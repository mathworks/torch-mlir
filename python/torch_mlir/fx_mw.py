# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import numpy as np
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass

from .compiler_utils import OutputType
from .compiler_utils_mw import (
    run_pipeline_mw,
    lower_mlir_module_mw,
)
from . import fx

from torch_mlir.ir import (
    Context,
    Location,
    Module,
    Value,
    Operation,
    StringAttr,
    DenseElementsAttr,
)
from torch_mlir.dialects.torch import register_dialect
from torch_mlir.extras.fx_importer import (
    GraphNodeImporter,
    FxImporterHooks,
    InputInfo,
    TORCH_DTYPE_TO_NPY_TYPE,
    TORCH_DTYPE_TO_MLIR_TYPE,
)


@dataclass
class LoweringOptions:
    """Options for controlling MLIR lowering passes."""

    globalize_torch_params: bool = True


def import_exported_model(
    prog: torch.export.ExportedProgram,
    output_type: str = "raw",
    experimental_support_mutation: bool = True,
    decomposition_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
    verbose: bool = False,
    enable_ir_printing: bool = False,
    lowering_options: Optional[LoweringOptions] = None,
):

    # Create hook with parameter order from graph signature
    hook = ParameterMetadataHook()
    hook.initialize_from_exported_program(prog)

    mlir_module = fx.export_and_import(
        prog,
        output_type=OutputType.RAW,
        experimental_support_mutation=experimental_support_mutation,
        verbose=verbose,
        enable_ir_printing=enable_ir_printing,
        decomposition_table=decomposition_table,
        hooks=hook,
    )

    if output_type != "raw":
        mlir_module = lower_module(
            mlir_module, output_type, verbose, enable_ir_printing, lowering_options
        )

    return mlir_module


def lower_module_from_file(
    mlir_file: str,
    output_type: str,
    verbose: bool = False,
    enable_ir_printing: bool = False,
    lowering_options: Optional[LoweringOptions] = None,
):
    src = open(mlir_file, "r").read()
    with Context() as ctx:
        register_dialect(ctx)
        with Location.unknown() as loc:
            mlir_module = Module.parse(src)

    return lower_module(
        mlir_module, output_type, verbose, enable_ir_printing, lowering_options
    )


def lower_module(
    mlir_module,
    output_type: str,
    verbose: bool = False,
    enable_ir_printing: bool = False,
    lowering_options: Optional[LoweringOptions] = None,
):

    backend_legal_ops = None

    match output_type:
        case "torch":
            output_type = OutputType.TORCH
        case "tosa":
            output_type = OutputType.TOSA
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.native_layer_norm",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.linear",
            ]
        case "linalg_on_tensors":
            output_type = OutputType.LINALG_ON_TENSORS
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.unflatten.int",
            ]
        case "tosa_linalg":
            output_type = OutputType.TOSA_LINALG
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.native_layer_norm",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.unflatten.int",
                "aten.as_strided",
            ]
        case "raw":
            output_type = OutputType.RAW
        case _:
            raise ValueError("Importing PyTorch model failed: Unsupported output type.")

    backend_legal_op_arg_str = ""
    if backend_legal_ops is not None:
        if not len(backend_legal_ops) == 0:
            backend_legal_op_arg_str = "backend-legal-ops=" + ",".join(
                backend_legal_ops
            )

    extra_library_file_name = ""
    option_string = (
        "{"
        + backend_legal_op_arg_str
        + " extra-library="
        + extra_library_file_name
        + "}"
    )

    pipeline_passes = []
    if lowering_options is not None and lowering_options.globalize_torch_params:
        pipeline_passes.append("globalize-torch-params")
    pipeline_passes.append(
        f"func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{option_string}"
    )

    pipeline = "builtin.module(" + ", ".join(pipeline_passes) + ")"

    run_pipeline_mw(
        mlir_module,
        pipeline,
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=enable_ir_printing,
    )

    return lower_mlir_module_mw(verbose, output_type, mlir_module)


class ParameterMetadataHook(FxImporterHooks):
    """Add parameter/buffer metadata to vtensor.literal ops."""

    def __init__(self):
        super().__init__()
        self.parameter_order = {}

    def initialize_from_exported_program(self, prog: torch.export.ExportedProgram):
        """Extract parameter order from ExportedProgram's graph signature."""
        for idx, input_spec in enumerate(prog.graph_signature.input_specs):
            if input_spec.kind in [
                torch.export.graph_signature.InputKind.PARAMETER,
                torch.export.graph_signature.InputKind.BUFFER,
            ]:
                self.parameter_order[input_spec.target] = idx

    def _create_tensor_literal(
        self,
        gni: "GraphNodeImporter",
        tensor: torch.Tensor,
        info: Optional[InputInfo] = None,
    ) -> Value:
        """Helper to create vtensor.literal with DenseElementsAttr."""
        vtensor_type = gni._cc.tensor_to_vtensor_type(tensor)

        npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
        np_tensor = np.array(tensor.detach().cpu().numpy()).astype(npy_dtype)

        element_type = TORCH_DTYPE_TO_MLIR_TYPE[tensor.dtype]()
        elements_attr = DenseElementsAttr.get(
            type=element_type, array=np_tensor, shape=np_tensor.shape
        )

        # Add metadata if info is available
        attributes = {"value": elements_attr}
        if info is not None:
            param_name = info.input_spec.target
            param_kind = info.input_spec.kind.name

            attributes["parameter_name"] = StringAttr.get(param_name)
            attributes["parameter_type"] = StringAttr.get(param_kind)

            # Add parameter index if available
            if param_name in self.parameter_order:
                attributes["parameter_index"] = gni._cc.integer_attr(
                    self.parameter_order[param_name], 64
                )

        # Create the operation
        return Operation.create(
            name="torch.vtensor.literal",
            results=[vtensor_type],
            attributes=attributes,
        ).result

    def resolve_literal(
        self, gni: "GraphNodeImporter", literal: Any, info: Optional[InputInfo]
    ) -> Optional[Value]:
        """Override to always create DenseElementsAttr for tensor literals."""
        if not isinstance(literal, torch.Tensor):
            return None  # Let default handle non-tensors

        return self._create_tensor_literal(gni, literal, info)

    def resolve_input(
        self, gni: "GraphNodeImporter", value: Any, info: InputInfo
    ) -> Optional[Value]:
        """Override to add metadata attributes to parameter/buffer literals."""
        if not isinstance(value, torch.Tensor):
            return None  # Let default handle non-tensors

        return self._create_tensor_literal(gni, value, info)
