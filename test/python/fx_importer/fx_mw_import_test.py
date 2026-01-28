# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir import fx_mw


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# Tests that constants and parameters work generally with the mutation path.
# This doesn't do mutation but ensures that the basics remain functional.
# CHECK-LABEL: test_import_frozen_exported_program
# CHECK:     func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[a:.+]] = torch.aten.randn
# CHECK-DAG: %[[b:.+]] = torch.vtensor.literal(dense<{{.*>+}} : tensor<3x1xf32>) : !torch.vtensor<[3,1],f32>
# CHECK-DAG: %[[p:.+]] = torch.vtensor.literal(dense<{{.*>+}} : tensor<1x1xf32>) {parameter_index = 0 : i64, parameter_name = "param", parameter_type = "PARAMETER"} : !torch.vtensor<[1,1],f32>
# CHECK-DAG: %[[tanh:.+]] = torch.aten.tanh %[[ARG0]]
# CHECK-DAG: %[[mul_a:.+]] = torch.aten.mul.Tensor %[[tanh]], %[[a]]
# CHECK-DAG: %[[mul_b:.+]] = torch.aten.mul.Tensor %[[mul_a]], %[[b]]
# CHECK-DAG: %[[mul_p:.+]] = torch.aten.mul.Tensor %[[mul_b]], %[[p]]
# CHECK:     return %[[mul_p]]
#
# Validate dialect resources does not exist.
# CHECK-NOT: dialect_resources:
# CHECK-NOT: torch_tensor_3_1_torch.float32
def test_import_frozen_exported_program():
    @torch._dynamo.assume_constant_result
    def get_a():
        return torch.randn(1, 4)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.randn(3, 1)
            self.param = nn.Parameter(torch.randn(1, 1))

        def forward(self, x):
            return torch.tanh(x) * get_a() * self.b * self.param

    m = fx_mw.import_exported_model(
        torch.export.export(
            Basic(),
            (torch.randn(3, 4),),
        )
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_frozen_buffer
# CHECK: %[[buffer_literal:.+]] = torch.vtensor.literal
# CHECK-SAME: {parameter_index = 0 : i64, parameter_name = "buffer", parameter_type = "BUFFER"}
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %arg0, %0
# CHECK: return %[[mul]]
def test_frozen_buffer():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            return x * self.buffer

    m = fx_mw.import_exported_model(torch.export.export(Basic(), (torch.randn(3, 4),)))
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_frozen_buffer_non_persistent
# CHECK: %[[buffer_literal:.+]] = torch.vtensor.literal
# CHECK-SAME: {parameter_index = 0 : i64, parameter_name = "buffer", parameter_type = "BUFFER"}
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %arg0, %0
# CHECK: return %[[mul]]
def test_frozen_buffer_non_persistent():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4), persistent=False)

        def forward(self, x):
            return x * self.buffer

    m = fx_mw.import_exported_model(torch.export.export(Basic(), (torch.randn(3, 4),)))
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_multiple_parameters
# CHECK:   func.func @main(
# CHECK-SAME:      %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK:           %[[VTENSOR_0:.*]] = torch.vtensor.literal
# CHECK-SAME:  tensor<4x4xf32>) {parameter_index = 2 : i64, parameter_name = "fc1.weight", parameter_type = "PARAMETER"}
# CHECK:           %[[VTENSOR_1:.*]] = torch.vtensor.literal
# CHECK-SAME:  tensor<4xf32>) {parameter_index = 3 : i64, parameter_name = "fc1.bias", parameter_type = "PARAMETER"}
# CHECK:           %[[ATEN_0:.*]] = torch.aten.linear %[[ARG0]], %[[VTENSOR_0]], %[[VTENSOR_1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[3,4],f32>
# CHECK:           %[[VTENSOR_2:.*]] = torch.vtensor.literal
# CHECK-SAME:  tensor<4x4xf32>) {parameter_index = 0 : i64, parameter_name = "fc2.weight", parameter_type = "PARAMETER"}
# CHECK:           %[[VTENSOR_3:.*]] = torch.vtensor.literal
# CHECK-SAME:  tensor<4xf32>) {parameter_index = 1 : i64, parameter_name = "fc2.bias", parameter_type = "PARAMETER"}
# CHECK:           %[[ATEN_1:.*]] = torch.aten.linear %[[ATEN_0]], %[[VTENSOR_2]], %[[VTENSOR_3]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[3,4],f32>
# CHECK:           return %[[ATEN_1]]
def test_multiple_parameters():
    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc2 = torch.nn.Linear(4, 4)
            self.fc1 = torch.nn.Linear(4, 4)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    m = fx_mw.import_exported_model(
        torch.export.export(LinearNet(), (torch.randn(3, 4),))
    )
    print(m)
    m.operation.verify()
