//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosaLinalg/GlobalizeTorchParams.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir::torch {

#define GEN_PASS_DEF_GLOBALIZETORCHPARAMS
#include "torch-mlir/Conversion/Passes.h.inc"

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

namespace {

// Helper function to check if an operation is a parameter or buffer
static bool isParameterOrBuffer(Operation *op) {
  auto paramNameAttr = op->getAttrOfType<StringAttr>("parameter_name");
  auto paramTypeAttr = op->getAttrOfType<StringAttr>("parameter_type");
  auto paramIndexAttr = op->getAttrOfType<IntegerAttr>("parameter_index");

  return paramNameAttr && paramTypeAttr && paramIndexAttr &&
         (paramTypeAttr.getValue() == "PARAMETER" ||
          paramTypeAttr.getValue() == "BUFFER");
}

} // namespace

// -----------------------------------------------------------------------------
// Patterns
// -----------------------------------------------------------------------------

namespace {

// Pattern to convert torch.vtensor.literal with parameter attributes to
// ml_program.global
struct GlobalizeValueTensorLiteralPattern
    : public OpConversionPattern<ValueTensorLiteralOp> {
  using OpConversionPattern<ValueTensorLiteralOp>::OpConversionPattern;
  using OpAdaptor = typename ValueTensorLiteralOp::Adaptor;

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Only convert if this is a parameter or buffer
    if (!isParameterOrBuffer(op))
      return rewriter.notifyMatchFailure(
          op, "Expected parameter or buffer attributes");

    auto paramNameAttr = op->getAttrOfType<StringAttr>("parameter_name");
    auto paramTypeAttr = op->getAttrOfType<StringAttr>("parameter_type");
    auto paramIndexAttr = op->getAttrOfType<IntegerAttr>("parameter_index");

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(
          op, "Expected op to be contained within a module");

    // Get the value attribute (dense_resource or dense elements)
    auto valueAttr = op.getValueAttr();
    if (!valueAttr)
      return rewriter.notifyMatchFailure(
          op, "Expected value to be dense elements or dense resource");

    // Convert torch.vtensor type to tensor type
    auto outputTy = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    if (!outputTy)
      return rewriter.notifyMatchFailure(
          op, "Expected ranked tensor type after conversion");

    // Handle integer types - convert to signless if needed
    ElementsAttr attr = cast<ElementsAttr>(valueAttr);
    if (auto elements = dyn_cast<DenseIntElementsAttr>(attr)) {
      if (elements.getElementType().isSignedInteger()) {
        Type builtinTensorElemTy = outputTy.getElementType();
        unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();
        DenseElementsAttr signlessAttr =
            elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
              return APInt(bitWidth, v.getSExtValue(), /*isSigned=*/false);
            });
        attr = signlessAttr;
      }
    }

    // Handle DenseResourceElementsAttr - retag to signless if needed
    if (auto res = dyn_cast<DenseResourceElementsAttr>(attr)) {
      auto shapedAttrTy = cast<ShapedType>(res.getType());
      if (auto intTy = dyn_cast<IntegerType>(shapedAttrTy.getElementType())) {
        if (!intTy.isSignless()) {
          auto signlessTy =
              IntegerType::get(rewriter.getContext(), intTy.getWidth());
          auto newTy =
              RankedTensorType::get(shapedAttrTy.getShape(), signlessTy);
          attr = DenseResourceElementsAttr::get(newTy, res.getRawHandle());
        }
      }
    }

    // Check if global already exists
    auto existingGlobal =
        module.lookupSymbol<ml_program::GlobalOp>(paramNameAttr.getValue());

    // Each parameter should only be encountered once, so global should not
    // exist
    assert(!existingGlobal && "Global for parameter should not already exist");

    // Create ml_program.global at module level
    // Insert after the last global or at the start if no globals exist to
    // maintain proper ordering
    OpBuilder::InsertionGuard guard(rewriter);

    // Find the position after the last global
    Operation *insertPoint = nullptr;
    for (auto &op : module.getBody()->getOperations()) {
      if (isa<ml_program::GlobalOp>(op)) {
        insertPoint = &op;
      }
    }

    if (insertPoint) {
      rewriter.setInsertionPointAfter(insertPoint);
    } else {
      rewriter.setInsertionPointToStart(module.getBody());
    }

    auto globalOp = ml_program::GlobalOp::create(
        rewriter, op.getLoc(), paramNameAttr.getValue(), outputTy,
        /*is_mutable=*/true, attr,
        /*sym_visibility=*/nullptr);
    globalOp.setPrivate();

    // Preserve parameter_index and parameter_type attributes on the global
    globalOp->setAttr("parameter_index", paramIndexAttr);
    globalOp->setAttr("parameter_type", paramTypeAttr);

    // Replace the vtensor.literal with a global_load
    rewriter.setInsertionPoint(op);
    auto globalSymbolRef =
        SymbolRefAttr::get(rewriter.getContext(), paramNameAttr.getValue());
    auto loadOp = ml_program::GlobalLoadOp::create(rewriter, op.getLoc(),
                                                   outputTy, globalSymbolRef);

    rewriter.replaceOp(op, loadOp);
    return success();
  }
};

} // namespace

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

namespace {

// Helper struct to hold global op with its index
struct IndexedGlobalOp {
  ml_program::GlobalOp globalOp;
  int64_t index;

  bool operator<(const IndexedGlobalOp &other) const {
    return index < other.index;
  }
};

// Helper function to generate a unique function name
static std::string getUniqueFunctionName(ModuleOp module, StringRef baseName) {
  std::string candidateName = baseName.str();
  int suffix = 0;

  while (module.lookupSymbol<func::FuncOp>(candidateName)) {
    candidateName = baseName.str() + "_" + std::to_string(suffix++);
  }

  return candidateName;
}

// Function to create a set_params function that stores values to
// ml_program.global ops. This creates a function that takes new values as
// parameters and stores them to globals
static void createGlobalSetFunction(ModuleOp module,
                                    ArrayRef<ml_program::GlobalOp> globalOps) {

  if (globalOps.empty())
    return;

  OpBuilder builder(module.getContext());

  SmallVector<Type> paramTypes;
  for (auto globalOp : globalOps) {
    paramTypes.push_back(globalOp.getType());
  }

  // Create set_params function with a unique name
  std::string setFuncName = getUniqueFunctionName(module, "set_params");

  // Create new set_params function that returns the updated values
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(paramTypes, paramTypes);
  auto setFunc = func::FuncOp::create(module.getLoc(), setFuncName, funcType);
  setFunc.setPublic();
  module.push_back(setFunc);

  // Create entry block with arguments
  Block *entryBlock = setFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create stores for each global using the corresponding function argument
  SmallVector<Value> storedValues;
  for (size_t idx = 0; idx < globalOps.size(); ++idx) {
    auto globalOp = globalOps[idx];
    auto globalSymbolRef =
        SymbolRefAttr::get(builder.getContext(), globalOp.getSymName());

    // Get the corresponding block argument
    Value argValue = entryBlock->getArgument(idx);

    // Create ml_program.global_store to store the argument value
    ml_program::GlobalStoreOp::create(builder, globalOp.getLoc(),
                                      globalSymbolRef, argValue);

    // Collect the stored value to return
    storedValues.push_back(argValue);
  }

  // Return the stored values
  func::ReturnOp::create(builder, module.getLoc(), storedValues);
}

// Function to create a get_params function that loads values from
// ml_program.global ops This creates a function that returns all parameter
// values as results
static void createGlobalGetFunction(ModuleOp module,
                                    ArrayRef<ml_program::GlobalOp> globalOps) {

  if (globalOps.empty())
    return;

  OpBuilder builder(module.getContext());

  SmallVector<Type> returnTypes;
  for (ml_program::GlobalOp globalOp : globalOps) {
    returnTypes.push_back(globalOp.getType());
  }

  // Create get_params function with a unique name
  std::string getFuncName = getUniqueFunctionName(module, "get_params");

  // Create new get_params function
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType({}, returnTypes);
  auto getFunc = func::FuncOp::create(module.getLoc(), getFuncName, funcType);
  getFunc.setPublic();
  module.push_back(getFunc);

  // Create entry block
  Block *entryBlock = getFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create loads for each global
  SmallVector<Value> loadedValues;
  for (auto globalOp : globalOps) {
    auto globalSymbolRef =
        SymbolRefAttr::get(builder.getContext(), globalOp.getSymName());

    // Create ml_program.global_load to load the global value
    auto loadOp = ml_program::GlobalLoadOp::create(
        builder, globalOp.getLoc(), globalOp.getType(), globalSymbolRef);
    loadedValues.push_back(loadOp);
  }

  // Return all loaded values
  func::ReturnOp::create(builder, module.getLoc(), loadedValues);
}

} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class GlobalizeTorchParams
    : public impl::GlobalizeTorchParamsBase<GlobalizeTorchParams> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<ml_program::MLProgramDialect,
                           TorchConversion::TorchConversionDialect,
                           Torch::TorchDialect, func::FuncDialect>();

    // Mark vtensor.literal ops with parameter attributes as illegal
    target.addDynamicallyLegalOp<ValueTensorLiteralOp>(
        [](ValueTensorLiteralOp op) {
          // Legal if it's NOT a parameter or buffer
          return !isParameterOrBuffer(op);
        });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<GlobalizeValueTensorLiteralPattern>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // Collect all ml_program.global ops with their parameter_index
    SmallVector<IndexedGlobalOp> indexedGlobals;
    for (auto globalOp : module.getOps<ml_program::GlobalOp>()) {
      int64_t index = -1;
      if (auto paramIndexAttr =
              globalOp->getAttrOfType<IntegerAttr>("parameter_index")) {
        index = paramIndexAttr.getInt();
      }
      indexedGlobals.push_back({globalOp, index});
    }

    // Sort globals by parameter_index
    std::sort(indexedGlobals.begin(), indexedGlobals.end());

    // Extract sorted global ops
    SmallVector<ml_program::GlobalOp> sortedGlobalOps;
    for (const auto &indexed : indexedGlobals) {
      sortedGlobalOps.push_back(indexed.globalOp);
    }

    // Create set_params function for updating globals (in sorted order)
    createGlobalSetFunction(module, sortedGlobalOps);

    // Create get_params function for retrieving globals (in sorted order)
    createGlobalGetFunction(module, sortedGlobalOps);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGlobalizeTorchParamsPass() {
  return std::make_unique<GlobalizeTorchParams>();
}

} // namespace mlir::torch
