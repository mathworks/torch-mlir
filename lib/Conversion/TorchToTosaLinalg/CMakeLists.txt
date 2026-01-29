add_mlir_conversion_library(TorchMLIRTorchToTosaLinalg
  TorchToTosaLinalg.cpp
  GlobalizeTorchParams.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTosaLinalg
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/GlobalizeTorchParams

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRLinalgDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTosaLinalg)
