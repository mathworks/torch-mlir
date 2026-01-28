// RUN: torch-mlir-opt %s -globalize-torch-params --split-input-file | FileCheck %s

// -----

// COM: Test basic single parameter conversion

// CHECK-LABEL: ml_program.global private mutable @weight(dense<1.000000e+00> : tensor<2x3xf32>) : tensor<2x3xf32>

// CHECK-LABEL:   func.func @forward() -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[GLOBAL_LOAD_0:.*]] = ml_program.global_load @weight : tensor<2x3xf32>
// CHECK:           %[[FROM_BUILTIN_TENSOR_0:.*]] = torch_c.from_builtin_tensor %[[GLOBAL_LOAD_0]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:           return %[[FROM_BUILTIN_TENSOR_0]]

// CHECK-LABEL: func.func @set_params(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-LABEL: func.func @get_params() -> tensor<2x3xf32>

func.func @forward() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.0> : tensor<2x3xf32>) {parameter_index = 0 : i64, parameter_name = "weight", parameter_type = "PARAMETER"} : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// COM: Test with multiple parameters

// CHECK-LABEL:   ml_program.global private mutable @weight(dense<1.000000e+00> : tensor<3x3xf32>) : tensor<3x3xf32>
// CHECK:         ml_program.global private mutable @bias(dense<0.000000e+00> : tensor<3xf32>) : tensor<3xf32>

// CHECK-LABEL:   func.func @forward() -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3],f32>) {
// CHECK:           %[[GLOBAL_LOAD_0:.*]] = ml_program.global_load @weight : tensor<3x3xf32>
// CHECK:           %[[FROM_BUILTIN_TENSOR_0:.*]] = torch_c.from_builtin_tensor %[[GLOBAL_LOAD_0]] : tensor<3x3xf32> -> !torch.vtensor<[3,3],f32>
// CHECK:           %[[GLOBAL_LOAD_1:.*]] = ml_program.global_load @bias : tensor<3xf32>
// CHECK:           %[[FROM_BUILTIN_TENSOR_1:.*]] = torch_c.from_builtin_tensor %[[GLOBAL_LOAD_1]] : tensor<3xf32> -> !torch.vtensor<[3],f32>
// CHECK:           return %[[FROM_BUILTIN_TENSOR_0]], %[[FROM_BUILTIN_TENSOR_1]]

// CHECK-LABEL: func.func @set_params(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>)
// CHECK-LABEL: func.func @get_params() -> (tensor<3x3xf32>, tensor<3xf32>)

func.func @forward() -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3],f32>) {
  %weight = torch.vtensor.literal(dense<1.0> : tensor<3x3xf32>) {parameter_index = 0 : i64, parameter_name = "weight", parameter_type = "PARAMETER"} : !torch.vtensor<[3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<3xf32>) {parameter_index = 1 : i64, parameter_name = "bias", parameter_type = "PARAMETER"} : !torch.vtensor<[3],f32>
  return %weight, %bias : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3],f32>
}

// -----

// COM: Test that parameter literals without all the required attributes are not converted

// CHECK-LABEL: func.func @forward
// CHECK: torch.vtensor.literal(dense<1.000000e+00> : tensor<3x3xf32>)
// CHECK: torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>)
// CHECK: torch.vtensor.literal(dense<2.000000e+00> : tensor<3xf32>)
// CHECK-NOT: ml_program.global

func.func @forward() -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) {
  %weight = torch.vtensor.literal(dense<1.0> : tensor<3x3xf32>) {parameter_name = "weight", parameter_type = "PARAMETER"} : !torch.vtensor<[3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<3xf32>) {parameter_index = 1 : i64, parameter_type = "PARAMETER"} : !torch.vtensor<[3],f32>
  %buf = torch.vtensor.literal(dense<2.0> : tensor<3xf32>) {parameter_index = 1 : i64, parameter_name = "bias"} : !torch.vtensor<[3],f32>
  return %weight, %bias, %buf : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
}

// -----

// COM: Test that non-parameter literals are not converted

// CHECK-LABEL: func.func @forward
// CHECK: torch.vtensor.literal(dense<1.000000e+00> : tensor<2xf32>)
// CHECK-NOT: ml_program.global

func.func @forward() -> !torch.vtensor<[2],f32> {
  %0 = torch.vtensor.literal(dense<1.0> : tensor<2xf32>) : !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// COM: Test with signed integer parameters (should convert to signless)
//
// CHECK-LABEL:   ml_program.global private mutable @indices(dense<[1, 2, 3]> : tensor<3xi32>) : tensor<3xi32>
func.func @forward() -> !torch.vtensor<[3],si32> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3]> : tensor<3xsi32>) {parameter_index = 0 : i64, parameter_name = "indices", parameter_type = "PARAMETER"} : !torch.vtensor<[3],si32>
  return %0 : !torch.vtensor<[3],si32>
}

// -----

// COM: Test get_params/set_params function generation

// CHECK-LABEL:   ml_program.global private mutable @param1(dense<1.000000e+00> : tensor<2xf32>) : tensor<2xf32>
// CHECK:         ml_program.global private mutable @param2(dense<2.000000e+00> : tensor<3xf32>) : tensor<3xf32>

// CHECK-LABEL:   func.func @set_params(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<3xf32>) -> (tensor<2xf32>, tensor<3xf32>) {
// CHECK:           ml_program.global_store @param1 = %[[ARG0]] : tensor<2xf32>
// CHECK:           ml_program.global_store @param2 = %[[ARG1]] : tensor<3xf32>
// CHECK:           return %[[ARG0]], %[[ARG1]]

// CHECK-LABEL:   func.func @get_params() -> (tensor<2xf32>, tensor<3xf32>) {
// CHECK:           %[[GLOBAL_LOAD_0:.*]] = ml_program.global_load @param1 : tensor<2xf32>
// CHECK:           %[[GLOBAL_LOAD_1:.*]] = ml_program.global_load @param2 : tensor<3xf32>
// CHECK:           return %[[GLOBAL_LOAD_0]], %[[GLOBAL_LOAD_1]]

func.func @forward() -> (!torch.vtensor<[2],f32>, !torch.vtensor<[3],f32>) {
  %0 = torch.vtensor.literal(dense<1.0> : tensor<2xf32>) {parameter_index = 0 : i64, parameter_name = "param1", parameter_type = "PARAMETER"} : !torch.vtensor<[2],f32>
  %1 = torch.vtensor.literal(dense<2.0> : tensor<3xf32>) {parameter_index = 1 : i64, parameter_name = "param2", parameter_type = "PARAMETER"} : !torch.vtensor<[3],f32>
  return %0, %1 : !torch.vtensor<[2],f32>, !torch.vtensor<[3],f32>
}

// -----

// COM: Test that literals without parameter_type="PARAMETER" are not converted

// CHECK-NOT: ml_program.global
func.func @forward() -> !torch.vtensor<[2],f32> {
  // CHECK: torch.vtensor.literal
  %0 = torch.vtensor.literal(dense<1.0> : tensor<2xf32>) {parameter_name = "not_param", parameter_type = "buffer"} : !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// COM: Test with dense resource elements attribute
//
// CHECK-LABEL:   ml_program.global private mutable @resource_param(dense_resource<resource_data> : tensor<4xf32>) : tensor<4xf32>

// CHECK-LABEL:   func.func @forward() -> !torch.vtensor<[4],f32> {
// CHECK:           %[[GLOBAL_LOAD_0:.*]] = ml_program.global_load @resource_param : tensor<4xf32>
// CHECK:           %[[FROM_BUILTIN_TENSOR_0:.*]] = torch_c.from_builtin_tensor %[[GLOBAL_LOAD_0]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:           return %[[FROM_BUILTIN_TENSOR_0]]

// CHECK-LABEL: func.func @set_params(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK-LABEL: func.func @get_params() -> tensor<4xf32>

func.func @forward() -> !torch.vtensor<[4],f32> {
  %0 = torch.vtensor.literal(dense_resource<resource_data> : tensor<4xf32>) {parameter_index = 0 : i64, parameter_name = "resource_param", parameter_type = "PARAMETER"} : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

{-#
  dialect_resources: {
    builtin: {
      resource_data: "0x080000000000803F0000003F0000803F0000003F"
    }
  }
#-}

// -----

// COM: Test with dense resource elements attribute - signed integer (should convert to signless)

// CHECK-LABEL:   ml_program.global private mutable @resource_int_param(dense_resource<int_resource> : tensor<3xi64>) : tensor<3xi64>
func.func @forward() -> !torch.vtensor<[3],si64> {
  %0 = torch.vtensor.literal(dense_resource<int_resource> : tensor<3xsi64>) {parameter_index = 0 : i64, parameter_name = "resource_int_param", parameter_type = "PARAMETER"} : !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}

{-#
  dialect_resources: {
    builtin: {
      int_resource: "0x080000000100000000000000020000000000000003000000"
    }
  }
#-}

// -----

// COM: Test mixed dense and dense resource parameters

// CHECK-LABEL:   ml_program.global private mutable @dense_param(dense<2.000000e+00> : tensor<2xf32>) : tensor<2xf32>
// CHECK:         ml_program.global private mutable @resource_param(dense_resource<mixed_resource> : tensor<3xf32>) : tensor<3xf32>

func.func @forward() -> (!torch.vtensor<[2],f32>, !torch.vtensor<[3],f32>) {
  %0 = torch.vtensor.literal(dense<2.0> : tensor<2xf32>) {parameter_index = 0 : i64, parameter_name = "dense_param", parameter_type = "PARAMETER"} : !torch.vtensor<[2],f32>
  %1 = torch.vtensor.literal(dense_resource<mixed_resource> : tensor<3xf32>) {parameter_index = 1 : i64, parameter_name = "resource_param", parameter_type = "PARAMETER"} : !torch.vtensor<[3],f32>
  return %0, %1 : !torch.vtensor<[2],f32>, !torch.vtensor<[3],f32>
}

{-#
  dialect_resources: {
    builtin: {
      mixed_resource: "0x080000000000803F0000003F0000003F"
    }
  }
#-}

// -----

// COM: Test dense resource with multi-dimensional tensor

// CHECK-LABEL:   ml_program.global private mutable @matrix_param(dense_resource<matrix_data> : tensor<2x3xf32>) : tensor<2x3xf32>

func.func @forward() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense_resource<matrix_data> : tensor<2x3xf32>) {parameter_index = 0 : i64, parameter_name = "matrix_param", parameter_type = "PARAMETER"} : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

{-#
  dialect_resources: {
    builtin: {
      matrix_data: "0x080000000000803F0000003F0000803F0000003F0000803F0000003F"
    }
  }
#-}

// -----

// COM: Test parameter ordering by parameter_index

// -----

// COM: Test parameter ordering by parameter_index

// CHECK-LABEL:   ml_program.global private mutable @fc1.weight(dense<1.000000e+00> : tensor<4x4xf32>) : tensor<4x4xf32> {parameter_index = 2 : i64}
// CHECK:         ml_program.global private mutable @fc1.bias(dense<0.000000e+00> : tensor<4xf32>) : tensor<4xf32> {parameter_index = 3 : i64}
// CHECK:         ml_program.global private mutable @fc2.weight(dense<2.000000e+00> : tensor<4x4xf32>) : tensor<4x4xf32> {parameter_index = 0 : i64}
// CHECK:         ml_program.global private mutable @fc2.bias(dense<1.000000e+00> : tensor<4xf32>) : tensor<4xf32> {parameter_index = 1 : i64}

// CHECK-LABEL:   func.func @set_params(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4xf32>, %[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4xf32>)
// CHECK-SAME:      -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) {
// CHECK:           ml_program.global_store @fc2.weight = %[[ARG0]] : tensor<4x4xf32>
// CHECK:           ml_program.global_store @fc2.bias = %[[ARG1]] : tensor<4xf32>
// CHECK:           ml_program.global_store @fc1.weight = %[[ARG2]] : tensor<4x4xf32>
// CHECK:           ml_program.global_store @fc1.bias = %[[ARG3]] : tensor<4xf32>
// CHECK:           return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]

// CHECK-LABEL:   func.func @get_params() -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) {
// CHECK:           %[[LOAD0:.*]] = ml_program.global_load @fc2.weight : tensor<4x4xf32>
// CHECK:           %[[LOAD1:.*]] = ml_program.global_load @fc2.bias : tensor<4xf32>
// CHECK:           %[[LOAD2:.*]] = ml_program.global_load @fc1.weight : tensor<4x4xf32>
// CHECK:           %[[LOAD3:.*]] = ml_program.global_load @fc1.bias : tensor<4xf32>
// CHECK:           return %[[LOAD0]], %[[LOAD1]], %[[LOAD2]], %[[LOAD3]]

func.func @forward(%arg0: !torch.vtensor<[4,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> {
  %fc1_weight = torch.vtensor.literal(dense<1.0> : tensor<4x4xf32>) {parameter_index = 2 : i64, parameter_name = "fc1.weight", parameter_type = "PARAMETER"} : !torch.vtensor<[4,4],f32>
  %fc1_bias = torch.vtensor.literal(dense<0.0> : tensor<4xf32>) {parameter_index = 3 : i64, parameter_name = "fc1.bias", parameter_type = "PARAMETER"} : !torch.vtensor<[4],f32>
  %0 = torch.aten.linear %arg0, %fc1_weight, %fc1_bias : !torch.vtensor<[4,4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4,4,4],f32>
  %fc2_weight = torch.vtensor.literal(dense<2.0> : tensor<4x4xf32>) {parameter_index = 0 : i64, parameter_name = "fc2.weight", parameter_type = "PARAMETER"} : !torch.vtensor<[4,4],f32>
  %fc2_bias = torch.vtensor.literal(dense<1.0> : tensor<4xf32>) {parameter_index = 1 : i64, parameter_name = "fc2.bias", parameter_type = "PARAMETER"} : !torch.vtensor<[4],f32>
  %1 = torch.aten.linear %0, %fc2_weight, %fc2_bias : !torch.vtensor<[4,4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4,4,4],f32>
  return %1 : !torch.vtensor<[4,4,4],f32>
}


// -----

// COM: Test that set_params/get_params functions are uniquified

// CHECK-LABEL:   ml_program.global private mutable @param(dense<1.000000e+00> : tensor<2xf32>) : tensor<2xf32> {parameter_index = 0 : i64}

// CHECK-LABEL:   func.func @set_params() {
// CHECK:           return

// CHECK-LABEL:   func.func @get_params() {
// CHECK:           return

// CHECK-LABEL:   func.func @set_params_0(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           ml_program.global_store @param = %[[ARG0]] : tensor<2xf32>
// CHECK:           return %[[ARG0]]

// CHECK-LABEL:   func.func @get_params_0() -> tensor<2xf32> {
// CHECK:           %[[GLOBAL_LOAD_0:.*]] = ml_program.global_load @param : tensor<2xf32>
// CHECK:           return %[[GLOBAL_LOAD_0]]

func.func @set_params() {
  return
}

func.func @get_params() {
  return
}

func.func @forward() -> !torch.vtensor<[2],f32> {
  %0 = torch.vtensor.literal(dense<1.0> : tensor<2xf32>) {parameter_index = 0 : i64, parameter_name = "param", parameter_type = "PARAMETER"} : !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}
