/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Output kernels for fusing computation into Eigen Tensor contractions:
//   (1) FusedConv2DOp
//   (2) FusedMatMulOp
//
// Supported fused computations:
//   (1) {Conv2D/MatMul} + BiasAdd + <Activation>
//   (2) {Conv2D/MatMul} + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...

#pragma once

#include "tfdml/runtime_adapter/op_kernel.h"
#include "tfdml/runtime_adapter/tensor.h"
#include "tfdml/runtime_adapter/types.h"

namespace tfdml
{

enum class FusedComputationType
{
    kUndefined,
    kBiasAdd,
    kBiasAddWithRelu,
    kBiasAddWithRelu6,
    kBiasAddWithElu,
    kBiasAddWithLeakyRelu,
    kFusedBatchNorm,
    kFusedBatchNormWithRelu,
    kFusedBatchNormWithRelu6,
    kFusedBatchNormWithElu,
    kFusedBatchNormWithLeakyRelu
};

// We have to pass around additional arguments for all possible fusion types.
struct FusedComputationArgs
{
    float epsilon = 0.0;         // Used by `FusedBatchNorm` fusion only
    float leakyrelu_alpha = 0.0; // Used by `LeakyRelu` fusion only
};

struct FusedComputationPattern
{
    FusedComputationType fused_computation;
    std::vector<std::string> fused_ops;
};

// Parse attributes from the kernel construction context, and verifies that they
// specify valid fused computation pattern.
Status InitializeFusedComputation(
    OpKernelConstruction* context,
    const std::string& kernel_name,
    const std::vector<FusedComputationPattern>& patterns,
    FusedComputationType* fused_computation,
    FusedComputationArgs* fused_computation_args);

// Type alias for the tensor contraction output mapper.
template <typename Scalar, typename StorageIndex>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<Scalar, StorageIndex, Eigen::ColMajor>;

// Returns input expression without any transformations.
struct Identity
{
    template <typename XprType>
    static auto apply(XprType expr) -> XprType
    {
        return expr;
    };
};

// Applies `Relu` to the passed input expression.
struct Relu
{
    template <typename XprType>
    static auto apply(XprType expr)
        -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>()))
    {
        return expr.cwiseMax(static_cast<typename XprType::Scalar>(0));
    };
};

// Applies `Relu6` to the passed input expression.
struct Relu6
{
    template <typename XprType>
    static auto apply(XprType expr)
        -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())
                        .cwiseMin(std::declval<typename XprType::Scalar>()))
    {
        return expr.cwiseMax(static_cast<typename XprType::Scalar>(0))
            .cwiseMin(static_cast<typename XprType::Scalar>(6));
    };
};

// Applies `Elu` to the passed input expression.
struct Elu
{
    template <typename XprType>
    static auto apply(XprType expr) -> decltype(
        (expr < std::declval<typename XprType::Scalar>())
            .select(
                expr.exp() -
                    expr.constant(std::declval<typename XprType::Scalar>()),
                expr))
    {
        return (expr < static_cast<typename XprType::Scalar>(0))
            .select(
                expr.exp() -
                    expr.constant(static_cast<typename XprType::Scalar>(1)),
                expr);
    };
};

// Applies `LeakyRelu` to the passed input expression.
struct LeakyRelu
{
    template <typename XprType>
    static auto apply(XprType expr, const float leakyrelu_alpha) -> decltype(
        (expr < std::declval<typename XprType::Scalar>())
            .select(
                expr * expr.constant(std::declval<typename XprType::Scalar>()),
                expr))
    {
        return (expr < static_cast<typename XprType::Scalar>(0))
            .select(
                expr * expr.constant(static_cast<typename XprType::Scalar>(
                           leakyrelu_alpha)),
                expr);
    };
};

template <typename T>
struct BiasAddArgs
{
    const T* bias_add_data = nullptr;
    float leakyrelu_alpha;

    static bool IsSupported(FusedComputationType fusion)
    {
        return fusion == FusedComputationType::kBiasAdd ||
               fusion == FusedComputationType::kBiasAddWithRelu ||
               fusion == FusedComputationType::kBiasAddWithRelu6 ||
               fusion == FusedComputationType::kBiasAddWithElu ||
               fusion == FusedComputationType::kBiasAddWithLeakyRelu;
    }
};

template <typename T>
struct FusedBatchNormArgs
{
    const T* scale_data = nullptr;
    const T* offset_data = nullptr;
    const T* estimated_mean_data = nullptr;
    const T* estimated_variance_data = nullptr;

    // Precomputed expression:
    //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
    Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor;

    float leakyrelu_alpha;

    static bool IsSupported(FusedComputationType fusion)
    {
        return fusion == FusedComputationType::kFusedBatchNorm ||
               fusion == FusedComputationType::kFusedBatchNormWithRelu ||
               fusion == FusedComputationType::kFusedBatchNormWithRelu6 ||
               fusion == FusedComputationType::kFusedBatchNormWithElu ||
               fusion == FusedComputationType::kFusedBatchNormWithLeakyRelu;
    }
};

} // namespace tfdml