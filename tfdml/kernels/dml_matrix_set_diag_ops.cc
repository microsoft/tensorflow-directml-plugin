/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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

#include "tfdml/kernels/dml_matrix_diag_helpers.h"
#include "tfdml/kernels/pch.h"
#include <numeric>

namespace tfdml {

class MatrixSetDiagInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  MatrixSetDiagInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr) {
    const Tensor& input = ctx->input(0);
    const Tensor& diag = ctx->input(1);

    // MatrixSetDiag, MatrixSetDiagV2, and MatrixSetDiagV3 both use this OpKernel. MatrixSetDiag
    // only has two inputs, so we have to check the number of inputs before
    // reading additional parameters in MatrixSetDiagV2/MatrixSetDiagV3.
    int32_t lower_diag_index = 0;
    int32_t upper_diag_index = 0;

    // MatrixSetDiagV2/V3-specific.
    if (ctx->num_inputs() > 2) {
      auto& diag_index = ctx->input(2);
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      lower_diag_index = diag_index.base<int32_t>()[0];
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            ctx, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.base<int32_t>()[1];
        }
      }
    }

    const TensorShape& input_shape = input.shape();
    const TensorShape& diag_shape = diag.shape();
    const int input_rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diag_shape.DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const Eigen::Index num_rows = input_shape.dim_size(input_rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(input_rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        ctx,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(ctx,
                (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                    upper_diag_index == 0,
                errors::InvalidArgument(
                    "upper_diag_index is out of bound: ", upper_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        ctx, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    // Check if diag size is consistent with input.
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(
        ctx,
        lower_diag_index == upper_diag_index ||
            (diag_shape.dim_size(input_rank - 2) == num_diags),
        errors::InvalidArgument("The number of diagonals provided in `diag` "
                                "is not consistent with `lower_diag_index` and "
                                "`upper_diag_index`"));

    TensorShape expected_diag_shape = input_shape;
    expected_diag_shape.RemoveLastDims(2);
    if (num_diags > 1) expected_diag_shape.AddDim(num_diags);
    const int32_t max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    expected_diag_shape.AddDim(max_diag_len);
    OP_REQUIRES(
        ctx, expected_diag_shape == diag_shape,
        errors::InvalidArgument(
            "Either first dimensions of diagonal don't match input.shape[:-2], "
            "or diagonal.shape[:-1] is not equal to the longests diagonal in "
            "range [lower_diag_index:upper_diag_index].\nInput shape: ",
            input_shape.DebugString(),
            "\nDiagonal shape: ", diag_shape.DebugString(),
            "\nExpected diagonal shape: ", expected_diag_shape.DebugString()));

    lower_diag_index_ = lower_diag_index;
    upper_diag_index_ = upper_diag_index;
  }

  int32_t GetLowerDiagIndex() const { return lower_diag_index_; }
  int32_t GetUpperDiagIndex() const { return upper_diag_index_; }

 private:
  int32_t lower_diag_index_;
  int32_t upper_diag_index_;
};

template <typename T>
class DmlMatrixSetDiagKernel : public DmlKernel {
 public:
  using InitHelper = MatrixSetDiagInitHelper;

  DmlMatrixSetDiagKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    const TensorShape& in_out_shape = ctx->GetInputTensorShape(0);
    const TensorShape& diag_shape = ctx->GetInputTensorShape(1);

    int64_t batch_size = 1;
    for (int i = 0; i < in_out_shape.dims() - 2; ++i) {
      batch_size *= in_out_shape.dim_size(i);
    }

    int64_t height = in_out_shape.dim_size(in_out_shape.dims() - 2);
    int64_t width = in_out_shape.dim_size(in_out_shape.dims() - 1);
    const TensorShape flattened_in_out_shape({1, batch_size, height, width});

    int32_t k0 = init_helper->GetLowerDiagIndex();
    int32_t k1 = init_helper->GetUpperDiagIndex();

    int64_t diag_tail_width = diag_shape.dim_size(diag_shape.dims() - 1);
    int64_t diag_tail_depth =
        k0 == k1 ? 1 : diag_shape.dim_size(diag_shape.dims() - 2);

    int64_t diag_head_size = 1;
    for (int i = 0; i < in_out_shape.dims() - 2; ++i) {
      diag_head_size *= diag_shape.dim_size(i);
    }

    const TensorShape flattened_diag_shape(
        {1, diag_head_size, diag_tail_depth, diag_tail_width});

    DmlTensorInfo in_out_tensor;
    in_out_tensor.kernel_index = 0;
    in_out_tensor.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(0), flattened_in_out_shape,
                              flattened_in_out_shape);

    DmlTensorInfo diag_tensor;
    diag_tensor.kernel_index = 1;
    diag_tensor.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(1), flattened_diag_shape, flattened_diag_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {in_out_tensor, diag_tensor};
    tensors.outputs = {in_out_tensor};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);
    auto diag = dml::InputTensor(scope, 1, inputs[1]);

    auto ones =
        dml::ScalarTensor<int32_t>(scope, 1, input.GetOutputDesc().sizes);
    auto ones_diag =
        dml::ScalarTensor<int32_t>(scope, 1, diag.GetOutputDesc().sizes);
    auto ones_matrix =
        dml::MatrixDiag(scope, ones_diag, k0, k1, 0, height, width);
    auto diag_matrix = dml::MatrixDiag(scope, diag, k0, k1, 0, height, width);

    auto result = dml::If(ones_matrix == ones, diag_matrix, input);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

template <typename Op, typename T>
using K = typename KernelDefinition<
    Op,
    DmlKernelWrapper<
        DmlMatrixSetDiagKernel<T>,
        GetOutputShapeAsInputShapeHelper>>;

template <typename T, typename... Ts>
static void RegisterMatrixSetDiag()
{
  using Op = ops::MatrixSetDiag;
  K<Op, T>::template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::Register();
  
  if constexpr (sizeof...(Ts) > 0) RegisterMatrixSetDiag<Ts...>();
}

template <typename T, typename... Ts>
static void RegisterMatrixSetDiagV2()
{
  using Op = ops::MatrixSetDiagV2;
    K<Op, T>
      ::template WithHostMemoryArguments<Op::Argument::k>
      ::template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::Register();
  
  if constexpr (sizeof...(Ts) > 0) RegisterMatrixSetDiagV2<Ts...>();
}

template <typename T, typename... Ts>
static void RegisterMatrixSetDiagV3()
{
  using Op = ops::MatrixSetDiagV3;
    K<Op, T>
      ::template WithHostMemoryArguments<Op::Argument::k>
      ::template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::Register();
  
  if constexpr (sizeof...(Ts) > 0) RegisterMatrixSetDiagV3<Ts...>();
}

template <typename T, typename... Ts>
static void RegisterBatchMatrixSetDiag()
{
  using Op = ops::BatchMatrixSetDiag;
  K<Op, T>::template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::Register();

  if constexpr (sizeof...(Ts) > 0) RegisterBatchMatrixSetDiag<Ts...>();
}

void RegisterKernels_MatrixSetDiag()
{
    RegisterMatrixSetDiag<float, Eigen::half, bool>();
    RegisterMatrixSetDiagV2<float, Eigen::half, bool>();
    RegisterMatrixSetDiagV3<float, Eigen::half, bool>();
    RegisterBatchMatrixSetDiag<float, Eigen::half>();
}

}  // namespace tfdml
