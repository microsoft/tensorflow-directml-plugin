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

#include "tfdml/core/common_runtime/dml/dml_operator_helper.h"
#include "tfdml/core/common_runtime/dml/dml_util.h"
#include "tfdml/core/kernels/dml_kernel_wrapper.h"
#include "tfdml/core/kernels/dml_ops_common.h"
#include "tfdml/core/util/kernel_def_builder.h"

namespace tfdml {

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

template <AxisArgumentName AxisArgName>
class ConcatInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  ConcatInitHelper(OpKernelContext* ctx,
                   std::shared_ptr<const Attributes> attr) {

    const char* axis_attribute_name = AxisArgName == NAME_IS_AXIS ? "axis"
                                      : AxisArgName == NAME_IS_CONCAT_DIM
                                          ? "concat_dim"
                                          : "<invalid>";

    const int num_inputs = ctx->num_inputs();
    int axis_index = AxisArgName == NAME_IS_CONCAT_DIM ? 0 : num_inputs - 1;

    const Tensor concat_dim_tensor = ctx->input(axis_index);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));
    
    std::vector<Tensor> values;
    int values_begin = AxisArgName == NAME_IS_CONCAT_DIM ? 1 : 0;
    int values_end = AxisArgName == NAME_IS_CONCAT_DIM ? num_inputs : num_inputs - 1;
    for (int i = values_begin; i < values_end; ++i) {
      values.push_back(std::move(ctx->input(i)));
    }

    const int input_dims = values[0].dims();
    first_input_shape_ = values[0].shape();

    CHECK(concat_dim_tensor.shape().dims() == 0);
    int64_t concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (AxisArgName == NAME_IS_AXIS) {
      CHECK(concat_dim_tensor.dtype() == TF_INT32 ||
            concat_dim_tensor.dtype() == TF_INT64);
    } else {
      CHECK(concat_dim_tensor.dtype() == TF_INT32);
    }
    if (concat_dim_tensor.dtype() == TF_INT32) {
      concat_dim = concat_dim_tensor.base<int32_t>()[0];
    } else {
      concat_dim = concat_dim_tensor.base<int64_t>()[0];
    }

    concat_axis_ = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(ctx, 0 <= concat_axis_ && concat_axis_ < input_dims,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    output_concat_dim_size_ = first_input_shape_.dim_size(concat_axis_);
    for (int i = 1; i < values.size(); ++i) {
      const auto& in = values[i];
      OP_REQUIRES(
          ctx, in.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              first_input_shape_.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      output_concat_dim_size_ += in.dim_size(concat_axis_);

      for (int j = 0; j < in.dims(); ++j) {
        if (j != concat_axis_) {
          OP_REQUIRES(
              ctx, in.dim_size(j) == first_input_shape_.dim_size(j),
              errors::InvalidArgument(
                  "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                  first_input_shape_.DebugString(), " vs. shape[", i,
                  "] = ", in.shape().DebugString()));
        }
      }
    }
  }

  // Concat is only a no-op if all input tensors (aside from the concat dim
  // tensor) are empty
  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    uint32_t concat_dim_tensor_index =
        AxisArgName == NAME_IS_CONCAT_DIM ? 0 : ctx->num_inputs() - 1;

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      if (i == concat_dim_tensor_index) {
        continue;
      }

      if (ctx->input(i).NumElements() != 0) {
        return false;
      }
    }

    return true;
  }

  int64_t GetConcatAxis() const { return concat_axis_; }
  int64_t GetOutputConcatDimSize() const { return output_concat_dim_size_; }
  const TensorShape& GetFirstInputShape() const { return first_input_shape_; }

 private:
  int64_t concat_axis_;
  int64_t output_concat_dim_size_;
  TensorShape first_input_shape_;
};

template <AxisArgumentName AxisArgName>
using InitHelper = ConcatInitHelper<AxisArgName>;

template <AxisArgumentName AxisArgName>
class ConcatShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const InitHelper<AxisArgName>*>(initialization_helper);

    int64_t concat_axis = init_helper->GetConcatAxis();
    int64_t output_concat_dim = init_helper->GetOutputConcatDimSize();
    const TensorShape& first_input_shape = init_helper->GetFirstInputShape();

    TensorShape output_shape(first_input_shape);
    output_shape.set_dim(concat_axis, output_concat_dim);

    return {std::move(output_shape)};
  }
};

template <AxisArgumentName AxisArgName, typename THostInputIndices>
class DmlConcatKernel : public DmlKernel {
 public:
  using InitHelper = InitHelper<AxisArgName>;

  // TODO: Remove this when/if the following PR gets merged
  // https://github.com/tensorflow/tensorflow/pull/51759
  static constexpr std::array<int, 1> host_input_indices =
      THostInputIndices::host_input_indices;
  static constexpr std::array<int, 0> host_output_indices = {};

  explicit DmlConcatKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    // We need to concatenate at least 2 tensors, and we need an additional
    // tensor for the concatenation axis
    CHECK(ctx->GetInputCount() >= 3);
    CHECK(ctx->GetOutputCount() == 1);

    uint32_t first_concat_tensor_index;
    uint32_t concat_dim_tensor_index;
    if constexpr (AxisArgName == NAME_IS_CONCAT_DIM) {
      // For Concat, the inputs come AFTER the axis
      first_concat_tensor_index = 1;
      concat_dim_tensor_index = 0;
    } else {
      // For ConcatV2, the inputs come BEFORE the axis
      first_concat_tensor_index = 0;
      concat_dim_tensor_index = ctx->GetInputCount() - 1;
    }

    DmlKernelTensors tensors;
    const TensorShape& first_input_shape = ctx->GetInputTensorShape(0);

    int64_t concat_axis = init_helper->GetConcatAxis();

    // We can collapse all dimensions to the left together and all dimensions
    // to the right together. This allows us to send tensors with an "unlimited"
    // number of dimensions to DirectML
    int left_dim_size = 1;
    int right_dim_size = 1;
    TensorShape output_shape = ctx->GetOutputTensorShape(0);

    for (int i = 0; i < concat_axis; ++i) {
      left_dim_size *= output_shape.dim_size(i);
    }

    for (int i = concat_axis + 1; i < output_shape.dims(); ++i) {
      right_dim_size *= output_shape.dim_size(i);
    }

    int axis_size = output_shape.dim_size(concat_axis);
    output_shape = TensorShape({left_dim_size, axis_size, right_dim_size});

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                        output_shape);
    tensors.outputs = {output};

    // DML doesn't support empty tensors, so filter them out when generating the
    // kernel input indices (which is what determines the mapping between kernel
    // inputs and DML op inputs)
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
      if (i == concat_dim_tensor_index) {
        continue;  // Ignore the concat axis tensor
      }

      if (ctx->GetInputTensorShape(i).num_elements() == 0) {
        // Empty tensor; ignore this input
        continue;
      }

      int axis_dim_size = ctx->GetInputTensorShape(i).dim_size(concat_axis);
      TensorShape tensor_shape({left_dim_size, axis_dim_size, right_dim_size});

      DmlTensorInfo input_info;
      input_info.kernel_index = i;
      input_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i),
                                              tensor_shape, tensor_shape);
      tensors.inputs.push_back(std::move(input_info));
    }

    // If all tensors are empty, this kernel should have already been no-op'd
    // earlier
    CHECK(!tensors.inputs.empty());

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());

    absl::InlinedVector<dml::Expression, 5> input_tensors;
    input_tensors.reserve(inputs.size());

    for (int i = 0; i < inputs.size(); ++i) {
      input_tensors.push_back(dml::InputTensor(scope, i, inputs[i]));
    }

    auto result = dml::Join(input_tensors, kNchwDimensionCount - 2);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

// TODO: Remove this when/if the following PR gets merged
// https://github.com/tensorflow/tensorflow/pull/51759
struct ConcatHostInputIndices {
  static constexpr std::array<int, 1> host_input_indices = {0};
};

struct ConcatV2HostInputIndices {
  static constexpr std::array<int, 1> host_input_indices = {-1};
};

template <AxisArgumentName AxisArgName, typename THostInputIndices>
using DmlConcatWrapper = DmlKernelWrapper<DmlConcatKernel<AxisArgName, THostInputIndices>,
                                          ConcatShapeHelper<AxisArgName>>;

#define REGISTER_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Concat")                        \
                              .Device(DEVICE_DML)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("concat_dim"),        \
                          DmlConcatWrapper<NAME_IS_CONCAT_DIM, ConcatHostInputIndices>) \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")                      \
                              .Device(DEVICE_DML)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("axis"),              \
                          DmlConcatWrapper<NAME_IS_AXIS, ConcatV2HostInputIndices>)

//TODO: add uint64 support
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_uint8(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tfdml