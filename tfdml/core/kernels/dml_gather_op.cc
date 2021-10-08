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
#include "tfdml/core/util/attribute.h"
#include "tfdml/core/util/kernel_def_builder.h"
#include "tfdml/core/util/refcount.h"
#include "tfdml/core/util/resource_var.h"
#include "tfdml/core/util/types.h"

namespace tfdml {

template <typename Index>
class GatherInitializationHelper : public InitializationHelper {
 public:
  struct Attributes : public BaseAttributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      // Set attr->batch_dims to 0 if the attribute does not exist.
      if (!ctx->GetAttr("batch_dims", &batch_dims).ok()) {
        batch_dims = 0;
      }

      named_attributes_ = {
          {"batch_dims", batch_dims},
      };
    }

    absl::Span<const NameAttributePair> GetNamedAttributes() const final {
      return named_attributes_;
    }

    int32_t batch_dims;
    absl::InlinedVector<NameAttributePair, 1> named_attributes_;
  };

  GatherInitializationHelper(OpKernelContext* ctx,
                             std::shared_ptr<const Attributes> attr) {
    if (ctx->input(0).dtype() == TF_RESOURCE) {
      const Tensor handle_input = ctx->input(0);

      OP_REQUIRES_OK(
          ctx, LookupResource(
                   ctx, handle_input.base<tensorflow::ResourceHandleProto>()[0],
                   &params_resource_));
      params_resource_->mu()->lock_shared();
    }

    const Tensor params = GetParamsTensor(ctx);
    const Tensor indices = ctx->input(1);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    axis_ = 0;
    bool axis_is_set = false;  // Indicates whether the axis argument was set.
    if (ctx->num_inputs() == 3) {
      axis_is_set = true;
      const Tensor axis_tensor = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == TF_INT32) {
        axis_ = axis_tensor.base<int32_t>()[0];
      } else if (axis_tensor.dtype() == TF_INT64) {
        axis_ = axis_tensor.base<int64_t>()[0];
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    }

    int64_t min_params_dim = axis_ < 0 ? -axis_ : axis_ + 1;
    OP_REQUIRES(
        ctx, params.dims() >= min_params_dim,
        errors::InvalidArgument("Shape must be at least rank ", min_params_dim,
                                " but is rank ", params.dims()));

    if (axis_ < 0) {
      axis_ = params.dims() + axis_;
    }

    batch_dims_ = attr->batch_dims;
    if (batch_dims_ != 0) {
      OP_REQUIRES(
          ctx, batch_dims_ >= -indices.dims() && batch_dims_ <= indices.dims(),
          errors::InvalidArgument("Expected batch_dims in the range [",
                                  -indices.dims(), ", ", indices.dims(),
                                  "], but got ", batch_dims_));

      if (batch_dims_ < 0) {
        batch_dims_ = indices.dims() + batch_dims_;
      }

      if (!axis_is_set) {
        axis_ = batch_dims_;
      }

      OP_REQUIRES(ctx, batch_dims_ < params.dims(),
                  errors::InvalidArgument("batch_dims (", batch_dims_,
                                          ") must be less than rank(params) (",
                                          params.dims(), ")."));

      OP_REQUIRES(ctx, axis_ >= batch_dims_,
                  errors::InvalidArgument("batch_dims (", batch_dims_,
                                          ") must be less than or equal to ",
                                          "axis (", axis_, ")."));
      for (int i = 0; i < batch_dims_; ++i) {
        OP_REQUIRES(ctx, params.dim_size(i) == indices.dim_size(i),
                    errors::InvalidArgument(
                        "params.shape[", i, "]: ", params.dim_size(i),
                        " should be equal to indices.shape[", i,
                        "]: ", indices.dim_size(i)));
      }
    }

    // Check that we have enough index space
    int64_t gather_dim_size = params.dim_size(axis_);
    OP_REQUIRES(
        ctx, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis_, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));
  }

  int32_t GetBatchDims() const { return batch_dims_; }
  int64_t GetAxis() const { return axis_; }

  const Tensor GetParamsTensor(OpKernelContext* ctx) const {
    return ctx->input(0).dtype() == TF_RESOURCE ? *params_resource_->tensor()
                                                : ctx->input(0);
  }

  void Unlock() const {
    if (params_resource_) {
      params_resource_->mu()->unlock_shared();
    }
  }

 private:
  int64_t axis_;
  int32_t batch_dims_;
  RefCountPtr<Var> params_resource_;
};

template <typename TIndex>
class GatherShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const GatherInitializationHelper<TIndex>*>(
        initialization_helper);

    const Tensor params = init_helper->GetParamsTensor(ctx);
    const Tensor indices = ctx->input(1);

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape output_shape;

    int32_t batch_dims = init_helper->GetBatchDims();
    int64_t axis = init_helper->GetAxis();

    for (int i = 0; i < batch_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int i = batch_dims; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int i = batch_dims; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int i = axis + 1; i < params.dims(); ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    return {std::move(output_shape)};
  }
};

struct SimpleGather {
  dml::TensorDesc::Dimensions params_shape;
  dml::TensorDesc::Dimensions indices_shape;
  dml::TensorDesc::Dimensions output_shape;
  uint32_t gather_axis;
  uint32_t index_dimensions;
};

SimpleGather SimplifyGather(const TensorShape& params_shape,
                            const TensorShape& indices_shape, int64_t axis,
                            int32_t batch_dims) {
  // Collapse the batch dimensions together
  uint32_t collapsed_batch_dims = 1;
  for (int i = 0; i < batch_dims; ++i) {
    collapsed_batch_dims *= params_shape.dim_size(i);
  }

  // Collapse the non-batch dimensions to the left of the axis together
  uint32_t left_collapsed_dims = 1;
  for (int i = batch_dims; i < axis; ++i) {
    left_collapsed_dims *= params_shape.dim_size(i);
  }

  // Collapse all non-batch dimensions in Indices together
  uint32_t collapsed_indices_elements = 1;
  for (int i = batch_dims; i < indices_shape.dims(); ++i) {
    collapsed_indices_elements *= indices_shape.dim_size(i);
  }

  // Collapse the dimensions to the right of the axis together
  uint32_t right_collapsed_dims = 1;
  for (int i = axis + 1; i < params_shape.dims(); ++i) {
    right_collapsed_dims *= params_shape.dim_size(i);
  }

  uint32_t gather_dims = params_shape.dim_size(axis);

  SimpleGather desc = {};
  desc.params_shape = {collapsed_batch_dims, left_collapsed_dims, gather_dims,
                       right_collapsed_dims};
  desc.gather_axis = 2;

  if (batch_dims < indices_shape.dims()) {
    desc.indices_shape = {1, 1, collapsed_batch_dims,
                          collapsed_indices_elements};
    desc.output_shape = {collapsed_batch_dims, left_collapsed_dims,
                         collapsed_indices_elements, right_collapsed_dims};
    desc.index_dimensions = 1;
  } else {
    desc.indices_shape = {1, 1, 1, collapsed_batch_dims};
    desc.output_shape = {1, collapsed_batch_dims, left_collapsed_dims,
                         right_collapsed_dims};
    desc.index_dimensions = 0;
  }

  return desc;
}

template <typename TIndex, typename THostInputIndices>
class DmlGatherKernel : public DmlKernel {
 public:
  using InitHelper = GatherInitializationHelper<TIndex>;

  // TODO: Remove this when/if the following PR gets merged
  // https://github.com/tensorflow/tensorflow/pull/51759
  static constexpr auto host_input_indices =
      THostInputIndices::host_input_indices;
  static constexpr std::array<int, 0> host_output_indices = {};

  explicit DmlGatherKernel(
      DmlKernelConstruction* ctx,
      const GatherInitializationHelper<TIndex>* init_helper) {
    CHECK(ctx->GetInputCount() == 2 || ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    int32_t batch_dims = init_helper->GetBatchDims();
    int64_t axis = init_helper->GetAxis();

    SimpleGather simple_gather =
        SimplifyGather(params_tensor.shape(), indices_shape, axis, batch_dims);

    DmlTensorInfo params_input;
    params_input.kernel_index = 0;
    params_input.desc =
        DmlTensorDesc::Create(params_tensor.dtype(), simple_gather.params_shape,
                              simple_gather.params_shape);

    DmlTensorInfo indices_input;
    indices_input.kernel_index = 1;
    indices_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                               simple_gather.indices_shape,
                                               simple_gather.indices_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        simple_gather.output_shape,
                                        simple_gather.output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {params_input, indices_input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto indices_tensor = dml::InputTensor(scope, 1, inputs[1]);

    auto result =
        dml::Gather(input_tensor, indices_tensor, simple_gather.gather_axis,
                    simple_gather.index_dimensions);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    Tensor& output = ctx->GetOutputTensor(0);

    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    D3D12BufferRegion input_buffers[] = {
        ctx->GetDmlDeviceContext()->GetBufferForTensor(
            init_helper->GetParamsTensor(ctx->GetOpKernelContext())),
        ctx->GetDmlDeviceContext()->GetBufferForTensor(ctx->GetInputTensor(1)),
    };

    D3D12BufferRegion output_buffers[] = {
        ctx->GetDmlDeviceContext()->GetBufferForTensor(output)};

    // Create bindings
    auto input_bindings = dml_util::GetBufferBindings(input_buffers);
    auto output_bindings = dml_util::GetBufferBindings(output_buffers);

    auto gpu_event_or_status =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);

    init_helper->Unlock();
    return gpu_event_or_status;
  }
};

// TODO: Remove this when/if the following PR gets merged
// https://github.com/tensorflow/tensorflow/pull/51759
struct GatherHostInputIndices {
  static constexpr std::array<int, 0> host_input_indices = {};
};

struct GatherV2HostInputIndices {
  static constexpr std::array<int, 1> host_input_indices = {2};
};

struct ResourceGatherHostInputIndices {
  static constexpr std::array<int, 1> host_input_indices = {0};
};

template <typename TIndex, typename THostInputIndices>
using DmlGatherWrapper =
    DmlKernelWrapper<DmlGatherKernel<TIndex, THostInputIndices>,
                     GatherShapeHelper<TIndex>>;

template <typename TIndex>
using DmlResourceGatherWrapper =
    DmlKernelWrapper<DmlGatherKernel<TIndex, ResourceGatherHostInputIndices>,
                     GatherShapeHelper<TIndex>, DmlKernelCachePolicy::Never>;

#define DML_REGISTER_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(Name("Gather")                                       \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("Tparams")                 \
                              .TypeConstraint<int32_t>("Tindices"),            \
                          DmlGatherWrapper<int32_t, GatherHostInputIndices>)   \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                                     \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("Tparams")                 \
                              .TypeConstraint<int32_t>("Tindices")             \
                              .HostMemory("axis"),                             \
                          DmlGatherWrapper<int32_t, GatherV2HostInputIndices>) \
  REGISTER_KERNEL_BUILDER(Name("Gather")                                       \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("Tparams")                 \
                              .TypeConstraint<int64_t>("Tindices"),            \
                          DmlGatherWrapper<int64_t, GatherHostInputIndices>)   \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                                     \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("Tparams")                 \
                              .TypeConstraint<int64_t>("Tindices")             \
                              .HostMemory("axis"),                             \
                          DmlGatherWrapper<int64_t, GatherV2HostInputIndices>)

TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_bool(DML_REGISTER_KERNELS);
TF_CALL_int32(DML_REGISTER_KERNELS);
TF_CALL_int64(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                    \
                              .Device(DEVICE_DML)                   \
                              .HostMemory("resource")               \
                              .TypeConstraint<type>("dtype")        \
                              .TypeConstraint<int32_t>("Tindices"), \
                          DmlResourceGatherWrapper<int32_t>)        \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                    \
                              .Device(DEVICE_DML)                   \
                              .HostMemory("resource")               \
                              .TypeConstraint<type>("dtype")        \
                              .TypeConstraint<int64_t>("Tindices"), \
                          DmlResourceGatherWrapper<int64_t>)

TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_bool(DML_REGISTER_KERNELS);
TF_CALL_int64(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tfdml
