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

#include "tfdml/kernels/pch.h"

namespace tfdml
{

class DmlFailureKernel : public OpKernel {
 public:
  explicit DmlFailureKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def)) {
    OP_REQUIRES_OK(ctx,
                   errors::Internal("Found instance of parallel_stack which "
                                    "could not be properly replaced."));
  }

  void Compute(OpKernelContext*) {}
};

class DmlParallelConcatStartKernel : public OpKernel
{
  public:
    explicit DmlParallelConcatStartKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    }

    void Compute(OpKernelContext* ctx)
    {
        StatusOr<Tensor> status_or_output_tensor =
            ctx->allocate_output(0, shape_);
        OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

        DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());
        Status sync_status = dml_device->Sync();
    }

  private:
    TensorShape shape_;
};

class DmlParallelConcatUpdateKernel : public OpKernel
{
  public:
    explicit DmlParallelConcatUpdateKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& value_tensor = ctx->input(0);

        // Value should be at least rank 1. Also the 0th dimension should be
        // at least loc_.
        OP_REQUIRES(ctx, value_tensor.dims() >= 1,
                    errors::InvalidArgument("value should be at least rank 1."));
        OP_REQUIRES(
            ctx, value_tensor.dim_size(0) > loc_,
            errors::InvalidArgument("0th dimension of value = ", value_tensor.dim_size(0),
                                    " is less than loc_=", loc_));

        const Tensor& update_tensor = ctx->input(1);

        OP_REQUIRES(
            ctx,
            value_tensor.dims() == update_tensor.dims(),
            errors::InvalidArgument(
                "value and update shape doesn't match: ",
                value_tensor.shape().DebugString(),
                " vs. ",
                update_tensor.shape().DebugString()));
        for (int i = 1; i < value_tensor.dims(); ++i)
        {
            OP_REQUIRES(
                ctx,
                value_tensor.dim_size(i) == update_tensor.dim_size(i),
                errors::InvalidArgument(
                    "value and update shape doesn't match ",
                    value_tensor.shape().DebugString(),
                    " vs. ",
                    update_tensor.shape().DebugString()));
        }
        OP_REQUIRES(
            ctx,
            1 == update_tensor.dim_size(0),
            errors::InvalidArgument(
                "update shape doesn't match: ",
                update_tensor.shape().DebugString()));

        DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());
        auto device_context = dml_device->GetDeviceContext();

        // This creates an alias intentionally
        Tensor output_tensor = value_tensor;

        D3D12BufferRegion update_buffer =
            device_context->GetBufferForTensor(update_tensor);
        D3D12BufferRegion output_buffer =
            device_context->GetBufferForTensor(output_tensor);

        const int64_t nrows = output_tensor.dim_size(0);
        const int dtype_size_in_bytes = DataTypeSize(output_tensor.dtype());
        const uint64_t stride =
            dtype_size_in_bytes * output_tensor.NumElements() / nrows;

        // Guard the row index range
        const int64_t row_index = (loc_ % nrows + nrows) % nrows;
        const uint64_t dst_offset = row_index * stride;

        device_context->CopyBufferToBuffer(
            output_buffer.Subregion(dst_offset),
            update_buffer.Subregion(0, stride));

        ctx->set_output(0, output_tensor);
    }

  private:
    int32_t loc_;
};

static void RegisterParallelConcat()
{
    using K = KernelDefinition<
        ops::ParallelConcat,
        DmlFailureKernel>;

    RegisterWithTypes<
        K,
        ops::ParallelConcat::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterParallelConcatStart()
{
    using K = KernelDefinition<
        ops::_ParallelConcatStart,
        DmlParallelConcatStartKernel>;

    RegisterWithTypes<
        K,
        ops::_ParallelConcatStart::Attribute::dtype,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterParallelConcatUpdate()
{
    using K = KernelDefinition<
        ops::_ParallelConcatUpdate,
        DmlParallelConcatUpdateKernel>;

    RegisterWithTypes<
        K,
        ops::_ParallelConcatUpdate::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_ParallelConcat()
{
    RegisterParallelConcat();
    RegisterParallelConcatStart();
    RegisterParallelConcatUpdate();
}

} // namespace tfdml
