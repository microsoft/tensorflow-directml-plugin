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

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/eager/c_api.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{
class DmlUnsortedSegmentReductionKernel : public OpKernel
{
  public:
    explicit DmlUnsortedSegmentReductionKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask_));
        OP_REQUIRES_OK(
            ctx,
            ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask_));

        TFE_ContextOptions* context_options = TFE_NewContextOptions();
        auto context_options_cleanup = absl::MakeCleanup(
            [context_options] { TFE_DeleteContextOptions(context_options); });

        Status status;
        eager_context_ = TFE_NewContext(context_options, status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlUnsortedSegmentReductionKernel() override
    {
        if (eager_context_)
        {
            TFE_DeleteContext(eager_context_);
        }
    }

    void Compute(OpKernelContext* ctx)
    {
        Status status;
        TFE_Op* strided_slice_op =
            TFE_NewOp(eager_context_, "StridedSlice", status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto strided_slice_op_cleanup = absl::MakeCleanup(
            [strided_slice_op] { TFE_DeleteOp(strided_slice_op); });

        TFE_OpSetAttrInt(strided_slice_op, "begin_mask", begin_mask_);
        TFE_OpSetAttrInt(strided_slice_op, "end_mask", end_mask_);
        TFE_OpSetAttrInt(strided_slice_op, "ellipsis_mask", ellipsis_mask_);
        TFE_OpSetAttrInt(strided_slice_op, "new_axis_mask", new_axis_mask_);
        TFE_OpSetAttrInt(
            strided_slice_op,
            "shrink_axis_mask",
            shrink_axis_mask_);

        TFE_OpSetDevice(strided_slice_op, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);

        const Tensor& input_tensor = ctx->input(0);
        TFE_TensorHandle* input_handle =
            TFE_NewTensorHandle(input_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto input_handle_cleanup = absl::MakeCleanup(
            [input_handle] { TFE_DeleteTensorHandle(input_handle); });
        TFE_OpAddInput(strided_slice_op, input_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        const Tensor& begin_tensor = ctx->input(1);
        TFE_TensorHandle* begin_handle =
            TFE_NewTensorHandle(begin_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto begin_handle_cleanup = absl::MakeCleanup(
            [begin_handle] { TFE_DeleteTensorHandle(begin_handle); });
        TFE_OpAddInput(strided_slice_op, begin_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        const Tensor& end_tensor = ctx->input(2);
        TFE_TensorHandle* end_handle =
            TFE_NewTensorHandle(end_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto end_handle_cleanup = absl::MakeCleanup(
            [end_handle] { TFE_DeleteTensorHandle(end_handle); });
        TFE_OpAddInput(strided_slice_op, end_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        const Tensor& strides_tensor = ctx->input(3);
        TFE_TensorHandle* strides_handle =
            TFE_NewTensorHandle(strides_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto strides_handle_cleanup = absl::MakeCleanup(
            [strides_handle] { TFE_DeleteTensorHandle(strides_handle); });
        TFE_OpAddInput(strided_slice_op, strides_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_TensorHandle* output_handle = nullptr;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup = absl::MakeCleanup(
            [output_handle] { TFE_DeleteTensorHandle(output_handle); });

        int num_retvals = 1;
        TFE_Execute(
            strided_slice_op,
            &output_handle,
            &num_retvals,
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        TF_Tensor* output =
            TFE_TensorHandleResolve(output_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        OP_REQUIRES_OK(ctx, ctx->set_output(0, Tensor(output)));
    }

  private:
    TFE_Context* eager_context_ = nullptr;
    int32_t begin_mask_;
    int32_t end_mask_;
    int32_t ellipsis_mask_;
    int32_t new_axis_mask_;
    int32_t shrink_axis_mask_;
};

#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                   \
    name,                                                                      \
    type,                                                                      \
    index_type,                                                                \
    initial_value_functor,                                                     \
    reduction_kernel_functor)                                                  \
    REGISTER_KERNEL_BUILDER(                                                   \
        Name(name)                                                             \
            .Device(DEVICE_GPU)                                                \
            .HostMemory("num_segments")                                        \
            .TypeConstraint<type>("T")                                         \
            .TypeConstraint<index_type>("Tindices"),                           \
        UnsortedSegmentReductionOp<                                            \
            type,                                                              \
            index_type,                                                        \
            functor::UnsortedSegmentFunctor<                                   \
                GPUDevice,                                                     \
                type,                                                          \
                index_type,                                                    \
                initial_value_functor,                                         \
                reduction_kernel_functor>>)

void RegisterUnsortedSegmentSum()
{
    using int32_kernel = KernelDefinition<
        ops::UnsortedSegmentSum,
        DmlUnsortedSegmentReductionKernel>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentSum::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentSum::Attribute::Tindices,
                TF_INT32>;

    using int64_kernel = KernelDefinition<
        ops::UnsortedSegmentSum,
        DmlUnsortedSegmentReductionKernel>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentSum::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentSum::Attribute::Tindices,
                TF_INT64>;

    RegisterWithTypes<
        int32_kernel,
        ops::UnsortedSegmentSum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();

    RegisterWithTypes<
        int64_kernel,
        ops::UnsortedSegmentSum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();
}

void RegisterKernels_SegmentReduction() { RegisterUnsortedSegmentSum(); }

} // namespace tfdml