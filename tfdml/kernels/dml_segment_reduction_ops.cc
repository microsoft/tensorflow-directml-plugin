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
template <DML_REDUCE_FUNCTION reduce_function>
class DmlUnsortedSegmentReductionKernel : public OpKernel
{
  public:
    explicit DmlUnsortedSegmentReductionKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
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
        const char* operator_type = nullptr;
        switch (reduce_function)
        {
        case DML_REDUCE_FUNCTION_SUM:
            operator_type = "UnsortedSegmentSum";
            break;
        case DML_REDUCE_FUNCTION_MAX:
            operator_type = "UnsortedSegmentMax";
            break;
        case DML_REDUCE_FUNCTION_MIN:
            operator_type = "UnsortedSegmentMin";
            break;
        case DML_REDUCE_FUNCTION_MULTIPLY:
            operator_type = "UnsortedSegmentProd";
            break;
        default:
            OP_REQUIRES(
                ctx,
                false,
                errors::InvalidArgument(
                    "Unsupported segment reduction function."));
        }

        Status status;
        TFE_Op* unsorted_segment_op =
            TFE_NewOp(eager_context_, operator_type, status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto unsorted_segment_op_cleanup = absl::MakeCleanup(
            [unsorted_segment_op] { TFE_DeleteOp(unsorted_segment_op); });

        TFE_OpSetDevice(unsorted_segment_op, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);

        // data_tensor is originally on the GPU, so copy it over to the CPU
        // before executing the CPU operator
        const Tensor& data_tensor = ctx->input(0);
        Tensor data_tensor_cpu;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                data_tensor.dtype(),
                data_tensor.shape(),
                &data_tensor_cpu,
                true));
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyDeviceTensorToCPU(
                &data_tensor,
                &data_tensor_cpu));
        TFE_TensorHandle* data_handle =
            TFE_NewTensorHandle(data_tensor_cpu.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto input_handle_cleanup = absl::MakeCleanup(
            [data_handle] { TFE_DeleteTensorHandle(data_handle); });
        TFE_OpAddInput(unsorted_segment_op, data_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // segment_ids is originally on the GPU, so copy it over to the CPU
        // before executing the CPU operator
        const Tensor& segment_ids_tensor = ctx->input(1);
        Tensor segment_ids_tensor_cpu;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                segment_ids_tensor.dtype(),
                segment_ids_tensor.shape(),
                &segment_ids_tensor_cpu,
                true));
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyDeviceTensorToCPU(
                &segment_ids_tensor,
                &segment_ids_tensor_cpu));
        TFE_TensorHandle* segment_ids_handle =
            TFE_NewTensorHandle(segment_ids_tensor_cpu.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto segment_ids_handle_cleanup =
            absl::MakeCleanup([segment_ids_handle]
                              { TFE_DeleteTensorHandle(segment_ids_handle); });
        TFE_OpAddInput(unsorted_segment_op, segment_ids_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // num_segments is already on the CPU
        const Tensor& num_segments_tensor = ctx->input(2);
        TFE_TensorHandle* num_segments_handle =
            TFE_NewTensorHandle(num_segments_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto num_segments_handle_cleanup =
            absl::MakeCleanup([num_segments_handle]
                              { TFE_DeleteTensorHandle(num_segments_handle); });
        TFE_OpAddInput(unsorted_segment_op, num_segments_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_TensorHandle* output_handle = nullptr;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup = absl::MakeCleanup(
            [output_handle] { TFE_DeleteTensorHandle(output_handle); });

        int num_retvals = 1;
        TFE_Execute(
            unsorted_segment_op,
            &output_handle,
            &num_retvals,
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        Tensor output_cpu =
            Tensor(TFE_TensorHandleResolve(output_handle, status.raw()));
        OP_REQUIRES_OK(ctx, status);

        // Copy the CPU output back to the device
        auto status_or_tensor = ctx->allocate_output(0, output_cpu.shape());
        OP_REQUIRES_OK(ctx, status_or_tensor.status());

        Tensor& output = status_or_tensor.ValueOrDie();
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyCPUTensorToDevice(&output_cpu, &output));
    }

  private:
    TFE_Context* eager_context_ = nullptr;
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
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_SUM>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentSum::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentSum::Attribute::Tindices,
                TF_INT32>;

    using int64_kernel = KernelDefinition<
        ops::UnsortedSegmentSum,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_SUM>>::
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

void RegisterUnsortedSegmentMax()
{
    using int32_kernel = KernelDefinition<
        ops::UnsortedSegmentMax,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MAX>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentMax::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentMax::Attribute::Tindices,
                TF_INT32>;

    using int64_kernel = KernelDefinition<
        ops::UnsortedSegmentMax,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MAX>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentMax::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentMax::Attribute::Tindices,
                TF_INT64>;

    RegisterWithTypes<
        int32_kernel,
        ops::UnsortedSegmentMax::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();

    RegisterWithTypes<
        int64_kernel,
        ops::UnsortedSegmentMax::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();
}

void RegisterUnsortedSegmentMin()
{
    using int32_kernel = KernelDefinition<
        ops::UnsortedSegmentMin,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MIN>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentMin::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentMin::Attribute::Tindices,
                TF_INT32>;

    using int64_kernel = KernelDefinition<
        ops::UnsortedSegmentMin,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MIN>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentMin::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentMin::Attribute::Tindices,
                TF_INT64>;

    RegisterWithTypes<
        int32_kernel,
        ops::UnsortedSegmentMin::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();

    RegisterWithTypes<
        int64_kernel,
        ops::UnsortedSegmentMin::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();
}

void RegisterUnsortedSegmentProd()
{
    using int32_kernel = KernelDefinition<
        ops::UnsortedSegmentProd,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MULTIPLY>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentProd::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentProd::Attribute::Tindices,
                TF_INT32>;

    using int64_kernel = KernelDefinition<
        ops::UnsortedSegmentProd,
        DmlUnsortedSegmentReductionKernel<DML_REDUCE_FUNCTION_MULTIPLY>>::
        WithHostMemoryArguments<
            ops::UnsortedSegmentProd::Argument::num_segments>::
            WithTypeConstraint<
                ops::UnsortedSegmentProd::Attribute::Tindices,
                TF_INT64>;

    RegisterWithTypes<
        int32_kernel,
        ops::UnsortedSegmentProd::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();

    RegisterWithTypes<
        int64_kernel,
        ops::UnsortedSegmentProd::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();
}

void RegisterKernels_SegmentReduction()
{
    RegisterUnsortedSegmentSum();
    RegisterUnsortedSegmentMax();
    RegisterUnsortedSegmentMin();
    RegisterUnsortedSegmentProd();
}

} // namespace tfdml