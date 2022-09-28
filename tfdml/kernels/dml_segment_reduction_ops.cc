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

        unsorted_segment_op_ =
            TFE_NewOp(eager_context_, operator_type, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_OpSetDevice(unsorted_segment_op_, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlUnsortedSegmentReductionKernel() override
    {
        TFE_DeleteOp(unsorted_segment_op_);
        TFE_DeleteContext(eager_context_);
    }

    void Compute(OpKernelContext* ctx)
    {
        // data and segment_ids are originally on the GPU, so copy them over to
        // the CPU before executing the CPU operator
        absl::InlinedVector<Tensor, 3> dml_tensors = {
            ctx->input(0),
            ctx->input(1),
        };
        absl::InlinedVector<Tensor, 3> cpu_tensors;

        for (int i = 0; i < dml_tensors.size(); ++i)
        {
            const Tensor& input_tensor = dml_tensors[i];
            Tensor input_tensor_cpu;
            OP_REQUIRES_OK(
                ctx,
                ctx->allocate_temp(
                    input_tensor.dtype(),
                    input_tensor.shape(),
                    &input_tensor_cpu,
                    true));

            cpu_tensors.push_back(std::move(input_tensor_cpu));
        }

        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyDeviceTensorsToCPU(
                dml_tensors,
                absl::Span<Tensor>(cpu_tensors)));

        // num_segments is already on the CPU
        cpu_tensors.push_back(ctx->input(2));

        absl::InlinedVector<TFE_TensorHandle*, 3> handles;
        auto handles_cleanup = absl::MakeCleanup(
            [&handles]
            {
                for (TFE_TensorHandle* handle : handles)
                {
                    TFE_DeleteTensorHandle(handle);
                }
            });

        Status status;
        for (const Tensor& cpu_tensor : cpu_tensors)
        {
            TFE_TensorHandle* handle =
                TFE_NewTensorHandle(cpu_tensor.raw(), status.raw());
            OP_REQUIRES_OK(ctx, status);
            handles.push_back(handle);

            TFE_OpAddInput(unsorted_segment_op_, handle, status.raw());
            OP_REQUIRES_OK(ctx, status);
        }

        TFE_TensorHandle* output_handle = nullptr;
        TFE_TensorHandle** output_handle_ptr = &output_handle;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup =
            absl::MakeCleanup([output_handle_ptr]
                              { TFE_DeleteTensorHandle(*output_handle_ptr); });

        int num_retvals = 1;
        TFE_Execute(
            unsorted_segment_op_,
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
    TFE_Op* unsorted_segment_op_ = nullptr;
};

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