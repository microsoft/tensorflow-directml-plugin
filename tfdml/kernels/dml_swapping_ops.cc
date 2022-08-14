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

class DmlCopyFromGpuToHost : public OpKernel
{
  public:
    explicit DmlCopyFromGpuToHost(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& input = ctx->input(0);
        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, input.shape());

        OP_REQUIRES_OK(ctx, status_or_output.status());
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyDeviceTensorToCPU(
                &input,
                &status_or_output.ValueOrDie()));
    }
};

class DmlCopyFromHostToGpu : public OpKernel
{
  public:
    explicit DmlCopyFromHostToGpu(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& input = ctx->input(0);
        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, input.shape());

        OP_REQUIRES_OK(ctx, status_or_output.status());
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyCPUTensorToDevice(
                &input,
                &status_or_output.ValueOrDie()));
    }
};

namespace ops
{
struct _CopyFromGpuToHost
{
    static constexpr const char* name = "_CopyFromGpuToHost";

    enum class Argument
    {
        input,
        output,
    };

    static constexpr uint32_t input_arg_count = 1;
    static constexpr uint32_t output_arg_count = 1;
    static constexpr std::
        array<ArgumentDesc, input_arg_count + output_arg_count>
            argument_descs{
                ArgumentDesc{"input", ArgumentDesc::TensorCount::Single},
                ArgumentDesc{"output", ArgumentDesc::TensorCount::Single},
            };

    enum class Attribute
    {
        T
    };

    static constexpr std::array<AttributeDesc, 1> attribute_descs{
        AttributeDesc{"T", AttributeType::Type},
    };
};

constexpr std::array<AttributeDesc, 1> _CopyFromGpuToHost::attribute_descs;
constexpr std::array<
    ArgumentDesc,
    _CopyFromGpuToHost::input_arg_count + _CopyFromGpuToHost::output_arg_count>
    _CopyFromGpuToHost::argument_descs;
} // namespace ops

namespace ops
{
struct _CopyFromHostToGpu
{
    static constexpr const char* name = "_CopyFromHostToGpu";

    enum class Argument
    {
        input,
        output,
    };

    static constexpr uint32_t input_arg_count = 1;
    static constexpr uint32_t output_arg_count = 1;
    static constexpr std::
        array<ArgumentDesc, input_arg_count + output_arg_count>
            argument_descs{
                ArgumentDesc{"input", ArgumentDesc::TensorCount::Single},
                ArgumentDesc{"output", ArgumentDesc::TensorCount::Single},
            };

    enum class Attribute
    {
        T
    };

    static constexpr std::array<AttributeDesc, 1> attribute_descs{
        AttributeDesc{"T", AttributeType::Type},
    };
};

constexpr std::array<AttributeDesc, 1> _CopyFromHostToGpu::attribute_descs;
constexpr std::array<
    ArgumentDesc,
    _CopyFromHostToGpu::input_arg_count + _CopyFromHostToGpu::output_arg_count>
    _CopyFromHostToGpu::argument_descs;
} // namespace ops

static void RegisterKernels_CopyFromGpuToHost()
{
    using K = KernelDefinition<ops::_CopyFromGpuToHost, DmlCopyFromGpuToHost>::
        WithHostMemoryArguments<ops::_CopyFromGpuToHost::Argument::output>;
    K::Register();
}

static void RegisterKernels_CopyFromHostToGpu()
{
    using K = KernelDefinition<ops::_CopyFromHostToGpu, DmlCopyFromHostToGpu>::
        WithHostMemoryArguments<ops::_CopyFromHostToGpu::Argument::input>;
    K::Register();
}

void RegisterKernels_Swapping()
{
    RegisterKernels_CopyFromGpuToHost();
    RegisterKernels_CopyFromHostToGpu();
}

} // namespace tfdml
