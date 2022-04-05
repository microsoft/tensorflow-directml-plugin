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

class DmlDeepCopyKernel : public OpKernel
{
  public:
    explicit DmlDeepCopyKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& input = ctx->input(0);
        const TensorShape& input_shape = input.shape();

        StatusOr<Tensor> status_or_output_tensor =
            ctx->allocate_output(0, input_shape);
        OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

        DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
        auto* device_context = device->GetDeviceContext();

        if (input.NumElements() == 0)
        {
            return;
        }

        D3D12BufferRegion input_buffer =
            device_context->GetBufferForTensor(input);

        D3D12BufferRegion output_buffer = device_context->GetBufferForTensor(
            status_or_output_tensor.ValueOrDie());

        uint64_t copy_size =
            std::min(output_buffer.SizeInBytes(), input_buffer.SizeInBytes());

        device_context->CopyBufferToBuffer(
            output_buffer,
            input_buffer.Subregion(0, copy_size));
    }
};

void RegisterKernels_DeepCopy()
{
    using K = KernelDefinition<ops::DeepCopy, DmlDeepCopyKernel>;

    RegisterWithTypes<
        K,
        ops::DeepCopy::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

} // namespace tfdml
