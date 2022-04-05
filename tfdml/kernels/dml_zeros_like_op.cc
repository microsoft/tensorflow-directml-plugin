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

static void SetTensorToZero(OpKernelContext* ctx, const Tensor& tensor)
{
    auto device_context =
        static_cast<DmlDevice*>(ctx->device())->GetDeviceContext();

    D3D12BufferRegion output_buffer =
        device_context->GetBufferForTensor(tensor);

    device_context->ZeroBuffer(output_buffer);
}

class DmlZerosLikeKernel : public OpKernel
{
  public:
    explicit DmlZerosLikeKernel(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        int candidate_input_indices[] = {0};

        StatusOr<Tensor> status_or_output =
            ctx->forward_input_or_allocate_output(
                candidate_input_indices,
                0,
                ctx->input(0).shape());

        OP_REQUIRES_OK(ctx, status_or_output.status());
        if (status_or_output.ValueOrDie().NumElements() > 0)
        {
            SetTensorToZero(ctx, status_or_output.ValueOrDie());
        }
    }
};

void RegisterKernels_ZerosLike()
{
    using K = KernelDefinition<ops::ZerosLike, DmlZerosLikeKernel>;

    RegisterWithTypes<
        K,
        ops::ZerosLike::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64,
        TF_BOOL>();
}

} // namespace tfdml
