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
#include "tfdml/runtime_adapter/stream.h"

namespace tfdml
{

static void SetTensorToZero(DmlDevice* dml_device, const Tensor& tensor)
{
    auto device_context = dml_device->GetDeviceContext();

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

  private:
    void ComputeImpl(OpKernelContext* ctx) final
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
            auto dml_device = static_cast<DmlDevice*>(ctx->device());
            SetTensorToZero(dml_device, status_or_output.ValueOrDie());
        }
    }
};

class DummyInitializationHelper : public InitializationHelper
{
};

static inline bool IsSupportedAddType(TF_DataType dtype)
{
    switch (dtype)
    {
    case TF_FLOAT:
    case TF_HALF:
    case TF_INT64:
    case TF_INT32:
    case TF_INT16:
    case TF_INT8:
    case TF_UINT64:
    case TF_UINT32:
    case TF_UINT16:
    case TF_UINT8:
    case TF_BOOL: return true;
    default: return false;
    }
}

static void ZerosLikeVariant(
    TF_OpKernelContext* ctx,
    TF_Tensor* raw_input_tensor,
    TF_Tensor* raw_out_tensor)
{
    Tensor input_tensor(raw_input_tensor);
    Tensor output_tensor(raw_out_tensor);

    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    DmlDevice* dml_device = static_cast<DmlDevice*>(device);

    if (!IsSupportedAddType(input_tensor.dtype()))
    {
        TF_OpKernelContext_Failure(
            ctx,
            errors::InvalidArgument(
                DataTypeString(input_tensor.dtype()),
                " is not a supported type for ZerosLike.")
                .raw());
        return;
    }

    SetTensorToZero(dml_device, output_tensor);
}

class DmlZerosLikeVariantKernel : public OpKernel
{
  public:
    explicit DmlZerosLikeVariantKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

  private:
    void ComputeImpl(OpKernelContext* ctx) final
    {
        OP_REQUIRES_OK(ctx, ctx->ZerosLikeVariant(ZerosLikeVariant));
    }
};

void RegisterKernels_ZerosLike()
{
    using K = KernelDefinition<ops::ZerosLike, DmlZerosLikeKernel>;

    constexpr auto T = ops::ZerosLike::Attribute::T;
    RegisterWithTypes<K, T, TF_FLOAT, TF_HALF, TF_INT64, TF_BOOL>();

    using VariantK =
        KernelDefinition<ops::ZerosLike, DmlZerosLikeVariantKernel>;
    VariantK::WithTypeConstraint<T, TF_VARIANT>::Register();
}

} // namespace tfdml
