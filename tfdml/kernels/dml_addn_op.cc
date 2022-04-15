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

class DmlAddNKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlAddNKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // AddN doesn't support broadcasting, so we can simply collapse all
        // dimensions into a single one to support more than 5D
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

        DmlKernelTensors tensors;

        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                tensor_shape,
                tensor_shape);

            tensors.inputs.push_back(std::move(input));
        }

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape);

        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        if (ctx->GetInputCount() == 1)
        {
            auto outputs = GetDmlTensorDescs(tensors.outputs);

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
            identity_desc.InputTensor = inputs.data();
            identity_desc.OutputTensor = outputs.data();

            DML_OPERATOR_DESC op_desc = {
                DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                &identity_desc};
            Initialize(ctx, std::move(tensors), op_desc);
        }
        else
        {
            auto scope = dml::Graph(ctx->GetDmlDevice());
            auto result = dml::InputTensor(scope, 0, inputs[0]);

            for (uint32_t i = 1; i < inputs.size(); ++i)
            {
                result += dml::InputTensor(scope, i, inputs[i]);
            }

            // TFDML #24881131
            if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
            {
                result = dml::ConvertInt32ToInt64(result);
            }

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());
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
    case TF_UINT64:
    case TF_UINT32: return true;
    default: return false;
    }
}

class DmlBinaryAddVariantHelper : public DmlKernel
{
  public:
    using DmlKernel::Compute;

    explicit DmlBinaryAddVariantHelper(
        DmlDevice* dml_device,
        TF_OpKernelContext* ctx,
        TF_DataType dtype,
        uint32_t num_elements)
    {
        uint32_t tensor_sizes[] = {1, 1, 1, num_elements};

        auto tensor_desc =
            DmlTensorDesc::Create(dtype, tensor_sizes, tensor_sizes);

        DmlTensorInfo a_info = {};
        a_info.kernel_index = 0;
        a_info.desc = tensor_desc;

        DmlTensorInfo b_info = {};
        b_info.kernel_index = 1;
        b_info.desc = tensor_desc;

        DmlKernelTensors tensors = {};
        tensors.inputs = {a_info, b_info};
        tensors.outputs = {a_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(dml_device->GetDmlDevice());
        const auto a = dml::InputTensor(scope, 0, inputs[0]);
        const auto b = dml::InputTensor(scope, 1, inputs[1]);
        auto result = a + b;

        // TFDML #24881131
        if (Is64BitSignedIntegerType(dtype))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        auto init_helper = std::make_shared<DummyInitializationHelper>();

        status_ = Initialize(
            ctx,
            std::move(tensors),
            compiled_op.Get(),
            std::move(init_helper),
            dml_device->GetDmlDevice(),
            dml_device->GetDeviceContext(),
            "AddN");
    }

    StatusOr<DmlGpuEvent> Compute(
        DmlDevice* dml_device,
        TF_OpKernelContext* ctx,
        const TF_Tensor* a_tensor,
        const TF_Tensor* b_tensor,
        TF_Tensor* out_tensor) const
    {
        D3D12BufferRegion a_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(a_tensor);
        D3D12BufferRegion b_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(b_tensor);
        D3D12BufferRegion out_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(out_tensor);

        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            a_buffer.GetBufferBinding(),
            b_buffer.GetBufferBinding(),
        };

        // Bind the first input as the output, to take advantage of in-place
        // execution
        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            out_buffer.GetBufferBinding(),
        };

        return DmlKernel::Compute(
            ctx,
            dml_device->GetDmlDevice(),
            dml_device->GetDeviceContext(),
            input_bindings,
            output_bindings);
    }

    const Status& GetStatus() const { return status_; }

  private:
    Status status_;
};

static void BinaryAddVariant(
    TF_OpKernelContext* ctx,
    const TF_Tensor* a_tensor,
    const TF_Tensor* b_tensor,
    TF_Tensor* out_tensor)
{
    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    DmlDevice* dml_device = static_cast<DmlDevice*>(device);

    TF_DataType dtype = TF_TensorType(a_tensor);
    int64_t num_elements = TF_TensorElementCount(a_tensor);

    if (!IsSupportedAddType(dtype))
    {
        TF_OpKernelContext_Failure(
            ctx,
            errors::InvalidArgument(
                DataTypeString(dtype),
                " is not a supported type for Add.")
                .raw());
        return;
    }

    if (num_elements == 0)
    {
        return;
    }

    DmlBinaryAddVariantHelper dml_kernel(dml_device, ctx, dtype, num_elements);

    if (!dml_kernel.GetStatus().ok())
    {
        TF_OpKernelContext_Failure(ctx, dml_kernel.GetStatus().raw());
        return;
    }

    StatusOr<DmlGpuEvent> status_or_event =
        dml_kernel.Compute(dml_device, ctx, a_tensor, b_tensor, out_tensor);

    if (!status_or_event.ok())
    {
        TF_OpKernelContext_Failure(ctx, status_or_event.status().raw());
    }
}

class DmlAddNVariantKernel : public OpKernel
{
  public:
    explicit DmlAddNVariantKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        OP_REQUIRES(
            ctx,
            ctx->ZerosLikeVariantSupported(),
            errors::InvalidArgument("AddN with the variant data type is not "
                                    "yet supported for pluggable devices in "
                                    "this version of TensorFlow."));

        OP_REQUIRES_OK(ctx, ctx->AddNVariant(BinaryAddVariant));
    }
};

void RegisterKernels_AddN()
{
    using K = KernelDefinition<
        ops::AddN,
        DmlKernelWrapper<DmlAddNKernel, GetOutputShapeAsInputShapeHelper>>;

    constexpr auto T = ops::AddN::Attribute::T;
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_UINT32>::Register();

    using VariantK = KernelDefinition<ops::AddN, DmlAddNVariantKernel>;
    VariantK::WithTypeConstraint<T, TF_VARIANT>::Register();
}

} // namespace tfdml