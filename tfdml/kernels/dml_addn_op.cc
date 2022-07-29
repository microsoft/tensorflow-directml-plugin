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

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());
        }
    }
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

class BinaryAddVariantInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    BinaryAddVariantInitHelper(
        DmlDevice* dml_device,
        TF_DataType dtype,
        uint32_t num_elements)
        : dml_device(dml_device),
          dtype(dtype),
          num_elements(num_elements)
    {
    }

    DmlDevice* dml_device;
    TF_DataType dtype;
    uint32_t num_elements;
};

class DmlBinaryAddVariantKernel : public DmlKernel
{
  public:
    using InitHelper = BinaryAddVariantInitHelper;

    explicit DmlBinaryAddVariantKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        uint32_t tensor_sizes[] = {1, 1, 1, init_helper->num_elements};

        auto tensor_desc = DmlTensorDesc::Create(
            init_helper->dtype,
            tensor_sizes,
            tensor_sizes);

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
        auto scope = dml::Graph(init_helper->dml_device->GetDmlDevice());
        const auto a = dml::InputTensor(scope, 0, inputs[0]);
        const auto b = dml::InputTensor(scope, 1, inputs[1]);
        auto result = a + b;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(
        OpKernelContext* ctx,
        const Tensor& a_tensor,
        const Tensor& b_tensor,
        Tensor& out_tensor) const
    {
        auto* dml_device = static_cast<DmlDevice*>(ctx->device());

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
            ctx->raw(),
            dml_device->GetDmlDevice(),
            dml_device->GetDeviceContext(),
            input_bindings,
            output_bindings);
    }
};

class DmlBinaryAddVariantKernelWrapper : public OpKernel
{
  public:
    explicit DmlBinaryAddVariantKernelWrapper(
        std::shared_ptr<NodeDef> node_def,
        DmlDevice* dml_device,
        const Tensor* a_tensor,
        const Tensor* b_tensor,
        Tensor* out_tensor)
        : OpKernel(node_def),
          node_def_(node_def),
          dml_device_(dml_device),
          a_tensor_(a_tensor),
          b_tensor_(b_tensor),
          out_tensor_(out_tensor)
    {
    }

    std::shared_ptr<DmlKernel> TryGetCachedKernel(
        const DmlKernelManager& kernel_manager,
        const DmlKernelKey& key) const
    {
        // Retrieve the kernel from the cache
        return kernel_manager.TryGetCachedKernel<DmlBinaryAddVariantKernel>(
            key);
    }

    void Compute(OpKernelContext* ctx)
    {
        DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());

        const DmlKernelManager& kernel_manager =
            *dml_device->GetKernelManager();
        std::shared_ptr<DmlKernel> kernel;
        DmlKernelKey key;

        // Construct a kernel key which uniquely identifies the kernel
        // instance we need
        key = CreateKernelKey(ctx);

        // Retrieve an appropriate DmlKernel from the cache. If the kernel
        // hasn't been cached yet, it will be null
        kernel = TryGetCachedKernel(kernel_manager, key);

        if (!kernel)
        {
            auto shared_helper = std::make_shared<BinaryAddVariantInitHelper>(
                dml_device_,
                a_tensor_->dtype(),
                static_cast<uint32_t>(a_tensor_->NumElements()));

            DmlKernelConstruction dml_construction(
                dml_device,
                ctx,
                {},
                shared_helper);

            kernel =
                kernel_manager.CreateCachedKernel<DmlBinaryAddVariantKernel>(
                    &dml_construction,
                    key,
                    shared_helper.get());

            // Check for validation done during kernel construction
            if (!ctx->status().ok())
            {
                return;
            }
        }

        assert(kernel != nullptr);

        // Check for errors triggered during the kernel context's constructor
        // (e.g. OOM when allocating the output buffers)
        if (!ctx->status().ok())
        {
            return;
        }

        auto status_or_event =
            static_cast<DmlBinaryAddVariantKernel*>(kernel.get())
                ->Compute(ctx, *a_tensor_, *b_tensor_, *out_tensor_);
        OP_REQUIRES_OK(ctx, status_or_event.status());

        // Keep this kernel alive at least until it's completed execution on the
        // GPU
        kernel_manager.QueueReference(
            kernel,
            status_or_event.ConsumeValueOrDie());
    }

    DmlKernelKey CreateKernelKey(OpKernelContext* ctx) const
    {
        DmlKernelKey key = {};
        key.op_type_name = "AddV2";
        key.node_def = node_def_;

        DmlInputTensorKey tensor_key = {};
        tensor_key.is_constant_cpu_input = false;
        tensor_key.tensor = TensorShapeAndType{
            TensorShape({a_tensor_->NumElements()}),
            a_tensor_->dtype()};

        key.input_tensors.push_back(tensor_key);
        key.input_tensors.push_back(std::move(tensor_key));

        return key;
    }

  private:
    std::shared_ptr<NodeDef> node_def_;
    DmlDevice* dml_device_;
    const Tensor* a_tensor_;
    const Tensor* b_tensor_;
    Tensor* out_tensor_;
};

static void BinaryAddVariant(
    TF_OpKernelContext* ctx,
    TF_Tensor* raw_a_tensor,
    TF_Tensor* raw_b_tensor,
    TF_Tensor* raw_out_tensor)
{
    Tensor a_tensor(raw_a_tensor);
    Tensor b_tensor(raw_b_tensor);
    Tensor out_tensor(raw_out_tensor);

    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    DmlDevice* dml_device = static_cast<DmlDevice*>(device);

    if (!IsSupportedAddType(a_tensor.dtype()))
    {
        TF_OpKernelContext_Failure(
            ctx,
            errors::InvalidArgument(
                DataTypeString(a_tensor.dtype()),
                " is not a supported type for Add.")
                .raw());
        return;
    }

    if (a_tensor.NumElements() == 0)
    {
        return;
    }

    auto node_def = std::make_shared<NodeDef>(
        "AddV2",
        "DmlAddV2Variant",
        absl::InlinedVector<MemoryType, 8>(3, MemoryType::DEVICE_MEMORY),
        absl::InlinedVector<AttributeValue, 4>(a_tensor.dtype()),
        2);

    DmlBinaryAddVariantKernelWrapper dml_kernel(
        std::move(node_def),
        dml_device,
        &a_tensor,
        &b_tensor,
        &out_tensor);

    OpKernelContext cc_ctx(ctx, &dml_kernel);
    dml_kernel.Compute(&cc_ctx);
}

class DmlAddNVariantKernel : public OpKernel
{
  public:
    explicit DmlAddNVariantKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(node_def)
    {
    }

    void Compute(OpKernelContext* ctx)
    {
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