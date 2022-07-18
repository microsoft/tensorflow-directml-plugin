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

#include "tfdml/core/dml_tracing.h"
#include "tfdml/runtime_adapter/stream.h"
#include "tfdml/runtime_adapter/training_op_helpers.h"

namespace tfdml
{
class DmlAssignVariableOp : public OpKernel
{
  public:
    explicit DmlAssignVariableOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
        if (c->HasAttr("validate_shape"))
        {
            OP_REQUIRES_OK(c, c->GetAttr("validate_shape", &validate_shape_));
        }
    }

    void Compute(OpKernelContext* ctx)
    {
        constexpr int var_index = 0;
        constexpr int value_index = 1;

        OP_REQUIRES(
            ctx,
            dtype_ == ctx->input(value_index).dtype(),
            errors::InvalidArgument(
                "Variable and value dtypes don't match; respectively, ",
                DataTypeString(dtype_),
                " and ",
                DataTypeString(ctx->input(value_index).dtype())));

        OP_REQUIRES_OK(
            ctx,
            ctx->AssignVariable(var_index, value_index, validate_shape_));
    }

  private:
    TF_DataType dtype_;
    bool validate_shape_ = false;
};

class DmlAssign : public OpKernel
{
  public:
    explicit DmlAssign(
        OpKernelConstruction* context,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(
            context,
            context->GetAttr("use_locking", &exclusive_lock_));
        OP_REQUIRES_OK(
            context,
            context->GetAttr("validate_shape", &validate_shape_));
    }

    void Compute(OpKernelContext* ctx)
    {
        constexpr int input_ref_index = 0;
        constexpr int output_ref_index = 0;
        constexpr int value_index = 1;

        OP_REQUIRES_OK(
            ctx,
            ctx->AssignRefVariable(
                input_ref_index,
                output_ref_index,
                value_index,
                exclusive_lock_,
                validate_shape_));
    }

  private:
    bool exclusive_lock_;
    bool validate_shape_;
};

template <typename Expression>
class DmlAssignUpdate : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlAssignUpdate(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // The Assign operators don't support broadcasting, so it's safe to
        // collapse all dimensions into a single one before sending it to DML.
        // This allows us to support tensors with more than 4 or 5 dimensions.
        TensorShape tensor_shape = {
            ctx->GetOutputTensorShape(1).num_elements()};

        TF_DataType data_type = ctx->GetInputDataType(1);

        DmlKernelTensors tensors;

        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc =
                DmlTensorDesc::Create(data_type, tensor_shape, tensor_shape);

            tensors.inputs.push_back(std::move(input));
        }

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc =
            DmlTensorDesc::Create(data_type, tensor_shape, tensor_shape);

        tensors.outputs = {output};

        // The input ref and the output ref must refer to the same memory
        tensors.output_refs_forwarding = {0};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto a_tensor = dml::InputTensor(scope, 0, inputs[0]);
        auto b_tensor = dml::InputTensor(scope, 1, inputs[1]);
        auto result = Expression()(a_tensor, b_tensor);

        // TFDML #24881131
        if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DummyInitializationHelper : public InitializationHelper
{
};

template <typename Expression>
const char* KernelName();

template <>
const char* KernelName<std::plus<dml::Expression>>()
{
    return "AssignAddVariableOp";
}

template <>
const char* KernelName<std::minus<dml::Expression>>()
{
    return "AssignSubVariableOp";
}

template <typename Expression>
class DmlUpdateVariableOpHelper : public DmlKernel
{
  public:
    using DmlKernel::Compute;

    explicit DmlUpdateVariableOpHelper(
        TF_OpKernelContext* ctx,
        DmlDevice* dml_device,
        TF_DataType dtype,
        int num_elements)
    {
        uint32_t tensor_sizes[] =
            {1, 1, 1, static_cast<uint32_t>(num_elements)};

        auto tensor_desc =
            DmlTensorDesc::Create(dtype, tensor_sizes, tensor_sizes);

        DmlTensorInfo lhs_info = {};
        lhs_info.kernel_index = 0;
        lhs_info.desc = tensor_desc;

        DmlTensorInfo rhs_info = {};
        rhs_info.kernel_index = 1;
        rhs_info.desc = tensor_desc;

        DmlKernelTensors tensors = {};
        tensors.inputs = {lhs_info, rhs_info};
        tensors.outputs = {lhs_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(dml_device->GetDmlDevice());
        const auto a = dml::InputTensor(scope, 0, inputs[0]);
        const auto b = dml::InputTensor(scope, 1, inputs[1]);
        auto result = Expression()(a, b);

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
            KernelName<Expression>());
    }

    StatusOr<DmlGpuEvent> Compute(
        TF_OpKernelContext* ctx,
        DmlDevice* dml_device,
        const TF_Tensor* var_tensor,
        const TF_Tensor* value_tensor) const
    {
        D3D12BufferRegion var_resource =
            dml_device->GetDeviceContext()->GetBufferForTensor(var_tensor);
        D3D12BufferRegion value_resource =
            dml_device->GetDeviceContext()->GetBufferForTensor(value_tensor);

        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            var_resource.GetBufferBinding(),
            value_resource.GetBufferBinding(),
        };

        // Bind the first input as the output, to take advantage of in-place
        // execution
        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            input_bindings[0],
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

template <typename Expression>
static void UpdateVariable(
    TF_OpKernelContext* ctx,
    TF_Tensor* var,
    TF_Tensor* value,
    int Op)
{
    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    DmlDevice* dml_device = static_cast<DmlDevice*>(device);

    DmlUpdateVariableOpHelper<Expression> dml_kernel(
        ctx,
        dml_device,
        TF_TensorType(value),
        TF_TensorElementCount(value));

    if (!dml_kernel.GetStatus().ok())
    {
        TF_OpKernelContext_Failure(ctx, dml_kernel.GetStatus().raw());
        return;
    }

    StatusOr<DmlGpuEvent> status_or_event =
        dml_kernel.Compute(ctx, dml_device, var, value);

    if (!status_or_event.ok())
    {
        TF_OpKernelContext_Failure(ctx, status_or_event.status().raw());
    }
}

template <typename Expression>
class DmlUpdateVariableOp : public OpKernel
{
  public:
    explicit DmlUpdateVariableOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    }

    void Compute(OpKernelContext* ctx)
    {
        constexpr int var_index = 0;
        constexpr int value_index = 1;
        OP_REQUIRES_OK(
            ctx,
            ctx->AssignUpdateVariable(
                var_index,
                value_index,
                UpdateVariable<Expression>));
    }

  private:
    TF_DataType dtype_;
};

void RegisterAssignVariableOp()
{
    using K = KernelDefinition<ops::AssignVariableOp, DmlAssignVariableOp>::
        WithHostMemoryArguments<ops::AssignVariableOp::Argument::resource>;

    // We deliberately register the same types here that CUDA does.
    constexpr auto T = ops::AssignVariableOp::Attribute::dtype;
    K::WithTypeConstraint<T, TF_BOOL>::Register();
    K::WithTypeConstraint<T, TF_COMPLEX64>::Register();
    K::WithTypeConstraint<T, TF_COMPLEX128>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_DOUBLE>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_UINT32>::Register();
    K::WithTypeConstraint<T, TF_BFLOAT16>::Register();
}

void RegisterAssignAddVariableOp()
{
    using K = KernelDefinition<
        ops::AssignAddVariableOp,
        DmlUpdateVariableOp<std::plus<dml::Expression>>>::
        WithHostMemoryArguments<ops::AssignAddVariableOp::Argument::resource>;

    constexpr auto T = ops::AssignAddVariableOp::Attribute::dtype;
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
}

void RegisterAssignSubVariableOp()
{
    using K = KernelDefinition<
        ops::AssignSubVariableOp,
        DmlUpdateVariableOp<std::minus<dml::Expression>>>::
        WithHostMemoryArguments<ops::AssignSubVariableOp::Argument::resource>;

    constexpr auto T = ops::AssignSubVariableOp::Attribute::dtype;
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
}

void RegisterAssign()
{
    using K = KernelDefinition<ops::Assign, DmlAssign>;

    // We deliberately register the same types here that CUDA does.
    constexpr auto T = ops::Assign::Attribute::T;
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_UINT32>::Register();
    K::WithTypeConstraint<T, TF_UINT8>::Register();
    K::WithTypeConstraint<T, TF_BOOL>::Register();
    K::WithTypeConstraint<T, TF_COMPLEX64>::Register();
    K::WithTypeConstraint<T, TF_COMPLEX128>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_DOUBLE>::Register();
}

void RegisterAssignAdd()
{
    using K = KernelDefinition<
        ops::AssignAdd,
        DmlKernelWrapper<
            DmlAssignUpdate<std::plus<dml::Expression>>,
            GetOutputShapeAsInputShapeHelper>>;

    // We deliberately register the same types here that CUDA does.
    constexpr auto T = ops::AssignAdd::Attribute::T;
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
}

void RegisterAssignSub()
{
    using K = KernelDefinition<
        ops::AssignSub,
        DmlKernelWrapper<
            DmlAssignUpdate<std::minus<dml::Expression>>,
            GetOutputShapeAsInputShapeHelper>>;

    // We deliberately register the same types here that CUDA does.
    constexpr auto T = ops::AssignSub::Attribute::T;
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
}

void RegisterKernels_AssignVariableOps()
{
    RegisterAssignVariableOp();
    RegisterAssignAddVariableOp();
    RegisterAssignSubVariableOp();
    RegisterAssignAdd();
    RegisterAssignSub();
    RegisterAssign();
}

} // namespace tfdml