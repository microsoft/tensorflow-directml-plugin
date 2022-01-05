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
        if (!c->GetAttr(
                  "_grappler_relax_allocator_constraints",
                  &relax_constraints_)
                 .ok())
        {
            relax_constraints_ = false;
        }
    }

    void Compute(OpKernelContext* context)
    {
        DmlDevice* dml_device = static_cast<DmlDevice*>(context->device());
        DmlTracing::KernelComputeEventScope event_scope(
            dml_device->GetDeviceOrdinal(),
            context->op_kernel().type_string(),
            context->op_kernel().name());

        OP_REQUIRES(
            context,
            dtype_ == context->input(1).dtype(),
            errors::InvalidArgument(
                "Variable and value dtypes don't match; respectively, ",
                DataTypeString(dtype_),
                " and ",
                DataTypeString(context->input(1).dtype())));
        RefCountPtr<Var> variable;
        const Tensor value = context->input(1);
        // Note: every resource-variable-manipulating op assumes copy-on-write
        // semantics, and creates a copy of the variable's Tensor if its
        // refcount is bigger than 1 when we try to modify it. This means we
        // never need to copy the original tensor for AssignVariableOp; even if
        // there are other live users of it we know none can modify it so this
        // is always safe (even in esoteric cases where the same tensor is used
        // to initialize multiple variables or the tensor is a constant this is
        // safe, as future writes will trigger copies).
        OP_REQUIRES_OK(
            context,
            LookupOrCreateResource<Var>(
                context,
                *HandleFromInput(context, 0),
                &variable,
                [this, &value](Var** ptr)
                {
                    *ptr = new Var(dtype_);
                    *(*ptr)->tensor() = value;
                    (*ptr)->is_initialized = true;
                    return Status::OK();
                }));
        std::unique_lock<std::shared_mutex> ml(*variable->mu());

        // TODO: Fix this when we update to a more recent TF version that has
        // better support for TF_RESOURCE
        // In graph mode, DestroyResource isn't called and resources are
        // expected to get cleaned up on session shutdown. The problem is that
        // the pluggable device API doesn't currently expose a callback for
        // session creation and shutdown, which doesn't give us the opportunity
        // to clean the resources.
        if (variable->tensor()->dtype() != dtype_)
        {
            Status status =
                DeleteResource(context, *HandleFromInput(context, 0));
            OP_REQUIRES_OK(
                context,
                LookupOrCreateResource<Var>(
                    context,
                    *HandleFromInput(context, 0),
                    &variable,
                    [this, &value](Var** ptr)
                    {
                        *ptr = new Var(dtype_);
                        *(*ptr)->tensor() = value;
                        (*ptr)->is_initialized = true;
                        return Status::OK();
                    }));
        }

        *variable->tensor() = value;
        variable->is_initialized = true;
    }

  private:
    TF_DataType dtype_;
    bool relax_constraints_;
};

class DmlDestroyResourceOp : public OpKernel
{
  public:
    explicit DmlDestroyResourceOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(
            ctx,
            ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
    }

    void Compute(OpKernelContext* ctx)
    {
        Status status = DeleteResource(ctx, *HandleFromInput(ctx, 0));
        if (ignore_lookup_error_ && errors::IsNotFound(status))
        {
            return;
        }
        OP_REQUIRES_OK(ctx, status);
    }

  private:
    bool ignore_lookup_error_;
};

Status CopyVariable(int output_idx, OpKernelContext* ctx, const Tensor* t)
{
    Status status;
    if (t->dtype() == TF_VARIANT)
    {
        LogFatal("TF_VARIANT is not supported yet");
    }
    StatusOr<Tensor> status_or_output =
        ctx->allocate_output(output_idx, t->shape());

    if (!status_or_output.ok())
    {
        return status_or_output.status();
    }

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
    device->GetDeviceContext()->CopyTensorInSameDevice(
        device,
        t,
        &status_or_output.ValueOrDie());
    return Status::OK();
}

class DmlReadVariableOp : public OpKernel
{
  public:
    explicit DmlReadVariableOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    }

    void Compute(OpKernelContext* ctx)
    {
        RefCountPtr<Var> variable;
        const Tensor handle_input = ctx->input(0);
        auto handle = HandleFromInput(ctx, 0);
        const auto status = LookupResource(ctx, *handle, &variable);
        OP_REQUIRES(
            ctx,
            status.ok(),
            errors::FailedPrecondition(
                "Could not find variable ",
                handle->name(),
                ". This could mean that the variable has been deleted. Debug "
                "info: container=",
                handle->container(),
                ", status error message=",
                status.error_message()));

        std::shared_lock<std::shared_mutex> ml(*variable->mu());

        // We're acquiring a reference to the underlying buffer while
        // holding a shared lock to guarantee ordering of reads and
        // writes when in copy-on-write mode.
        const Tensor* t = variable->tensor();
        if (!variable->copy_on_read_mode.load())
        {
            OP_REQUIRES(
                ctx,
                dtype_ == t->dtype(),
                errors::InvalidArgument(
                    "Trying to read variable with wrong dtype. Expected ",
                    DataTypeString(dtype_),
                    " got ",
                    DataTypeString(t->dtype())));
            ctx->set_output(0, *t);
        }
        else
        {
            OP_REQUIRES_OK(ctx, CopyVariable(0, ctx, t));
        }
    }

  private:
    TF_DataType dtype_;
};

template <typename Expression>
class DmlUpdateVariableOp : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlUpdateVariableOp(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        uint32_t tensor_sizes[] = {
            1,
            1,
            1,
            static_cast<uint32_t>(ctx->GetInputTensorShape(1).num_elements())};

        auto tensor_desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            tensor_sizes,
            tensor_sizes);

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
        auto scope = dml::Graph(ctx->GetDmlDevice());
        const auto a = dml::InputTensor(scope, 0, inputs[0]);
        const auto b = dml::InputTensor(scope, 1, inputs[1]);
        auto result = Expression()(a, b);

        // TFDML #24881131
        if (Is64BitSignedIntegerType(ctx->GetInputDataType(1)))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        RefCountPtr<Var> variable;
        TF_RETURN_IF_ERROR(
            LookupResource(op_ctx, *HandleFromInput(op_ctx, 0), &variable));

        std::unique_lock<std::shared_mutex> ml(*variable->mu());

        Tensor* var_tensor = variable->tensor();
        const TensorShape& var_shape = variable->tensor()->shape();
        const Tensor& value = ctx->GetInputTensor(1);
        const TensorShape& value_shape = value.shape();

        if (!var_shape.IsSameSize(value_shape))
        {
            return errors::InvalidArgument(
                "Cannot update variable with shape ",
                var_shape.DebugString(),
                " using a Tensor with shape ",
                value_shape.DebugString(),
                ", shapes must be equal.");
        }

        TF_RETURN_IF_ERROR(PrepareToUpdateVariable(
            op_ctx,
            var_tensor,
            variable->copy_on_read_mode.load()));

        D3D12BufferRegion var_resource =
            ctx->GetDmlDeviceContext()->GetBufferForTensor(*var_tensor);
        D3D12BufferRegion value_resource =
            ctx->GetDmlDeviceContext()->GetBufferForTensor(value);

        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            var_resource.GetBufferBinding(),
            value_resource.GetBufferBinding(),
        };

        // Bind the first input as the output, to take advantage of in-place
        // execution
        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            input_bindings[0],
        };

        return DmlKernel::Compute(ctx, input_bindings, output_bindings);
    }
};

// TODO: Remove when we update to a more recent TF version
template <typename T>
class DmlVariableShapeOp : public OpKernel
{
  public:
    explicit DmlVariableShapeOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        RefCountPtr<Var> variable;
        OP_REQUIRES_OK(
            ctx,
            LookupResource(ctx, *HandleFromInput(ctx, 0), &variable));
        variable->mu()->lock_shared();
        TensorShape shape = variable->tensor()->shape();
        variable->mu()->unlock_shared();
        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, {shape.dims()});
        OP_REQUIRES_OK(ctx, status_or_output.status());

        Tensor& output = status_or_output.ValueOrDie();
        for (int i = 0; i < shape.dims(); ++i)
        {
            output.base<T>()[i] = shape.dim_size(i);
        }
    }
};

class DmlVarIsInitializedOp : public OpKernel
{
  public:
    explicit DmlVarIsInitializedOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, TensorShape({}));
        OP_REQUIRES_OK(ctx, status_or_output.status());
        Tensor& output = status_or_output.ValueOrDie();
        auto output_tensor = output.base<bool>();
        RefCountPtr<Var> variable;
        Status s = LookupResource(ctx, *HandleFromInput(ctx, 0), &variable);
        if (!s.ok())
        {
            output_tensor[0] = false;
            return;
        }
        std::unique_lock<std::shared_mutex> ml(*variable->mu());
        output_tensor[0] = variable->is_initialized;
    }
};

void RegisterAssignVariableOp()
{
    using K = KernelDefinition<ops::AssignVariableOp, DmlAssignVariableOp>::
        WithHostMemoryArgument<ops::AssignVariableOp::Argument::resource>;

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
}

void RegisterReadVariableOp()
{
    KernelDefinition<ops::ReadVariableOp, DmlReadVariableOp>::
        WithHostMemoryArgument<
            ops::ReadVariableOp::Argument::resource>::Register();
}

void RegisterDestroyResourceOp()
{
    KernelDefinition<ops::DestroyResourceOp, DmlDestroyResourceOp>::
        WithHostMemoryArgument<
            ops::DestroyResourceOp::Argument::resource>::Register();
}

void RegisterAssignAddVariableOp()
{
    using K = KernelDefinition<
        ops::AssignAddVariableOp,
        DmlKernelWrapper<
            DmlUpdateVariableOp<std::plus<dml::Expression>>,
            NoOutputShapeHelper>>::
        WithHostMemoryArgument<ops::AssignAddVariableOp::Argument::resource>;

    constexpr auto T = ops::AssignAddVariableOp::Attribute::dtype;
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
}

void RegisterAssignSubVariableOp()
{
    using K = KernelDefinition<
        ops::AssignSubVariableOp,
        DmlKernelWrapper<
            DmlUpdateVariableOp<std::minus<dml::Expression>>,
            NoOutputShapeHelper>>::
        WithHostMemoryArgument<ops::AssignSubVariableOp::Argument::resource>;

    constexpr auto T = ops::AssignSubVariableOp::Attribute::dtype;
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
}

void RegisterVariableShapeOp()
{
    KernelDefinition<ops::VariableShape, DmlVariableShapeOp<int32_t>>::
        WithTypeConstraint<ops::VariableShape::Attribute::out_type, TF_INT32>::
            WithHostMemoryArgument<ops::VariableShape::Argument::input>::
                WithHostMemoryArgument<
                    ops::VariableShape::Argument::output>::Register();

    KernelDefinition<ops::VariableShape, DmlVariableShapeOp<int64_t>>::
        WithTypeConstraint<ops::VariableShape::Attribute::out_type, TF_INT64>::
            WithHostMemoryArgument<ops::VariableShape::Argument::input>::
                WithHostMemoryArgument<
                    ops::VariableShape::Argument::output>::Register();
}

void RegisterVarIsInitializedOp()
{
    KernelDefinition<ops::VarIsInitializedOp, DmlVarIsInitializedOp>::
        WithHostMemoryArgument<ops::VarIsInitializedOp::Argument::resource>::
            WithHostMemoryArgument<
                ops::VarIsInitializedOp::Argument::is_initialized>::Register();
}

void RegisterKernels_VariableOps()
{
    RegisterAssignVariableOp();
    RegisterAssignAddVariableOp();
    RegisterAssignSubVariableOp();
    RegisterReadVariableOp();
    RegisterDestroyResourceOp();
    RegisterVariableShapeOp();
    RegisterVarIsInitializedOp();
}

} // namespace tfdml
