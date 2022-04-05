/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (ctx) Microsoft Corporation.

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
#include "tfdml/kernels/pch.h"
#include "tfdml/runtime_adapter/variable_lock.h"

namespace tfdml
{

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidRefShapes(
    const Tensor& params,
    const Tensor& updates,
    const Tensor& indices)
{
    if (updates.dims() == 0) return true;
    if (updates.dims() != indices.dims() + params.dims() - 1) return false;
    for (int d = 0; d < indices.dims(); d++)
    {
        if (updates.dim_size(d) != indices.dim_size(d))
        {
            return false;
        }
    }
    for (int d = 1; d < params.dims(); d++)
    {
        if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims()))
        {
            return false;
        }
    }
    return true;
}

static Status ValidateResourceScatter(
    const Tensor& indices,
    const Tensor& updates)
{
    int64_t num_updates = updates.NumElements();
    int64_t num_indices = indices.NumElements();
    if (num_indices > 0 && !TensorShapeUtils::IsScalar(updates.shape()) &&
        num_updates % num_indices != 0)
    {
        return errors::InvalidArgument(
            "shape of indices (",
            indices.shape().DebugString(),
            ") is not compatible with the shape of updates (",
            updates.shape().DebugString(),
            ")");
    }

    return Status::OK();
}

template <typename Index>
static Status ValidateCommonScatter(const Tensor& params, const Tensor& indices)
{
    // Check that we have enough index space
    const int64_t N_big = indices.NumElements();

    if (N_big > std::numeric_limits<Index>::max())
    {
        return errors::InvalidArgument(
            "indices has too many elements for ",
            DataTypeString(DataTypeToEnum<Index>()),
            " indexing: ",
            N_big,
            " > ",
            std::numeric_limits<Index>::max());
    }

    const Index N = static_cast<Index>(indices.NumElements());

    if (params.dim_size(0) > std::numeric_limits<Index>::max())
    {
        return errors::InvalidArgument(
            "params.shape[0] too large for ",
            DataTypeString(DataTypeToEnum<Index>()),
            " indexing: ",
            params.dim_size(0),
            " > ",
            std::numeric_limits<Index>::max());
    }

    return Status::OK();
}

template <typename Index>
class ScatterUpdateInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    ScatterUpdateInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : var_lock_(ctx)
    {
        if (ctx->input(0).dtype() == TF_RESOURCE)
        {
            constexpr bool exclusive_lock = false;
            constexpr bool is_variant = false;
            variable_tensor_.emplace();
            OP_REQUIRES_OK(
                ctx,
                ctx->GetInputTensorFromVariable(
                    0,
                    exclusive_lock,
                    is_variant,
                    &*variable_tensor_));

            constexpr int lock_indices[1] = {0};
            var_lock_.LockShared(lock_indices);
        }

        const Tensor params = GetParamsTensor(ctx);
        const Tensor& indices = ctx->input(1);
        const Tensor& updates = ctx->input(2);

        OP_REQUIRES_OK(ctx, ValidateCommonScatter<Index>(params, indices));

        if (ctx->input(0).dtype() == TF_RESOURCE)
        {
            OP_REQUIRES_OK(ctx, ValidateResourceScatter(indices, updates));
        }
    }

    Tensor GetParamsTensor(OpKernelContext* ctx) const
    {
        return variable_tensor_ ? *variable_tensor_ : ctx->input(0);
    }

    void Unlock() const
    {
        if (variable_tensor_)
        {
            var_lock_.Unlock();
        }
    }

  private:
    absl::optional<Tensor> variable_tensor_;
    mutable VariableLock var_lock_;
};

struct ScatterUpdateOperation
{
    static constexpr bool inplace_allowed = true;

    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression params,
        dml::Expression indices,
        dml::Expression updates,
        uint32_t scatter_axis,
        bool int64_indices,
        bool scalar_updates)
    {
        return dml::ScatterElements(params, indices, updates, scatter_axis);
    }
};

struct BinaryMinOperation
{
    dml::Expression operator()(dml::Expression a, dml::Expression b)
    {
        return dml::Min(a, b);
    }
};

struct BinaryMaxOperation
{
    dml::Expression operator()(dml::Expression a, dml::Expression b)
    {
        return dml::Max(a, b);
    }
};

template <typename BinaryOperation, typename TParams>
static constexpr TParams BinaryOperationIdentityValue()
{
    if (std::is_same<BinaryOperation, std::multiplies<dml::Expression>>::value)
    {
        return static_cast<TParams>(1);
    }

    if (std::is_same<BinaryOperation, std::divides<dml::Expression>>::value)
    {
        return static_cast<TParams>(1);
    }

    if (std::is_same<BinaryOperation, std::plus<dml::Expression>>::value)
    {
        return static_cast<TParams>(0);
    }

    if (std::is_same<BinaryOperation, std::minus<dml::Expression>>::value)
    {
        return static_cast<TParams>(0);
    }

    if (std::is_same<BinaryOperation, BinaryMinOperation>::value)
    {
        return std::numeric_limits<TParams>::max();
    }

    if (std::is_same<BinaryOperation, BinaryMaxOperation>::value)
    {
        return std::numeric_limits<TParams>::lowest();
    }
}

// For arithmetic Scatter operations, TensorFlow supports duplicate indices so
// we can't use DirectML's Scatter. For now, we can use the graph as a
// workaround but we should revisit it in the future and add a DirectML API if
// we get signals that this implementation is a bottleneck.
template <
    typename BinaryOperation,
    DML_REDUCE_FUNCTION reduce_function,
    typename TParams>
struct ScatterBinaryOperation
{
    static constexpr bool inplace_allowed = false;

    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression params,
        dml::Expression indices,
        dml::Expression updates,
        uint32_t scatter_axis,
        bool int64_indices,
        bool scalar_updates)
    {
        auto params_sizes = params.GetOutputDesc().sizes;
        uint32_t row_count = params_sizes[scatter_axis];

        dml::TensorDesc::Dimensions row_indices_sizes({1, 1, row_count, 1});

        auto row_indices = dml::FillValueSequence(
            scope,
            row_indices_sizes,
            indices.GetOutputDesc().dataType,
            dml::ScalarUnion(0, indices.GetOutputDesc().dataType),
            dml::ScalarUnion(1, indices.GetOutputDesc().dataType));

        auto indices_sizes = indices.GetOutputDesc().sizes;
        dml::TensorDesc::Dimensions broadcasted_sizes({
            1,
            indices_sizes[2],
            row_count,
            params_sizes[3],
        });

        auto broadcasted_row_indices = dml::Reinterpret(
            row_indices,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 0, 1, 0}));

        uint32_t indices_stride_multiplier = int64_indices ? 2 : 1;
        auto broadcasted_indices = dml::Reinterpret(
            indices,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, indices_stride_multiplier, 0, 0}));

        dml::Expression broadcasted_updates =
            scalar_updates
                ? dml::Reinterpret(
                      updates,
                      broadcasted_sizes,
                      dml::TensorDesc::Dimensions({0, 0, 0, 0}))
                : dml::Reinterpret(
                      updates,
                      broadcasted_sizes,
                      dml::TensorDesc::Dimensions({0, indices_sizes[3], 0, 1}));

        const TParams identity_value =
            BinaryOperationIdentityValue<BinaryOperation, TParams>();

        auto identity = dml::ScalarTensor<TParams>(
            scope,
            identity_value,
            broadcasted_sizes);

        auto sparse_updates = dml::If(
            broadcasted_indices == broadcasted_row_indices,
            broadcasted_updates,
            identity);

        auto reduced_updates =
            dml::Reduce(sparse_updates, reduce_function, {1});
        auto result = BinaryOperation()(params, reduced_updates);

        return result;
    }
};

template <typename Index, typename ScatterOp>
class DmlScatterUpdateKernel : public DmlKernel
{
  public:
    using InitHelper = ScatterUpdateInitializationHelper<Index>;

    explicit DmlScatterUpdateKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const Tensor params_tensor =
            init_helper->GetParamsTensor(ctx->GetOpKernelContext());

        const TensorShape& params_shape = params_tensor.shape();
        const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
        const TensorShape& updates_shape = ctx->GetInputTensorShape(2);
        bool scalar_updates = TensorShapeUtils::IsScalar(updates_shape);

        const TensorShape flat_params_shape({
            params_shape.dim_size(0),
            params_shape.num_elements() / params_shape.dim_size(0),
        });

        const TensorShape flat_indices_shape({
            indices_shape.num_elements(),
            params_shape.num_elements() / params_shape.dim_size(0),
        });

        const TensorShape non_broadcast_flat_indices_shape({
            indices_shape.num_elements(),
            1,
        });

        const TensorShape flat_updates_shape({
            indices_shape.num_elements(),
            params_shape.num_elements() / params_shape.dim_size(0),
        });

        const TensorShape& non_broadcast_flat_updates_shape =
            scalar_updates ? updates_shape : flat_updates_shape;

        DmlTensorInfo input_tensor_info;
        input_tensor_info.kernel_index = 0;
        input_tensor_info.desc = DmlTensorDesc::Create(
            params_tensor.dtype(),
            flat_params_shape,
            flat_params_shape);

        DmlTensorInfo indices_tensor_info;
        indices_tensor_info.kernel_index = 1;
        indices_tensor_info.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            flat_indices_shape,
            non_broadcast_flat_indices_shape);

        DmlTensorInfo updates_tensor_info;
        updates_tensor_info.kernel_index = 2;
        updates_tensor_info.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            flat_updates_shape,
            non_broadcast_flat_updates_shape);

        DmlTensorInfo output_tensor_info;
        output_tensor_info.kernel_index = 0;
        output_tensor_info.desc = DmlTensorDesc::Create(
            params_tensor.dtype(),
            params_shape,
            params_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {
            input_tensor_info,
            indices_tensor_info,
            updates_tensor_info,
        };

        tensors.outputs = {output_tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto params = dml::InputTensor(scope, 0, inputs[0]);
        auto indices = dml::InputTensor(scope, 1, inputs[1]);
        auto updates = dml::InputTensor(scope, 2, inputs[2]);

        const uint32_t scatter_axis =
            params.GetOutputDesc().sizes.size() - flat_params_shape.dims();

        // TODO: Remove the Is64BitIntegerType hack when DML has a more solid
        // solution for 64 bit datatypes
        // TFDML #24881131
        auto result = ScatterOp()(
            scope,
            params,
            indices,
            updates,
            scatter_axis,
            Is64BitIntegerType(ctx->GetInputDataType(1)),
            scalar_updates);

        // TODO: Remove the Is64BitSignedIntegerType hack when DML has a more
        // solid solution for 64 bit datatypes
        // TFDML #24881131
        if (Is64BitSignedIntegerType(params_tensor.dtype()))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto init_helper = ctx->GetInitializationHelper<InitHelper>();
        absl::Cleanup lock_cleanup = [init_helper] { init_helper->Unlock(); };

        const Tensor params_tensor =
            init_helper->GetParamsTensor(ctx->GetOpKernelContext());

        // Create input buffers
        D3D12BufferRegion input_buffers[] = {
            ctx->GetDmlDeviceContext()->GetBufferForTensor(params_tensor),
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                ctx->GetInputTensor(1)),
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                ctx->GetInputTensor(2)),
        };

        // Create input bindings
        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            input_buffers[0].GetBufferBinding(),
            input_buffers[1].GetBufferBinding(),
            input_buffers[2].GetBufferBinding(),
        };

        DmlGpuEvent gpu_event;

        if (ScatterOp::inplace_allowed)
        {
            absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
                input_bindings[0],
            };

            auto status_or_event =
                DmlKernel::Compute(ctx, input_bindings, output_bindings);
            if (!status_or_event.ok())
            {
                return status_or_event;
            }

            gpu_event = status_or_event.ValueOrDie();
        }
        else
        {
            DmlBuffer output_buffer =
                ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
                    ctx->GetOpKernelContext()->raw(),
                    input_buffers[0].SizeInBytes());

            absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
                output_buffer.GetBufferBinding(),
            };

            auto status_or_event =
                DmlKernel::Compute(ctx, input_bindings, output_bindings);
            if (!status_or_event.ok())
            {
                return status_or_event;
            }

            ctx->GetDmlDeviceContext()->CopyBufferToBuffer(
                input_buffers[0],
                output_buffer.Region());

            gpu_event = ctx->GetDmlDeviceContext()->InsertUavBarrier();
        }

        return gpu_event;
    }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, name, op, index_type)              \
    REGISTER_KERNEL_BUILDER(                                                   \
        Name(name)                                                             \
            .Device(DEVICE_DML)                                                \
            .TypeConstraint<type>("T")                                         \
            .TypeConstraint<index_type>("Tindices"),                           \
        DmlKernelWrapper<                                                      \
            DmlScatterUpdateKernel<index_type, op>,                            \
            GetOutputShapeAsInputShapeHelper>)

#define REGISTER_SCATTER_KERNEL(type, name, op)                                \
    REGISTER_SCATTER_KERNEL_INDEX(type, name, op, int32_t);                    \
    REGISTER_SCATTER_KERNEL_INDEX(type, name, op, int64_t);

#define REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, index_type)     \
    REGISTER_KERNEL_BUILDER(                                                   \
        Name(name)                                                             \
            .Device(DEVICE_DML)                                                \
            .HostMemory("resource")                                            \
            .TypeConstraint<type>("dtype")                                     \
            .TypeConstraint<index_type>("Tindices"),                           \
        DmlKernelWrapper<                                                      \
            DmlScatterUpdateKernel<index_type, op>,                            \
            NoOutputShapeHelper,                                               \
            DmlKernelCachePolicy::Never>)

#define REGISTER_RESOURCE_SCATTER_KERNEL(type, name, op)                       \
    REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, int32_t);           \
    REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, int64_t);

template <typename type>
using ScatterPlusOp = ScatterBinaryOperation<
    std::plus<dml::Expression>,
    DML_REDUCE_FUNCTION_SUM,
    type>;

template <typename type>
using ScatterMinusOp = ScatterBinaryOperation<
    std::minus<dml::Expression>,
    DML_REDUCE_FUNCTION_SUM,
    type>;

template <typename type>
using ScatterMulOp = ScatterBinaryOperation<
    std::multiplies<dml::Expression>,
    DML_REDUCE_FUNCTION_MULTIPLY,
    type>;

template <typename type>
using ScatterDivOp = ScatterBinaryOperation<
    std::divides<dml::Expression>,
    DML_REDUCE_FUNCTION_MULTIPLY,
    type>;

template <typename type>
using ScatterMinOp =
    ScatterBinaryOperation<BinaryMinOperation, DML_REDUCE_FUNCTION_MIN, type>;

template <typename type>
using ScatterMaxOp =
    ScatterBinaryOperation<BinaryMaxOperation, DML_REDUCE_FUNCTION_MAX, type>;

template <typename Index, typename DataType>
using ResourceScatterAddKernel = typename KernelDefinition<
    ops::ResourceScatterAdd,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterPlusOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterAdd::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterAdd::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterAdd::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterAdd()
{
    ResourceScatterAddKernel<int32_t, float>::Register();
    ResourceScatterAddKernel<int64_t, float>::Register();
    ResourceScatterAddKernel<int32_t, Eigen::half>::Register();
    ResourceScatterAddKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterSubKernel = typename KernelDefinition<
    ops::ResourceScatterSub,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterMinusOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterSub::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterSub::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterSub::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterSub()
{
    ResourceScatterSubKernel<int32_t, float>::Register();
    ResourceScatterSubKernel<int64_t, float>::Register();
    ResourceScatterSubKernel<int32_t, Eigen::half>::Register();
    ResourceScatterSubKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterMulKernel = typename KernelDefinition<
    ops::ResourceScatterMul,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterMulOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterMul::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterMul::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterMul::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterMul()
{
    ResourceScatterMulKernel<int32_t, float>::Register();
    ResourceScatterMulKernel<int64_t, float>::Register();
    ResourceScatterMulKernel<int32_t, Eigen::half>::Register();
    ResourceScatterMulKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterDivKernel = typename KernelDefinition<
    ops::ResourceScatterDiv,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterDivOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterDiv::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterDiv::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterDiv::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterDiv()
{
    ResourceScatterDivKernel<int32_t, float>::Register();
    ResourceScatterDivKernel<int64_t, float>::Register();
    ResourceScatterDivKernel<int32_t, Eigen::half>::Register();
    ResourceScatterDivKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterMinKernel = typename KernelDefinition<
    ops::ResourceScatterMin,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterMinOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterMin::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterMin::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterMin::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterMin()
{
    ResourceScatterMinKernel<int32_t, float>::Register();
    ResourceScatterMinKernel<int64_t, float>::Register();
    ResourceScatterMinKernel<int32_t, Eigen::half>::Register();
    ResourceScatterMinKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterMaxKernel = typename KernelDefinition<
    ops::ResourceScatterMax,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterMaxOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterMax::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterMax::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterMax::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterMax()
{
    ResourceScatterMaxKernel<int32_t, float>::Register();
    ResourceScatterMaxKernel<int64_t, float>::Register();
    ResourceScatterMaxKernel<int32_t, Eigen::half>::Register();
    ResourceScatterMaxKernel<int64_t, Eigen::half>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterUpdateKernel = typename KernelDefinition<
    ops::ResourceScatterUpdate,
    DmlKernelWrapper<
        DmlScatterUpdateKernel<Index, ScatterUpdateOperation>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterUpdate::Argument::resource>::
        template WithTypeConstraint<
            ops::ResourceScatterUpdate::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterUpdate::Attribute::dtype,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterUpdate()
{
    ResourceScatterUpdateKernel<int32_t, bool>::Register();
    ResourceScatterUpdateKernel<int64_t, int64_t>::Register();
    ResourceScatterUpdateKernel<int32_t, float>::Register();
    ResourceScatterUpdateKernel<int64_t, float>::Register();
    ResourceScatterUpdateKernel<int32_t, Eigen::half>::Register();
    ResourceScatterUpdateKernel<int64_t, Eigen::half>::Register();
}

void RegisterKernels_Scatter()
{
    RegisterResourceScatterAdd();
    RegisterResourceScatterSub();
    RegisterResourceScatterMul();
    RegisterResourceScatterDiv();
    RegisterResourceScatterMin();
    RegisterResourceScatterMax();
    RegisterResourceScatterUpdate();
}

} // namespace tfdml
