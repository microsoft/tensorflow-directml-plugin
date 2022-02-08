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
#include "tfdml/kernels/pch.h"
#include "tfdml/runtime_adapter/variable_lock.h"

namespace tfdml
{

static bool ValidEmptyOutputShape(
    int64_t num_inputs,
    int64_t num_indices,
    int64_t num_updates)
{
    if (num_indices == 0 && num_updates == 0)
    {
        return true; // regardless of num_inputs ?= 0, covers both cases
    }
    // now we want all 3 tensors to have values
    return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

static Status ValidateUpdateShape(
    const TensorShape& params_shape,
    const Tensor& indices,
    const Tensor& updates)
{
    const int64_t slice_dim =
        (indices.dims() > 1) ? indices.dim_size(indices.dims() - 1) : 1;
    const int64_t batch_dim = (indices.dims() > 1) ? indices.dims() - 1 : 1;

    auto shape_err = [&]()
    {
        return errors::InvalidArgument(
            "Must have updates.shape = indices.shape[:batch_dim] + ",
            "params_shape[slice_dim:], got updates.shape: ",
            updates.shape().DebugString(),
            ", indices.shape: ",
            indices.shape().DebugString(),
            ", params_shape: ",
            params_shape.DebugString(),
            ", slice_dim: ",
            slice_dim,
            ", and batch_dim: ",
            batch_dim);
    };

    if (updates.dims() < batch_dim) return shape_err();
    if (params_shape.dims() < slice_dim + (updates.dims() - batch_dim))
    {
        return shape_err();
    }
    if (updates.dims() != batch_dim + params_shape.dims() - slice_dim)
    {
        return shape_err();
    }
    for (int d = 0; d < batch_dim; ++d)
    {
        if (updates.dim_size(d) != indices.dim_size(d)) return shape_err();
    }
    for (int d = 0; d < updates.dims() - batch_dim; ++d)
    {
        if (updates.dim_size(d + batch_dim) !=
            params_shape.dim_size(d + slice_dim))
        {
            return shape_err();
        }
    }
    return Status::OK();
}

template <typename Index>
static Status ValidateCommonScatter(
    const Tensor& params,
    const Tensor& indices,
    const Tensor& updates)
{
    if (!TensorShapeUtils::IsVectorOrHigher(params.shape()))
    {
        return errors::InvalidArgument(
            "Output must be at least 1-D, ",
            "got shape: ",
            params.shape().DebugString());
    }

    if (!ValidEmptyOutputShape(
            params.NumElements(),
            indices.NumElements(),
            updates.NumElements()))
    {
        return errors::InvalidArgument(
            "Indices and updates specified for empty output.  indices shape: ",
            indices.shape().DebugString());
    }

    if (updates.dim_size(0) != indices.dim_size(0))
    {
        return errors::InvalidArgument(
            "The outermost dimension of updates and indices ",
            "must match. Got indices.shape ",
            indices.shape().DebugString(),
            ", updates.shape ",
            updates.shape().DebugString());
    }

    TF_RETURN_IF_ERROR(ValidateUpdateShape(params.shape(), indices, updates));

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
class ResourceScatterNDInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    explicit ResourceScatterNDInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : var_lock_(ctx),
          isTensorInput_(ctx->input(0).dtype() != TF_RESOURCE)
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

        OP_REQUIRES_OK(
            ctx,
            ValidateCommonScatter<Index>(params, indices, updates));
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

    bool IsTensorInput() const { return isTensorInput_; }

    virtual ~ResourceScatterNDInitHelper() { Unlock(); }

  private:
    bool isTensorInput_;
    absl::optional<Tensor> variable_tensor_;
    mutable VariableLock var_lock_;
};

template <typename Index>
class DmlScatterNDUpdateKernel : public DmlKernel
{
  public:
    using InitHelper = ResourceScatterNDInitHelper<Index>;

    DmlScatterNDUpdateKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const Tensor params_tensor =
            init_helper->GetParamsTensor(ctx->GetOpKernelContext());

        const TensorShape& in_out_shape = params_tensor.shape();
        const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
        const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

        DmlTensorInfo in_out_tensor;
        in_out_tensor.desc = DmlTensorDesc::Create(
            params_tensor.dtype(),
            in_out_shape,
            in_out_shape);

        DmlTensorInfo indices_tensor;
        indices_tensor.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            indices_shape,
            indices_shape);

        DmlTensorInfo updates_tensor;
        updates_tensor.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            updates_shape,
            updates_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {in_out_tensor, indices_tensor, updates_tensor};
        tensors.outputs = {in_out_tensor};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto graph = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(graph, 0, inputs[0]);
        auto indices = dml::InputTensor(graph, 1, inputs[1]);
        auto updates = dml::InputTensor(graph, 2, inputs[2]);

        // First, perform the scatter on an empty tensor
        auto result = dml::ScatterND(
            input,
            indices,
            updates,
            in_out_shape.dims(),
            indices_shape.dims());

        // TFDML #24881131
        if (Is64BitSignedIntegerType(params_tensor.dtype()))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            graph.Compile(DML_EXECUTION_FLAG_NONE, {result});

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

        return gpu_event;
    }
};

// For arithmetic ScatterNd operations, TensorFlow supports duplicate indices so
// we can't use DirectML's ScatterNd. For now, we can use the graph as a
// workaround but we should revisit it in the future and add a DirectML API if
// we get signals that this implementation is a bottleneck.
template <
    typename BinaryOperation,
    DML_REDUCE_FUNCTION reduce_function,
    typename TParams>
struct ScatterNdBinaryOperation
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression params,
        dml::Expression indices,
        dml::Expression updates,
        dml::Expression strides,
        bool int64_indices,
        bool int64_data)
    {
        // First, compute the 1D version of the indices as if we were indexing
        // into a 1D array
        const auto broadcasted_strides = dml::Reinterpret(
            strides,
            indices.GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions({0, 0, 0, int64_indices ? 2u : 1u}));

        const auto global_indices = dml::Reduce(
            indices * broadcasted_strides,
            DML_REDUCE_FUNCTION_SUM,
            {3});

        const auto params_sizes = params.GetOutputDesc().sizes;
        const uint32_t row_count = params_sizes[2];
        const dml::TensorDesc::Dimensions row_indices_sizes(
            {1, 1, row_count, 1});

        const auto indices_dtype = global_indices.GetOutputDesc().dataType;

        const auto row_indices = dml::FillValueSequence(
            scope,
            row_indices_sizes,
            indices_dtype,
            dml::ScalarUnion(0, indices_dtype),
            dml::ScalarUnion(1, indices_dtype));

        const auto indices_sizes = indices.GetOutputDesc().sizes;
        const dml::TensorDesc::Dimensions broadcasted_sizes({
            1,
            indices_sizes[2],
            row_count,
            params_sizes[3],
        });

        const auto broadcasted_row_indices = dml::Reinterpret(
            row_indices,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 0, 1, 0}));

        const auto broadcasted_indices = dml::Reinterpret(
            global_indices,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 1, 0, 0}));

        const auto updates_sizes = updates.GetOutputDesc().sizes;
        const auto broadcasted_updates = dml::Reinterpret(
            updates,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, updates_sizes[3], 0, 1}));

        const auto identity =
            dml::ScalarTensor<TParams>(scope, TParams(0), broadcasted_sizes);

        const auto sparse_updates = dml::If(
            broadcasted_indices == broadcasted_row_indices,
            broadcasted_updates,
            identity);

        const auto reduced_updates =
            dml::Reduce(sparse_updates, reduce_function, {1});

        auto result = BinaryOperation()(params, reduced_updates);

        // TFDML #24881131
        if (int64_data)
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        return result;
    }
};

template <typename Index, typename BinaryOp>
class DmlScatterNDBinaryKernel : public DmlKernel
{
    absl::optional<DmlBuffer> strides_buffer_;

  public:
    using InitHelper = ResourceScatterNDInitHelper<Index>;

    DmlScatterNDBinaryKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const Tensor params_tensor =
            init_helper->GetParamsTensor(ctx->GetOpKernelContext());

        const TensorShape& in_out_shape = params_tensor.shape();
        const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
        const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

        const int64_t indices_last_dim =
            indices_shape.dim_size(indices_shape.dims() - 1);

        const TensorShape flat_indices_shape = {
            indices_shape.num_elements() / indices_last_dim,
            indices_last_dim,
        };

        const int64_t slice_dim =
            (indices_shape.dims() > 1)
                ? indices_shape.dim_size(indices_shape.dims() - 1)
                : 1;

        int64_t slice_size = 1;
        for (int64_t i = slice_dim; i < in_out_shape.dims(); ++i)
        {
            slice_size *= in_out_shape.dim_size(i);
        }

        const int64_t safe_slice_dim = (slice_dim < 1) ? 1 : slice_dim;
        const int64_t num_updates =
            indices_shape.num_elements() / safe_slice_dim;

        const TensorShape flat_updates_shape = {
            num_updates,
            slice_size,
        };

        const TensorShape flat_in_out_shape = {
            in_out_shape.num_elements() / slice_size,
            slice_size,
        };

        const TensorShape strides_shape = {indices_last_dim};
        const TF_DataType indices_dtype = ctx->GetInputDataType(1);

        DmlTensorInfo in_out_tensor;
        in_out_tensor.desc = DmlTensorDesc::Create(
            params_tensor.dtype(),
            flat_in_out_shape,
            flat_in_out_shape);

        DmlTensorInfo indices_tensor;
        indices_tensor.desc = DmlTensorDesc::Create(
            indices_dtype,
            flat_indices_shape,
            flat_indices_shape);

        DmlTensorInfo updates_tensor;
        updates_tensor.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            flat_updates_shape,
            flat_updates_shape);

        DmlTensorInfo strides_tensor;
        strides_tensor.desc =
            DmlTensorDesc::Create(indices_dtype, strides_shape, strides_shape);

        DmlKernelTensors tensors;
        tensors.inputs =
            {in_out_tensor, indices_tensor, updates_tensor, strides_tensor};
        tensors.outputs = {in_out_tensor};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto graph = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(graph, 0, inputs[0]);
        auto indices = dml::InputTensor(graph, 1, inputs[1]);
        auto updates = dml::InputTensor(graph, 2, inputs[2]);
        auto strides = dml::InputTensor(graph, 3, inputs[3]);

        // TODO: Remove the Is64BitIntegerType hack when DML has a more solid
        // solution for 64 bit datatypes
        // TFDML #24881131
        auto result = BinaryOp()(
            graph,
            input,
            indices,
            updates,
            strides,
            Is64BitIntegerType(indices_dtype),
            Is64BitSignedIntegerType(params_tensor.dtype()));

        const uint32_t buffer_size =
            indices_last_dim * DataTypeSize(indices_dtype);
        strides_buffer_ = ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
            ctx->GetOpKernelContext()->raw(),
            buffer_size);

        OP_REQUIRES(
            ctx->GetOpKernelContext(),
            strides_buffer_,
            errors::ResourceExhausted(
                "OOM when allocating a buffer of ",
                buffer_size,
                " bytes"));

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            graph.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto init_helper = ctx->GetInitializationHelper<InitHelper>();
        absl::Cleanup lock_cleanup = [init_helper] { init_helper->Unlock(); };

        const Tensor params_tensor =
            init_helper->GetParamsTensor(ctx->GetOpKernelContext());

        const Tensor indices_tensor = ctx->GetInputTensor(1);

        const int64_t indices_last_dim =
            indices_tensor.dim_size(indices_tensor.dims() - 1);

        absl::InlinedVector<Index, 8> strides(indices_last_dim);
        Index stride = 1;

        for (int i = indices_last_dim - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= params_tensor.dim_size(i);
        }

        auto byte_ptr = reinterpret_cast<const uint8_t*>(strides.data());
        auto byte_span =
            absl::MakeSpan(byte_ptr, strides.size() * sizeof(Index));

        const auto status_or_event =
            ctx->GetDmlDeviceContext()->CopyHostToBuffer(
                strides_buffer_->Region(),
                byte_span);

        TF_RETURN_IF_ERROR(status_or_event.status());

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
            strides_buffer_->GetBufferBinding(),
        };

        DmlGpuEvent gpu_event;
        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1>
            output_bindings;
        const bool isTensorInput = init_helper->IsTensorInput();
        if (isTensorInput)
        {
            D3D12BufferRegion output_buffer =
                ctx->GetDmlDeviceContext()->GetBufferForTensor(
                    ctx->GetOutputTensor(0));
            output_bindings.push_back(output_buffer.GetBufferBinding());

            auto status_or_event =
                DmlKernel::Compute(ctx, input_bindings, output_bindings);
            if (!status_or_event.ok())
            {
                return status_or_event;
            }
        }
        else
        {
            DmlBuffer output_buffer =
                ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
                    ctx->GetOpKernelContext()->raw(),
                    input_buffers[0].SizeInBytes());
            output_bindings.push_back(output_buffer.GetBufferBinding());

            auto status_or_event =
                DmlKernel::Compute(ctx, input_bindings, output_bindings);
            if (!status_or_event.ok())
            {
                return status_or_event;
            }
            ctx->GetDmlDeviceContext()->CopyBufferToBuffer(
                input_buffers[0],
                output_buffer.Region());
        }
        gpu_event = ctx->GetDmlDeviceContext()->InsertUavBarrier();
        return gpu_event;
    }
};

template <typename type>
using ScatterNdPlusOp = ScatterNdBinaryOperation<
    std::plus<dml::Expression>,
    DML_REDUCE_FUNCTION_SUM,
    type>;

template <typename type>
using ScatterNdMinusOp = ScatterNdBinaryOperation<
    std::minus<dml::Expression>,
    DML_REDUCE_FUNCTION_SUM,
    type>;

template <typename Index, typename DataType>
using ScatterNdNonAliasingAddKernel = typename KernelDefinition<
    ops::ScatterNdNonAliasingAdd,
    DmlKernelWrapper<
        DmlScatterNDBinaryKernel<Index, ScatterNdPlusOp<DataType>>,
        GetOutputShapeAsInputShapeHelper>>::
    template WithTypeConstraint<
        ops::ScatterNdNonAliasingAdd::Attribute::Tindices,
        DataTypeToEnum<Index>()>::
        template WithTypeConstraint<
            ops::ScatterNdNonAliasingAdd::Attribute::T,
            DataTypeToEnum<DataType>()>;

static void RegisterScatterNdNonAliasingAdd()
{
    ScatterNdNonAliasingAddKernel<int32_t, float>::Register();
    ScatterNdNonAliasingAddKernel<int64_t, float>::Register();
    ScatterNdNonAliasingAddKernel<int32_t, Eigen::half>::Register();
    ScatterNdNonAliasingAddKernel<int64_t, Eigen::half>::Register();
    ScatterNdNonAliasingAddKernel<int32_t, int64_t>::Register();
    ScatterNdNonAliasingAddKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ScatterNdAddKernel = typename KernelDefinition<
    ops::ScatterNdAdd,
    DmlKernelWrapper<
        DmlScatterNDBinaryKernel<Index, ScatterNdPlusOp<DataType>>,
        GetOutputShapeAsInputShapeHelper>>::
    template WithTypeConstraint<
        ops::ScatterNdAdd::Attribute::Tindices,
        DataTypeToEnum<Index>()>::
        template WithTypeConstraint<
            ops::ScatterNdAdd::Attribute::T,
            DataTypeToEnum<DataType>()>;

static void RegisterScatterNdAdd()
{
    ScatterNdAddKernel<int32_t, float>::Register();
    ScatterNdAddKernel<int64_t, float>::Register();
    ScatterNdAddKernel<int32_t, Eigen::half>::Register();
    ScatterNdAddKernel<int64_t, Eigen::half>::Register();
    ScatterNdAddKernel<int32_t, int64_t>::Register();
    ScatterNdAddKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ScatterNdSubKernel = typename KernelDefinition<
    ops::ScatterNdSub,
    DmlKernelWrapper<
        DmlScatterNDBinaryKernel<Index, ScatterNdMinusOp<DataType>>,
        GetOutputShapeAsInputShapeHelper>>::
    template WithTypeConstraint<
        ops::ScatterNdSub::Attribute::Tindices,
        DataTypeToEnum<Index>()>::
        template WithTypeConstraint<
            ops::ScatterNdSub::Attribute::T,
            DataTypeToEnum<DataType>()>;

static void RegisterScatterNdSub()
{
    ScatterNdSubKernel<int32_t, float>::Register();
    ScatterNdSubKernel<int64_t, float>::Register();
    ScatterNdSubKernel<int32_t, Eigen::half>::Register();
    ScatterNdSubKernel<int64_t, Eigen::half>::Register();
    ScatterNdSubKernel<int32_t, int64_t>::Register();
    ScatterNdSubKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ScatterNdUpdateKernel = typename KernelDefinition<
    ops::ScatterNdUpdate,
    DmlKernelWrapper<
        DmlScatterNDUpdateKernel<Index>,
        GetOutputShapeAsInputShapeHelper>>::
    template WithTypeConstraint<
        ops::ScatterNdUpdate::Attribute::Tindices,
        DataTypeToEnum<Index>()>::
        template WithTypeConstraint<
            ops::ScatterNdUpdate::Attribute::T,
            DataTypeToEnum<DataType>()>;

static void RegisterScatterNdUpdate()
{
    ScatterNdUpdateKernel<int32_t, float>::Register();
    ScatterNdUpdateKernel<int64_t, float>::Register();
    ScatterNdUpdateKernel<int32_t, Eigen::half>::Register();
    ScatterNdUpdateKernel<int64_t, Eigen::half>::Register();
    ScatterNdUpdateKernel<int32_t, int64_t>::Register();
    ScatterNdUpdateKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterNdAddKernel = typename KernelDefinition<
    ops::ResourceScatterNdAdd,
    DmlKernelWrapper<
        DmlScatterNDBinaryKernel<Index, ScatterNdPlusOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<ops::ResourceScatterNdAdd::Argument::ref>::
        template WithTypeConstraint<
            ops::ResourceScatterNdAdd::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterNdAdd::Attribute::T,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterNdAdd()
{
    ResourceScatterNdAddKernel<int32_t, float>::Register();
    ResourceScatterNdAddKernel<int64_t, float>::Register();
    ResourceScatterNdAddKernel<int32_t, Eigen::half>::Register();
    ResourceScatterNdAddKernel<int64_t, Eigen::half>::Register();
    ResourceScatterNdAddKernel<int32_t, int64_t>::Register();
    ResourceScatterNdAddKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterNdSubKernel = typename KernelDefinition<
    ops::ResourceScatterNdSub,
    DmlKernelWrapper<
        DmlScatterNDBinaryKernel<Index, ScatterNdMinusOp<DataType>>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<ops::ResourceScatterNdSub::Argument::ref>::
        template WithTypeConstraint<
            ops::ResourceScatterNdSub::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterNdSub::Attribute::T,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterNdSub()
{
    ResourceScatterNdSubKernel<int32_t, float>::Register();
    ResourceScatterNdSubKernel<int64_t, float>::Register();
    ResourceScatterNdSubKernel<int32_t, Eigen::half>::Register();
    ResourceScatterNdSubKernel<int64_t, Eigen::half>::Register();
    ResourceScatterNdSubKernel<int32_t, int64_t>::Register();
    ResourceScatterNdSubKernel<int64_t, int64_t>::Register();
}

template <typename Index, typename DataType>
using ResourceScatterNdUpdateKernel = typename KernelDefinition<
    ops::ResourceScatterNdUpdate,
    DmlKernelWrapper<
        DmlScatterNDUpdateKernel<Index>,
        NoOutputShapeHelper,
        DmlKernelCachePolicy::Never>>::
    template WithHostMemoryArguments<
        ops::ResourceScatterNdUpdate::Argument::ref>::
        template WithTypeConstraint<
            ops::ResourceScatterNdUpdate::Attribute::Tindices,
            DataTypeToEnum<Index>()>::
            template WithTypeConstraint<
                ops::ResourceScatterNdUpdate::Attribute::T,
                DataTypeToEnum<DataType>()>;

static void RegisterResourceScatterNdUpdate()
{
    ResourceScatterNdUpdateKernel<int32_t, float>::Register();
    ResourceScatterNdUpdateKernel<int64_t, float>::Register();
    ResourceScatterNdUpdateKernel<int32_t, Eigen::half>::Register();
    ResourceScatterNdUpdateKernel<int64_t, Eigen::half>::Register();
    ResourceScatterNdUpdateKernel<int32_t, int64_t>::Register();
    ResourceScatterNdUpdateKernel<int64_t, int64_t>::Register();
}

void RegisterKernels_ScatterNd()
{
    RegisterScatterNdNonAliasingAdd();
    RegisterScatterNdAdd();
    RegisterScatterNdSub();
    RegisterScatterNdUpdate();
    RegisterResourceScatterNdAdd();
    RegisterResourceScatterNdSub();
    RegisterResourceScatterNdUpdate();
}

} // namespace tfdml