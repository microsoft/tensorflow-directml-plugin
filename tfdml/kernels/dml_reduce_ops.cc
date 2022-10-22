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

#include "absl/strings/str_join.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

template <typename Tperm>
Status SimplifyHelper(
    const TensorShape& data,
    const Tensor& axis,
    absl::InlinedVector<bool, 4>& bitmap)
{
    auto axis_vec = reinterpret_cast<const Tperm*>(axis.raw_data());
    for (int64_t i = 0; i < axis.NumElements(); ++i)
    {
        Tperm index = axis_vec[i];
        if (index < -data.dims() || index >= data.dims())
        {
            return errors::InvalidArgument(
                "Invalid reduction dimension (",
                index,
                " for input with ",
                data.dims(),
                " dimension(s)");
        }
        index = (index + data.dims()) % data.dims();
        if (bitmap[index])
        {
            return errors::InvalidArgument(
                "Invalid reduction arguments: Axes contains duplicate "
                "dimension: ",
                index);
        }
        bitmap[index] = true;
    }
    return Status::OK();
}

class ReductionHelper
{
  public:
    ReductionHelper() : reduce_first_axis_(false) {}

    Status Simplify(
        const TensorShape& data,
        const Tensor& axis,
        const bool keep_dims)
    {
        // bitmap[i] indicates whether to reduce data along i-th axis.
        absl::InlinedVector<bool, 4> bitmap(data.dims(), false);
        if (axis.dtype() == TF_INT32)
        {
            TF_RETURN_IF_ERROR(SimplifyHelper<int32_t>(data, axis, bitmap));
        }
        else
        {
            TF_RETURN_IF_ERROR(SimplifyHelper<int64_t>(data, axis, bitmap));
        }
        // Output tensor's dim sizes.
        out_shape_.clear();
        for (int i = 0; i < data.dims(); ++i)
        {
            if (!bitmap[i])
            {
                // If we are not reducing along dimension i.
                out_shape_.push_back(data.dim_size(i));
            }
            else if (keep_dims)
            {
                // We are reducing along dimension i, but we want to keep the
                // same number of dimensions, so we set the dimension of i to
                // '1'.
                out_shape_.push_back(1);
            }
        }

        // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
        // the input data before doing the reduction on the resulting
        // tensor.  The shape of the reduction is a reshape of the final
        // output.

        // We'll skip the leading 1s.
        int dim_index = 0;
        for (; dim_index < data.dims(); ++dim_index)
        {
            if (data.dim_size(dim_index) != 1) break;
        }
        if (dim_index >= data.dims())
        {
            // Special case. The input is essentially a scalar.
            reduce_first_axis_ = true;
        }
        else
        {
            // Starting from the (dim_index)-th dimension, dimensions
            // alternates between runs that need to be reduced and runs that
            // don't.
            //
            // NOTE: If a dimension has size 1, we group it as the current
            // run so that we can minimize the number of runs.
            //
            // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
            // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
            // and reduce by axes = [1] (i.e., the output is shape [6]).
            reduce_first_axis_ = bitmap[dim_index];
            data_reshape_.push_back(data.dim_size(dim_index));
            ++dim_index;
            for (; dim_index < data.dims(); ++dim_index)
            {
                const auto size = data.dim_size(dim_index);
                if (size == 1)
                {
                    bitmap[dim_index] = bitmap[dim_index - 1];
                }
                if (bitmap[dim_index - 1] != bitmap[dim_index])
                {
                    // Starts a new run of reduce or !reduce.
                    data_reshape_.push_back(size);
                }
                else
                {
                    // Continue a run of reduce or !reduce.
                    data_reshape_.back() *= size;
                }
            }
            // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
            // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
            // otherwise, data_reshape_[0, 2, 4, ...] is.
            for (size_t i = reduce_first_axis_ ? 1 : 0;
                 i < data_reshape_.size();
                 i += 2)
            {
                out_reshape_.push_back(data_reshape_[i]);
            }
        }

        TF_VLog(
            1,
            "data reshape: %s",
            absl::StrJoin(data_reshape_, ",").c_str());
        TF_VLog(
            1,
            "out  reshape: %s",
            absl::StrJoin(out_reshape_, ",").c_str());
        TF_VLog(1, "out    shape: %s", absl::StrJoin(out_shape_, ",").c_str());
        return Status::OK();
    }

    // The final output shape must be allocated with this shape.
    TensorShape out_shape() const
    {
        TensorShape shape;
        for (auto size : out_shape_)
        {
            shape.AddDim(size);
        }
        return shape;
    }

    // The reduction is on a reshaped tensor of this rank.
    int ndims() const { return data_reshape_.size(); }

    // True if need to reduce the 0-th dimension.
    bool reduce_first_axis() const { return reduce_first_axis_; }

    // Shape of shuffled input
    TensorShape data_reshape() const
    {
        TensorShape shape;
        for (auto s : data_reshape_)
            shape.AddDim(s);
        return shape;
    }

  private:
    bool reduce_first_axis_; // True if need to reduce the 0-th dimension.
    absl::InlinedVector<int64_t, 4>
        data_reshape_; // Reshape data before reduction.
    absl::InlinedVector<int64_t, 4> out_shape_; // The final output shape.
    absl::InlinedVector<int64_t, 4>
        out_reshape_; // Reshape output for reduction.
};

template <DML_REDUCE_FUNCTION reduce_function>
class ReduceInitializationHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            if (!ctx->GetAttr("keep_dims", &keep_dims).ok())
            {
                keep_dims = false;
            }
        }

        bool keep_dims;
    };

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        const Tensor& data_tensor = ctx->input(0);

        if (output_shapes[0].num_elements() == 0)
        {
            return true;
        }

        // TF's Prod and All operators are different to the other reductions in
        // that reduction of an empty tensor is defined to return 1, not 0.
        // Because of this, reduction of empty tensors needs to be handled by
        // the kernel to explicitly output 1. Min and Max are also different
        // because they need to return inf and -inf, respectively.
        if (reduce_function == DML_REDUCE_FUNCTION_MULTIPLY ||
            reduce_function == DML_REDUCE_FUNCTION_MIN ||
            reduce_function == DML_REDUCE_FUNCTION_MAX)
        {
            return false;
        }

        // Ignore the axes tensor when deciding whether to no-op. This is
        // because it's legal to have an empty axes tensor, which turns this
        // reduction into an identity.
        if (data_tensor.NumElements() == 0)
        {
            return true;
        }

        return false;
    }

    absl::optional<int> GetForwardableInputIndex(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes,
        int outputIndex) const override
    {
        // For reduce, we can only forward input 0 to output 0
        if (outputIndex != 0)
        {
            return {};
        }

        absl::optional<int> return_val;
        if (IsIdentity())
        {
            return_val = 0;
        }

        return return_val;
    }

    ReduceInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        // We delegate most of the work to TF's existing ReductionHelper
        const Tensor& data_tensor = ctx->input(0);
        const Tensor& axes_tensor = ctx->input(1);

        static constexpr bool is_arg_function =
            reduce_function == DML_REDUCE_FUNCTION_ARGMIN ||
            reduce_function == DML_REDUCE_FUNCTION_ARGMAX;

        if (is_arg_function)
        {
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(axes_tensor.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    axes_tensor.shape().DebugString()));

            const int32_t dim =
                reinterpret_cast<const int32_t*>(axes_tensor.raw_data())[0];
            const int input_dims = data_tensor.dims();

            int axis = dim < 0 ? dim + input_dims : dim;

            OP_REQUIRES(
                ctx,
                axis < input_dims,
                errors::InvalidArgument(
                    "Expected dimension in the range [",
                    -input_dims,
                    ", ",
                    input_dims,
                    "), but got ",
                    dim));
            OP_REQUIRES(
                ctx,
                data_tensor.dim_size(axis) > 0,
                errors::InvalidArgument(
                    "Reduction axis ",
                    dim,
                    " is empty in shape ",
                    data_tensor.shape().DebugString()));
        }

        OP_REQUIRES_OK(
            ctx,
            reduction_helper_
                .Simplify(data_tensor.shape(), axes_tensor, attr->keep_dims));

        OP_REQUIRES(
            ctx,
            reduction_helper_.data_reshape().dims() <= 8,
            errors::InvalidArgument(
                "DML doesn't support more than 8 dimensions for Reduction "
                "after "
                "simplifying the inputs and collapsing axes together."));

        // Euclidean Norm scalar is a special case that is never identity
        bool is_euclidean_scalar = reduce_function == DML_REDUCE_FUNCTION_L2 &&
                                   data_tensor.NumElements() == 1;

        is_identity_ = !is_arg_function && !is_euclidean_scalar &&
                       (reduction_helper_.ndims() == 0 ||
                        (reduction_helper_.ndims() == 1 &&
                         !reduction_helper_.reduce_first_axis()));
    }

    const ReductionHelper& GetReductionHelper() const
    {
        return reduction_helper_;
    }

    bool IsIdentity() const { return is_identity_; }

  private:
    ReductionHelper reduction_helper_;
    bool is_identity_;
};

template <DML_REDUCE_FUNCTION reduce_function>
using InitHelper = ReduceInitializationHelper<reduce_function>;

template <DML_REDUCE_FUNCTION reduce_function>
class ReduceOutputShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper = static_cast<const InitHelper<reduce_function>*>(
            initialization_helper);
        return {init_helper->GetReductionHelper().out_shape()};
    }
};

template <typename T>
T EmptyKernelReturnValue(DML_REDUCE_FUNCTION reduce_function)
{
    switch (reduce_function)
    {
    case DML_REDUCE_FUNCTION_MULTIPLY: return static_cast<T>(1);
    case DML_REDUCE_FUNCTION_MIN:
        return std::numeric_limits<T>::has_infinity
                   ? std::numeric_limits<T>::infinity()
                   : std::numeric_limits<T>::max();
    case DML_REDUCE_FUNCTION_MAX:
        return std::numeric_limits<T>::has_infinity
                   ? -std::numeric_limits<T>::infinity()
                   : std::numeric_limits<T>::min();
    default: LogFatal("Invalid reduce function type.");
    }
}

template <>
bool EmptyKernelReturnValue(DML_REDUCE_FUNCTION reduce_function)
{
    switch (reduce_function)
    {
    case DML_REDUCE_FUNCTION_MIN: return true;
    case DML_REDUCE_FUNCTION_MAX: return false;
    default: LogFatal("Invalid reduce function type.");
    }
}

template <DML_REDUCE_FUNCTION reduce_function>
class DmlReduceKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper<reduce_function>;

    explicit DmlReduceKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        // Use TF's existing ReductionHelper to help compute axes for reduction.
        // The ReductionHelper does a couple of useful things.
        //
        // One, it collapses adjacent reduction axes together. For example if
        // you have a 5D (e.g. NCDHW) tensor and want to reduce along the 2nd
        // axis (e.g. 'D'), you can collapse the NC dimensions together and the
        // HW dimensions together, then perform a 3D reduction along the 1st
        // axis. This allows us to perform reductions across higher-dimensional
        // tensors than we would usually support, since DML only supports 4D
        // reductions.
        //
        // Second, if the number of axes that need reduction exceed what's
        // supported, the ReductionHelper also provides shapes for transposing
        // the dimensions of the input tensor such that all the reduction axes
        // are shuffled to the end. This allows an N-dimensional reduction to be
        // performed as a 2D reduction, albeit at the cost of a tensor copy. We
        // don't use this facility in DML, though. If the dimensionality of the
        // reduction (after collapsing adjacent axes) exceeds what DML supports,
        // this kernel returns an error.
        const ReductionHelper& reduce_helper =
            init_helper->GetReductionHelper();

        constexpr bool is_special_empty_kernel =
            reduce_function == DML_REDUCE_FUNCTION_MULTIPLY ||
            reduce_function == DML_REDUCE_FUNCTION_MIN ||
            reduce_function == DML_REDUCE_FUNCTION_MAX;

        // Special-case for Prod operator: when reducing an empty tensor, we
        // explicitly need to return a value of 1.0 (not zero, which is the
        // default for a no-op'd operator)
        if (is_special_empty_kernel &&
            ctx->GetInputTensorShape(0).num_elements() == 0)
        {
            TF_DataType output_type = ctx->GetOutputDataType(0);

            DmlKernelTensors tensors;
            tensors.outputs.resize(1);

            TensorShape dml_output_shape({
                ctx->GetOutputTensorShape(0).num_elements(),
            });

            tensors.outputs[0].emplace();
            tensors.outputs[0]->desc = DmlTensorDesc::Create(
                output_type,
                dml_output_shape,
                dml_output_shape);
            tensors.outputs[0]->kernel_index = 0;

            auto value_datatype = GetDmlDataTypeFromTfDataType(output_type);
            DML_SCALAR_UNION value{};

            switch (value_datatype)
            {
            case DML_TENSOR_DATA_TYPE_FLOAT32: {
                value.Float32 = EmptyKernelReturnValue<float>(reduce_function);
            }
            break;

            case DML_TENSOR_DATA_TYPE_FLOAT16: {
                // Copy the bits as a UINT16
                value.UInt16 =
                    EmptyKernelReturnValue<Eigen::half>(reduce_function).x;
            }
            break;

            case DML_TENSOR_DATA_TYPE_UINT32: {
                value.UInt32 =
                    EmptyKernelReturnValue<uint32_t>(reduce_function);
            }
            break;

            case DML_TENSOR_DATA_TYPE_INT32: {
                value.Int32 = EmptyKernelReturnValue<int32_t>(reduce_function);
            }
            break;

            case DML_TENSOR_DATA_TYPE_UINT8: {
                value.UInt8 = EmptyKernelReturnValue<bool>(reduce_function);
            }
            break;

            case DML_TENSOR_DATA_TYPE_UINT64: {
                value.UInt64 =
                    EmptyKernelReturnValue<uint64_t>(reduce_function);
            }
            break;

            case DML_TENSOR_DATA_TYPE_INT64: {
                value.Int64 = EmptyKernelReturnValue<int64_t>(reduce_function);
            }
            break;

            default: {
                assert(false);
                LogFatal("Unsupported datatype");
            }
            }

            const auto output_sizes = tensors.outputs[0]->desc.GetSizes();

            auto scope = dml::Graph(ctx->GetDmlDevice());
            auto result = dml::FillValueConstant(
                scope,
                dml::TensorDimensions(output_sizes.begin(), output_sizes.end()),
                value_datatype,
                value);

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());

            return;
        }

        if (init_helper->IsIdentity())
        {
            // Since the reduce helper may have removed dimensions of size 1, we
            // can't just take its shape as-is. We know that this is an identity
            // scenario, so we can just collapse all dimensions into one and use
            // the same tensor desc for both the input and the output.
            TensorShape in_out_shape(
                {1, 1, 1, ctx->GetInputTensorShape(0).num_elements()});

            DmlTensorInfo in_out_tensor;
            in_out_tensor.kernel_index = 0;
            in_out_tensor.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(0),
                in_out_shape,
                in_out_shape);

            DmlKernelTensors tensors;
            tensors.inputs = {in_out_tensor};
            tensors.outputs = {in_out_tensor};

            auto input_descs = GetDmlTensorDescs(tensors.inputs);
            auto scope = dml::Graph(ctx->GetDmlDevice());
            auto result =
                dml::Identity(dml::InputTensor(scope, 0, input_descs[0]));

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());

            return;
        }

        // Unlike other reduction operators, the behavior for arg functions when
        // there are no axes to reduce is to output 0's everywhere
        is_no_op_ = is_arg_function_ && (reduce_helper.ndims() == 0 ||
                                         (reduce_helper.ndims() == 1 &&
                                          !reduce_helper.reduce_first_axis()));

        if (is_no_op_)
        {
            zero_outputs_ = true;
            InitializeAsNoOp(ctx);
            return;
        }

        // The input shape after adjacent reduction axes have been collapsed.
        const TensorShape& input_shape = reduce_helper.data_reshape();

        uint32_t reduce_axis_offset =
            input_shape.dims() >= kNchwDimensionCount
                ? 0
                : kNchwDimensionCount - input_shape.dims();

        // Compute the DML reduce axes based on the input shape. If
        // reduce_first_axis() is true we reduce over axes 0 and 2, otherwise we
        // reduce over axes 1 and 3 (up to the dimension count of the input.)
        uint32_t first_reduce_axis = reduce_helper.reduce_first_axis() ? 0 : 1;
        absl::InlinedVector<uint32_t, 4> reduce_axes;
        for (uint32_t axis = first_reduce_axis; axis < input_shape.dims();
             axis += 2)
        {
            reduce_axes.push_back(axis + reduce_axis_offset);
        }

        // Use the axes and input shape to compute the output shape as required
        // by DML. We can't use the TF-style output shape as computed by the
        // ReductionHelper, because DML requires a very specific output tensor
        // shape for reduction (namely, that reduced axes must have dimension 1
        // in the output tensor)
        TensorShape output_shape;
        for (int i = 0; i < input_shape.dims(); ++i)
        {
            uint32_t axis = i + reduce_axis_offset;
            const bool is_reduce_axis =
                std::count(reduce_axes.begin(), reduce_axes.end(), axis) > 0;

            if (is_reduce_axis)
            {
                output_shape.AddDim(1);
            }
            else
            {
                output_shape.AddDim(input_shape.dim_size(i));
            }
        }

        assert(input_shape.dims() == output_shape.dims());

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_shape,
            input_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto result = dml::InputTensor(scope, 0, input_descs[0]);

        if (is_arg_function_)
        {
            const DML_TENSOR_DATA_TYPE dml_output_data_type =
                GetDmlDataTypeFromTfDataType(ctx->GetOutputDataType(0));

            // ARGMAX and ARGMIN in DML don't support outputs with less than 32
            // bit precision
            const bool is_low_precision_output =
                dml_output_data_type == DML_TENSOR_DATA_TYPE_INT16 ||
                dml_output_data_type == DML_TENSOR_DATA_TYPE_INT8;

            auto casted_dml_output_data_type = dml_output_data_type;

            if (is_low_precision_output)
            {
                casted_dml_output_data_type = DML_TENSOR_DATA_TYPE_INT32;
            }
            else
            {
                const bool is_low_precision_unsigned_output =
                    dml_output_data_type == DML_TENSOR_DATA_TYPE_UINT16 ||
                    dml_output_data_type == DML_TENSOR_DATA_TYPE_UINT8;

                if (is_low_precision_unsigned_output)
                {
                    casted_dml_output_data_type = DML_TENSOR_DATA_TYPE_UINT32;
                }
            }

            result = dml::Reduce(
                result,
                reduce_function,
                reduce_axes,
                casted_dml_output_data_type);

            // Cast back to the original TensorFlow low precision type
            if (dml_output_data_type != casted_dml_output_data_type)
            {
                result = dml::Cast(result, dml_output_data_type);
            }
        }
        else
        {
            result = dml::Reduce(result, reduce_function, reduce_axes);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const
    {
        if (zero_outputs_)
        {
            Tensor& output = ctx->GetOutputTensor(0);
            ctx->GetDmlDeviceContext()->ZeroBuffer(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
        }

        if (is_no_op_)
        {
            return ctx->GetDmlDeviceContext()->GetCurrentCompletionEvent();
        }

        return DmlKernel::Compute(ctx);
    }

  private:
    bool is_no_op_ = false;
    bool zero_outputs_ = false;

    // ARGMIN and ARGMAX are special reduce functions that can never be replaced
    // by identity
    static constexpr bool is_arg_function_ =
        reduce_function == DML_REDUCE_FUNCTION_ARGMIN ||
        reduce_function == DML_REDUCE_FUNCTION_ARGMAX;
};

template <DML_REDUCE_FUNCTION reduce_function>
using DmlReduceWrapper = DmlKernelWrapper<
    DmlReduceKernel<reduce_function>,
    ReduceOutputShapeHelper<reduce_function>>;

template <typename op, DML_REDUCE_FUNCTION reduce_function, TF_DataType Tidx>
using HostReductionIndicesKernel =
    typename KernelDefinition<op, DmlReduceWrapper<reduce_function>>::
        template WithHostMemoryArguments<op::Argument::reduction_indices>::
            template WithTypeConstraint<op::Attribute::Tidx, Tidx>;

template <typename op, DML_REDUCE_FUNCTION reduce_function, TF_DataType Tidx>
using HostDimensionKernel =
    typename KernelDefinition<op, DmlReduceWrapper<reduce_function>>::
        template WithHostMemoryArguments<op::Argument::dimension>::
            template WithTypeConstraint<op::Attribute::Tidx, Tidx>;

void RegisterSum()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Sum, DML_REDUCE_FUNCTION_SUM, TF_INT32>,
        ops::Sum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();

    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Sum, DML_REDUCE_FUNCTION_SUM, TF_INT64>,
        ops::Sum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

void RegisterMean()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::Mean,
            DML_REDUCE_FUNCTION_AVERAGE,
            TF_INT32>,
        ops::Mean::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::Mean,
            DML_REDUCE_FUNCTION_AVERAGE,
            TF_INT64>,
        ops::Mean::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterProd()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::Prod,
            DML_REDUCE_FUNCTION_MULTIPLY,
            TF_INT32>,
        ops::Prod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();

    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::Prod,
            DML_REDUCE_FUNCTION_MULTIPLY,
            TF_INT64>,
        ops::Prod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32>();
}

void RegisterMin()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Min, DML_REDUCE_FUNCTION_MIN, TF_INT32>,
        ops::Min::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Min, DML_REDUCE_FUNCTION_MIN, TF_INT64>,
        ops::Min::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterMax()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Max, DML_REDUCE_FUNCTION_MAX, TF_INT32>,
        ops::Max::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();

    RegisterWithTypes<
        HostReductionIndicesKernel<ops::Max, DML_REDUCE_FUNCTION_MAX, TF_INT64>,
        ops::Max::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

void RegisterEuclideanNorm()
{
    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::EuclideanNorm,
            DML_REDUCE_FUNCTION_L2,
            TF_INT32>,
        ops::EuclideanNorm::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    RegisterWithTypes<
        HostReductionIndicesKernel<
            ops::EuclideanNorm,
            DML_REDUCE_FUNCTION_L2,
            TF_INT64>,
        ops::EuclideanNorm::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterArgMin()
{
    RegisterWithTypes<
        HostDimensionKernel<ops::ArgMin, DML_REDUCE_FUNCTION_ARGMIN, TF_INT32>,
        ops::ArgMin::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    RegisterWithTypes<
        HostDimensionKernel<ops::ArgMin, DML_REDUCE_FUNCTION_ARGMIN, TF_INT64>,
        ops::ArgMin::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterArgMax()
{
    RegisterWithTypes<
        HostDimensionKernel<ops::ArgMax, DML_REDUCE_FUNCTION_ARGMAX, TF_INT32>,
        ops::ArgMax::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    RegisterWithTypes<
        HostDimensionKernel<ops::ArgMax, DML_REDUCE_FUNCTION_ARGMAX, TF_INT64>,
        ops::ArgMax::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterAny()
{
    HostReductionIndicesKernel<ops::Any, DML_REDUCE_FUNCTION_MAX, TF_INT32>::
        Register();

    HostReductionIndicesKernel<ops::Any, DML_REDUCE_FUNCTION_MAX, TF_INT64>::
        Register();
}

void RegisterAll()
{
    HostReductionIndicesKernel<ops::All, DML_REDUCE_FUNCTION_MIN, TF_INT32>::
        Register();

    HostReductionIndicesKernel<ops::All, DML_REDUCE_FUNCTION_MIN, TF_INT64>::
        Register();
}

void RegisterKernels_Reduce()
{
    RegisterSum();
    RegisterMean();
    RegisterProd();
    RegisterMin();
    RegisterMax();
    RegisterEuclideanNorm();
    RegisterArgMin();
    RegisterArgMax();
    RegisterAny();
    RegisterAll();
}

} // namespace tfdml