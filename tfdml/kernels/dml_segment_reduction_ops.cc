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

#include "tfdml/kernels/pch.h"

namespace tfdml
{

class SegmentReductionInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    SegmentReductionInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& data = ctx->input(0);
        const Tensor& segment_ids = ctx->input(1);
        const Tensor& num_segments = ctx->input(2);

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(num_segments.shape()),
            errors::InvalidArgument(
                "num_segments should be a scalar, not shape ",
                num_segments.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
            errors::InvalidArgument(
                "data.shape = ",
                data.shape().DebugString(),
                " does not start with segment_ids.shape = ",
                segment_ids.shape().DebugString()));

        const int64_t output_rows = static_cast<int64_t>(
            num_segments.dtype() == TF_INT32 ? num_segments.base<int32_t>()[0]
                                             : num_segments.base<int64_t>()[0]);
        OP_REQUIRES(
            ctx,
            output_rows >= 0,
            errors::InvalidArgument(
                "Input num_segments == ",
                output_rows,
                " must not be negative."));

        output_shape_.AddDim(output_rows);
        for (int i = segment_ids.dims(); i < data.dims(); i++)
        {
            output_shape_.AddDim(data.dim_size(i));
        }
    }

    const TensorShape& GetOutputShape() const { return output_shape_; }

  private:
    TensorShape output_shape_;
};

class SegmentReductionShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const SegmentReductionInitializationHelper*>(
                initialization_helper);

        return {init_helper->GetOutputShape()};
    }
};

template <DML_REDUCE_FUNCTION reduce_function, typename TData>
constexpr TData GetIdentityValue()
{
    if (reduce_function == DML_REDUCE_FUNCTION_MULTIPLY)
    {
        return static_cast<TData>(1);
    }

    if (reduce_function == DML_REDUCE_FUNCTION_SUM)
    {
        return static_cast<TData>(0);
    }

    if (reduce_function == DML_REDUCE_FUNCTION_MIN)
    {
        return std::numeric_limits<TData>::max();
    }

    if (reduce_function == DML_REDUCE_FUNCTION_MAX)
    {
        return std::numeric_limits<TData>::lowest();
    }
}

template <DML_REDUCE_FUNCTION reduce_function, typename TData>
struct SegmentReductionFunctor
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression data,
        dml::Expression segment_ids,
        uint32_t num_segments)
    {
        dml::TensorDesc::Dimensions row_indices_sizes({1, num_segments, 1, 1});

        auto row_indices = dml::FillValueSequence(
            scope,
            row_indices_sizes,
            segment_ids.GetOutputDesc().dataType,
            dml::ScalarUnion(0, segment_ids.GetOutputDesc().dataType),
            dml::ScalarUnion(1, segment_ids.GetOutputDesc().dataType));

        auto data_sizes = data.GetOutputDesc().sizes;

        dml::TensorDesc::Dimensions broadcasted_sizes({
            1,
            num_segments,
            data_sizes[data_sizes.size() - 2],
            data_sizes[data_sizes.size() - 1],
        });

        auto broadcasted_row_indices = dml::Reinterpret(
            row_indices,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 1, 0, 0}));

        auto broadcasted_data = dml::Reinterpret(
            data,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 0, broadcasted_sizes[3], 1}));

        auto broadcasted_segment_ids = dml::Reinterpret(
            segment_ids,
            broadcasted_sizes,
            dml::TensorDesc::Dimensions({0, 0, 1, 0}));

        TData identity_value = GetIdentityValue<reduce_function, TData>();

        auto broadcasted_identity =
            dml::ScalarTensor<TData>(scope, identity_value, broadcasted_sizes);

        auto sparse_data = dml::If(
            broadcasted_row_indices == broadcasted_segment_ids,
            broadcasted_data,
            broadcasted_identity);

        auto result = dml::Reduce(sparse_data, reduce_function, {2});

        return result;
    }
};

template <typename SegmentReductionOp>
class DmlSegmentReductionKernel : public DmlKernel
{
  public:
    using InitHelper = SegmentReductionInitializationHelper;

    explicit DmlSegmentReductionKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& data_shape = ctx->GetInputTensorShape(0);
        const TensorShape& segment_ids_shape = ctx->GetInputTensorShape(1);
        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

        const TensorShape flat_data_shape({
            segment_ids_shape.num_elements(),
            data_shape.num_elements() / segment_ids_shape.num_elements(),
        });

        const TensorShape flat_segment_ids_shape({
            segment_ids_shape.num_elements(),
        });

        const TensorShape flat_output_shape({
            output_shape.dim_size(0),
            output_shape.num_elements() / output_shape.dim_size(0),
        });

        DmlTensorInfo data_tensor_info;
        data_tensor_info.kernel_index = 0;
        data_tensor_info.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            flat_data_shape,
            flat_data_shape);

        DmlTensorInfo segment_ids_tensor_info;
        segment_ids_tensor_info.kernel_index = 1;
        segment_ids_tensor_info.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            flat_segment_ids_shape,
            flat_segment_ids_shape);

        DmlTensorInfo output_tensor_info;
        output_tensor_info.kernel_index = 0;
        output_tensor_info.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            flat_output_shape,
            flat_output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {data_tensor_info, segment_ids_tensor_info};
        tensors.outputs = {output_tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto data = dml::InputTensor(scope, 0, inputs[0]);
        auto segment_ids = dml::InputTensor(scope, 1, inputs[1]);

        auto result = SegmentReductionOp()(
            scope,
            data,
            segment_ids,
            output_shape.dim_size(0));

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

// The memory of this implementation is too big at the moment, so we should
// revisit this operator later if we want to support it with a new DML API
/*
#define REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL_INDEX(type, name, op,  \
                                                         index_type)      \
  REGISTER_KERNEL_BUILDER(Name(name)                                      \
                              .Device(DEVICE_DML)                         \
                              .HostMemory("num_segments")                 \
                              .TypeConstraint<type>("T")                  \
                              .TypeConstraint<index_type>("Tindices"),    \
                          DmlKernelWrapper<DmlSegmentReductionKernel<op>, \
                                           SegmentReductionShapeHelper>)

#define REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL(type, name, op)         \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL_INDEX(type, name, op, int32); \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL_INDEX(type, name, op, int64);


#define REGISTER_UNSORTED_SEGMENT_REDUCTION_DML_KERNEL(type)               \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL(type, "UnsortedSegmentSum",   \
                                             SegmentReductionSumOp<type>); \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL(type, "UnsortedSegmentMax",   \
                                             SegmentReductionMaxOp<type>); \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL(type, "UnsortedSegmentMin",   \
                                             SegmentReductionMinOp<type>); \
  REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL(type, "UnsortedSegmentProd",  \
                                             SegmentReductionProdOp<type>);

TF_CALL_float(REGISTER_UNSORTED_SEGMENT_REDUCTION_DML_KERNEL);
TF_CALL_half(REGISTER_UNSORTED_SEGMENT_REDUCTION_DML_KERNEL);
TF_CALL_int32(REGISTER_UNSORTED_SEGMENT_REDUCTION_DML_KERNEL);
#undef REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL
#undef REGISTER_UNSORTED_SEGMENT_REDUCTION_KERNEL_INDEX
#undef REGISTER_UNSORTED_SEGMENT_REDUCTION_DML_KERNEL
*/

template <typename type>
using SegmentReductionSumOp =
    SegmentReductionFunctor<DML_REDUCE_FUNCTION_SUM, type>;

template <typename type>
using SegmentReductionMaxOp =
    SegmentReductionFunctor<DML_REDUCE_FUNCTION_MAX, type>;

template <typename type>
using SegmentReductionMinOp =
    SegmentReductionFunctor<DML_REDUCE_FUNCTION_MIN, type>;

template <typename type>
using SegmentReductionProdOp =
    SegmentReductionFunctor<DML_REDUCE_FUNCTION_MULTIPLY, type>;

template <typename DataType, typename IndicesType>
using UnsortedSegmentSum = typename KernelDefinition<
    ops::UnsortedSegmentSum,
    DmlKernelWrapper<
        DmlSegmentReductionKernel<
            SegmentReductionFunctor<DML_REDUCE_FUNCTION_SUM, DataType>>,
        SegmentReductionShapeHelper>>::
    template WithHostMemoryArguments<
        ops::UnsortedSegmentSum::Argument::num_segments>::
        template WithTypeConstraint<
            ops::UnsortedSegmentSum::Attribute::T,
            DataTypeToEnum<DataType>()>::
            template WithTypeConstraint<
                ops::UnsortedSegmentSum::Attribute::Tindices,
                DataTypeToEnum<IndicesType>()>;

static void RegisterUnsortedSegmentSum()
{
    UnsortedSegmentSum<float, int32_t>::Register();
    UnsortedSegmentSum<float, int64_t>::Register();
    UnsortedSegmentSum<Eigen::half, int32_t>::Register();
    UnsortedSegmentSum<Eigen::half, int64_t>::Register();
    UnsortedSegmentSum<int32_t, int32_t>::Register();
    UnsortedSegmentSum<int32_t, int64_t>::Register();
}

template <typename DataType, typename IndicesType>
using UnsortedSegmentMax = typename KernelDefinition<
    ops::UnsortedSegmentMax,
    DmlKernelWrapper<
        DmlSegmentReductionKernel<
            SegmentReductionFunctor<DML_REDUCE_FUNCTION_MAX, DataType>>,
        SegmentReductionShapeHelper>>::
    template WithHostMemoryArguments<
        ops::UnsortedSegmentMax::Argument::num_segments>::
        template WithTypeConstraint<
            ops::UnsortedSegmentMax::Attribute::T,
            DataTypeToEnum<DataType>()>::
            template WithTypeConstraint<
                ops::UnsortedSegmentMax::Attribute::Tindices,
                DataTypeToEnum<IndicesType>()>;

static void RegisterUnsortedSegmentMax()
{
    UnsortedSegmentMax<float, int32_t>::Register();
    UnsortedSegmentMax<float, int64_t>::Register();
    UnsortedSegmentMax<Eigen::half, int32_t>::Register();
    UnsortedSegmentMax<Eigen::half, int64_t>::Register();
    UnsortedSegmentMax<int32_t, int32_t>::Register();
    UnsortedSegmentMax<int32_t, int64_t>::Register();
}

template <typename DataType, typename IndicesType>
using UnsortedSegmentMin = typename KernelDefinition<
    ops::UnsortedSegmentMin,
    DmlKernelWrapper<
        DmlSegmentReductionKernel<
            SegmentReductionFunctor<DML_REDUCE_FUNCTION_MIN, DataType>>,
        SegmentReductionShapeHelper>>::
    template WithHostMemoryArguments<
        ops::UnsortedSegmentMin::Argument::num_segments>::
        template WithTypeConstraint<
            ops::UnsortedSegmentMin::Attribute::T,
            DataTypeToEnum<DataType>()>::
            template WithTypeConstraint<
                ops::UnsortedSegmentMin::Attribute::Tindices,
                DataTypeToEnum<IndicesType>()>;

static void RegisterUnsortedSegmentMin()
{
    UnsortedSegmentMin<float, int32_t>::Register();
    UnsortedSegmentMin<float, int64_t>::Register();
    UnsortedSegmentMin<Eigen::half, int32_t>::Register();
    UnsortedSegmentMin<Eigen::half, int64_t>::Register();
    UnsortedSegmentMin<int32_t, int32_t>::Register();
    UnsortedSegmentMin<int32_t, int64_t>::Register();
}

template <typename DataType, typename IndicesType>
using UnsortedSegmentProd = typename KernelDefinition<
    ops::UnsortedSegmentProd,
    DmlKernelWrapper<
        DmlSegmentReductionKernel<
            SegmentReductionFunctor<DML_REDUCE_FUNCTION_MULTIPLY, DataType>>,
        SegmentReductionShapeHelper>>::
    template WithHostMemoryArguments<
        ops::UnsortedSegmentProd::Argument::num_segments>::
        template WithTypeConstraint<
            ops::UnsortedSegmentProd::Attribute::T,
            DataTypeToEnum<DataType>()>::
            template WithTypeConstraint<
                ops::UnsortedSegmentProd::Attribute::Tindices,
                DataTypeToEnum<IndicesType>()>;

static void RegisterUnsortedSegmentProd()
{
    UnsortedSegmentProd<float, int32_t>::Register();
    UnsortedSegmentProd<float, int64_t>::Register();
    UnsortedSegmentProd<Eigen::half, int32_t>::Register();
    UnsortedSegmentProd<Eigen::half, int64_t>::Register();
    UnsortedSegmentProd<int32_t, int32_t>::Register();
    UnsortedSegmentProd<int32_t, int64_t>::Register();
}

void RegisterKernels_SegmentReduction()
{
    RegisterUnsortedSegmentSum();
    RegisterUnsortedSegmentMax();
    RegisterUnsortedSegmentMin();
    RegisterUnsortedSegmentProd();
}

} // namespace tfdml