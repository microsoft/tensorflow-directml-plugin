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

template <typename Tidx>
class ScanInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive));
        }

        bool reverse;
        bool exclusive;
    };

    ScanInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor& input = ctx->input(0);
        const Tensor& tensor_axis = ctx->input(1);

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(tensor_axis.shape()),
            errors::InvalidArgument(
                "ScanOp: axis must be a scalar, not ",
                tensor_axis.shape().DebugString()));

        const Tidx axis_arg = tensor_axis.base<Tidx>()[0];
        axis_ = (axis_arg < 0) ? input.dims() + axis_arg : axis_arg;
        OP_REQUIRES(
            ctx,
            axis_ >= 0 && axis_ < input.dims(),
            errors::InvalidArgument(
                "ScanOp: Expected scan axis in the range [",
                -input.dims(),
                ", ",
                input.dims(),
                "), but got ",
                axis_));
    }

    bool IsReverse() const { return attr_->reverse; }
    bool IsExclusive() const { return attr_->exclusive; }
    int64_t GetAxis() const { return axis_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    int64_t axis_;
};

template <typename Tidx, typename ScanFunctor>
class DmlScanKernel : public DmlKernel
{
  public:
    using InitHelper = ScanInitHelper<Tidx>;

    explicit DmlScanKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() == 2);
        assert(ctx->GetOutputCount() == 1);

        const TensorShape& original_input_shape = ctx->GetInputTensorShape(0);

        int64_t axis = init_helper->GetAxis();
        DML_AXIS_DIRECTION axis_direction = init_helper->IsReverse()
                                                ? DML_AXIS_DIRECTION_DECREASING
                                                : DML_AXIS_DIRECTION_INCREASING;

        // Collapse the dimensions to the left and to the right of the axis
        // together
        int left_dim_size = 1;
        for (int i = 0; i < axis; ++i)
        {
            left_dim_size *= original_input_shape.dim_size(i);
        }

        int right_dim_size = 1;
        for (int i = axis + 1; i < original_input_shape.dims(); ++i)
        {
            right_dim_size *= original_input_shape.dim_size(i);
        }

        int axis_dim_size = original_input_shape.dim_size(axis);

        TensorShape tensor_shape(
            {1, left_dim_size, axis_dim_size, right_dim_size});

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            tensor_shape,
            tensor_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        // The non-axis dimensions have already been collapsed together, so the
        // dml axis is always "2"
        constexpr uint32_t dml_axis = 2;

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        const auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
        auto result = ScanFunctor()(
            input_tensor,
            dml_axis,
            axis_direction,
            init_helper->IsExclusive());

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

struct CumsumFunctor
{
    dml::Expression operator()(
        dml::Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axis_direction,
        bool has_exclusive_sum)
    {
        return dml::CumulativeSummation(
            input,
            axis,
            axis_direction,
            has_exclusive_sum);
    }
};

struct CumprodFunctor
{
    dml::Expression operator()(
        dml::Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axis_direction,
        bool has_exclusive_product)
    {
        return dml::CumulativeProduct(
            input,
            axis,
            axis_direction,
            has_exclusive_product);
    }
};

void RegisterCumsum()
{
    using int32_kernel = KernelDefinition<
        ops::Cumsum,
        DmlKernelWrapper<
            DmlScanKernel<int32_t, CumsumFunctor>,
            GetOutputShapeAsInputShapeHelper>>::
        WithTypeConstraint<ops::Cumsum::Attribute::Tidx, TF_INT32>::
            WithHostMemoryArguments<ops::Cumsum::Argument::axis>;

    using int64_kernel = KernelDefinition<
        ops::Cumsum,
        DmlKernelWrapper<
            DmlScanKernel<int64_t, CumsumFunctor>,
            GetOutputShapeAsInputShapeHelper>>::
        WithTypeConstraint<ops::Cumsum::Attribute::Tidx, TF_INT64>::
            WithHostMemoryArguments<ops::Cumsum::Argument::axis>;

    RegisterWithTypes<
        int32_kernel,
        ops::Cumsum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();

    RegisterWithTypes<
        int64_kernel,
        ops::Cumsum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();
}

void RegisterCumprod()
{
    using int32_kernel = KernelDefinition<
        ops::Cumprod,
        DmlKernelWrapper<
            DmlScanKernel<int32_t, CumprodFunctor>,
            GetOutputShapeAsInputShapeHelper>>::
        WithTypeConstraint<ops::Cumprod::Attribute::Tidx, TF_INT32>::
            WithHostMemoryArguments<ops::Cumprod::Argument::axis>;

    using int64_kernel = KernelDefinition<
        ops::Cumprod,
        DmlKernelWrapper<
            DmlScanKernel<int64_t, CumprodFunctor>,
            GetOutputShapeAsInputShapeHelper>>::
        WithTypeConstraint<ops::Cumprod::Attribute::Tidx, TF_INT64>::
            WithHostMemoryArguments<ops::Cumprod::Argument::axis>;

    RegisterWithTypes<
        int32_kernel,
        ops::Cumprod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();

    RegisterWithTypes<
        int64_kernel,
        ops::Cumprod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();
}

void RegisterKernels_Scan()
{
    RegisterCumsum();
    RegisterCumprod();
}

} // namespace tfdml