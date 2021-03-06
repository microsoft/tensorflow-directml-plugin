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

class UnpackInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num", &output_count));
        }

        int axis;
        int output_count;
    };

    UnpackInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor& input = ctx->input(0);
        input_shape_ = input.shape();

        positive_axis_ =
            attr->axis < 0 ? attr->axis + input_shape_.dims() : attr->axis;

        OP_REQUIRES(
            ctx,
            0 <= positive_axis_ && positive_axis_ < input_shape_.dims(),
            errors::InvalidArgument(
                "axis = ",
                attr->axis,
                " not in [",
                -input_shape_.dims(),
                ", ",
                input_shape_.dims(),
                ")"));

        OP_REQUIRES(
            ctx,
            input_shape_.dims() > 0 &&
                input_shape_.dim_size(positive_axis_) == attr->output_count,
            errors::InvalidArgument(
                "Input shape axis ",
                positive_axis_,
                " must equal ",
                attr->output_count,
                ", got shape ",
                input_shape_.DebugString()));
    }

    int GetAxis() const { return positive_axis_; }
    int GetOutputCount() const { return attr_->output_count; }
    const TensorShape& GetInputShape() const { return input_shape_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    TensorShape input_shape_;
    int positive_axis_;
};

using InitHelper = UnpackInitHelper;

class UnpackShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        int axis = init_helper->GetAxis();
        int output_count = init_helper->GetOutputCount();
        const TensorShape& input_shape = init_helper->GetInputShape();

        TensorShape output_shape(input_shape);
        output_shape.RemoveDim(axis);

        return std::vector<TensorShape>(output_count, output_shape);
    }
};

class DmlUnpackKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper;

    explicit DmlUnpackKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        TensorShape original_input_shape = ctx->GetInputTensorShape(0);
        int axis = init_helper->GetAxis();

        // We can collapse all dimensions to the left together and all
        // dimensions to the right together. This allows us to send tensors with
        // an "unlimited" number of dimensions to DirectML
        int left_dim_size = 1;
        int right_dim_size = 1;

        for (int i = 0; i < axis; ++i)
        {
            left_dim_size *= original_input_shape.dim_size(i);
        }

        for (int i = axis + 1; i < original_input_shape.dims(); ++i)
        {
            right_dim_size *= original_input_shape.dim_size(i);
        }

        int axis_size = original_input_shape.dim_size(axis);
        TensorShape input_shape({left_dim_size, axis_size, right_dim_size});

        DmlKernelTensors tensors;

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_shape,
            input_shape);
        tensors.inputs = {input};

        TensorShape output_shape({left_dim_size, 1, right_dim_size});

        for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i)
        {
            DmlTensorInfo output_info;
            output_info.kernel_index = i;
            output_info.desc = DmlTensorDesc::Create(
                ctx->GetOutputDataType(i),
                output_shape,
                output_shape);

            tensors.outputs.push_back(std::move(output_info));
        }

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_SPLIT_OPERATOR_DESC desc = {};
        desc.Axis = kNchwDimensionCount - 2;
        desc.OutputCount = outputs.size();
        desc.InputTensor = &inputs[0];
        desc.OutputTensors = outputs.data();

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SPLIT, &desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

void RegisterKernels_Unpack()
{
    using K = KernelDefinition<
        ops::Unpack,
        DmlKernelWrapper<DmlUnpackKernel, UnpackShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::Unpack::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_UINT8,
        TF_BOOL,
        TF_INT32>();
}

} // namespace tfdml