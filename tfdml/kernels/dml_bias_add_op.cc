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

#include "tfdml/core/dml_operator_helper.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

class BiasAddInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::string data_format_attr;

            if (ctx->GetAttr("data_format", &data_format_attr).ok())
            {
                OP_REQUIRES(
                    ctx,
                    FormatFromString(data_format_attr, &data_format),
                    errors::InvalidArgument("Invalid data format"));
            }
        }

        TensorFormat data_format = FORMAT_NHWC;
    };

    BiasAddInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor input = ctx->input(0);
        const Tensor bias = ctx->input(1);
        const TensorShape& input_shape = input.shape();
        const TensorShape& bias_shape = bias.shape();

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsMatrixOrHigher(input_shape),
            errors::InvalidArgument(
                "Input tensor must be at least 2D: ",
                input_shape.DebugString()));
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsVector(bias_shape),
            errors::InvalidArgument(
                "Biases must be 1D: ",
                bias_shape.DebugString()));

        int channel_count =
            attr->data_format == FORMAT_NCHW
                ? input_shape.dim_size(1) // NCHW always has channel dim in 1
                : input_shape.dim_size(input_shape.dims() - 1);

        OP_REQUIRES(
            ctx,
            bias_shape.dim_size(0) == channel_count,
            errors::InvalidArgument(
                "Must provide as many biases as the last dimension "
                "of the input tensor: ",
                bias_shape.DebugString(),
                " vs. ",
                input_shape.DebugString()));

        // When data_format == FORMAT_NCHW, the broadcasting is done differently
        // from what DML expects (i.e. the trailing dimensions should be padded
        // instead of the leading ones). Therefore, to calculate the correct
        // strides and sizes for DML, we need to manually broadcast the
        // dimensions.
        if (attr->data_format == FORMAT_NCHW)
        {
            bias_shape_ = TensorShape({channel_count, 1, 1});

            if (input_shape.dims() == kNcdhwDimensionCount)
            {
                bias_shape_.AddDim(1);
            }
        }
        else
        {
            bias_shape_ = bias_shape;
        }
    }

    const TensorShape& GetBiasShape() const { return bias_shape_; }
    TensorFormat GetDataFormat() const { return attr_->data_format; }

  private:
    TensorShape bias_shape_;
    const std::shared_ptr<const Attributes> attr_;
};

class DmlBiasAddKernel : public DmlKernel
{
  public:
    using InitHelper = BiasAddInitHelper;

    explicit DmlBiasAddKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        TensorFormat data_format = init_helper->GetDataFormat();
        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        TensorShape output_shape = ctx->GetOutputTensorShape(0);

        // When data_format_ == FORMAT_NCHW, the broadcasting is done
        // differently than what DML expects (i.e. the trailing dimensions
        // should be padded instead of the leading ones). Therefore, to
        // calculate the correct strides and sizes for DML, we need to manually
        // broadcast the dimensions.
        if (data_format == FORMAT_NCHW &&
            input_shape.dims() < kNchwDimensionCount)
        {
            const int64_t missing_dims =
                kNchwDimensionCount - input_shape.dims();
            for (int64_t i = 0; i < missing_dims; i++)
            {
                output_shape.AddDim(1);
            }
        }

        DmlKernelTensors tensors;
        tensors.inputs.resize(2);
        tensors.outputs.resize(1);

        tensors.inputs[0].emplace();
        tensors.inputs[0]->desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                                    output_shape, output_shape);
        tensors.inputs[0]->kernel_index = 0;

        // Broadcast the bias tensor over the output shape
        const TensorShape& bias_physical_shape = init_helper->GetBiasShape();
        tensors.inputs[1].emplace();
        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            output_shape,
            bias_physical_shape);
        tensors.inputs[1]->kernel_index = 1;

        tensors.outputs[0].emplace();
        tensors.outputs[0]->desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);
        tensors.outputs[0]->kernel_index = 0;

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
        add_desc.ATensor = &input_descs[0];
        add_desc.BTensor = &input_descs[1];
        add_desc.OutputTensor = &output_descs[0];

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class BiasAddGradInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::string data_format_attr;

            if (ctx->GetAttr("data_format", &data_format_attr).ok())
            {
                OP_REQUIRES(
                    ctx,
                    FormatFromString(data_format_attr, &data_format),
                    errors::InvalidArgument("Invalid data format"));
            }
        }

        TensorFormat data_format = FORMAT_NHWC;
    };

    BiasAddGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor output_backprop = ctx->input(0);
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
            errors::InvalidArgument(
                "Input tensor must be at least 2D: ",
                output_backprop.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            output_backprop.NumElements() < std::numeric_limits<int32_t>::max(),
            errors::InvalidArgument(
                "BiasGrad requires tensor size <= int32_t max"));

        batch_ = 1;
        height_ = 1;
        width_ = 1;
        depth_ = 1;
        channel_ = 1;

        const TensorShape& output_backprop_shape = output_backprop.shape();

        if (attr->data_format == FORMAT_NHWC)
        {
            int32_t channel_dim = output_backprop_shape.dims() - 1;
            channel_ = static_cast<int32_t>(
                output_backprop_shape.dim_size(channel_dim));
            for (int32_t i = 0; i < channel_dim; i++)
            {
                batch_ *=
                    static_cast<int32_t>(output_backprop_shape.dim_size(i));
            }
        }
        else if (attr->data_format == FORMAT_NCHW)
        {
            batch_ = static_cast<int32_t>(output_backprop_shape.dim_size(0));
            channel_ = static_cast<int32_t>(output_backprop_shape.dim_size(1));
            height_ = static_cast<int32_t>(output_backprop_shape.dim_size(2));
            if (output_backprop_shape.dims() > 3)
            {
                width_ =
                    static_cast<int32_t>(output_backprop_shape.dim_size(3));
            }
            if (output_backprop_shape.dims() > 4)
            {
                depth_ =
                    static_cast<int32_t>(output_backprop_shape.dim_size(4));
            }
        }
    }

    TensorFormat GetDataFormat() const { return attr_->data_format; }
    int32_t GetBatch() const { return batch_; }
    int32_t GetHeight() const { return height_; }
    int32_t GetWidth() const { return width_; }
    int32_t GetDepth() const { return depth_; }
    int32_t GetChannel() const { return channel_; }

  private:
    int32_t batch_, height_, width_, depth_, channel_;
    const std::shared_ptr<const Attributes> attr_;
};

class BiasAddGradShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        const Tensor output_backprop = ctx->input(0);

        auto init_helper =
            static_cast<const BiasAddGradInitHelper*>(initialization_helper);

        TensorShape output_shape{init_helper->GetChannel()};
        return {std::move(output_shape)};
    }
};

class DmlBiasAddGradKernel : public DmlKernel
{
  public:
    using InitHelper = BiasAddGradInitHelper;

    explicit DmlBiasAddGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        int32_t batch = init_helper->GetBatch();
        int32_t height = init_helper->GetHeight();
        int32_t width = init_helper->GetWidth();
        int32_t depth = init_helper->GetDepth();
        int32_t channel = init_helper->GetChannel();

        using namespace DmlTensorAxes;

        // Compute the desired shape of the tensor. Since this is a simple
        // reduction and we know the tensor is packed, we don't actually need
        // all the stated dimensions. We just need to reduce over the channel
        // (whose location depends on layout) and coerce all other dimensions.
        TensorShape input_shape;
        absl::InlinedVector<DmlTensorAxis, DML_TENSOR_DIMENSION_COUNT_MAX>
            input_layout;
        if (init_helper->GetDataFormat() == FORMAT_NHWC)
        {
            // Reduce all but the last dimension, coercing all other dimensions
            // into the batch
            input_shape.AddDim(batch * height * width * depth); // N
            input_shape.AddDim(channel);                        // C
            input_layout = {N, C};
        }
        else
        {
            CHECK(init_helper->GetDataFormat() == FORMAT_NCHW);

            // Reduce the middle dimension, keeping the batch but coercing all
            // spatial dimensions into height
            input_shape.AddDim(batch);                  // N
            input_shape.AddDim(channel);                // C
            input_shape.AddDim(height * width * depth); // H
            input_layout = {N, C, H};
        }

        DmlTensorInfo input_info = {};
        input_info.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0), input_shape, input_shape, input_layout);
        input_info.kernel_index = 0;

        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
        auto output_layout = {C};
        DmlTensorInfo output_info = {};
        output_info.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0), output_shape, output_shape, output_layout);
        output_info.kernel_index = 0;

        DmlKernelTensors tensors = {};
        tensors.inputs.push_back(std::move(input_info));
        tensors.outputs.push_back(std::move(output_info));

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        // Reduce across every channel to produce an output of size {1, C, 1, 1}
        uint32_t reduce_axes[] = {0, 2, 3};

        DML_REDUCE_OPERATOR_DESC reduce_desc = {};
        reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
        reduce_desc.InputTensor = &input_descs[0];
        reduce_desc.OutputTensor = &output_descs[0];
        reduce_desc.AxisCount = TF_ARRAYSIZE(reduce_axes);
        reduce_desc.Axes = reduce_axes;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

static void RegisterBiasAdd()
{
    using K = KernelDefinition<
        ops::BiasAdd,
        DmlKernelWrapper<DmlBiasAddKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::BiasAdd::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterBiasAddV1()
{
    using K = KernelDefinition<
        ops::BiasAddV1,
        DmlKernelWrapper<DmlBiasAddKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::BiasAddV1::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterBiasAddGrad()
{
    using K = KernelDefinition<
        ops::BiasAddGrad,
        DmlKernelWrapper<DmlBiasAddGradKernel, BiasAddGradShapeHelper>>;

    RegisterWithTypes<K, ops::BiasAddGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterKernels_BiasAdd()
{
    RegisterBiasAdd();
    RegisterBiasAddV1();
    RegisterBiasAddGrad();
}

} // namespace tfdml