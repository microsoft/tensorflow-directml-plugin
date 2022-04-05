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

static float CalculateResizeScale(
    int64_t in_size,
    int64_t out_size,
    bool align_corners)
{
    return (align_corners && out_size > 1)
               ? (in_size - 1) / static_cast<float>(out_size - 1)
               : in_size / static_cast<float>(out_size);
}

struct ImageResizerGradientState
{
    explicit ImageResizerGradientState(
        bool align_corners,
        bool half_pixel_centers)
        : align_corners_(align_corners),
          half_pixel_centers_(half_pixel_centers)
    {
    }

    void ValidateAndCalculateScales(
        OpKernelContext* context,
        const Tensor& input,
        const Tensor& original_image)
    {
        OP_REQUIRES(
            context,
            !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
            errors::InvalidArgument("If half_pixel_centers is True, "
                                    "align_corners must be False."));

        OP_REQUIRES(
            context,
            input.dims() == 4,
            errors::InvalidArgument(
                "input_grad must be 4-dimensional",
                input.shape().DebugString()));
        // Resizers always produce float images, so input gradient must
        // always be a float.
        OP_REQUIRES(
            context,
            input.dtype() == TF_FLOAT,
            errors::InvalidArgument(
                "input_grad must be of type float",
                DataTypeString(input.dtype())));

        OP_REQUIRES(
            context,
            original_image.dims() == 4,
            errors::InvalidArgument(
                "original_image must be 4-dimensional",
                original_image.shape().DebugString()));

        // Allocate output and initialize to zeros.
        batch_size = input.dim_size(0);
        channels = input.dim_size(3);
        resized_height = input.dim_size(1);
        resized_width = input.dim_size(2);
        original_height = original_image.dim_size(1);
        original_width = original_image.dim_size(2);

        OP_REQUIRES(
            context,
            original_height < std::numeric_limits<int32_t>::max() &&
                original_width < std::numeric_limits<int32_t>::max(),
            errors::InvalidArgument(
                "original sizes must be between 0 and max int32_t"));

        height_scale = CalculateResizeScale(
            original_height,
            resized_height,
            align_corners_);
        width_scale =
            CalculateResizeScale(original_width, resized_width, align_corners_);
    }

    int64_t batch_size;
    int64_t channels;
    int64_t resized_height;
    int64_t resized_width;
    int64_t original_height;
    int64_t original_width;
    float height_scale;
    float width_scale;
    Tensor* output;

  private:
    bool align_corners_;
    bool half_pixel_centers_;
};

template <DML_INTERPOLATION_MODE interpolation_mode>
class ResizeGradInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("align_corners", &align_corners));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("half_pixel_centers", &half_pixel_centers));
        }

        bool align_corners;
        bool half_pixel_centers;
    };

    ResizeGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor& input = ctx->input(0);

        if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR)
        {
            const Tensor& original_image = ctx->input(1);

            ImageResizerGradientState st(
                attr->align_corners,
                attr->half_pixel_centers);
            st.ValidateAndCalculateScales(ctx, input, original_image);

            if (!ctx->status().ok())
            {
                return;
            }

            out_height_ = original_image.dim_size(1);
            out_width_ = original_image.dim_size(2);
        }
        else
        {
            assert(
                interpolation_mode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR);

            const Tensor& shape_t = ctx->input(1);

            // Validate the input:
            OP_REQUIRES(
                ctx,
                input.dims() == 4,
                errors::InvalidArgument(
                    "input must be 4-dimensional",
                    input.shape().DebugString()));

            // Grab and validate the output shape:
            OP_REQUIRES(
                ctx,
                shape_t.dims() == 1,
                errors::InvalidArgument(
                    "shape_t must be 1-dimensional",
                    shape_t.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                shape_t.NumElements() == 2,
                errors::InvalidArgument(
                    "shape_t must have two elements",
                    shape_t.shape().DebugString()));

            auto sizes = shape_t.base<int32_t>();

            OP_REQUIRES(
                ctx,
                sizes[0] > 0 && sizes[1] > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

            out_height_ = sizes[0];
            out_width_ = sizes[1];
        }

        batch_size_ = input.dim_size(0);
        const int64_t in_height = input.dim_size(1);
        const int64_t in_width = input.dim_size(2);
        channels_ = input.dim_size(3);

        height_scale_ =
            CalculateResizeScale(in_height, out_height_, attr->align_corners);
        width_scale_ =
            CalculateResizeScale(in_width, out_width_, attr->align_corners);
    }

    bool AlignCorners() const { return attr_->align_corners; }
    bool HalfPixelCenters() const { return attr_->half_pixel_centers; }
    int64_t GetBatchSize() const { return batch_size_; }
    int64_t GetChannels() const { return channels_; }
    int64_t GetOutputHeight() const { return out_height_; }
    int64_t GetOutputWidth() const { return out_width_; }
    float GetHeightScale() const { return height_scale_; }
    float GetWidthScale() const { return width_scale_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    int64_t batch_size_;
    int64_t channels_;
    int64_t out_height_;
    int64_t out_width_;
    float height_scale_;
    float width_scale_;
};

template <DML_INTERPOLATION_MODE interpolation_mode>
using InitHelper = ResizeGradInitHelper<interpolation_mode>;

template <DML_INTERPOLATION_MODE interpolation_mode>
class ResizeGradShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const
    {
        const Tensor& input = ctx->input(0);

        auto init_helper = static_cast<const InitHelper<interpolation_mode>*>(
            initialization_helper);

        // TF's Resize tensors are always in NHWC format
        TensorShape output_shape(
            {init_helper->GetBatchSize(),
             init_helper->GetOutputHeight(),
             init_helper->GetOutputWidth(),
             init_helper->GetChannels()});

        return {std::move(output_shape)};
    }
};

template <DML_INTERPOLATION_MODE interpolation_mode>
class DmlResizeGradKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper<interpolation_mode>;

    explicit DmlResizeGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        DmlKernelParams params;
        params.kernel_input_indices = {0};
        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto result = dml::InputTensor(scope, 0, inputs[0]);

        bool align_corners = init_helper->AlignCorners();
        bool half_pixel_centers = init_helper->HalfPixelCenters();
        float input_offset = -0.5f;
        float output_offset = 0.5f;

        if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR)
        {
            input_offset = half_pixel_centers ? -0.5f : 0.0f;
            output_offset = half_pixel_centers ? 0.5f : 0.0f;
        }
        else
        {
            input_offset = half_pixel_centers ? -0.5f : 0.0f;
            output_offset = align_corners ? 0.0f : 0.5f;
        }

        float height_scale = init_helper->GetHeightScale();
        float width_scale = init_helper->GetWidthScale();

        float scales[] = {1.0f, height_scale, width_scale, 1.0f};

        float input_pixel_offsets[] = {-0.5, input_offset, input_offset, -0.5};
        float output_pixel_offsets[] = {0.5, output_offset, output_offset, 0.5};

        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
        auto output_sizes = NarrowTensorShape(output_shape);

        result = dml::ResampleGrad(
            result,
            dml::TensorDesc::Dimensions(
                output_sizes.begin(),
                output_sizes.end()),
            interpolation_mode,
            scales,
            input_pixel_offsets,
            output_pixel_offsets);

        TF_DataType tf_output_data_type = ctx->GetOutputDataType(0);
        auto dml_out_data_type =
            GetDmlDataTypeFromTfDataType(tf_output_data_type);

        if (result.GetOutputDesc().dataType != dml_out_data_type)
        {
            result = dml::Cast(result, dml_out_data_type);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterResizeBilinearGrad()
{
    using K = KernelDefinition<
        ops::ResizeBilinearGrad,
        DmlKernelWrapper<
            DmlResizeGradKernel<DML_INTERPOLATION_MODE_LINEAR>,
            ResizeGradShapeHelper<DML_INTERPOLATION_MODE_LINEAR>>>;

    RegisterWithTypes<
        K,
        ops::ResizeBilinearGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResizeNearestNeighborGrad()
{
    using K = KernelDefinition<
        ops::ResizeNearestNeighborGrad,
        DmlKernelWrapper<
            DmlResizeGradKernel<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>,
            ResizeGradShapeHelper<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>>>::
        WithHostMemoryArguments<ops::ResizeNearestNeighborGrad::Argument::size>;

    RegisterWithTypes<
        K,
        ops::ResizeNearestNeighborGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_ResizeGrad()
{
    RegisterResizeBilinearGrad();
    RegisterResizeNearestNeighborGrad();
}

} // namespace tfdml