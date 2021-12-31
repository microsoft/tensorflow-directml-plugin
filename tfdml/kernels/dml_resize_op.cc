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

// CalculateResizeScale determines the float scaling factor.
float CalculateResizeScale(
    int64_t in_size,
    int64_t out_size,
    bool align_corners)
{
    return (align_corners && out_size > 1)
               ? (in_size - 1) / static_cast<float>(out_size - 1)
               : in_size / static_cast<float>(out_size);
}

struct ImageResizerState
{
    explicit ImageResizerState(bool align_corners, bool half_pixel_centers)
        : align_corners_(align_corners),
          half_pixel_centers_(half_pixel_centers)
    {
    }

    // ValidateAndCalculateOutputSize checks the bounds on the input tensors
    // and requested size, sets up some of the resizing state such as the
    // height_scale and width_scale, and calculates the output size.
    // If any of these operations fails, it sets an error status in
    // the context, which the caller must check.
    void ValidateAndCalculateOutputSize(
        OpKernelContext* context,
        const Tensor& input)
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
                "input must be 4-dimensional",
                input.shape().DebugString()));
        const Tensor& shape_t = context->input(1);
        OP_REQUIRES(
            context,
            shape_t.dims() == 1,
            errors::InvalidArgument(
                "shape_t must be 1-dimensional",
                shape_t.shape().DebugString()));
        OP_REQUIRES(
            context,
            shape_t.NumElements() == 2,
            errors::InvalidArgument(
                "shape_t must have two elements",
                shape_t.shape().DebugString()));
        auto shape_data = reinterpret_cast<const int32_t*>(shape_t.raw_data());
        batch_size = input.dim_size(0);
        out_height = shape_data[0];
        out_width = shape_data[1];
        OP_REQUIRES(
            context,
            input.dim_size(1) < std::numeric_limits<int32_t>::max() &&
                input.dim_size(2) < std::numeric_limits<int32_t>::max(),
            errors::InvalidArgument(
                "input sizes must be between 0 and max int32_t"));

        in_height = static_cast<int32_t>(input.dim_size(1));
        in_width = static_cast<int32_t>(input.dim_size(2));
        channels = input.dim_size(3);
        OP_REQUIRES(
            context,
            out_height > 0 && out_width > 0,
            errors::InvalidArgument("output dimensions must be positive"));
        OP_REQUIRES(
            context,
            channels > 0,
            errors::InvalidArgument("image must have at least one channel"));
        OP_REQUIRES(
            context,
            input.dim_size(1) > 0 && input.dim_size(2) > 0,
            errors::InvalidArgument("input image must be of non-zero size"));
        height_scale =
            CalculateResizeScale(in_height, out_height, align_corners_);
        width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

        // Guard against overflows
        OP_REQUIRES(
            context,
            ceilf((out_height - 1) * height_scale) <=
                static_cast<float>(std::numeric_limits<int64_t>::max()),
            errors::InvalidArgument(
                "input image height scale would cause an overflow"));
        OP_REQUIRES(
            context,
            ceilf((out_width - 1) * width_scale) <= static_cast<float>(INT_MAX),
            errors::InvalidArgument(
                "input image width scale would cause an overflow"));
    }

    int64_t batch_size;
    int64_t out_height;
    int64_t out_width;
    int64_t in_height;
    int64_t in_width;
    int64_t channels;
    float height_scale;
    float width_scale;
    Tensor* output = nullptr;

  private:
    bool align_corners_;
    bool half_pixel_centers_;
};

class ResizeInitHelper : public InitializationHelper
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

    ResizeInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        image_resizer_state_.emplace(
            ImageResizerState(attr->align_corners, attr->half_pixel_centers));

        image_resizer_state_->ValidateAndCalculateOutputSize(
            ctx,
            ctx->input(0));
    }

    bool AlignCorners() const { return attr_->align_corners; }
    bool HalfPixelCenters() const { return attr_->half_pixel_centers; }

    const ImageResizerState& GetImageResizerState() const
    {
        assert(image_resizer_state_.has_value());
        return *image_resizer_state_;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
    absl::optional<ImageResizerState> image_resizer_state_;

}; // namespace tensorflow

using InitHelper = ResizeInitHelper;

class GetResizeShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);

        const ImageResizerState& image_resizer_state =
            init_helper->GetImageResizerState();

        // TF's Resize tensors are always in NHWC format
        TensorShape output_shape(
            {image_resizer_state.batch_size,
             image_resizer_state.out_height,
             image_resizer_state.out_width,
             image_resizer_state.channels});

        return {std::move(output_shape)};
    }
};

template <DML_INTERPOLATION_MODE interpolation_mode>
class DmlResizeKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper;

    explicit DmlResizeKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        TensorShape input_tensor_shape = ctx->GetInputTensorShape(0);
        TensorShape output_tensor_shape = ctx->GetOutputTensorShape(0);

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_tensor_shape,
            input_tensor_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_tensor_shape,
            output_tensor_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto result = dml::InputTensor(scope, 0, inputs[0]);

        if (input_tensor_shape == output_tensor_shape)
        {
            result = dml::Identity(result);
        }
        else
        {
            bool align_corners = init_helper->AlignCorners();
            bool half_pixel_centers = init_helper->HalfPixelCenters();
            float input_offset = 0.5f;
            float output_offset = -0.5f;

            if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR)
            {
                input_offset = half_pixel_centers ? 0.5f : 0.0f;
                output_offset = half_pixel_centers ? -0.5f : 0.0f;
            }
            else
            {
                input_offset = align_corners ? 0.0f : 0.5f;
                output_offset = half_pixel_centers ? -0.5f : 0.0f;
            }

            const ImageResizerState& image_resizer_state =
                init_helper->GetImageResizerState();

            float height_scale = 1.0f / image_resizer_state.height_scale;
            float width_scale = 1.0f / image_resizer_state.width_scale;
            float scales[] = {1.0f, height_scale, width_scale, 1.0f};

            float input_pixel_offsets[] =
                {0.5f, input_offset, input_offset, 0.5f};
            float output_pixel_offsets[] =
                {-0.5f, output_offset, output_offset, -0.5f};

            auto output_sizes = NarrowTensorShape(output_tensor_shape);

            result = dml::Resample(
                result,
                dml::TensorDesc::Dimensions(
                    output_sizes.begin(),
                    output_sizes.end()),
                interpolation_mode,
                scales,
                input_pixel_offsets,
                output_pixel_offsets);
        }

        // For Bilinear, the output is always in float32 but the input can be
        // either float16 or float32
        if (result.GetOutputDesc().dataType != output.desc.GetDmlDataType())
        {
            result = dml::Cast(result, output.desc.GetDmlDataType());
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterResizeBilinear()
{
    using K = KernelDefinition<
        ops::ResizeBilinear,
        DmlKernelWrapper<
            DmlResizeKernel<DML_INTERPOLATION_MODE_LINEAR>,
            GetResizeShapeHelper>>::
        WithHostMemoryArgument<ops::ResizeBilinear::Argument::size>;

    RegisterWithTypes<
        K,
        ops::ResizeBilinear::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResizeNearestNeighbor()
{
    using K = KernelDefinition<
        ops::ResizeNearestNeighbor,
        DmlKernelWrapper<
            DmlResizeKernel<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>,
            GetResizeShapeHelper>>::
        WithHostMemoryArgument<ops::ResizeNearestNeighbor::Argument::size>;

    RegisterWithTypes<
        K,
        ops::ResizeNearestNeighbor::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_Resize()
{
    RegisterResizeBilinear();
    RegisterResizeNearestNeighbor();
}

} // namespace tfdml
