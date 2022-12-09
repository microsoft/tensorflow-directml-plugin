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

class CropAndResizeGradImageInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::string method_attr;

            OP_REQUIRES_OK(ctx, ctx->GetAttr("method", &method_attr));
            OP_REQUIRES(
                ctx,
                method_attr == "bilinear" || method_attr == "nearest",
                errors::InvalidArgument(
                    "method must be 'bilinear' or 'nearest'",
                    method_attr));

            interpolation_mode = method_attr == "bilinear"
                                     ? DML_INTERPOLATION_MODE_LINEAR
                                     : DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR;
        }

        DML_INTERPOLATION_MODE interpolation_mode;
    };

    CropAndResizeGradImageInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
        const Tensor& grads = ctx->input(0);
        // The shape of 'boxes' is [num_boxes, 4].
        const Tensor& boxes = ctx->input(1);
        // The shape of 'box_index' is [num_boxes].
        const Tensor& box_index = ctx->input(2);
        // The shape of 'image_size' is [4].
        const Tensor& image_size = ctx->input(3);

        // Validate input shapes.
        OP_REQUIRES(
            ctx,
            grads.dims() == 4,
            errors::InvalidArgument(
                "grads image must be 4-D",
                grads.shape().DebugString()));
        const int crop_height = grads.dim_size(1);
        const int crop_width = grads.dim_size(2);

        OP_REQUIRES(
            ctx,
            crop_height > 0 && crop_width > 0,
            errors::InvalidArgument("grads dimensions must be positive"));

        int num_boxes = 0;
        const TensorShape& boxes_shape = ctx->input(1).shape();
        const TensorShape& box_index_shape = ctx->input(2).shape();

        if (boxes_shape.num_elements() != 0 ||
            box_index_shape.num_elements() != 0)
        {
            OP_REQUIRES(
                ctx,
                boxes_shape.dims() == 2,
                errors::InvalidArgument(
                    "boxes must be 2-D",
                    boxes_shape.DebugString()));

            num_boxes = boxes_shape.dim_size(0);

            OP_REQUIRES(
                ctx,
                boxes_shape.dim_size(1) == 4,
                errors::InvalidArgument("boxes must have 4 columns"));

            OP_REQUIRES(
                ctx,
                box_index_shape.dims() == 1,
                errors::InvalidArgument(
                    "box_index must be 1-D",
                    boxes_shape.DebugString()));

            OP_REQUIRES(
                ctx,
                box_index_shape.dim_size(0) == num_boxes,
                errors::InvalidArgument("box_index has incompatible shape"));
        }

        OP_REQUIRES(
            ctx,
            grads.dim_size(0) == num_boxes,
            errors::InvalidArgument("boxes and grads have incompatible shape"));

        OP_REQUIRES(
            ctx,
            image_size.dims() == 1,
            errors::InvalidArgument(
                "image_size must be 1-D",
                image_size.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            image_size.dim_size(0) == 4,
            errors::InvalidArgument(
                "image_size must have 4 elements",
                image_size.shape().DebugString()));
        auto image_size_vec = image_size.base<int32_t>();
        const int batch_size = image_size_vec[0];
        const int image_height = image_size_vec[1];
        const int image_width = image_size_vec[2];
        const int depth = image_size_vec[3];

        OP_REQUIRES(
            ctx,
            image_height > 0 && image_width > 0,
            errors::InvalidArgument("image dimensions must be positive"));

        OP_REQUIRES(
            ctx,
            grads.dim_size(3) == depth,
            errors::InvalidArgument("image_size and grads are incompatible"));

        output_shape_ =
            TensorShape({batch_size, image_height, image_width, depth});
    }

    TensorShape GetOutputShape() const { return output_shape_; }

    DML_INTERPOLATION_MODE GetInterpolationMode() const
    {
        return attr_->interpolation_mode;
    }

  private:
    TensorShape output_shape_;
    std::shared_ptr<const Attributes> attr_;
};

class CropAndResizeGradImageShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper = static_cast<const CropAndResizeGradImageInitHelper*>(
            initialization_helper);

        return {init_helper->GetOutputShape()};
    }
};

class DmlCropAndResizeGradImageKernel : public DmlKernel
{
  public:
    using InitHelper = CropAndResizeGradImageInitHelper;

    explicit DmlCropAndResizeGradImageKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& output_gradient_shape = ctx->GetOutputTensorShape(0);

        DmlKernelParams params;
        params.kernel_input_indices = {0, 1, 2};
        params.kernel_output_indices = {0};

        using namespace DmlTensorAxes;
        auto layout = {N, H, W, C};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc = CreateTensorDescFromInput(ctx, 0, layout);

        // RoiAlignImageGrad only supports uint32 for the batch indices
        tensors.inputs[2]->desc.ForceUnsignedDataType();
        tensors.outputs[0]->desc = CreateTensorDescFromOutput(ctx, 0, layout);

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        // auto input_gradient = &inputs[0];
        // auto roi = &inputs[1];
        // auto batch_indices = &inputs[2];

        // TF's ROIs are in {y1, x1, y2, x2} order, but DML's are in {x1, y1,
        // x2, y2} order
        // auto roi_sizes = roi->GetOutputDesc().sizes;

        // roi = dml::Reinterpret(roi, {1, roi_sizes[2], 2, 2}, {});

        // auto seq_lengths =
        //     dml::ScalarTensor<uint32_t>(scope, 2, {1, roi_sizes[2], 2, 1});
        // roi = dml::ReverseSubsequences(roi, seq_lengths, 3);

        // // NHWC strides for sizes [1, 1, roiCount, 4]
        // dml::TensorDimensions roiStrides{
        //     roi_sizes[2] * 4,
        //     roi_sizes[2] * 4,
        //     1,
        //     roi_sizes[2],
        // };

        // roi = dml::Reinterpret(roi, roi_sizes, roiStrides);

        // for (int i = 0; i < 4; i++) {
        //     auto x1 = roi[0, 0, i, 1];
        //     roi[0, 0, i, 1] = roi[0, 0, i, 0];
        //     roi[0, 0, i, 0] = x1;
        //     auto x2 = roi[0, 0, i, 3];
        //     roi[0, 0, i, 3] = roi[0, 0, i, 2];
        //     roi[0, 0, i, 2] = x2;
        // }

        const float spatial_scale_x = output_gradient_shape.dim_size(2) - 1;
        const float spatial_scale_y = output_gradient_shape.dim_size(1) - 1;
        constexpr uint32_t minimum_samples_per_output = 1;
        constexpr uint32_t maximum_samples_per_output = 1;
        constexpr float input_pixel_offset = 0;
        constexpr float output_pixel_offset = 0;
        constexpr bool align_region_to_corners = true;
        const uint32_t batch_size = output_gradient_shape.dim_size(0);
        const uint32_t image_height = output_gradient_shape.dim_size(1);
        const uint32_t image_width = output_gradient_shape.dim_size(2);
        constexpr bool compute_output_gradient = true;
        constexpr bool compute_output_roi_gradient = false;

        auto inputGradientTensor = &inputs[0];
        auto roiTensor = &inputs[1];
        auto batchIndicesTensor = &inputs[2];

        DML_ROI_ALIGN_GRAD_OPERATOR_DESC roi_align_grad_desc = {};
        roi_align_grad_desc.InputTensor = nullptr;
        roi_align_grad_desc.InputGradientTensor = inputGradientTensor;
        roi_align_grad_desc.ROITensor = roiTensor;
        roi_align_grad_desc.BatchIndicesTensor = batchIndicesTensor;
        roi_align_grad_desc.OutputGradientTensor = &outputs[0];
        roi_align_grad_desc.OutputROIGradientTensor = nullptr;
        roi_align_grad_desc.ReductionFunction = DML_REDUCE_FUNCTION_AVERAGE; 
        roi_align_grad_desc.InterpolationMode = init_helper->GetInterpolationMode();
        roi_align_grad_desc.SpatialScaleX = spatial_scale_x;
        roi_align_grad_desc.SpatialScaleY = spatial_scale_y;
        roi_align_grad_desc.InputPixelOffset = input_pixel_offset;
        roi_align_grad_desc.OutputPixelOffset = output_pixel_offset;
        roi_align_grad_desc.MinimumSamplesPerOutput = minimum_samples_per_output;
        roi_align_grad_desc.MaximumSamplesPerOutput = maximum_samples_per_output;
        roi_align_grad_desc.AlignRegionsToCorners = align_region_to_corners;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ROI_ALIGN_GRAD, &roi_align_grad_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

void RegisterKernels_CropAndResizeGradImage()
{
    using K = KernelDefinition<
        ops::CropAndResizeGradImage,
        DmlKernelWrapper<
            DmlCropAndResizeGradImageKernel,
            CropAndResizeGradImageShapeHelper>>::
        WithHostMemoryArguments<
            ops::CropAndResizeGradImage::Argument::image_size>;

    RegisterWithTypes<
        K,
        ops::CropAndResizeGradImage::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

} // namespace tfdml
