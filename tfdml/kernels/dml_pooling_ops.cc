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
#include "tfdml/runtime_adapter/kernel_shape_util.h"
#include "tfdml/runtime_adapter/padding.h"

namespace tfdml
{

static Status CheckPaddingSize(
    int64_t window_rows,
    int64_t window_cols,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_left,
    int64_t pad_right)
{
    if (pad_top >= window_rows)
    {
        return errors::InvalidArgument(
            "Top padding ",
            pad_top,
            " needs to be smaller than the window size ",
            window_rows);
    }
    if (pad_bottom >= window_rows)
    {
        return errors::InvalidArgument(
            "Bottom padding ",
            pad_bottom,
            " needs to be smaller than the window size ",
            window_rows);
    }
    if (pad_left >= window_cols)
    {
        return errors::InvalidArgument(
            "Left padding ",
            pad_left,
            " needs to be smaller than the window size ",
            window_cols);
    }
    if (pad_right >= window_cols)
    {
        return errors::InvalidArgument(
            "Right padding ",
            pad_right,
            " needs to be smaller than the window size ",
            window_cols);
    }
    return Status::OK();
}

// A helper class to manage sizes and shapes for pooling operations.
struct PoolParameters
{
    // Updates context->status if there is an invalid input.
    // explicit_paddings has eight elements if padding==EXPLIICT, and zero
    // elements otherwise.
    PoolParameters(
        OpKernelContext* context,
        const std::vector<int32_t>& ksize,
        const std::vector<int32_t>& stride,
        Padding padding,
        absl::Span<const int64_t> explicit_paddings,
        TensorFormat data_format,
        const TensorShape& tensor_in_shape)
    {
        // For maxpooling, tensor_in should have 2 spatial dimensions.
        // Note: the total number of dimensions could be 4 for NHWC, NCHW,
        // or 5 for NCHW_VECT_C.
        OP_REQUIRES(
            context,
            GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
            errors::InvalidArgument(
                "tensor_in_shape must have 2 spatial dimensions. ",
                tensor_in_shape.dims(),
                " ",
                data_format));

        this->data_format = data_format;
        depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
                (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
        tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
        tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
        tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
        window_rows = GetTensorDim(ksize, data_format, 'H');
        window_cols = GetTensorDim(ksize, data_format, 'W');
        depth_window = GetTensorDim(ksize, data_format, 'C');
        row_stride = GetTensorDim(stride, data_format, 'H');
        col_stride = GetTensorDim(stride, data_format, 'W');
        depth_stride = GetTensorDim(stride, data_format, 'C');

        // We only support 2D pooling across width/height and depthwise
        // pooling, not a combination.
        OP_REQUIRES(
            context,
            (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
            errors::Unimplemented(
                "MaxPooling supports exactly one of pooling across depth "
                "or pooling across width/height."));
        if (padding == Padding::EXPLICIT)
        {
            OP_REQUIRES_OK(
                context,
                CheckValidPadding(
                    padding,
                    explicit_paddings,
                    /*num_dims=*/4,
                    data_format));
            GetExplicitPaddingForDim(
                explicit_paddings,
                data_format,
                'H',
                &pad_top,
                &pad_bottom);
            GetExplicitPaddingForDim(
                explicit_paddings,
                data_format,
                'W',
                &pad_left,
                &pad_right);
            OP_REQUIRES_OK(
                context,
                CheckPaddingSize(
                    window_rows,
                    window_cols,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right));
        }

        if (depth_window == 1)
        {
            OP_REQUIRES_OK(
                context,
                GetWindowedOutputSizeVerbose(
                    tensor_in_rows,
                    window_rows,
                    row_stride,
                    padding,
                    &out_height,
                    &pad_top,
                    &pad_bottom));
            OP_REQUIRES_OK(
                context,
                GetWindowedOutputSizeVerbose(
                    tensor_in_cols,
                    window_cols,
                    col_stride,
                    padding,
                    &out_width,
                    &pad_left,
                    &pad_right));
            pad_depth = 0;
            out_depth = depth;
        }
        else
        {
            OP_REQUIRES(
                context,
                depth_window > 0,
                errors::InvalidArgument("depth_window must not be 0"));
            // Our current version of depthwise max pooling does not support
            // any padding, and expects the depth_window to equal the
            // depth_stride (no overlapping).
            OP_REQUIRES(
                context,
                depth % depth_window == 0,
                errors::Unimplemented(
                    "Depthwise max pooling requires the depth window to evenly "
                    "divide the input depth"));
            OP_REQUIRES(
                context,
                depth_stride == depth_window,
                errors::Unimplemented(
                    "Depthwise max pooling requires the depth window to equal "
                    "the depth stride"));

            // The current version of depthwise max is only implemented on CPU.
            OP_REQUIRES(
                context,
                false,
                errors::Unimplemented("Depthwise max pooling is currently only "
                                      "implemented for CPU devices."));

            pad_depth = 0;
            out_depth = depth / depth_window;
        }
    }

    // Returns the shape of the output for "forward" pooling operations.
    TensorShape forward_output_shape()
    {
        if (depth_window == 1)
        {
            // Spatial pooling
            return ShapeFromFormat(
                data_format,
                tensor_in_batch,
                out_height,
                out_width,
                depth);
        }
        else
        {
            // Depthwise pooling
            return TensorShape(
                {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
        }
    }

    int depth;

    int tensor_in_cols;
    int tensor_in_rows;
    int tensor_in_batch;

    int window_rows;
    int window_cols;
    int depth_window;

    int row_stride;
    int col_stride;
    int depth_stride;

    int64_t out_height;
    int64_t out_width;
    int out_depth;

    int64_t pad_top;
    int64_t pad_bottom;
    int64_t pad_left;
    int64_t pad_right;

    int pad_depth;

    TensorFormat data_format;
};

// A helper class to manage sizes and shapes for 3d pooling operations.
struct Pool3dParameters
{
    // Updates context->status if there is an invalid input.
    Pool3dParameters(
        OpKernelContext* context,
        const std::vector<int32_t>& ksize,
        const std::vector<int32_t>& stride,
        Padding padding,
        TensorFormat data_format,
        const TensorShape& tensor_in_shape)
    {
        // For maxpooling, tensor_in should have 4 dimensions.
        OP_REQUIRES(
            context,
            tensor_in_shape.dims() == 5,
            errors::InvalidArgument("tensor_in must be 4-dimensional"));

        this->data_format = data_format;
        depth = GetTensorDim(tensor_in_shape, data_format, 'C');
        tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
        tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
        tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
        tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
        window_planes = GetTensorDim(ksize, data_format, '0');
        window_rows = GetTensorDim(ksize, data_format, '1');
        window_cols = GetTensorDim(ksize, data_format, '2');
        depth_window = GetTensorDim(ksize, data_format, 'C');
        plane_stride = GetTensorDim(stride, data_format, '0');
        row_stride = GetTensorDim(stride, data_format, '1');
        col_stride = GetTensorDim(stride, data_format, '2');
        depth_stride = GetTensorDim(stride, data_format, 'C');

        // We only support 3D pooling across plane/width/height. Depthwise
        // pooling is not supported.
        OP_REQUIRES(
            context,
            depth_window == 1 && depth_stride == 1,
            errors::Unimplemented(
                "Pooling3d only supports pooling across plane/width/height."));

        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSize(
                tensor_in_planes,
                window_planes,
                plane_stride,
                padding,
                &out_plane,
                &pad_planes));
        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSize(
                tensor_in_rows,
                window_rows,
                row_stride,
                padding,
                &out_height,
                &pad_rows));
        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSize(
                tensor_in_cols,
                window_cols,
                col_stride,
                padding,
                &out_width,
                &pad_cols));
    }

    // Returns the shape of the output for "forward" pooling operations.
    TensorShape forward_output_shape()
    {
        return ShapeFromFormat(
            data_format,
            tensor_in_batch,
            {{out_plane, out_height, out_width}},
            depth);
    }

    int depth;

    int tensor_in_planes;
    int tensor_in_cols;
    int tensor_in_rows;
    int tensor_in_batch;

    int window_planes;
    int window_cols;
    int window_rows;
    int depth_window;

    int plane_stride;
    int col_stride;
    int row_stride;
    int depth_stride;

    int64_t out_plane;
    int64_t out_height;
    int64_t out_width;

    int64_t pad_planes;
    int64_t pad_cols;
    int64_t pad_rows;

    TensorFormat data_format;
};

struct DmlPoolValues
{
    absl::InlinedVector<uint32_t, 3> strides;
    absl::InlinedVector<uint32_t, 3> window_size;
    absl::InlinedVector<uint32_t, 3> start_padding;
    absl::InlinedVector<uint32_t, 3> end_padding;
    TensorFormat data_format;
};

class PoolInitHelper : public InitializationHelper
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

            // MaxPoolV2 has ksize and strides in attributes instead of inputs
            if (ctx->HasAttr("ksize"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize));
                OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride));

                OP_REQUIRES(
                    ctx,
                    GetTensorDim(ksize, data_format, 'N') == 1 &&
                        GetTensorDim(stride, data_format, 'N') == 1,
                    errors::Unimplemented("Pooling is not yet supported on the "
                                          "batch dimension."));
            }

            OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));

            if (padding == Padding::EXPLICIT)
            {
                OP_REQUIRES_OK(
                    ctx,
                    ctx->GetAttr("explicit_paddings", &explicit_paddings));
                OP_REQUIRES_OK(
                    ctx,
                    CheckValidPadding(
                        padding,
                        explicit_paddings,
                        data_format_attr.length(),
                        data_format));
            }
        }

        std::vector<int32_t> ksize;
        std::vector<int32_t> stride;
        Padding padding;
        TensorFormat data_format = FORMAT_NHWC;
        std::vector<int64_t> explicit_paddings;
    };

    PoolInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(attr),
          ksize_(attr_->ksize),
          stride_(attr_->stride)
    {
        if (ctx->num_inputs() == 3)
        {
            const Tensor& ksize_tensor = ctx->input(1);
            const Tensor& stride_tensor = ctx->input(2);
            auto ksize_values =
                reinterpret_cast<const int32_t*>(ksize_tensor.raw_data());
            auto stride_values =
                reinterpret_cast<const int32_t*>(stride_tensor.raw_data());

            ksize_.assign(
                ksize_values,
                ksize_values + ksize_tensor.NumElements());
            stride_.assign(
                stride_values,
                stride_values + stride_tensor.NumElements());

            OP_REQUIRES(
                ctx,
                GetTensorDim(ksize_, attr_->data_format, 'N') == 1 &&
                    GetTensorDim(stride_, attr_->data_format, 'N') == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
        }

        OP_REQUIRES(
            ctx,
            ksize_.size() == ctx->input(0).dims(),
            errors::InvalidArgument(
                "Sliding window ksize field must specify",
                ctx->input(0).dims(),
                "dimensions"));

        OP_REQUIRES(
            ctx,
            stride_.size() == ctx->input(0).dims(),
            errors::InvalidArgument(
                "Sliding window stride field must specify",
                ctx->input(0).dims(),
                "dimensions"));

        for (int i = 0; i < ksize_.size(); ++i)
        {
            OP_REQUIRES(
                ctx,
                ksize_[i] != 0,
                errors::InvalidArgument("ksize cannot be zero"));
        }

        if (ctx->input(0).shape().dims() == kNcdhwDimensionCount)
        {
            Pool3dParameters params(
                ctx,
                ksize_,
                stride_,
                attr_->padding,
                attr_->data_format,
                ctx->input(0).shape());

            OP_REQUIRES_OK(ctx, ctx->status());

            output_shape_ = params.forward_output_shape();
        }
        else
        {
            PoolParameters params(
                ctx,
                ksize_,
                stride_,
                attr_->padding,
                attr_->explicit_paddings,
                attr_->data_format,
                ctx->input(0).shape());

            OP_REQUIRES_OK(ctx, ctx->status());

            output_shape_ = params.forward_output_shape();
        }
    }

    const std::vector<int32_t>& GetKernelSizes() const { return ksize_; }
    const std::vector<int32_t>& GetKernelStrides() const { return stride_; }

    absl::Span<const int64_t> GetExplicitPaddings() const
    {
        return attr_->explicit_paddings;
    }

    Padding GetPadding() const { return attr_->padding; }
    TensorFormat GetDataFormat() const { return attr_->data_format; }
    const TensorShape& GetOutputShape() const { return output_shape_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    std::vector<int32_t> ksize_;
    std::vector<int32_t> stride_;
    TensorShape output_shape_;
};

class MaxPoolGradInitHelper : public InitializationHelper
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

            if (ctx->HasAttr("ksize"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize));
                OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride));
            }

            OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));

            if (padding == Padding::EXPLICIT)
            {
                OP_REQUIRES_OK(
                    ctx,
                    ctx->GetAttr("explicit_paddings", &explicit_paddings));
                OP_REQUIRES_OK(
                    ctx,
                    CheckValidPadding(
                        padding,
                        explicit_paddings,
                        data_format_attr.length(),
                        data_format));
            }
        }

        std::vector<int32_t> ksize;
        std::vector<int32_t> stride;
        Padding padding;
        TensorFormat data_format = FORMAT_NHWC;
        std::vector<int64_t> explicit_paddings;
    };

    MaxPoolGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr),
          ksize_(attr_->ksize),
          stride_(attr_->stride)
    {
        if (ctx->num_inputs() == 5)
        {
            const Tensor& ksize_tensor = ctx->input(3);
            const Tensor& stride_tensor = ctx->input(4);

            auto ksize_values =
                reinterpret_cast<const int32_t*>(ksize_tensor.raw_data());
            auto stride_values =
                reinterpret_cast<const int32_t*>(stride_tensor.raw_data());

            ksize_.assign(
                ksize_values,
                ksize_values + ksize_tensor.NumElements());
            stride_.assign(
                stride_values,
                stride_values + stride_tensor.NumElements());
        }

        OP_REQUIRES(
            ctx,
            ksize_.size() == ctx->input(0).dims(),
            errors::InvalidArgument(
                "Sliding window ksize field must specify",
                ctx->input(0).dims(),
                "dimensions"));

        OP_REQUIRES(
            ctx,
            stride_.size() == ctx->input(0).dims(),
            errors::InvalidArgument(
                "Sliding window stride field must specify",
                ctx->input(0).dims(),
                "dimensions"));

        OP_REQUIRES(
            ctx,
            GetTensorDim(ksize_, attr_->data_format, 'N') == 1 &&
                GetTensorDim(stride_, attr_->data_format, 'N') == 1,
            errors::Unimplemented(
                "Pooling is not yet supported on the batch dimension."));

        OP_REQUIRES(
            ctx,
            GetTensorDim(ksize_, attr_->data_format, 'C') == 1 &&
                GetTensorDim(stride_, attr_->data_format, 'C') == 1,
            errors::Unimplemented(
                "MaxPoolingGrad is not yet supported on the depth dimension."));

        const Tensor& input = ctx->input(0);
        const Tensor& tensor_out = ctx->input(1);
        const Tensor& out_backprop = ctx->input(2);
        TensorShape forward_output_shape;

        if (input.shape().dims() == kNcdhwDimensionCount)
        {
            Pool3dParameters params(
                ctx,
                ksize_,
                stride_,
                attr_->padding,
                attr_->data_format,
                input.shape());

            OP_REQUIRES_OK(ctx, ctx->status());

            forward_output_shape = params.forward_output_shape();
        }
        else
        {
            PoolParameters params(
                ctx,
                ksize_,
                stride_,
                attr_->padding,
                attr_->explicit_paddings,
                attr_->data_format,
                input.shape());

            OP_REQUIRES_OK(ctx, ctx->status());

            forward_output_shape = params.forward_output_shape();
        }

        OP_REQUIRES(
            ctx,
            tensor_out.shape() == forward_output_shape,
            errors::InvalidArgument(
                "Expected orig_output shape to be ",
                forward_output_shape.DebugString(),
                ", but got ",
                tensor_out.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            out_backprop.shape() == forward_output_shape,
            errors::InvalidArgument(
                "Expected grad shape to be ",
                forward_output_shape.DebugString(),
                ", but got ",
                out_backprop.shape().DebugString()));
    }

    const std::vector<int32_t>& GetKernelSizes() const { return ksize_; }
    const std::vector<int32_t>& GetKernelStrides() const { return stride_; }
    Padding GetPadding() const { return attr_->padding; }
    TensorFormat GetDataFormat() const { return attr_->data_format; }

    absl::Span<const int64_t> GetExplicitPaddings() const
    {
        return attr_->explicit_paddings;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
    std::vector<int32_t> ksize_;
    std::vector<int32_t> stride_;
};

class AvgPoolGradInitHelper : public InitializationHelper
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

            OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize));
            OP_REQUIRES(
                ctx,
                ksize.size() == data_format_attr.length(),
                errors::InvalidArgument(
                    "Sliding window ksize field must specify",
                    data_format_attr.length(),
                    "dimensions"));

            OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride));
            OP_REQUIRES(
                ctx,
                stride.size() == data_format_attr.length(),
                errors::InvalidArgument(
                    "Sliding window stride field must specify",
                    data_format_attr.length(),
                    "dimensions"));

            OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));

            OP_REQUIRES(
                ctx,
                GetTensorDim(ksize, data_format, 'N') == 1 &&
                    GetTensorDim(stride, data_format, 'N') == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
        }

        std::vector<int32_t> ksize;
        std::vector<int32_t> stride;
        Padding padding;
        TensorFormat data_format = FORMAT_NHWC;
    };

    AvgPoolGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
    }

    const std::vector<int32_t>& GetKernelSizes() const { return attr_->ksize; }

    const std::vector<int32_t>& GetKernelStrides() const
    {
        return attr_->stride;
    }

    absl::Span<const int64_t> GetExplicitPaddings() const { return {}; }
    Padding GetPadding() const { return attr_->padding; }
    TensorFormat GetDataFormat() const { return attr_->data_format; }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class PoolingShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const PoolInitHelper*>(initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

// Helper to get DML API pooling values (window size, strides, and padding) that
// may be either attributes or tensors in TF.
template <typename TInitHelper>
DmlPoolValues GetPoolValuesFromAttributesOrTensors(
    DmlKernelConstruction* ctx,
    const TInitHelper* init_helper,
    const TensorShape& tensor_in_shape)
{
    const std::vector<int32_t>& ksize = init_helper->GetKernelSizes();
    const std::vector<int32_t>& stride = init_helper->GetKernelStrides();
    Padding padding = init_helper->GetPadding();
    TensorFormat data_format = init_helper->GetDataFormat();

    const int64_t tensor_in_rows =
        GetTensorDim(tensor_in_shape, data_format, 'H');
    const int64_t window_rows = GetTensorDim(ksize, data_format, 'H');
    const int64_t row_stride = GetTensorDim(stride, data_format, 'H');
    int64_t out_height = 0;
    int64_t pad_rows_start = 0;
    int64_t pad_rows_end = 0;

    if (padding == Padding::EXPLICIT)
    {
        GetExplicitPaddingForDim(
            init_helper->GetExplicitPaddings(),
            data_format,
            'H',
            &pad_rows_start,
            &pad_rows_end);
    }

    TF_CHECK_OK(GetWindowedOutputSizeVerbose(
        tensor_in_rows,
        window_rows,
        row_stride,
        padding,
        &out_height,
        &pad_rows_start,
        &pad_rows_end));

    const int64_t tensor_in_cols =
        GetTensorDim(tensor_in_shape, data_format, 'W');
    const int64_t window_cols = GetTensorDim(ksize, data_format, 'W');
    const int64_t col_stride = GetTensorDim(stride, data_format, 'W');
    int64_t out_width = 0;
    int64_t pad_cols_start = 0;
    int64_t pad_cols_end = 0;

    if (padding == Padding::EXPLICIT)
    {
        GetExplicitPaddingForDim(
            init_helper->GetExplicitPaddings(),
            data_format,
            'W',
            &pad_cols_start,
            &pad_cols_end);
    }

    TF_CHECK_OK(GetWindowedOutputSizeVerbose(
        tensor_in_cols,
        window_cols,
        col_stride,
        padding,
        &out_width,
        &pad_cols_start,
        &pad_cols_end));

    DmlPoolValues poolValues = {};

    if (tensor_in_shape.dims() == kNcdhwDimensionCount)
    {
        const int64_t tensor_in_depth =
            GetTensorDim(tensor_in_shape, data_format, '0');
        const int64_t window_depth = GetTensorDim(ksize, data_format, '0');
        const int64_t depth_stride = GetTensorDim(stride, data_format, '0');
        int64_t out_depth = 0;
        int64_t pad_depth_start = 0;
        int64_t pad_depth_end = 0;

        if (padding == Padding::EXPLICIT)
        {
            GetExplicitPaddingForDim(
                init_helper->GetExplicitPaddings(),
                data_format,
                'D',
                &pad_depth_start,
                &pad_depth_end);
        }

        TF_CHECK_OK(GetWindowedOutputSizeVerbose(
            tensor_in_depth,
            window_depth,
            depth_stride,
            padding,
            &out_depth,
            &pad_depth_start,
            &pad_depth_end));

        poolValues.strides.push_back(depth_stride);
        poolValues.window_size.push_back(window_depth);
        poolValues.start_padding.push_back(pad_depth_start);
        poolValues.end_padding.push_back(pad_depth_end);
    }

    poolValues.strides.push_back(row_stride);
    poolValues.strides.push_back(col_stride);
    poolValues.window_size.push_back(window_rows);
    poolValues.window_size.push_back(window_cols);
    poolValues.start_padding.push_back(pad_rows_start);
    poolValues.start_padding.push_back(pad_cols_start);
    poolValues.end_padding.push_back(pad_rows_end);
    poolValues.end_padding.push_back(pad_cols_end);
    poolValues.data_format = data_format;

    return poolValues;
}

// Implements the AvgPool, MaxPool, and MaxPoolV2 ops.
template <DML_OPERATOR_TYPE op_type, typename OperatorDesc>
class DmlPoolingKernel : public DmlKernel
{
  public:
    using InitHelper = PoolInitHelper;

    explicit DmlPoolingKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& tensor_in_shape = ctx->GetInputTensorShape(0);
        DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
            ctx,
            init_helper,
            tensor_in_shape);

        // Ignore the kernel size/stride tensors, because DML takes them as
        // attributes and not input tensors
        DmlKernelParams params;
        params.kernel_input_indices = {0};

        // The layout of the input/output tensors is determined by the
        // "data_format"
        auto input_output_layout = GetDmlTensorLayout(
            poolValues.data_format,
            ctx->GetOutputTensorShape(0).dims());

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        OperatorDesc pooling_desc = {};
        pooling_desc.InputTensor = &input_descs[0];
        pooling_desc.OutputTensor = &output_descs[0];
        pooling_desc.DimensionCount = poolValues.strides.size();
        pooling_desc.Strides = poolValues.strides.data();
        pooling_desc.WindowSize = poolValues.window_size.data();
        pooling_desc.StartPadding = poolValues.start_padding.data();
        pooling_desc.EndPadding = poolValues.end_padding.data();
        // AvgPool in TF never includes padding, so for
        // DML_AVERAGE_POOLING_OPERATOR_DESC::IncludePadding we can just leave
        // it as its default-initialized value (false)

        DML_OPERATOR_DESC op_desc = {op_type, &pooling_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlAvgPoolingGradKernel : public DmlKernel
{
  public:
    using InitHelper = AvgPoolGradInitHelper;

    explicit DmlAvgPoolingGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        const TensorShape& tensor_in_shape = ctx->GetOutputTensorShape(0);
        DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
            ctx,
            init_helper,
            tensor_in_shape);

        DmlKernelParams params;
        params.kernel_input_indices = {1};

        auto input_output_layout = GetDmlTensorLayout(
            poolValues.data_format,
            ctx->GetOutputTensorShape(0).dims());

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 1, input_output_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC avg_pooling_grad_desc = {};
        avg_pooling_grad_desc.InputGradientTensor = inputs.data();
        avg_pooling_grad_desc.OutputGradientTensor = outputs.data();
        avg_pooling_grad_desc.DimensionCount = poolValues.strides.size();
        avg_pooling_grad_desc.Strides = poolValues.strides.data();
        avg_pooling_grad_desc.WindowSize = poolValues.window_size.data();
        avg_pooling_grad_desc.StartPadding = poolValues.start_padding.data();
        avg_pooling_grad_desc.EndPadding = poolValues.end_padding.data();
        // AvgPoolGrad in TF never includes padding, so for
        // DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC::IncludePadding we can just
        // leave it as its default-initialized value (false)

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_AVERAGE_POOLING_GRAD,
            &avg_pooling_grad_desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlMaxPoolGradKernel : public DmlKernel
{
  public:
    using InitHelper = MaxPoolGradInitHelper;

    explicit DmlMaxPoolGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& tensor_in_shape = ctx->GetInputTensorShape(0);
        DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
            ctx,
            init_helper,
            tensor_in_shape);

        // TF doesn't use dilations, but DML needs default values of 1.
        uint32_t dilations[] = {1, 1, 1};

        // Ignore the kernel size/stride tensors, because DML takes them as
        // attributes and not input tensors
        DmlKernelParams params;
        params.kernel_input_indices = {0, 2};

        // The layout of the input/output tensors is determined by the
        // "data_format"
        auto input_output_layout = GetDmlTensorLayout(
            poolValues.data_format,
            ctx->GetOutputTensorShape(0).dims());

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 2, input_output_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_MAX_POOLING_GRAD_OPERATOR_DESC max_pooling_grad_desc = {};
        max_pooling_grad_desc.InputTensor = &input_descs[0];
        max_pooling_grad_desc.InputGradientTensor = &input_descs[1];
        max_pooling_grad_desc.OutputGradientTensor = output_descs.data();
        max_pooling_grad_desc.DimensionCount = poolValues.strides.size();
        max_pooling_grad_desc.Strides = poolValues.strides.data();
        max_pooling_grad_desc.WindowSize = poolValues.window_size.data();
        max_pooling_grad_desc.StartPadding = poolValues.start_padding.data();
        max_pooling_grad_desc.EndPadding = poolValues.end_padding.data();
        max_pooling_grad_desc.Dilations = dilations;

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_MAX_POOLING_GRAD,
            &max_pooling_grad_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

using DmlAvgPoolKernel = DmlPoolingKernel<
    DML_OPERATOR_AVERAGE_POOLING,
    DML_AVERAGE_POOLING_OPERATOR_DESC>;
using DmlMaxPoolKernel =
    DmlPoolingKernel<DML_OPERATOR_MAX_POOLING, DML_MAX_POOLING_OPERATOR_DESC>;

void RegisterAvgPool()
{
    using K = KernelDefinition<
        ops::AvgPool,
        DmlKernelWrapper<DmlAvgPoolKernel, PoolingShapeHelper>>;

    RegisterWithTypes<K, ops::AvgPool::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterAvgPool3D()
{
    using K = KernelDefinition<
        ops::AvgPool3D,
        DmlKernelWrapper<DmlAvgPoolKernel, PoolingShapeHelper>>;

    RegisterWithTypes<K, ops::AvgPool3D::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPool()
{
    using K = KernelDefinition<
        ops::MaxPool,
        DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>>;

    RegisterWithTypes<K, ops::MaxPool::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPool3D()
{
    using K = KernelDefinition<
        ops::MaxPool3D,
        DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>>;

    RegisterWithTypes<K, ops::MaxPool3D::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPoolV2()
{
    using K = KernelDefinition<
        ops::MaxPoolV2,
        DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>>::
        WithHostMemoryArguments<
            ops::MaxPoolV2::Argument::ksize,
            ops::MaxPoolV2::Argument::strides>;

    RegisterWithTypes<K, ops::MaxPoolV2::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterAvgPoolGrad()
{
    using K = KernelDefinition<
        ops::AvgPoolGrad,
        DmlKernelWrapper<
            DmlAvgPoolingGradKernel,
            GetOutputShapeFromDimsTensorHelper<0>>>::
        WithHostMemoryArguments<ops::AvgPoolGrad::Argument::orig_input_shape>;

    RegisterWithTypes<K, ops::AvgPoolGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterAvgPool3DGrad()
{
    using K = KernelDefinition<
        ops::AvgPool3DGrad,
        DmlKernelWrapper<
            DmlAvgPoolingGradKernel,
            GetOutputShapeFromDimsTensorHelper<0>>>::
        WithHostMemoryArguments<ops::AvgPool3DGrad::Argument::orig_input_shape>;

    RegisterWithTypes<K, ops::AvgPool3DGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPoolGrad()
{
    using K = KernelDefinition<
        ops::MaxPoolGrad,
        DmlKernelWrapper<
            DmlMaxPoolGradKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::MaxPoolGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPoolGradV2()
{
    using K = KernelDefinition<
        ops::MaxPoolGradV2,
        DmlKernelWrapper<
            DmlMaxPoolGradKernel,
            GetOutputShapeAsInputShapeHelper>>::
        WithHostMemoryArguments<
            ops::MaxPoolGradV2::Argument::ksize,
            ops::MaxPoolGradV2::Argument::strides>;

    RegisterWithTypes<K, ops::MaxPoolGradV2::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterMaxPool3DGrad()
{
    using K = KernelDefinition<
        ops::MaxPool3DGrad,
        DmlKernelWrapper<
            DmlMaxPoolGradKernel,
            GetOutputShapeAsInputShapeHelper>>;

    constexpr auto T = ops::MaxPool3DGrad::Attribute::T;
    constexpr auto TInput = ops::MaxPool3DGrad::Attribute::TInput;

    K::WithTypeConstraint<T, TF_HALF>::WithTypeConstraint<TInput, TF_HALF>::
        Register();

    K::WithTypeConstraint<T, TF_FLOAT>::WithTypeConstraint<TInput, TF_FLOAT>::
        Register();
}

void RegisterKernels_Pooling()
{
    RegisterAvgPool();
    RegisterAvgPool3D();
    RegisterMaxPool();
    RegisterMaxPool3D();
    RegisterMaxPoolV2();
    RegisterAvgPoolGrad();
    RegisterAvgPool3DGrad();
    RegisterMaxPoolGrad();
    RegisterMaxPoolGradV2();
    RegisterMaxPool3DGrad();
}

} // namespace tfdml
