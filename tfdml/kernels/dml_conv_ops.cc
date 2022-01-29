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
#include "tfdml/runtime_adapter/fused_eigen_output_kernels.h"
#include "tfdml/runtime_adapter/kernel_shape_util.h"
#include "tfdml/runtime_adapter/padding.h"

namespace tfdml
{

// Convolution parameters specified by Op attributes.
struct Conv2DParameters
{
    std::vector<int32_t> dilations;
    std::vector<int32_t> strides;
    Padding padding;
    TensorFormat data_format;
    std::vector<int64_t> explicit_paddings;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DDimensions
{
    int batch;
    int input_rows;
    int input_cols;
    int in_depth;

    int filter_rows;
    int filter_cols;
    int patch_depth;
    int out_depth;

    int stride_rows;
    int stride_cols;

    int dilation_rows;
    int dilation_cols;

    int64_t out_rows;
    int64_t out_cols;
    int64_t pad_rows_before;
    int64_t pad_rows_after;
    int64_t pad_cols_before;
    int64_t pad_cols_after;
};

// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension
{
    int64_t input_size;
    int64_t filter_size;
    int64_t output_size;
    int64_t stride;
    int64_t dilation;

    // Output size after scaling by the stride.
    int64_t expanded_output_size;

    // Number of padding elements to be added before/after this dimension of
    // the input when computing Conv?DBackpropInput.
    int64_t pad_before, pad_after;
};

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions
{
    // Information about each spatial dimension.
    absl::InlinedVector<ConvBackpropSpatialDimension, 3> spatial_dims;

    // Batch size.
    int64_t batch_size;

    // Input and output feature depth.
    int64_t in_depth, out_depth;

    // Convenience access methods for spatial dimensions properties.
    int64_t input_size(int dim) const { return spatial_dims[dim].input_size; }
    int64_t filter_size(int dim) const { return spatial_dims[dim].filter_size; }
    int64_t output_size(int dim) const { return spatial_dims[dim].output_size; }
    int64_t stride(int dim) const { return spatial_dims[dim].stride; }
    int64_t dilation(int dim) const { return spatial_dims[dim].dilation; }

    // Compute padding for the given spatial dimension.
    int SpatialPadding(const Padding& padding, int dim) const
    {
        return (padding == VALID)
                   ? 0
                   : std::max<int>(
                         0,
                         static_cast<int>(
                             (output_size(dim) - 1) * stride(dim) +
                             (filter_size(dim) - 1) * dilation(dim) + 1 -
                             input_size(dim)));
    }
};

class BaseConv2DInitHelper : public InitializationHelper
{
  public:
    virtual TensorFormat GetDataFormat() const = 0;
    virtual int64_t GetBatch() const = 0;
    virtual int64_t GetOutRows() const = 0;
    virtual int64_t GetOutCols() const = 0;
    virtual int64_t GetOutDepth() const = 0;
};

Status InitConv2DParameters(
    const OpKernelConstruction* context,
    Conv2DParameters* params)
{
    TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
    TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
    TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
    if (context->HasAttr("explicit_paddings"))
    {
        TF_RETURN_IF_ERROR(
            context->GetAttr("explicit_paddings", &params->explicit_paddings));
    }
    std::string data_format_string;
    TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));

    if (!FormatFromString(data_format_string, &params->data_format))
    {
        return errors::InvalidArgument("Invalid data format");
    }

    const auto& strides = params->strides;
    const auto& dilations = params->dilations;
    const auto& data_format = params->data_format;

    if (dilations.size() != 4)
    {
        return errors::InvalidArgument("Sliding window dilations field must "
                                       "specify 4 dimensions");
    }

    if (strides.size() != 4)
    {
        return errors::InvalidArgument("Sliding window strides field must "
                                       "specify 4 dimensions");
    }

    const int64_t stride_n = GetTensorDim(strides, data_format, 'N');
    const int64_t stride_c = GetTensorDim(strides, data_format, 'C');
    const int64_t stride_h = GetTensorDim(strides, data_format, 'H');
    const int64_t stride_w = GetTensorDim(strides, data_format, 'W');

    if (stride_n != 1 || stride_c != 1)
    {
        return errors::Unimplemented(
            "Current implementation does not yet support "
            "strides in the batch and depth dimensions.");
    }

    if (stride_h <= 0 || stride_w <= 0)
    {
        return errors::InvalidArgument(
            "Row and column strides should be larger than 0.");
    }

    const int64_t dilation_n = GetTensorDim(dilations, data_format, 'N');
    const int64_t dilation_c = GetTensorDim(dilations, data_format, 'C');
    const int64_t dilation_h = GetTensorDim(dilations, data_format, 'H');
    const int64_t dilation_w = GetTensorDim(dilations, data_format, 'W');

    if (dilation_n != 1 || dilation_c != 1)
    {
        return errors::Unimplemented(
            "Current implementation does not yet support "
            "dilations in the batch and depth dimensions.");
    }

    if (dilation_h <= 0 || dilation_w <= 0)
    {
        return errors::InvalidArgument(
            "Dilated rates should be larger than 0.");
    }

    constexpr int num_dims = 4;

    TF_RETURN_IF_ERROR(CheckValidPadding(
        params->padding,
        params->explicit_paddings,
        num_dims,
        data_format));

    return Status::OK();
}

Status ComputeConv2DDimension(
    const Conv2DParameters& params,
    const TensorShape& input_shape,
    const TensorShape& filter_shape,
    Conv2DDimensions* dimensions)
{
    // Check that 2D convolution input and filter have exactly 4 dimensions.

    if (input_shape.dims() != 4)
    {
        return errors::InvalidArgument(
            "input must be 4-dimensional",
            input_shape.DebugString());
    }

    if (filter_shape.dims() != 4)
    {
        return errors::InvalidArgument(
            "filter must be 4-dimensional: ",
            filter_shape.DebugString());
    }
    for (int i = 0; i < 3; i++)
    {
        if (filter_shape.dim_size(i) >= std::numeric_limits<int>::max())
        {
            return errors::InvalidArgument("filter too large");
        }
    }

    // The last dimension for input is in_depth. Check that it is the same as
    // the filter's in_depth or it is evenly divisible by filter's in_depth.
    const int64_t in_depth_raw =
        GetTensorDim(input_shape, params.data_format, 'C');
    const int64_t patch_depth_raw = filter_shape.dim_size(2);
    if (in_depth_raw >= std::numeric_limits<int>::max())
    {
        return errors::InvalidArgument("Input depth too large");
    }

    if (patch_depth_raw >= std::numeric_limits<int>::max())
    {
        return errors::InvalidArgument("Patch depth too large");
    }
    const int in_depth = static_cast<int>(in_depth_raw);
    const int patch_depth = static_cast<int>(patch_depth_raw);
    if (patch_depth <= 0)
    {
        return errors::InvalidArgument(
            "filter depth must be stricly positive, got ",
            patch_depth);
    }
    if (in_depth % patch_depth != 0)
    {
        return errors::InvalidArgument(
            "input depth must be evenly divisible by filter depth: ",
            in_depth,
            " vs ",
            patch_depth);
    }
    if (filter_shape.num_elements() <= 0)
    {
        return errors::InvalidArgument(
            "filter must not have zero elements "
            "(i.e. all dimensions must be non-zero)");
    }

    const int64_t num_groups = in_depth / patch_depth;
    if (num_groups <= 0)
    {
        return errors::InvalidArgument(
            "number of groups must be stricly positive, got ",
            num_groups);
    }

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter_shape.dim_size(3));

    if (out_depth % num_groups != 0 || out_depth < num_groups)
    {
        return errors::InvalidArgument(
            "output depth must be evenly divisible by number of groups: ",
            out_depth,
            " vs ",
            num_groups);
    }

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64_t input_rows_raw =
        GetTensorDim(input_shape, params.data_format, 'H');
    if (input_rows_raw >= std::numeric_limits<int>::max())
    {
        return errors::InvalidArgument("Input rows too large");
    }
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter_shape.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64_t input_cols_raw =
        GetTensorDim(input_shape, params.data_format, 'W');
    if (input_cols_raw >= std::numeric_limits<int>::max())
    {
        return errors::InvalidArgument("Input cols too large");
    }
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter_shape.dim_size(1));

    // The first dimension for input is batch.
    const int64_t batch_raw =
        GetTensorDim(input_shape, params.data_format, 'N');
    if (batch_raw >= std::numeric_limits<int>::max())
    {
        return errors::InvalidArgument("batch is too large");
    }
    const int batch = static_cast<int>(batch_raw);

    // Take the stride and dilation from the second and third dimensions only
    // (we do not support striding or dilation on the batch or depth dimension).
    const int stride_rows =
        GetTensorDim(params.strides, params.data_format, 'H');
    const int stride_cols =
        GetTensorDim(params.strides, params.data_format, 'W');
    const int dilation_rows =
        GetTensorDim(params.dilations, params.data_format, 'H');
    const int dilation_cols =
        GetTensorDim(params.dilations, params.data_format, 'W');

    int64_t pad_rows_before, pad_rows_after, pad_cols_before, pad_cols_after;
    if (params.padding == Padding::EXPLICIT)
    {
        GetExplicitPaddingForDim(
            params.explicit_paddings,
            params.data_format,
            'H',
            &pad_rows_before,
            &pad_rows_after);
        GetExplicitPaddingForDim(
            params.explicit_paddings,
            params.data_format,
            'W',
            &pad_cols_before,
            &pad_cols_after);
    }

    // Compute windowed output sizes for rows and columns.
    int64_t out_rows = 0, out_cols = 0;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
        input_rows,
        filter_rows,
        dilation_rows,
        stride_rows,
        params.padding,
        &out_rows,
        &pad_rows_before,
        &pad_rows_after));
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
        input_cols,
        filter_cols,
        dilation_cols,
        stride_cols,
        params.padding,
        &out_cols,
        &pad_cols_before,
        &pad_cols_after));

    dimensions->batch = batch;
    dimensions->input_rows = input_rows;
    dimensions->input_cols = input_cols;
    dimensions->in_depth = in_depth;
    dimensions->filter_rows = filter_rows;
    dimensions->filter_cols = filter_cols;
    dimensions->patch_depth = patch_depth;
    dimensions->out_depth = out_depth;
    dimensions->stride_rows = stride_rows;
    dimensions->stride_cols = stride_cols;
    dimensions->dilation_rows = dilation_rows;
    dimensions->dilation_cols = dilation_cols;
    dimensions->out_rows = out_rows;
    dimensions->out_cols = out_cols;
    dimensions->pad_rows_before = pad_rows_before;
    dimensions->pad_rows_after = pad_rows_after;
    dimensions->pad_cols_before = pad_cols_before;
    dimensions->pad_cols_after = pad_cols_after;

    return Status::OK();
}

Status ConvBackpropExtractAndVerifyDimension(
    const char* label,
    const TensorShape& input_shape,
    const TensorShape& filter_shape,
    const TensorShape& output_shape,
    const absl::Span<const int32_t> dilations,
    const std::vector<int32_t>& strides,
    Padding padding,
    int64_t padding_before,
    int64_t padding_after,
    int spatial_dim,
    int filter_spatial_dim,
    ConvBackpropSpatialDimension* dim)
{
    dim->input_size = input_shape.dim_size(spatial_dim);
    dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
    dim->output_size = output_shape.dim_size(spatial_dim);
    dim->stride = strides[spatial_dim];
    dim->dilation = dilations[spatial_dim];
    int64_t out_size = 0;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
        dim->input_size,
        dim->filter_size,
        dim->dilation,
        dim->stride,
        padding,
        &out_size,
        &padding_before,
        &padding_after));
    if (dim->output_size != out_size)
    {
        return errors::InvalidArgument(
            label,
            ": Size of out_backprop doesn't match computed: ",
            "actual = ",
            dim->output_size,
            ", computed = ",
            out_size,
            " spatial_dim: ",
            spatial_dim,
            " input: ",
            dim->input_size,
            " filter: ",
            dim->filter_size,
            " output: ",
            dim->output_size,
            " stride: ",
            dim->stride,
            " dilation: ",
            dim->dilation);
    }

    int64_t effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
    dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
    const auto padded_out_size = dim->input_size + effective_filter_size - 1;
    dim->pad_before = effective_filter_size - 1 - padding_before;
    dim->pad_after =
        padded_out_size - dim->expanded_output_size - dim->pad_before;
    TF_VLog(
        2,
        "%s: expanded_out = %lld, effective_filter_size = %lld, padded_out = "
        "%lld, pad_before = %lld, pad_after = %lld, dilation = %lld, strides = "
        "%lld",
        label,
        dim->expanded_output_size,
        effective_filter_size,
        padded_out_size,
        dim->pad_before,
        dim->pad_after,
        dim->dilation,
        dim->stride);
    return Status::OK();
}

Status ConvBackpropComputeDimensionsV2(
    const char* label,
    int num_spatial_dims,
    const TensorShape& input_shape,
    const TensorShape& filter_shape,
    const TensorShape& out_backprop_shape,
    const absl::Span<const int32_t>& dilations,
    const std::vector<int32_t>& strides,
    Padding padding,
    absl::Span<const int64_t> explicit_paddings,
    TensorFormat data_format,
    ConvBackpropDimensions* dims)
{
    // The + 2 in the following line is for the batch and feature dimensions.
    const int num_dims = num_spatial_dims + 2;
    if (input_shape.dims() != num_dims)
    {
        return errors::InvalidArgument(
            label,
            ": input must be ",
            num_dims,
            "-dimensional");
    }
    if (filter_shape.dims() != num_dims)
    {
        return errors::InvalidArgument(
            label,
            ": filter must be ",
            num_dims,
            "-dimensional");
    }
    if (out_backprop_shape.dims() != num_dims)
    {
        return errors::InvalidArgument(
            label,
            ": out_backprop must be ",
            num_dims,
            "-dimensional");
    }
    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
    dims->batch_size = input_shape.dim_size(batch_dim);
    if (dims->batch_size != out_backprop_shape.dim_size(batch_dim))
    {
        return errors::InvalidArgument(
            label,
            ": input and out_backprop must have the same batch size.",
            " Input batch: ",
            dims->batch_size,
            ", outbackprop batch: ",
            out_backprop_shape.dim_size(batch_dim),
            ", batch_dim: ",
            batch_dim);
    }

    int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
    dims->in_depth = input_shape.dim_size(feature_dim);
    // The input and output feature dimensions are the second last and last
    // dimensions of the filter Tensor.
    TF_VLog(
        2,
        "input vs filter_in depth %d %lld",
        dims->in_depth,
        filter_shape.dim_size(num_dims - 2));
    if (filter_shape.dim_size(num_dims - 2) <= 0)
    {
        return errors ::InvalidArgument(
            label,
            ": filter depth must be strictly greated than zero");
    }
    if (dims->in_depth % filter_shape.dim_size(num_dims - 2))
    {
        return errors::InvalidArgument(
            label,
            ": input depth must be evenly divisible by filter depth");
    }
    dims->out_depth = filter_shape.dim_size(num_dims - 1);
    if (dims->out_depth != out_backprop_shape.dim_size(feature_dim))
    {
        return errors::InvalidArgument(
            label,
            ": filter and out_backprop must have the same out_depth");
    }
    dims->spatial_dims.resize(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; ++i)
    {
        int image_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
        int64_t padding_before = -1, padding_after = -1;
        if (padding == EXPLICIT)
        {
            padding_before = explicit_paddings[2 * image_dim];
            padding_after = explicit_paddings[2 * image_dim + 1];
        }
        TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
            label,
            input_shape,
            filter_shape,
            out_backprop_shape,
            dilations,
            strides,
            padding,
            padding_before,
            padding_after,
            image_dim,
            i,
            &dims->spatial_dims[i]));
    }
    return Status::OK();
}

class DepthwiseConv2DNativeInitHelper : public BaseConv2DInitHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::vector<int32_t> strides;
            std::vector<int32_t> dilations;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations));
            std::string data_format_attr;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_attr));
            OP_REQUIRES(
                ctx,
                FormatFromString(data_format_attr, &data_format),
                errors::InvalidArgument("Invalid data format"));

            OP_REQUIRES(
                ctx,
                strides.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

            OP_REQUIRES(
                ctx,
                dilations.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));

            stride_h = GetTensorDim(strides, data_format, 'H');
            stride_w = GetTensorDim(strides, data_format, 'W');
            const int64_t stride_n = GetTensorDim(strides, data_format, 'N');
            const int64_t stride_c = GetTensorDim(strides, data_format, 'C');

            OP_REQUIRES(
                ctx,
                (stride_n == 1 && stride_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "strides in the batch and depth dimensions."));

            dilation_h = GetTensorDim(dilations, data_format, 'H');
            dilation_w = GetTensorDim(dilations, data_format, 'W');
            const int64_t dilation_n = GetTensorDim(strides, data_format, 'N');
            const int64_t dilation_c = GetTensorDim(strides, data_format, 'C');

            OP_REQUIRES(
                ctx,
                (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "strides in the batch and depth dimensions."));

            OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));

            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("explicit_paddings", &explicit_paddings));

            constexpr int num_dims = 4;
            OP_REQUIRES_OK(
                ctx,
                CheckValidPadding(
                    padding,
                    explicit_paddings,
                    num_dims,
                    data_format));
        }

        TensorFormat data_format;
        Padding padding;
        int32_t stride_h;
        int32_t stride_w;
        int32_t dilation_h;
        int32_t dilation_w;
        std::vector<int64_t> explicit_paddings;
    };

    DepthwiseConv2DNativeInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        // Input tensor is of the following dimensions:
        // [ batch, in_rows, in_cols, in_depth ]
        const Tensor input = ctx->input(0);

        // Input filter is of the following dimensions:
        // [ filter_rows, filter_cols, in_depth, depth_multiplier]
        const Tensor filter = ctx->input(1);

        // For 2D convolution, there should be 4 dimensions.
        OP_REQUIRES(
            ctx,
            input.dims() == 4,
            errors::InvalidArgument(
                "input must be 4-dimensional",
                input.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            filter.dims() == 4,
            errors::InvalidArgument(
                "filter must be 4-dimensional: ",
                filter.shape().DebugString()));

        // in_depth for input and filter must match.
        in_depth_ = GetTensorDim(input, attr_->data_format, 'C');
        OP_REQUIRES(
            ctx,
            in_depth_ == filter.dim_size(2),
            errors::InvalidArgument(
                "input and filter must have the same depth: ",
                in_depth_,
                " vs ",
                filter.dim_size(2)));

        group_count_ = filter.dim_size(2);

        // The last dimension for filter is depth multiplier.
        int32_t depth_multiplier = filter.dim_size(3);

        // The output depth is input depth x depth multipler
        out_depth_ = in_depth_ * depth_multiplier;

        const int64_t input_rows_raw =
            GetTensorDim(input, attr->data_format, 'H');
        OP_REQUIRES(
            ctx,
            input_rows_raw < std::numeric_limits<int32_t>::max(),
            errors::InvalidArgument("Input rows too large"));
        const int32_t input_rows = static_cast<int32_t>(input_rows_raw);
        filter_rows_ = filter.dim_size(0);

        const int64_t input_cols_raw =
            GetTensorDim(input, attr->data_format, 'W');
        OP_REQUIRES(
            ctx,
            input_cols_raw < std::numeric_limits<int32_t>::max(),
            errors::InvalidArgument("Input cols too large"));
        const int32_t input_cols = static_cast<int32_t>(input_cols_raw);
        filter_cols_ = filter.dim_size(1);

        // The first dimension for input is batch.
        batch_ = input.dim_size(0);

        if (attr_->padding == Padding::EXPLICIT)
        {
            GetExplicitPaddingForDim(
                attr_->explicit_paddings,
                attr_->data_format,
                'H',
                &pad_rows_before_,
                &pad_rows_after_);
            GetExplicitPaddingForDim(
                attr_->explicit_paddings,
                attr_->data_format,
                'W',
                &pad_cols_before_,
                &pad_cols_after_);
        }

        OP_REQUIRES_OK(
            ctx,
            GetWindowedOutputSizeVerboseV2(
                input_rows,
                filter_rows_,
                attr->dilation_h,
                attr_->stride_h,
                attr->padding,
                &out_rows_,
                &pad_rows_before_,
                &pad_rows_after_));

        OP_REQUIRES_OK(
            ctx,
            GetWindowedOutputSizeVerboseV2(
                input_cols,
                filter_cols_,
                attr_->dilation_w,
                attr_->stride_w,
                attr->padding,
                &out_cols_,
                &pad_cols_before_,
                &pad_cols_after_));
    }

    TensorFormat GetDataFormat() const final { return attr_->data_format; }
    int64_t GetBatch() const final { return batch_; }
    int64_t GetOutRows() const final { return out_rows_; }
    int64_t GetOutCols() const final { return out_cols_; }
    int64_t GetInDepth() const { return in_depth_; }
    int64_t GetOutDepth() const final { return out_depth_; }
    int32_t GetStrideH() const { return attr_->stride_h; }
    int32_t GetStrideW() const { return attr_->stride_w; }
    int32_t GetDilationH() const { return attr_->dilation_h; }
    int32_t GetDilationW() const { return attr_->dilation_w; }
    int32_t GetFilterRows() const { return filter_rows_; }
    int32_t GetFilterCols() const { return filter_cols_; }
    int32_t GetGroupCount() const { return group_count_; }
    int64_t GetPadRowsBefore() const { return pad_rows_before_; }
    int64_t GetPadColsBefore() const { return pad_cols_before_; }
    int64_t GetPadRowsAfter() const { return pad_rows_after_; }
    int64_t GetPadColsAfter() const { return pad_cols_after_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    int32_t filter_rows_;
    int32_t filter_cols_;
    int64_t batch_;
    int64_t out_rows_;
    int64_t out_cols_;
    int64_t in_depth_;
    int64_t out_depth_;
    int32_t group_count_;
    int64_t pad_rows_before_;
    int64_t pad_cols_before_;
    int64_t pad_rows_after_;
    int64_t pad_cols_after_;
};

class ConvInitHelper : public BaseConv2DInitHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params));
        }

        Conv2DParameters params;
    };

    ConvInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        const Tensor input = ctx->input(0);
        const Tensor filter = ctx->input(1);

        OP_REQUIRES_OK(
            ctx,
            ComputeConv2DDimension(
                attr->params,
                input.shape(),
                filter.shape(),
                &dimensions_));
    }

    const Conv2DParameters& GetParams() const { return attr_->params; }
    TensorFormat GetDataFormat() const final
    {
        return attr_->params.data_format;
    }
    int64_t GetBatch() const final { return dimensions_.batch; }
    int64_t GetOutRows() const final { return dimensions_.out_rows; }
    int64_t GetOutCols() const final { return dimensions_.out_cols; }
    int64_t GetOutDepth() const final { return dimensions_.out_depth; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    Conv2DDimensions dimensions_;
};

class FusedConvInitHelper : public ConvInitHelper
{
  public:
    struct Attributes : public ConvInitHelper::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : ConvInitHelper::Attributes(ctx)
        {
            using FCT = FusedComputationType;
            std::vector<FusedComputationPattern> patterns = {
                {FCT::kBiasAdd, {"BiasAdd"}},
                {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
                {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
                {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
                {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
                {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
                {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
                {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
            };

            OP_REQUIRES_OK(
                ctx,
                InitializeFusedComputation(
                    ctx,
                    "DmlFusedConv2d",
                    patterns,
                    &fused_computation_type,
                    &fused_computation_args));
        }

        FusedComputationType fused_computation_type;
        FusedComputationArgs fused_computation_args;
    };

    FusedConvInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : ConvInitHelper(ctx, attr),
          attr_(attr)
    {
    }

    FusedComputationType GetFusedComputationType() const
    {
        return attr_->fused_computation_type;
    }

    FusedComputationArgs GetFusedComputationArgs() const
    {
        return attr_->fused_computation_args;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class Conv2DGradInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params));
        }

        Conv2DParameters params;
    };

    Conv2DGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
    }

    const Conv2DParameters& GetParams() const { return attr_->params; }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class ConvShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const BaseConv2DInitHelper*>(initialization_helper);

        TensorShape out_shape = ShapeFromFormat(
            init_helper->GetDataFormat(),
            init_helper->GetBatch(),
            init_helper->GetOutRows(),
            init_helper->GetOutCols(),
            init_helper->GetOutDepth());

        return {std::move(out_shape)};
    }
};

class DmlDepthwiseConv2DNativeKernel : public DmlKernel
{
  public:
    using InitHelper = DepthwiseConv2DNativeInitHelper;

    explicit DmlDepthwiseConv2DNativeKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() == 2);
        assert(ctx->GetOutputCount() == 1);
        assert(ctx->GetInputTensorShape(0).dims() == kNchwDimensionCount);
        assert(ctx->GetInputTensorShape(1).dims() == kNchwDimensionCount);
        assert(ctx->GetOutputTensorShape(0).dims() == kNchwDimensionCount);

        uint32_t strides[] = {
            static_cast<uint32_t>(init_helper->GetStrideH()),
            static_cast<uint32_t>(init_helper->GetStrideW())};
        uint32_t dilations[] = {
            static_cast<uint32_t>(init_helper->GetDilationH()),
            static_cast<uint32_t>(init_helper->GetDilationW())};
        uint32_t start_padding[] = {
            static_cast<uint32_t>(init_helper->GetPadRowsBefore()),
            static_cast<uint32_t>(init_helper->GetPadColsBefore())};
        uint32_t end_padding[] = {
            static_cast<uint32_t>(init_helper->GetPadRowsAfter()),
            static_cast<uint32_t>(init_helper->GetPadColsAfter())};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count =
            static_cast<uint32_t>(init_helper->GetGroupCount());

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects
        auto filter_layout = {H, W, C, N};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout = GetDmlTensorLayout(
            init_helper->GetDataFormat(),
            kNchwDimensionCount);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);

        TensorShape filter_shape = {
            init_helper->GetFilterRows(),
            init_helper->GetFilterCols(),
            init_helper->GetInDepth() / group_count,
            init_helper->GetOutDepth()};

        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            filter_shape,
            filter_shape,
            filter_layout);

        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kNchwSpatialDimensionCount;
        conv_desc.Strides = strides;
        conv_desc.Dilations = dilations;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = group_count;
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlConv2DKernel : public DmlKernel
{
  public:
    using InitHelper = ConvInitHelper;

    explicit DmlConv2DKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        // 2D conv requires 4D tensors
        static const uint32_t kDimensionCount = 4;
        static const uint32_t kSpatialDimensionCount = 2;

        CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
        CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
        CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

        const Conv2DParameters& conv_params = init_helper->GetParams();

        Conv2DDimensions conv_dims;
        TF_CHECK_OK(ComputeConv2DDimension(
            conv_params,
            ctx->GetInputTensorShape(0),
            ctx->GetInputTensorShape(1),
            &conv_dims));

        uint32_t strides[] = {
            static_cast<uint32_t>(conv_dims.stride_rows),
            static_cast<uint32_t>(conv_dims.stride_cols)};
        uint32_t dilations[] = {
            static_cast<uint32_t>(conv_dims.dilation_rows),
            static_cast<uint32_t>(conv_dims.dilation_cols)};
        uint32_t start_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_before),
            static_cast<uint32_t>(conv_dims.pad_cols_before)};
        uint32_t end_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_after),
            static_cast<uint32_t>(conv_dims.pad_cols_after)};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count = conv_dims.in_depth / conv_dims.patch_depth;

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects
        auto filter_layout = {H, W, C, N};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 1, filter_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        conv_desc.Strides = strides;
        conv_desc.Dilations = dilations;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = group_count;
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <typename T>
class DmlFusedConv2DKernel : public DmlKernel
{
  public:
    using InitHelper = FusedConvInitHelper;

    explicit DmlFusedConv2DKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() > 2);
        CHECK(ctx->GetOutputCount() == 1);

        // 2D conv requires 4D tensors
        static const uint32_t kDimensionCount = 4;

        CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
        CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
        CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

        const Conv2DParameters& conv_params = init_helper->GetParams();
        const auto fused_computation_type =
            init_helper->GetFusedComputationType();
        const auto fused_computation_args =
            init_helper->GetFusedComputationArgs();

        Conv2DDimensions conv_dims;
        TF_CHECK_OK(ComputeConv2DDimension(
            conv_params,
            ctx->GetInputTensorShape(0),
            ctx->GetInputTensorShape(1),
            &conv_dims));

        uint32_t strides[] = {
            static_cast<uint32_t>(conv_dims.stride_rows),
            static_cast<uint32_t>(conv_dims.stride_cols)};
        uint32_t dilations[] = {
            static_cast<uint32_t>(conv_dims.dilation_rows),
            static_cast<uint32_t>(conv_dims.dilation_cols)};
        uint32_t start_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_before),
            static_cast<uint32_t>(conv_dims.pad_cols_before)};
        uint32_t end_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_after),
            static_cast<uint32_t>(conv_dims.pad_cols_after)};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count =
            static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

        DmlKernelParams params;

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects
        auto filter_layout = {H, W, C, N};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 1, filter_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        auto scope = dml::Graph(
            ctx->GetDmlDevice(),
            GetDmlXTensorPolicy(conv_params.data_format));
        auto input = dml::InputTensor(scope, 0, input_descs[0]);
        auto filter = dml::InputTensor(scope, 1, input_descs[1]);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op;

        if (BiasAddArgs<T>::IsSupported(fused_computation_type))
        {
            CHECK(ctx->GetInputCount() == 3);

            const TensorShape& bias_tensor_shape = ctx->GetInputTensorShape(2);
            // bias must be 1-dimensional
            CHECK(bias_tensor_shape.dims() == 1);
            uint32_t bias_size = bias_tensor_shape.dim_size(0);

            // dml expects bias to be 4d tensor
            dml::TensorDesc::Dimensions bias_sizes{1, bias_size, 1, 1};
            dml::TensorDesc::Dimensions bias_strides{bias_size, 1, 0, 0};
            auto bias = dml::Reinterpret(
                dml::InputTensor(scope, 2, input_descs[2]),
                bias_sizes,
                bias_strides);
            switch (fused_computation_type)
            {
            case FusedComputationType::kBiasAdd: {
                auto conv2d = dml::Convolution(
                    input,
                    filter,
                    bias,
                    DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                    DML_CONVOLUTION_DIRECTION_FORWARD,
                    strides,
                    dilations,
                    start_padding,
                    end_padding,
                    output_padding,
                    group_count);
                compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
            }
            break;
            case FusedComputationType::kBiasAddWithRelu: {
                auto conv2d = dml::Convolution(
                    input,
                    filter,
                    bias,
                    DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                    DML_CONVOLUTION_DIRECTION_FORWARD,
                    strides,
                    dilations,
                    start_padding,
                    end_padding,
                    output_padding,
                    group_count,
                    dml::FusedActivation::Relu());
                compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
            }
            break;
            case FusedComputationType::kBiasAddWithRelu6: {
                auto conv2d = dml::Convolution(
                    input,
                    filter,
                    bias,
                    DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                    DML_CONVOLUTION_DIRECTION_FORWARD,
                    strides,
                    dilations,
                    start_padding,
                    end_padding,
                    output_padding,
                    group_count);
                auto relu6 = dml::ActivationRelu6(conv2d);
                compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {relu6});
            }
            break;
            case FusedComputationType::kBiasAddWithElu: {
                auto conv2d = dml::Convolution(
                    input,
                    filter,
                    bias,
                    DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                    DML_CONVOLUTION_DIRECTION_FORWARD,
                    strides,
                    dilations,
                    start_padding,
                    end_padding,
                    output_padding,
                    group_count,
                    dml::FusedActivation::Elu(1.0f));
                compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
            }
            break;
            default: CHECK(false);
            }
        }
        else if (FusedBatchNormArgs<T>::IsSupported(fused_computation_type))
        {
            CHECK(ctx->GetInputCount() == 6);
            const TensorShape& scale_tensor_shape = ctx->GetInputTensorShape(2);
            const TensorShape& offset_tensor_shape =
                ctx->GetInputTensorShape(3);
            const TensorShape& mean_tensor_shape = ctx->GetInputTensorShape(4);
            const TensorShape& variance_tensor_shape =
                ctx->GetInputTensorShape(5);

            // all arguments must be 1-dimensional
            CHECK(scale_tensor_shape.dims() == 1);
            CHECK(offset_tensor_shape.dims() == 1);
            CHECK(mean_tensor_shape.dims() == 1);
            CHECK(variance_tensor_shape.dims() == 1);

            input_descs[2] =
                CreateTensorDescFromInput(ctx, 2, {C}).GetDmlDesc();
            input_descs[3] =
                CreateTensorDescFromInput(ctx, 3, {C}).GetDmlDesc();
            input_descs[4] =
                CreateTensorDescFromInput(ctx, 4, {C}).GetDmlDesc();
            input_descs[5] =
                CreateTensorDescFromInput(ctx, 5, {C}).GetDmlDesc();

            auto scale = dml::InputTensor(scope, 2, input_descs[2]);
            auto offset = dml::InputTensor(scope, 3, input_descs[3]);
            auto mean = dml::InputTensor(scope, 4, input_descs[4]);
            auto variance = dml::InputTensor(scope, 5, input_descs[5]);

            auto conv2d = dml::Convolution(
                input,
                filter,
                absl::nullopt,
                DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                DML_CONVOLUTION_DIRECTION_FORWARD,
                strides,
                dilations,
                start_padding,
                end_padding,
                output_padding,
                group_count);
            switch (fused_computation_type)
            {
            case FusedComputationType::kFusedBatchNorm: {
                auto batch_norm = dml::BatchNormalization(
                    conv2d,
                    mean,
                    variance,
                    scale,
                    offset,
                    true,
                    fused_computation_args.epsilon);
                compiled_op =
                    scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
            }
            break;
            case FusedComputationType::kFusedBatchNormWithRelu: {
                auto batch_norm = dml::BatchNormalization(
                    conv2d,
                    mean,
                    variance,
                    scale,
                    offset,
                    true,
                    fused_computation_args.epsilon,
                    dml::FusedActivation::Relu());
                compiled_op =
                    scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
            }
            break;
            case FusedComputationType::kFusedBatchNormWithRelu6: {
                auto batch_norm = dml::BatchNormalization(
                    conv2d,
                    mean,
                    variance,
                    scale,
                    offset,
                    true,
                    fused_computation_args.epsilon);
                auto relu6 = dml::ActivationRelu6(batch_norm);
                compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {relu6});
            }
            break;
            case FusedComputationType::kFusedBatchNormWithElu: {
                auto batch_norm = dml::BatchNormalization(
                    conv2d,
                    mean,
                    variance,
                    scale,
                    offset,
                    true,
                    fused_computation_args.epsilon,
                    dml::FusedActivation::Elu(1.0f));
                compiled_op =
                    scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
            }
            break;
            default: CHECK(false);
            }
        }

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlConv2DBackpropInputKernel : public DmlKernel
{
  public:
    using InitHelper = Conv2DGradInitHelper;

    explicit DmlConv2DBackpropInputKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1);

        // 2D conv requires 4D tensors
        static const uint32_t kDimensionCount = 4;
        static const uint32_t kSpatialDimensionCount = 2;

        // Tensor 0 is a 1-d vector of input shapes
        CHECK(ctx->GetInputTensorShape(0).dims() == 1);
        CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
        CHECK(ctx->GetInputTensorShape(2).dims() == kDimensionCount);
        CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

        const auto& input_sizes = ctx->GetConstantInputTensor(0);
        TensorShape input_shape = TensorShapeUtils::MakeShape(input_sizes);

        DmlKernelParams params;
        params.kernel_input_indices = {
            2,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        const Conv2DParameters& conv_params = init_helper->GetParams();

        Conv2DDimensions conv_dims;
        TF_CHECK_OK(ComputeConv2DDimension(
            conv_params,
            input_shape,
            ctx->GetInputTensorShape(1),
            &conv_dims));

        uint32_t strides[] = {
            static_cast<uint32_t>(conv_dims.stride_rows),
            static_cast<uint32_t>(conv_dims.stride_cols)};
        uint32_t dilations[] = {
            static_cast<uint32_t>(conv_dims.dilation_rows),
            static_cast<uint32_t>(conv_dims.dilation_cols)};
        uint32_t start_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_before),
            static_cast<uint32_t>(conv_dims.pad_cols_before)};
        uint32_t end_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_after),
            static_cast<uint32_t>(conv_dims.pad_cols_after)};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count =
            static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects.
        auto filter_layout = {H, W, C, N};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 2, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 1, filter_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];

        // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
        // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
        // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION
        // here.
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        conv_desc.Strides = strides;
        conv_desc.Dilations = dilations;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = group_count;
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlConv2DBackpropFilterKernel : public DmlKernel
{
  public:
    using InitHelper = Conv2DGradInitHelper;

    explicit DmlConv2DBackpropFilterKernel(
        DmlKernelConstruction* ctx,
        const Conv2DGradInitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1);

        // 2D conv requires 4D tensors
        static const uint32_t kDimensionCount = 4;
        static const uint32_t kSpatialDimensionCount = 2;

        CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
        CHECK(ctx->GetInputTensorShape(1).dims() == 1);
        CHECK(ctx->GetInputTensorShape(2).dims() == kDimensionCount);
        CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

        const auto& filter_sizes = ctx->GetConstantInputTensor(1);
        TensorShape filter_shape = TensorShapeUtils::MakeShape(filter_sizes);

        const Conv2DParameters& conv_params = init_helper->GetParams();

        Conv2DDimensions conv_dims;
        TF_CHECK_OK(ComputeConv2DDimension(
            conv_params,
            ctx->GetInputTensorShape(0),
            filter_shape,
            &conv_dims));

        uint32_t strides[] = {
            static_cast<uint32_t>(conv_dims.stride_rows),
            static_cast<uint32_t>(conv_dims.stride_cols)};
        uint32_t dilations[] = {
            static_cast<uint32_t>(conv_dims.dilation_rows),
            static_cast<uint32_t>(conv_dims.dilation_cols)};
        uint32_t start_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_before),
            static_cast<uint32_t>(conv_dims.pad_cols_before)};
        uint32_t end_padding[] = {
            static_cast<uint32_t>(conv_dims.pad_rows_after),
            static_cast<uint32_t>(conv_dims.pad_cols_after)};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count =
            static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            2,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects. Note order of C and N
        // are reversed as we convolve with the backprop_output
        // which has N channels, where N is the number of output
        // feature maps in the forward conv2d case.
        auto filter_layout = {H, W, N, C};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

        // Swap N and C channels. Note: Channel 'N' of output
        // contains the 'K' feature maps produced in the forward
        // direction. Swapping is required because we
        // convolve the incoming backprop_output gradient containing K
        //  features with the 'K' channelled filter
        DmlTensorAxis axis;
        switch (conv_params.data_format)
        {
        case FORMAT_NHWC:
            axis = input_output_layout[0];
            input_output_layout[0] = input_output_layout[3];
            input_output_layout[3] = axis;
            break;

        case FORMAT_NCHW:
            axis = input_output_layout[0];
            input_output_layout[0] = input_output_layout[1];
            input_output_layout[1] = axis;
            break;

        case FORMAT_NCHW_VECT_C:
            LogFatal("FORMAT_NCHW_VECT_C is not supported for DML devices.");
            break;

        case FORMAT_NHWC_VECT_W:
            LogFatal("FORMAT_NHWC_VECT_W is not supported for DML devices.");
            break;

        case FORMAT_HWNC:
            LogFatal("FORMAT_HWNC is not supported for DML devices.");
            break;

        case FORMAT_HWCN:
            LogFatal("FORMAT_HWCN is not supported for DML devices.");
            break;
        }

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 2, input_output_layout);

        // The output tensor gets the filter_layout as we are computing the
        // back-prop for the filter.
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, filter_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        conv_desc.Strides = dilations;
        conv_desc.Dilations = strides;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = group_count;
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

// Depthwise Conv2D has two input tensors (input and filter) so it has two
// gradient ops as well. The initialization logic for both gradient ops is
// the same. Each gradient op receives only one of the original input
// tensors from the forward pass (the "non-backprop tensor"); the gradient op
// receives a 1D host-memory shape tensor for the tensor whose gradients the op
// is computing (the "backprop tensor").
//
// - DepthwiseConv2dNativeBackpropInput : computes gradients w.r.t. input
//   inputs = [input_sizes, filter, out_gradients]
//   BackpropTensorIndex = 0 (input)
//   NonBackpropTensorIndex = 1 (filter)
//
// - DepthwiseConv2dNativeBackpropFilter : computes gradients w.r.t. filter
//   inputs = [input, filter_sizes, out_gradients]
//   BackpropTensorIndex = 1 (filter)
//   NonBackpropTensorIndex = 0 (input)
template <int BackpropTensorIndex>
class DepthwiseConv2DBackpropInitHelper : public InitializationHelper
{
  public:
    using Attributes = DepthwiseConv2DNativeInitHelper::Attributes;

    DepthwiseConv2DBackpropInitHelper(
        OpKernelContext* context,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        constexpr int NonBackpropTensorIndex = !BackpropTensorIndex;

        const char* label = nullptr;
        const char* backprop_sizes_name = nullptr;
        if constexpr (BackpropTensorIndex == 0)
        {
            label = "Conv2DBackpropInput";
            backprop_sizes_name = "input_sizes";
        }
        else
        {
            label = "Conv2DBackpropFilter";
            backprop_sizes_name = "filter_sizes";
        }

        const Tensor backprop_tensor_shape_tensor =
            context->input(BackpropTensorIndex);
        OP_REQUIRES(
            context,
            TensorShapeUtils::IsVector(backprop_tensor_shape_tensor.shape()),
            errors::InvalidArgument(
                label,
                ":",
                backprop_sizes_name,
                " input must be 1-dim, not ",
                backprop_tensor_shape_tensor.dims()));
        TensorShape backprop_tensor_shape;
        const int32_t* backprop_tensor_shape_data =
            reinterpret_cast<const int32_t*>(
                backprop_tensor_shape_tensor.tensor_data().data());
        for (int i = 0; i < backprop_tensor_shape_tensor.NumElements(); ++i)
        {
            OP_REQUIRES(
                context,
                backprop_tensor_shape_data[i] >= 0,
                errors::InvalidArgument(
                    "Dimension ",
                    i,
                    " of ",
                    backprop_sizes_name,
                    " must be >= 0"));
            backprop_tensor_shape.AddDim(backprop_tensor_shape_data[i]);
        }

        const Tensor non_backprop_tensor =
            context->input(NonBackpropTensorIndex);
        const TensorShape& non_backprop_tensor_shape =
            non_backprop_tensor.shape();

        const TensorShape& input_shape = (BackpropTensorIndex == 0)
                                             ? backprop_tensor_shape
                                             : non_backprop_tensor_shape;
        const TensorShape& filter_shape = (BackpropTensorIndex == 0)
                                              ? non_backprop_tensor_shape
                                              : backprop_tensor_shape;

        const Tensor out_backprop = context->input(2);

        OP_REQUIRES(
            context,
            input_shape.dims() == 4,
            errors::InvalidArgument(label, ": input must be 4-dimensional"));
        OP_REQUIRES(
            context,
            filter_shape.dims() == 4,
            errors::InvalidArgument(label, ": filter must be 4-dimensional"));
        OP_REQUIRES(
            context,
            out_backprop.dims() == 4,
            errors::InvalidArgument(
                label,
                ": out_backprop must be 4-dimensional"));

        const int64_t batch_size_raw = input_shape.dim_size(0);
        OP_REQUIRES(
            context,
            batch_size_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Batch size too large"));
        OP_REQUIRES(
            context,
            batch_size_raw == out_backprop.dim_size(0),
            errors::InvalidArgument(
                label,
                ": input and out_backprop must have the same batch size"));
        batch_size_ = static_cast<uint32_t>(batch_size_raw);

        const int64_t input_depth_raw =
            GetTensorDim(input_shape, attr_->data_format, 'C');
        OP_REQUIRES(
            context,
            input_depth_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Input depth too large"));
        in_channels_ = static_cast<uint32_t>(input_depth_raw);

        const int64_t input_rows_raw =
            GetTensorDim(input_shape, attr_->data_format, 'H');
        OP_REQUIRES(
            context,
            input_rows_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Input rows too large"));
        in_height_ = static_cast<uint32_t>(input_rows_raw);

        const int64_t input_cols_raw =
            GetTensorDim(input_shape, attr_->data_format, 'W');
        OP_REQUIRES(
            context,
            input_cols_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Input cols too large"));
        in_width_ = static_cast<uint32_t>(input_cols_raw);

        const int64_t filter_rows_raw = filter_shape.dim_size(0);
        OP_REQUIRES(
            context,
            filter_rows_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Filter rows too large"));
        filter_height_ = static_cast<uint32_t>(filter_rows_raw);

        const int64_t filter_cols_raw = filter_shape.dim_size(1);
        OP_REQUIRES(
            context,
            filter_cols_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Filter cols too large"));
        filter_width_ = static_cast<uint32_t>(filter_cols_raw);

        OP_REQUIRES(
            context,
            in_channels_ == filter_shape.dim_size(2),
            errors::InvalidArgument(
                label,
                ": input and filter must have the same in_depth"));

        const int64_t depth_multiplier = filter_shape.dim_size(3);

        const int64_t out_channels_raw =
            GetTensorDim(out_backprop.shape(), attr_->data_format, 'C');
        OP_REQUIRES(
            context,
            out_channels_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Output depth too large"));
        OP_REQUIRES(
            context,
            (depth_multiplier * in_channels_) == out_channels_raw,
            errors::InvalidArgument(
                label,
                ": depth_multiplier * in_depth not equal to out_depth"));
        out_channels_ = static_cast<uint32_t>(out_channels_raw);

        const int64_t output_rows_raw =
            GetTensorDim(out_backprop.shape(), attr_->data_format, 'H');
        OP_REQUIRES(
            context,
            output_rows_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Output rows too large"));
        out_height_ = static_cast<uint32_t>(output_rows_raw);

        const int64_t output_cols_raw =
            GetTensorDim(out_backprop.shape(), attr_->data_format, 'W');
        OP_REQUIRES(
            context,
            output_cols_raw < std::numeric_limits<uint32_t>::max(),
            errors::InvalidArgument("Output cols too large"));
        out_width_ = static_cast<uint32_t>(output_cols_raw);

        int64_t out_height_calculated;
        int64_t pad_rows_before_raw;
        int64_t pad_rows_after_raw;

        if (attr_->padding == Padding::EXPLICIT)
        {
            GetExplicitPaddingForDim(
                attr_->explicit_paddings,
                attr_->data_format,
                'H',
                &pad_rows_before_raw,
                &pad_rows_after_raw);
        }

        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSizeVerboseV2(
                in_height_,
                filter_height_,
                attr_->dilation_h,
                attr_->stride_h,
                attr_->padding,
                &out_height_calculated,
                &pad_rows_before_raw,
                &pad_rows_after_raw));
        OP_REQUIRES(
            context,
            out_height_calculated == out_height_,
            errors::InvalidArgument(
                label,
                ": Number of rows of out_backprop doesn't match computed: ",
                "actual = ",
                out_height_,
                ", computed = ",
                out_height_calculated));
        pad_rows_before_ = static_cast<uint32_t>(pad_rows_before_raw);
        pad_rows_after_ = static_cast<uint32_t>(pad_rows_after_raw);

        int64_t out_width_calculated;
        int64_t pad_cols_before_raw;
        int64_t pad_cols_after_raw;

        if (attr_->padding == Padding::EXPLICIT)
        {
            GetExplicitPaddingForDim(
                attr_->explicit_paddings,
                attr_->data_format,
                'W',
                &pad_cols_before_raw,
                &pad_cols_after_raw);
        }

        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSizeVerboseV2(
                in_width_,
                filter_width_,
                attr_->dilation_w,
                attr_->stride_w,
                attr_->padding,
                &out_width_calculated,
                &pad_cols_before_raw,
                &pad_cols_after_raw));
        OP_REQUIRES(
            context,
            out_width_calculated == out_width_,
            errors::InvalidArgument(
                label,
                ": Number of cols of out_backprop doesn't match computed: ",
                "actual = ",
                out_width_,
                ", computed = ",
                out_width_calculated));
        pad_cols_before_ = static_cast<uint32_t>(pad_cols_before_raw);
        pad_cols_after_ = static_cast<uint32_t>(pad_cols_after_raw);
    }

    TensorFormat GetDataFormat() const { return attr_->data_format; }

    uint32_t GetBatchSize() const { return batch_size_; }
    uint32_t GetInChannels() const { return in_channels_; }
    uint32_t GetInHeight() const { return in_height_; }
    uint32_t GetInWidth() const { return in_width_; }
    uint32_t GetFilterHeight() const { return filter_height_; }
    uint32_t GetFilterWidth() const { return filter_width_; }
    uint32_t GetOutChannels() const { return out_channels_; }
    uint32_t GetOutHeight() const { return out_height_; }
    uint32_t GetOutWidth() const { return out_width_; }
    uint32_t GetStrideH() const { return attr_->stride_h; }
    uint32_t GetStrideW() const { return attr_->stride_w; }
    uint32_t GetDilationH() const { return attr_->dilation_h; }
    uint32_t GetDilationW() const { return attr_->dilation_w; }
    uint32_t GetGroupCount() const { return in_channels_; }
    uint32_t GetPadRowsBefore() const { return pad_rows_before_; }
    uint32_t GetPadColsBefore() const { return pad_cols_before_; }
    uint32_t GetPadRowsAfter() const { return pad_rows_after_; }
    uint32_t GetPadColsAfter() const { return pad_cols_after_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    uint32_t batch_size_;
    uint32_t in_channels_;
    uint32_t in_height_;
    uint32_t in_width_;
    int32_t filter_channels_;
    int32_t filter_height_;
    int32_t filter_width_;
    uint32_t out_channels_;
    uint32_t out_height_;
    uint32_t out_width_;
    uint32_t pad_rows_before_;
    uint32_t pad_cols_before_;
    uint32_t pad_rows_after_;
    uint32_t pad_cols_after_;
};

using DepthwiseConv2DBackpropInputInitHelper =
    DepthwiseConv2DBackpropInitHelper<0>;

class DmlDepthwiseConv2DBackpropFilterKernel : public DmlKernel
{
  public:
    using InitHelper = DepthwiseConv2DBackpropInitHelper<1>;

    explicit DmlDepthwiseConv2DBackpropFilterKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() == 3);
        assert(ctx->GetOutputCount() == 1);

        uint32_t strides[] = {
            1,
            init_helper->GetStrideH(),
            init_helper->GetStrideW()};
        uint32_t dilations[] = {
            1,
            init_helper->GetDilationH(),
            init_helper->GetDilationW()};
        uint32_t start_padding[] = {
            0,
            init_helper->GetPadRowsBefore(),
            init_helper->GetPadColsBefore()};
        uint32_t end_padding[] = {
            0,
            init_helper->GetPadRowsAfter(),
            init_helper->GetPadColsAfter()};
        uint32_t output_padding[] = {0, 0, 0};

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            2,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        // The depthwise 2D conv filter backprop calculation can be expressed as
        // a convolution, just like standard 2D convolution, but it requires an
        // extra dimension to correctly reinterpret and transpose dimensions
        // without copying to intermediate tensors.
        DmlTensorLayout input_layout;
        DmlTensorLayout output_backprop_layout;
        TensorShape input_shape;
        TensorShape output_backprop_shape;
        switch (init_helper->GetDataFormat())
        {
        case FORMAT_NCHW:
            input_layout = {N, D, C, H, W};
            input_shape = {
                1,                            // N
                init_helper->GetBatchSize(),  // D
                init_helper->GetInChannels(), // C
                init_helper->GetInHeight(),   // H
                init_helper->GetInWidth(),    // W
            };
            output_backprop_layout = {D, N, C, H, W};
            output_backprop_shape = {
                init_helper->GetBatchSize(),   // D
                init_helper->GetOutChannels(), // N
                1,                             // C
                init_helper->GetOutHeight(),   // H
                init_helper->GetOutWidth(),    // W
            };
            break;
        case FORMAT_NHWC:
            input_layout = {N, D, H, W, C};
            input_shape = {
                1,                            // N
                init_helper->GetBatchSize(),  // D
                init_helper->GetInHeight(),   // H
                init_helper->GetInWidth(),    // W
                init_helper->GetInChannels(), // C
            };
            output_backprop_layout = {D, C, H, W, N};
            output_backprop_shape = {
                init_helper->GetBatchSize(),   // D
                1,                             // C
                init_helper->GetOutHeight(),   // H
                init_helper->GetOutWidth(),    // W
                init_helper->GetOutChannels(), // N
            };
            break;
        case FORMAT_NCHW_VECT_C:
            LogFatal("FORMAT_NCHW_VECT_C is not supported for DML devices.");
            break;
        case FORMAT_NHWC_VECT_W:
            LogFatal("FORMAT_NHWC_VECT_W is not supported for DML devices.");
            break;
        case FORMAT_HWNC:
            LogFatal("FORMAT_HWNC is not supported for DML devices.");
            break;
        case FORMAT_HWCN:
            LogFatal("FORMAT_HWCN is not supported for DML devices.");
            break;
        }

        auto filter_backprop_layout = {N, H, W, C, D};
        TensorShape filter_backprop_shape = {
            1,                              // N
            init_helper->GetFilterHeight(), // H
            init_helper->GetFilterWidth(),  // W
            init_helper->GetOutChannels(),  // C
            1                               // D
        };

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_shape,
            input_shape,
            input_layout);

        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            output_backprop_shape,
            output_backprop_shape,
            output_backprop_layout);

        tensors.outputs[0]->desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            filter_backprop_shape,
            filter_backprop_shape,
            filter_backprop_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kNcdhwSpatialDimensionCount;
        conv_desc.Strides = strides;
        conv_desc.Dilations = dilations;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = init_helper->GetGroupCount();
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlDepthwiseConv2DBackpropInputKernel : public DmlKernel
{
  public:
    using InitHelper = DepthwiseConv2DBackpropInitHelper<0>;

    explicit DmlDepthwiseConv2DBackpropInputKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() == 3);
        assert(ctx->GetOutputCount() == 1);

        uint32_t strides[] = {
            init_helper->GetStrideH(),
            init_helper->GetStrideW()};
        uint32_t dilations[] = {
            init_helper->GetDilationH(),
            init_helper->GetDilationW()};
        uint32_t start_padding[] = {
            init_helper->GetPadRowsBefore(),
            init_helper->GetPadColsBefore()};
        uint32_t end_padding[] = {
            init_helper->GetPadRowsAfter(),
            init_helper->GetPadColsAfter()};
        uint32_t output_padding[] = {0, 0};
        uint32_t group_count = init_helper->GetGroupCount();

        DmlKernelParams params;
        params.kernel_input_indices = {
            2,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        auto output_layout = GetDmlTensorLayout(
            init_helper->GetDataFormat(),
            kNchwDimensionCount);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 2, output_layout);

        auto filter_layout = {H, W, C, N};
        TensorShape filter_shape = {
            init_helper->GetFilterHeight(),
            init_helper->GetFilterWidth(),
            init_helper->GetInChannels() / group_count,
            init_helper->GetOutChannels()};

        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            filter_shape,
            filter_shape,
            filter_layout);

        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
        // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
        // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION
        // here.
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
        conv_desc.DimensionCount = kNchwSpatialDimensionCount;
        conv_desc.Strides = strides;
        conv_desc.Dilations = dilations;
        conv_desc.StartPadding = start_padding;
        conv_desc.EndPadding = end_padding;
        conv_desc.OutputPadding = output_padding;
        conv_desc.GroupCount = group_count;
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <bool HasDataFormatAttribute>
struct Conv3DAttributes
{
    explicit Conv3DAttributes(OpKernelConstruction* context)
    {
        if constexpr (HasDataFormatAttribute)
        {
            std::string data_format_str;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("data_format", &data_format_str));
            OP_REQUIRES(
                context,
                FormatFromString(data_format_str, &data_format),
                errors::InvalidArgument("Invalid data format"));
        }
        else
        {
            // V2 conv3d grad ops have a data format attribute but the original
            // ops assume NHWC format.
            data_format = FORMAT_NHWC;
        }

        std::vector<int32_t> strides;
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
        OP_REQUIRES(
            context,
            strides.size() == 5,
            errors::InvalidArgument("Sliding window strides field must "
                                    "specify 5 dimensions"));

        int32_t stride_n = GetTensorDim(strides, data_format, 'N');
        int32_t stride_c = GetTensorDim(strides, data_format, 'C');
        stride_d = GetTensorDim(strides, data_format, '0');
        stride_h = GetTensorDim(strides, data_format, '1');
        stride_w = GetTensorDim(strides, data_format, '2');

        OP_REQUIRES(
            context,
            (stride_n == 1 && stride_c == 1),
            errors::InvalidArgument(
                "Current implementation does not yet support "
                "strides in the batch and depth dimensions."));
        OP_REQUIRES(
            context,
            (stride_d > 0 && stride_h > 0 && stride_w > 0),
            errors::InvalidArgument(
                "Spatial strides should be larger than 0."));

        std::vector<int32_t> dilations;
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
        OP_REQUIRES(
            context,
            dilations.size() == 5,
            errors::InvalidArgument("Dilation rates field must "
                                    "specify 5 dimensions"));

        int32_t dilation_n = GetTensorDim(dilations, data_format, 'N');
        int32_t dilation_c = GetTensorDim(dilations, data_format, 'C');
        dilation_d = GetTensorDim(dilations, data_format, '0');
        dilation_h = GetTensorDim(dilations, data_format, '1');
        dilation_w = GetTensorDim(dilations, data_format, '2');

        OP_REQUIRES(
            context,
            (dilation_n == 1 && dilation_c == 1),
            errors::InvalidArgument(
                "Current implementation does not yet support "
                "dilation rates in the batch and depth dimensions."));
        OP_REQUIRES(
            context,
            (dilation_d > 0 && dilation_h > 0 && dilation_w > 0),
            errors::InvalidArgument("Dilated rates should be larger than 0."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    }

    TensorFormat data_format;
    Padding padding;
    int32_t stride_d;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_d;
    int32_t dilation_h;
    int32_t dilation_w;
};

class Conv3DInitHelper : public InitializationHelper
{
  public:
    using Attributes = Conv3DAttributes<true>;

    Conv3DInitHelper(
        OpKernelContext* context,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        // Input tensor is of the following dimensions:
        // [ batch, in_z, in_y, in_x, in_channels ]
        const Tensor input = context->input(0);

        // Input filter is of the following dimensions:
        // [ filter_z, filter_y, filter_x, in_channels, out_channels]
        const Tensor filter = context->input(1);

        // NOTE: The ordering of the spatial dimensions is arbitrary, but has to
        // be kept consistent between input/filter/output.
        OP_REQUIRES(
            context,
            input.dims() == 5,
            errors::InvalidArgument("input must be 5-dimensional"));
        OP_REQUIRES(
            context,
            filter.dims() == 5,
            errors::InvalidArgument("filter must be 5-dimensional"));

        const int64_t in_batch = GetTensorDim(input, attr->data_format, 'N');
        const int64_t in_channels = GetTensorDim(input, attr->data_format, 'C');
        const int64_t in_depth = GetTensorDim(input, attr->data_format, '0');
        const int64_t in_height = GetTensorDim(input, attr->data_format, '1');
        const int64_t in_width = GetTensorDim(input, attr->data_format, '2');

        const int64_t filter_depth = filter.dim_size(0);
        const int64_t filter_height = filter.dim_size(1);
        const int64_t filter_width = filter.dim_size(2);
        const int64_t filter_channels = filter.dim_size(3);
        const int64_t out_channels = filter.dim_size(4);

        OP_REQUIRES(
            context,
            filter_depth != 0,
            errors::InvalidArgument("filter_depth must be non-zero"));

        OP_REQUIRES(
            context,
            in_channels % filter_channels == 0,
            errors::InvalidArgument(
                "Input depth must be evenly divisible by filter depth: ",
                in_channels,
                " vs ",
                filter_channels));

        OP_REQUIRES(
            context,
            filter.NumElements() > 0,
            errors::InvalidArgument("filter must not have zero elements "
                                    "(i.e. all dimensions must be non-zero)"));

        // Dimension order for these arrays is: z, y, x.
        std::array<int64_t, 3> input_size = {{in_depth, in_height, in_width}};
        std::array<int64_t, 3> filter_size = {
            {filter_depth, filter_height, filter_width}};
        std::array<int64_t, 3> dilations = {
            {attr->dilation_d, attr->dilation_h, attr->dilation_w}};
        std::array<int64_t, 3> strides = {
            {attr->stride_d, attr->stride_h, attr->stride_w}};

        std::array<int64_t, 3> out, padding;

        OP_REQUIRES_OK(
            context,
            Get3dOutputSizeV2(
                input_size,
                filter_size,
                dilations,
                strides,
                attr->padding,
                &out,
                &padding));

        batch_size_ = static_cast<uint32_t>(in_batch);
        in_channels_ = static_cast<uint32_t>(in_channels);
        in_depth_ = static_cast<uint32_t>(in_depth);
        in_height_ = static_cast<uint32_t>(in_height);
        in_width_ = static_cast<uint32_t>(in_width);
        filter_channels_ = static_cast<uint32_t>(filter_channels);
        filter_depth_ = static_cast<uint32_t>(filter_depth);
        filter_height_ = static_cast<uint32_t>(filter_height);
        filter_width_ = static_cast<uint32_t>(filter_width);
        out_channels_ = static_cast<uint32_t>(out_channels);
        out_depth_ = static_cast<uint32_t>(out[0]);
        out_height_ = static_cast<uint32_t>(out[1]);
        out_width_ = static_cast<uint32_t>(out[2]);
        strides_[0] = attr->stride_d;
        strides_[1] = attr->stride_h;
        strides_[2] = attr->stride_w;
        dilations_[0] = attr->dilation_d;
        dilations_[1] = attr->dilation_h;
        dilations_[2] = attr->dilation_w;
        start_padding_[0] = static_cast<uint32_t>(padding[0]);
        start_padding_[1] = static_cast<uint32_t>(padding[1]);
        start_padding_[2] = static_cast<uint32_t>(padding[2]);
    }

    TensorFormat GetDataFormat() const { return attr_->data_format; }

    uint32_t GetBatchSize() const { return batch_size_; }
    uint32_t GetInChannels() const { return in_channels_; }
    uint32_t GetInDepth() const { return in_depth_; }
    uint32_t GetInHeight() const { return in_height_; }
    uint32_t GetInWidth() const { return in_width_; }
    uint32_t GetFilterChannels() const { return filter_channels_; }
    uint32_t GetFilterDepth() const { return filter_depth_; }
    uint32_t GetFilterHeight() const { return filter_height_; }
    uint32_t GetFilterWidth() const { return filter_width_; }
    uint32_t GetOutChannels() const { return out_channels_; }
    uint32_t GetOutDepth() const { return out_depth_; }
    uint32_t GetOutHeight() const { return out_height_; }
    uint32_t GetOutWidth() const { return out_width_; }
    const std::array<uint32_t, 3>& GetStrides() const { return strides_; }
    const std::array<uint32_t, 3>& GetDilations() const { return dilations_; }
    const std::array<uint32_t, 3>& GetStartPadding() const
    {
        return start_padding_;
    }
    const std::array<uint32_t, 3>& GetEndPadding() const
    {
        return end_padding_;
    }
    const std::array<uint32_t, 3>& GetOutPadding() const
    {
        return out_padding_;
    }
    uint32_t GetGroupCount() const { return in_channels_ / filter_channels_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    uint32_t batch_size_;
    uint32_t in_channels_;
    uint32_t in_depth_;
    uint32_t in_height_;
    uint32_t in_width_;
    uint32_t filter_channels_;
    uint32_t filter_depth_;
    uint32_t filter_height_;
    uint32_t filter_width_;
    uint32_t out_channels_;
    uint32_t out_depth_;
    uint32_t out_height_;
    uint32_t out_width_;
    std::array<uint32_t, 3> strides_;
    std::array<uint32_t, 3> dilations_;
    std::array<uint32_t, 3> start_padding_;
    std::array<uint32_t, 3> end_padding_ = {0, 0, 0};
    std::array<uint32_t, 3> out_padding_ = {0, 0, 0};
};

class Conv3DShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const Conv3DInitHelper*>(initialization_helper);

        TensorShape out_shape = ShapeFromFormat(
            init_helper->GetDataFormat(),
            init_helper->GetBatchSize(),
            {init_helper->GetOutDepth(),
             init_helper->GetOutHeight(),
             init_helper->GetOutWidth()},
            init_helper->GetOutChannels());

        return {std::move(out_shape)};
    }
};

class DmlConv3DKernel : public DmlKernel
{
  public:
    using InitHelper = Conv3DInitHelper;

    explicit DmlConv3DKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        // 3D conv requires 5D tensors
        static const uint32_t kDimensionCount = 5;
        static const uint32_t kSpatialDimensionCount = 3;

        CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
        CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
        CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        auto filter_layout = {D, H, W, C, N};

        auto input_output_layout =
            GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 1, filter_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        conv_desc.Strides = init_helper->GetStrides().data();
        conv_desc.Dilations = init_helper->GetDilations().data();
        conv_desc.StartPadding = init_helper->GetStartPadding().data();
        conv_desc.EndPadding = init_helper->GetEndPadding().data();
        conv_desc.OutputPadding = init_helper->GetOutPadding().data();
        conv_desc.GroupCount = init_helper->GetGroupCount();
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <bool HasDataFormatAttribute, bool BackpropInput>
class Conv3DGradInitHelper : public InitializationHelper
{
  public:
    using Attributes = Conv3DAttributes<HasDataFormatAttribute>;

    Conv3DGradInitHelper(
        OpKernelContext* context,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        std::string label;
        TensorShape input_shape;
        TensorShape filter_shape;
        if constexpr (BackpropInput)
        {
            label = "Conv3DBackpropInputOp";
            const Tensor input_sizes = context->input(0);
            OP_REQUIRES_OK(
                context,
                TensorShapeUtils::MakeShape(input_sizes, &input_shape));
            filter_shape = context->input(1).shape();
        }
        else
        {
            label = "Conv3DBackpropFilterOp";
            input_shape = context->input(0).shape();
            const Tensor filter_sizes = context->input(1);
            OP_REQUIRES_OK(
                context,
                TensorShapeUtils::MakeShape(filter_sizes, &filter_shape));
        }

        const Tensor out_backprop = context->input(2);
        const TensorShape& out_backprop_shape = out_backprop.shape();

        std::vector<int32_t> strides;
        std::vector<int32_t> dilations;
        if (attr->data_format == FORMAT_NCHW)
        {
            strides = {1, 1, attr->stride_d, attr->stride_h, attr->stride_w};
            dilations =
                {1, 1, attr->dilation_d, attr->dilation_h, attr->dilation_w};
        }
        else
        {
            strides = {1, attr->stride_d, attr->stride_h, attr->stride_w, 1};
            dilations =
                {1, attr->dilation_d, attr->dilation_h, attr->dilation_w, 1};
        }

        ConvBackpropDimensions dims;
        OP_REQUIRES_OK(
            context,
            ConvBackpropComputeDimensionsV2(
                label.c_str(),
                /*num_spatial_dims=*/3,
                input_shape,
                filter_shape,
                out_backprop_shape,
                dilations,
                strides,
                attr->padding,
                {},
                attr->data_format,
                &dims));

        uint32_t pad_d =
            static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 0));
        uint32_t pad_h =
            static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 1));
        uint32_t pad_w =
            static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 2));

        batch_size_ = static_cast<uint32_t>(dims.batch_size);
        in_channels_ = static_cast<uint32_t>(dims.in_depth);
        in_depth_ = static_cast<uint32_t>(dims.input_size(0));
        in_height_ = static_cast<uint32_t>(dims.input_size(1));
        in_width_ = static_cast<uint32_t>(dims.input_size(2));
        filter_channels_ = static_cast<uint32_t>(filter_shape.dim_size(3));
        filter_depth_ = static_cast<uint32_t>(dims.filter_size(0));
        filter_height_ = static_cast<uint32_t>(dims.filter_size(1));
        filter_width_ = static_cast<uint32_t>(dims.filter_size(2));
        out_channels_ = static_cast<uint32_t>(dims.out_depth);
        out_depth_ = static_cast<uint32_t>(dims.output_size(0));
        out_height_ = static_cast<uint32_t>(dims.output_size(1));
        out_width_ = static_cast<uint32_t>(dims.output_size(2));
        strides_[0] = attr->stride_d;
        strides_[1] = attr->stride_h;
        strides_[2] = attr->stride_w;
        dilations_[0] = attr->dilation_d;
        dilations_[1] = attr->dilation_h;
        dilations_[2] = attr->dilation_w;
        start_padding_[0] = pad_d / 2;
        start_padding_[1] = pad_h / 2;
        start_padding_[2] = pad_w / 2;
        end_padding_[0] = pad_d / 2 + pad_d % 2;
        end_padding_[1] = pad_h / 2 + pad_h % 2;
        end_padding_[2] = pad_w / 2 + pad_w % 2;
    }

    TensorFormat GetDataFormat() const { return attr_->data_format; }

    uint32_t GetBatchSize() const { return batch_size_; }
    uint32_t GetInChannels() const { return in_channels_; }
    uint32_t GetInDepth() const { return in_depth_; }
    uint32_t GetInHeight() const { return in_height_; }
    uint32_t GetInWidth() const { return in_width_; }
    uint32_t GetFilterChannels() const { return filter_channels_; }
    uint32_t GetFilterDepth() const { return filter_depth_; }
    uint32_t GetFilterHeight() const { return filter_height_; }
    uint32_t GetFilterWidth() const { return filter_width_; }
    uint32_t GetOutChannels() const { return out_channels_; }
    uint32_t GetOutDepth() const { return out_depth_; }
    uint32_t GetOutHeight() const { return out_height_; }
    uint32_t GetOutWidth() const { return out_width_; }
    const std::array<uint32_t, 3>& GetStrides() const { return strides_; }
    const std::array<uint32_t, 3>& GetDilations() const { return dilations_; }
    const std::array<uint32_t, 3>& GetStartPadding() const
    {
        return start_padding_;
    }
    const std::array<uint32_t, 3>& GetEndPadding() const
    {
        return end_padding_;
    }
    const std::array<uint32_t, 3>& GetOutPadding() const
    {
        return out_padding_;
    }
    uint32_t GetGroupCount() const { return in_channels_ / filter_channels_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    uint32_t batch_size_;
    uint32_t in_channels_;
    uint32_t in_depth_;
    uint32_t in_height_;
    uint32_t in_width_;
    uint32_t filter_channels_;
    uint32_t filter_depth_;
    uint32_t filter_height_;
    uint32_t filter_width_;
    uint32_t out_channels_;
    uint32_t out_depth_;
    uint32_t out_height_;
    uint32_t out_width_;
    std::array<uint32_t, 3> strides_;
    std::array<uint32_t, 3> dilations_;
    std::array<uint32_t, 3> start_padding_;
    std::array<uint32_t, 3> end_padding_;
    std::array<uint32_t, 3> out_padding_ = {0, 0, 0};
};

template <bool HasDataFormatAttribute>
class DmlConv3DBackpropInputKernel : public DmlKernel
{
  public:
    using InitHelper = Conv3DGradInitHelper<HasDataFormatAttribute, true>;

    explicit DmlConv3DBackpropInputKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1);

        // 3D conv requires 5D tensors
        static const uint32_t kDimensionCount = 5;
        static const uint32_t kSpatialDimensionCount = 3;

        const auto& input_sizes = ctx->GetConstantInputTensor(0);
        TensorShape input_shape = TensorShapeUtils::MakeShape(input_sizes);

        DmlKernelParams params;
        params.kernel_input_indices = {
            2,
            1,
            absl::nullopt // We don't use the DML bias tensor
        };

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects.
        auto filter_layout = {D, H, W, C, N};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 2, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 1, filter_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];

        // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
        // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
        // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION
        // here.
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        conv_desc.Strides = init_helper->GetStrides().data();
        conv_desc.Dilations = init_helper->GetDilations().data();
        conv_desc.StartPadding = init_helper->GetStartPadding().data();
        conv_desc.EndPadding = init_helper->GetEndPadding().data();
        conv_desc.OutputPadding = init_helper->GetOutPadding().data();
        conv_desc.GroupCount = init_helper->GetGroupCount();
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <bool HasDataFormatAttribute>
class DmlConv3DBackpropFilterKernel : public DmlKernel
{
  public:
    using InitHelper = Conv3DGradInitHelper<HasDataFormatAttribute, false>;

    explicit DmlConv3DBackpropFilterKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1);

        // 2D conv requires 4D tensors
        static const uint32_t kDimensionCount = 5;
        static const uint32_t kSpatialDimensionCount = 3;

        const auto& filter_sizes = ctx->GetConstantInputTensor(1);
        TensorShape filter_shape = TensorShapeUtils::MakeShape(filter_sizes);

        DmlKernelParams params;
        params.kernel_input_indices = {
            0,
            2,
            absl::nullopt // We don't use the DML bias tensor
        };

        using namespace DmlTensorAxes;

        // The dimensions of the filter tensor are laid out a little differently
        // than what DML expects. Note order of C and N
        // are reversed as we convolve with the backprop_output
        // which has N channels, where N is the number of output
        // feature maps in the forward conv2d case.
        auto filter_layout = {D, H, W, N, C};

        // The layout of the input/output tensors is determined by the
        // "data_format" attribute
        auto input_output_layout =
            GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

        // Swap N and C channels. Note: Channel 'N' of output
        // contains the 'K' feature maps produced in the forward
        // direction. Swapping is required because we
        // convolve the incoming backprop_output gradient containing K
        //  features with the 'K' channelled filter
        switch (init_helper->GetDataFormat())
        {
        case FORMAT_NHWC:
            std::swap(input_output_layout[0], input_output_layout[4]);
            break;

        case FORMAT_NCHW:
            std::swap(input_output_layout[0], input_output_layout[1]);
            break;

        case FORMAT_NCHW_VECT_C:
            LogFatal("FORMAT_NCHW_VECT_C is not supported for DML devices.");
            break;

        case FORMAT_NHWC_VECT_W:
            LogFatal("FORMAT_NHWC_VECT_W is not supported for DML devices.");
            break;

        case FORMAT_HWNC:
            LogFatal("FORMAT_HWNC is not supported for DML devices.");
            break;

        case FORMAT_HWCN:
            LogFatal("FORMAT_HWCN is not supported for DML devices.");
            break;
        }

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.inputs[1]->desc =
            CreateTensorDescFromInput(ctx, 2, input_output_layout);

        // The output tensor gets the filter_layout as we are computing the
        // back-prop for the filter.
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, filter_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
        conv_desc.InputTensor = &input_descs[0];
        conv_desc.FilterTensor = &input_descs[1];
        conv_desc.BiasTensor = nullptr;
        conv_desc.OutputTensor = &output_descs[0];
        conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        conv_desc.DimensionCount = kSpatialDimensionCount;
        // Not a typo: strides and dilations are swapped when computing filter
        // backprop:
        conv_desc.Strides = init_helper->GetDilations().data();
        conv_desc.Dilations = init_helper->GetStrides().data();
        conv_desc.StartPadding = init_helper->GetStartPadding().data();
        conv_desc.EndPadding = init_helper->GetEndPadding().data();
        conv_desc.OutputPadding = init_helper->GetOutPadding().data();
        conv_desc.GroupCount = init_helper->GetGroupCount();
        conv_desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

void RegisterDepthwiseConv2dNative()
{
    using K = KernelDefinition<
        ops::DepthwiseConv2dNative,
        DmlKernelWrapper<DmlDepthwiseConv2DNativeKernel, ConvShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::DepthwiseConv2dNative::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv2D()
{
    using K = KernelDefinition<
        ops::Conv2D,
        DmlKernelWrapper<DmlConv2DKernel, ConvShapeHelper>>;

    RegisterWithTypes<K, ops::Conv2D::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterFusedConv2D()
{
    // FusedConv2D only supports float32
    using K = KernelDefinition<
        ops::_FusedConv2D,
        DmlKernelWrapper<DmlFusedConv2DKernel<float>, ConvShapeHelper>>;

    RegisterWithTypes<K, ops::_FusedConv2D::Attribute::T, TF_FLOAT>();
}

void RegisterConv2DBackpropInput()
{
    using K = KernelDefinition<
        ops::Conv2DBackpropInput,
        DmlKernelWrapper<
            DmlConv2DBackpropInputKernel,
            GetOutputShapeFromDimsTensorHelper<int32_t, 0>>>::
        WithHostMemoryArguments<
            ops::Conv2DBackpropInput::Argument::input_sizes>;

    RegisterWithTypes<
        K,
        ops::Conv2DBackpropInput::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv2DBackpropFilter()
{
    using K = KernelDefinition<
        ops::Conv2DBackpropFilter,
        DmlKernelWrapper<
            DmlConv2DBackpropFilterKernel,
            GetOutputShapeFromDimsTensorHelper<int32_t, 1>>>::
        WithHostMemoryArguments<
            ops::Conv2DBackpropFilter::Argument::filter_sizes>;

    RegisterWithTypes<
        K,
        ops::Conv2DBackpropFilter::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterDepthwiseConv2dNativeBackpropFilter()
{
    using K = KernelDefinition<
        ops::DepthwiseConv2dNativeBackpropFilter,
        DmlKernelWrapper<
            DmlDepthwiseConv2DBackpropFilterKernel,
            GetOutputShapeFromDimsTensorHelper<int32_t, 1>>>::
        WithHostMemoryArguments<
            ops::DepthwiseConv2dNativeBackpropFilter::Argument::filter_sizes>;

    RegisterWithTypes<
        K,
        ops::DepthwiseConv2dNativeBackpropFilter::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterDepthwiseConv2dNativeBackpropInput()
{
    using K = KernelDefinition<
        ops::DepthwiseConv2dNativeBackpropInput,
        DmlKernelWrapper<
            DmlDepthwiseConv2DBackpropInputKernel,
            GetOutputShapeFromDimsTensorHelper<int32_t, 0>>>::
        WithHostMemoryArguments<
            ops::DepthwiseConv2dNativeBackpropInput::Argument::input_sizes>;

    RegisterWithTypes<
        K,
        ops::DepthwiseConv2dNativeBackpropInput::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv3D()
{
    using K = KernelDefinition<
        ops::Conv3D,
        DmlKernelWrapper<DmlConv3DKernel, Conv3DShapeHelper>>;

    RegisterWithTypes<K, ops::Conv3D::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterConv3DBackpropInput()
{
    using K = KernelDefinition<
        ops::Conv3DBackpropInput,
        DmlKernelWrapper<
            DmlConv3DBackpropInputKernel<false>,
            GetOutputShapeFromInputShapeHelper<0>>>;

    RegisterWithTypes<
        K,
        ops::Conv3DBackpropInput::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv3DBackpropInputV2()
{
    using K = KernelDefinition<
        ops::Conv3DBackpropInputV2,
        DmlKernelWrapper<
            DmlConv3DBackpropInputKernel<true>,
            GetOutputShapeFromDimsTensorHelper<int32_t, 0>>>::
        WithHostMemoryArguments<
            ops::Conv3DBackpropInputV2::Argument::input_sizes>;

    RegisterWithTypes<
        K,
        ops::Conv3DBackpropInputV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv3DBackpropFilter()
{
    using K = KernelDefinition<
        ops::Conv3DBackpropFilter,
        DmlKernelWrapper<
            DmlConv3DBackpropFilterKernel<false>,
            GetOutputShapeFromInputShapeHelper<1>>>;

    RegisterWithTypes<
        K,
        ops::Conv3DBackpropFilter::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterConv3DBackpropFilterV2()
{
    using K = KernelDefinition<
        ops::Conv3DBackpropFilterV2,
        DmlKernelWrapper<
            DmlConv3DBackpropFilterKernel<true>,
            GetOutputShapeFromDimsTensorHelper<int32_t, 1>>>::
        WithHostMemoryArguments<
            ops::Conv3DBackpropFilterV2::Argument::filter_sizes>;

    RegisterWithTypes<
        K,
        ops::Conv3DBackpropFilterV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_Conv()
{
    RegisterDepthwiseConv2dNative();
    RegisterConv2D();
    RegisterFusedConv2D();
    RegisterConv2DBackpropInput();
    RegisterConv2DBackpropFilter();
    RegisterDepthwiseConv2dNativeBackpropFilter();
    RegisterDepthwiseConv2dNativeBackpropInput();
    RegisterConv3D();
    RegisterConv3DBackpropInput();
    RegisterConv3DBackpropInputV2();
    RegisterConv3DBackpropFilter();
    RegisterConv3DBackpropFilterV2();
}

} // namespace tfdml