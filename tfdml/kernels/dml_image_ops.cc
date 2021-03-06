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

class ColorConversionInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    ColorConversionInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const TensorShape input_shape = ctx->input(0).shape();

        OP_REQUIRES(
            ctx,
            input_shape.dims() >= 1,
            errors::InvalidArgument(
                "input must be at least 1D",
                input_shape.DebugString()));

        num_channels_ = input_shape.dim_size(input_shape.dims() - 1);
        OP_REQUIRES(
            ctx,
            num_channels_ == 3,
            errors::FailedPrecondition(
                "input must have 3 channels but input only has ",
                num_channels_,
                " channels."));

        // Each pixel in an RGB/HSV color conversion is independent; the tensor
        // shape does not matter, since the channel is the last dimension.
        // Collapse all leading dimensions into a single pixel count so DML can
        // handle arbitrarily shaped tensors.
        num_pixels_ = input_shape.num_elements() /
                      input_shape.dim_size(input_shape.dims() - 1);
    }

    int64_t GetNumChannels() const { return num_channels_; }
    int64_t GetNumPixels() const { return num_pixels_; }

  private:
    int64_t num_channels_;
    int64_t num_pixels_;
};

template <bool needs_three_channels_input>
class AdjustImageInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    AdjustImageInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const TensorShape input_shape = ctx->input(0).shape();

        OP_REQUIRES(
            ctx,
            input_shape.dims() >= 3,
            errors::InvalidArgument(
                "input must be at least 3-D, got shape",
                input_shape.DebugString()));

        const TensorShape adjustment_shape = ctx->input(1).shape();
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(adjustment_shape),
            errors::InvalidArgument(
                "second input must be scalar: ",
                adjustment_shape.DebugString()));

        if (ctx->num_inputs() == 4)
        {
            const Tensor& min_value = ctx->input(2);
            const Tensor& max_value = ctx->input(3);

            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(min_value.shape()),
                errors::InvalidArgument(
                    "min_value must be scalar: ",
                    min_value.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(max_value.shape()),
                errors::InvalidArgument(
                    "max_value must be scalar: ",
                    max_value.shape().DebugString()));
        }

        height_ = input_shape.dim_size(input_shape.dims() - 3);
        width_ = input_shape.dim_size(input_shape.dims() - 2);
        channels_ = input_shape.dim_size(input_shape.dims() - 1);

        if (needs_three_channels_input)
        {
            OP_REQUIRES(
                ctx,
                channels_ == 3,
                errors::InvalidArgument(
                    "input must have 3 channels but instead has ",
                    channels_,
                    " channels."));
        }
    }

    int64_t GetWidth() const { return width_; }
    int64_t GetHeight() const { return height_; }
    int64_t GetChannels() const { return channels_; }

  private:
    int64_t height_;
    int64_t width_;
    int64_t channels_;
};

template <typename T>
std::vector<dml::Expression> RGBToHSVPlanes(
    dml::Graph& scope,
    dml::Expression input)
{
    // This helper can only be called with an NHWC tensor as input.
    auto& inputSizes = input.GetOutputDesc().sizes;
    assert(inputSizes.size() == 4);

    const uint32_t img_n = inputSizes[0];
    const uint32_t img_h = inputSizes[1];
    const uint32_t img_w = inputSizes[2];
    const uint32_t img_c = inputSizes[3];
    uint32_t channel_dim = 3;

    dml::TensorDesc::Dimensions img_flat_size =
        {1, 1, 1, img_n * img_h * img_w * img_c};
    dml::TensorDesc::Dimensions plane_flat_size =
        {1, 1, 1, img_n * img_h * img_w};

    const uint32_t window_size_and_strides[2] = {1, 3};
    auto max_pool_out = dml::MaxPoolingBuilder(input, window_size_and_strides)
                            .Strides(window_size_and_strides)
                            .OutputIndices(true)
                            .Build();

    auto& c_max = max_pool_out.values;
    auto& c_max_indices = max_pool_out.indices;

    auto c_min = dml::Reduce(input, DML_REDUCE_FUNCTION_MIN, {3});
    auto delta = c_max - c_min;

    auto six = dml::ScalarTensor<T>(
        scope,
        TfTensorTypeTraits<T>::FromFloat(6.0f),
        delta.GetOutputDesc().sizes);
    auto zero = dml::ScalarTensor<T>(
        scope,
        TfTensorTypeTraits<T>::FromFloat(0.0f),
        delta.GetOutputDesc().sizes);

    auto c_planes = dml::Split(input, channel_dim, {1, 1, 1});
    auto& r = c_planes[0];
    auto& g = c_planes[1];
    auto& b = c_planes[2];

    constexpr auto sixty_degrees_pct = 60.0f / 360.0f;
    auto h_r = sixty_degrees_pct *
               dml::ModulusFloor((g - b) / delta, six);     // hue if R is max
    auto h_g = sixty_degrees_pct * (((b - r) / delta) + 2); // hue if G is max
    auto h_b = sixty_degrees_pct * (((r - g) / delta) + 4); // hue if B is max
    auto h_rgb = dml::Join({h_r, h_g, h_b}, channel_dim);

    auto h_rgb_flat = dml::Reinterpret(h_rgb, img_flat_size, dml::NullOpt);
    auto c_max_indices_flat =
        dml::Reinterpret(c_max_indices, plane_flat_size, dml::NullOpt);
    auto h_max = dml::Gather(h_rgb_flat, c_max_indices_flat, channel_dim, 4);
    auto h_max_resized =
        dml::Reinterpret(h_max, r.GetOutputDesc().sizes, dml::NullOpt);

    auto delta_greater_than_zero = delta > zero;
    auto h = dml::If(delta_greater_than_zero, h_max_resized, zero);
    auto s = dml::If(delta_greater_than_zero, delta / c_max, zero);

    return {h, s, c_max};
}

static dml::Expression HSVPlanesToRGB(
    dml::Graph& scope,
    dml::Expression h,
    dml::Expression s,
    dml::Expression v)
{
    // This helper can only be called with {N,H,W,1}-shaped tensors with equal
    // sizes.
    assert(
        h.GetOutputDesc().sizes.size() == 4 && h.GetOutputDesc().sizes[3] == 1);
    assert(
        s.GetOutputDesc().sizes.size() == 4 && s.GetOutputDesc().sizes[3] == 1);
    assert(
        v.GetOutputDesc().sizes.size() == 4 && v.GetOutputDesc().sizes[3] == 1);
    for (size_t i = 0; i < 3; ++i)
    {
        assert(h.GetOutputDesc().sizes[i] == s.GetOutputDesc().sizes[i]);
        assert(h.GetOutputDesc().sizes[i] == v.GetOutputDesc().sizes[i]);
    }

    auto output_shape = h.GetOutputDesc().sizes;
    output_shape[3] = 3;

    dml::TensorDesc::Dimensions bcast_strides =
        {output_shape[2] * output_shape[1], output_shape[2], 1, 0};
    auto s_bcast = dml::Reinterpret(s, output_shape, bcast_strides);
    auto v_bcast = dml::Reinterpret(v, output_shape, bcast_strides);

    // Calculate partial RGB planes, join them, then perform final math.
    auto r = dml::Abs(h, DML_SCALE_BIAS{6, -3}) - 1;
    auto g = 2 - dml::Abs(h, DML_SCALE_BIAS{6, -2});
    auto b = 2 - dml::Abs(h, DML_SCALE_BIAS{6, -4});
    auto rgb = dml::Join(std::vector<dml::Expression>{r, g, b}, 3);
    rgb = ((dml::Clip(rgb, 0, 1) - 1) * s_bcast + 1) * v_bcast;

    return rgb;
}

template <typename ConversionFunctor>
class DmlColorConversionKernel : public DmlKernel
{
  public:
    using InitHelper = ColorConversionInitHelper;

    explicit DmlColorConversionKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        auto num_channels =
            static_cast<uint32_t>(init_helper->GetNumChannels());
        auto num_pixels = static_cast<uint32_t>(init_helper->GetNumPixels());

        uint32_t tensor_sizes[4] = {1, 1, num_pixels, num_channels};
        auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
        DmlTensorInfo tensor_info = {};
        tensor_info.kernel_index = 0;
        tensor_info.desc = DmlTensorDesc{data_type, tensor_sizes};

        DmlKernelTensors tensors = {};
        tensors.inputs = {tensor_info};
        tensors.outputs = {tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);

        ConversionFunctor f;
        auto converted = f(scope, input);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {converted});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename T>
struct DmlRGBToHSVFunctor
{
    dml::Expression operator()(dml::Graph& scope, dml::Expression input)
    {
        return dml::Join(RGBToHSVPlanes<T>(scope, input), 3);
    }
};

struct DmlHSVToRGBFunctor
{
    dml::Expression operator()(dml::Graph& scope, dml::Expression input)
    {
        auto hsv_planes = dml::Split(input, 3, {1, 1, 1});
        return HSVPlanesToRGB(
            scope,
            hsv_planes[0],
            hsv_planes[1],
            hsv_planes[2]);
    }
};

template <typename Functor, bool needs_three_channels_input>
class DmlAdjustImageKernel : public DmlKernel
{
  public:
    using InitHelper = AdjustImageInitHelper<needs_three_channels_input>;

    explicit DmlAdjustImageKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        auto width = static_cast<uint32_t>(init_helper->GetWidth());
        auto height = static_cast<uint32_t>(init_helper->GetHeight());
        auto channels = static_cast<uint32_t>(init_helper->GetChannels());

        // Conversion helpers and DML require 4D input, but the leading
        // dimensions may be coalsced into N.
        auto num_images = static_cast<uint32_t>(
            input_shape.num_elements() / (width * height * channels));
        uint32_t input_shape_dml[4] = {num_images, height, width, channels};
        uint32_t zero_strides[4] = {0, 0, 0, 0};
        auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));

        DmlTensorInfo input_tensor_info = {};
        input_tensor_info.kernel_index = 0;
        input_tensor_info.desc = DmlTensorDesc{data_type, input_shape_dml};

        DmlTensorInfo adjustment_tensor_info = {};
        adjustment_tensor_info.kernel_index = 1;
        adjustment_tensor_info.desc =
            DmlTensorDesc{data_type, input_shape_dml, zero_strides};

        DmlKernelTensors tensors = {};
        tensors.inputs = {input_tensor_info, adjustment_tensor_info};
        tensors.outputs = {input_tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);
        auto adjustment = dml::InputTensor(scope, 1, inputs[1]);

        Functor f;
        auto result = f(scope, input, adjustment);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename Functor, bool needs_three_channels_input>
class DmlAdjustImageClampedKernel : public DmlKernel
{
  public:
    using InitHelper = AdjustImageInitHelper<needs_three_channels_input>;

    explicit DmlAdjustImageClampedKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 4);
        CHECK(ctx->GetOutputCount() == 1);

        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        auto width = static_cast<uint32_t>(init_helper->GetWidth());
        auto height = static_cast<uint32_t>(init_helper->GetHeight());
        auto channels = static_cast<uint32_t>(init_helper->GetChannels());

        // Conversion helpers and DML require 4D input, but the leading
        // dimensions may be coalsced into N.
        auto num_images = static_cast<uint32_t>(
            input_shape.num_elements() / (width * height * channels));
        uint32_t input_shape_dml[4] = {num_images, height, width, channels};
        uint32_t zero_strides[4] = {0, 0, 0, 0};
        auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));

        DmlTensorInfo input_tensor_info = {};
        input_tensor_info.kernel_index = 0;
        input_tensor_info.desc = DmlTensorDesc{data_type, input_shape_dml};

        DmlTensorInfo adjustment_tensor_info = {};
        adjustment_tensor_info.kernel_index = 1;
        adjustment_tensor_info.desc =
            DmlTensorDesc{data_type, input_shape_dml, zero_strides};

        DmlTensorInfo min_value_tensor_info = {};
        min_value_tensor_info.kernel_index = 2;
        min_value_tensor_info.desc =
            DmlTensorDesc{data_type, input_shape_dml, zero_strides};

        DmlTensorInfo max_value_tensor_info = {};
        max_value_tensor_info.kernel_index = 3;
        max_value_tensor_info.desc =
            DmlTensorDesc{data_type, input_shape_dml, zero_strides};

        DmlKernelTensors tensors = {};
        tensors.inputs = {
            input_tensor_info,
            adjustment_tensor_info,
            min_value_tensor_info,
            max_value_tensor_info,
        };

        tensors.outputs = {input_tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);
        auto adjustment = dml::InputTensor(scope, 1, inputs[1]);
        auto min_value = dml::InputTensor(scope, 2, inputs[2]);
        auto max_value = dml::InputTensor(scope, 3, inputs[3]);

        Functor f;
        auto result = f(scope, input, adjustment, min_value, max_value);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename T>
struct DmlAdjustSaturationFunctor
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression input,
        dml::Expression scale)
    {
        auto hsv_planes = RGBToHSVPlanes<T>(scope, input);
        auto scale_bcast = dml::Reinterpret(
            scale,
            hsv_planes[0].GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{0, 0, 0, 0});
        auto s_adjusted = dml::Clip(hsv_planes[1] * scale_bcast, 0.0f, 1.0f);
        return HSVPlanesToRGB(scope, hsv_planes[0], s_adjusted, hsv_planes[2]);
    }
};

template <typename T>
struct DmlAdjustHueFunctor
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression input,
        dml::Expression delta)
    {
        auto hsv_planes = RGBToHSVPlanes<T>(scope, input);

        auto one = dml::ScalarTensor<T>(
            scope,
            TfTensorTypeTraits<T>::FromFloat(1.0f),
            hsv_planes[0].GetOutputDesc().sizes);

        auto delta_bcast = dml::Reinterpret(
            delta,
            hsv_planes[0].GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{0, 0, 0, 0});

        auto h_adjusted = dml::ModulusFloor(hsv_planes[0] + delta_bcast, one);

        return HSVPlanesToRGB(scope, h_adjusted, hsv_planes[1], hsv_planes[2]);
    }
};

struct DmlAdjustContrastFunctor
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression input,
        dml::Expression contrast_factor,
        dml::Expression min_value,
        dml::Expression max_value)
    {
        // AdjustContrast can take inputs of different types, but only supports
        // float32 for its output
        if (input.GetOutputDesc().dataType != DML_TENSOR_DATA_TYPE_FLOAT32)
        {
            input = dml::Cast(input, DML_TENSOR_DATA_TYPE_FLOAT32);
        }

        contrast_factor = dml::Reinterpret(
            contrast_factor,
            input.GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{0, 0, 0, 0});

        auto mean = dml::Reinterpret(
            dml::Reduce(input, DML_REDUCE_FUNCTION_AVERAGE, {1, 2}),
            input.GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{3, 0, 0, 1});

        auto result = contrast_factor * (input - mean) + mean;
        result = dml::Min(max_value, dml::Max(result, min_value));

        return result;
    }
};

struct DmlAdjustContrastFunctorV2
{
    dml::Expression operator()(
        dml::Graph& scope,
        dml::Expression input,
        dml::Expression contrast_factor)
    {
        contrast_factor = dml::Reinterpret(
            contrast_factor,
            input.GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{0, 0, 0, 0});

        auto mean = dml::Reinterpret(
            dml::Reduce(input, DML_REDUCE_FUNCTION_AVERAGE, {1, 2}),
            input.GetOutputDesc().sizes,
            dml::TensorDesc::Dimensions{3, 0, 0, 1});

        return contrast_factor * (input - mean) + mean;
    }
};

static void RegisterRGBToHSV()
{

    using K = KernelDefinition<
        ops::RGBToHSV,
        DmlKernelWrapper<
            DmlColorConversionKernel<DmlRGBToHSVFunctor<float>>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::RGBToHSV::Attribute::T, TF_FLOAT>();
}

static void RegisterHSVToRGB()
{
    using K = KernelDefinition<
        ops::HSVToRGB,
        DmlKernelWrapper<
            DmlColorConversionKernel<DmlHSVToRGBFunctor>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::HSVToRGB::Attribute::T, TF_FLOAT>();
}

static void RegisterAdjustSaturation()
{
    using half_kernel = KernelDefinition<
        ops::AdjustSaturation,
        DmlKernelWrapper<
            DmlAdjustImageKernel<DmlAdjustSaturationFunctor<Eigen::half>, true>,
            GetOutputShapeAsInputShapeHelper>>;

    using float_kernel = KernelDefinition<
        ops::AdjustSaturation,
        DmlKernelWrapper<
            DmlAdjustImageKernel<DmlAdjustSaturationFunctor<float>, true>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        half_kernel,
        ops::AdjustSaturation::Attribute::T,
        TF_HALF>();
    RegisterWithTypes<
        float_kernel,
        ops::AdjustSaturation::Attribute::T,
        TF_FLOAT>();
}

static void RegisterAdjustHue()
{
    using half_kernel = KernelDefinition<
        ops::AdjustHue,
        DmlKernelWrapper<
            DmlAdjustImageKernel<DmlAdjustHueFunctor<Eigen::half>, true>,
            GetOutputShapeAsInputShapeHelper>>;

    using float_kernel = KernelDefinition<
        ops::AdjustHue,
        DmlKernelWrapper<
            DmlAdjustImageKernel<DmlAdjustHueFunctor<float>, true>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<half_kernel, ops::AdjustHue::Attribute::T, TF_HALF>();
    RegisterWithTypes<float_kernel, ops::AdjustHue::Attribute::T, TF_FLOAT>();
}

static void RegisterAdjustContrast()
{
    using K = KernelDefinition<
        ops::AdjustContrast,
        DmlKernelWrapper<
            DmlAdjustImageClampedKernel<DmlAdjustContrastFunctor, true>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::AdjustContrast::Attribute::T,
        TF_FLOAT,
        TF_UINT8,
        TF_INT8,
        TF_INT16,
        TF_INT32>();
}

static void RegisterAdjustContrastV2()
{
    using K = KernelDefinition<
        ops::AdjustContrastv2,
        DmlKernelWrapper<
            DmlAdjustImageKernel<DmlAdjustContrastFunctorV2, false>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::AdjustContrastv2::Attribute::T,
        TF_FLOAT,
        TF_UINT8,
        TF_INT8,
        TF_INT16,
        TF_INT32>();
}

void RegisterKernels_Image()
{
    RegisterRGBToHSV();
    RegisterHSVToRGB();
    RegisterAdjustSaturation();
    RegisterAdjustHue();
    RegisterAdjustContrast();
    RegisterAdjustContrastV2();
}

} // namespace tfdml