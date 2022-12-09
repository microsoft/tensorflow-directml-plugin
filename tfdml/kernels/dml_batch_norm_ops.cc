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
#include "tfdml/runtime_adapter/tensor_format.h"

namespace tfdml
{

// FusedBatchNormEx op supports side inputs and activations:
//   (1) batch_norm + activation
//   (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode
{
    kIdentity,
    kRelu
};

Status ParseActivationMode(
    OpKernelConstruction* context,
    FusedBatchNormActivationMode* activation_mode)
{
    std::string activation_mode_str;
    TF_RETURN_IF_ERROR(
        context->GetAttr("activation_mode", &activation_mode_str));

    if (activation_mode_str == "Identity")
    {
        *activation_mode = FusedBatchNormActivationMode::kIdentity;
        return Status::OK();
    }
    if (activation_mode_str == "Relu")
    {
        *activation_mode = FusedBatchNormActivationMode::kRelu;
        return Status::OK();
    }
    return errors::InvalidArgument(
        "Unsupported activation mode: ",
        activation_mode_str);
}

class FusedBatchNormInitializationHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::string tensor_format_attr;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("data_format", &tensor_format_attr));
            OP_REQUIRES(
                ctx,
                FormatFromString(tensor_format_attr, &tensor_format),
                errors::InvalidArgument("Invalid data format"));

            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr(
                    "exponential_avg_factor",
                    &exponential_avg_factor));

            if (ctx->GetAttr("num_side_inputs", &num_side_inputs).ok())
            {
                OP_REQUIRES_OK(ctx, ParseActivationMode(ctx, &activation_mode));

                OP_REQUIRES(
                    ctx,
                    activation_mode ==
                            FusedBatchNormActivationMode::kIdentity ||
                        activation_mode == FusedBatchNormActivationMode::kRelu,
                    errors::InvalidArgument("FusedBatchNorm only supports "
                                            "Identity and Relu for now."));

                OP_REQUIRES(
                    ctx,
                    num_side_inputs >= 0 && num_side_inputs <= 1,
                    errors::InvalidArgument(
                        "FusedBatchNorm accepts at most one side input."));

                if (num_side_inputs > 0 && is_training)
                {
                    OP_REQUIRES(
                        ctx,
                        activation_mode !=
                            FusedBatchNormActivationMode::kIdentity,
                        errors::InvalidArgument(
                            "Identity activation is not supported with "
                            "non-empty side input"));
                }
            }
            else
            {
                num_side_inputs = 0;
                activation_mode = FusedBatchNormActivationMode::kIdentity;
            }
        }

        float epsilon;
        bool is_training;
        TensorFormat tensor_format;
        int num_side_inputs;
        FusedBatchNormActivationMode activation_mode;
        float exponential_avg_factor;
    };

    FusedBatchNormInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& x = ctx->input(0);
        const Tensor& scale = ctx->input(1);
        const Tensor& offset = ctx->input(2);
        const Tensor& estimated_mean = ctx->input(3);
        const Tensor& estimated_variance = ctx->input(4);

        OP_REQUIRES(
            ctx,
            x.dims() == 4 || x.dims() == 5,
            errors::InvalidArgument(
                "input must be 4 or 5-dimensional",
                x.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            scale.dims() == 1,
            errors::InvalidArgument(
                "scale must be 1-dimensional",
                scale.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            offset.dims() == 1,
            errors::InvalidArgument(
                "offset must be 1-dimensional",
                offset.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            estimated_mean.dims() == 1,
            errors::InvalidArgument(
                "estimated_mean must be 1-dimensional",
                estimated_mean.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            estimated_variance.dims() == 1,
            errors::InvalidArgument(
                "estimated_variance must be 1-dimensional",
                estimated_variance.shape().DebugString()));

        const auto num_channels = GetTensorDim(x, attr_->tensor_format, 'C');
        OP_REQUIRES(
            ctx,
            scale.NumElements() == num_channels,
            errors::InvalidArgument(
                "scale must have the same number of elements as the channels "
                "of x, got ",
                scale.NumElements(),
                " and ",
                num_channels));
        OP_REQUIRES(
            ctx,
            offset.NumElements() == num_channels,
            errors::InvalidArgument(
                "offset must have the same number of elements as the channels "
                "of x, got ",
                offset.NumElements(),
                " and ",
                num_channels));
        if (!attr_->is_training || attr_->exponential_avg_factor != 1.0f)
        {
            std::string prefix_msg = attr_->is_training
                                         ? "When exponential_avg_factor != 1"
                                         : "When is_training=false";
            OP_REQUIRES(
                ctx,
                estimated_mean.NumElements() == num_channels,
                errors::InvalidArgument(
                    prefix_msg,
                    ", mean must have the same number of elements as the "
                    "channels of x, got ",
                    estimated_mean.NumElements(),
                    " and ",
                    num_channels));
            OP_REQUIRES(
                ctx,
                estimated_variance.NumElements() == num_channels,
                errors::InvalidArgument(
                    prefix_msg,
                    ", variance must have the same number of elements as the "
                    "channels of x, got ",
                    estimated_variance.NumElements(),
                    " and ",
                    num_channels));
        }

        if (attr_->num_side_inputs > 0)
        {
            const Tensor& side_input = ctx->input(5);

            OP_REQUIRES(
                ctx,
                side_input.shape() == x.shape(),
                errors::InvalidArgument(
                    "side_input shape must be equal to input shape: ",
                    side_input.shape().DebugString(),
                    " != ",
                    x.shape().DebugString()));
        }
    }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        // FusedBatchNorm can legitimately have empty tensors depending on
        // whether is_training is true or not. So this kernel is only truly a
        // no-op when the input tensor is empty.
        return (ctx->input(0).NumElements() == 0) &&
               attr_->exponential_avg_factor == 1.0f;
    }

    float GetEpsilon() const { return attr_->epsilon; }
    bool IsTraining() const { return attr_->is_training; }
    TensorFormat GetFormat() const { return attr_->tensor_format; }
    bool AddSideInput() const { return attr_->num_side_inputs > 0; }

    FusedBatchNormActivationMode GetActivationMode() const
    {
        return attr_->activation_mode;
    }

    float GetExponentialAvgFactor() const
    {
        return attr_->exponential_avg_factor;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class BatchGlobalNormInitializationHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("variance_epsilon", &variance_epsilon));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr(
                    "scale_after_normalization",
                    &scale_after_normalization));
        }
        float variance_epsilon;
        bool scale_after_normalization;
    };

    BatchGlobalNormInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& t = ctx->input(0);
        const Tensor& m = ctx->input(1);
        const Tensor& v = ctx->input(2);
        const Tensor& beta = ctx->input(3);
        const Tensor& gamma = ctx->input(4);
        OP_REQUIRES(
            ctx,
            t.dims() == 4 || t.dims() == 5,
            errors::InvalidArgument(
                "input must be 4 or 5-dimensional",
                t.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            gamma.dims() == 1,
            errors::InvalidArgument(
                "scale must be 1-dimensional",
                gamma.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            beta.dims() == 1,
            errors::InvalidArgument(
                "offset must be 1-dimensional",
                beta.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            m.dims() == 1,
            errors::InvalidArgument(
                "estimated_mean must be 1-dimensional",
                m.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            v.dims() == 1,
            errors::InvalidArgument(
                "estimated_variance must be 1-dimensional",
                v.shape().DebugString()));
    }

    float GetVarianceEpsilon() const { return attr_->variance_epsilon; }
    bool ScaleAfterNormalization() const
    {
        return attr_->scale_after_normalization;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class BatchGlobalNormGradInitializationHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("variance_epsilon", &variance_epsilon));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr(
                    "scale_after_normalization",
                    &scale_after_normalization));
        }
        float variance_epsilon;
        bool scale_after_normalization;
    };

    BatchGlobalNormGradInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& t = ctx->input(0);
        const Tensor& m = ctx->input(1);
        const Tensor& v = ctx->input(2);
        const Tensor& gamma = ctx->input(3);
        const Tensor& backprop = ctx->input(4);
        OP_REQUIRES(
            ctx,
            t.dims() == 4 || t.dims() == 5,
            errors::InvalidArgument(
                "input must be 4 or 5-dimensional",
                t.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            gamma.dims() == 1,
            errors::InvalidArgument(
                "scale must be 1-dimensional",
                gamma.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            backprop.dims() == 4,
            errors::InvalidArgument(
                "offset must be 1-dimensional",
                backprop.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            m.dims() == 1,
            errors::InvalidArgument(
                "estimated_mean must be 1-dimensional",
                m.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            v.dims() == 1,
            errors::InvalidArgument(
                "estimated_variance must be 1-dimensional",
                v.shape().DebugString()));
    }

    float GetVarianceEpsilon() const { return attr_->variance_epsilon; }
    bool ScaleAfterNormalization() const
    {
        return attr_->scale_after_normalization;
    }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

static dml::Expression CreateBatchNormNode(
    dml::Expression x,
    dml::Expression mean,
    dml::Expression variance,
    dml::Expression scale,
    dml::Expression offset,
    float epsilon,
    FusedBatchNormActivationMode activation_mode)
{
    // This should already have been validated in the init helper
    assert(
        activation_mode == FusedBatchNormActivationMode::kIdentity ||
        activation_mode == FusedBatchNormActivationMode::kRelu);

    auto fused_activation =
        activation_mode == FusedBatchNormActivationMode::kIdentity
            ? dml::FusedActivation::None()
            : dml::FusedActivation::Relu();

    // TODO: Remove when the bug causing intermediate value overflow is fixed
    // TFDML #41739122
    bool float16_inputs =
        x.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_FLOAT16;
    if (float16_inputs)
    {
        x = dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    if (mean.GetOutputDesc().dataType != DML_TENSOR_DATA_TYPE_FLOAT32)
    {
        mean = dml::Cast(mean, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    if (variance.GetOutputDesc().dataType != DML_TENSOR_DATA_TYPE_FLOAT32)
    {
        variance = dml::Cast(variance, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    if (scale.GetOutputDesc().dataType != DML_TENSOR_DATA_TYPE_FLOAT32)
    {
        scale = dml::Cast(scale, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    if (offset.GetOutputDesc().dataType != DML_TENSOR_DATA_TYPE_FLOAT32)
    {
        offset = dml::Cast(offset, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    constexpr bool is_spatial = true;
    auto result = dml::BatchNormalization(
        x,
        mean,
        variance,
        scale,
        offset,
        is_spatial,
        epsilon,
        fused_activation);

    if (float16_inputs)
    {
        result = dml::Cast(result, DML_TENSOR_DATA_TYPE_FLOAT16);
    }

    return result;
}

static dml::BatchNormalizationTrainingOutputs CreateBatchNormTrainingNode(
    dml::Expression x,
    dml::Expression scale,
    dml::Expression offset,
    absl::optional<dml::Expression> fused_add,
    float epsilon,
    FusedBatchNormActivationMode activation_mode)
{
    // This should already have been validated in the init helper
    assert(
        activation_mode == FusedBatchNormActivationMode::kIdentity ||
        activation_mode == FusedBatchNormActivationMode::kRelu);

    auto fused_activation =
        activation_mode == FusedBatchNormActivationMode::kIdentity
            ? dml::FusedActivation::None()
            : dml::FusedActivation::Relu();

    return dml::BatchNormalizationTraining(
        x,
        scale,
        offset,
        fused_add,
        epsilon,
        fused_activation);
}

class DmlFusedBatchNormKernel : public DmlKernel
{
    enum InputIndex
    {
        kX,
        kScale,
        kOffset,
        kMean,
        kVariance,
        kSideInput,
    };

  public:
    using InitHelper = FusedBatchNormInitializationHelper;

    explicit DmlFusedBatchNormKernel(
        DmlKernelConstruction* ctx,
        const FusedBatchNormInitializationHelper* init_helper)
    {
        // FusedBatchNormEx takes an additional side input
        CHECK(ctx->GetInputCount() == 5 || ctx->GetInputCount() == 6);

        // FusedBatchNormV3 returns an additional output
        CHECK(ctx->GetOutputCount() == 5 || ctx->GetOutputCount() == 6);

        float epsilon = init_helper->GetEpsilon();
        bool is_training = init_helper->IsTraining();
        TensorFormat tensor_format = init_helper->GetFormat();

        if (ctx->GetInputTensorShape(0).num_elements() == 0)
        {
            assert(init_helper->GetExponentialAvgFactor() != 1.0f);
            InitializeForNoOp(
                ctx,
                epsilon,
                tensor_format,
                init_helper->GetExponentialAvgFactor());
        }
        else if (is_training)
        {
            InitializeForTraining(
                ctx,
                epsilon,
                tensor_format,
                init_helper->AddSideInput(),
                init_helper->GetActivationMode(),
                init_helper->GetExponentialAvgFactor());
        }
        else
        {
            InitializeForInference(
                ctx,
                epsilon,
                tensor_format,
                init_helper->AddSideInput(),
                init_helper->GetActivationMode());
        }
    }

    void InitializeForNoOp(
        DmlKernelConstruction* ctx,
        float epsilon,
        TensorFormat tensor_format,
        float exponential_avg_factor)
    {
        DmlKernelParams params;
        params.kernel_input_indices = {kMean, kVariance};

        // Computed mean, computed variance
        params.kernel_output_indices = {1, 2};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        const uint32_t dim_count = ctx->GetInputTensorShape(0).dims();

        // Mean and variance are 1D, in the C dimension
        TensorShape mean_shape = ctx->GetInputTensorShape(kMean);
        TensorShape variance_shape = ctx->GetInputTensorShape(kVariance);

        // The order of the N, D, H and W dimensions doesn't matter because they
        // are all 1's. The only thing we are trying to do here is reorder the C
        // dimension to the right position.
        const int missing_dims = dim_count - mean_shape.dims();
        for (int i = 0; i < missing_dims; ++i)
        {
            mean_shape.AddDim(1);
            variance_shape.AddDim(1);
        }

        using namespace DmlTensorAxes;
        assert(dim_count == 4 || dim_count == 5);
        const auto scalar_layout =
            dim_count == 4 ? DmlTensorLayout::Cnhw() : DmlTensorLayout::Cndhw();

        tensors.inputs[0]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kMean),
            mean_shape,
            mean_shape,
            scalar_layout);
        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kVariance),
            variance_shape,
            variance_shape,
            scalar_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        auto scope =
            dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));
        auto mean = dml::InputTensor(scope, 0, input_descs[0]);
        auto variance = dml::InputTensor(scope, 1, input_descs[1]);

        float one_minus_factor = 1.0f - exponential_avg_factor;
        auto corrected_variance = variance * one_minus_factor;
        auto corrected_mean = mean * one_minus_factor;

        auto outputs = {
            corrected_mean,     // batch_mean
            corrected_variance, // batch_variance
        };

        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    // Initializes the batch norm kernel for training. In training mode, we
    // don't receive the mean/variance and need to compute it ourselves.
    void InitializeForTraining(
        DmlKernelConstruction* ctx,
        float epsilon,
        TensorFormat tensor_format,
        bool add_side_input,
        FusedBatchNormActivationMode activation_mode,
        float exponential_avg_factor)
    {
        DmlKernelParams params;

        params.kernel_input_indices = {kX, kScale, kOffset};

        int mean_index = -1;
        int variance_index = -1;
        int side_input_index = -1;

        if (exponential_avg_factor != 1.0f)
        {
            mean_index = params.kernel_input_indices.size();
            params.kernel_input_indices.push_back(kMean);

            variance_index = params.kernel_input_indices.size();
            params.kernel_input_indices.push_back(kVariance);
        }

        if (add_side_input)
        {
            side_input_index = params.kernel_input_indices.size();
            params.kernel_input_indices.push_back(kSideInput);
        }

        // Normalized output, computed mean, computed variance, saved mean,
        // saved variance
        params.kernel_output_indices = {0, 1, 2, 3, 4};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        const uint32_t dim_count = ctx->GetInputTensorShape(0).dims();

        // Input and output tensors have their layout specified by the
        // "data_format" attribute
        DmlTensorLayout input_output_layout =
            GetDmlTensorLayout(tensor_format, dim_count);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        // Scale and bias are 1D, in the C dimension
        TensorShape scale_shape = ctx->GetInputTensorShape(1);
        TensorShape offset_shape = ctx->GetInputTensorShape(2);

        // The order of the N, D, H and W dimensions doesn't matter because they
        // are all 1's. The only thing we are trying to do here is reorder the C
        // dimension to the right position.
        const int missing_dims = dim_count - scale_shape.dims();
        for (int i = 0; i < missing_dims; ++i)
        {
            scale_shape.AddDim(1);
            offset_shape.AddDim(1);
        }

        using namespace DmlTensorAxes;
        assert(dim_count == 4 || dim_count == 5);
        const auto scalar_layout =
            dim_count == 4 ? DmlTensorLayout::Cnhw() : DmlTensorLayout::Cndhw();

        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            scale_shape,
            scale_shape,
            scalar_layout);
        tensors.inputs[2]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            offset_shape,
            offset_shape,
            scalar_layout);

        if (exponential_avg_factor != 1.0f)
        {
            TensorShape mean_shape = ctx->GetInputTensorShape(mean_index);
            TensorShape variance_shape =
                ctx->GetInputTensorShape(variance_index);

            for (int i = 0; i < missing_dims; ++i)
            {
                scale_shape.AddDim(1);
                offset_shape.AddDim(1);
                mean_shape.AddDim(1);
                variance_shape.AddDim(1);
            }

            tensors.inputs[mean_index]->desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(mean_index),
                mean_shape,
                mean_shape,
                scalar_layout);
            tensors.inputs[variance_index]->desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(variance_index),
                variance_shape,
                variance_shape,
                scalar_layout);
        }

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        auto scope =
            dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));
        auto x = dml::InputTensor(scope, 0, input_descs[0]);
        auto scale = dml::InputTensor(scope, 1, input_descs[1]);
        auto offset = dml::InputTensor(scope, 2, input_descs[2]);

        DML_TENSOR_DATA_TYPE input_type = x.GetOutputDesc().dataType;

        // scale and offset are always float32, but the input data type might
        // not be. If that's the case, we need to insert a cast.
        bool is_cast_required = (input_type != scale.GetOutputDesc().dataType);

        // TODO: Remove when the bug causing intermediate value overflow is
        // fixed
        // TFDML #41739122
        is_cast_required = false;

        if (is_cast_required)
        {
            scale = dml::Cast(scale, input_type);
            offset = dml::Cast(offset, input_type);
        }

        absl::optional<dml::Expression> side_input;
        if (add_side_input)
        {
            side_input = dml::InputTensor(
                scope,
                side_input_index,
                input_descs[side_input_index]);
        }

        // TODO: Remove when the bug causing intermediate value overflow is
        // fixed
        // TFDML #41739122
        if (input_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            x = dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32);

            if (side_input)
            {
                *side_input =
                    dml::Cast(*side_input, DML_TENSOR_DATA_TYPE_FLOAT32);
            }
        }

        dml::BatchNormalizationTrainingOutputs dml_outputs =
            CreateBatchNormTrainingNode(
                x,
                scale,
                offset,
                side_input,
                epsilon,
                activation_mode);

        // TODO: Remove when the bug causing intermediate value overflow is
        // fixed
        // TFDML #41739122
        if (input_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            dml_outputs.output =
                dml::Cast(dml_outputs.output, DML_TENSOR_DATA_TYPE_FLOAT16);
        }

        // Apply Bessel's correction to the variance
        auto input_sizes = x.GetOutputDesc().sizes;
        uint32_t norm_elem_count =
            input_sizes[0] * input_sizes[2] * input_sizes[3];
        float bessel_correction_factor = 1.0f;
        if (norm_elem_count > 1)
        {
            // Prevent division by 0
            bessel_correction_factor =
                (float)norm_elem_count / (norm_elem_count - 1);
        }

        auto corrected_variance =
            dml_outputs.variance * bessel_correction_factor;

        if (is_cast_required)
        {
            // These output tensors are defined to always be float32, so insert
            // a cast if necessary
            dml_outputs.mean =
                dml::Cast(dml_outputs.mean, DML_TENSOR_DATA_TYPE_FLOAT32);
            corrected_variance =
                dml::Cast(corrected_variance, DML_TENSOR_DATA_TYPE_FLOAT32);
            dml_outputs.variance =
                dml::Cast(dml_outputs.variance, DML_TENSOR_DATA_TYPE_FLOAT32);
        }

        auto corrected_mean = dml_outputs.mean;

        // Mean and Variance are always float32, so execute the operation after
        // the cast
        if (exponential_avg_factor != 1.0f)
        {
            auto old_mean =
                dml::InputTensor(scope, mean_index, input_descs[mean_index]);
            auto old_variance = dml::InputTensor(
                scope,
                variance_index,
                input_descs[variance_index]);

            float one_minus_factor = 1.0f - exponential_avg_factor;
            corrected_variance = corrected_variance * exponential_avg_factor +
                                 old_variance * one_minus_factor;

            corrected_mean = corrected_mean * exponential_avg_factor +
                             old_mean * one_minus_factor;
        }

        auto outputs = {
            dml_outputs.output,
            corrected_mean,      // batch_mean
            corrected_variance,  // batch_variance
            dml_outputs.mean,    // saved_mean
            dml_outputs.variance // saved_variance
        };

        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    // Initializes the kernel for the is_training=false case.
    void InitializeForInference(
        DmlKernelConstruction* ctx,
        float epsilon,
        TensorFormat tensor_format,
        bool add_side_input,
        FusedBatchNormActivationMode activation_mode)
    {
        DmlKernelParams params;

        // DML's BatchNorm operator takes inputs in a different order than the
        // kernel receives them
        params.kernel_input_indices = {kX, kMean, kVariance, kScale, kOffset};

        if (add_side_input)
        {
            params.kernel_input_indices.push_back(kSideInput);
        }

        // Normalized output, computed mean, computed variance. We don't use any
        // of the reserved_space outputs for the inference case.
        params.kernel_output_indices = {0, 1, 2};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        const uint32_t dim_count = ctx->GetInputTensorShape(0).dims();

        // Input and output tensors have their layout specified by the
        // "data_format" attribute
        DmlTensorLayout input_output_layout =
            GetDmlTensorLayout(tensor_format, dim_count);
        tensors.inputs[0]->desc =
            CreateTensorDescFromInput(ctx, 0, input_output_layout);
        tensors.outputs[0]->desc =
            CreateTensorDescFromOutput(ctx, 0, input_output_layout);

        // Mean, variance, scale, and bias are 1D; in the C dimension

        TensorShape mean_shape = ctx->GetInputTensorShape(1);
        TensorShape variance_shape = ctx->GetInputTensorShape(2);
        TensorShape scale_shape = ctx->GetInputTensorShape(3);
        TensorShape offset_shape = ctx->GetInputTensorShape(4);

        // The order of the N, D, H and W dimensions doesn't matter because they
        // are all 1's. The only thing we are trying to do here is reorder the C
        // dimension to the right position.
        const int missing_dims = dim_count - mean_shape.dims();
        for (int i = 0; i < missing_dims; ++i)
        {
            mean_shape.AddDim(1);
            variance_shape.AddDim(1);
            scale_shape.AddDim(1);
            offset_shape.AddDim(1);
        }

        using namespace DmlTensorAxes;
        assert(dim_count == 4 || dim_count == 5);
        const auto scalar_layout =
            dim_count == 4 ? DmlTensorLayout::Cnhw() : DmlTensorLayout::Cndhw();

        tensors.inputs[1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            mean_shape,
            mean_shape,
            scalar_layout);
        tensors.inputs[2]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            variance_shape,
            variance_shape,
            scalar_layout);
        tensors.inputs[3]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(3),
            scale_shape,
            scale_shape,
            scalar_layout);
        tensors.inputs[4]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(4),
            offset_shape,
            offset_shape,
            scalar_layout);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        auto scope =
            dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));
        auto x = dml::InputTensor(scope, 0, input_descs[0]);
        auto mean = dml::InputTensor(scope, 1, input_descs[1]);
        auto variance = dml::InputTensor(scope, 2, input_descs[2]);
        auto scale = dml::InputTensor(scope, 3, input_descs[3]);
        auto offset = dml::InputTensor(scope, 4, input_descs[4]);

        DML_TENSOR_DATA_TYPE input_type = x.GetOutputDesc().dataType;

        // mean/variance in its original datatype
        auto original_mean = mean;
        auto original_variance = variance;

        bool is_cast_required = (input_type != scale.GetOutputDesc().dataType);

        // TODO: Remove when the intermediate accumulation bug in DML is fixed
        // TFDML #41739122
        is_cast_required = false;

        // scale, offset, mean, and variance are always float32, but the input
        // data type might not be. If that's the case, we need to insert a cast.
        if (is_cast_required)
        {
            mean = dml::Cast(mean, input_type);
            variance = dml::Cast(variance, input_type);
            scale = dml::Cast(scale, input_type);
            offset = dml::Cast(offset, input_type);
        }

        // If we need to add a side input, we cannot fuse the activation with
        // BatchNorm since the side input needs to be added before the
        // activation
        auto normalized_output = CreateBatchNormNode(
            x,
            mean,
            variance,
            scale,
            offset,
            epsilon,
            add_side_input ? FusedBatchNormActivationMode::kIdentity
                           : activation_mode);

        // FusedBatchNormEx can provide a side input that we need to add before
        // activation
        if (add_side_input)
        {
            auto side_input = dml::InputTensor(scope, 5, input_descs[5]);
            normalized_output += side_input;

            if (activation_mode != FusedBatchNormActivationMode::kIdentity)
            {
                // Only Relu is supported
                assert(activation_mode == FusedBatchNormActivationMode::kRelu);
                normalized_output = dml::ActivationRelu(normalized_output);
            }
        }

        // TF requires that we output batch_mean and batch_variance in addition
        // to the normalized output. Since this is the inference case, we just
        // copy the original mean/variance to the batch_mean and batch_variance
        // outputs.
        auto outputs = {
            normalized_output,
            dml::Identity(original_mean),
            dml::Identity(original_variance),
        };

        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlBatchNormWithGlobalNormalizationKernel : public DmlKernel
{
    enum InputIndex
    {
        kT,
        kM,
        kV,
        kBeta,
        kGamma,
    };

  public:
    using InitHelper = BatchGlobalNormInitializationHelper;
    explicit DmlBatchNormWithGlobalNormalizationKernel(
        DmlKernelConstruction* ctx,
        const BatchGlobalNormInitializationHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 1);

        float variance_epsilon = init_helper->GetVarianceEpsilon();
        bool scale_after_normalization = init_helper->ScaleAfterNormalization();

        DmlKernelParams params;

        // DML's BatchNorm operator takes inputs in a different order than the
        // kernel receives them
        if (scale_after_normalization)
        {
            params.kernel_input_indices = {kT, kM, kV, kGamma, kBeta};
        }
        else
        {
            params.kernel_input_indices = {kT, kM, kV, kBeta};
        }
        params.kernel_output_indices = {0};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        const uint32_t beta_index = scale_after_normalization ? 4 : 3;
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto t = dml::InputTensor(scope, 0, input_descs[0]);
        auto m = dml::InputTensor(scope, 1, input_descs[1]);
        auto v = dml::InputTensor(scope, 2, input_descs[2]);
        auto beta =
            dml::InputTensor(scope, beta_index, input_descs[beta_index]);

        // Just use DML's BATCH_NORMALIZATION operator to compute the output.
        dml::Expression normalized_output;
        if (scale_after_normalization)
        {
            auto gamma = dml::InputTensor(scope, 3, input_descs[3]);
            normalized_output = CreateBatchNormNode(
                t,
                m,
                v,
                gamma,
                beta,
                variance_epsilon,
                FusedBatchNormActivationMode::kIdentity);
        }
        else
        {
            auto ones = dml::ScalarTensor(scope, 1.0f, v.GetOutputDesc().sizes);
            normalized_output = CreateBatchNormNode(
                t,
                m,
                v,
                ones,
                beta,
                variance_epsilon,
                FusedBatchNormActivationMode::kIdentity);
        }
        auto outputs = {normalized_output};
        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class FusedBatchNormGradInitializationHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            std::string tensor_format_attr;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("data_format", &tensor_format_attr));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
            OP_REQUIRES(
                ctx,
                FormatFromString(tensor_format_attr, &tensor_format),
                errors::InvalidArgument("Invalid data format"));
        }

        float epsilon;
        bool is_training;
        TensorFormat tensor_format;
    };

    FusedBatchNormGradInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& y_backprop = ctx->input(0);
        const Tensor& x = ctx->input(1);
        const Tensor& scale = ctx->input(2);
        const Tensor& reserved_space_1 = ctx->input(3);
        const Tensor& reserved_space_2 = ctx->input(4);

        OP_REQUIRES(
            ctx,
            x.shape() == y_backprop.shape(),
            errors::InvalidArgument(
                "x and y_backprop must have same shape, but x has shape ",
                x.shape().DebugString(),
                " and y_backprop has shape ",
                y_backprop.shape().DebugString()));

        const auto num_channels = GetTensorDim(x, attr_->tensor_format, 'C');
        OP_REQUIRES(
            ctx,
            scale.NumElements() == num_channels,
            errors::InvalidArgument(
                "scale must have the same number of elements as the channels "
                "of x, got ",
                scale.NumElements(),
                " and ",
                num_channels));
        OP_REQUIRES(
            ctx,
            reserved_space_1.NumElements() == num_channels,
            errors::InvalidArgument(
                "reserve_space_1 must have the same number of "
                "elements as the channels of x, got ",
                reserved_space_1.NumElements(),
                " and ",
                num_channels));
        OP_REQUIRES(
            ctx,
            reserved_space_2.NumElements() == num_channels,
            errors::InvalidArgument(
                "reserve_space_2 must have the same number of "
                "elements as the channels of x, got ",
                reserved_space_2.NumElements(),
                " and ",
                num_channels));
    }

    float GetEpsilon() const { return attr_->epsilon; }
    bool IsTraining() const { return attr_->is_training; }
    TensorFormat GetFormat() const { return attr_->tensor_format; }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class DmlFusedBatchNormGradKernel : public DmlKernel
{
    enum InputIndex
    {
        kYBackprop,
        kX,
        kScale,
        kReserveSpace1,
        kReserveSpace2,
        kReserveSpace3,
    };

    enum OutputIndex
    {
        kXBackprop,
        kScaleBackprop,
        kOffsetBackprop,
    };

  public:
    using InitHelper = FusedBatchNormGradInitializationHelper;

    explicit DmlFusedBatchNormGradKernel(
        DmlKernelConstruction* ctx,
        const FusedBatchNormGradInitializationHelper* init_helper)
    {
        // FusedBatchNormGradV3 takes an additional input (which we don't use)
        CHECK(ctx->GetInputCount() == 5 || ctx->GetInputCount() == 6);
        CHECK(ctx->GetOutputCount() == 5);

        DmlKernelParams params;

        // FusedBatchNormGradV3 receives an additional (6th) input which we
        // don't use, so we explicitly only supply 5 indices here
        params.kernel_input_indices = {0, 1, 2, 3, 4};

        // Only the first 3 outputs are used, the remainder are placeholders
        params.kernel_output_indices = {0, 1, 2};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        // y_backprop, x, and x_backprop have their layout specified by the
        // "data_format" attribute. The rest are 1D tensors in the C dimension
        using namespace DmlTensorAxes;

        int64_t dim_count = ctx->GetInputTensorShape(kX).dims();

        DmlTensorLayout layout =
            GetDmlTensorLayout(init_helper->GetFormat(), dim_count);

        TensorShape scale_dml_shape;
        TensorShape reserve_space_1_dml_shape;
        TensorShape reserve_space_2_dml_shape;
        TensorShape scale_backprop_dml_shape;
        TensorShape offset_backprop_dml_shape;

        for (int i = 0; i < dim_count; ++i)
        {
            scale_dml_shape.AddDim(1);
            reserve_space_1_dml_shape.AddDim(1);
            reserve_space_2_dml_shape.AddDim(1);
            scale_backprop_dml_shape.AddDim(1);
            offset_backprop_dml_shape.AddDim(1);
        }

        scale_dml_shape.set_dim(
            1,
            ctx->GetInputTensorShape(kScale).num_elements());
        reserve_space_1_dml_shape.set_dim(
            1,
            ctx->GetInputTensorShape(kReserveSpace1).num_elements());
        reserve_space_2_dml_shape.set_dim(
            1,
            ctx->GetInputTensorShape(kReserveSpace2).num_elements());

        // Inputs
        tensors.inputs[kYBackprop]->desc =
            CreateTensorDescFromInput(ctx, kYBackprop, layout);
        tensors.inputs[kX]->desc = CreateTensorDescFromInput(ctx, kX, layout);
        tensors.inputs[kScale]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kScale),
            scale_dml_shape,
            scale_dml_shape);
        tensors.inputs[kReserveSpace1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kReserveSpace1),
            reserve_space_1_dml_shape,
            reserve_space_1_dml_shape);
        tensors.inputs[kReserveSpace2]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kReserveSpace2),
            reserve_space_2_dml_shape,
            reserve_space_2_dml_shape);

        // Outputs
        tensors.outputs[kXBackprop]->desc =
            CreateTensorDescFromInput(ctx, kYBackprop, layout);
        tensors.inputs[kX]->desc = CreateTensorDescFromInput(ctx, kX, layout);
        tensors.inputs[kScale]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kScale),
            scale_dml_shape,
            scale_dml_shape);
        tensors.inputs[kReserveSpace1]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kReserveSpace1),
            reserve_space_1_dml_shape,
            reserve_space_1_dml_shape);
        tensors.inputs[kReserveSpace2]->desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(kReserveSpace2),
            reserve_space_2_dml_shape,
            reserve_space_2_dml_shape);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(
            ctx->GetDmlDevice(),
            GetDmlXTensorPolicy(init_helper->GetFormat()));

        auto y_backprop =
            dml::InputTensor(scope, kYBackprop, input_descs[kYBackprop]);
        auto x = dml::InputTensor(scope, kX, input_descs[kX]);
        auto scale = dml::InputTensor(scope, kScale, input_descs[kScale]);
        auto mean = dml::InputTensor(
            scope,
            kReserveSpace1,
            input_descs[kReserveSpace1]);
        auto variance = dml::InputTensor(
            scope,
            kReserveSpace2,
            input_descs[kReserveSpace2]);

        DML_TENSOR_DATA_TYPE input_type = y_backprop.GetOutputDesc().dataType;

        // y_backprop, x, and x_backprop may be float16 or float32, but
        // everything else is always float32. If the types don't match, we need
        // to insert casts.
        bool is_cast_required = (input_type != scale.GetOutputDesc().dataType);

        // TODO: Remove when the intermediate accumulation bug in DML is fixed
        // TFDML #41739122
        is_cast_required = false;
        if (input_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            x = dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32);
            y_backprop = dml::Cast(y_backprop, DML_TENSOR_DATA_TYPE_FLOAT32);
        }

        if (is_cast_required)
        {
            mean = dml::Cast(mean, input_type);
            variance = dml::Cast(variance, input_type);
            scale = dml::Cast(scale, input_type);
        }

        dml::BatchNormalizationGradOutputs batch_norm_grad_op_outputs;
        if (init_helper->IsTraining())
        {
            batch_norm_grad_op_outputs = dml::BatchNormalizationTrainingGrad(
                x,
                y_backprop,
                mean,
                variance,
                scale,
                init_helper->GetEpsilon());
        }
        else
        {
            batch_norm_grad_op_outputs = dml::BatchNormalizationGrad(
                x,
                y_backprop,
                mean,
                variance,
                scale,
                init_helper->GetEpsilon());
        }

        auto x_backprop = batch_norm_grad_op_outputs.gradient;
        auto scale_backprop = batch_norm_grad_op_outputs.scaleGradient;
        auto offset_backprop = batch_norm_grad_op_outputs.biasGradient;

        // TODO: Remove when the intermediate accumulation bug in DML is fixed
        // TFDML #41739122
        if (input_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            x_backprop = dml::Cast(x_backprop, DML_TENSOR_DATA_TYPE_FLOAT16);
        }

        // If necessary, cast outputs to their required types
        if (is_cast_required)
        {
            scale_backprop =
                dml::Cast(scale_backprop, DML_TENSOR_DATA_TYPE_FLOAT32);
            offset_backprop =
                dml::Cast(offset_backprop, DML_TENSOR_DATA_TYPE_FLOAT32);
        }

        auto outputs = {
            x_backprop,
            scale_backprop,
            offset_backprop,
        };

        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlBatchGlobalNormGradKernel : public DmlKernel
{
    enum InputIndex
    {
        kT,
        kM,
        kV,
        kGamma,
        kBackProp,
    };

    enum OutputIndex
    {
        kDX,
        kDM,
        kDV,
        kDB,
        kDG,
    };

  public:
    using InitHelper = BatchGlobalNormGradInitializationHelper;
    explicit DmlBatchGlobalNormGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 5);

        float epsilon = init_helper->GetVarianceEpsilon();
        bool scale_after_normalization = init_helper->ScaleAfterNormalization();

        DmlKernelParams params;

        if (scale_after_normalization)
        {
            params.kernel_input_indices = {kT, kM, kV, kGamma, kBackProp};
        }
        else
        {
            params.kernel_input_indices = {kT, kM, kV, kBackProp};
        }
        params.kernel_output_indices = {0, 1, 2, 3, 4};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());

        const uint32_t back_prop_index =
            scale_after_normalization ? kBackProp : kBackProp - 1;
        auto input = dml::InputTensor(scope, kT, input_descs[kT]);
        auto mean = dml::InputTensor(scope, kM, input_descs[kM]);
        auto variance = dml::InputTensor(scope, kV, input_descs[kV]);
        auto back_prop = dml::InputTensor(
            scope,
            back_prop_index,
            input_descs[back_prop_index]);

        auto input_sizes = back_prop.GetOutputDesc().sizes;
        // The strides we need to set to broadcast C across an entire tensor
        dml::TensorDesc::Dimensions broadcast_c_strides = {
            /*N*/ 0,
            /*H*/ 0,
            /*W*/ 0,
            /*C*/ 1};

        // Formulae copied from tensorflow\core\kernels\batch_norm_op.h:
        // db = out_backprop
        //
        // dg = out_backprop * ((x - m) * rsqrt(v + epsilon))
        //
        // dv = sum_over_rest(out_backprop * gamma * (x - m)) *
        //      (-1/2) * (v + epsilon) ^ (-3/2)
        //
        // dm = sum_over_rest(out_backprop * gamma) * (-1 / rsqrt(v + epsilon))
        //
        // dx = out_backprop * (gamma * rsqrt(v + epsilon))

        auto variance_e = variance + epsilon;
        auto sqrt_variance_e = dml::Sqrt(variance_e);
        auto mean_bcast =
            dml::Reinterpret(mean, input_sizes, broadcast_c_strides);

        // scratch1 = rsqrt(v + epsilon)
        auto scratch1 = 1.0f / sqrt_variance_e;
        auto scratch1_bcast =
            dml::Reinterpret(scratch1, input_sizes, broadcast_c_strides);

        // scratch2 = sum_over_rest(out_backprop * (x - m))
        auto scratch2_s = back_prop * (input - mean_bcast);
        auto scratch2 =
            dml::Reduce(scratch2_s, DML_REDUCE_FUNCTION_SUM, {0, 1, 2});

        // scratch3 = - 1/2 * (var + epsilon) ^ (-3/2)
        auto scratch3 = -1.0f / 2 * scratch1 / variance_e;

        auto db = dml::Reduce(back_prop, DML_REDUCE_FUNCTION_SUM, {0, 1, 2});
        dml::Expression dx, dv, dg, dm;

        if (scale_after_normalization)
        {
            auto gamma = dml::InputTensor(scope, kGamma, input_descs[kGamma]);
            auto scratch1_gamma_bcast = dml::Reinterpret(
                gamma * scratch1,
                input_sizes,
                broadcast_c_strides);
            dx = back_prop * scratch1_gamma_bcast;
            dm = -db * scratch1 * gamma;
            dv = scratch2 * scratch3 * gamma;
            dg = scratch2 * scratch1;
        }
        else
        {
            dx = back_prop * scratch1_bcast;
            dm = -db * scratch1;
            dv = scratch2 * scratch3;
            dg = dml::ScalarTensor(
                scope,
                0.0f,
                variance.GetOutputDesc().sizes); // Gamma is not learned.
        }

        auto outputs = {
            dx,
            dm,
            dv,
            db,
            dg,
        };

        auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterFusedBatchNorm()
{
    using K = KernelDefinition<
        ops::FusedBatchNorm,
        DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>>;

    RegisterWithTypes<K, ops::FusedBatchNorm::Attribute::T, TF_FLOAT>();
}

void RegisterFusedBatchNormV2()
{
    using K = KernelDefinition<
        ops::FusedBatchNormV2,
        DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>>::
        WithTypeConstraint<ops::FusedBatchNormV2::Attribute::U, TF_FLOAT>;

    RegisterWithTypes<
        K,
        ops::FusedBatchNormV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterFusedBatchNormV3()
{
    using K = KernelDefinition<
        ops::FusedBatchNormV3,
        DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>>::
        WithTypeConstraint<ops::FusedBatchNormV3::Attribute::U, TF_FLOAT>;

    RegisterWithTypes<
        K,
        ops::FusedBatchNormV3::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterFusedBatchNormEx()
{
    using K = KernelDefinition<
        ops::_FusedBatchNormEx,
        DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>>::
        WithTypeConstraint<ops::_FusedBatchNormEx::Attribute::U, TF_FLOAT>;

    RegisterWithTypes<
        K,
        ops::_FusedBatchNormEx::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterBatchNormWithGlobalNormalization()
{
    using K = KernelDefinition<
        ops::BatchNormWithGlobalNormalization,
        DmlKernelWrapper<
            DmlBatchNormWithGlobalNormalizationKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::BatchNormWithGlobalNormalization::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterFusedBatchNormGrad()
{
    using K = KernelDefinition<
        ops::FusedBatchNormGrad,
        DmlKernelWrapper<
            DmlFusedBatchNormGradKernel,
            BatchNormGradShapeHelper>>;

    RegisterWithTypes<K, ops::FusedBatchNormGrad::Attribute::T, TF_FLOAT>();
}

void RegisterFusedBatchNormGradV2()
{
    using K = KernelDefinition<
        ops::FusedBatchNormGradV2,
        DmlKernelWrapper<
            DmlFusedBatchNormGradKernel,
            BatchNormGradShapeHelper>>::
        WithTypeConstraint<ops::FusedBatchNormGradV2::Attribute::U, TF_FLOAT>;

    RegisterWithTypes<
        K,
        ops::FusedBatchNormGradV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterFusedBatchNormGradV3()
{
    using K = KernelDefinition<
        ops::FusedBatchNormGradV3,
        DmlKernelWrapper<
            DmlFusedBatchNormGradKernel,
            BatchNormGradShapeHelper>>::
        WithTypeConstraint<ops::FusedBatchNormGradV3::Attribute::U, TF_FLOAT>;

    RegisterWithTypes<
        K,
        ops::FusedBatchNormGradV3::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterBatchNormWithGlobalNormalizationGrad()
{
    using K = KernelDefinition<
        ops::BatchNormWithGlobalNormalizationGrad,
        DmlKernelWrapper<DmlBatchGlobalNormGradKernel, BatchNormShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::BatchNormWithGlobalNormalizationGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_BatchNorm()
{
    RegisterFusedBatchNorm();
    RegisterFusedBatchNormV2();
    RegisterFusedBatchNormV3();
    RegisterFusedBatchNormEx();
    RegisterBatchNormWithGlobalNormalization();
    RegisterFusedBatchNormGrad();
    RegisterFusedBatchNormGradV2();
    RegisterFusedBatchNormGradV3();
    RegisterBatchNormWithGlobalNormalizationGrad();
}

} // namespace tfdml