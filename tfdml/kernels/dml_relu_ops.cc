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
#include "tfdml/runtime_adapter/bcast.h"

namespace tfdml
{

template <typename TAttributes = InitializationHelper::EmptyAttributes>
class LuGradInitHelper : public InitializationHelper
{
  public:
    using Attributes = TAttributes;

    LuGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        BCast bcast_helper(
            ctx->input(1).shape().dim_sizes(),
            ctx->input(0).shape().dim_sizes());
        feature_shape_ = TensorShape(bcast_helper.x_reshape());
        input_gradient_shape_ = TensorShape(bcast_helper.y_reshape());
        broadcasted_output_shape_ =
            BroadcastTensorShapes({feature_shape_, input_gradient_shape_});

        OP_REQUIRES(
            ctx,
            broadcasted_output_shape_.dims() <= kNcdhwDimensionCount,
            errors::InvalidArgument(
                "DML doesn't support more than ",
                kNcdhwDimensionCount,
                " dimensions for this operator, but ",
                broadcasted_output_shape_.dims(),
                " were provided."));
    }

    const TensorShape& GetBroadcastedOutputShape() const
    {
        return broadcasted_output_shape_;
    }

    const TensorShape& GetFeatureShape() const { return feature_shape_; }

    const TensorShape& GetInputGradientShape() const
    {
        return input_gradient_shape_;
    }

  private:
    TensorShape feature_shape_;
    TensorShape input_gradient_shape_;
    TensorShape broadcasted_output_shape_;
};

class DmlReluKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlReluKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        auto num_elements =
            static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());
        uint32_t tensor_sizes[4] = {1, 1, 1, num_elements};

        auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
        DmlTensorInfo tensor_info = {};
        tensor_info.kernel_index = 0;
        tensor_info.desc = DmlTensorDesc{data_type, tensor_sizes};

        DmlKernelTensors tensors = {};
        tensors.supports_in_place_execution = true;
        tensors.inputs = {tensor_info};
        tensors.outputs = {tensor_info};

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
        relu_desc.InputTensor = &input_descs[0];
        relu_desc.OutputTensor = output_descs.data();

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ACTIVATION_RELU, &relu_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlReluIntKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlReluIntKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        auto num_elements =
            static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());
        uint32_t tensor_sizes[4] = {1, 1, 1, num_elements};

        auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
        DmlTensorInfo tensor_info = {};
        tensor_info.kernel_index = 0;
        tensor_info.desc = DmlTensorDesc{data_type, tensor_sizes};

        DmlKernelTensors tensors = {};
        tensors.supports_in_place_execution = true;
        tensors.inputs = {tensor_info};
        tensors.outputs = {tensor_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);
        auto zero = dml::ZeroTensor(
            scope,
            input.GetOutputDesc().dataType,
            input.GetOutputDesc().sizes);

        auto result = dml::If(input < zero, zero, input);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

// Base CRTP class for linear unit (LU) grad ops: ReluGrad, SeluGrad, etc.
template <typename Impl, typename TInitHelper = LuGradInitHelper<>>
class DmlLUGradKernel : public DmlKernel
{
  public:
    using InitHelper = TInitHelper;

    explicit DmlLUGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        const TensorShape& feature_shape = init_helper->GetFeatureShape();
        const TensorShape& input_gradient_shape =
            init_helper->GetInputGradientShape();
        const TensorShape& output_shape =
            init_helper->GetBroadcastedOutputShape();

        DmlTensorInfo feature_tensor;
        feature_tensor.kernel_index = 1;
        feature_tensor.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(1),
            feature_shape,
            feature_shape);

        DmlTensorInfo input_gradient_tensor;
        input_gradient_tensor.kernel_index = 0;
        input_gradient_tensor.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_gradient_shape,
            input_gradient_shape);

        DmlTensorInfo output_tensor;
        output_tensor.kernel_index = 0;
        output_tensor.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);

        DmlKernelTensors tensors = {};
        tensors.supports_in_place_execution = true;
        tensors.inputs = {feature_tensor, input_gradient_tensor};
        tensors.outputs = {output_tensor};

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        static_cast<Impl*>(this)->Init(
            ctx,
            init_helper,
            std::move(tensors),
            input_descs[0],
            input_descs[1],
            output_descs[0]);
    }
};

class DmlReluGradKernel : public DmlLUGradKernel<DmlReluGradKernel>
{
  public:
    using InitHelper = LuGradInitHelper<>;

    explicit DmlReluGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlLUGradKernel<DmlReluGradKernel>(ctx, init_helper)
    {
    }

    void Init(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper,
        DmlKernelTensors&& tensors,
        const DML_TENSOR_DESC& gradient_desc,
        const DML_TENSOR_DESC& feature_desc,
        const DML_TENSOR_DESC& output_desc)
    {
        DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC relu_grad_desc = {};
        relu_grad_desc.InputTensor = &feature_desc;
        relu_grad_desc.InputGradientTensor = &gradient_desc;
        relu_grad_desc.OutputGradientTensor = &output_desc;

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ACTIVATION_RELU_GRAD,
            &relu_grad_desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <typename T>
class DmlRelu6GradKernel : public DmlLUGradKernel<DmlRelu6GradKernel<T>>
{
  public:
    using InitHelper = LuGradInitHelper<>;

    explicit DmlRelu6GradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlLUGradKernel<DmlRelu6GradKernel>(ctx, init_helper)
    {
    }

    void Init(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper,
        DmlKernelTensors&& tensors,
        const DML_TENSOR_DESC& gradient_desc,
        const DML_TENSOR_DESC& feature_desc,
        const DML_TENSOR_DESC& output_desc)
    {
        DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC clip_grad_desc;
        clip_grad_desc.InputTensor = &feature_desc;
        clip_grad_desc.InputGradientTensor = &gradient_desc;
        clip_grad_desc.OutputGradientTensor = &output_desc;
        clip_grad_desc.Min = 0.0f;
        clip_grad_desc.Max = 6.0f;

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD,
            &clip_grad_desc};

        DmlKernel::Initialize(ctx, std::move(tensors), op_desc);
    }
};

struct LeakyReluGradAttributes
{
    explicit LeakyReluGradAttributes(OpKernelConstruction* ctx)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha));
    }

    float alpha;
};

class LeakyReluGradInitHelper : public LuGradInitHelper<LeakyReluGradAttributes>
{
  public:
    using Attributes = LeakyReluGradAttributes;

    LeakyReluGradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : LuGradInitHelper<LeakyReluGradAttributes>(ctx, attr),
          alpha_(attr->alpha)
    {
    }

    float GetAlpha() const { return alpha_; }

  private:
    float alpha_;
};

class DmlLeakyReluGradKernel
    : public DmlLUGradKernel<DmlLeakyReluGradKernel, LeakyReluGradInitHelper>
{
  public:
    using InitHelper = LeakyReluGradInitHelper;

    explicit DmlLeakyReluGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlLUGradKernel<DmlLeakyReluGradKernel, InitHelper>(ctx, init_helper)
    {
    }

    void Init(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper,
        DmlKernelTensors&& tensors,
        const DML_TENSOR_DESC& gradient_desc,
        const DML_TENSOR_DESC& feature_desc,
        const DML_TENSOR_DESC& output_desc)
    {
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto feature = dml::InputTensor(scope, 0, feature_desc);
        auto gradient = dml::InputTensor(scope, 1, gradient_desc);

        DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
        const auto& feature_sizes = feature.GetOutputDesc().sizes;
        auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);
        float alpha = init_helper->GetAlpha();
        auto result = dml::If(feature > zero, gradient, gradient * alpha);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlEluGradKernel : public DmlLUGradKernel<DmlEluGradKernel>
{
  public:
    using InitHelper = LuGradInitHelper<>;

    explicit DmlEluGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlLUGradKernel<DmlEluGradKernel>(ctx, init_helper)
    {
    }

    void Init(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper,
        DmlKernelTensors&& tensors,
        const DML_TENSOR_DESC& gradient_desc,
        const DML_TENSOR_DESC& feature_desc,
        const DML_TENSOR_DESC& output_desc)
    {
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto feature = dml::InputTensor(scope, 0, feature_desc);
        auto gradient = dml::InputTensor(scope, 1, gradient_desc);

        DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
        const auto& feature_sizes = feature.GetOutputDesc().sizes;

        auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

        auto result =
            dml::If(feature < zero, (feature + 1.0f) * gradient, gradient);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlSeluGradKernel : public DmlLUGradKernel<DmlSeluGradKernel>
{
  public:
    using InitHelper = LuGradInitHelper<>;

    explicit DmlSeluGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlLUGradKernel<DmlSeluGradKernel>(ctx, init_helper)
    {
    }

    void Init(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper,
        DmlKernelTensors&& tensors,
        const DML_TENSOR_DESC& gradient_desc,
        const DML_TENSOR_DESC& feature_desc,
        const DML_TENSOR_DESC& output_desc)
    {
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto feature = dml::InputTensor(scope, 0, feature_desc);
        auto gradient = dml::InputTensor(scope, 1, gradient_desc);

        DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
        const auto& feature_sizes = feature.GetOutputDesc().sizes;

        auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

        constexpr float scale = 1.0507009873554804934193349852946f;
        constexpr float scale_alpha = 1.7580993408473768599402175208123f;

        auto result = dml::If(
            feature < zero,
            gradient * (feature + scale_alpha),
            gradient * scale);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void Relu()
{
    using float_kernel = KernelDefinition<
        ops::Relu,
        DmlKernelWrapper<DmlReluKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        float_kernel,
        ops::Relu::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    // TODO: Remove the specialized int implementation once DML supports all
    // datatypes
    // TFDML #41313814
    using int64_kernel = KernelDefinition<
        ops::Relu,
        DmlKernelWrapper<DmlReluIntKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        int64_kernel,
        ops::Relu::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32,
        TF_UINT64>();
}

void ReluGrad()
{
    using K = KernelDefinition<
        ops::ReluGrad,
        DmlKernelWrapper<DmlReluGradKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::ReluGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void Relu6Grad()
{
    using half_kernel = KernelDefinition<
        ops::Relu6Grad,
        DmlKernelWrapper<
            DmlRelu6GradKernel<Eigen::half>,
            GetOutputShapeAsInputShapeHelper>>;

    using float_kernel = KernelDefinition<
        ops::Relu6Grad,
        DmlKernelWrapper<
            DmlRelu6GradKernel<float>,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<half_kernel, ops::Relu6Grad::Attribute::T, TF_HALF>();
    RegisterWithTypes<float_kernel, ops::Relu6Grad::Attribute::T, TF_FLOAT>();
}

void LeakyReluGrad()
{
    using K = KernelDefinition<
        ops::LeakyReluGrad,
        DmlKernelWrapper<
            DmlLeakyReluGradKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::LeakyReluGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterEluGrad()
{
    using K = KernelDefinition<
        ops::EluGrad,
        DmlKernelWrapper<DmlEluGradKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::EluGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterSeluGrad()
{
    using K = KernelDefinition<
        ops::SeluGrad,
        DmlKernelWrapper<DmlSeluGradKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::SeluGrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterKernels_Relu()
{
    Relu();
    ReluGrad();
    Relu6Grad();
    LeakyReluGrad();
    RegisterEluGrad();
    RegisterSeluGrad();
}

} // namespace tfdml