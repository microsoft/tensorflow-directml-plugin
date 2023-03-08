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
#include <limits>

namespace tfdml
{

using Microsoft::WRL::ComPtr;

static absl::InlinedVector<TensorShape, 2> GetCollapsedShapes(
    OpKernelContext* ctx)
{
    if (ctx->num_inputs() == 1)
    {
        return {TensorShape({ctx->input(0).NumElements()})};
    }

    absl::InlinedVector<TensorShape, 2> shapes;

    // Shape collapsing for more than 2 inputs is not implemented
    if (ctx->num_inputs() > 2)
    {
        for (uint32_t i = 0; i < ctx->num_inputs(); ++i)
        {
            shapes.push_back(ctx->input(i).shape());
        }

        return shapes;
    }

    BCast bcast_helper(
        ctx->input(0).shape().dim_sizes(),
        ctx->input(1).shape().dim_sizes());

    shapes.emplace_back(bcast_helper.x_reshape());
    shapes.emplace_back(bcast_helper.y_reshape());

    return shapes;
}

template <uint32_t max_dim_count>
class ElementWiseInitHelper : public GetBroadcastedOutputShapeHelper::InitHelper
{
  public:
    struct Attributes
        : public GetBroadcastedOutputShapeHelper::InitHelper::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : GetBroadcastedOutputShapeHelper::InitHelper::Attributes(ctx)
        {
        }
    };

    ElementWiseInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : GetBroadcastedOutputShapeHelper::InitHelper(ctx, attr)
    {
        collapsed_input_shapes_ = GetCollapsedShapes(ctx);
        collapsed_output_shape_ =
            BroadcastTensorShapes(collapsed_input_shapes_);

        OP_REQUIRES(
            ctx,
            collapsed_output_shape_.dims() <= max_dim_count,
            errors::InvalidArgument(
                "DML doesn't support more than ",
                max_dim_count,
                " dimensions for this operator, but ",
                collapsed_output_shape_.dims(),
                " were provided."));
    }

    absl::Span<const TensorShape> GetCollapsedInputShapes() const
    {
        return collapsed_input_shapes_;
    }

    const TensorShape& GetCollapsedOutputShape() const
    {
        return collapsed_output_shape_;
    }

  private:
    absl::InlinedVector<TensorShape, 2> collapsed_input_shapes_;
    TensorShape collapsed_output_shape_;
};

static DmlKernelTensors CreateKernelTensors(
    DmlKernelConstruction* ctx,
    absl::Span<const TensorShape> input_shapes,
    const TensorShape& output_shape,
    bool supports_in_place_execution)
{
    DmlKernelTensors tensors;
    tensors.supports_in_place_execution = supports_in_place_execution;

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
    {
        DmlTensorInfo input;
        input.kernel_index = i;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(i),
            output_shape,
            input_shapes[i]);

        tensors.inputs.push_back(std::move(input));
    }

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0),
        output_shape,
        output_shape);

    tensors.outputs = {output};

    return tensors;
}

template <
    typename Functor,
    uint32_t max_dim_count,
    bool supports_in_place_execution>
class DmlBinaryWithZeroKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<max_dim_count>;

    explicit DmlBinaryWithZeroKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        auto input_shapes = init_helper->GetCollapsedInputShapes();
        const TensorShape& output_shape =
            init_helper->GetCollapsedOutputShape();

        DmlKernelTensors tensors = CreateKernelTensors(
            ctx,
            input_shapes,
            output_shape,
            supports_in_place_execution);
        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto x = dml::InputTensor(scope, 0, inputs[0]);
        auto y = dml::InputTensor(scope, 1, inputs[1]);
        auto zero = dml::ZeroTensor(
            scope,
            x.GetOutputDesc().dataType,
            x.GetOutputDesc().sizes);

        Functor f;
        auto result = f(zero, x, y);

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <
    typename ExpressionFunctor,
    uint32_t max_dim_count,
    bool supports_in_place_execution>
class DmlCompositeBinaryKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<max_dim_count>;

    explicit DmlCompositeBinaryKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        auto input_shapes = init_helper->GetCollapsedInputShapes();
        const TensorShape& output_shape =
            init_helper->GetCollapsedOutputShape();

        DmlKernelTensors tensors = CreateKernelTensors(
            ctx,
            input_shapes,
            output_shape,
            supports_in_place_execution);
        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto x = dml::InputTensor(scope, 0, inputs[0]);
        auto y = dml::InputTensor(scope, 1, inputs[1]);

        ExpressionFunctor expression;
        auto result = expression(x, y);

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlMaxActivationKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

    explicit DmlMaxActivationKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        const TensorShape& input_shape = ctx->GetInputTensorShape(0);

        int batch_size = 1;
        int logits_size = input_shape.dim_size(input_shape.dims() - 1);

        // DML doesn't support tensors with rank > 2 for the max activation
        // functions, so collapse all the batch dimensions together
        for (int i = 0; i < input_shape.dims() - 1; ++i)
        {
            batch_size *= input_shape.dim_size(i);
        }

        TensorShape dml_tensor_shape;
        dml_tensor_shape.AddDim(batch_size);
        dml_tensor_shape.AddDim(logits_size);

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            dml_tensor_shape,
            dml_tensor_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            dml_tensor_shape,
            dml_tensor_shape);

        DmlKernelTensors tensors;
        tensors.supports_in_place_execution = true;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto output_descs = GetDmlTensorDescs(tensors.outputs);

        DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {};
        op_specific_desc.InputTensor = input_descs.data();
        op_specific_desc.OutputTensor = output_descs.data();

        DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <typename ExpressionFunctor, bool supports_in_place_execution>
class DmlCompositeUnaryKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

    explicit DmlCompositeUnaryKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
        DmlKernelTensors tensors = CreateKernelTensors(
            ctx,
            {tensor_shape},
            tensor_shape,
            supports_in_place_execution);

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto x = dml::InputTensor(scope, 0, inputs[0]);

        ExpressionFunctor expression;
        auto result = expression(x);

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlClipByValueKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

    explicit DmlClipByValueKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelParams params;
        params.supports_in_place_execution = true;

        // Broadcast inputs to match output shape
        params.input_shape = ctx->GetOutputTensorShape(0);

        // The DML operator takes fewer inputs than the TF kernel receives, so
        // we need to explicitly specify the kernel indices. In this case, the
        // DML op takes a single input which corresponds to the 0th input on the
        // kernel.
        params.kernel_input_indices = {0};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto input_type = ctx->GetInputDataType(0);

        // Min/max are supplied as tensors for ClipByValue, which are required
        // to be constant CPU inputs
        const Tensor& min_tensor = ctx->GetConstantInputTensor(1);
        const Tensor& max_tensor = ctx->GetConstantInputTensor(2);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;

        // Since int64 to float is a narrowing conversion, we use max/min
        // instead to preserve the clipping values
        if (input_type == TF_INT64)
        {
            dml::TensorDesc::Dimensions input_dims =
                input.GetOutputDesc().sizes;
            auto min_exp = dml::ScalarTensor<int64_t>(
                scope,
                min_tensor.base<int64_t>()[0],
                input_dims);
            auto max_exp = dml::ScalarTensor<int64_t>(
                scope,
                max_tensor.base<int64_t>()[0],
                input_dims);
            result = dml::Max(input, min_exp);
            result = dml::Min(result, max_exp);
        }
        else
        {
            result = dml::Clip(
                input,
                min_tensor.base<float>()[0],
                max_tensor.base<float>()[0]);
        }

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlSeluKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

    explicit DmlSeluKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelParams params;
        params.supports_in_place_execution = true;

        // Broadcast inputs to match output shape
        params.input_shape = ctx->GetOutputTensorShape(0);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC selu_desc = {
            &inputs[0],
            outputs.data(),
            1.67326319217681884765625f, // alpha
            1.05070102214813232421875f  // gamma
        };

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ACTIVATION_SCALED_ELU,
            &selu_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class LeakyReluInitHelper : public ElementWiseInitHelper<UINT32_MAX>
{
  public:
    struct Attributes : public ElementWiseInitHelper<UINT32_MAX>::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : ElementWiseInitHelper<UINT32_MAX>::Attributes(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha));
        }

        float alpha;
    };

    LeakyReluInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : ElementWiseInitHelper<UINT32_MAX>(ctx, attr),
          alpha_(attr->alpha)
    {
    }

    float GetAlpha() const { return alpha_; }

  private:
    float alpha_;
};

class DmlLeakyReluKernel : public DmlKernel
{
  public:
    using InitHelper = LeakyReluInitHelper;

    explicit DmlLeakyReluKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelParams params;
        params.supports_in_place_execution = true;

        // Broadcast inputs to match output shape
        params.input_shape = ctx->GetOutputTensorShape(0);

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leaky_relu_desc = {
            &inputs[0],
            outputs.data(),
            init_helper->GetAlpha()};

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ACTIVATION_LEAKY_RELU,
            &leaky_relu_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class ApproximateEqualInitHelper
    : public ElementWiseInitHelper<kBinaryCwiseOpMaxDimCount>
{
  public:
    struct Attributes
        : public ElementWiseInitHelper<kBinaryCwiseOpMaxDimCount>::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : ElementWiseInitHelper<kBinaryCwiseOpMaxDimCount>::Attributes(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("tolerance", &tolerance));
        }

        float tolerance;
    };

    ApproximateEqualInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : ElementWiseInitHelper<kBinaryCwiseOpMaxDimCount>(ctx, attr),
          tolerance_(attr->tolerance)
    {
        const Tensor& x_input = ctx->input(0);
        const Tensor& y_input = ctx->input(1);
        OP_REQUIRES(
            ctx,
            x_input.shape() == y_input.shape(),
            errors::InvalidArgument(
                "x and y must be of the same shape. ",
                "x shape: ",
                x_input.shape().DebugString(),
                ". y shape: ",
                y_input.shape().DebugString()));
    }

    float GetTolerance() const { return tolerance_; }

  private:
    float tolerance_;
};

template <typename T>
class DmlApproximateEqualKernel : public DmlKernel
{
  public:
    using InitHelper = ApproximateEqualInitHelper;

    explicit DmlApproximateEqualKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        auto input_shapes = init_helper->GetCollapsedInputShapes();
        const TensorShape& output_shape =
            init_helper->GetCollapsedOutputShape();

        DmlKernelTensors tensors =
            CreateKernelTensors(ctx, input_shapes, output_shape, false);
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto x = dml::InputTensor(scope, 0, inputs[0]);
        auto y = dml::InputTensor(scope, 1, inputs[1]);

        auto tolerance_tensor = dml::ScalarTensor<T>(
            scope,
            TfTensorTypeTraits<T>::FromFloat(init_helper->GetTolerance()),
            x.GetOutputDesc().sizes);

        auto result = dml::Abs(x - y) < tolerance_tensor;

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlBitwiseNotKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlBitwiseNotKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        auto num_elements =
            static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());

        auto dtype = ctx->GetInputDataType(0);
        assert(dtype == ctx->GetOutputDataType(0));

        std::array<uint32_t, 4> sizes = {1, 1, 1, num_elements};

        DmlTensorInfo in;
        in.kernel_index = 0;
        in.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
        in.desc.ForceUnsignedDataType();
        auto in_desc = in.desc.GetDmlDesc();

        DmlTensorInfo out;
        out.kernel_index = 0;
        out.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
        out.desc.ForceUnsignedDataType();
        auto out_desc = out.desc.GetDmlDesc();

        DmlKernelTensors tensors;
        tensors.supports_in_place_execution = true;
        tensors.inputs = {in};
        tensors.outputs = {out};

        DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC desc = {};
        desc.InputTensor = &in_desc;
        desc.OutputTensor = &out_desc;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_BIT_NOT, &desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlBinaryBitwiseKernel : public DmlKernel
{
  public:
    using InitHelper = ElementWiseInitHelper<kBinaryCwiseOpMaxDimCount>;

    explicit DmlBinaryBitwiseKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        auto input_shapes = init_helper->GetCollapsedInputShapes();
        const TensorShape& output_shape =
            init_helper->GetCollapsedOutputShape();

        DmlKernelTensors tensors =
            CreateKernelTensors(ctx, input_shapes, output_shape, true);

        // DML only supports unsigned types, but sign doesn't matter for
        // bitwise.
        tensors.inputs[0]->desc.ForceUnsignedDataType();
        tensors.inputs[1]->desc.ForceUnsignedDataType();
        tensors.outputs[0]->desc.ForceUnsignedDataType();

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_OPERATOR_SPECIFIC_DESC desc = {
            &inputs[0],
            &inputs[1],
            outputs.data(),
        };

        DML_OPERATOR_DESC op_desc = {op_type, &desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

class DmlBitCountKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlBitCountKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        auto num_elements =
            static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());

        std::array<uint32_t, 4> sizes = {1, 1, 1, num_elements};

        DmlTensorInfo in;
        in.kernel_index = 0;
        in.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), sizes, sizes);
        in.desc.ForceUnsignedDataType();
        auto in_desc = in.desc.GetDmlDesc();

        DmlTensorInfo out;
        out.kernel_index = 0;
        out.desc =
            DmlTensorDesc::Create(ctx->GetOutputDataType(0), sizes, sizes);
        out.desc.ForceUnsignedDataType();
        auto out_desc = out.desc.GetDmlDesc();

        DmlKernelTensors tensors;
        tensors.supports_in_place_execution = true;
        tensors.inputs = {in};
        tensors.outputs = {out};

        DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC desc = {};
        desc.InputTensor = &in_desc;
        desc.OutputTensor = &out_desc;

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ELEMENT_WISE_BIT_COUNT,
            &desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

struct DmlDivNoNanFunctor
{
    dml::Expression operator()(
        dml::Expression zero,
        dml::Expression x,
        dml::Expression y)
    {
        return dml::If(y == zero, zero, x / y);
    }
};

struct DmlMulNoNanFunctor
{
    dml::Expression operator()(
        dml::Expression zero,
        dml::Expression x,
        dml::Expression y)
    {
        return dml::If(y == zero, zero, x * y);
    }
};

struct DmlXdivyFunctor
{
    dml::Expression operator()(
        dml::Expression zero,
        dml::Expression x,
        dml::Expression y)
    {
        return dml::If(x == zero, zero, x / y);
    }
};

struct DmlXlogyFunctor
{
    dml::Expression operator()(
        dml::Expression zero,
        dml::Expression x,
        dml::Expression y)
    {
        return dml::If(x == zero, zero, x * dml::Log(y));
    }
};

#define REGISTER_DML_COMPOSITE_BINARY_STRUCT(                                  \
    opName,                                                                    \
    op,                                                                        \
    expression,                                                                \
    max_dim_count,                                                             \
    supports_in_place_execution)                                               \
    struct Dml##opName##Functor                                                \
    {                                                                          \
        dml::Expression operator()(dml::Expression x, dml::Expression y)       \
        {                                                                      \
            return (expression);                                               \
        }                                                                      \
    };                                                                         \
    using K_##opName = KernelDefinition<                                       \
        op,                                                                    \
        DmlKernelWrapper<                                                      \
            DmlCompositeBinaryKernel<                                          \
                Dml##opName##Functor,                                          \
                max_dim_count,                                                 \
                supports_in_place_execution>,                                  \
            GetBroadcastedOutputShapeHelper>>;

#define REGISTER_DML_COMPOSITE_UNARY_STRUCT(                                   \
    opName,                                                                    \
    op,                                                                        \
    expression,                                                                \
    supports_in_place_execution)                                               \
    struct Dml##opName##Functor                                                \
    {                                                                          \
        dml::Expression operator()(dml::Expression x) { return (expression); } \
    };                                                                         \
    using K_##opName = KernelDefinition<                                       \
        op,                                                                    \
        DmlKernelWrapper<                                                      \
            DmlCompositeUnaryKernel<                                           \
                Dml##opName##Functor,                                          \
                supports_in_place_execution>,                                  \
            GetBroadcastedOutputShapeHelper>>;

static void RegisterAbs()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Abs, ops::Abs, dml::Abs(x), true)

    RegisterWithTypes<
        K_Abs,
        ops::Abs::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterAcos()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Acos, ops::Acos, dml::ACos(x), true)

    RegisterWithTypes<K_Acos, ops::Acos::Attribute::T, TF_FLOAT>();
}

static void RegisterAcosh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Acosh, ops::Acosh, dml::ACosh(x), true)

    RegisterWithTypes<K_Acosh, ops::Acosh::Attribute::T, TF_FLOAT>();
}

static void RegisterAdd()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        AddUint8,
        ops::Add,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) +
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
            DML_TENSOR_DATA_TYPE_UINT8),
        8,
        false)
    K_AddUint8::template WithTypeConstraint<ops::Add::Attribute::T, TF_UINT8>::
        Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Add, ops::Add, x + y, 8, true)

    RegisterWithTypes<
        K_Add,
        ops::Add::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterAddV2()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        AddV2Uint8,
        ops::AddV2,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) +
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
            DML_TENSOR_DATA_TYPE_UINT8),
        8,
        false)
    K_AddV2Uint8::template WithTypeConstraint<
        ops::AddV2::Attribute::T,
        TF_UINT8>::Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(AddV2, ops::AddV2, x + y, 8, true)
    RegisterWithTypes<
        K_AddV2,
        ops::AddV2::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterApproximateEqual()
{
    using half_kernel = KernelDefinition<
        ops::ApproximateEqual,
        DmlKernelWrapper<
            DmlApproximateEqualKernel<Eigen::half>,
            GetBroadcastedOutputShapeHelper>>;

    using float_kernel = KernelDefinition<
        ops::ApproximateEqual,
        DmlKernelWrapper<
            DmlApproximateEqualKernel<float>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        half_kernel,
        ops::ApproximateEqual::Attribute::T,
        TF_HALF>();
    RegisterWithTypes<
        float_kernel,
        ops::ApproximateEqual::Attribute::T,
        TF_FLOAT>();
}

static void RegisterAsin()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Asin, ops::Asin, dml::ASin(x), true)

    RegisterWithTypes<K_Asin, ops::Asin::Attribute::T, TF_FLOAT>();
}

static void RegisterAsinh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Asinh, ops::Asinh, dml::ASinh(x), true)

    RegisterWithTypes<K_Asinh, ops::Asinh::Attribute::T, TF_FLOAT>();
}

static void RegisterAtan()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Atan, ops::Atan, dml::ATan(x), true)

    RegisterWithTypes<K_Atan, ops::Atan::Attribute::T, TF_FLOAT>();
}

static void RegisterAtan2()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        Atan2,
        ops::Atan2,
        dml::ATanYX(x, y),
        8,
        true)

    RegisterWithTypes<K_Atan2, ops::Atan2::Attribute::T, TF_FLOAT>();
}

static void RegisterAtanh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Atanh, ops::Atanh, dml::ATanh(x), true)

    RegisterWithTypes<K_Atanh, ops::Atanh::Attribute::T, TF_FLOAT>();
}

static void RegisterBitwiseAnd()
{
    using K = KernelDefinition<
        ops::BitwiseAnd,
        DmlKernelWrapper<
            DmlBinaryBitwiseKernel<
                DML_OPERATOR_ELEMENT_WISE_BIT_AND,
                DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::BitwiseAnd::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();
}

static void RegisterBitwiseOr()
{
    using K = KernelDefinition<
        ops::BitwiseOr,
        DmlKernelWrapper<
            DmlBinaryBitwiseKernel<
                DML_OPERATOR_ELEMENT_WISE_BIT_OR,
                DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::BitwiseOr::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();
}

static void RegisterBitwiseXor()
{
    using K = KernelDefinition<
        ops::BitwiseXor,
        DmlKernelWrapper<
            DmlBinaryBitwiseKernel<
                DML_OPERATOR_ELEMENT_WISE_BIT_XOR,
                DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::BitwiseXor::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();
}

static void RegisterCeil()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Ceil, ops::Ceil, dml::Ceil(x), true)

    RegisterWithTypes<K_Ceil, ops::Ceil::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterClipByValue()
{
    using K = KernelDefinition<
        ops::ClipByValue,
        DmlKernelWrapper<
            DmlClipByValueKernel,
            GetBroadcastedOutputShapeHelper>>::
        template WithHostMemoryArguments<
            ops::ClipByValue::Argument::clip_value_min>::
            template WithHostMemoryArguments<
                ops::ClipByValue::Argument::clip_value_max>;

    RegisterWithTypes<
        K,
        ops::ClipByValue::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8,
        TF_UINT16>();
}

static void RegisterCos()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Cos, ops::Cos, dml::Cos(x), true)

    RegisterWithTypes<K_Cos, ops::Cos::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterCosh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Cosh, ops::Cosh, dml::Cosh(x), true)

    RegisterWithTypes<K_Cosh, ops::Cosh::Attribute::T, TF_FLOAT>();
}

static void RegisterDiv()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Div, ops::Div, x / y, 8, true)

    RegisterWithTypes<
        K_Div,
        ops::Div::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_UINT8,
        TF_UINT16,
        TF_INT16>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        DivInt64,
        ops::Div,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) /
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
            DML_TENSOR_DATA_TYPE_INT64),
        8,
        false)

    RegisterWithTypes<K_DivInt64, ops::Div::Attribute::T, TF_INT64>();
}

static void RegisterDivNoNan()
{
    using K = KernelDefinition<
        ops::DivNoNan,
        DmlKernelWrapper<
            DmlBinaryWithZeroKernel<
                DmlDivNoNanFunctor,
                kNchwDimensionCount,
                false>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::DivNoNan::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterElu()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Elu,
        ops::Elu,
        dml::ActivationElu(x),
        true)

    RegisterWithTypes<K_Elu, ops::Elu::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterEqual()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Equal, ops::Equal, x == y, 8, true)

    RegisterWithTypes<
        K_Equal,
        ops::Equal::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterErf()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Erf, ops::Erf, dml::Erf(x), true)

    RegisterWithTypes<K_Erf, ops::Erf::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterErfc()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Erfc,
        ops::Erfc,
        1.0f - dml::Erf(x),
        true)

    RegisterWithTypes<K_Erfc, ops::Erfc::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterExp()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Exp, ops::Exp, dml::Exp(x), true)

    RegisterWithTypes<K_Exp, ops::Exp::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterExpm1()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Expm1,
        ops::Expm1,
        dml::Exp(x) - 1.0f,
        true)

    RegisterWithTypes<K_Expm1, ops::Expm1::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterFloor()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Floor, ops::Floor, dml::Floor(x), true)

    RegisterWithTypes<K_Floor, ops::Floor::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterFloorDiv()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        FloorDiv,
        ops::FloorDiv,
        dml::Floor(x / y),
        8,
        true)

    RegisterWithTypes<
        K_FloorDiv,
        ops::FloorDiv::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterFloorMod()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        FloorMod,
        ops::FloorMod,
        dml::ModulusFloor(x, y),
        8,
        true)

    RegisterWithTypes<
        K_FloorMod,
        ops::FloorMod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        FloorModInt64,
        ops::FloorMod,
        dml::Cast(
            dml::ModulusFloor(
                dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32),
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32)),
            DML_TENSOR_DATA_TYPE_INT64),
        8,
        false)

    RegisterWithTypes<K_FloorModInt64, ops::FloorMod::Attribute::T, TF_INT64>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        FloorModUInt64,
        ops::FloorMod,
        dml::Cast(
            dml::ModulusFloor(
                dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32),
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32)),
            DML_TENSOR_DATA_TYPE_UINT64),
        8,
        false)

    RegisterWithTypes<
        K_FloorModUInt64,
        ops::FloorMod::Attribute::T,
        TF_UINT64>();
}

static void RegisterGreater()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Greater, ops::Greater, x > y, 8, false)

    RegisterWithTypes<
        K_Greater,
        ops::Greater::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterGreaterEqual()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        GreaterEqual,
        ops::GreaterEqual,
        x >= y,
        8,
        false)

    RegisterWithTypes<
        K_GreaterEqual,
        ops::GreaterEqual::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterInv()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        InvInt64,
        ops::Inv,
        dml::Cast(
            dml::Recip(dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32)),
            DML_TENSOR_DATA_TYPE_INT64),
        false)

    RegisterWithTypes<K_InvInt64, ops::Inv::Attribute::T, TF_INT64>();

    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Inv, ops::Inv, dml::Recip(x), true)

    RegisterWithTypes<K_Inv, ops::Inv::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterInvGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        InvGrad,
        ops::InvGrad,
        (-y * x * x),
        8,
        false)

    RegisterWithTypes<
        K_InvGrad,
        ops::InvGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterInvert()
{
    using K = KernelDefinition<
        ops::Invert,
        DmlKernelWrapper<
            DmlBitwiseNotKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::Invert::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_INT64,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32,
        TF_UINT64>();
}

static void RegisterIsFinite()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        IsFinite,
        ops::IsFinite,
        !(dml::IsNaN(x) || dml::IsInfinity(x)),
        false)

    RegisterWithTypes<
        K_IsFinite,
        ops::IsFinite::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterIsInf()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        IsInf,
        ops::IsInf,
        dml::IsInfinity(x),
        false)

    RegisterWithTypes<K_IsInf, ops::IsInf::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterIsNan()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(IsNan, ops::IsNan, dml::IsNaN(x), false)

    RegisterWithTypes<K_IsNan, ops::IsNan::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterLeakyRelu()
{
    using K = KernelDefinition<
        ops::LeakyRelu,
        DmlKernelWrapper<DmlLeakyReluKernel, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::LeakyRelu::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterLeftShift()
{
    using K = KernelDefinition<
        ops::LeftShift,
        DmlKernelWrapper<
            DmlBinaryBitwiseKernel<
                DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT,
                DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::LeftShift::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();
}

static void RegisterLess()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Less, ops::Less, x < y, 8, false)

    RegisterWithTypes<
        K_Less,
        ops::Less::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterLessEqual()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        LessEqual,
        ops::LessEqual,
        x <= y,
        8,
        false)

    RegisterWithTypes<
        K_LessEqual,
        ops::LessEqual::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterLog()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Log, ops::Log, dml::Log(x), true)

    RegisterWithTypes<K_Log, ops::Log::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterLog1p()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Log1p,
        ops::Log1p,
        dml::Log(x, DML_SCALE_BIAS{1, 1}),
        true)

    RegisterWithTypes<K_Log1p, ops::Log1p::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterLogicalAnd()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        LogicalAnd,
        ops::LogicalAnd,
        dml::LogicalAnd(x, y),
        8,
        true)

    K_LogicalAnd::Register();
}

static void RegisterLogicalNot()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        LogicalNot,
        ops::LogicalNot,
        dml::LogicalNot(x),
        true)

    K_LogicalNot::Register();
}

static void RegisterLogicalOr()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        LogicalOr,
        ops::LogicalOr,
        dml::LogicalOr(x, y),
        8,
        true)

    K_LogicalOr::Register();
}

static void RegisterLogSoftmax()
{
    using K = KernelDefinition<
        ops::LogSoftmax,
        DmlKernelWrapper<
            DmlMaxActivationKernel<
                DML_OPERATOR_ACTIVATION_LOG_SOFTMAX,
                DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::LogSoftmax::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterMaximum()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        Maximum,
        ops::Maximum,
        dml::Max(x, y),
        8,
        true)

    RegisterWithTypes<
        K_Maximum,
        ops::Maximum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterMinimum()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        Minimum,
        ops::Minimum,
        dml::Min(x, y),
        8,
        true)

    RegisterWithTypes<
        K_Minimum,
        ops::Minimum::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterMod()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        Mod,
        ops::Mod,
        dml::ModulusTruncate(x, y),
        8,
        true)

    RegisterWithTypes<
        K_Mod,
        ops::Mod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32,
        TF_UINT64>();
}

static void RegisterMul()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        MulUint8,
        ops::Mul,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) *
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
            DML_TENSOR_DATA_TYPE_UINT8),
        8,
        false)
    K_MulUint8::template WithTypeConstraint<ops::Mul::Attribute::T, TF_UINT8>::
        Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        MulInt8,
        ops::Mul,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) *
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
            DML_TENSOR_DATA_TYPE_INT8),
        8,
        false)
    K_MulInt8::template WithTypeConstraint<ops::Mul::Attribute::T, TF_INT8>::
        Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        MulUint16,
        ops::Mul,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) *
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
            DML_TENSOR_DATA_TYPE_UINT16),
        8,
        false)
    K_MulUint16::template WithTypeConstraint<
        ops::Mul::Attribute::T,
        TF_UINT16>::Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        MulInt16,
        ops::Mul,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) *
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
            DML_TENSOR_DATA_TYPE_INT16),
        8,
        false)
    K_MulInt16::template WithTypeConstraint<ops::Mul::Attribute::T, TF_INT16>::
        Register();

    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Mul, ops::Mul, (x * y), 8, true)
    RegisterWithTypes<
        K_Mul,
        ops::Mul::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterMulNoNan()
{
    using K = KernelDefinition<
        ops::MulNoNan,
        DmlKernelWrapper<
            DmlBinaryWithZeroKernel<
                DmlMulNoNanFunctor,
                kNchwDimensionCount,
                false>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::MulNoNan::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterNeg()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Neg, ops::Neg, -x, true)

    RegisterWithTypes<
        K_Neg,
        ops::Neg::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterNotEqual()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        NotEqual,
        ops::NotEqual,
        x != y,
        8,
        true)

    RegisterWithTypes<
        K_NotEqual,
        ops::NotEqual::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_INT16,
        TF_INT64,
        TF_UINT8>();
}

static void RegisterPopulationCount()
{
    using K = KernelDefinition<
        ops::PopulationCount,
        DmlKernelWrapper<DmlBitCountKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::PopulationCount::Attribute::T,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_INT64,
        TF_UINT8,
        TF_UINT16>();
}

static void RegisterPow()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Pow, ops::Pow, dml::Pow(x, y), 8, true)
    RegisterWithTypes<K_Pow, ops::Pow::Attribute::T, TF_FLOAT, TF_HALF>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        PowInt64,
        ops::Pow,
        dml::Cast(
            dml::Pow(
                dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32),
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32)),
            DML_TENSOR_DATA_TYPE_INT64),
        8,
        false)
    RegisterWithTypes<K_PowInt64, ops::Pow::Attribute::T, TF_INT64>();
}

static void RegisterRealDiv()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(RealDiv, ops::RealDiv, x / y, 8, true)

    RegisterWithTypes<
        K_RealDiv,
        ops::RealDiv::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterReciprocal()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        ReciprocalInt64,
        ops::Reciprocal,
        dml::Cast(
            dml::Recip(dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32)),
            DML_TENSOR_DATA_TYPE_INT64),
        true)

    RegisterWithTypes<
        K_ReciprocalInt64,
        ops::Reciprocal::Attribute::T,
        TF_INT64>();

    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Reciprocal,
        ops::Reciprocal,
        dml::Recip(x),
        true)

    RegisterWithTypes<
        K_Reciprocal,
        ops::Reciprocal::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterReciprocalGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        ReciprocalGrad,
        ops::ReciprocalGrad,
        (-y * x * x),
        8,
        false)

    RegisterWithTypes<
        K_ReciprocalGrad,
        ops::ReciprocalGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterRelu6()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Relu6,
        ops::Relu6,
        dml::Clip(x, 0, 6),
        true)

    RegisterWithTypes<K_Relu6, ops::Relu6::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterRightShift()
{
    using K = KernelDefinition<
        ops::RightShift,
        DmlKernelWrapper<
            DmlBinaryBitwiseKernel<
                DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT,
                DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::RightShift::Attribute::T,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();
}

static void RegisterRint()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Rint, ops::Rint, dml::Round(x), true)

    RegisterWithTypes<K_Rint, ops::Rint::Attribute::T, TF_FLOAT>();
}

static void RegisterRound()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        RoundFloat,
        ops::Round,
        dml::Round(x),
        true)

    RegisterWithTypes<
        K_RoundFloat,
        ops::Round::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        RoundInt,
        ops::Round,
        dml::Identity(x),
        true)

    RegisterWithTypes<
        K_RoundInt,
        ops::Round::Attribute::T,
        TF_INT32,
        TF_INT64>();
}

static void RegisterRsqrt()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Rsqrt,
        ops::Rsqrt,
        1.0f / dml::Sqrt(x),
        true)

    RegisterWithTypes<K_Rsqrt, ops::Rsqrt::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterRsqrtGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        RsqrtGrad,
        ops::RsqrtGrad,
        (y * (-0.5f * x) * (x * x)),
        8,
        false)

    RegisterWithTypes<
        K_RsqrtGrad,
        ops::RsqrtGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSelu()
{
    using K = KernelDefinition<
        ops::Selu,
        DmlKernelWrapper<DmlSeluKernel, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Selu::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterSigmoid()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Sigmoid,
        ops::Sigmoid,
        dml::ActivationSigmoid(x),
        true)

    RegisterWithTypes<
        K_Sigmoid,
        ops::Sigmoid::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSigmoidGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SigmoidGrad,
        ops::SigmoidGrad,
        (y * x * (1 - x)),
        8,
        false)

    RegisterWithTypes<
        K_SigmoidGrad,
        ops::SigmoidGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSign()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Sign, ops::Sign, dml::Sign(x), true)

    RegisterWithTypes<
        K_Sign,
        ops::Sign::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterSin()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Sin, ops::Sin, dml::Sin(x), true)

    RegisterWithTypes<K_Sin, ops::Sin::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterSinh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Sinh, ops::Sinh, dml::Sinh(x), true)

    RegisterWithTypes<K_Sinh, ops::Sinh::Attribute::T, TF_FLOAT>();
}

static void RegisterSoftmax()
{
    using K = KernelDefinition<
        ops::Softmax,
        DmlKernelWrapper<
            DmlMaxActivationKernel<
                DML_OPERATOR_ACTIVATION_SOFTMAX,
                DML_ACTIVATION_SOFTMAX_OPERATOR_DESC>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Softmax::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterSoftplus()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Softplus,
        ops::Softplus,
        dml::ActivationSoftplus(x),
        true)

    RegisterWithTypes<
        K_Softplus,
        ops::Softplus::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSoftplusGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SoftplusGrad,
        ops::SoftplusGrad,
        x / (dml::Exp(-y) + 1),
        8,
        false)

    RegisterWithTypes<
        K_SoftplusGrad,
        ops::SoftplusGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSoftsign()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(
        Softsign,
        ops::Softsign,
        dml::ActivationSoftsign(x),
        true)

    RegisterWithTypes<
        K_Softsign,
        ops::Softsign::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSoftsignGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SoftsignGrad,
        ops::SoftsignGrad,
        x / dml::Pow(1 + dml::Abs(y), 2),
        8,
        false)

    RegisterWithTypes<
        K_SoftsignGrad,
        ops::SoftsignGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSqrt()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Sqrt, ops::Sqrt, dml::Sqrt(x), true)

    RegisterWithTypes<K_Sqrt, ops::Sqrt::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterSqrtGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SqrtGrad,
        ops::SqrtGrad,
        (y * 0.5f / x),
        8,
        false)

    RegisterWithTypes<
        K_SqrtGrad,
        ops::SqrtGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterSquare()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Square, ops::Square, (x * x), true)

    RegisterWithTypes<
        K_Square,
        ops::Square::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterSquaredDifference()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SquaredDifference,
        ops::SquaredDifference,
        dml::DifferenceSquare(x, y),
        8,
        true)

    RegisterWithTypes<
        K_SquaredDifference,
        ops::SquaredDifference::Attribute::T,
        TF_FLOAT,
        TF_HALF>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        SquaredDifferenceInt64,
        ops::SquaredDifference,
        (x - y) * (x - y),
        8,
        false)

    RegisterWithTypes<
        K_SquaredDifferenceInt64,
        ops::SquaredDifference::Attribute::T,
        TF_INT64>();
}

static void RegisterSub()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(Sub, ops::Sub, x - y, 8, true)

    RegisterWithTypes<
        K_Sub,
        ops::Sub::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64>();
}

static void RegisterTan()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Tan, ops::Tan, dml::Tan(x), true)

    RegisterWithTypes<K_Tan, ops::Tan::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterTanh()
{
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(Tanh, ops::Tanh, dml::Tanh(x), true)

    RegisterWithTypes<K_Tanh, ops::Tanh::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterTanhGrad()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TanhGrad,
        ops::TanhGrad,
        (y * (1 - x * x)),
        8,
        false)

    RegisterWithTypes<
        K_TanhGrad,
        ops::TanhGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

static void RegisterTruncateDiv()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TruncateDiv,
        ops::TruncateDiv,
        x / y,
        8,
        true)

    RegisterWithTypes<
        K_TruncateDiv,
        ops::TruncateDiv::Attribute::T,
        TF_UINT8,
        TF_UINT16,
        TF_INT16>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TruncateDivInt64,
        ops::TruncateDiv,
        dml::Cast(
            dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) /
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
            DML_TENSOR_DATA_TYPE_INT64),
        8,
        false)

    RegisterWithTypes<
        K_TruncateDivInt64,
        ops::TruncateDiv::Attribute::T,
        TF_INT64>();
}

static void RegisterTruncateMod()
{
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TruncateMod,
        ops::TruncateMod,
        dml::ModulusTruncate(x, y),
        8,
        true)

    RegisterWithTypes<
        K_TruncateMod,
        ops::TruncateMod::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT8,
        TF_INT16,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TruncateModInt64,
        ops::TruncateMod,
        dml::Cast(
            dml::ModulusTruncate(
                dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32),
                dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32)),
            DML_TENSOR_DATA_TYPE_INT64),
        8,
        true)

    RegisterWithTypes<
        K_TruncateModInt64,
        ops::TruncateMod::Attribute::T,
        TF_INT64>();

    // TODO: Revisit this and consider having a native int64 alternative
    // TFDML #41163316
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(
        TruncateModUInt64,
        ops::TruncateMod,
        dml::Cast(
            dml::ModulusTruncate(
                dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32),
                dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32)),
            DML_TENSOR_DATA_TYPE_UINT64),
        8,
        true)

    RegisterWithTypes<
        K_TruncateModUInt64,
        ops::TruncateMod::Attribute::T,
        TF_UINT64>();
}

static void RegisterXdivy()
{
    using K = KernelDefinition<
        ops::Xdivy,
        DmlKernelWrapper<
            DmlBinaryWithZeroKernel<
                DmlXdivyFunctor,
                kNchwDimensionCount,
                false>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Xdivy::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterXlogy()
{
    using K = KernelDefinition<
        ops::Xlogy,
        DmlKernelWrapper<
            DmlBinaryWithZeroKernel<
                DmlXlogyFunctor,
                kNchwDimensionCount,
                false>,
            GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Xlogy::Attribute::T, TF_FLOAT, TF_HALF>();
}

#undef REGISTER_DML_COMPOSITE_BINARY_STRUCT
#undef REGISTER_DML_COMPOSITE_UNARY_STRUCT

void RegisterKernels_Cwise()
{
    RegisterAbs();
    RegisterAcos();
    RegisterAcosh();
    RegisterAdd();
    RegisterAddV2();
    RegisterApproximateEqual();
    RegisterAsin();
    RegisterAsinh();
    RegisterAtan();
    RegisterAtan2();
    RegisterAtanh();
    RegisterBitwiseAnd();
    RegisterBitwiseOr();
    RegisterBitwiseXor();
    RegisterCeil();
    RegisterClipByValue();
    RegisterCos();
    RegisterCosh();
    RegisterDiv();
    RegisterDivNoNan();
    RegisterElu();
    RegisterEqual();
    RegisterErf();
    RegisterErfc();
    RegisterExp();
    RegisterExpm1();
    RegisterFloor();
    RegisterFloorDiv();
    RegisterFloorMod();
    RegisterGreater();
    RegisterGreaterEqual();
    RegisterInv();
    RegisterInvGrad();
    RegisterInvert();
    RegisterIsFinite();
    RegisterIsInf();
    RegisterIsNan();
    RegisterLeakyRelu();
    RegisterLeftShift();
    RegisterLess();
    RegisterLessEqual();
    RegisterLog();
    RegisterLog1p();
    RegisterLogicalAnd();
    RegisterLogicalNot();
    RegisterLogicalOr();
    RegisterLogSoftmax();
    RegisterMaximum();
    RegisterMinimum();
    RegisterMod();
    RegisterMul();
    RegisterMulNoNan();
    RegisterNeg();
    RegisterNotEqual();
    RegisterPopulationCount();
    RegisterPow();
    RegisterReciprocal();
    RegisterRealDiv();
    RegisterReciprocalGrad();
    RegisterRelu6();
    RegisterRightShift();
    RegisterRint();
    RegisterRound();
    RegisterRsqrt();
    RegisterRsqrtGrad();
    RegisterSelu();
    RegisterSigmoid();
    RegisterSigmoidGrad();
    RegisterSign();
    RegisterSin();
    RegisterSinh();
    RegisterSoftmax();
    RegisterSoftplus();
    RegisterSoftplusGrad();
    RegisterSoftsign();
    RegisterSoftsignGrad();
    RegisterSqrt();
    RegisterSqrtGrad();
    RegisterSquare();
    RegisterSquaredDifference();
    RegisterSub();
    RegisterTan();
    RegisterTanh();
    RegisterTanhGrad();
    RegisterTruncateDiv();
    RegisterTruncateMod();
    RegisterXdivy();
    RegisterXlogy();
}

} // namespace tfdml