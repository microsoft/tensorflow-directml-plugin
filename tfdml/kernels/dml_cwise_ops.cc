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

using Microsoft::WRL::ComPtr;

template <DML_OPERATOR_TYPE op_type>
static constexpr uint32_t GetMaxDimCount() {
  switch (op_type) {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
    case DML_OPERATOR_ELEMENT_WISE_ADD:
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
      return 5;
    default:
      return 4;
  }
}

static absl::InlinedVector<TensorShape, 2> GetCollapsedShapes(
    OpKernelContext* ctx) {
  if (ctx->num_inputs() == 1) {
    return {TensorShape({ctx->input(0).NumElements()})};
  }

  absl::InlinedVector<TensorShape, 2> shapes;

  // Shape collapsing for more than 2 inputs is not implemented
  if (ctx->num_inputs() > 2) {
    for (uint32_t i = 0; i < ctx->num_inputs(); ++i) {
      shapes.push_back(ctx->input(i).shape());
    }

    return shapes;
  }

  BCast bcast_helper(ctx->input(0).shape().dim_sizes(),
                     ctx->input(1).shape().dim_sizes());

  shapes.emplace_back(bcast_helper.x_reshape());
  shapes.emplace_back(bcast_helper.y_reshape());

  return shapes;
}

template <uint32_t max_dim_count>
class ElementWiseInitHelper
    : public GetBroadcastedOutputShapeHelper::InitHelper {
 public:
  struct Attributes
      : public GetBroadcastedOutputShapeHelper::InitHelper::Attributes {
    explicit Attributes(OpKernelConstruction* ctx)
        : GetBroadcastedOutputShapeHelper::InitHelper::Attributes(ctx) {}
  };

  ElementWiseInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr)
      : GetBroadcastedOutputShapeHelper::InitHelper(ctx, attr) {
    collapsed_input_shapes_ = GetCollapsedShapes(ctx);
    collapsed_output_shape_ = BroadcastTensorShapes(collapsed_input_shapes_);

    OP_REQUIRES(ctx, collapsed_output_shape_.dims() <= max_dim_count,
                errors::InvalidArgument(
                    "DML doesn't support more than ", max_dim_count,
                    " dimensions for this operator, but ",
                    collapsed_output_shape_.dims(), " were provided."));
  }

  absl::Span<const TensorShape> GetCollapsedInputShapes() const {
    return collapsed_input_shapes_;
  }

  const TensorShape& GetCollapsedOutputShape() const {
    return collapsed_output_shape_;
  }

 private:
  absl::InlinedVector<TensorShape, 2> collapsed_input_shapes_;
  TensorShape collapsed_output_shape_;
};

static DmlKernelTensors CreateKernelTensors(
    DmlKernelConstruction* ctx, absl::Span<const TensorShape> input_shapes,
    const TensorShape& output_shape) {
  const auto tensor_layout =
      GetDmlTensorLayout(FORMAT_NCHW, output_shape.dims());

  DmlKernelTensors tensors;

  for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
    DmlTensorInfo input;
    input.kernel_index = i;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i), output_shape,
                                       input_shapes[i], tensor_layout);

    tensors.inputs.push_back(std::move(input));
  }

  DmlTensorInfo output;
  output.kernel_index = 0;
  output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                      output_shape, tensor_layout);

  tensors.outputs = {output};

  return tensors;
}

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlBinaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<GetMaxDimCount<op_type>()>;

  explicit DmlBinaryKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0)))
    {
        zero_outputs_ = true;
    }

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {
        &inputs[0],
        &inputs[1],
        outputs.data(),
    };

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {

    Tensor& output = ctx->GetOutputTensor(0);

    if (zero_outputs_) {
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
    }

    return DmlKernel::Compute(ctx);
  }

  private:
    bool zero_outputs_ = false;
};

template <typename Functor, uint32_t max_dim_count>
class DmlBinaryWithZeroKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<max_dim_count>;

  explicit DmlBinaryWithZeroKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0)))
    {
        zero_outputs_ = true;
    }

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);
    auto zero = dml::ZeroTensor(scope, x.GetOutputDesc().dataType,
                                x.GetOutputDesc().sizes);

    Functor f;
    auto result = f(zero, x, y);

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {
    Tensor& output = ctx->GetOutputTensor(0);
    if (zero_outputs_) {
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
    }
    return DmlKernel::Compute(ctx);
  }

  private:
    bool zero_outputs_ = false;
};

template <typename ExpressionFunctor, uint32_t max_dim_count>
class DmlCompositeUnaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlCompositeUnaryKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0)))
    {
        zero_outputs_ = true;
    }

    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, {tensor_shape}, tensor_shape);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);

    ExpressionFunctor expression;
    auto result = expression(x);

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {
    Tensor& output = ctx->GetOutputTensor(0);
    if (zero_outputs_) {
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
    }
    return DmlKernel::Compute(ctx);
  }

  private:
    bool zero_outputs_ = false;
};


template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC,
          int scale = 1, int bias = 0, int... constants>
class DmlUnaryScaleBiasKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlUnaryScaleBiasKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0)))
    {
        zero_outputs_ = true;
    }

    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, {tensor_shape}, tensor_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_SCALE_BIAS scale_bias = {scale, bias};
    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {
        &inputs[0],
        &outputs[0],
        &scale_bias,
        constants...,
    };

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {
    Tensor& output = ctx->GetOutputTensor(0);
    if (zero_outputs_) {
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
    }
    return DmlKernel::Compute(ctx);
  }

  private:
    bool zero_outputs_ = false;
};

struct DmlDivNoNanFunctor {
  dml::Expression operator()(dml::Expression zero, dml::Expression x,
                             dml::Expression y) {
    return dml::If(y == zero, zero, x / y);
  }
};

struct DmlRsqrtFunctor {
    dml::Expression operator()(dml::Expression x) { return (1.0f / dml::Sqrt(x)); }
};

static void RegisterAdd()
{
    using K = KernelDefinition<
        ops::Add,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_ADD, DML_ELEMENT_WISE_ADD_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Add::Attribute::T, TF_FLOAT, TF_HALF, TF_INT64, TF_UINT32, TF_UINT64>();
}

static void RegisterAddV2()
{
    using K = KernelDefinition<
        ops::AddV2,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_ADD, DML_ELEMENT_WISE_ADD_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::AddV2::Attribute::T, TF_FLOAT, TF_HALF, TF_INT64, TF_UINT32, TF_UINT64>();
}

static void RegisterDivNoNan()
{
    using K = KernelDefinition<
        ops::DivNoNan,
        DmlKernelWrapper<DmlBinaryWithZeroKernel<DmlDivNoNanFunctor, kNchwDimensionCount>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::DivNoNan::Attribute::T, TF_FLOAT, TF_HALF, TF_INT32, TF_INT64, TF_UINT32, TF_UINT64>();
}

static void RegisterEqual()
{
    using K = KernelDefinition<
        ops::Equal,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS, DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Equal::Attribute::T, TF_FLOAT, TF_HALF, TF_INT8, TF_INT16, TF_INT64, TF_UINT8, TF_UINT16, TF_UINT32, TF_UINT64>();
}

static void RegisterLog()
{
    using K = KernelDefinition<
        ops::Log,
        DmlKernelWrapper<DmlUnaryScaleBiasKernel<DML_OPERATOR_ELEMENT_WISE_LOG, DML_ELEMENT_WISE_LOG_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Log::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterLogicalAnd()
{
    using K = KernelDefinition<
        ops::LogicalAnd,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND, DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    K::Register();
}

static void RegisterMaximum()
{
    using K = KernelDefinition<
        ops::Maximum,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_MAX, DML_ELEMENT_WISE_MAX_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Maximum::Attribute::T, TF_FLOAT, TF_HALF, TF_INT8, TF_INT16, TF_INT64, TF_UINT8, TF_UINT16, TF_UINT32, TF_UINT64>();
}

static void RegisterMinimum()
{
    using K = KernelDefinition<
        ops::Minimum,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_MIN, DML_ELEMENT_WISE_MIN_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Minimum::Attribute::T, TF_FLOAT, TF_HALF, TF_INT8, TF_INT16, TF_INT64, TF_UINT8, TF_UINT16, TF_UINT32, TF_UINT64>();
}

static void RegisterMul()
{
    using K = KernelDefinition<
        ops::Mul,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_MULTIPLY, DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Mul::Attribute::T, TF_FLOAT, TF_HALF, TF_INT64, TF_UINT32, TF_UINT64>();
}

static void RegisterReciprocal()
{
    using K = KernelDefinition<
        ops::Reciprocal,
        DmlKernelWrapper<DmlUnaryScaleBiasKernel<DML_OPERATOR_ELEMENT_WISE_RECIP, DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Reciprocal::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterRelu6()
{
    using K = KernelDefinition<
        ops::Relu6,
        DmlKernelWrapper<DmlUnaryScaleBiasKernel<DML_OPERATOR_ELEMENT_WISE_CLIP, DML_ELEMENT_WISE_CLIP_OPERATOR_DESC, 1, 0, 0, 6>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Relu6::Attribute::T, TF_FLOAT, TF_HALF, TF_INT8, TF_INT16, TF_INT32, TF_INT64, TF_UINT8, TF_UINT16, TF_UINT32, TF_UINT64>();
}

static void RegisterRsqrt()
{
    using K = KernelDefinition<
        ops::Rsqrt,
        DmlKernelWrapper<DmlCompositeUnaryKernel<DmlRsqrtFunctor, kNchwDimensionCount>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Rsqrt::Attribute::T, TF_FLOAT, TF_HALF>();
}

static void RegisterSub()
{
    using K = KernelDefinition<
        ops::Sub,
        DmlKernelWrapper<DmlBinaryKernel<DML_OPERATOR_ELEMENT_WISE_SUBTRACT, DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>, GetBroadcastedOutputShapeHelper>>;

    RegisterWithTypes<K, ops::Sub::Attribute::T, TF_FLOAT, TF_HALF, TF_INT64, TF_UINT32, TF_UINT64>();
}

void RegisterKernels_Cwise()
{
    RegisterAdd();
    RegisterAddV2();
    RegisterDivNoNan();
    RegisterEqual();
    RegisterLog();
    RegisterLogicalAnd();
    RegisterMaximum();
    RegisterMinimum();
    RegisterMul();
    RegisterRelu6();
    RegisterRsqrt();
    RegisterSub();
}

} // namespace tfdml