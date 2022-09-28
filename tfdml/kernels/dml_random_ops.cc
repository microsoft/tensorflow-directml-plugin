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

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/eager/c_api.h"
#include "tfdml/kernels/pch.h"
#include "tfdml/runtime_adapter/guarded_philox_random.h"
#include "tfdml/runtime_adapter/random_ops_util.h"
#include "tfdml/runtime_adapter/rng_alg.h"
#include "tfdml/runtime_adapter/stateless_random_ops.h"
#include "tfdml/runtime_adapter/variable_lock.h"

namespace tfdml
{

// Some kernels that don't have a DML implementation are emulated on the CPU to
// force optimizatizer to be placed on DML devices
enum class EmulatedKernelType
{
    kRandomStandardNormal,
    kTruncatedNormal,
};

// Helpers to convert random uniform bits to a real uniform distribution. This
// approach outputs a floating-point value with sign=0 (positive), exponent=2^0,
// and mantissa set to the lowest-order M bits from the random generator output
// (M = bits in the floating-point mantissa). For example, FP32 consumes the
// lowest 23 bits from each 32-bit generator value; FP16 consumes the lowest 10
// bits from each 32-bit generator value. FP64 (not implemented) would require
// 2 generator values per output vaule, and it would use the lowest 52 bits.
dml::Expression UniformFloat(
    dml::Graph& scope,
    dml::Expression input_state,
    uint32_t element_count)
{
    // FP32 has 1 sign bit, 8 exponent bits, and 23 mantissa bits.
    constexpr uint32_t sign_and_exponent_value = ((1 << (8 - 1)) - 1) << 23;
    constexpr uint32_t mantissa_mask_value = (1 << 23) - 1;

    auto generator_outputs =
        dml::RandomGenerator(input_state, {1, 1, 1, element_count}, false);
    auto random_bits = generator_outputs.values;

    auto sign_and_exponent = dml::ScalarTensor(
        scope,
        sign_and_exponent_value,
        random_bits.GetOutputDesc().sizes);

    auto mantissa_mask = dml::ScalarTensor(
        scope,
        mantissa_mask_value,
        random_bits.GetOutputDesc().sizes);

    auto result = sign_and_exponent | (random_bits & mantissa_mask);

    return dml::Reinterpret(result, DML_TENSOR_DATA_TYPE_FLOAT32) - 1.0f;
}

dml::Expression UniformHalf(
    dml::Graph& scope,
    dml::Expression input_state,
    uint32_t element_count)
{
    // FP16 has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    constexpr uint32_t sign_and_exponent_value = ((1 << (5 - 1)) - 1) << 10;
    constexpr uint32_t mantissa_mask_value = (1 << 10) - 1;

    auto generator_outputs =
        dml::RandomGenerator(input_state, {1, 1, 1, element_count}, false);
    auto random_bits = generator_outputs.values;

    auto sign_and_exponent = dml::ScalarTensor(
        scope,
        sign_and_exponent_value,
        random_bits.GetOutputDesc().sizes);

    auto mantissa_mask = dml::ScalarTensor(
        scope,
        mantissa_mask_value,
        random_bits.GetOutputDesc().sizes);

    auto result = sign_and_exponent | (random_bits & mantissa_mask);

    result = dml::Cast(result, DML_TENSOR_DATA_TYPE_UINT16);

    return dml::Reinterpret(result, DML_TENSOR_DATA_TYPE_FLOAT16) - 1.0f;
}

// Compute a + b where a is a signed type and b is unsigned. Requires the result
// is representable in the range of a's data type. See SignedAdd from
// random_distributions.h.
dml::Expression SignedAdd(dml::Expression a, dml::Expression b)
{
    auto b_div_2 = b / 2;
    return a + dml::Reinterpret(b_div_2, a.GetOutputDesc().dataType) +
           dml::Reinterpret(b - b_div_2, a.GetOutputDesc().dataType);
}

// TODO: Remove this when int64/uint64 support division
// TFDML #41163316
dml::Expression SignedAdd64(
    dml::Graph& graph,
    dml::Expression a,
    dml::Expression b)
{
    assert(a.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_INT64);
    assert(b.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_UINT64);

    // int64 doesn't support divisions yet, so do a BitShiftRight instead
    auto b_div_2 =
        b >> dml::ScalarTensor<uint64_t>(graph, 1, b.GetOutputDesc().sizes);
    return a + dml::Reinterpret(b_div_2, a.GetOutputDesc().dataType) +
           dml::Reinterpret(b - b_div_2, a.GetOutputDesc().dataType);
}

// Produces a uniform distribution of integers in the range [min_value,
// max_value). See UniformDistribution<Generator, int32> from
// random_distributions.h. Requires min_value < max_value.
template <typename T>
dml::Expression UniformInt(
    dml::Graph& graph,
    dml::Expression input_state,
    T min_value,
    T max_value,
    uint32_t element_count)
{
    using TUnsigned = typename std::make_unsigned<T>::type;

    const dml::TensorDimensions shape = {1, 1, 1, element_count};

    // The generator always generates uint32_t value, so we need to generate
    // more or less depending on the desired output type
    uint32_t generator_num_elements =
        element_count * sizeof(TUnsigned) / sizeof(uint32_t);
    const dml::TensorDimensions generator_shape =
        {1, 1, 1, generator_num_elements};

    auto generator = dml::RandomGenerator(input_state, generator_shape, false);

    auto random_bits = generator.values;
    auto lo = dml::ScalarTensor(graph, min_value, shape);

    TUnsigned max_value_unsigned = static_cast<TUnsigned>(max_value);
    TUnsigned min_value_unsigned = static_cast<TUnsigned>(min_value);
    TUnsigned range_value = max_value_unsigned - min_value_unsigned;

    dml::Expression mod_result;
    if (std::is_same<T, int64_t>::value)
    {
        auto invariant_operand = (1 << 16) % range_value;
        invariant_operand *= invariant_operand;

        // We make sure that we never overflow even if we have the biggest
        // values possible on the GPU. We should have fell back to the fallback
        // kernel before we can even reach this assert.
        assert(
            invariant_operand * (range_value - 1) + (range_value - 1) <=
            UINT32_MAX);

        auto range = dml::ScalarTensor<uint32_t>(graph, range_value, shape);
        auto invariant_operand_scalar =
            dml::ScalarTensor<uint32_t>(graph, invariant_operand, shape);

        random_bits = dml::Reinterpret(
            random_bits,
            TfTensorTypeTraits<TUnsigned>::dml_type,
            shape,
            {});

        auto random_bits_low = dml::Reinterpret(
            random_bits,
            DML_TENSOR_DATA_TYPE_UINT32,
            shape,
            dml::TensorDimensions({1, 1, 1, 2}));

        auto random_bits_high = dml::Reinterpret(
            random_bits >> dml::ScalarTensor<uint64_t>(graph, 32, shape),
            DML_TENSOR_DATA_TYPE_UINT32,
            shape,
            dml::TensorDimensions({1, 1, 1, 2}));

        mod_result = (invariant_operand_scalar * (random_bits_high % range) +
                      (random_bits_low % range)) %
                     range;

        mod_result = dml::Cast(mod_result, DML_TENSOR_DATA_TYPE_UINT64);

        return SignedAdd64(graph, lo, mod_result);
    }
    else
    {
        auto range = dml::ScalarTensor(graph, range_value, shape);
        return SignedAdd(lo, random_bits % range);
    }
}

static Status CheckKeyCounterShape(
    int minimum_counter_size,
    TensorShape const& key_shape,
    TensorShape const& counter_shape)
{
    if (!(key_shape.dims() == 1 && key_shape.dim_size(0) == RNG_KEY_SIZE))
    {
        return errors::InvalidArgument(
            "key must have shape [",
            RNG_KEY_SIZE,
            "], not ",
            key_shape.DebugString(),
            ". (Note that batched keys are not supported yet.)");
    }
    if (!(counter_shape.dims() == 1 &&
          counter_shape.dim_size(0) >= minimum_counter_size))
    {
        return errors::InvalidArgument(
            "counter must be a vector with length at least ",
            minimum_counter_size,
            "; got shape: ",
            counter_shape.DebugString(),
            ". (Note that batched counters are not supported yet.)");
    }
    return Status::OK();
}

template <typename T>
static Status GetScalar(const Tensor& tensor, int input_idx, T* result)
{
    auto dtype = DataTypeToEnum<T>();
    if (tensor.dims() != 0)
    {
        return errors::InvalidArgument(
            "input ",
            std::to_string(input_idx),
            " (0-based) must have shape [], not ",
            tensor.shape().DebugString());
    }
    if (tensor.dtype() != dtype)
    {
        return errors::InvalidArgument(
            "dtype of input ",
            std::to_string(input_idx),
            " (0-based) must be ",
            DataTypeString(dtype),
            ", not ",
            DataTypeString(tensor.dtype()));
    }
    *result = tensor.base<T>()[0];
    return Status::OK();
}

class BaseStatelessRandomUniformInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;
    virtual const TensorShape& GetOutputShape() const = 0;
};

class StatelessRandomUniformInitHelper
    : public BaseStatelessRandomUniformInitHelper
{
  public:
    StatelessRandomUniformInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& shape_t = ctx->input(0);
        const Tensor& seed_t = ctx->input(1);
        TensorShape shape;
        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(shape_t, &shape));

        OP_REQUIRES(
            ctx,
            seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
            errors::InvalidArgument(
                "seed must have shape [2], not ",
                seed_t.shape().DebugString()));

        output_shape_ = std::move(shape);
        if (output_shape_.num_elements() == 0) return;

        OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key_, &counter_));

        // This init helper is shared for both "StatelessRandomUniform" (real
        // types) and "StatelessRandomUniformInt" (integral types). The latter
        // has two extra host-memory tensors for the min and max of the output
        // range.
        if (ctx->num_inputs() == 4)
        {
            const Tensor& minval = ctx->input(2);
            const Tensor& maxval = ctx->input(3);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument(
                    "minval must be 0-D, got shape ",
                    minval.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument(
                    "maxval must be 0-D, got shape ",
                    maxval.shape().DebugString()));

            // Verify that minval < maxval. Note that we'll never reach this
            // point for empty output.  Zero impossible things are fine.
            const auto lo = minval.base<int32_t>()[0];
            const auto hi = maxval.base<int32_t>()[0];
            OP_REQUIRES(
                ctx,
                lo < hi,
                errors::InvalidArgument(
                    "Need minval < maxval, got ",
                    lo,
                    " >= ",
                    hi));
        }
    }

    const TensorShape& GetOutputShape() const final { return output_shape_; }
    const random::PhiloxRandom::Key GetKey() const { return key_; }
    const random::PhiloxRandom::ResultType GetCounter() const
    {
        return counter_;
    }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        for (size_t i = 0; i < output_shapes.size(); ++i)
        {
            if (output_shapes[i].num_elements() != 0)
            {
                return false;
            }
        }
        return true;
    }

  private:
    TensorShape output_shape_;
    random::PhiloxRandom::Key key_;
    random::PhiloxRandom::ResultType counter_;
};

class StatelessRandomUniformV2InitHelper
    : public BaseStatelessRandomUniformInitHelper
{
  public:
    StatelessRandomUniformV2InitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& key_tensor = ctx->input(1);
        const Tensor& counter_tensor = ctx->input(2);
        const Tensor& alg_tensor = ctx->input(3);

        int alg_id;
        OP_REQUIRES_OK(ctx, GetScalar(alg_tensor, 3, &alg_id));
        Algorithm alg = Algorithm(alg_id);
        if (alg == RNG_ALG_AUTO_SELECT)
        {
            alg = RNG_ALG_PHILOX;
        }

        OP_REQUIRES_OK(
            ctx,
            CheckKeyCounterShape(
                alg,
                key_tensor.shape(),
                counter_tensor.shape()));

        OP_REQUIRES(
            ctx,
            alg == RNG_ALG_PHILOX,
            errors::InvalidArgument("Unsupported algorithm id: ", alg));

        TensorShape shape;
        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(ctx->input(0), &shape));

        output_shape_ = std::move(shape);
        if (output_shape_.num_elements() == 0) return;

        // This init helper is shared for both "StatelessRandomUniformV2" (real
        // types) and "StatelessRandomUniformIntV2" (integral types). The latter
        // has two extra host-memory tensors for the min and max of the output
        // range.
        if (ctx->num_inputs() == 6)
        {
            const Tensor& minval = ctx->input(4);
            const Tensor& maxval = ctx->input(5);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument(
                    "minval must be 0-D, got shape ",
                    minval.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument(
                    "maxval must be 0-D, got shape ",
                    maxval.shape().DebugString()));

            // Verify that minval < maxval. Note that we'll never reach this
            // point for empty output.  Zero impossible things are fine.
            const auto lo = minval.base<int32_t>()[0];
            const auto hi = maxval.base<int32_t>()[0];
            OP_REQUIRES(
                ctx,
                lo < hi,
                errors::InvalidArgument(
                    "Need minval < maxval, got ",
                    lo,
                    " >= ",
                    hi));
        }
    }

    const TensorShape& GetOutputShape() const final { return output_shape_; }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        for (size_t i = 0; i < output_shapes.size(); ++i)
        {
            if (output_shapes[i].num_elements() != 0)
            {
                return false;
            }
        }
        return true;
    }

  private:
    TensorShape output_shape_;
};

class StatelessRandomUniformShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const BaseStatelessRandomUniformInitHelper*>(
                initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

class DmlStatelessRandomUniformKernel : public DmlKernel
{
    std::array<uint32_t, 6> input_state_;

  public:
    using InitHelper = StatelessRandomUniformInitHelper;

    explicit DmlStatelessRandomUniformKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        auto num_elements =
            static_cast<uint32_t>(init_helper->GetOutputShape().num_elements());

        // Copy counter & key into the input_state_ buffer.
        auto counter = init_helper->GetCounter();
        auto key = init_helper->GetKey();
        input_state_[0] = counter[0];
        input_state_[1] = counter[1];
        input_state_[2] = counter[2];
        input_state_[3] = counter[3];
        input_state_[4] = key[0];
        input_state_[5] = key[1];

        // Reserve an input binding, even though TF doesn't provide a (device)
        // input tensor. We will swap in a temporary buffer and upload the CPU
        // state at compute time.
        DmlTensorInfo state_info;
        state_info.kernel_index = 0;
        std::array<uint32_t, 4> state_sizes = {1, 1, 1, 6};
        state_info.desc =
            DmlTensorDesc::Create(TF_UINT32, state_sizes, state_sizes);

        const auto out_dtype = ctx->GetOutputDataType(0);

        // Flatten output shape for DirectML.
        DmlTensorInfo output_info;
        output_info.kernel_index = 0;
        std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_elements};
        output_info.desc =
            DmlTensorDesc::Create(out_dtype, output_sizes, output_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {state_info};
        tensors.outputs = {output_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_state = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;
        if (out_dtype == TF_FLOAT)
        {
            result = UniformFloat(scope, input_state, num_elements);
        }
        else if (out_dtype == TF_HALF)
        {
            result = UniformHalf(scope, input_state, num_elements);
        }
        else if (out_dtype == TF_INT32)
        {
            auto min_value = ctx->GetConstantInputTensor(2).base<int32_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(3).base<int32_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_elements);
        }
        else
        {
            assert(out_dtype == TF_INT64);
            auto min_value = ctx->GetConstantInputTensor(2).base<int64_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(3).base<int64_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_elements);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        DmlBuffer input_state_buffer =
            ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
                ctx->GetOpKernelContext()->raw(),
                6 * sizeof(uint32_t));

        Tensor output_tensor = ctx->GetOutputTensor(0);
        D3D12BufferRegion output_buffer =
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output_tensor);

        if (!input_state_buffer)
        {
            return errors::ResourceExhausted(
                "OOM when allocating a buffer of ",
                6 * sizeof(uint32_t),
                " bytes");
        }

        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1>
            input_bindings;
        input_bindings.push_back(input_state_buffer.GetBufferBinding());

        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1>
            output_bindings;
        output_bindings.push_back(output_buffer.GetBufferBinding());

        // Upload generator input state.
        auto byte_ptr = reinterpret_cast<const uint8_t*>(input_state_.data());
        auto byte_span = absl::MakeSpan(
            byte_ptr,
            input_state_.size() * sizeof(input_state_[0]));

        ctx->GetDmlDeviceContext()->CopyHostToBuffer(
            input_state_buffer.Region(),
            byte_span);

        return DmlKernel::Compute(ctx, input_bindings, output_bindings);
    }
};

class DmlStatelessRandomUniformV2Kernel : public DmlKernel
{
  public:
    using InitHelper = StatelessRandomUniformV2InitHelper;

    explicit DmlStatelessRandomUniformV2Kernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        auto num_elements =
            static_cast<uint32_t>(init_helper->GetOutputShape().num_elements());

        std::array<uint32_t, 4> key_sizes = {1, 1, 1, 2};
        DmlTensorInfo key_tensor;
        key_tensor.kernel_index = 1;
        key_tensor.desc =
            DmlTensorDesc::Create(TF_UINT32, key_sizes, key_sizes);

        std::array<uint32_t, 4> counter_sizes = {1, 1, 1, 4};
        DmlTensorInfo counter_tensor;
        counter_tensor.kernel_index = 2;
        counter_tensor.desc =
            DmlTensorDesc::Create(TF_UINT32, counter_sizes, counter_sizes);

        const auto out_dtype = ctx->GetOutputDataType(0);

        // Flatten output shape for DirectML.
        DmlTensorInfo output_info;
        output_info.kernel_index = 0;
        std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_elements};
        output_info.desc =
            DmlTensorDesc::Create(out_dtype, output_sizes, output_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {key_tensor, counter_tensor};
        tensors.outputs = {output_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto key = dml::InputTensor(scope, 0, inputs[0]);
        auto counter = dml::InputTensor(scope, 1, inputs[1]);
        auto input_state = dml::Join({counter, key}, 3);

        dml::Expression result;
        if (out_dtype == TF_FLOAT)
        {
            result = UniformFloat(scope, input_state, num_elements);
        }
        else if (out_dtype == TF_HALF)
        {
            result = UniformHalf(scope, input_state, num_elements);
        }
        else if (out_dtype == TF_INT32)
        {
            auto min_value = ctx->GetConstantInputTensor(4).base<int32_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(5).base<int32_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_elements);
        }
        else
        {
            assert(out_dtype == TF_INT64);
            auto min_value = ctx->GetConstantInputTensor(4).base<int64_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(5).base<int64_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_elements);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

// ----------------------------------------------------------------------------

template <
    typename TKernel,
    typename TShapeHelper,
    DmlKernelCachePolicy cache_policy = DmlKernelCachePolicy::Never>
class DmlPhiloxWrapper
    : public DmlKernelWrapper<TKernel, TShapeHelper, cache_policy>
{
  public:
    explicit DmlPhiloxWrapper(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : DmlKernelWrapper<TKernel, TShapeHelper, cache_policy>(
              ctx,
              std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, generator_.Init(ctx));
    }

    StatusOr<DmlGpuEvent> ComputeKernel(
        DmlKernel* kernel,
        DmlKernelContext* context) const override
    {
        return static_cast<TKernel*>(kernel)->Compute(context, generator_);
    }

  protected:
    mutable GuardedPhiloxRandom generator_;
};

class RandomUniformInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    RandomUniformInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& shape_t = ctx->input(0);
        TensorShape shape;
        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(shape_t, &shape));
        output_shape_ = std::move(shape);

        // This init helper is shared for both "RandomUniform" (real types) and
        // "RandomUniformInt" (integral types). The latter has two extra
        // host-memory tensors for the min and max of the output range.
        if (ctx->num_inputs() == 3)
        {
            const Tensor& minval = ctx->input(1);
            const Tensor& maxval = ctx->input(2);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument(
                    "minval must be 0-D, got shape ",
                    minval.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument(
                    "maxval must be 0-D, got shape ",
                    maxval.shape().DebugString()));

            if (output_shape_.num_elements() == 0) return;

            // Verify that minval < maxval. Note that we'll never reach this
            // point for empty output.  Zero impossible things are fine.
            const auto lo = minval.base<int32_t>()[0];
            const auto hi = maxval.base<int32_t>()[0];
            OP_REQUIRES(
                ctx,
                lo < hi,
                errors::InvalidArgument(
                    "Need minval < maxval, got ",
                    lo,
                    " >= ",
                    hi));
        }
    }

    const TensorShape& GetOutputShape() const { return output_shape_; }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        for (size_t i = 0; i < output_shapes.size(); ++i)
        {
            if (output_shapes[i].num_elements() != 0)
            {
                return false;
            }
        }
        return true;
    }

  private:
    TensorShape output_shape_;
};

class RandomUniformShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const RandomUniformInitHelper*>(initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

class DmlRandomUniformKernel : public DmlKernel
{
    absl::optional<DmlBuffer> state_buffer_;
    uint32_t num_output_elements_;

  public:
    using InitHelper = tfdml::RandomUniformInitHelper;

    explicit DmlRandomUniformKernel(
        DmlKernelConstruction* ctx,
        const RandomUniformInitHelper* init_helper)
    {
        num_output_elements_ =
            static_cast<uint32_t>(init_helper->GetOutputShape().num_elements());

        state_buffer_ = ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
            ctx->GetOpKernelContext()->raw(),
            6 * sizeof(uint32_t));

        OP_REQUIRES(
            ctx->GetOpKernelContext(),
            state_buffer_,
            errors::ResourceExhausted(
                "OOM when allocating a buffer of ",
                6 * sizeof(uint32_t),
                " bytes"));

        // Reserve input state binding. This will point at state_buffer_.
        DmlTensorInfo state_info;
        state_info.kernel_index = 0;
        std::array<uint32_t, 4> state_sizes = {1, 1, 1, 6};
        state_info.desc =
            DmlTensorDesc::Create(TF_UINT32, state_sizes, state_sizes);

        const auto out_dtype = ctx->GetOutputDataType(0);

        // Flatten output shape for DirectML.
        DmlTensorInfo output_info;
        output_info.kernel_index = 0;
        std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_output_elements_};
        output_info.desc =
            DmlTensorDesc::Create(out_dtype, output_sizes, output_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {state_info};
        tensors.outputs = {output_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_state = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;
        if (out_dtype == TF_FLOAT)
        {
            result = UniformFloat(scope, input_state, num_output_elements_);
        }
        else if (out_dtype == TF_HALF)
        {
            result = UniformHalf(scope, input_state, num_output_elements_);
        }
        else if (out_dtype == TF_INT32)
        {
            auto min_value = ctx->GetConstantInputTensor(1).base<int32_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(2).base<int32_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_output_elements_);
        }
        else
        {
            assert(out_dtype == TF_INT64);
            auto min_value = ctx->GetConstantInputTensor(1).base<int64_t>()[0];
            auto max_value = ctx->GetConstantInputTensor(2).base<int64_t>()[0];

            result = UniformInt(
                scope,
                input_state,
                min_value,
                max_value,
                num_output_elements_);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(
        DmlKernelContext* ctx,
        GuardedPhiloxRandom& generator) const
    {
        Tensor output_tensor = ctx->GetOutputTensor(0);
        D3D12BufferRegion output_buffer =
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output_tensor);

        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1>
            input_bindings;
        input_bindings.push_back(state_buffer_->GetBufferBinding());

        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1>
            output_bindings;
        output_bindings.push_back(output_buffer.GetBufferBinding());

        // Upload generator state. Note that generator_.ReserveRandomOutputs()
        // is thread safe and doesn't actually invoke the Philox generator; it
        // simply returns the current counter and then advances its internal
        // counter.
        std::array<uint32_t, 6> state_buf;
        auto philox_state =
            generator.ReserveRandomOutputs(num_output_elements_, 256);
        state_buf[0] = philox_state.counter()[0];
        state_buf[1] = philox_state.counter()[1];
        state_buf[2] = philox_state.counter()[2];
        state_buf[3] = philox_state.counter()[3];
        state_buf[4] = philox_state.key()[0];
        state_buf[5] = philox_state.key()[1];

        auto byte_ptr = reinterpret_cast<const uint8_t*>(state_buf.data());
        auto byte_span =
            absl::MakeSpan(byte_ptr, state_buf.size() * sizeof(state_buf[0]));

        ctx->GetDmlDeviceContext()->CopyHostToBuffer(
            state_buffer_->Region(),
            byte_span);

        return DmlKernel::Compute(ctx, input_bindings, output_bindings);
    }
};

template <typename DmlRandomKernelWrapperImpl, bool is_stateless, bool is_v2>
class RandomUniformInt64KernelSelector : public OpKernel
{
  public:
    explicit RandomUniformInt64KernelSelector(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(node_def),
          dml_kernel_wrapper_(ctx, node_def)
    {
        if (!is_stateless)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
        }
    }

    ~RandomUniformInt64KernelSelector()
    {
        TFE_DeleteOp(random_uniform_int_op_);
        TFE_DeleteContext(eager_context_);
    }

    void Compute(OpKernelContext* ctx)
    {
        int min_value_index = is_stateless ? 2 : 1;
        int max_value_index = is_stateless ? 3 : 2;

        const Tensor& min_value_tensor = ctx->input(min_value_index);
        const Tensor& max_value_tensor = ctx->input(max_value_index);
        int64_t min_value = min_value_tensor.base<int64_t>()[0];
        int64_t max_value = max_value_tensor.base<int64_t>()[0];
        uint64_t range_value =
            static_cast<uint64_t>(max_value) - static_cast<uint64_t>(min_value);

        auto invariant_operand = range_value == 0 ? 0 : (1 << 16) % range_value;
        invariant_operand *= invariant_operand;

        // The DML kernel implementation doesn't support real int64 modulus yet,
        // so fall back to the emulated kernel if we don't support the range
        if (invariant_operand * (range_value - 1) + (range_value - 1) >
            UINT32_MAX)
        {
            Status status;

            // Lazily create the eager op
            if (!eager_context_)
            {
                std::string op_name;
                op_name.reserve(28);

                if (is_stateless)
                {
                    op_name += "Stateless";
                }

                op_name += "RandomUniformInt";

                if (is_v2)
                {
                    op_name += "V2";
                }

                TFE_ContextOptions* context_options = TFE_NewContextOptions();
                auto context_options_cleanup = absl::MakeCleanup(
                    [context_options]
                    { TFE_DeleteContextOptions(context_options); });

                eager_context_ = TFE_NewContext(context_options, status.raw());
                OP_REQUIRES_OK(ctx, status);

                random_uniform_int_op_ =
                    TFE_NewOp(eager_context_, op_name.c_str(), status.raw());
                OP_REQUIRES_OK(ctx, status);

                TFE_OpSetDevice(
                    random_uniform_int_op_,
                    "/device:CPU",
                    status.raw());
                OP_REQUIRES_OK(ctx, status);

                if (!is_stateless)
                {
                    TFE_OpSetAttrInt(random_uniform_int_op_, "seed", seed_);
                    TFE_OpSetAttrInt(random_uniform_int_op_, "seed2", seed2_);
                }
            }

            absl::InlinedVector<TFE_TensorHandle*, 4> input_handles;
            auto input_handles_cleanup = absl::MakeCleanup(
                [&input_handles]
                {
                    for (TFE_TensorHandle* handle : input_handles)
                    {
                        TFE_DeleteTensorHandle(handle);
                    }
                });

            for (int i = 0; i < ctx->num_inputs(); ++i)
            {
                const Tensor& input_tensor = ctx->input(i);
                TFE_TensorHandle* input_handle =
                    TFE_NewTensorHandle(input_tensor.raw(), status.raw());
                OP_REQUIRES_OK(ctx, status);

                input_handles.push_back(input_handle);

                TFE_OpAddInput(
                    random_uniform_int_op_,
                    input_handle,
                    status.raw());
                OP_REQUIRES_OK(ctx, status);
            }

            TFE_TensorHandle* output_handle = nullptr;
            TFE_TensorHandle** output_handle_ptr = &output_handle;
            OP_REQUIRES_OK(ctx, status);
            auto output_handle_cleanup = absl::MakeCleanup(
                [output_handle_ptr]
                { TFE_DeleteTensorHandle(*output_handle_ptr); });

            int num_retvals = 1;
            TFE_Execute(
                random_uniform_int_op_,
                &output_handle,
                &num_retvals,
                status.raw());
            OP_REQUIRES_OK(ctx, status);

            Tensor output_cpu =
                Tensor(TFE_TensorHandleResolve(output_handle, status.raw()));
            OP_REQUIRES_OK(ctx, status);

            // Copy the CPU output back to the device
            StatusOr<Tensor> status_or_output =
                ctx->allocate_output(0, output_cpu.shape());
            OP_REQUIRES_OK(ctx, status_or_output.status());

            Tensor& output = status_or_output.ValueOrDie();
            OP_REQUIRES_OK(
                ctx,
                ctx->device()->CopyCPUTensorToDevice(&output_cpu, &output));
        }
        else
        {
            return dml_kernel_wrapper_.Compute(ctx);
        }
    }

  private:
    DmlRandomKernelWrapperImpl dml_kernel_wrapper_;
    int64_t seed_;
    int64_t seed2_;
    TFE_Context* eager_context_ = nullptr;
    TFE_Op* random_uniform_int_op_ = nullptr;
};

// ----------------------------------------------------------------------------

// Emulates a DML philox PRNG+distribution by executing it on the CPU and
// copying the results to the GPU.
template <EmulatedKernelType emulated_kernel_type>
class DmlEmulatedPhiloxRandomKernel : public OpKernel
{
  public:
    explicit DmlEmulatedPhiloxRandomKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        int seed;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));

        int seed2;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2));

        TF_DataType dtype;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));

        TFE_ContextOptions* context_options = TFE_NewContextOptions();
        auto context_options_cleanup = absl::MakeCleanup(
            [context_options] { TFE_DeleteContextOptions(context_options); });

        Status status;
        eager_context_ = TFE_NewContext(context_options, status.raw());
        OP_REQUIRES_OK(ctx, status);

        const char* emulated_kernel_name;
        switch (emulated_kernel_type)
        {
        case EmulatedKernelType::kRandomStandardNormal:
            emulated_kernel_name = "RandomStandardNormal";
            break;
        case EmulatedKernelType::kTruncatedNormal:
            emulated_kernel_name = "TruncatedNormal";
            break;
        }

        random_op_ =
            TFE_NewOp(eager_context_, emulated_kernel_name, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_OpSetAttrInt(random_op_, "seed", seed);
        TFE_OpSetAttrInt(random_op_, "seed2", seed2);
        TFE_OpSetAttrType(random_op_, "dtype", dtype);

        TFE_OpSetDevice(random_op_, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlEmulatedPhiloxRandomKernel() override
    {
        TFE_DeleteOp(random_op_);
        TFE_DeleteContext(eager_context_);
    }

    void Compute(OpKernelContext* ctx)
    {
        Status status;
        const Tensor& shape_tensor = ctx->input(0);
        TFE_TensorHandle* shape_handle =
            TFE_NewTensorHandle(shape_tensor.raw(), status.raw());
        OP_REQUIRES_OK(ctx, status);
        auto shape_handle_cleanup = absl::MakeCleanup(
            [shape_handle] { TFE_DeleteTensorHandle(shape_handle); });
        TFE_OpAddInput(random_op_, shape_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_TensorHandle* output_handle = nullptr;
        TFE_TensorHandle** output_handle_ptr = &output_handle;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup =
            absl::MakeCleanup([output_handle_ptr]
                              { TFE_DeleteTensorHandle(*output_handle_ptr); });

        int num_retvals = 1;
        TFE_Execute(random_op_, &output_handle, &num_retvals, status.raw());
        OP_REQUIRES_OK(ctx, status);

        Tensor output_cpu =
            Tensor(TFE_TensorHandleResolve(output_handle, status.raw()));
        OP_REQUIRES_OK(ctx, status);

        // Copy the CPU output back to the device
        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, output_cpu.shape());
        OP_REQUIRES_OK(ctx, status_or_output.status());

        Tensor& output = status_or_output.ValueOrDie();
        OP_REQUIRES_OK(
            ctx,
            ctx->device()->CopyCPUTensorToDevice(&output_cpu, &output));
    }

  private:
    TFE_Context* eager_context_ = nullptr;
    TFE_Op* random_op_ = nullptr;
};

template <typename AlgEnumType>
static Status GetAlg(OpKernelContext* ctx, int input_idx, Algorithm* alg)
{
    AlgEnumType alg_id;
    TF_RETURN_IF_ERROR(GetScalar(ctx->input(input_idx), input_idx, &alg_id));
    *alg = Algorithm(alg_id);
    return Status::OK();
}

// 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained
// in b/111604096 and cl/171681867), so we use signed int here. We choose int64
// instead of int32 because `VarHandleOp` doesn't support int32 on GPU, and
// because of the "int32 problem".
using StateElementType = int64_t;

static constexpr int64_t PHILOX_MIN_STATE_SIZE =
    (random::PhiloxRandom::ResultType::kElementCount +
     random::PhiloxRandom::Key::kElementCount) /
    2;

static Status CheckPhiloxState(const Tensor& state, int64_t alg_tag_skip = 0)
{
    static_assert(
        std::is_same<StateElementType, int64_t>::value,
        "StateElementType must be int64");
    static_assert(
        std::is_same<random::PhiloxRandom::ResultElementType, uint32_t>::value,
        "PhiloxRandom::ResultElementType must be uint32");
    auto min_size = alg_tag_skip + PHILOX_MIN_STATE_SIZE;
    if (state.NumElements() < min_size)
    {
        return errors::InvalidArgument(
            "For the Philox algorithm, the size of state"
            " must be at least ",
            min_size,
            "; got ",
            state.NumElements());
    }
    return Status::OK();
}

template <typename AlgEnumType, typename DeltaType>
class RngSkipInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    RngSkipInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : state_var_lock_(ctx)
    {
        constexpr int alg_input_index = 1;
        OP_REQUIRES_OK(
            ctx,
            GetAlg<AlgEnumType>(ctx, alg_input_index, &algorithm_));

        constexpr int delta_input_idx = 2;
        OP_REQUIRES_OK(
            ctx,
            GetScalar(ctx->input(delta_input_idx), delta_input_idx, &delta_));

        // Like for the CPU and CUDA devices, only Philox is supported for now
        OP_REQUIRES(
            ctx,
            algorithm_ == RNG_ALG_PHILOX,
            errors::InvalidArgument("Unsupported algorithm id: ", algorithm_));

        constexpr int state_input_index = 0;
        constexpr bool exclusive_lock = false;
        constexpr bool is_variant = false;
        OP_REQUIRES_OK(
            ctx,
            ctx->GetInputTensorFromVariable(
                state_input_index,
                exclusive_lock,
                is_variant,
                &state_tensor_));
        constexpr int lock_indices[1] = {state_input_index};
        state_var_lock_.LockShared(lock_indices);

        OP_REQUIRES_OK(ctx, CheckPhiloxState(state_tensor_));
    }

    void Unlock() const { state_var_lock_.Unlock(); }
    Algorithm GetAlgorithm() const { return algorithm_; }
    DeltaType GetDelta() const { return delta_; }
    const Tensor& GetStateTensor() const { return state_tensor_; }

  private:
    Algorithm algorithm_;
    DeltaType delta_;
    Tensor state_tensor_;
    mutable VariableLock state_var_lock_;
};

template <typename AlgEnumType, typename DeltaType>
class RngReadAndSkipShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        return {{RNG_MAX_COUNTER_SIZE + RNG_KEY_SIZE}};
    }
};

template <typename AlgEnumType, typename DeltaType, bool read_old_value>
class DmlRngSkipKernel : public DmlKernel
{
  public:
    using InitHelper = RngSkipInitializationHelper<AlgEnumType, DeltaType>;

    explicit DmlRngSkipKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const Tensor& state_tensor = init_helper->GetStateTensor();

        TensorShape input_output_shape = {1, 1, 1, state_tensor.NumElements()};

        DmlTensorInfo input_output;
        input_output.kernel_index = 0;
        input_output.desc = DmlTensorDesc::Create(
            state_tensor.dtype(),
            input_output_shape,
            input_output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input_output};
        tensors.outputs = {input_output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input = dml::InputTensor(scope, 0, inputs[0]);
        input = dml::Reinterpret(
            input,
            DML_TENSOR_DATA_TYPE_UINT32,
            {1, 1, 1, static_cast<uint32_t>(state_tensor.NumElements()) * 2},
            {});

        constexpr uint32_t split_axis = 3;
        auto split_inputs = dml::Split(
            input,
            split_axis,
            {1, 1, 1, 1, input.GetOutputDesc().sizes[3] - 4});

        const int64_t count = init_helper->GetDelta() * 256;

        auto dml_data_type = input.GetOutputDesc().dataType;

        auto count_lo = dml::FillValueConstant(
            scope,
            {1, 1, 1, 1},
            dml_data_type,
            dml::ScalarUnion(static_cast<uint32_t>(count), dml_data_type));

        auto count_hi = dml::FillValueConstant(
            scope,
            {1, 1, 1, 1},
            dml_data_type,
            dml::ScalarUnion(
                static_cast<uint32_t>(count >> 32),
                dml_data_type));

        split_inputs[0] += count_lo;
        count_hi = dml::If(split_inputs[0] < count_lo, count_hi + 1, count_hi);

        split_inputs[1] += count_hi;

        auto split_input_1_lower = split_inputs[1] < count_hi;

        split_inputs[2] =
            dml::If(split_input_1_lower, split_inputs[2] + 1, split_inputs[2]);

        auto zero = dml::ZeroTensor(
            scope,
            dml_data_type,
            split_inputs[2].GetOutputDesc().sizes);

        split_inputs[3] = dml::If(
            split_input_1_lower && split_inputs[2] == zero,
            split_inputs[3] + 1,
            split_inputs[3]);

        auto result = dml::Join(split_inputs, split_axis);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto init_helper = ctx->GetInitializationHelper<InitHelper>();
        const Tensor& state_input = init_helper->GetStateTensor();

        // We compute the new state in-place, which saves us from doing an
        // output -> input copy at the end
        D3D12BufferRegion input_output_buffers[] = {
            ctx->GetDmlDeviceContext()->GetBufferForTensor(state_input),
        };

        // Create bindings
        auto input_bindings = dml_util::GetBufferBindings(input_output_buffers);
        auto output_bindings =
            dml_util::GetBufferBindings(input_output_buffers);

        // For RngReadAndSkip, we copy the old value to the output before we
        // start computing
        if (read_old_value)
        {
            Tensor& output = ctx->GetOutputTensor(0);

            ctx->GetOpKernelContext()->device()->CopyTensorInSameDevice(
                &state_input,
                &output);
        }

        auto gpu_event_or_status =
            DmlKernel::Compute(ctx, input_bindings, output_bindings);

        init_helper->Unlock();
        return gpu_event_or_status;
    }
};

class GetKeyCounterAlgOp : public OpKernel
{
  public:
    explicit GetKeyCounterAlgOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& seed_t = ctx->input(0);
        OP_REQUIRES(
            ctx,
            seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
            errors::InvalidArgument(
                "seed must have shape [2], not ",
                seed_t.shape().DebugString()));
        // Allocate outputs
        auto status_or_key_output =
            ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}));
        OP_REQUIRES_OK(ctx, status_or_key_output.status());
        Tensor key_output = status_or_key_output.ConsumeValueOrDie();

        auto status_or_counter_output =
            ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}));
        OP_REQUIRES_OK(ctx, status_or_counter_output.status());
        Tensor counter_output = status_or_counter_output.ConsumeValueOrDie();

        auto status_or_alg_output = ctx->allocate_output(2, TensorShape({}));
        OP_REQUIRES_OK(ctx, status_or_alg_output.status());
        Tensor alg_output = status_or_alg_output.ConsumeValueOrDie();

        random::PhiloxRandom::Key key;
        random::PhiloxRandom::ResultType counter;
        OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
        WriteKeyToMem(key, key_output.base<uint64_t>());
        WriteCounterToMem(counter, counter_output.base<uint64_t>());
        alg_output.base<int>()[0] = RNG_ALG_PHILOX;
    }
};

class GetKeyCounterOp : public OpKernel
{
  public:
    explicit GetKeyCounterOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& seed_t = ctx->input(0);
        OP_REQUIRES(
            ctx,
            seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
            errors::InvalidArgument(
                "seed must have shape [2], not ",
                seed_t.shape().DebugString()));
        // Allocate outputs
        auto status_or_key_output =
            ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}));
        OP_REQUIRES_OK(ctx, status_or_key_output.status());
        Tensor key_output = status_or_key_output.ConsumeValueOrDie();

        auto status_or_counter_output =
            ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}));
        OP_REQUIRES_OK(ctx, status_or_counter_output.status());
        Tensor counter_output = status_or_counter_output.ConsumeValueOrDie();

        random::PhiloxRandom::Key key;
        random::PhiloxRandom::ResultType counter;
        OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
        WriteKeyToMem(key, key_output.base<uint64_t>());
        WriteCounterToMem(counter, counter_output.base<uint64_t>());
    }
};

class GetAlgOp : public OpKernel
{
  public:
    explicit GetAlgOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        auto status_or_alg_output = ctx->allocate_output(0, TensorShape({}));
        OP_REQUIRES_OK(ctx, status_or_alg_output.status());
        Tensor alg_output = status_or_alg_output.ConsumeValueOrDie();

        alg_output.base<int>()[0] = RNG_ALG_PHILOX;
    }
};

void RegisterStatelessRandomUniform()
{
    using K = KernelDefinition<
        ops::StatelessRandomUniform,
        DmlKernelWrapper<
            DmlStatelessRandomUniformKernel,
            StatelessRandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniform::Argument::shape,
            ops::StatelessRandomUniform::Argument::seed>;

    RegisterWithTypes<
        K,
        ops::StatelessRandomUniform::Attribute::dtype,
        TF_HALF,
        TF_FLOAT>();
}

void RegisterStatelessRandomUniformV2()
{
    using K = KernelDefinition<
        ops::StatelessRandomUniformV2,
        DmlKernelWrapper<
            DmlStatelessRandomUniformV2Kernel,
            StatelessRandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformV2::Argument::shape,
            ops::StatelessRandomUniformV2::Argument::alg>;

    RegisterWithTypes<
        K,
        ops::StatelessRandomUniformV2::Attribute::dtype,
        TF_HALF,
        TF_FLOAT>();
}

void RegisterStatelessRandomUniformInt()
{
    using int32_kernel = KernelDefinition<
        ops::StatelessRandomUniformInt,
        DmlKernelWrapper<
            DmlStatelessRandomUniformKernel,
            StatelessRandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformInt::Argument::shape,
            ops::StatelessRandomUniformInt::Argument::seed,
            ops::StatelessRandomUniformInt::Argument::minval,
            ops::StatelessRandomUniformInt::Argument::maxval>::
            WithTypeConstraint<
                ops::StatelessRandomUniformInt::Attribute::dtype,
                TF_INT32>;
    int32_kernel::Register();

    using int64_kernel = KernelDefinition<
        ops::StatelessRandomUniformInt,
        RandomUniformInt64KernelSelector<
            DmlKernelWrapper<
                DmlStatelessRandomUniformKernel,
                StatelessRandomUniformShapeHelper>,
            true,
            false>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformInt::Argument::shape,
            ops::StatelessRandomUniformInt::Argument::seed,
            ops::StatelessRandomUniformInt::Argument::minval,
            ops::StatelessRandomUniformInt::Argument::maxval>::
            WithTypeConstraint<
                ops::StatelessRandomUniformInt::Attribute::dtype,
                TF_INT64>;
    int64_kernel::Register();
}

void RegisterStatelessRandomUniformIntV2()
{
    using int32_kernel = KernelDefinition<
        ops::StatelessRandomUniformIntV2,
        DmlKernelWrapper<
            DmlStatelessRandomUniformV2Kernel,
            StatelessRandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformIntV2::Argument::shape,
            ops::StatelessRandomUniformIntV2::Argument::alg,
            ops::StatelessRandomUniformIntV2::Argument::minval,
            ops::StatelessRandomUniformIntV2::Argument::maxval>::
            WithTypeConstraint<
                ops::StatelessRandomUniformIntV2::Attribute::dtype,
                TF_INT32>;
    int32_kernel::Register();

    using int64_kernel = KernelDefinition<
        ops::StatelessRandomUniformIntV2,
        RandomUniformInt64KernelSelector<
            DmlKernelWrapper<
                DmlStatelessRandomUniformV2Kernel,
                StatelessRandomUniformShapeHelper>,
            true,
            true>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformIntV2::Argument::shape,
            ops::StatelessRandomUniformIntV2::Argument::alg,
            ops::StatelessRandomUniformIntV2::Argument::minval,
            ops::StatelessRandomUniformIntV2::Argument::maxval>::
            WithTypeConstraint<
                ops::StatelessRandomUniformIntV2::Attribute::dtype,
                TF_INT64>;
    int64_kernel::Register();
}

void RegisterRandomUniform()
{
    using K = KernelDefinition<
        ops::RandomUniform,
        DmlPhiloxWrapper<DmlRandomUniformKernel, RandomUniformShapeHelper>>::
        WithHostMemoryArguments<ops::RandomUniform::Argument::shape>::
            WithTypeConstraint<ops::RandomUniform::Attribute::T, TF_INT32>;

    RegisterWithTypes<
        K,
        ops::RandomUniform::Attribute::dtype,
        TF_HALF,
        TF_FLOAT>();
}

void RegisterRandomUniformInt()
{
    using int32_kernel = KernelDefinition<
        ops::RandomUniformInt,
        DmlPhiloxWrapper<DmlRandomUniformKernel, RandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::RandomUniformInt::Argument::shape,
            ops::RandomUniformInt::Argument::minval,
            ops::RandomUniformInt::Argument::maxval>::
            WithTypeConstraint<ops::RandomUniformInt::Attribute::T, TF_INT32>::
                WithTypeConstraint<
                    ops::RandomUniformInt::Attribute::Tout,
                    TF_INT32>;
    int32_kernel::Register();

    using int64_kernel = KernelDefinition<
        ops::RandomUniformInt,
        RandomUniformInt64KernelSelector<
            DmlPhiloxWrapper<DmlRandomUniformKernel, RandomUniformShapeHelper>,
            false,
            false>>::
        WithHostMemoryArguments<
            ops::RandomUniformInt::Argument::shape,
            ops::RandomUniformInt::Argument::minval,
            ops::RandomUniformInt::Argument::maxval>::
            WithTypeConstraint<ops::RandomUniformInt::Attribute::T, TF_INT32>::
                WithTypeConstraint<
                    ops::RandomUniformInt::Attribute::Tout,
                    TF_INT64>;
    int64_kernel::Register();
}

void RegisterRandomStandardNormal()
{
    using half_kernel = KernelDefinition<
        ops::RandomStandardNormal,
        DmlEmulatedPhiloxRandomKernel<
            EmulatedKernelType::kRandomStandardNormal>>::
        WithHostMemoryArguments<ops::RandomStandardNormal::Argument::shape>;

    using float_kernel = KernelDefinition<
        ops::RandomStandardNormal,
        DmlEmulatedPhiloxRandomKernel<
            EmulatedKernelType::kRandomStandardNormal>>::
        WithHostMemoryArguments<ops::RandomStandardNormal::Argument::shape>;

    half_kernel::WithTypeConstraint<
        ops::RandomStandardNormal::Attribute::dtype,
        TF_HALF>::Register();
    float_kernel::WithTypeConstraint<
        ops::RandomStandardNormal::Attribute::dtype,
        TF_FLOAT>::Register();
}

void RegisterTruncatedNormal()
{
    using half_kernel = KernelDefinition<
        ops::TruncatedNormal,
        DmlEmulatedPhiloxRandomKernel<EmulatedKernelType::kTruncatedNormal>>::
        WithHostMemoryArguments<ops::TruncatedNormal::Argument::shape>;

    using float_kernel = KernelDefinition<
        ops::TruncatedNormal,
        DmlEmulatedPhiloxRandomKernel<EmulatedKernelType::kTruncatedNormal>>::
        WithHostMemoryArguments<ops::TruncatedNormal::Argument::shape>;

    half_kernel::WithTypeConstraint<
        ops::TruncatedNormal::Attribute::dtype,
        TF_HALF>::Register();
    float_kernel::WithTypeConstraint<
        ops::TruncatedNormal::Attribute::dtype,
        TF_FLOAT>::Register();
}

void RegisterRngSkip()
{
    using K = KernelDefinition<
        ops::RngSkip,
        DmlKernelWrapper<
            DmlRngSkipKernel<int64_t, int64_t, false>,
            NoOutputShapeHelper,
            DmlKernelCachePolicy::Never>>::
        WithHostMemoryArguments<
            ops::RngSkip::Argument::resource,
            ops::RngSkip::Argument::algorithm,
            ops::RngSkip::Argument::delta>;

    K::Register();
}

void RegisterRngReadAndSkip()
{
    using K = KernelDefinition<
        ops::RngReadAndSkip,
        DmlKernelWrapper<
            DmlRngSkipKernel<int32_t, uint64_t, true>,
            RngReadAndSkipShapeHelper<int32_t, uint64_t>,
            DmlKernelCachePolicy::Never>>::
        WithHostMemoryArguments<
            ops::RngReadAndSkip::Argument::resource,
            ops::RngReadAndSkip::Argument::alg,
            ops::RngReadAndSkip::Argument::delta>;

    K::Register();
}

void RegisterStatelessRandomGetKeyCounterAlg()
{
    using K = KernelDefinition<
        ops::StatelessRandomGetKeyCounterAlg,
        GetKeyCounterAlgOp>::
        WithHostMemoryArguments<
            ops::StatelessRandomGetKeyCounterAlg::Argument::seed,
            ops::StatelessRandomGetKeyCounterAlg::Argument::key,
            ops::StatelessRandomGetKeyCounterAlg::Argument::counter,
            ops::StatelessRandomGetKeyCounterAlg::Argument::alg>;

    K::Register();
}

void RegisterStatelessRandomGetKeyCounter()
{
    using K =
        KernelDefinition<ops::StatelessRandomGetKeyCounter, GetKeyCounterOp>::
            WithHostMemoryArguments<
                ops::StatelessRandomGetKeyCounter::Argument::seed,
                ops::StatelessRandomGetKeyCounter::Argument::key,
                ops::StatelessRandomGetKeyCounter::Argument::counter>;

    K::Register();
}

void RegisterGetAlgOp()
{
    using K = KernelDefinition<ops::StatelessRandomGetAlg, GetAlgOp>::
        WithHostMemoryArguments<ops::StatelessRandomGetAlg::Argument::alg>;

    K::Register();
}

void RegisterKernels_Random()
{
    RegisterStatelessRandomUniform();
    RegisterStatelessRandomUniformV2();
    RegisterStatelessRandomUniformInt();
    RegisterStatelessRandomUniformIntV2();
    RegisterRandomUniform();
    RegisterRandomUniformInt();
    RegisterRandomStandardNormal();
    RegisterTruncatedNormal();
    RegisterRngSkip();
    RegisterRngReadAndSkip();
    RegisterStatelessRandomGetKeyCounterAlg();
    RegisterStatelessRandomGetKeyCounter();
    RegisterGetAlgOp();
}

} // namespace tfdml