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
#include "tfdml/kernels/pch.h"
#include "tfdml/runtime_adapter/guarded_philox_random.h"
#include "tfdml/runtime_adapter/stateless_random_ops.h"

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

// Produces a uniform distribution of integers in the range [min_value,
// max_value). See UniformDistribution<Generator, int32> from
// random_distributions.h. Requires min_value < max_value.
dml::Expression UniformInt(
    dml::Graph& graph,
    dml::Expression input_state,
    int32_t min_value,
    int32_t max_value,
    uint32_t element_count)
{
    dml::TensorDimensions shape = {1, 1, 1, element_count};

    auto generator = dml::RandomGenerator(input_state, shape, false);
    auto random_bits = generator.values;

    auto lo = dml::ScalarTensor(graph, min_value, shape);

    uint32_t max_value_unsigned = static_cast<uint32_t>(max_value);
    uint32_t min_value_unsigned = static_cast<uint32_t>(min_value);
    uint32_t range_value = max_value_unsigned - min_value_unsigned;
    auto range = dml::ScalarTensor(graph, range_value, shape);

    return SignedAdd(lo, random_bits % range);
}

class StatelessRandomUniformInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

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

    const TensorShape& GetOutputShape() const { return output_shape_; }
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

using InitHelper = StatelessRandomUniformInitHelper;

class StatelessRandomUniformShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

class DmlStatelessRandomUniformKernel : public DmlKernel
{
    std::array<uint32_t, 6> input_state_;

  public:
    using InitHelper = tfdml::InitHelper;

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

        // Flatten output shape for DirectML.
        DmlTensorInfo output_info;
        output_info.kernel_index = 0;
        std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_elements};
        output_info.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_sizes,
            output_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {state_info};
        tensors.outputs = {output_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_state = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;
        if (ctx->GetOutputDataType(0) == TF_FLOAT)
        {
            result = UniformFloat(scope, input_state, num_elements);
        }
        else if (ctx->GetOutputDataType(0) == TF_HALF)
        {
            result = UniformHalf(scope, input_state, num_elements);
        }
        else
        {
            assert(ctx->GetOutputDataType(0) == TF_INT32);
            int min_value = ctx->GetConstantInputTensor(2).base<int32_t>()[0];
            int max_value = ctx->GetConstantInputTensor(3).base<int32_t>()[0];
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

        // Flatten output shape for DirectML.
        DmlTensorInfo output_info;
        output_info.kernel_index = 0;
        std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_output_elements_};
        output_info.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_sizes,
            output_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {state_info};
        tensors.outputs = {output_info};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_state = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;
        if (ctx->GetOutputDataType(0) == TF_FLOAT)
        {
            result = UniformFloat(scope, input_state, num_output_elements_);
        }
        else if (ctx->GetOutputDataType(0) == TF_HALF)
        {
            result = UniformHalf(scope, input_state, num_output_elements_);
        }
        else
        {
            assert(ctx->GetOutputDataType(0) == TF_INT32);
            int min_value = ctx->GetConstantInputTensor(1).base<int32_t>()[0];
            int max_value = ctx->GetConstantInputTensor(2).base<int32_t>()[0];
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
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));

        TF_Graph* graph = TF_NewGraph();
        auto graph_cleanup =
            absl::MakeCleanup([graph] { TF_DeleteGraph(graph); });

        // Initialize the placeholder that sets the shape for the random op on
        // the CPU
        TF_OperationDescription* shape_desc =
            TF_NewOperation(graph, "Placeholder", "shape");
        TF_SetDevice(shape_desc, "/device:CPU");

        TF_DataType shape_tensor_dtype;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &shape_tensor_dtype));
        TF_SetAttrType(shape_desc, "dtype", shape_tensor_dtype);

        Status status;
        shape_op_ = TF_FinishOperation(shape_desc, status.raw());
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

        TF_DataType output_dtype;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &output_dtype));

        // Initialize the random op on the CPU
        TF_OperationDescription* random_desc = TF_NewOperation(
            graph,
            emulated_kernel_name,
            "DmlEmulatedPhiloxRandomKernel");
        TF_SetDevice(random_desc, "/device:CPU");
        TF_AddInput(random_desc, TF_Output{shape_op_, 0});
        TF_SetAttrType(random_desc, "dtype", output_dtype);
        TF_SetAttrType(random_desc, "T", shape_tensor_dtype);
        TF_SetAttrInt(random_desc, "seed", seed_);
        TF_SetAttrInt(random_desc, "seed2", seed2_);

        random_op_ = TF_FinishOperation(random_desc, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // Create a new session that will be executed on the CPU
        TF_SessionOptions* opts = TF_NewSessionOptions();
        auto session_opts_cleanup =
            absl::MakeCleanup([opts] { TF_DeleteSessionOptions(opts); });

        sess_ = TF_NewSession(graph, opts, status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlEmulatedPhiloxRandomKernel() override
    {
        if (sess_)
        {
            Status status;
            TF_DeleteSession(sess_, status.raw());
            TF_CHECK_OK(status);
        }
    }

    void Compute(OpKernelContext* ctx)
    {
        Tensor shape_tensor = ctx->input(0);
        const TensorShape& input_shape = shape_tensor.shape();

        TF_Output feeds[] = {TF_Output{shape_op_, 0}};
        TF_Tensor* feedValues[] = {shape_tensor.raw()};
        TF_Output fetches[] = {TF_Output{random_op_, 0}};
        TF_Tensor* fetchValues[] = {nullptr};

        Status status;
        TF_SessionRun(
            sess_,
            nullptr,
            feeds,
            feedValues,
            1,
            fetches,
            fetchValues,
            1,
            nullptr,
            0,
            nullptr,
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        TensorShape output_shape;
        OP_REQUIRES_OK(
            ctx,
            TensorShapeUtils::MakeShape(shape_tensor, &output_shape));

        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, output_shape);
        OP_REQUIRES_OK(ctx, status_or_output.status());

        Tensor host_output(fetchValues[0]);
        ctx->device()->CopyCPUTensorToDevice(
            &host_output,
            &status_or_output.ValueOrDie());
    }

  private:
    int seed_;
    int seed2_;
    TF_Operation* shape_op_ = nullptr;
    TF_Operation* random_op_ = nullptr;
    TF_Session* sess_ = nullptr;
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

void RegisterStatelessRandomUniformInt()
{
    using K = KernelDefinition<
        ops::StatelessRandomUniformInt,
        DmlKernelWrapper<
            DmlStatelessRandomUniformKernel,
            StatelessRandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::StatelessRandomUniformInt::Argument::shape,
            ops::StatelessRandomUniformInt::Argument::seed,
            ops::StatelessRandomUniformInt::Argument::minval,
            ops::StatelessRandomUniformInt::Argument::maxval>;

    RegisterWithTypes<
        K,
        ops::StatelessRandomUniformInt::Attribute::dtype,
        TF_INT32>();
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
    using K = KernelDefinition<
        ops::RandomUniformInt,
        DmlPhiloxWrapper<DmlRandomUniformKernel, RandomUniformShapeHelper>>::
        WithHostMemoryArguments<
            ops::RandomUniformInt::Argument::shape,
            ops::RandomUniformInt::Argument::minval,
            ops::RandomUniformInt::Argument::maxval>::
            WithTypeConstraint<ops::RandomUniformInt::Attribute::T, TF_INT32>;

    RegisterWithTypes<K, ops::RandomUniformInt::Attribute::Tout, TF_INT32>();
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

void RegisterKernels_Random()
{
    RegisterStatelessRandomUniform();
    RegisterStatelessRandomUniformInt();
    RegisterRandomUniform();
    RegisterRandomUniformInt();
    RegisterRandomStandardNormal();
    RegisterTruncatedNormal();
}

} // namespace tfdml