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
#include "tfdml/runtime_adapter/variable_lock.h"
#include <cfloat>

namespace tfdml
{

// Provides common implementations for the training kernels. These training
// kernels all generally follow the same pattern:
//   * They modify their input tensors as opposed to producing outputs. This
//     requires staging in temporary buffers, and then copying back to the
//     input.
//   * They have resource- and non-resource versions. The non-resource versions
//     use ref tensors which require some additional logic.
//   * Because we want to maintain common code between the resource and
//     non-resource versions, retrieval of input tensors requires some special
//     handling due to differences between resource tensors vs refs tensors.
//

template <int num_input_refs>
class TrainingInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
        }

        bool use_locking;
        TF_DataType dtype;
    };

    TrainingInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : use_exclusive_lock_(attr->use_locking && ctx->num_outputs() == 0),
          dtype_(attr->dtype)
    {
    }

    bool UseExclusiveLock() const { return use_exclusive_lock_; }
    TF_DataType GetDataType() const { return dtype_; }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        constexpr bool lock_held = false;
        constexpr bool is_variant = false;

        for (int i = 0; i < num_input_refs; ++i)
        {
            Tensor input_ref_tensor;
            Status status = ctx->GetInputTensorFromVariable(
                i,
                lock_held,
                is_variant,
                &input_ref_tensor);
            CHECK(status.ok());

            if (input_ref_tensor.NumElements() == 0)
            {
                return true;
            }
        }

        for (int i = num_input_refs; i < ctx->num_inputs(); ++i)
        {
            if (ctx->input(i).NumElements() == 0)
            {
                return true;
            }
        }

        for (const auto& output_shape : output_shapes)
        {
            if (output_shape.num_elements() == 0)
            {
                return true;
            }
        }

        return false;
    }

  private:
    bool use_exclusive_lock_;
    TF_DataType dtype_;
};

// Helper class to ensure that variable tensors are only ever accessed under
// the lock.
class VariableTensorAccessor
{
  public:
    VariableTensorAccessor(
        OpKernelContext* op_ctx,
        bool use_exclusive_lock,
        VariableLock&& locks)
        : op_ctx_(op_ctx),
          use_exclusive_lock_(use_exclusive_lock),
          locks_(std::move(locks))
    {
    }

    Tensor Get(int index) const
    {
        constexpr bool is_variant = false;

        Tensor tensor;
        TF_CHECK_OK(op_ctx_->GetInputTensorFromVariable(
            index,
            use_exclusive_lock_,
            is_variant,
            &tensor));
        CHECK(tensor.IsInitialized());

        return tensor;
    }

    TensorShape GetShape(int index) const { return Get(index).shape(); }

  private:
    OpKernelContext* op_ctx_;
    bool use_exclusive_lock_;
    VariableLock locks_;
};

// A helper overload of DmlKernel::GetTrainingTensorInfos that constructs a
// DmlKernelTensors which conforms to the pattern followed by all training
// ops. (i.e. broadcasted inputs, uniform datatypes, and graph input/output
// order conforming to kernel definition)
static DmlKernelTensors GetTrainingTensorInfos(
    DmlKernelConstruction* ctx,
    const absl::optional<TensorShape>& desired_shape,
    absl::Span<const TensorShape> input_shapes,
    absl::Span<const TensorShape> output_shapes,
    TF_DataType dtype)
{
    uint32_t input_count = static_cast<uint32_t>(input_shapes.size());
    uint32_t output_count = static_cast<uint32_t>(output_shapes.size());
    CHECK(input_count == ctx->GetInputCount());

    DmlKernelTensors tensors;

    for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx)
    {
        const TensorShape& input_shape = input_shapes[input_idx];
        auto desc = DmlTensorDesc::Create(
            dtype,
            desired_shape ? *desired_shape : input_shape,
            input_shape);
        tensors.inputs.push_back(DmlTensorInfo{std::move(desc), input_idx});
    }

    for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx)
    {
        const TensorShape& output_shape = output_shapes[output_idx];
        auto desc = DmlTensorDesc::Create(
            dtype,
            desired_shape ? *desired_shape : output_shape,
            output_shape);
        tensors.outputs.push_back(DmlTensorInfo{std::move(desc), output_idx});
    }

    return tensors;
}

template <int num_input_refs>
class DmlTrainingKernel : public DmlKernel
{
  public:
    using InitHelper = TrainingInitHelper<num_input_refs>;

    explicit DmlTrainingKernel(
        DmlKernelConstruction* ctx,
        bool use_exclusive_lock,
        bool inplace_allowed = false)
        : inplace_allowed_(inplace_allowed),
          use_exclusive_lock_(use_exclusive_lock)
    {
        // Kernels which use resource variables don't have output tensors
        is_resource_op_ = ctx->GetOutputCount() == 0;
    }

    // Prepares the input tensors at the given indices for access. This must be
    // called prior to GetRefOrResourceTensor.
    void PrepareVariableTensors(
        OpKernelContext* op_ctx,
        absl::Span<const int> input_indices)
    {
        CHECK(!prepare_tensors_called_); // This can only be called once

        // Keep track of which inputs are variables and which are not
        is_variable_input_.resize(op_ctx->num_inputs());
        for (int i : input_indices)
        {
            is_variable_input_[i] = true;
        }

        prepare_tensors_called_ = true;
    }

    VariableTensorAccessor LockVariableTensors(OpKernelContext* op_ctx) const
    {
        // PrepareVariableTensors must be called exactly once before
        // ref/resource tensors can be accessed
        CHECK(prepare_tensors_called_);

        // Retrieve indices of all variable tensors
        std::vector<int> input_indices;
        for (int i = 0; i < is_variable_input_.size(); ++i)
        {
            if (is_variable_input_[i])
            {
                input_indices.push_back(i);
            }
        }

        VariableLock locks(op_ctx, use_exclusive_lock_, input_indices);

        return VariableTensorAccessor(
            op_ctx,
            use_exclusive_lock_,
            std::move(locks));
    }

    // Sets the correct output_refs_forwarding on the DmlKernelTensors,
    // depending on if it's a ref tensor or a resource tensor.
    void MaybeForwardRefInputToRefOutput(
        DmlKernelTensors& tensors,
        uint32_t input,
        uint32_t output)
    {
        // Forwarding refs don't apply to resource tensors
        if (IsResourceOp())
        {
            return;
        }

        if (tensors.output_refs_forwarding.size() <= output)
        {
            tensors.output_refs_forwarding.resize(output + 1);
        }

        tensors.output_refs_forwarding[output] = input;
    }

    // True if this kernel receives resource tensors, false if it receives ref
    // tensors.
    bool IsResourceOp() const { return is_resource_op_; }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        CHECK(prepare_tensors_called_);

        auto* op_ctx = ctx->GetOpKernelContext();
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        // Retrieve the inputs to this kernel. Because some of our inputs might
        // be resource tensors, we need to hold a ref on them until the end of
        // Compute().
        absl::InlinedVector<Tensor, 16> input_tensors;
        absl::InlinedVector<uint32_t, 4> variable_tensor_indices;
        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            if (is_variable_input_[i])
            {
                input_tensors.push_back(var_accessor.Get(i));
                variable_tensor_indices.push_back(i);
            }
            else
            {
                input_tensors.push_back(ctx->GetInputTensor(i));
            }
        }

        // Create input buffers
        absl::InlinedVector<D3D12BufferRegion, 16> input_buffers;
        for (const Tensor& input_tensor : input_tensors)
        {
            input_buffers.push_back(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(input_tensor));
        }

        // Create input bindings
        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 16>
            input_bindings;
        for (const D3D12BufferRegion& input_buffer : input_buffers)
        {
            input_bindings.push_back(input_buffer.GetBufferBinding());
        }

        // Allocate intermediates for variable inputs and produce output
        // bindings. All training ops "output" by writing back to one of their
        // inputs, so we need to buffer the output of our DML op into an
        // intermediate.
        absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 4>
            output_bindings;
        absl::InlinedVector<DmlBuffer, 4> intermediate_buffers;
        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            if (!is_variable_input_[i])
            {
                continue;
            }

            // If inplace is allowed, we don't need intermediate buffers and can
            // directly reuse the input bindings for the output
            if (inplace_allowed_)
            {
                output_bindings.push_back(input_bindings[i]);
            }
            else
            {
                uint64_t intermediate_resource_size =
                    input_tensors[i].TotalBytes();
                DmlBuffer intermediate_buffer =
                    ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
                        op_ctx->raw(),
                        intermediate_resource_size);
                output_bindings.push_back(
                    intermediate_buffer.GetBufferBinding());
                intermediate_buffers.push_back(std::move(intermediate_buffer));
            }
        }

        // Sanity
        assert(
            inplace_allowed_ ||
            output_bindings.size() == intermediate_buffers.size());

        assert(output_bindings.size() == variable_tensor_indices.size());

        auto status_or_gpu_event =
            DmlKernel::Compute(ctx, input_bindings, output_bindings);
        if (!status_or_gpu_event.ok())
        {
            return status_or_gpu_event;
        }

        if (!inplace_allowed_)
        {
            // Copy the intermediate buffers back to their respective inputs.
            // ExecuteOperator already inserts a barrier at the end, so we don't
            // need to add another one here.
            for (uint32_t output_index = 0;
                 output_index < output_bindings.size();
                 ++output_index)
            {
                uint32_t input_index = variable_tensor_indices[output_index];

                auto& input_buffer = input_buffers[input_index];

                ctx->GetDmlDeviceContext()->CopyBufferToBuffer(
                    input_buffer,
                    intermediate_buffers[output_index].Region());
            }

            status_or_gpu_event =
                ctx->GetDmlDeviceContext()->InsertUavBarrier();
        }

        return status_or_gpu_event;
    }

  private:
    bool is_resource_op_ = false;
    bool prepare_tensors_called_ = false;
    bool use_exclusive_lock_ = false;
    bool inplace_allowed_ = false;

    // For each input to this kernel, determines whether it's a variable input
    // or not. The size of this vector is equal to the number of input tensors.
    std::vector<bool> is_variable_input_;
};

template <int num_input_refs>
class NesterovInitHelper : public TrainingInitHelper<num_input_refs>
{
  public:
    struct Attributes : public TrainingInitHelper<num_input_refs>::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : TrainingInitHelper<num_input_refs>::Attributes(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov));
        }

        bool use_nesterov;
    };

    NesterovInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : TrainingInitHelper<num_input_refs>(ctx, attr),
          use_nesterov_(attr->use_nesterov)
    {
    }

    bool UseNesterov() const { return use_nesterov_; }

  private:
    bool use_nesterov_;
};

template <int num_input_refs>
class DmlApplyAdamKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = NesterovInitHelper<num_input_refs>;

    enum TensorIndices
    {
        kVar,
        kM,
        kV,
        kBeta1Power,
        kBeta2Power,
        kLR,
        kBeta1,
        kBeta2,
        kEpsilon,
        kGrad,
    };

    explicit DmlApplyAdamKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock(),
              true /* inplace_allowed */)
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 10);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        // This option is deprecated
        OP_REQUIRES(
            op_ctx,
            !init_helper->UseNesterov(),
            errors::InvalidArgument("use_nesterov is not supported; use "
                                    "tf.keras.optimizers.Nadam instead."));

        this->PrepareVariableTensors(op_ctx, {kVar, kM, kV});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(kVar);
        const TensorShape& m_shape = var_accessor.GetShape(kM);
        const TensorShape& v_shape = var_accessor.GetShape(kV);

        const TensorShape& beta1_power_shape =
            ctx->GetInputTensorShape(kBeta1Power);
        const TensorShape& beta2_power_shape =
            ctx->GetInputTensorShape(kBeta2Power);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(kLR);
        const TensorShape& beta1_shape = ctx->GetInputTensorShape(kBeta1);
        const TensorShape& beta2_shape = ctx->GetInputTensorShape(kBeta2);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(kEpsilon);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_power_shape),
            errors::InvalidArgument(
                "beta1_power is not a scalar: ",
                beta1_power_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta2_power_shape),
            errors::InvalidArgument(
                "beta2_power is not a scalar: ",
                beta2_power_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar : ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_shape),
            errors::InvalidArgument(
                "beta1 is not a scalar: ",
                beta1_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta2_shape),
            errors::InvalidArgument(
                "beta2 is not a scalar: ",
                beta2_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        const TensorShape& grad_shape = ctx->GetInputTensorShape(kGrad);

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(m_shape),
            errors::InvalidArgument(
                "var and m do not have the same shape",
                var_shape.DebugString(),
                " ",
                m_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(v_shape),
            errors::InvalidArgument(
                "var and v do not have the same shape",
                var_shape.DebugString(),
                " ",
                v_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        auto input_shapes = {
            var_shape,
            m_shape,
            v_shape,
            beta1_power_shape,
            beta2_power_shape,
            lr_shape,
            beta1_shape,
            beta2_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, m_shape, v_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            absl::nullopt,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);
        auto outputs = this->GetDmlTensorDescs(tensors.outputs);

        float lr = GetScalarFromTensor(ctx->GetConstantInputTensor(kLR));
        beta1_ = GetScalarFromTensor(ctx->GetConstantInputTensor(kBeta1));
        beta2_ = GetScalarFromTensor(ctx->GetConstantInputTensor(kBeta2));
        float epsilon =
            GetScalarFromTensor(ctx->GetConstantInputTensor(kEpsilon));

        // This scalar tensor isn't actually provided to the kernel; it's
        // computed manually and uploaded
        DmlTensorDesc trainingStepTensor =
            DmlTensorDesc::Create(TF_UINT32, TensorShape{}, TensorShape{});
        DML_TENSOR_DESC trainingStepDesc = trainingStepTensor.GetDmlDesc();

        DML_ADAM_OPTIMIZER_OPERATOR_DESC adam_desc = {};
        adam_desc.InputParametersTensor = &inputs[kVar];
        adam_desc.InputFirstMomentTensor = &inputs[kM];
        adam_desc.InputSecondMomentTensor = &inputs[kV];
        adam_desc.GradientTensor = &inputs[kGrad];
        adam_desc.TrainingStepTensor = &trainingStepDesc;
        adam_desc.OutputParametersTensor = &outputs[0];
        adam_desc.OutputFirstMomentTensor = &outputs[1];
        adam_desc.OutputSecondMomentTensor = &outputs[2];
        adam_desc.LearningRate = lr;
        adam_desc.Beta1 = beta1_;
        adam_desc.Beta2 = beta2_;
        adam_desc.Epsilon = epsilon;

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ADAM_OPTIMIZER, &adam_desc};
        this->Initialize(ctx, std::move(tensors), op_desc);
    }

    static float GetScalarFromTensor(const Tensor& t)
    {
        switch (t.dtype())
        {
        case TF_FLOAT: return t.base<float>()[0];
        case TF_HALF: return static_cast<float>(t.base<Eigen::half>()[0]);
        default: LogFatal("Unsupported datatype");
        }
    }

    static uint32_t GetTrainingStep(float beta, float beta_power)
    {
        // Compute T by taking the log-base-beta of beta_power and rounding off
        float t = log(beta_power + FLT_EPSILON) / log(beta + FLT_EPSILON);
        t = std::round(t);

        return static_cast<uint32_t>(t);
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto* op_ctx = ctx->GetOpKernelContext();
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        // beta1_power and beta2_power are treated specially: they are host
        // memory tensors, but aren't treated as constant CPU input tensors
        // (meaning they don't participate in caching). This means that these
        // values can change on each call to Compute().
        float beta1_power = GetScalarFromTensor(op_ctx->input(kBeta1Power));
        float beta2_power = GetScalarFromTensor(op_ctx->input(kBeta2Power));

        // DML's Adam operator takes a training step count T, whereas TF
        // supplies the values of beta1^T and beta2^T. Therefore we reconstruct
        // T using either beta1 or beta2. We choose whichever one is larger,
        // since that's more likely to produce a numerically-stable result.
        float beta_max = beta1_ > beta2_ ? beta1_ : beta2_;
        float beta_power_max = beta1_ > beta2_ ? beta1_power : beta2_power;
        uint32_t t = GetTrainingStep(beta_max, beta_power_max);

        // Allocate a scalar tensor to hold the training step count
        constexpr bool on_host = false;
        Tensor t_tensor;
        TF_RETURN_IF_ERROR(
            op_ctx->allocate_temp(TF_UINT32, {}, &t_tensor, on_host));

        // Upload the training step scalar T to the GPU buffer
        D3D12BufferRegion dst =
            ctx->GetDmlDeviceContext()->GetBufferForTensor(t_tensor);
        auto src =
            absl::MakeSpan(reinterpret_cast<const uint8_t*>(&t), sizeof(t));
        TF_RETURN_IF_ERROR(
            ctx->GetDmlDeviceContext()->CopyHostToBuffer(dst, src).status());

        Tensor input_tensors[] = {
            var_accessor.Get(kVar),     // InputParameters (var)
            var_accessor.Get(kM),       // InputFirstMoment (m)
            var_accessor.Get(kV),       // InputSecondMoment (v)
            ctx->GetInputTensor(kGrad), // Gradient (grad)
            t_tensor,                   // TrainingStep
        };

        D3D12BufferRegion input_buffers[] = {
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                input_tensors[0]), // InputParameters (var)
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                input_tensors[1]), // InputFirstMoment (m)
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                input_tensors[2]), // InputSecondMoment (v)
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                input_tensors[3]), // Gradient (grad)
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                input_tensors[4]), // TrainingStep
        };

        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            input_buffers[0].GetBufferBinding(), // InputParameters (var)
            input_buffers[1].GetBufferBinding(), // InputFirstMoment (m)
            input_buffers[2].GetBufferBinding(), // InputSecondMoment (v)
            input_buffers[3].GetBufferBinding(), // Gradient (grad)
            input_buffers[4].GetBufferBinding(), // TrainingStep
        };

        // DML's Adam op can be executed in-place, so we simply bind our inputs
        // as outputs
        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            input_bindings[0], // OutputParameters
            input_bindings[1], // OutputFirstMoment
            input_bindings[2], // OutputSecondMoment
        };

        return DmlKernel::Compute(ctx, input_bindings, output_bindings);
    }

  private:
    float beta1_;
    float beta2_;
};

template <typename ShapeHelper, int num_input_refs>
class DmlAdamKernelWrapper
    : public DmlKernelWrapper<DmlApplyAdamKernel<num_input_refs>, ShapeHelper>
{
  public:
    explicit DmlAdamKernelWrapper(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : DmlKernelWrapper<DmlApplyAdamKernel<num_input_refs>, ShapeHelper>(
              ctx,
              node_def),
          node_def_(std::move(node_def))
    {
    }

  protected:
    DmlKernelKey CreateKernelKey(OpKernelContext* ctx) const override
    {
        DmlKernelKey key = {};
        key.op_type_name = std::string(this->type_string());
        key.node_def = node_def_;

        using TensorIndices =
            typename DmlApplyAdamKernel<num_input_refs>::TensorIndices;

        // Add constant CPU input tensors. Note that this doesn't include
        // beta1_power/beta2_power, since those are handled dynamically by the
        // kernel.
        key.input_tensors.push_back({ctx->input(TensorIndices::kLR), true});
        key.input_tensors.push_back({ctx->input(TensorIndices::kBeta1), true});
        key.input_tensors.push_back({ctx->input(TensorIndices::kBeta2), true});
        key.input_tensors.push_back(
            {ctx->input(TensorIndices::kEpsilon), true});

        // For ApplyAdam/ResourceApplyAdam, the only non-constant tensor we need
        // to consider for the purposes of caching is the 'grad' tensor. This is
        // because we know the other tensors (var, m, and v) all must match the
        // shape and datatype of grad.
        const Tensor& grad = ctx->input(TensorIndices::kGrad);
        key.input_tensors.push_back(
            {TensorShapeAndType{grad.shape(), grad.dtype()}, false});

        return key;
    }

  private:
    std::shared_ptr<const NodeDef> node_def_;
};

template <int num_input_refs>
class DmlApplyAdamWithAmsgradKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = typename DmlTrainingKernel<num_input_refs>::InitHelper;

    explicit DmlApplyAdamWithAmsgradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 11);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1, 2, 3});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& m_shape = var_accessor.GetShape(1);
        const TensorShape& v_shape = var_accessor.GetShape(2);
        const TensorShape& vhat_shape = var_accessor.GetShape(3);

        const TensorShape& beta1_power_shape = ctx->GetInputTensorShape(4);
        const TensorShape& beta2_power_shape = ctx->GetInputTensorShape(5);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(6);
        const TensorShape& beta1_shape = ctx->GetInputTensorShape(7);
        const TensorShape& beta2_shape = ctx->GetInputTensorShape(8);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(9);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_power_shape),
            errors::InvalidArgument(
                "beta1_power is not a scalar: ",
                beta1_power_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta2_power_shape),
            errors::InvalidArgument(
                "beta2_power is not a scalar: ",
                beta2_power_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar : ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_shape),
            errors::InvalidArgument(
                "beta1 is not a scalar: ",
                beta1_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta2_shape),
            errors::InvalidArgument(
                "beta2 is not a scalar: ",
                beta2_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        const TensorShape& grad_shape = ctx->GetInputTensorShape(10);

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(m_shape),
            errors::InvalidArgument(
                "var and m do not have the same shape",
                var_shape.DebugString(),
                " ",
                m_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(v_shape),
            errors::InvalidArgument(
                "var and v do not have the same shape",
                var_shape.DebugString(),
                " ",
                v_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        auto input_shapes = {
            var_shape,
            m_shape,
            v_shape,
            vhat_shape,
            beta1_power_shape,
            beta2_power_shape,
            lr_shape,
            beta1_shape,
            beta2_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, m_shape, v_shape, vhat_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            absl::nullopt,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto m = dml::InputTensor(scope, 1, inputs[1]);
        auto v = dml::InputTensor(scope, 2, inputs[2]);
        auto vhat = dml::InputTensor(scope, 3, inputs[3]);
        auto beta1_power = dml::InputTensor(scope, 4, inputs[4]);
        auto beta2_power = dml::InputTensor(scope, 5, inputs[5]);
        auto lr = dml::InputTensor(scope, 6, inputs[6]);
        auto beta1 = dml::InputTensor(scope, 7, inputs[7]);
        auto beta2 = dml::InputTensor(scope, 8, inputs[8]);
        auto epsilon = dml::InputTensor(scope, 9, inputs[9]);
        auto grad = dml::InputTensor(scope, 10, inputs[10]);

        const auto& desired_shape = var.GetOutputDesc().sizes;

        // Broadcasts a scalar into a tensor with our desired shape.
        auto BCast = [&](dml::Expression input)
        {
            // Because we're broadcasting a scalar, just set all strides to 0
            dml::TensorDesc::Dimensions bcast_strides = {0, 0, 0, 0};
            return dml::Reinterpret(input, desired_shape, bcast_strides);
        };

        auto alpha = lr * (dml::Sqrt(1 - beta2_power) / (1 - beta1_power));

        m += (grad - m) * BCast(1 - beta1);
        v += (grad * grad - v) * BCast(1 - beta2);
        vhat = dml::Max(vhat, v);
        var -= m * BCast(alpha) / (dml::Sqrt(vhat) + BCast(epsilon));

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, m, v, vhat});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyAdaMaxKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = typename DmlTrainingKernel<num_input_refs>::InitHelper;

    explicit DmlApplyAdaMaxKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 9);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1, 2});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& m_shape = var_accessor.GetShape(1);
        const TensorShape& v_shape = var_accessor.GetShape(2);

        const TensorShape& beta1_power_shape = ctx->GetInputTensorShape(3);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(4);
        const TensorShape& beta1_shape = ctx->GetInputTensorShape(5);
        const TensorShape& beta2_shape = ctx->GetInputTensorShape(6);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(7);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_power_shape),
            errors::InvalidArgument(
                "beta1_power is not a scalar: ",
                beta1_power_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar : ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta1_shape),
            errors::InvalidArgument(
                "beta1 is not a scalar: ",
                beta1_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta2_shape),
            errors::InvalidArgument(
                "beta2 is not a scalar: ",
                beta2_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        const TensorShape& grad_shape = ctx->GetInputTensorShape(8);

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(m_shape),
            errors::InvalidArgument(
                "var and m do not have the same shape",
                var_shape.DebugString(),
                " ",
                m_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(v_shape),
            errors::InvalidArgument(
                "var and v do not have the same shape",
                var_shape.DebugString(),
                " ",
                v_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            m_shape,
            v_shape,
            beta1_power_shape,
            lr_shape,
            beta1_shape,
            beta2_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, m_shape, v_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto m = dml::InputTensor(scope, 1, inputs[1]);
        auto v = dml::InputTensor(scope, 2, inputs[2]);
        auto beta1_power = dml::InputTensor(scope, 3, inputs[3]);
        auto lr = dml::InputTensor(scope, 4, inputs[4]);
        auto beta1 = dml::InputTensor(scope, 5, inputs[5]);
        auto beta2 = dml::InputTensor(scope, 6, inputs[6]);
        auto epsilon = dml::InputTensor(scope, 7, inputs[7]);
        auto grad = dml::InputTensor(scope, 8, inputs[8]);

        m += (grad - m) * (1 - beta1);
        v = dml::Max(beta2 * v, dml::Abs(grad));
        var -= lr / (1 - beta1_power) * (m / (v + epsilon));

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, m, v});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyGradientDescentKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = typename DmlTrainingKernel<num_input_refs>::InitHelper;

    DmlApplyGradientDescentKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 3);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& alpha_shape = ctx->GetInputTensorShape(1);
        const TensorShape& delta_shape = ctx->GetInputTensorShape(2);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(alpha_shape),
            errors::InvalidArgument(
                "alpha is not a scalar: ",
                alpha_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(delta_shape),
            errors::InvalidArgument(
                "var and delta do not have the same shape",
                var_shape.DebugString(),
                " ",
                delta_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {var_shape, alpha_shape, delta_shape};
        auto output_shapes = {var_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto alpha = dml::InputTensor(scope, 1, inputs[1]);
        auto delta = dml::InputTensor(scope, 2, inputs[2]);

        // Alpha is also referred to as the learning rate. Delta is the gradient
        // tensor.
        var -= alpha * delta;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyAdadeltaKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = typename DmlTrainingKernel<num_input_refs>::InitHelper;

    DmlApplyAdadeltaKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 7);
        CHECK(ctx->GetOutputCount() <= 1);

        this->PrepareVariableTensors(op_ctx, {0, 1, 2});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& accum_shape = var_accessor.GetShape(1);
        const TensorShape& accum_update_shape = var_accessor.GetShape(2);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(3);
        const TensorShape& rho_shape = ctx->GetInputTensorShape(4);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(5);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(6);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(rho_shape),
            errors::InvalidArgument(
                "rho is not a scalar: ",
                rho_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(accum_shape),
            errors::InvalidArgument(
                "var and accum do not have the same shape",
                var_shape.DebugString(),
                " ",
                accum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            accum_shape,
            accum_update_shape,
            lr_shape,
            rho_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, accum_shape, accum_update_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto accum = dml::InputTensor(scope, 1, inputs[1]);
        auto accum_update = dml::InputTensor(scope, 2, inputs[2]);
        auto lr = dml::InputTensor(scope, 3, inputs[3]);
        auto rho = dml::InputTensor(scope, 4, inputs[4]);
        auto epsilon = dml::InputTensor(scope, 5, inputs[5]);
        auto grad = dml::InputTensor(scope, 6, inputs[6]);

        accum = accum * rho + grad * grad * (1.0f - rho);
        // Note: we can't simplify this into just a single sqrt because it
        // introduces too much numerical instability, especially in cases where
        // the denominator approaches zero.
        auto update = dml::Sqrt(accum_update + epsilon) /
                      dml::Sqrt(accum + epsilon) * grad;
        var -= update * lr;
        accum_update = accum_update * rho + update * update * (1.0f - rho);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, accum, accum_update});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class ApplyAdagradInitHelper : public TrainingInitHelper<num_input_refs>
{
  public:
    struct Attributes : public TrainingInitHelper<num_input_refs>::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : TrainingInitHelper<num_input_refs>::Attributes(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots));
        }

        bool update_slots;
    };

    ApplyAdagradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : TrainingInitHelper<num_input_refs>(ctx, attr),
          update_slots_(attr->update_slots)
    {
    }

    bool UpdateSlots() const { return update_slots_; }

  private:
    bool update_slots_;
};

template <int num_input_refs>
class DmlApplyAdagradKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = ApplyAdagradInitHelper<num_input_refs>;

    DmlApplyAdagradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        // ApplyAdagradV2 has an additional input
        CHECK(ctx->GetInputCount() == 4 || ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() <= 1);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const bool has_epsilon = (ctx->GetInputCount() == 5);

        // The "grad" tensor is the 3rd or 4th tensor, depending on whether this
        // is Adagrad or AdagradV2.
        uint32_t grad_index = has_epsilon ? 4u : 3u;

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& accum_shape = var_accessor.GetShape(1);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(grad_index);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(accum_shape),
            errors::InvalidArgument(
                "var and accum do not have the same shape",
                var_shape.DebugString(),
                " ",
                accum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        absl::InlinedVector<TensorShape, 5> input_shapes = {
            var_shape,
            accum_shape,
            lr_shape};

        if (has_epsilon)
        {
            const TensorShape& epsilon_shape = ctx->GetInputTensorShape(3);

            OP_REQUIRES(
                op_ctx,
                TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument(
                    "epsilon is not a scalar: ",
                    epsilon_shape.DebugString()));

            input_shapes.push_back(epsilon_shape);
        }

        input_shapes.push_back(grad_shape);
        auto output_shapes = {var_shape, accum_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto accum = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto grad = dml::InputTensor(scope, grad_index, inputs[grad_index]);

        if (init_helper->UpdateSlots())
        {
            accum += grad * grad;
        }

        // AdagradV2 applies an epsilon before performing the division
        if (has_epsilon)
        {
            auto epsilon = dml::InputTensor(scope, 3, inputs[3]);

            var -= grad * lr / (dml::Sqrt(accum) + epsilon);
        }
        else
        {
            var -= grad * lr / dml::Sqrt(accum);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, accum});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class SparseApplyAdagradInitHelper : public TrainingInitHelper<num_input_refs>
{
  public:
    struct Attributes : public TrainingInitHelper<num_input_refs>::Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
            : TrainingInitHelper<num_input_refs>::Attributes(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots));
        }

        bool update_slots;
    };

    SparseApplyAdagradInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : TrainingInitHelper<num_input_refs>(ctx, attr),
          update_slots_(attr->update_slots)
    {
    }

    bool UpdateSlots() const { return update_slots_; }

  private:
    bool update_slots_;
};

template <int num_input_refs>
class DmlSparseApplyAdagradKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = SparseApplyAdagradInitHelper<num_input_refs>;

    DmlSparseApplyAdagradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        // SparseApplyAdagradV2 has an additional input
        CHECK(ctx->GetInputCount() == 5 || ctx->GetInputCount() == 6);
        CHECK(ctx->GetOutputCount() <= 1);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        OP_REQUIRES(
            op_ctx,
            var_accessor.Get(0).IsInitialized(),
            errors::FailedPrecondition(
                "Attempting to use uninitialized variables: var"));
        OP_REQUIRES(
            op_ctx,
            var_accessor.Get(1).IsInitialized(),
            errors::FailedPrecondition(
                "Attempting to use uninitialized variables: accum"));

        const bool has_epsilon = (ctx->GetInputCount() == 6);

        // The "grad" tensor is the 3rd or 4th tensor, depending on whether this
        // is Adagrad or AdagradV2.
        uint32_t grad_index = has_epsilon ? 4u : 3u;
        uint32_t indices_index = has_epsilon ? 5u : 4u;

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& accum_shape = var_accessor.GetShape(1);
        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(grad_index);
        const TensorShape& indices_shape =
            ctx->GetInputTensorShape(indices_index);

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(accum_shape),
            errors::InvalidArgument(
                "var and accum do not have the same shape",
                var_shape.DebugString(),
                " ",
                accum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsVectorOrHigher(var_shape),
            errors::InvalidArgument("var must be at least 1 dimensional"));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsVector(indices_shape),
            errors::InvalidArgument("indices must be one-dimensional"));
        int64_t inner_dim = 1;
        for (int d = 1; d < var_shape.dims(); d++)
        {
            OP_REQUIRES(
                op_ctx,
                var_shape.dim_size(d) == grad_shape.dim_size(d),
                errors::InvalidArgument(
                    absl::StrCat("var and grad must match in dimension ", d)));
            inner_dim *= grad_shape.dim_size(d);
        }
        const int N = indices_shape.dim_size(0);
        OP_REQUIRES(
            op_ctx,
            grad_shape.dim_size(0) == N,
            errors::InvalidArgument("grad must be the same size as indices in "
                                    "the first dimension."));

        OP_REQUIRES(
            op_ctx,
            inner_dim > 0,
            errors::InvalidArgument(
                "Inner dimension should be greater than zero."));

        DmlKernelTensors tensors;

        auto var_accum_desc = DmlTensorDesc::Create(
            init_helper->GetDataType(),
            var_shape,
            var_shape);
        tensors.inputs.push_back(DmlTensorInfo{var_accum_desc, 0});
        tensors.inputs.push_back(DmlTensorInfo{std::move(var_accum_desc), 1});

        auto lr_desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(2),
            grad_shape,
            lr_shape);
        tensors.inputs.push_back(DmlTensorInfo{std::move(lr_desc), 2});

        if (has_epsilon)
        {
            const TensorShape& epsilon_shape = ctx->GetInputTensorShape(3);

            OP_REQUIRES(
                op_ctx,
                TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument(
                    "epsilon is not a scalar: ",
                    epsilon_shape.DebugString()));

            auto epsilon_desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(3),
                grad_shape,
                epsilon_shape);
            tensors.inputs.push_back(DmlTensorInfo{std::move(epsilon_desc), 3});
        }

        auto grad_desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(grad_index),
            grad_shape,
            grad_shape);
        tensors.inputs.push_back(
            DmlTensorInfo{std::move(grad_desc), grad_index});

        auto indices_desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(indices_index),
            indices_shape,
            indices_shape);
        tensors.inputs.push_back(
            DmlTensorInfo{std::move(indices_desc), indices_index});

        auto output_desc = DmlTensorDesc::Create(
            init_helper->GetDataType(),
            var_shape,
            var_shape);
        tensors.outputs.push_back(DmlTensorInfo{output_desc, 0});
        tensors.outputs.push_back(DmlTensorInfo{std::move(output_desc), 1});

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto accum = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto grad = dml::InputTensor(scope, grad_index, inputs[grad_index]);

        auto indices =
            dml::InputTensor(scope, indices_index, inputs[indices_index]);

        if (var_shape.dims() > 1)
        {
            auto indices_sizes = indices.GetOutputDesc().sizes;
            std::swap(
                indices_sizes[indices_sizes.size() - 1],
                indices_sizes[indices_sizes.size() - 2]);

            dml::TensorStrides indices_strides(indices_sizes.size());
            indices_strides[indices_sizes.size() - 2] = 1;

            indices = dml::Reinterpret(indices, indices_sizes, indices_strides);
        }

        const int indices_dim_count = var_shape.dims() > 1 ? 2 : 1;

        auto gathered_var =
            dml::GatherND(var, indices, var_shape.dims(), indices_dim_count, 0);
        auto gathered_accum = dml::GatherND(
            accum,
            indices,
            accum_shape.dims(),
            indices_dim_count,
            0);

        if (init_helper->UpdateSlots())
        {
            gathered_accum += grad * grad;
        }

        // AdagradV2 applies an epsilon before performing the division
        if (has_epsilon)
        {
            auto epsilon = dml::InputTensor(scope, 3, inputs[3]);

            gathered_var -= grad * lr / (dml::Sqrt(gathered_accum) + epsilon);
        }
        else
        {
            gathered_var -= grad * lr / dml::Sqrt(gathered_accum);
        }

        auto scattered_var = dml::ScatterND(
            var,
            indices,
            gathered_var,
            var_shape.dims(),
            indices_dim_count);
        auto scattered_accum = dml::ScatterND(
            accum,
            indices,
            gathered_accum,
            accum_shape.dims(),
            indices_dim_count);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(
                DML_EXECUTION_FLAG_NONE,
                {scattered_var, scattered_accum});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyMomentumKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = NesterovInitHelper<num_input_refs>;

    explicit DmlApplyMomentumKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& accum_shape = var_accessor.GetShape(1);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(3);
        const TensorShape& momentum_shape = ctx->GetInputTensorShape(4);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(accum_shape),
            errors::InvalidArgument(
                "var and accum do not have the same shape",
                var_shape.DebugString(),
                " ",
                accum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(momentum_shape),
            errors::InvalidArgument(
                "momentum is not a scalar: ",
                momentum_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes =
            {var_shape, accum_shape, lr_shape, grad_shape, momentum_shape};
        auto output_shapes = {var_shape, accum_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto accum = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto grad = dml::InputTensor(scope, 3, inputs[3]);
        auto momentum = dml::InputTensor(scope, 4, inputs[4]);

        accum = accum * momentum + grad;

        if (init_helper->UseNesterov())
        {
            var -= grad * lr + accum * momentum * lr;
        }
        else
        {
            var -= accum * lr;
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, accum});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyKerasMomentumKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = NesterovInitHelper<num_input_refs>;

    explicit DmlApplyKerasMomentumKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& accum_shape = var_accessor.GetShape(1);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(3);
        const TensorShape& momentum_shape = ctx->GetInputTensorShape(4);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(accum_shape),
            errors::InvalidArgument(
                "var and accum do not have the same shape",
                var_shape.DebugString(),
                " ",
                accum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(momentum_shape),
            errors::InvalidArgument(
                "momentum is not a scalar: ",
                momentum_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes =
            {var_shape, accum_shape, lr_shape, grad_shape, momentum_shape};
        auto output_shapes = {var_shape, accum_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto accum = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto grad = dml::InputTensor(scope, 3, inputs[3]);
        auto momentum = dml::InputTensor(scope, 4, inputs[4]);

        accum = accum * momentum - grad * lr;

        if (init_helper->UseNesterov())
        {
            var += accum * momentum - grad * lr;
        }
        else
        {
            var += accum;
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, accum});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyRMSPropKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = TrainingInitHelper<num_input_refs>;

    explicit DmlApplyRMSPropKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 8);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1, 2});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& ms_shape = var_accessor.GetShape(1);
        const TensorShape& mom_shape = var_accessor.GetShape(2);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(3);
        const TensorShape& rho_shape = ctx->GetInputTensorShape(4);
        const TensorShape& momentum_shape = ctx->GetInputTensorShape(5);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(6);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(7);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar : ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(rho_shape),
            errors::InvalidArgument(
                "rho is not a scalar: ",
                rho_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(momentum_shape),
            errors::InvalidArgument(
                "momentum is not a scalar: ",
                momentum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(ms_shape),
            errors::InvalidArgument(
                "var and ms do not have the same shape",
                var_shape.DebugString(),
                " ",
                ms_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(mom_shape),
            errors::InvalidArgument(
                "var and mom do not have the same shape",
                var_shape.DebugString(),
                " ",
                mom_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            ms_shape,
            mom_shape,
            lr_shape,
            rho_shape,
            momentum_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, ms_shape, mom_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto ms = dml::InputTensor(scope, 1, inputs[1]);
        auto mom = dml::InputTensor(scope, 2, inputs[2]);
        auto lr = dml::InputTensor(scope, 3, inputs[3]);
        auto rho = dml::InputTensor(scope, 4, inputs[4]);
        auto momentum = dml::InputTensor(scope, 5, inputs[5]);
        auto epsilon = dml::InputTensor(scope, 6, inputs[6]);
        auto grad = dml::InputTensor(scope, 7, inputs[7]);

        ms += (grad * grad - ms) * (1 - rho);
        mom = (mom * momentum) + (grad * lr) / dml::Sqrt(ms + epsilon);
        var -= mom;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, ms, mom});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyCenteredRMSPropKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = NesterovInitHelper<num_input_refs>;

    explicit DmlApplyCenteredRMSPropKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 9);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1, 2, 3});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& mg_shape = var_accessor.GetShape(1);
        const TensorShape& ms_shape = var_accessor.GetShape(2);
        const TensorShape& mom_shape = var_accessor.GetShape(3);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(4);
        const TensorShape& rho_shape = ctx->GetInputTensorShape(5);
        const TensorShape& momentum_shape = ctx->GetInputTensorShape(6);
        const TensorShape& epsilon_shape = ctx->GetInputTensorShape(7);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(8);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar : ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(rho_shape),
            errors::InvalidArgument(
                "rho is not a scalar: ",
                rho_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(momentum_shape),
            errors::InvalidArgument(
                "momentum is not a scalar: ",
                momentum_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(epsilon_shape),
            errors::InvalidArgument(
                "epsilon is not a scalar: ",
                epsilon_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(mg_shape),
            errors::InvalidArgument(
                "var and mg do not have the same shape",
                var_shape.DebugString(),
                " ",
                ms_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(ms_shape),
            errors::InvalidArgument(
                "var and ms do not have the same shape",
                var_shape.DebugString(),
                " ",
                ms_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(mom_shape),
            errors::InvalidArgument(
                "var and mom do not have the same shape",
                var_shape.DebugString(),
                " ",
                mom_shape.DebugString()));

        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            mg_shape,
            ms_shape,
            mom_shape,
            lr_shape,
            rho_shape,
            momentum_shape,
            epsilon_shape,
            grad_shape};
        auto output_shapes = {var_shape, mg_shape, ms_shape, mom_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto mg = dml::InputTensor(scope, 1, inputs[1]);
        auto ms = dml::InputTensor(scope, 2, inputs[2]);
        auto mom = dml::InputTensor(scope, 3, inputs[3]);
        auto lr = dml::InputTensor(scope, 4, inputs[4]);
        auto rho = dml::InputTensor(scope, 5, inputs[5]);
        auto momentum = dml::InputTensor(scope, 6, inputs[6]);
        auto epsilon = dml::InputTensor(scope, 7, inputs[7]);
        auto grad = dml::InputTensor(scope, 8, inputs[8]);

        ms += (grad * grad - ms) * (1 - rho);
        mg += (grad - mg) * (1 - rho);
        auto denom = (ms - mg * mg) + epsilon;
        mom = (mom * momentum) + (grad * lr) / dml::Sqrt(denom);
        var -= mom;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, mg, ms, mom});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyAddSignKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = TrainingInitHelper<num_input_refs>;

    explicit DmlApplyAddSignKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 7);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& m_shape = var_accessor.GetShape(1);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& alpha_shape = ctx->GetInputTensorShape(3);
        const TensorShape& sign_decay_shape = ctx->GetInputTensorShape(4);
        const TensorShape& beta_shape = ctx->GetInputTensorShape(5);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(6);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(alpha_shape),
            errors::InvalidArgument(
                "alpha is not a scalar: ",
                alpha_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(alpha_shape),
            errors::InvalidArgument(
                "sign_decay is not a scalar: ",
                sign_decay_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta_shape),
            errors::InvalidArgument(
                "beta is not a scalar: ",
                beta_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(m_shape),
            errors::InvalidArgument(
                "var and m do not have the same shape",
                var_shape.DebugString(),
                " ",
                m_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            m_shape,
            lr_shape,
            alpha_shape,
            sign_decay_shape,
            beta_shape,
            grad_shape};
        auto output_shapes = {var_shape, m_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto m = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto alpha = dml::InputTensor(scope, 3, inputs[3]);
        auto sign_decay = dml::InputTensor(scope, 4, inputs[4]);
        auto beta = dml::InputTensor(scope, 5, inputs[5]);
        auto grad = dml::InputTensor(scope, 6, inputs[6]);

        m = m * beta + grad * (1 - beta);
        auto sign_gm = dml::Sign(grad) * dml::Sign(m);
        var -= lr * (alpha + sign_decay * sign_gm) * grad;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, m});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <int num_input_refs>
class DmlApplyPowerSignKernel : public DmlTrainingKernel<num_input_refs>
{
  public:
    using InitHelper = TrainingInitHelper<num_input_refs>;

    explicit DmlApplyPowerSignKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
        : DmlTrainingKernel<num_input_refs>(
              ctx,
              init_helper->UseExclusiveLock())
    {
        auto* op_ctx = ctx->GetOpKernelContext();

        CHECK(ctx->GetInputCount() == 7);
        CHECK(ctx->GetOutputCount() == 1 || ctx->GetOutputCount() == 0);

        this->PrepareVariableTensors(op_ctx, {0, 1});
        VariableTensorAccessor var_accessor = this->LockVariableTensors(op_ctx);

        const TensorShape& var_shape = var_accessor.GetShape(0);
        const TensorShape& m_shape = var_accessor.GetShape(1);

        const TensorShape& lr_shape = ctx->GetInputTensorShape(2);
        const TensorShape& logbase_shape = ctx->GetInputTensorShape(3);
        const TensorShape& sign_decay_shape = ctx->GetInputTensorShape(4);
        const TensorShape& beta_shape = ctx->GetInputTensorShape(5);
        const TensorShape& grad_shape = ctx->GetInputTensorShape(6);

        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(lr_shape),
            errors::InvalidArgument(
                "lr is not a scalar: ",
                lr_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(logbase_shape),
            errors::InvalidArgument(
                "logbase is not a scalar: ",
                logbase_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(logbase_shape),
            errors::InvalidArgument(
                "sign_decay is not a scalar: ",
                sign_decay_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            TensorShapeUtils::IsScalar(beta_shape),
            errors::InvalidArgument(
                "beta is not a scalar: ",
                beta_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(m_shape),
            errors::InvalidArgument(
                "var and m do not have the same shape",
                var_shape.DebugString(),
                " ",
                m_shape.DebugString()));
        OP_REQUIRES(
            op_ctx,
            var_shape.IsSameSize(grad_shape),
            errors::InvalidArgument(
                "var and grad do not have the same shape",
                var_shape.DebugString(),
                " ",
                grad_shape.DebugString()));

        // Broadcast inputs to match var_shape
        const TensorShape& desired_shape = var_shape;

        auto input_shapes = {
            var_shape,
            m_shape,
            lr_shape,
            logbase_shape,
            sign_decay_shape,
            beta_shape,
            grad_shape};
        auto output_shapes = {var_shape, m_shape};
        DmlKernelTensors tensors = GetTrainingTensorInfos(
            ctx,
            desired_shape,
            input_shapes,
            output_shapes,
            init_helper->GetDataType());

        this->MaybeForwardRefInputToRefOutput(tensors, 0, 0);

        auto inputs = this->GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto var = dml::InputTensor(scope, 0, inputs[0]);
        auto m = dml::InputTensor(scope, 1, inputs[1]);
        auto lr = dml::InputTensor(scope, 2, inputs[2]);
        auto logbase = dml::InputTensor(scope, 3, inputs[3]);
        auto sign_decay = dml::InputTensor(scope, 4, inputs[4]);
        auto beta = dml::InputTensor(scope, 5, inputs[5]);
        auto grad = dml::InputTensor(scope, 6, inputs[6]);

        m = m * beta + grad * (1 - beta);
        auto sign_gm = dml::Sign(grad) * dml::Sign(m);
        auto grad_scale = dml::Exp(logbase * sign_decay * sign_gm);
        var -= lr * grad_scale * grad;

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {var, m});

        this->Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterApplyAdam()
{
    using K = KernelDefinition<
        ops::ApplyAdam,
        DmlAdamKernelWrapper<GetOutputShapeAsRefInputShapeHelper, 3>>::
        WithHostMemoryArguments<
            ops::ApplyAdam::Argument::beta1_power,
            ops::ApplyAdam::Argument::beta2_power,
            ops::ApplyAdam::Argument::lr,
            ops::ApplyAdam::Argument::beta1,
            ops::ApplyAdam::Argument::beta2,
            ops::ApplyAdam::Argument::epsilon>;

    RegisterWithTypes<K, ops::ApplyAdam::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyAdam()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdam,
        DmlAdamKernelWrapper<NoOutputShapeHelper, 0>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdam::Argument::var,
            ops::ResourceApplyAdam::Argument::m,
            ops::ResourceApplyAdam::Argument::v,
            ops::ResourceApplyAdam::Argument::beta1_power,
            ops::ResourceApplyAdam::Argument::beta2_power,
            ops::ResourceApplyAdam::Argument::lr,
            ops::ResourceApplyAdam::Argument::beta1,
            ops::ResourceApplyAdam::Argument::beta2,
            ops::ResourceApplyAdam::Argument::epsilon>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdam::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyAdamWithAmsgrad()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdamWithAmsgrad,
        DmlKernelWrapper<
            DmlApplyAdamWithAmsgradKernel<0>,
            NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdamWithAmsgrad::Argument::var,
            ops::ResourceApplyAdamWithAmsgrad::Argument::m,
            ops::ResourceApplyAdamWithAmsgrad::Argument::v>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdamWithAmsgrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyAdaMax()
{
    using K = KernelDefinition<
        ops::ApplyAdaMax,
        DmlKernelWrapper<
            DmlApplyAdaMaxKernel<3>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyAdaMax::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyAdaMax()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdaMax,
        DmlKernelWrapper<DmlApplyAdaMaxKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdaMax::Argument::var,
            ops::ResourceApplyAdaMax::Argument::m,
            ops::ResourceApplyAdaMax::Argument::v>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdaMax::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyGradientDescent()
{
    using K = KernelDefinition<
        ops::ApplyGradientDescent,
        DmlKernelWrapper<
            DmlApplyGradientDescentKernel<1>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ApplyGradientDescent::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyGradientDescent()
{
    using K = KernelDefinition<
        ops::ResourceApplyGradientDescent,
        DmlKernelWrapper<
            DmlApplyGradientDescentKernel<0>,
            NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyGradientDescent::Argument::var>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyGradientDescent::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyAdadelta()
{
    using K = KernelDefinition<
        ops::ApplyAdadelta,
        DmlKernelWrapper<
            DmlApplyAdadeltaKernel<3>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyAdadelta::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyAdadelta()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdadelta,
        DmlKernelWrapper<DmlApplyAdadeltaKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdadelta::Argument::var,
            ops::ResourceApplyAdadelta::Argument::accum,
            ops::ResourceApplyAdadelta::Argument::accum_update>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdadelta::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyAdagrad()
{
    using K = KernelDefinition<
        ops::ApplyAdagrad,
        DmlKernelWrapper<
            DmlApplyAdagradKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyAdagrad::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterApplyAdagradV2()
{
    using K = KernelDefinition<
        ops::ApplyAdagradV2,
        DmlKernelWrapper<
            DmlApplyAdagradKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ApplyAdagradV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyAdagrad()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdagrad,
        DmlKernelWrapper<DmlApplyAdagradKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdagrad::Argument::var,
            ops::ResourceApplyAdagrad::Argument::accum>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdagrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyAdagradV2()
{
    using K = KernelDefinition<
        ops::ResourceApplyAdagradV2,
        DmlKernelWrapper<DmlApplyAdagradKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAdagradV2::Argument::var,
            ops::ResourceApplyAdagradV2::Argument::accum>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAdagradV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterSparseApplyAdagrad()
{
    using K = KernelDefinition<
        ops::SparseApplyAdagrad,
        DmlKernelWrapper<
            DmlSparseApplyAdagradKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::SparseApplyAdagrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterSparseApplyAdagradV2()
{
    using K = KernelDefinition<
        ops::SparseApplyAdagradV2,
        DmlKernelWrapper<
            DmlSparseApplyAdagradKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::SparseApplyAdagradV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceSparseApplyAdagrad()
{
    using K = KernelDefinition<
        ops::ResourceSparseApplyAdagrad,
        DmlKernelWrapper<DmlSparseApplyAdagradKernel<0>, NoOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ResourceSparseApplyAdagrad::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceSparseApplyAdagradV2()
{
    using K = KernelDefinition<
        ops::ResourceSparseApplyAdagradV2,
        DmlKernelWrapper<DmlSparseApplyAdagradKernel<0>, NoOutputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ResourceSparseApplyAdagradV2::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyMomentum()
{
    using K = KernelDefinition<
        ops::ApplyMomentum,
        DmlKernelWrapper<
            DmlApplyMomentumKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyMomentum::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyMomentum()
{
    using K = KernelDefinition<
        ops::ResourceApplyMomentum,
        DmlKernelWrapper<DmlApplyMomentumKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyMomentum::Argument::var,
            ops::ResourceApplyMomentum::Argument::accum>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyMomentum::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyKerasMomentum()
{
    using K = KernelDefinition<
        ops::ResourceApplyKerasMomentum,
        DmlKernelWrapper<DmlApplyKerasMomentumKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyKerasMomentum::Argument::var,
            ops::ResourceApplyKerasMomentum::Argument::accum>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyKerasMomentum::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyRMSProp()
{
    using K = KernelDefinition<
        ops::ApplyRMSProp,
        DmlKernelWrapper<
            DmlApplyRMSPropKernel<3>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyRMSProp::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyRMSProp()
{
    using K = KernelDefinition<
        ops::ResourceApplyRMSProp,
        DmlKernelWrapper<DmlApplyRMSPropKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyRMSProp::Argument::var,
            ops::ResourceApplyRMSProp::Argument::ms,
            ops::ResourceApplyRMSProp::Argument::mom>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyRMSProp::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyCenteredRMSProp()
{
    using K = KernelDefinition<
        ops::ApplyCenteredRMSProp,
        DmlKernelWrapper<
            DmlApplyCenteredRMSPropKernel<4>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ApplyCenteredRMSProp::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyCenteredRMSProp()
{
    using K = KernelDefinition<
        ops::ResourceApplyCenteredRMSProp,
        DmlKernelWrapper<
            DmlApplyCenteredRMSPropKernel<0>,
            NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyCenteredRMSProp::Argument::var,
            ops::ResourceApplyCenteredRMSProp::Argument::ms,
            ops::ResourceApplyCenteredRMSProp::Argument::mom>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyCenteredRMSProp::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyAddSign()
{
    using K = KernelDefinition<
        ops::ApplyAddSign,
        DmlKernelWrapper<
            DmlApplyAddSignKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<K, ops::ApplyAddSign::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterResourceApplyAddSign()
{
    using K = KernelDefinition<
        ops::ResourceApplyAddSign,
        DmlKernelWrapper<DmlApplyAddSignKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyAddSign::Argument::var,
            ops::ResourceApplyAddSign::Argument::m>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyAddSign::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterApplyPowerSign()
{
    using K = KernelDefinition<
        ops::ApplyPowerSign,
        DmlKernelWrapper<
            DmlApplyPowerSignKernel<2>,
            GetOutputShapeAsRefInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::ApplyPowerSign::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterResourceApplyPowerSign()
{
    using K = KernelDefinition<
        ops::ResourceApplyPowerSign,
        DmlKernelWrapper<DmlApplyPowerSignKernel<0>, NoOutputShapeHelper>>::
        WithHostMemoryArguments<
            ops::ResourceApplyPowerSign::Argument::var,
            ops::ResourceApplyPowerSign::Argument::m>;

    RegisterWithTypes<
        K,
        ops::ResourceApplyPowerSign::Attribute::T,
        TF_FLOAT,
        TF_HALF>();
}

void RegisterKernels_Training()
{
    RegisterApplyAdam();
    RegisterResourceApplyAdam();
    RegisterResourceApplyAdamWithAmsgrad();
    RegisterApplyAdaMax();
    RegisterResourceApplyAdaMax();
    RegisterApplyGradientDescent();
    RegisterResourceApplyGradientDescent();
    RegisterApplyAdadelta();
    RegisterResourceApplyAdadelta();
    RegisterApplyAdagrad();
    RegisterApplyAdagradV2();
    RegisterResourceApplyAdagrad();
    RegisterResourceApplyAdagradV2();
    RegisterSparseApplyAdagrad();
    RegisterSparseApplyAdagradV2();
    RegisterResourceSparseApplyAdagrad();
    RegisterResourceSparseApplyAdagradV2();
    RegisterApplyMomentum();
    RegisterResourceApplyMomentum();
    RegisterResourceApplyKerasMomentum();
    RegisterApplyRMSProp();
    RegisterResourceApplyRMSProp();
    RegisterApplyCenteredRMSProp();
    RegisterResourceApplyCenteredRMSProp();
    RegisterApplyAddSign();
    RegisterResourceApplyAddSign();
    RegisterApplyPowerSign();
    RegisterResourceApplyPowerSign();
}

} // namespace tfdml