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

#include "tensorflow/c/eager/c_api.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

template <typename TIndex>
class FillInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        // The output shape of this kernel is determined by a dims tensor. Since
        // an empty dims tensor is valid, we only look at the output tensor
        // shapes when determining when to no-op.
        return output_shapes[0].num_elements() == 0;
    }

    FillInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& dims_tensor = ctx->input(0);
        const Tensor& value_tensor = ctx->input(1);
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsVector(dims_tensor.shape()),
            errors::InvalidArgument(
                "dims must be a vector, got shape ",
                dims_tensor.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(value_tensor.shape()),
            errors::InvalidArgument(
                "value must be a scalar, got shape ",
                value_tensor.shape().DebugString()));

        // We only call TensorShapeUtils::MakeShape for the validation that it
        // provides
        TensorShape output_shape;
        OP_REQUIRES_OK(
            ctx,
            TensorShapeUtils::MakeShape(dims_tensor, &output_shape));
    }
};

template <typename TIndex>
class DmlFillKernel : public DmlKernel
{
  public:
    using InitHelper = FillInitializationHelper<TIndex>;

    explicit DmlFillKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelParams params;
        params.supports_in_place_execution = true;

        // Broadcast inputs to match output shape
        params.input_shape = ctx->GetOutputTensorShape(0);

        // The value tensor is at index 1 in TF's kernel
        params.kernel_input_indices = {1};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
        auto result = dml::Identity(input_tensor);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});
        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlFillCpuKernel : public OpKernel
{
  public:
    explicit DmlFillCpuKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        TFE_ContextOptions* context_options = TFE_NewContextOptions();
        auto context_options_cleanup = absl::MakeCleanup(
            [context_options] { TFE_DeleteContextOptions(context_options); });

        Status status;
        eager_context_ = TFE_NewContext(context_options, status.raw());
        OP_REQUIRES_OK(ctx, status);

        fill_op_ = TFE_NewOp(eager_context_, "Fill", status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_OpSetDevice(fill_op_, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlFillCpuKernel() override
    {
        TFE_DeleteOp(fill_op_);
        TFE_DeleteContext(eager_context_);
    }

  private:
    void ComputeImpl(OpKernelContext* ctx) final
    {
        absl::InlinedVector<TFE_TensorHandle*, 2> input_handles;
        auto input_handles_cleanup = absl::MakeCleanup(
            [&input_handles]
            {
                for (TFE_TensorHandle* handle : input_handles)
                {
                    TFE_DeleteTensorHandle(handle);
                }
            });

        Status status;
        for (int i = 0; i < ctx->num_inputs(); ++i)
        {
            const Tensor& input_tensor = ctx->input(i);
            TFE_TensorHandle* input_handle =
                TFE_NewTensorHandle(input_tensor.raw(), status.raw());
            OP_REQUIRES_OK(ctx, status);
            input_handles.push_back(input_handle);

            TFE_OpAddInput(fill_op_, input_handle, status.raw());
            OP_REQUIRES_OK(ctx, status);
        }

        TFE_TensorHandle* output_handle = nullptr;
        TFE_TensorHandle** output_handle_ptr = &output_handle;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup =
            absl::MakeCleanup([output_handle_ptr]
                              { TFE_DeleteTensorHandle(*output_handle_ptr); });

        int num_retvals = 1;
        TFE_Execute(fill_op_, &output_handle, &num_retvals, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TF_Tensor* output =
            TFE_TensorHandleResolve(output_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        OP_REQUIRES_OK(ctx, ctx->set_output(0, Tensor(output)));
    }

    TFE_Context* eager_context_ = nullptr;
    TFE_Op* fill_op_ = nullptr;
};

template <typename TIndex>
using DmlFillWrapper = DmlKernelWrapper<
    DmlFillKernel<TIndex>,
    GetOutputShapeFromDimsTensorHelper<0>>;

void RegisterFill()
{
    using int32_kernel = KernelDefinition<ops::Fill, DmlFillWrapper<int32_t>>::
        WithTypeConstraint<ops::Fill::Attribute::index_type, TF_INT32>::
            WithHostMemoryArguments<ops::Fill::Argument::dims>;

    using int64_kernel = KernelDefinition<ops::Fill, DmlFillWrapper<int64_t>>::
        WithTypeConstraint<ops::Fill::Attribute::index_type, TF_INT64>::
            WithHostMemoryArguments<ops::Fill::Argument::dims>;

    RegisterWithTypes<
        int32_kernel,
        ops::Fill::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_UINT8,
        TF_INT8,
        TF_UINT16,
        TF_INT16,
        TF_INT64,
        TF_BOOL>();

    RegisterWithTypes<
        int64_kernel,
        ops::Fill::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_UINT8,
        TF_INT8,
        TF_UINT16,
        TF_INT16,
        TF_INT64,
        TF_BOOL>();
}

void RegisterFillCpu()
{
    KernelDefinition<ops::Fill, DmlFillCpuKernel>::WithHostMemoryArguments<
        ops::Fill::Argument::dims,
        ops::Fill::Argument::value,
        ops::Fill::Argument::output>::
        WithTypeConstraint<ops::Fill::Attribute::T, TF_INT32>::
            WithTypeConstraint<ops::Fill::Attribute::index_type, TF_INT32>::
                Register();
}

void RegisterKernels_Fill()
{
    RegisterFill();
    RegisterFillCpu();
}

} // namespace tfdml
