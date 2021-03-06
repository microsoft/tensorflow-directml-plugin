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

class CheckNumericsInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            // message is used as the prefix for the assertion error message.
            // For instance, this can be the name of the input op that produced
            // the tensor.
            OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message));
        }

        std::string message;
    };

    CheckNumericsInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
    }

    const std::string& GetMessage() const { return attr_->message; }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class DmlCheckNumericsKernel : public DmlKernel
{
  public:
    using InitHelper = CheckNumericsInitHelper;

    explicit DmlCheckNumericsKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        message_ = init_helper->GetMessage();

        const TensorShape& input_shape = ctx->GetInputTensorShape(0);

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_shape,
            input_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc =
            DmlTensorDesc::Create(TF_INT32, TensorShape({}), TensorShape({}));

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);

        // Reduce doesn't support less than 32bit integer datatypes, so we need
        // to cast to uint32 beforehand
        auto has_nan = dml::Reduce(
            dml::Cast(dml::IsNaN(input_tensor), DML_TENSOR_DATA_TYPE_UINT32),
            DML_REDUCE_FUNCTION_MAX);
        auto has_inf = dml::Reduce(
            dml::Cast(
                dml::IsInfinity(input_tensor),
                DML_TENSOR_DATA_TYPE_UINT32),
            DML_REDUCE_FUNCTION_MAX);

        // We pack the NaN and Inf bits into 1 byte, where NaN is the bit at 2^1
        // and Inf is the bit at 2^0
        auto result =
            dml::Cast(has_nan * 2 + has_inf, DML_TENSOR_DATA_TYPE_UINT8);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        DmlKernel::Compute(ctx);

        OpKernelContext* op_ctx = ctx->GetOpKernelContext();

        // Copy the result to the CPU
        Tensor is_error_tensor;

        TF_RETURN_IF_ERROR(op_ctx->allocate_temp(
            op_ctx->input(0).dtype(),
            {},
            &is_error_tensor,
            true));

        auto dml_device = static_cast<DmlDevice*>(op_ctx->device());
        Tensor output_tensor = ctx->GetOutputTensor(0);

        TF_RETURN_IF_ERROR(
            dml_device->GetDeviceContext()->CopyDeviceTensorToCPU(
                dml_device,
                &output_tensor,
                &is_error_tensor));

        uint8_t nan_inf_bits = is_error_tensor.base<uint8_t>()[0];

        // The NaN bit is 2^1 and the Inf bit is 2^0
        if (nan_inf_bits)
        {
            bool is_nan = nan_inf_bits & 2;
            bool is_inf = nan_inf_bits & 1;
            std::string status;

            if (is_nan && is_inf)
            {
                status = "Inf and NaN";
            }
            else if (is_nan)
            {
                status = "NaN";
            }
            else if (is_inf)
            {
                status = "Inf";
            }

            return errors::InvalidArgument(
                message_,
                " : Tensor had ",
                status,
                " values");
        }
        else
        {
            // If everything is fine, we simply copy the input to the output
            D3D12BufferRegion input_buffer =
                ctx->GetDmlDeviceContext()->GetBufferForTensor(
                    ctx->GetInputTensor(0));

            D3D12BufferRegion output_buffer =
                ctx->GetDmlDeviceContext()->GetBufferForTensor(output_tensor);

            ctx->GetDmlDeviceContext()->CopyBufferToBuffer(
                output_buffer,
                input_buffer.Subregion(0, output_tensor.TotalBytes()));
        }

        return ctx->GetDmlDeviceContext()->GetCurrentCompletionEvent();
    }

  private:
    std::string message_;
};

static void RegisterCheckNumerics()
{
    using K = KernelDefinition<
        ops::CheckNumerics,
        DmlKernelWrapper<
            DmlCheckNumericsKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<K, ops::CheckNumerics::Attribute::T, TF_FLOAT, TF_HALF>();
}

void RegisterKernels_CheckNumerics() { RegisterCheckNumerics(); }

} // namespace tfdml