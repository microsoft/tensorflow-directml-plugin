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

class DmlCastKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlCastKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 1);
        CHECK(ctx->GetOutputCount() == 1);

        // Currently, 64-bit integers in DML are emulated using 32-bit integers
        // using striding to emulate a larger type. Because we can't guarantee
        // that our output tensor's memory is zero'd, we need to do so manually
        // prior to performing the cast.
        if (Is64BitIntegerType(ctx->GetOutputDataType(0)))
        {
            zero_outputs_ = true;
        }

        // Tensor shape doesn't matter for Cast, so don't bother with DML's 4D
        // restrictions
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

        auto tensor_layout =
            GetDmlTensorLayout(FORMAT_NCHW, tensor_shape.dims());

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            tensor_shape,
            tensor_shape,
            tensor_layout);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape,
            tensor_layout);

        DmlKernelTensors tensors;
        tensors.outputs = {output};
        tensors.inputs = {input};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_CAST_OPERATOR_DESC cast_desc = {};
        cast_desc.InputTensor = inputs.data();
        cast_desc.OutputTensor = outputs.data();

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CAST, &cast_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const
    {
        if (zero_outputs_)
        {
            Tensor& output = ctx->GetOutputTensor(0);
            ctx->GetDmlDeviceContext()->ZeroBuffer(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
        }

        return DmlKernel::Compute(ctx);
    }

  private:
    bool zero_outputs_ = false;
};

template <TF_DataType SrcT, TF_DataType T, TF_DataType... Ts>
void RegisterCastDstT()
{
    using Op = ops::Cast;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<DmlCastKernel, GetOutputShapeAsInputShapeHelper>>::
        template WithTypeConstraint<Op::Attribute::DstT, T>;
    K::template WithTypeConstraint<Op::Attribute::SrcT, SrcT>::Register();

    if constexpr (sizeof...(Ts) > 0) RegisterCastDstT<SrcT, Ts...>();
}

template <TF_DataType T, TF_DataType... Ts>
void RegisterCastSrcT()
{
    RegisterCastDstT<
        T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32,
        TF_UINT64,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_INT64>();
    if constexpr (sizeof...(Ts) > 0) RegisterCastDstT<Ts...>();
}

void RegisterKernels_Cast()
{
    RegisterCastSrcT<
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_UINT8,
        TF_UINT16,
        TF_UINT32,
        TF_UINT64,
        TF_INT8,
        TF_INT16,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml