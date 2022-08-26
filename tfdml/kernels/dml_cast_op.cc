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

        TF_DataType input_dtype = ctx->GetInputDataType(0);
        TF_DataType output_dtype = ctx->GetOutputDataType(0);
        const DML_TENSOR_DATA_TYPE dml_out_dtype =
            GetDmlDataTypeFromTfDataType(output_dtype);

        // Tensor shape doesn't matter for Cast, so don't bother with DML's 4D
        // restrictions
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc =
            DmlTensorDesc::Create(input_dtype, tensor_shape, tensor_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc =
            DmlTensorDesc::Create(output_dtype, tensor_shape, tensor_shape);

        DmlKernelTensors tensors;
        tensors.supports_in_place_execution = true;
        tensors.outputs = {output};
        tensors.inputs = {input};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto result = dml::InputTensor(scope, 0, inputs[0]);

        // Bool is a special case since it doesn't behave the same as uint8. The
        // uint8 version simply drops the decimals, but bool converts anything
        // that is not 0.0 to True.
        if (output_dtype == TF_BOOL &&
            (input_dtype == TF_HALF || input_dtype == TF_FLOAT))
        {
            result = dml::Ceil(dml::Abs(result));
        }

        result = dml::Cast(result, dml_out_dtype);

        if (output_dtype == TF_BOOL)
        {
            result = dml::Clip(result, 0.0, 1.0);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <TF_DataType T>
void RegisterCastDstT()
{
    using Op = ops::Cast;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<DmlCastKernel, GetOutputShapeAsInputShapeHelper>>::
        template WithTypeConstraint<Op::Attribute::DstT, T>;

    RegisterWithTypes<
        K,
        Op::Attribute::SrcT,
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

void RegisterKernels_Cast()
{
    RegisterCastDstT<TF_FLOAT>();
    RegisterCastDstT<TF_HALF>();
    RegisterCastDstT<TF_BOOL>();
    RegisterCastDstT<TF_UINT8>();
    RegisterCastDstT<TF_UINT16>();
    RegisterCastDstT<TF_UINT32>();
    RegisterCastDstT<TF_UINT64>();
    RegisterCastDstT<TF_INT8>();
    RegisterCastDstT<TF_INT16>();
    RegisterCastDstT<TF_INT32>();
    RegisterCastDstT<TF_INT64>();
}

} // namespace tfdml