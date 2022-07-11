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

class DmlOnesLikeKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlOnesLikeKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape);

        DmlKernelTensors tensors;
        tensors.outputs = {output};

        auto dml_dtype = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
        DML_SCALAR_UNION one_value = dml::ScalarUnion(1, dml_dtype);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto result = dml::FillValueConstant(
            scope,
            {static_cast<uint32_t>(tensor_shape.num_elements())},
            dml_dtype,
            one_value);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterKernels_OnesLike()
{
    using K = KernelDefinition<
        ops::OnesLike,
        DmlKernelWrapper<DmlOnesLikeKernel, GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::OnesLike::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64,
        TF_BOOL>();
}

} // namespace tfdml