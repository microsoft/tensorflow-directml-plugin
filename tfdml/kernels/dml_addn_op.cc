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

#include "tfdml/core/dml_operator_helper.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/core/kernels/dml_kernel_wrapper.h"
#include "tfdml/core/kernels/dml_ops_common.h"
#include "tfdml/core/kernel_def_builder.h"
#include "tfdml/external/macros.h"

namespace tfdml
{

class DmlAddNKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlAddNKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // AddN doesn't support broadcasting, so we can simply collapse all
        // dimensions into a single one to support more than 5D
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

        DmlKernelTensors tensors;

        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                tensor_shape,
                tensor_shape);

            tensors.inputs.push_back(std::move(input));
        }

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape);

        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        if (ctx->GetInputCount() == 1)
        {
            auto outputs = GetDmlTensorDescs(tensors.outputs);

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
            identity_desc.InputTensor = inputs.data();
            identity_desc.OutputTensor = outputs.data();

            DML_OPERATOR_DESC op_desc = {
                DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                &identity_desc};
            Initialize(ctx, std::move(tensors), op_desc);
        }
        else
        {
            auto scope = dml::Graph(ctx->GetDmlDevice());
            auto result = dml::InputTensor(scope, 0, inputs[0]);

            for (uint32_t i = 1; i < inputs.size(); ++i)
            {
                result += dml::InputTensor(scope, i, inputs[i]);
            }

            // TFDML #24881131
            if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
            {
                result = dml::ConvertInt32ToInt64(result);
            }

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());
        }
    }
};

void RegisterKernels_AddN()
{
    using K = KernelDefinition<
        ops::AddN,
        DmlKernelWrapper<DmlAddNKernel, GetOutputShapeAsInputShapeHelper>>;

    constexpr auto T = ops::AddN::Attribute::T;
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
}

} // namespace tfdml