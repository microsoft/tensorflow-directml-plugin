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

class DiagPartInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    DiagPartInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& tensor = ctx->input(0);
        const int num_dims = tensor.dims();
        const int out_dims = num_dims / 2;
        OP_REQUIRES(
            ctx,
            0 == num_dims % 2,
            errors::InvalidArgument(
                "The rank of the tensor should be \
                                         even and positive, got shape ",
                tensor.shape().DebugString()));
        for (int i = 0; i < out_dims; i++)
        {
            OP_REQUIRES(
                ctx,
                tensor.dim_size(i) == tensor.dim_size(i + out_dims),
                errors::InvalidArgument(
                    "Invalid shape ",
                    tensor.shape().DebugString(),
                    ": dimensions ",
                    i,
                    " and ",
                    i + out_dims,
                    " do not match."));
        }
    }
};

class DiagPartShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        const Tensor& tensor = ctx->input(0);
        const int out_dims = tensor.dims() / 2;

        TensorShape output_shape;
        for (int i = 0; i < out_dims; ++i)
        {
            output_shape.AddDim(tensor.dim_size(i));
        }

        return {std::move(output_shape)};
    }
};

class DmlDiagPartKernel : public DmlKernel
{
  public:
    using InitHelper = DiagPartInitHelper;

    explicit DmlDiagPartKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // Flatten the input into a vector
        TensorShape output_shape(
            {1, 1, 1, ctx->GetOutputTensorShape(0).num_elements()});

        auto dtype_tf = ctx->GetInputDataType(0);
        auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);

        // Flatten the input into a vector and use strides to skip over zeros
        auto out_num_elements =
            static_cast<uint32_t>(output_shape.num_elements());
        uint32_t input_sizes[] = {1, 1, 1, out_num_elements};
        uint32_t input_strides[] = {
            0,
            0,
            0,
            (out_num_elements + 1),
        };

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc(dtype_dml, input_sizes, input_strides);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
        auto result = dml::Identity(input_tensor);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterKernels_DiagPart()
{
    using K = KernelDefinition<
        ops::DiagPart,
        DmlKernelWrapper<DmlDiagPartKernel, DiagPartShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::DiagPart::Attribute::T,
        TF_FLOAT,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml