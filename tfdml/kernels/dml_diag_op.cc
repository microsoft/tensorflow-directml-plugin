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

class DiagInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    DiagInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
    {
        const Tensor& diagonal = ctx->input(0);
        const int num_dims = diagonal.dims();
        OP_REQUIRES(
            ctx,
            0 != num_dims,
            errors::InvalidArgument("Input must be at least rank 1, got 0"));
    }
};

class DiagShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        const Tensor& diagonal = ctx->input(0);
        const int num_dims = diagonal.dims();

        TensorShape output_shape;
        for (int i = 0; i < num_dims; ++i)
        {
            output_shape.AddDim(diagonal.dim_size(i));
        }
        for (int i = 0; i < num_dims; ++i)
        {
            output_shape.AddDim(diagonal.dim_size(i));
        }

        return {std::move(output_shape)};
    }
};

class DmlDiagKernel : public DmlKernel
{
  public:
    using InitHelper = DiagInitHelper;

    explicit DmlDiagKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // Flatten the input into a vector
        TensorShape input_shape(
            {1, 1, 1, ctx->GetInputTensorShape(0).num_elements()});

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            input_shape,
            input_shape);

        const auto dtype_tf = ctx->GetOutputDataType(0);
        const auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);

        // Flatten the output into a vector and use strides to skip over zeros
        auto num_elements = static_cast<uint32_t>(input_shape.num_elements());
        uint32_t output_sizes[] = {1, 1, 1, num_elements};
        uint32_t output_strides[] = {
            0,
            0,
            0,
            (num_elements + 1),
        };

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc =
            DmlTensorDesc(dtype_dml, output_sizes, output_strides, 0, 0);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        const auto out_policy = dml::TensorPolicy(
            [](DML_TENSOR_DATA_TYPE dataType,
               DML_TENSOR_FLAGS flags,
               dml::Span<const uint32_t> sizes)
            {
                uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

                const uint32_t num_elements = std::accumulate(
                    sizes.begin(),
                    sizes.end(),
                    1u,
                    std::multiplies<uint32_t>());

                dml::TensorDimensions strides(dimension_count);
                strides.back() = (num_elements + 1);

                dml::TensorProperties props = {};
                props.guaranteedBaseOffsetAlignment = 0;
                props.strides = std::move(strides);
                props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
                    dataType,
                    dimension_count,
                    sizes.data(),
                    props.strides->data());
                return props;
            });

        auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
        auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
        auto result = dml::Identity(input_tensor);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        // Zero the buffer since we use strides to skip over elements
        Tensor output = ctx->GetOutputTensor(0);
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));

        return DmlKernel::Compute(ctx);
    }
};

void RegisterKernels_Diag()
{
    using K = KernelDefinition<
        ops::Diag,
        DmlKernelWrapper<DmlDiagKernel, DiagShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::Diag::Attribute::T,
        TF_FLOAT,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml