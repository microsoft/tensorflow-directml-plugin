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

class DmlTopKInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("sorted", &sorted_));
            if (ctx->HasAttr("k"))
            {
                // k is an attr (TopK).
                OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
            }
            else
            {
                // k is an input (TopKV2), so we won't know it until Compute.
                k_ = -1;
            }
        }
        int k_;
        // sorted_ attr is not used because tensorflow has no specification for
        // sorted_=False
        bool sorted_;
    };

    DmlTopKInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        k = attr_->k_;
        if (ctx->num_inputs() >= 2)
        {
            const auto& k_in = ctx->input(1);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(k_in.shape()),
                errors::InvalidArgument(
                    "k must be scalar, got shape ",
                    k_in.shape().DebugString()));
            k = k_in.base<int32_t>()[0];
        }
        OP_REQUIRES(
            ctx,
            k >= 0,
            errors::InvalidArgument("Need k >= 0, got ", k));
        const auto& input_in = ctx->input(0);
        OP_REQUIRES(
            ctx,
            input_in.dims() >= 1,
            errors::InvalidArgument(
                "input must be >= 1-D, got shape ",
                input_in.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            input_in.dim_size(input_in.dims() - 1) >= k,
            errors::InvalidArgument(
                "input must have at least k columns. Had ",
                input_in.dim_size(input_in.dims() - 1),
                ", needed ",
                k));
    }
    int GetK() const { return k; }

  private:
    int k = -1;
    const std::shared_ptr<const Attributes> attr_;
};

class TopKShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const DmlTopKInitHelper*>(initialization_helper);
        int k = init_helper->GetK();
        const TensorShape& input_shape = ctx->input(0).shape();
        TensorShape output_shape;
        for (int i = 0; i < input_shape.dims() - 1; ++i)
        {
            output_shape.AddDim(input_shape.dim_size(i));
        }
        output_shape.AddDim(k);
        return {output_shape, output_shape};
    }
};

class DmlTopKKernel : public DmlKernel
{
  public:
    using InitHelper = DmlTopKInitHelper;
    explicit DmlTopKKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() == 1 || ctx->GetInputCount() == 2);
        assert(ctx->GetOutputCount() == 2);

        const TensorShape& tensor_shape = ctx->GetInputTensorShape(0);
        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            tensor_shape,
            tensor_shape);

        DmlTensorInfo values_output;
        values_output.kernel_index = 0;
        values_output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);

        DmlTensorInfo indices_output;
        indices_output.kernel_index = 1;
        indices_output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(1),
            output_shape,
            output_shape);
        indices_output.desc.ForceUnsignedDataType();

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {values_output, indices_output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);

        uint32_t k = init_helper->GetK();
        uint32_t axis = input.desc.GetDimensionCount() - 1;

        dml::TopKOutputs result =
            dml::TopK(input_tensor, axis, k, DML_AXIS_DIRECTION_DECREASING);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(
                DML_EXECUTION_FLAG_NONE,
                {result.value, result.index});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterTopK()
{
    using K = KernelDefinition<
        ops::TopK,
        DmlKernelWrapper<DmlTopKKernel, TopKShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::TopK::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64,
        TF_INT32,
        TF_UINT16,
        TF_INT16,
        TF_UINT8,
        TF_INT8>();
}

void RegisterTopKV2()
{
    using K = KernelDefinition<
        ops::TopKV2,
        DmlKernelWrapper<DmlTopKKernel, TopKShapeHelper>>::
        WithHostMemoryArguments<ops::TopKV2::Argument::k>;

    RegisterWithTypes<
        K,
        ops::TopKV2::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64,
        TF_INT32,
        TF_UINT16,
        TF_INT16,
        TF_UINT8,
        TF_INT8>();
}

void RegisterKernels_TopK()
{
    RegisterTopK();
    RegisterTopKV2();
}

} // namespace tfdml