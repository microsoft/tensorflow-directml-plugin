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
#include "tfdml/runtime_adapter/bcast.h"

namespace tfdml
{

class BroadcastToInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    BroadcastToInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& input_tensor = ctx->input(0);
        const Tensor& shape_tensor = ctx->input(1);
        const TensorShape& input_shape = input_tensor.shape();

        TensorShape output_shape;
        OP_REQUIRES_OK(
            ctx,
            TensorShapeUtils::MakeShape(shape_tensor, &output_shape));

        OP_REQUIRES(
            ctx,
            input_shape.dims() <= output_shape.dims(),
            errors::InvalidArgument(
                "Rank of input (",
                input_shape.dims(),
                ") must be no greater than rank of output shape (",
                output_shape.dims(),
                ")."));

        BCast bcast(
            BCast::FromShape(input_shape),
            BCast::FromShape(output_shape));

        OP_REQUIRES(
            ctx,
            bcast.IsValid(),
            errors::InvalidArgument(
                "Incompatible shapes: ",
                input_shape.DebugString(),
                " vs. ",
                output_shape.DebugString()));
        OP_REQUIRES(
            ctx,
            BCast::ToShape(bcast.output_shape()) == output_shape,
            errors::InvalidArgument(
                "Unable to broadcast tensor of shape ",
                input_shape.DebugString(),
                " to tensor of shape ",
                output_shape.DebugString()));

        collapsed_input_shape_ = BCast::ToShape(bcast.x_reshape());
        collapsed_output_shape_ = BCast::ToShape(bcast.y_reshape());

        OP_REQUIRES(
            ctx,
            collapsed_output_shape_.dims() <= kNcdhwDimensionCount,
            errors::InvalidArgument(
                "DML doesn't support more than 5D for BroadcastTo after "
                "collapsing dimensions together, but the output has ",
                collapsed_output_shape_.dims(),
                " dimensions."));
    }

    const TensorShape& GetCollapsedInputShape() const
    {
        return collapsed_input_shape_;
    }

    const TensorShape& GetCollapsedOutputShape() const
    {
        return collapsed_output_shape_;
    }

  private:
    TensorShape collapsed_input_shape_;
    TensorShape collapsed_output_shape_;
};

class DmlBroadcastToKernel : public DmlKernel
{
  public:
    using InitHelper = BroadcastToInitHelper;

    explicit DmlBroadcastToKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelTensors tensors;

        const TensorShape input_shape = init_helper->GetCollapsedInputShape();
        const TensorShape output_shape = init_helper->GetCollapsedOutputShape();

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            output_shape,
            input_shape);
        tensors.inputs = {input};

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
        identity_desc.InputTensor = &inputs[0];
        identity_desc.OutputTensor = &outputs[0];

        DML_OPERATOR_DESC op_desc = {
            DML_OPERATOR_ELEMENT_WISE_IDENTITY,
            &identity_desc};
        Initialize(ctx, std::move(tensors), op_desc);
    }
};

void RegisterKernels_BroadcastTo()
{
    using int32_kernel = KernelDefinition<
        ops::BroadcastTo,
        DmlKernelWrapper<
            DmlBroadcastToKernel,
            GetOutputShapeFromDimsTensorHelper<int32_t, 1>>>::
        WithTypeConstraint<ops::BroadcastTo::Attribute::Tidx, TF_INT32>::
            WithHostMemoryArguments<ops::BroadcastTo::Argument::shape>;

    using int64_kernel = KernelDefinition<
        ops::BroadcastTo,
        DmlKernelWrapper<
            DmlBroadcastToKernel,
            GetOutputShapeFromDimsTensorHelper<int64_t, 1>>>::
        WithTypeConstraint<ops::BroadcastTo::Attribute::Tidx, TF_INT64>::
            WithHostMemoryArguments<ops::BroadcastTo::Argument::shape>;

    RegisterWithTypes<
        int32_kernel,
        ops::BroadcastTo::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL>();

    RegisterWithTypes<
        int64_kernel,
        ops::BroadcastTo::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL>();
}

} // namespace tfdml