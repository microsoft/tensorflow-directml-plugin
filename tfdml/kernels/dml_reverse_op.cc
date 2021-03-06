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

class ReverseInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const final
    {
        return ctx->input(0).NumElements() == 0;
    }

    ReverseInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const TensorShape& input_shape = ctx->input(0).shape();
        const Tensor& dims_tensor = ctx->input(1);

        OP_REQUIRES(
            ctx,
            input_shape.dims() <= kNcdhwDimensionCount,
            errors::Unimplemented(
                "DML doesn't support tensors of rank > 5 for Reverse."));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsVector(dims_tensor.shape()),
            errors::InvalidArgument(
                "'dims' must be 1-dimension, not ",
                dims_tensor.dims()));

        TF_DataType dims_type = dims_tensor.dtype();
        uint32_t axis_bits = 0;

        // Check whether we are executing Reverse (bool vector) or ReverseV2
        // (int32 or int64 vector)
        if (dims_type == TF_BOOL)
        {
            OP_REQUIRES(
                ctx,
                input_shape.dims() == dims_tensor.dim_size(0),
                errors::InvalidArgument(
                    "'dims' must have the same number of values as 'input' has "
                    "dimensions. 'input' has ",
                    input_shape.dims(),
                    "'dims' has ",
                    dims_tensor.dim_size(0),
                    " values"));

            for (int32_t i = 0; i < dims_tensor.dims(); ++i)
            {
                axis_bits |= 1 << i;
            }
        }
        else
        {
            const auto dims_vector = IntTensorToVec<int64_t>(dims_tensor);

            for (int i = 0; i < dims_vector.size(); ++i)
            {
                int64_t axis = dims_vector[i];
                int64_t canonical_axis =
                    axis < 0 ? input_shape.dims() + axis : axis;
                OP_REQUIRES(
                    ctx,
                    canonical_axis >= 0 && canonical_axis < input_shape.dims(),
                    errors::InvalidArgument(
                        "'axis'[",
                        i,
                        "] = ",
                        axis,
                        " is out of valid range [",
                        0,
                        ", ",
                        input_shape.dims() - 1));

                uint32_t mask = 1 << canonical_axis;

                OP_REQUIRES(
                    ctx,
                    (axis_bits & mask) == 0,
                    errors::InvalidArgument(
                        "axis ",
                        canonical_axis,
                        " specified more than once."));
                axis_bits |= mask;
            }
        }

        axis_bits_ = axis_bits;
    }

    uint32_t GetAxisBits() const { return axis_bits_; }

  private:
    uint32_t axis_bits_;
};

class DmlReverseKernel : public DmlKernel
{
  public:
    using InitHelper = ReverseInitializationHelper;

    explicit DmlReverseKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        DmlKernelParams params;
        params.kernel_input_indices = {0};

        DmlKernelTensors tensors = GetTensorInfos(ctx, params);
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto outputs = GetDmlTensorDescs(tensors.outputs);

        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        const Tensor& dims_tensor = ctx->GetConstantInputTensor(1);

        bool is_identity = TensorShapeUtils::IsScalar(input_shape) ||
                           dims_tensor.NumElements() == 0;

        if (is_identity)
        {
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
            const int input_dims = input_shape.dims();

            TF_DataType dims_type = ctx->GetInputDataType(1);
            const uint32_t axis_bits = init_helper->GetAxisBits();
            const uint32_t axis_padding = input_dims < kNchwDimensionCount
                                              ? kNchwDimensionCount - input_dims
                                              : 0;

            absl::InlinedVector<int32_t, 5> dml_strides(axis_padding, 1);
            absl::InlinedVector<uint32_t, 5> dml_sizes(axis_padding, 1);

            for (int i = 0; i < input_dims; ++i)
            {
                if (axis_bits & (1 << i))
                {
                    dml_strides.push_back(-1);
                }
                else
                {
                    dml_strides.push_back(1);
                }

                dml_sizes.push_back(input_shape.dim_size(i));
            }

            absl::InlinedVector<uint32_t, 5> dml_offsets(dml_sizes.size(), 0);

            DML_SLICE1_OPERATOR_DESC slice_desc = {};
            slice_desc.InputTensor = inputs.data();
            slice_desc.OutputTensor = outputs.data();
            slice_desc.InputWindowSizes = dml_sizes.data();
            slice_desc.InputWindowStrides = dml_strides.data();
            slice_desc.InputWindowOffsets = dml_offsets.data();
            slice_desc.DimensionCount = dml_sizes.size();

            DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE1, &slice_desc};
            Initialize(ctx, std::move(tensors), op_desc);
        }
    }
};

static void RegisterReverse()
{
    using K = KernelDefinition<
        ops::Reverse,
        DmlKernelWrapper<DmlReverseKernel, GetOutputShapeAsInputShapeHelper>>::
        template WithHostMemoryArguments<ops::Reverse::Argument::dims>;

    RegisterWithTypes<
        K,
        ops::Reverse::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_UINT8,
        TF_INT8>();
}

static void RegisterReverseV2()
{
    using K = KernelDefinition<
        ops::ReverseV2,
        DmlKernelWrapper<DmlReverseKernel, GetOutputShapeAsInputShapeHelper>>::
        template WithHostMemoryArguments<ops::ReverseV2::Argument::axis>;

    RegisterWithTypes<
        K,
        ops::ReverseV2::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_UINT8,
        TF_INT8>();
}

void RegisterKernels_Reverse()
{
    RegisterReverse();
    RegisterReverseV2();
}

} // namespace tfdml