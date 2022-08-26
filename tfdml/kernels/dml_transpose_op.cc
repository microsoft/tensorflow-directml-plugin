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

struct SimpleTranspose
{
    TensorShape input_shape;
    TensorShape output_shape;
    absl::InlinedVector<int, 8> permutations;
};

template <typename TPerm>
static SimpleTranspose SimplifyTranspose(
    const TensorShape& input_shape,
    const Tensor& perm_tensor)
{
    absl::InlinedVector<int64_t, 8> simple_input_sizes;
    absl::InlinedVector<int, 8> input_to_perm_map;

    if (perm_tensor.NumElements() == 0)
    {
        // A null permutation tensor means that we have an identity operation
        simple_input_sizes = {1};
        input_to_perm_map = {0};
    }
    else
    {
        simple_input_sizes.resize(input_shape.dims());
        input_to_perm_map.resize(input_shape.dims());

        auto perm_values =
            reinterpret_cast<const TPerm*>(perm_tensor.raw_data());
        int input_index = static_cast<int>(perm_values[0]);
        int dim_size = input_shape.dim_size(input_index);
        input_to_perm_map[input_index] = 0;

        // Merge all adjacent dimensions that will still be adjacent after the
        // transpose operation (i.e. increasing sequences with a step of 1)
        for (int64_t i = 1; i < perm_tensor.NumElements(); ++i)
        {
            int prev_index = static_cast<int>(perm_values[i - 1]);
            int index = static_cast<int>(perm_values[i]);
            input_to_perm_map[index] = i;

            if (index == prev_index + 1)
            {
                dim_size *= input_shape.dim_size(index);

                // Set the dimension to -1 to signal that it should be removed
                // later
                simple_input_sizes[index] = -1;
            }
            else
            {
                simple_input_sizes[input_index] = dim_size;
                dim_size = input_shape.dim_size(index);
                input_index = index;
            }
        }

        simple_input_sizes[input_index] = dim_size;
    }

    // Shift collapsed input dimensions to the left and adjust the permutation
    // indices accordingly
    int input_left_shift = 0;
    absl::InlinedVector<int, 8> simple_permutations(simple_input_sizes.size());
    for (int i = 0; i < simple_input_sizes.size(); ++i)
    {
        int perm_index = input_to_perm_map[i];

        if (simple_input_sizes[i] == -1)
        {
            input_left_shift++;

            // Set the dimension to -1 to signal that it should be removed later
            simple_permutations[perm_index] = -1;
        }
        else
        {
            int new_index = i - input_left_shift;
            simple_input_sizes[new_index] = simple_input_sizes[i];
            simple_permutations[perm_index] = new_index;
        }
    }

    simple_input_sizes.resize(simple_input_sizes.size() - input_left_shift);

    // Shift the permutations to the left
    int perm_left_shift = 0;
    for (int i = 0; i < simple_permutations.size(); ++i)
    {
        if (simple_permutations[i] == -1)
        {
            perm_left_shift++;
        }
        else
        {
            simple_permutations[i - perm_left_shift] = simple_permutations[i];
        }
    }

    simple_permutations.resize(simple_permutations.size() - perm_left_shift);

    // Finally, create the output shape
    TensorShape simple_output_shape;
    for (int i = 0; i < simple_permutations.size(); ++i)
    {
        int dim_size = simple_input_sizes[simple_permutations[i]];
        simple_output_shape.AddDim(dim_size);
    }

    SimpleTranspose simple_transpose;
    simple_transpose.input_shape = TensorShape(simple_input_sizes);
    simple_transpose.output_shape = std::move(simple_output_shape);
    simple_transpose.permutations = std::move(simple_permutations);

    return simple_transpose;
}

template <typename TPerm>
static std::vector<TensorShape> GetOutputShapesHelper(OpKernelContext* ctx)
{
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& perm_tensor = ctx->input(1);
    const TensorShape& input_shape = input_tensor.shape();
    TensorShape output_shape(input_shape);

    for (int64_t i = 0; i < perm_tensor.NumElements(); ++i)
    {
        auto perm_values =
            reinterpret_cast<const TPerm*>(perm_tensor.raw_data());
        TPerm input_dim_index = perm_values[i];
        CHECK(input_dim_index < input_shape.dims());
        output_shape.set_dim(i, input_shape.dim_size(input_dim_index));
    }

    return {std::move(output_shape)};
}

static dml::TensorStrides ComputeStrides(
    const TensorShape& input_shape,
    absl::Span<const int> permutations,
    uint32_t expected_dim_count)
{
    assert(input_shape.dims() == permutations.size());

    int leading_dims = permutations.size() < expected_dim_count
                           ? expected_dim_count - permutations.size()
                           : 0;

    dml::TensorStrides output_strides(leading_dims + permutations.size(), 1);

    uint32_t stride = 1;

    for (int64_t i = permutations.size() - 1; i >= 0; --i)
    {
        int input_dim_index = permutations[i];
        assert(input_dim_index < output_strides.size());

        output_strides[leading_dims + input_dim_index] = stride;
        stride *= input_shape.dim_size(input_dim_index);
    }

    return output_strides;
}

class TransposeInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    TransposeInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& input = ctx->input(0);
        const Tensor& perm_tensor = ctx->input(1);
        const TensorShape& input_shape = input.shape();

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsVector(perm_tensor.shape()),
            errors::InvalidArgument(
                "perm must be a vector, not ",
                perm_tensor.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            input_shape.dims() == perm_tensor.NumElements(),
            errors::InvalidArgument(
                "transpose expects a vector of size ",
                input_shape.dims(),
                ". But input(1) is a vector of size ",
                perm_tensor.NumElements()));

        assert(
            perm_tensor.dtype() == TF_INT32 || perm_tensor.dtype() == TF_INT64);
        simple_transpose_ =
            perm_tensor.dtype() == TF_INT32
                ? SimplifyTranspose<int32_t>(input_shape, perm_tensor)
                : SimplifyTranspose<int64_t>(input_shape, perm_tensor);

        OP_REQUIRES(
            ctx,
            simple_transpose_.input_shape.dims() <= 8,
            errors::InvalidArgument(
                "DML doesn't support more than 8D for Transpose, but ",
                simple_transpose_.input_shape.dims(),
                " dimensions were provided."));
    }

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const final
    {
        // An empty perm tensor is valid (it represents an identity operation),
        // so we're only a no-op kernel if the input is empty
        return ctx->input(0).NumElements() == 0;
    }

    SimpleTranspose GetSimpleTranspose() const { return simple_transpose_; }

  private:
    SimpleTranspose simple_transpose_;
};

class TransposeShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        const Tensor& perm_tensor = ctx->input(1);

        assert(
            perm_tensor.dtype() == TF_INT32 || perm_tensor.dtype() == TF_INT64);
        return perm_tensor.dtype() == TF_INT32
                   ? GetOutputShapesHelper<int32_t>(ctx)
                   : GetOutputShapesHelper<int64_t>(ctx);
    }
};

class DmlTransposeKernel : public DmlKernel
{
  public:
    using InitHelper = TransposeInitHelper;

    explicit DmlTransposeKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 2);
        CHECK(ctx->GetOutputCount() == 1);

        auto simple_transpose = init_helper->GetSimpleTranspose();

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            simple_transpose.input_shape,
            simple_transpose.input_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            simple_transpose.output_shape,
            simple_transpose.output_shape);

        const auto out_policy = dml::TensorPolicy(
            [ctx, simple_transpose](
                DML_TENSOR_DATA_TYPE dataType,
                DML_TENSOR_FLAGS flags,
                dml::Span<const uint32_t> sizes)
            {
                uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

                dml::TensorProperties props = {};
                props.strides = ComputeStrides(
                    simple_transpose.input_shape,
                    simple_transpose.permutations,
                    dimension_count);

                props.guaranteedBaseOffsetAlignment = 0;

                props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
                    dataType,
                    dimension_count,
                    sizes.data(),
                    props.strides->data());

                return props;
            });

        DmlKernelTensors tensors;
        tensors.supports_in_place_execution = true;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
        auto result = dml::Identity(dml::InputTensor(scope, 0, inputs[0]));

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

void RegisterKernels_Transpose()
{
    using K = KernelDefinition<
        ops::Transpose,
        DmlKernelWrapper<DmlTransposeKernel, TransposeShapeHelper>>::
        WithHostMemoryArguments<ops::Transpose::Argument::perm>;

    RegisterWithTypes<
        K,
        ops::Transpose::Attribute::T,
        TF_BOOL,
        TF_HALF,
        TF_FLOAT,
        TF_UINT8,
        TF_INT8,
        TF_UINT16,
        TF_INT16,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml