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

#include "tfdml/kernels/dml_matrix_diag_helpers.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

template <typename T>
class MatrixDiagPartInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            if (ctx->HasAttr("align"))
            {
                std::string align;
                OP_REQUIRES_OK(ctx, ctx->GetAttr("align", &align));
                align_sup_left = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
                align_sub_left = align == "LEFT_LEFT" || align == "RIGHT_LEFT";
            }
        }

        bool align_sup_left = true;
        bool align_sub_left = true;
    };

    MatrixDiagPartInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& input = ctx->input(0);

        // MatrixDiagPart, MatrixDiagPartV2, and MatrixDiagPartV3 all use this
        // OpKernel. MatrixDiagPart only has one input, so we have to check the
        // number of inputs before reading additional parameters in
        // MatrixDiagV2/MatrixDiagV3.
        int32_t lower_diag_index = 0;
        int32_t upper_diag_index = 0;
        T padding_value(0);

        // MatrixDiagPartV2/V3-specific.
        if (ctx->num_inputs() > 1)
        {
            const Tensor& diag_index = ctx->input(1);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(diag_index.shape()) ||
                    TensorShapeUtils::IsVector(diag_index.shape()),
                errors::InvalidArgument(
                    "diag_index must be a scalar or vector, received shape: ",
                    diag_index.shape().DebugString()));
            lower_diag_index = diag_index.base<int32_t>()[0];
            upper_diag_index = lower_diag_index;
            if (TensorShapeUtils::IsVector(diag_index.shape()))
            {
                auto diag_index_size = diag_index.dim_size(0);
                OP_REQUIRES(
                    ctx,
                    0 < diag_index_size && diag_index_size <= 2,
                    errors::InvalidArgument(
                        "diag_index must have only one or two elements, "
                        "received ",
                        diag_index_size,
                        " elements."));
                if (diag_index_size > 1)
                {
                    upper_diag_index = diag_index.base<int32_t>()[1];
                }
            }
            padding_value = ctx->input(2).base<T>()[0];
        }
        const TensorShape& input_shape = input.shape();

        // Preliminary validation of sizes.
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsMatrixOrHigher(input_shape),
            errors::InvalidArgument(
                "input must be at least 2-dim, received shape: ",
                input.shape().DebugString()));

        // Make sure lower_diag_index and upper_diag_index is valid.
        const int rank = input_shape.dims();
        const Eigen::Index num_rows = input_shape.dim_size(rank - 2);
        const Eigen::Index num_cols = input_shape.dim_size(rank - 1);
        OP_REQUIRES( // Checks lower_diag_index == 0 for when matrix shape = 0.
            ctx,
            (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
                lower_diag_index == 0,
            errors::InvalidArgument(
                "lower_diag_index is out of bound: ",
                lower_diag_index,
                ". It must be between ",
                -num_rows,
                " and ",
                num_cols));
        OP_REQUIRES(
            ctx,
            (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                upper_diag_index == 0,
            errors::InvalidArgument(
                "upper_diag_index is out of bound: ",
                upper_diag_index,
                " It must be between ",
                -num_rows,
                " and ",
                num_cols));
        OP_REQUIRES(
            ctx,
            lower_diag_index <= upper_diag_index,
            errors::InvalidArgument(
                "lower_diag_index must not be larger than upper_diag_index: ",
                lower_diag_index,
                " > ",
                upper_diag_index));

        for (int i = 0; i < rank - 2; ++i)
        {
            output_shape_.AddDim(input_shape.dim_size(i));
        }
        const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
        if (num_diags > 1) output_shape_.AddDim(num_diags);
        const int32_t max_diag_len = std::min(
            num_rows + std::min(upper_diag_index, 0),
            num_cols - std::max(lower_diag_index, 0));
        output_shape_.AddDim(max_diag_len);

        padding_value_ = padding_value;
        lower_diag_index_ = lower_diag_index;
        upper_diag_index_ = upper_diag_index;
        align_sup_left_ = attr->align_sup_left;
        align_sub_left_ = attr->align_sub_left;
    }

    TensorShape GetOutputShape() const { return output_shape_; }
    int32_t GetLowerDiagIndex() const { return lower_diag_index_; }
    int32_t GetUpperDiagIndex() const { return upper_diag_index_; }
    T GetPaddingValue() const { return padding_value_; }
    bool GetAlignSupLeft() const { return align_sup_left_; }
    bool GetAlignSubLeft() const { return align_sub_left_; }

  private:
    TensorShape output_shape_;
    T padding_value_;
    int32_t lower_diag_index_;
    int32_t upper_diag_index_;
    bool align_sup_left_ = true;
    bool align_sub_left_ = true;
};

template <typename T>
class MatrixDiagPartShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper = static_cast<const MatrixDiagPartInitHelper<T>*>(
            initialization_helper);

        return {init_helper->GetOutputShape()};
    }
};

template <typename T>
class DmlMatrixDiagPartKernel : public DmlKernel
{
  public:
    using InitHelper = MatrixDiagPartInitHelper<T>;

    DmlMatrixDiagPartKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& in_shape = ctx->GetInputTensorShape(0);
        int32_t k_min = init_helper->GetLowerDiagIndex();
        int32_t k_max = init_helper->GetUpperDiagIndex();

        // Fast path for MatrixDiag and MatrixDiagV2/V3 when k=0,
        // num_rows=num_cols
        const bool is_square_matrix = in_shape.dim_size(in_shape.dims() - 2) ==
                                      in_shape.dim_size(in_shape.dims() - 1);

        const bool use_fast_path = is_square_matrix && k_min == 0 && k_max == 0;

        if (use_fast_path)
        {
            ExtractDiagPartFromSimpleMatrix(ctx, init_helper);
        }
        else
        {
            ExtractDiagPartFromComplexMatrix(ctx, init_helper);
        }
    }

  private:
    void ExtractDiagPartFromSimpleMatrix(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

        uint32_t batch_size = 1u;
        for (int i = 0; i < input_shape.dims() - 2; ++i)
        {
            batch_size *= input_shape.dim_size(i);
        }

        auto elem_count_per_batch =
            static_cast<uint32_t>(output_shape.num_elements() / batch_size);
        auto input_height =
            static_cast<uint32_t>(input_shape.dim_size(input_shape.dims() - 2));
        auto input_width =
            static_cast<uint32_t>(input_shape.dim_size(input_shape.dims() - 1));

        // Flatten the output batches of vectors
        TensorShape flattened_output_shape(
            {batch_size, 1, 1, elem_count_per_batch});

        auto dtype_tf = ctx->GetInputDataType(0);
        auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);

        // Flatten the input into a vector and use strides to skip over zeros
        uint32_t input_sizes[] = {batch_size, 1, 1, elem_count_per_batch};
        uint32_t input_strides[] = {
            input_height * input_width,
            0,
            0,
            (elem_count_per_batch + 1),
        };

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc(dtype_dml, input_sizes, input_strides);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            flattened_output_shape,
            flattened_output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
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

    void ExtractDiagPartFromComplexMatrix(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
        uint32_t xlen = input_shape.dim_size(input_shape.dims() - 1);
        uint32_t ylen = input_shape.dim_size(input_shape.dims() - 2);
        uint32_t leading_dims_size = input_shape.num_elements() / xlen / ylen;
        dml::TensorDesc::Dimensions m_shape({1, leading_dims_size, ylen, xlen});

        int32_t k0 = init_helper->GetLowerDiagIndex();
        int32_t k1 = init_helper->GetUpperDiagIndex();

        uint32_t out_cols = output_shape.dim_size(output_shape.dims() - 1);
        uint32_t out_rows =
            k0 == k1 ? 1 : output_shape.dim_size(output_shape.dims() - 2);
        uint32_t out_leading_dim_size =
            output_shape.num_elements() / out_cols / out_rows;

        dml::TensorDesc::Dimensions flattened_out_shape(
            {1, out_leading_dim_size, out_rows, out_cols});

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc =
            DmlTensorDesc::Create(ctx->GetInputDataType(0), m_shape, m_shape);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            flattened_out_shape,
            flattened_out_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto m = dml::InputTensor(scope, 0, inputs[0]);
        auto diags = dml::MatrixDiagPart(
            scope,
            m,
            k0,
            k1,
            static_cast<float>(init_helper->GetPaddingValue()),
            out_rows,
            out_cols,
            init_helper->GetAlignSupLeft(),
            init_helper->GetAlignSubLeft());

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {diags});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename Op, typename T>
using K = KernelDefinition<
    Op,
    DmlKernelWrapper<DmlMatrixDiagPartKernel<T>, MatrixDiagPartShapeHelper<T>>>;

template <typename T>
static void RegisterMatrixDiagPart()
{
    using Op = ops::MatrixDiagPart;
    K<Op, T>::template WithTypeConstraint<
        Op::Attribute::T,
        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiagPart()
{
    RegisterMatrixDiagPart<T>();
    RegisterMatrixDiagPart<Ts...>();
}

template <typename T>
static void RegisterMatrixDiagPartV2()
{
    using Op = ops::MatrixDiagPartV2;
    K<Op, T>::template WithHostMemoryArguments<Op::Argument::k>::
        template WithHostMemoryArguments<Op::Argument::padding_value>::
            template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::
                Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiagPartV2()
{
    RegisterMatrixDiagPartV2<T>();
    RegisterMatrixDiagPartV2<Ts...>();
}

template <typename T>
static void RegisterMatrixDiagPartV3()
{
    using Op = ops::MatrixDiagPartV3;
    K<Op, T>::template WithHostMemoryArguments<Op::Argument::k>::
        template WithHostMemoryArguments<Op::Argument::padding_value>::
            template WithTypeConstraint<Op::Attribute::T, DataTypeToEnum<T>()>::
                Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiagPartV3()
{
    RegisterMatrixDiagPartV3<T>();
    RegisterMatrixDiagPartV3<Ts...>();
}

template <typename T>
static void RegisterBatchMatrixDiagPart()
{
    using Op = ops::BatchMatrixDiagPart;
    K<Op, T>::template WithTypeConstraint<
        Op::Attribute::T,
        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterBatchMatrixDiagPart()
{
    RegisterBatchMatrixDiagPart<T>();
    RegisterBatchMatrixDiagPart<Ts...>();
}

void RegisterKernels_MatrixDiagPart()
{
    RegisterMatrixDiagPart<float, Eigen::half, bool>();
    RegisterMatrixDiagPartV2<float, Eigen::half, bool>();
    RegisterMatrixDiagPartV3<float, Eigen::half, bool>();
    RegisterBatchMatrixDiagPart<float, Eigen::half>();
}

} // namespace tfdml