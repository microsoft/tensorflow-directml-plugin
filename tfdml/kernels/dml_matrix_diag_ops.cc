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
#include <numeric>

namespace tfdml
{

template <typename T>
class MatrixDiagInitHelper : public InitializationHelper
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

    MatrixDiagInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& diagonal = ctx->input(0);

        // MatrixDiag, MatrixDiagV2 and MatrixDiagV3 all use this init helper.
        // MatrixDiag only has one input, so we have to check the number of
        // inputs before reading additional parameters in
        // MatrixDiagV2/MatrixDiagV3.
        int32_t lower_diag_index = 0;
        int32_t upper_diag_index = 0;
        int32_t num_rows = -1;
        int32_t num_cols = -1;
        padding_value_ = T(0);

        // MatrixDiagOpV2-specific.
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
            num_rows = ctx->input(2).base<int32_t>()[0];
            num_cols = ctx->input(3).base<int32_t>()[0];
            padding_value_ = ctx->input(4).base<T>()[0];
        }

        // Size validations.
        const TensorShape& diagonal_shape = diagonal.shape();
        const int diag_rank = diagonal_shape.dims();
        const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsVectorOrHigher(diagonal_shape),
            errors::InvalidArgument(
                "diagonal must be at least 1-dim, received shape: ",
                diagonal.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            lower_diag_index <= upper_diag_index,
            errors::InvalidArgument(
                "lower_diag_index must not be larger than upper_diag_index: ",
                lower_diag_index,
                " > ",
                upper_diag_index));
        OP_REQUIRES(
            ctx,
            lower_diag_index == upper_diag_index ||
                diagonal_shape.dim_size(diag_rank - 2) == num_diags,
            errors::InvalidArgument(
                "The number of diagonals provided in the input does not "
                "match the lower_diag_index and upper_diag_index range."));

        const Eigen::Index max_diag_len =
            diagonal_shape.dim_size(diag_rank - 1);
        const int32_t min_num_rows =
            max_diag_len - std::min(upper_diag_index, 0);
        const int32_t min_num_cols =
            max_diag_len + std::max(lower_diag_index, 0);
        OP_REQUIRES(
            ctx,
            num_rows == -1 || num_rows >= min_num_rows,
            errors::InvalidArgument("The number of rows is too small."));
        OP_REQUIRES(
            ctx,
            num_cols == -1 || num_cols >= min_num_cols,
            errors::InvalidArgument("The number of columns is too small."));

        // If both num_rows and num_cols are unknown, assume that output is
        // square. Otherwise, use smallest possible values.
        if (num_rows == -1 && num_cols == -1)
        {
            num_rows = std::max(min_num_rows, min_num_cols);
            num_cols = num_rows;
        }
        else if (num_rows == -1)
        {
            num_rows = min_num_rows;
        }
        else if (num_cols == -1)
        {
            num_cols = min_num_cols;
        }
        OP_REQUIRES(
            ctx,
            num_rows == min_num_rows || num_cols == min_num_cols,
            errors::InvalidArgument(
                "The number of rows or columns is not consistent with "
                "the specified d_lower, d_upper, and diagonal."));

        output_shape_ = diagonal_shape;
        if (num_diags == 1)
        { // Output has rank `rank+1`.
            output_shape_.set_dim(diag_rank - 1, num_rows);
            output_shape_.AddDim(num_cols);
        }
        else
        { // Output has rank `rank`.
            output_shape_.set_dim(diag_rank - 2, num_rows);
            output_shape_.set_dim(diag_rank - 1, num_cols);
        }

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
    int32_t lower_diag_index_;
    int32_t upper_diag_index_;
    T padding_value_;
    bool align_sup_left_ = true;
    bool align_sub_left_ = true;
};

template <typename T>
class MatrixDiagShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const MatrixDiagInitHelper<T>*>(initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

template <typename T>
class DmlMatrixDiagKernel : public DmlKernel
{
  public:
    using InitHelper = MatrixDiagInitHelper<T>;
    DmlMatrixDiagKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& out_shape = ctx->GetOutputTensorShape(0);
        int32_t k_min = init_helper->GetLowerDiagIndex();
        int32_t k_max = init_helper->GetUpperDiagIndex();
        padding_value_ = init_helper->GetPaddingValue();

        // Fast path for MatrixDiag and MatrixDiagV2 when k=0, num_rows=num_cols
        const bool is_square_matrix =
            out_shape.dim_size(out_shape.dims() - 2) ==
            out_shape.dim_size(out_shape.dims() - 1);

        use_fast_path_ = is_square_matrix && k_min == 0 && k_max == 0;

        if (use_fast_path_)
        {
            DiagonalizeSimpleMatrix(ctx, init_helper);
        }
        else
        {
            DiagonalizeComplexMatrix(ctx, init_helper);
        }
    }

  private:
    void DiagonalizeSimpleMatrix(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const TensorShape& input_shape = ctx->GetInputTensorShape(0);
        const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

        uint32_t batch_size = 1u;
        for (int i = 0; i < input_shape.dims() - 1; ++i)
        {
            batch_size *= input_shape.dim_size(i);
        }

        auto elem_count_per_batch =
            static_cast<uint32_t>(input_shape.num_elements() / batch_size);
        auto output_height = static_cast<uint32_t>(
            output_shape.dim_size(output_shape.dims() - 2));
        auto output_width = static_cast<uint32_t>(
            output_shape.dim_size(output_shape.dims() - 1));

        // Flatten the input into a batch of vectors
        TensorShape flattened_input_shape(
            {batch_size, 1, 1, elem_count_per_batch});

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            flattened_input_shape,
            flattened_input_shape);

        // Flatten the output into a batch of vectors and use strides to skip
        // over zeros
        uint32_t output_sizes[] = {batch_size, 1, 1, elem_count_per_batch};
        uint32_t output_strides[] = {
            output_height * output_width,
            0,
            0,
            (elem_count_per_batch + 1),
        };

        auto dtype_tf = ctx->GetOutputDataType(0);
        auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc(dtype_dml, output_sizes, output_strides);

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

    void DiagonalizeComplexMatrix(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        TensorShape input_shape = ctx->GetInputTensorShape(0);
        const TensorShape& out_shape = ctx->GetOutputTensorShape(0);
        int32_t k_min = init_helper->GetLowerDiagIndex();
        int32_t k_max = init_helper->GetUpperDiagIndex();
        int32_t k_depth = k_max + 1 - k_min;

        if (input_shape.dims() == 1)
        {
            input_shape.InsertDim(0, 1);
        }

        int32_t old_diag_depth = input_shape.dim_size(input_shape.dims() - 2);

        if (input_shape.dims() < 4)
        {
            int missing_dims = 4 - input_shape.dims();

            if (k_depth != old_diag_depth)
            {
                missing_dims--;
            }

            for (int i = 0; i < missing_dims; ++i)
            {
                input_shape.InsertDim(0, 1);
            }
        }

        // If the depth of the diagonal is not the same as the depth of the
        // band, we we reshape the diags by placing each vector on a different
        // batch
        if (k_depth != old_diag_depth)
        {
            input_shape.InsertDim(input_shape.dims() - 1, 1);
        }

        // Flatten the extra dimensions if the input shape is bigger than 4D
        int batch_dims = input_shape.dims() - 3;
        int64_t batch_size = 1;

        for (int i = 0; i < batch_dims; ++i)
        {
            batch_size *= input_shape.dim_size(i);
        }

        TensorShape diag_shape({
            batch_size,
            input_shape.dim_size(input_shape.dims() - 3),
            input_shape.dim_size(input_shape.dims() - 2),
            input_shape.dim_size(input_shape.dims() - 1),
        });

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            diag_shape,
            diag_shape);

        TensorShape flattened_output_shape({out_shape.num_elements()});

        // Output shape doesn't matter here, it only needs to contain the right
        // number of elements
        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            flattened_output_shape,
            flattened_output_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        int64_t out_height = out_shape.dim_size(out_shape.dims() - 2);
        int64_t out_width = out_shape.dim_size(out_shape.dims() - 1);

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto diag = dml::InputTensor(scope, 0, inputs[0]);
        auto result = dml::MatrixDiag(
            scope,
            diag,
            k_min,
            k_max,
            static_cast<float>(padding_value_),
            out_height,
            out_width,
            init_helper->GetAlignSupLeft(),
            init_helper->GetAlignSubLeft());

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        if (use_fast_path_)
        {
            // Fill the buffer with the padding value since we use strides to
            // skip over elements
            Tensor output = ctx->GetOutputTensor(0);
            ctx->GetDmlDeviceContext()->FillBufferWithValue(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(output),
                static_cast<float>(padding_value_));
        }

        return DmlKernel::Compute(ctx);
    }

    T padding_value_ = T(0);
    bool use_fast_path_ = false;
};

template <typename Op, typename T>
using K = KernelDefinition<
    Op,
    DmlKernelWrapper<DmlMatrixDiagKernel<T>, MatrixDiagShapeHelper<T>>>;

template <typename T>
static void RegisterMatrixDiag()
{
    using Op = ops::MatrixDiag;
    K<Op, T>::template WithTypeConstraint<
        Op::Attribute::T,
        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiag()
{
    RegisterMatrixDiag<T>();
    RegisterMatrixDiag<Ts...>();
}

template <typename T>
static void RegisterMatrixDiagV2()
{
    using Op = ops::MatrixDiagV2;
    K<Op, T>::template WithHostMemoryArguments<Op::Argument::k>::
        template WithHostMemoryArguments<Op::Argument::num_rows>::
            template WithHostMemoryArguments<Op::Argument::num_cols>::
                template WithHostMemoryArguments<Op::Argument::padding_value>::
                    template WithTypeConstraint<
                        Op::Attribute::T,
                        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiagV2()
{
    RegisterMatrixDiagV2<T>();
    RegisterMatrixDiagV2<Ts...>();
}

template <typename T>
static void RegisterMatrixDiagV3()
{
    using Op = ops::MatrixDiagV3;
    K<Op, T>::template WithHostMemoryArguments<Op::Argument::k>::
        template WithHostMemoryArguments<Op::Argument::num_rows>::
            template WithHostMemoryArguments<Op::Argument::num_cols>::
                template WithHostMemoryArguments<Op::Argument::padding_value>::
                    template WithTypeConstraint<
                        Op::Attribute::T,
                        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterMatrixDiagV3()
{
    RegisterMatrixDiagV3<T>();
    RegisterMatrixDiagV3<Ts...>();
}

template <typename T>
static void RegisterBatchMatrixDiag()
{
    using Op = ops::BatchMatrixDiag;
    K<Op, T>::template WithTypeConstraint<
        Op::Attribute::T,
        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterBatchMatrixDiag()
{
    RegisterBatchMatrixDiag<T>();
    RegisterBatchMatrixDiag<Ts...>();
}

void RegisterKernels_MatrixDiag()
{
    RegisterMatrixDiag<float, Eigen::half, bool>();
    RegisterMatrixDiagV2<float, Eigen::half, bool>();
    RegisterMatrixDiagV3<float, Eigen::half, bool>();
    RegisterBatchMatrixDiag<float, Eigen::half>();
}

} // namespace tfdml