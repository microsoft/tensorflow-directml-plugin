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
#include "tfdml/runtime_adapter/mirror_pad_mode.h"

namespace tfdml
{

struct SimplePad
{
    absl::InlinedVector<uint32_t, 4> in_shape;
    absl::InlinedVector<uint32_t, 4> out_shape;
    absl::InlinedVector<uint32_t, 4> start_padding;
    absl::InlinedVector<uint32_t, 4> end_padding;
};

// Coalesces padded dimensions with all contiguous non-padded dimensions that
// follow. For example:
//
// original input shape = [2,2,2,3]
// original paddings = [[1,1],[0,0],[1,2],[0,0]]
// original output shape = [4,2,5,3]
//
// coalesced input shape = [4,6]
// coalesced paddings = [[2,2],[3,6]]
// coalesced output shape = [8,15]
//
// This is more aggressive than the implementation in the CPU kernel, which only
// collapses adjacent non-padded dimensions (none in the above example). This
// function also inflates the shapes and paddings to meet DML requirements,
// converts paddings to two separate arrays, and casts signed integers to
// unsigned.
template <typename Tpadding>
absl::optional<SimplePad> SimplifyPad(
    const TensorShape& input_shape,
    const Tensor& paddings_tensor,
    bool constant_pad,
    size_t min_output_size = 4,
    size_t max_output_size = 8)
{
    auto paddings = paddings_tensor.matrix<Tpadding>();
    assert(input_shape.dims() == paddings.dimension(0));

    SimplePad simple_pad = {};

    int i = 0;
    while (i < paddings.dimension(0))
    {
        auto size = static_cast<uint32_t>(input_shape.dim_size(i));
        auto start_pad = static_cast<uint32_t>(paddings(i, 0));
        auto end_pad = static_cast<uint32_t>(paddings(i, 1));

        // Coalesce subsequent non-padded dims into the current dim.
        int j = i + 1;

        // Constant padding can collapse adjacent non-padded and padded
        // dimensions because there's a single value to be padded, but the
        // padding values for reflect and symmetric depend on the shape.
        if (constant_pad || (paddings(i, 0) == 0 && paddings(i, 1) == 0))
        {
            while (j < paddings.dimension(0) && paddings(j, 0) == 0 &&
                   paddings(j, 1) == 0)
            {
                auto other_dim_size =
                    static_cast<uint32_t>(input_shape.dim_size(j));
                size *= other_dim_size;
                start_pad *= other_dim_size;
                end_pad *= other_dim_size;
                j++;
            }
        }

        i = j;

        simple_pad.in_shape.push_back(size);
        simple_pad.out_shape.push_back(size + start_pad + end_pad);
        simple_pad.start_padding.push_back(start_pad);
        simple_pad.end_padding.push_back(end_pad);
    }

    if (simple_pad.in_shape.size() > max_output_size)
    {
        return absl::nullopt;
    }

    // Inflate DML shapes/pads to the minimum required size.
    for (size_t i = simple_pad.in_shape.size(); i < min_output_size; i++)
    {
        simple_pad.in_shape.insert(simple_pad.in_shape.begin(), 1);
        simple_pad.out_shape.insert(simple_pad.out_shape.begin(), 1);
        simple_pad.start_padding.insert(simple_pad.start_padding.begin(), 0);
        simple_pad.end_padding.insert(simple_pad.end_padding.begin(), 0);
    }

    return simple_pad;
}

template <typename T, typename Tpadding>
class PadInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            MirrorPadMode mode;
            if (ctx->GetAttr("mode", &mode).ok())
            {
                switch (mode)
                {
                case SYMMETRIC:
                    padding_mode = DML_PADDING_MODE_SYMMETRIC;
                    break;
                case REFLECT: padding_mode = DML_PADDING_MODE_REFLECTION; break;
                default:
                    OP_REQUIRES(
                        ctx,
                        false,
                        errors::InvalidArgument(
                            "mode must be either REFLECT or SYMMETRIC."));
                }
            }
            else
            {
                padding_mode = DML_PADDING_MODE_CONSTANT;
            }
        }

        DML_PADDING_MODE padding_mode;
    };

    PadInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : padding_mode_(attr->padding_mode)
    {
        const Tensor& input = ctx->input(0);
        const Tensor& paddings = ctx->input(1);

        const int dims = input.dims();
        static const int kMinDims = 0;
        static const int kMaxDims = 6;
        OP_REQUIRES(
            ctx,
            kMinDims <= dims && dims <= kMaxDims,
            errors::Unimplemented(
                "inputs rank not in [",
                kMinDims,
                ",",
                kMaxDims,
                "]: ",
                dims));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsMatrix(paddings.shape()) &&
                paddings.dim_size(1) == 2,
            errors::InvalidArgument(
                "paddings must be a matrix with 2 columns: ",
                paddings.shape().DebugString()));

        const int fixed_dims =
            (dims == 0 && paddings.dim_size(0) == 1) ? 1 : dims;

        OP_REQUIRES(
            ctx,
            dims == paddings.dim_size(0),
            errors::InvalidArgument(
                "The first dimension of paddings must be the rank of inputs",
                paddings.shape().DebugString(),
                " ",
                input.shape().DebugString()));

        pad_value_ = T();
        if (ctx->num_inputs() == 3)
        {
            const Tensor& constant_values = ctx->input(2);
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::IsScalar(constant_values.shape()),
                errors::InvalidArgument(
                    "constant_values must be a scalar. Found: ",
                    constant_values.shape().DebugString()));
            pad_value_ = ctx->input(2).base<T>()[0];
        }

        typename TTypes<Tpadding>::ConstMatrix pads =
            paddings.matrix<Tpadding>();

        for (int d = 0; d < fixed_dims; ++d)
        {
            const Tpadding before_d = pads(d, 0);
            const Tpadding after_d = pads(d, 1);
            OP_REQUIRES(
                ctx,
                before_d >= 0 && after_d >= 0,
                errors::InvalidArgument(
                    "Paddings must be non-negative: ",
                    before_d,
                    " ",
                    after_d));

            if (padding_mode_ == DML_PADDING_MODE_SYMMETRIC)
            {
                OP_REQUIRES(
                    ctx,
                    before_d <= input.dim_size(d) &&
                        after_d <= input.dim_size(d),
                    errors::InvalidArgument(
                        "paddings must be no greater "
                        "than the dimension size: ",
                        before_d,
                        ", ",
                        after_d,
                        " greater than ",
                        input.dim_size(d)));
            }
            else if (padding_mode_ == DML_PADDING_MODE_REFLECTION)
            {
                OP_REQUIRES(
                    ctx,
                    before_d < input.dim_size(d) && after_d < input.dim_size(d),
                    errors::InvalidArgument(
                        "paddings must be less than"
                        " the dimension size: ",
                        before_d,
                        ", ",
                        after_d,
                        " not less than ",
                        input.dim_size(d)));
            }

            const int64_t size_d = (d == input.dims()) ? 1 : input.dim_size(d);
            output_shape_.AddDim(before_d + size_d + after_d);
        }

        bool constant_pad = padding_mode_ == DML_PADDING_MODE_CONSTANT;
        simple_pad_ =
            SimplifyPad<Tpadding>(input.shape(), paddings, constant_pad);
        OP_REQUIRES(
            ctx,
            simple_pad_.has_value(),
            errors::InvalidArgument(
                "DML can only handle up to 8D padding, but the given shape "
                "and paddings cannot be simplified to 8D."));
    }

    const TensorShape& GetOutputShape() const { return output_shape_; }

    T GetPadValue() const { return pad_value_; }

    DML_PADDING_MODE GetPaddingMode() const { return padding_mode_; }

    const absl::optional<SimplePad>& GetSimplePad() const
    {
        return simple_pad_;
    }

  private:
    TensorShape output_shape_;
    T pad_value_;
    absl::optional<SimplePad> simple_pad_;
    DML_PADDING_MODE padding_mode_;
};

template <typename T, typename Tpadding>
class PadShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper = static_cast<const PadInitHelper<T, Tpadding>*>(
            initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

template <typename T, typename Tpadding>
class DmlPadKernel : public DmlKernel
{
  public:
    using InitHelper = PadInitHelper<T, Tpadding>;

    explicit DmlPadKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        auto dtype = ctx->GetInputDataType(0);
        assert(dtype == ctx->GetOutputDataType(0));
        auto pad = init_helper->GetSimplePad();
        assert(pad.has_value()); // Validated in init_helper

        DmlTensorInfo in;
        in.kernel_index = 0;
        in.desc = DmlTensorDesc::Create(dtype, pad->in_shape, pad->in_shape);
        auto in_desc = in.desc.GetDmlDesc();

        DmlTensorInfo out;
        out.kernel_index = 0;
        out.desc = DmlTensorDesc::Create(dtype, pad->out_shape, pad->out_shape);
        auto out_desc = out.desc.GetDmlDesc();

        DmlKernelTensors tensors;
        tensors.inputs = {in};
        tensors.outputs = {out};

        DML_PADDING_OPERATOR_DESC desc = {};
        desc.InputTensor = &in_desc;
        desc.OutputTensor = &out_desc;
        desc.PaddingMode = init_helper->GetPaddingMode();
        desc.PaddingValue = static_cast<float>(init_helper->GetPadValue());
        desc.DimensionCount = pad->in_shape.size();
        desc.StartPadding = pad->start_padding.data();
        desc.EndPadding = pad->end_padding.data();

        DML_OPERATOR_DESC op_desc = {DML_OPERATOR_PADDING, &desc};

        Initialize(ctx, std::move(tensors), op_desc);
    }
};

template <typename Op, TF_DataType type, typename TPadding>
using K = typename KernelDefinition<
    Op,
    DmlKernelWrapper<
        DmlPadKernel<typename EnumToDataType<type>::T, TPadding>,
        PadShapeHelper<typename EnumToDataType<type>::T, TPadding>>>::
    template WithTypeConstraint<
        Op::Attribute::Tpaddings,
        DataTypeToEnum<TPadding>()>::
        template WithTypeConstraint<Op::Attribute::T, type>;

template <TF_DataType T, TF_DataType... Ts>
void RegisterPad()
{
    using Op = ops::Pad;
    K<Op, T, int32_t>::template WithHostMemoryArguments<
        Op::Argument::paddings>::Register();
    K<Op, T, int64_t>::template WithHostMemoryArguments<
        Op::Argument::paddings>::Register();
    if constexpr (sizeof...(Ts) > 0) RegisterPad<Ts...>();
}

template <TF_DataType T, TF_DataType... Ts>
void RegisterPadV2()
{
    using Op = ops::PadV2;
    K<Op, T, int32_t>::template WithHostMemoryArguments<
        Op::Argument::paddings,
        Op::Argument::constant_values>::Register();
    K<Op, T, int64_t>::template WithHostMemoryArguments<
        Op::Argument::paddings,
        Op::Argument::constant_values>::Register();
    if constexpr (sizeof...(Ts) > 0) RegisterPadV2<Ts...>();
}

template <TF_DataType T, TF_DataType... Ts>
void RegisterMirrorPad()
{
    using Op = ops::MirrorPad;
    K<Op, T, int32_t>::template WithHostMemoryArguments<
        Op::Argument::paddings>::Register();
    K<Op, T, int64_t>::template WithHostMemoryArguments<
        Op::Argument::paddings>::Register();
    if constexpr (sizeof...(Ts) > 0) RegisterMirrorPad<Ts...>();
}

void RegisterKernels_Pad()
{
    RegisterPad<TF_HALF, TF_FLOAT, TF_UINT8, TF_INT8>();
    RegisterPadV2<TF_HALF, TF_FLOAT, TF_UINT8, TF_INT8>();
    RegisterMirrorPad<TF_HALF, TF_FLOAT>();
}

} // namespace tfdml