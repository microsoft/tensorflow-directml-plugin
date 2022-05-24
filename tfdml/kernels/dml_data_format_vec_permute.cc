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

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/ops.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

// Ensure that `src` and `dst` define a valid permutation.
// Ops defined in this file assume that user specifies a permutation via two
// string attributes. This check validates that these attributes properly define
// it to prevent security vulnerabilities.
static bool IsValidPermutation(const std::string& src, const std::string& dst)
{
    if (src.size() != dst.size())
    {
        return false;
    }

    std::array<bool, 256> characters{};

    // Every character in `src` must be present only once
    for (const auto c : src)
    {
        const uint8_t char_index = static_cast<uint8_t>(c);
        if (characters[char_index])
        {
            return false;
        }
        characters[char_index] = true;
    }

    // Every character in `dst` must show up in `src` exactly once
    for (const auto c : dst)
    {
        const uint8_t char_index = static_cast<uint8_t>(c);
        if (!characters[char_index])
        {
            return false;
        }
        characters[char_index] = false;
    }

    // At this point, characters[] has been switched to true and false exactly
    // once for all character in `src` (and `dst`) so we have a valid
    // permutation
    return true;
}

class DmlDataFormatVecPermuteKernel : public OpKernel
{
  public:
    explicit DmlDataFormatVecPermuteKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        std::string src_format;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
        OP_REQUIRES(
            ctx,
            src_format.size() == 4 || src_format.size() == 5,
            errors::InvalidArgument(
                "Source format must be of length 4 or 5, received "
                "src_format = ",
                src_format));
        std::string dst_format;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));
        OP_REQUIRES(
            ctx,
            dst_format.size() == 4 || dst_format.size() == 5,
            errors::InvalidArgument(
                "Destination format must be of length "
                "4 or 5, received dst_format = ",
                dst_format));
        OP_REQUIRES(
            ctx,
            IsValidPermutation(src_format, dst_format),
            errors::InvalidArgument(
                "Destination and source format must determine a permutation, "
                "got ",
                src_format,
                " and ",
                dst_format));

        src_format_ = src_format;
        dst_format_ = dst_format;
    }

    void Compute(OpKernelContext* ctx)
    {
        const Tensor& input = ctx->input(0);
        const TensorShape& input_shape = input.shape();

        OP_REQUIRES(
            ctx,
            input_shape.dims() == 1 || input_shape.dims() == 2,
            errors::InvalidArgument(
                "input must be a vector or 2D tensor, but got shape ",
                input_shape.DebugString()));

        const int full_dim_count = src_format_.size();
        const int spatial_dim_count = full_dim_count - 2;

        if (input_shape.dims() == 1)
        {
            OP_REQUIRES(
                ctx,
                input.NumElements() == spatial_dim_count ||
                    input.NumElements() == full_dim_count,
                errors::InvalidArgument(
                    "1D input must be of size ",
                    spatial_dim_count,
                    " or ",
                    full_dim_count,
                    ", but got shape ",
                    input.shape().DebugString()));
        }
        else if (input_shape.dims() == 2)
        {
            OP_REQUIRES(
                ctx,
                input.dim_size(0) == spatial_dim_count ||
                    input.dim_size(0) == full_dim_count,
                errors::InvalidArgument(
                    "First dimension of 2D input must be "
                    "of size ",
                    spatial_dim_count,
                    " or ",
                    full_dim_count,
                    ", but got shape ",
                    input.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                input_shape.dim_size(1) == 2,
                errors::InvalidArgument(
                    "Second dimension of 2D input must be of size 2, but got "
                    "shape ",
                    input_shape.DebugString()));
        }
        std::string src_format = src_format_;
        std::string dst_format = dst_format_;

        if (input.dim_size(0) == spatial_dim_count)
        {
            // If the input is a vector of size 2, treat the two elements as
            // spatial dimensions.
            auto keep_only_spatial_dimensions =
                [spatial_dim_count](std::string* format_str) -> void
            {
                auto new_end = std::remove_if(
                    format_str->begin(),
                    format_str->end(),
                    [spatial_dim_count](const char dim) {
                        return dim != 'H' && dim != 'W' &&
                               (spatial_dim_count == 2 || dim != 'D');
                    });
                format_str->erase(new_end, format_str->end());
            };
            keep_only_spatial_dimensions(&src_format);
            keep_only_spatial_dimensions(&dst_format);

            if (spatial_dim_count == 3)
            {
                OP_REQUIRES(
                    ctx,
                    src_format.size() == 3 && dst_format.size() == 3,
                    errors::InvalidArgument("Format specifier must contain D, "
                                            "H and W for 2D case"));
            }
            else
            {
                assert(spatial_dim_count == 2);
                OP_REQUIRES(
                    ctx,
                    src_format.size() == 2 && dst_format.size() == 2,
                    errors::InvalidArgument(
                        "Format specifier must contain H and W for 2D case"));
            }
        }

        absl::InlinedVector<uint32_t, 5> permutations;

        for (size_t dst_index = 0; dst_index < dst_format.length(); ++dst_index)
        {
            for (size_t src_index = 0; src_index < src_format.length();
                 ++src_index)
            {
                if (dst_format[dst_index] == src_format[src_index])
                {
                    permutations.push_back(src_index);
                    break;
                }
            }
        }

        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, input_shape);
        OP_REQUIRES_OK(ctx, status_or_output.status());

        DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());
        auto device_context = dml_device->GetDeviceContext();

        D3D12BufferRegion input_buffer =
            device_context->GetBufferForTensor(input);

        D3D12BufferRegion output_buffer =
            device_context->GetBufferForTensor(status_or_output.ValueOrDie());

        const int perm_stride =
            DataTypeSize(input.dtype()) * input_shape.dims();

        for (uint32_t i = 0; i < permutations.size(); ++i)
        {
            device_context->CopyBufferToBuffer(
                output_buffer.Subregion(i * perm_stride),
                input_buffer.Subregion(
                    permutations[i] * perm_stride,
                    perm_stride));
        }
    }

  private:
    std::string src_format_;
    std::string dst_format_;
};

// TODO: Remove once TensorFlow core implements host for DEVICE_DEFAULT
// https://github.com/tensorflow/tensorflow/pull/55558
class DmlDataFormatVecPermuteHostOp : public OpKernel
{
  public:
    explicit DmlDataFormatVecPermuteHostOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format_));

        TF_Graph* graph = TF_NewGraph();
        auto graph_cleanup = absl::MakeCleanup([graph] { TF_DeleteGraph(graph); });

        // Initialize the placeholder that sets the input for the
        // DataFormatVecPermute op on the CPU
        TF_OperationDescription* placeholder_desc =
            TF_NewOperation(graph, "Placeholder", "data_format");
        TF_SetDevice(placeholder_desc, "/device:CPU");

        TF_DataType dtype;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
        TF_SetAttrType(placeholder_desc, "dtype", dtype);

        Status status;
        placeholder_op_ = TF_FinishOperation(placeholder_desc, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // Initialize the DataFormatVecPermute op on the CPU
        TF_OperationDescription* data_format_vec_permute_desc = TF_NewOperation(
            graph,
            "DataFormatVecPermute",
            "DmlDataFormatVecPermuteHost");
        TF_SetDevice(data_format_vec_permute_desc, "/device:CPU");
        TF_AddInput(
            data_format_vec_permute_desc,
            TF_Output{placeholder_op_, 0});
        TF_SetAttrType(data_format_vec_permute_desc, "T", dtype);
        TF_SetAttrString(
            data_format_vec_permute_desc,
            "src_format",
            src_format_.c_str(),
            src_format_.length());
        TF_SetAttrString(
            data_format_vec_permute_desc,
            "dst_format",
            dst_format_.c_str(),
            dst_format_.length());

        data_format_vec_permute_op_ =
            TF_FinishOperation(data_format_vec_permute_desc, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // Create a new session that will be executed on the CPU
        TF_SessionOptions* opts = TF_NewSessionOptions();
        auto session_opts_cleanup = absl::MakeCleanup([opts]
        { TF_DeleteSessionOptions(opts); });

        sess_ = TF_NewSession(graph, opts, status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlDataFormatVecPermuteHostOp() override
    {
        if (sess_)
        {
            Status status;
            TF_DeleteSession(sess_, status.raw());
            TF_CHECK_OK(status);
        }
    }

    void Compute(OpKernelContext* ctx)
    {
        Tensor input_tensor = ctx->input(0);
        TF_Output feeds[] = {TF_Output{placeholder_op_, 0}};
        TF_Tensor* feedValues[] = {input_tensor.raw()};
        TF_Output fetches[] = {TF_Output{data_format_vec_permute_op_, 0}};
        TF_Tensor* fetchValues[] = {nullptr};

        Status status;
        TF_SessionRun(
            sess_,
            nullptr,
            feeds,
            feedValues,
            1,
            fetches,
            fetchValues,
            1,
            nullptr,
            0,
            nullptr,
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        Tensor output_tensor(fetchValues[0]);
        OP_REQUIRES_OK(ctx, ctx->set_output(0, output_tensor));
    }

  private:
    TF_Operation* placeholder_op_ = nullptr;
    TF_Operation* data_format_vec_permute_op_ = nullptr;
    TF_Session* sess_ = nullptr;
    std::string src_format_;
    std::string dst_format_;
};

namespace ops
{
struct DmlDataFormatVecPermuteHost
{
    static constexpr const char* name = "DmlDataFormatVecPermuteHost";

    enum class Argument
    {
        x,
        y
    };

    static constexpr uint32_t input_arg_count = 1;
    static constexpr uint32_t output_arg_count = 1;
    static constexpr std::
        array<ArgumentDesc, input_arg_count + output_arg_count>
            argument_descs{
                ArgumentDesc{"x", ArgumentDesc::TensorCount::Single},
                ArgumentDesc{"y", ArgumentDesc::TensorCount::Single}};

    enum class Attribute
    {
        T,
        src_format,
        dst_format
    };

    static constexpr std::array<AttributeDesc, 3> attribute_descs{
        AttributeDesc{"T", AttributeType::Type},
        AttributeDesc{"src_format", AttributeType::String},
        AttributeDesc{"dst_format", AttributeType::String}};
};
} // namespace ops

void DataFormatVecPermuteShapeInferenceFn(
    TF_ShapeInferenceContext* ctx,
    TF_Status* status)
{
    TF_ShapeHandle* handle = TF_NewShapeHandle();
    auto handle_cleanup = absl::MakeCleanup([handle] { TF_DeleteShapeHandle(handle); });

    TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
    CHECK(TF_GetCode(status) == TF_OK);

    TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
    CHECK(TF_GetCode(status) == TF_OK);
}

void RegisterKernels_DataFormatVecPermute()
{
    using K = KernelDefinition<
        ops::DataFormatVecPermute,
        DmlDataFormatVecPermuteKernel>;

    RegisterWithTypes<
        K,
        ops::DataFormatVecPermute::Attribute::T,
        TF_INT32,
        TF_INT64>();

    // TODO: Remove once TensorFlow core implements host for DEVICE_DEFAULT
    // https://github.com/tensorflow/tensorflow/pull/55558
    TF_OpDefinitionBuilder* builder =
        TF_NewOpDefinitionBuilder("DmlDataFormatVecPermuteHost");
    TF_OpDefinitionBuilderAddInput(builder, "x: T");
    TF_OpDefinitionBuilderAddOutput(builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(builder, "T: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(builder, "src_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(builder, "dst_format: string = 'NCHW'");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(
        builder,
        DataFormatVecPermuteShapeInferenceFn);

    Status status;
    TF_RegisterOpDefinition(builder, status.raw());
    CHECK(status.ok());

    using HostK = KernelDefinition<
        ops::DmlDataFormatVecPermuteHost,
        DmlDataFormatVecPermuteHostOp>::
        WithHostMemoryArguments<
            ops::DmlDataFormatVecPermuteHost::Argument::x,
            ops::DmlDataFormatVecPermuteHost::Argument::y>;

    RegisterWithTypes<
        HostK,
        ops::DmlDataFormatVecPermuteHost::Attribute::T,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml