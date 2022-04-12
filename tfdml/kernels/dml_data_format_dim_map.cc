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
static bool IsValidPermutation(absl::string_view src, absl::string_view dst)
{
    if (src.size() != dst.size())
    {
        return false;
    }

    std::array<bool, 256> characters{};

    // Every character in `src` must be present only once
    for (const char c : src)
    {
        const uint8_t char_index = static_cast<uint8_t>(c);
        if (characters[char_index])
        {
            return false;
        }
        characters[char_index] = true;
    }

    // Every character in `dst` must show up in `src` exactly once
    for (const char c : dst)
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

class DataFormatDimMapInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));
        }

        std::string src_format;
        std::string dst_format;
    };

    DataFormatDimMapInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
    }

    absl::string_view GetSrcFormat() const { return attr_->src_format; }
    absl::string_view GetDstFormat() const { return attr_->dst_format; }

  private:
    const std::shared_ptr<const Attributes> attr_;
};

class DmlDataFormaDimMapKernel : public DmlKernel
{
  public:
    using InitHelper = DataFormatDimMapInitHelper;

    explicit DmlDataFormaDimMapKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        auto src_format = init_helper->GetSrcFormat();
        auto dst_format = init_helper->GetDstFormat();

        // TODO #37895137: Investigate why DataFormatDimMap's validation makes
        // other operators fail when done in the constructor instead of Compute
        OP_REQUIRES(
            ctx->GetOpKernelContext(),
            src_format.size() == 4 || src_format.size() == 5,
            errors::InvalidArgument(absl::StrCat(
                "Source format must be of length 4 or 5, received "
                "src_format = ",
                src_format)));
        OP_REQUIRES(
            ctx->GetOpKernelContext(),
            dst_format.size() == 4 || dst_format.size() == 5,
            errors::InvalidArgument(absl::StrCat(
                "Destination format must be of length "
                "4 or 5, received dst_format = ",
                dst_format)));
        OP_REQUIRES(
            ctx->GetOpKernelContext(),
            IsValidPermutation(src_format, dst_format),
            errors::InvalidArgument(
                "Destination and source format must "
                "determine a permutation, got ",
                src_format,
                " and ",
                dst_format));

        // Put all the indices into a single uint32 scalar that we use to fill
        // the buffer. Since the indices are forced to be within the [0, 3]
        // range and it has been validated earlier, we can represent them as 4
        // uint8 values. We can then reinterpret them as a tensor of 4 uint8
        // values before doing the gather operation.
        uint64_t src_dst_mapping_packed = 0;
        uint64_t left_shift = 0;

        for (uint64_t i = 0; i < src_format.size(); ++i)
        {
            for (uint64_t j = 0; j < dst_format.size(); ++j)
            {
                if (dst_format[j] == src_format[i])
                {
                    src_dst_mapping_packed |= j << left_shift;
                    left_shift += 8;
                    break;
                }
            }
        }

        TensorShape collapsed_shape(
            {ctx->GetInputTensorShape(0).num_elements()});

        DmlTensorInfo in_out;
        in_out.kernel_index = 0;
        in_out.desc = DmlTensorDesc::Create(
            ctx->GetInputDataType(0),
            collapsed_shape,
            collapsed_shape);

        DmlKernelTensors tensors;
        tensors.inputs = {in_out};
        tensors.outputs = {in_out};

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto indices = dml::InputTensor(scope, 0, inputs[0]);

        const uint32_t src_dst_mapping_right =
            static_cast<uint32_t>(src_dst_mapping_packed);

        auto params = dml::ScalarTensor<uint32_t>(
            scope,
            src_dst_mapping_right,
            {1, 1, 1, 1});

        params = dml::Reinterpret(
            params,
            DML_TENSOR_DATA_TYPE_UINT8,
            {1, 1, 1, 4},
            {});

        constexpr uint32_t gather_axis = 3;

        if (src_format.size() == 5)
        {
            const uint8_t src_dst_mapping_left =
                static_cast<uint8_t>(src_dst_mapping_packed >> 32);

            auto additional_params = dml::ScalarTensor<uint8_t>(
                scope,
                src_dst_mapping_left,
                {1, 1, 1, 1});

            params = dml::Join({params, additional_params}, gather_axis);
        }

        // We need strides of 4 for int32 and strides of 8 for int64 since the
        // params are uint8
        // TFDML #24881131
        const uint32_t element_stride =
            Is64BitIntegerType(ctx->GetOutputDataType(0)) ? 8 : 4;

        const auto out_policy = dml::TensorPolicy(
            [element_stride](
                DML_TENSOR_DATA_TYPE dataType,
                DML_TENSOR_FLAGS flags,
                dml::Span<const uint32_t> sizes)
            {
                uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
                dml::TensorDimensions strides(dimension_count);
                strides.back() = element_stride;

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

        scope.SetTensorPolicy(out_policy);

        constexpr uint32_t index_dimensions = 1;
        auto result =
            dml::Gather(params, indices, gather_axis, index_dimensions);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        // Since we gather uint8 values with strides of 4 or 8, we always need
        // to zero the buffer
        Tensor& output = ctx->GetOutputTensor(0);
        ctx->GetDmlDeviceContext()->ZeroBuffer(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
        return DmlKernel::Compute(ctx);
    }
};

// TODO: Remove once TensorFlow core implements host for DEVICE_DEFAULT
// https://github.com/tensorflow/tensorflow/pull/55558
class DmlDataFormatDimMapHostOp : public OpKernel
{
  public:
    explicit DmlDataFormatDimMapHostOp(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format_));

        TF_Graph* graph = TF_NewGraph();
        absl::Cleanup graph_cleanup = [graph] { TF_DeleteGraph(graph); };

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

        // Initialize the DataFormatDimMap op on the CPU
        TF_OperationDescription* data_format_dim_map_desc = TF_NewOperation(
            graph,
            "DataFormatDimMap",
            "DmlDataFormatDimMapHost");
        TF_SetDevice(data_format_dim_map_desc, "/device:CPU");
        TF_AddInput(data_format_dim_map_desc, TF_Output{placeholder_op_, 0});
        TF_SetAttrType(data_format_dim_map_desc, "T", dtype);
        TF_SetAttrString(
            data_format_dim_map_desc,
            "src_format",
            src_format_.c_str(),
            src_format_.length());
        TF_SetAttrString(
            data_format_dim_map_desc,
            "dst_format",
            dst_format_.c_str(),
            dst_format_.length());

        data_format_dim_map_op_ =
            TF_FinishOperation(data_format_dim_map_desc, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // Create a new session that will be executed on the CPU
        TF_SessionOptions* opts = TF_NewSessionOptions();
        absl::Cleanup session_opts_cleanup = [opts]
        { TF_DeleteSessionOptions(opts); };

        sess_ = TF_NewSession(graph, opts, status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlDataFormatDimMapHostOp() override
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
        TF_Output fetches[] = {TF_Output{data_format_dim_map_op_, 0}};
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
    TF_Operation* data_format_dim_map_op_ = nullptr;
    TF_Session* sess_ = nullptr;
    std::string src_format_;
    std::string dst_format_;
};

namespace ops
{
struct DmlDataFormatDimMapHost
{
    static constexpr const char* name = "DmlDataFormatDimMapHost";

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

void DataFormatDimMapShapeInferenceFn(
    TF_ShapeInferenceContext* ctx,
    TF_Status* status)
{
    TF_ShapeHandle* handle = TF_NewShapeHandle();
    absl::Cleanup handle_cleanup = [handle] { TF_DeleteShapeHandle(handle); };

    TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
    CHECK(TF_GetCode(status) == TF_OK);

    TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
    CHECK(TF_GetCode(status) == TF_OK);
}

void RegisterKernels_DataFormatDimMap()
{
    using K = KernelDefinition<
        ops::DataFormatDimMap,
        DmlKernelWrapper<
            DmlDataFormaDimMapKernel,
            GetOutputShapeAsInputShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::DataFormatDimMap::Attribute::T,
        TF_INT32,
        TF_INT64>();

    // TODO: Remove once TensorFlow core implements host for DEVICE_DEFAULT
    // https://github.com/tensorflow/tensorflow/pull/55558
    TF_OpDefinitionBuilder* builder =
        TF_NewOpDefinitionBuilder("DmlDataFormatDimMapHost");
    TF_OpDefinitionBuilderAddInput(builder, "x: T");
    TF_OpDefinitionBuilderAddOutput(builder, "y: T");
    TF_OpDefinitionBuilderAddAttr(builder, "T: {int32, int64} = DT_INT32");
    TF_OpDefinitionBuilderAddAttr(builder, "src_format: string = 'NHWC'");
    TF_OpDefinitionBuilderAddAttr(builder, "dst_format: string = 'NCHW'");

    TF_OpDefinitionBuilderSetShapeInferenceFunction(
        builder,
        DataFormatDimMapShapeInferenceFn);

    Status status;
    TF_RegisterOpDefinition(builder, status.raw());
    CHECK(status.ok());

    using HostK = KernelDefinition<
        ops::DmlDataFormatDimMapHost,
        DmlDataFormatDimMapHostOp>::
        WithHostMemoryArguments<
            ops::DmlDataFormatDimMapHost::Argument::x,
            ops::DmlDataFormatDimMapHost::Argument::y>;

    RegisterWithTypes<
        HostK,
        ops::DmlDataFormatDimMapHost::Attribute::T,
        TF_INT32,
        TF_INT64>();
}

} // namespace tfdml