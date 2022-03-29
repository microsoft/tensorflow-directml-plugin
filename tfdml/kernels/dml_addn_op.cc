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
#include "tfdml/kernels/pch.h"

namespace tfdml
{

class DmlAddNKernel : public DmlKernel
{
  public:
    using InitHelper = NoOpInitializationHelper;

    explicit DmlAddNKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        // AddN doesn't support broadcasting, so we can simply collapse all
        // dimensions into a single one to support more than 5D
        TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

        DmlKernelTensors tensors;

        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                tensor_shape,
                tensor_shape);

            tensors.inputs.push_back(std::move(input));
        }

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            tensor_shape,
            tensor_shape);

        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);

        if (ctx->GetInputCount() == 1)
        {
            auto outputs = GetDmlTensorDescs(tensors.outputs);

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
            auto scope = dml::Graph(ctx->GetDmlDevice());
            auto result = dml::InputTensor(scope, 0, inputs[0]);

            for (uint32_t i = 1; i < inputs.size(); ++i)
            {
                result += dml::InputTensor(scope, i, inputs[i]);
            }

            // TFDML #24881131
            if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
            {
                result = dml::ConvertInt32ToInt64(result);
            }

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
                scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

            Initialize(ctx, std::move(tensors), compiled_op.Get());
        }
    }
};

class DmlAddNVariantKernel : public OpKernel
{
  public:
    explicit DmlAddNVariantKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        TF_Graph* graph = TF_NewGraph();
        absl::Cleanup graph_cleanup = [graph] { TF_DeleteGraph(graph); };

        // Initialize the placeholders that takes the Variant inputs

        int num_inputs;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_inputs));

        TF_DataType dtype;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));

        Status status;

        for (int i = 0; i < num_inputs; ++i)
        {
            TF_OperationDescription* placeholder_desc = TF_NewOperation(
                graph,
                "Placeholder",
                std::to_string(i).c_str());
            TF_SetDevice(placeholder_desc, "/device:CPU");
            TF_SetAttrType(placeholder_desc, "dtype", dtype);

            placeholder_ops_.push_back(
                TF_FinishOperation(placeholder_desc, status.raw()));
            OP_REQUIRES_OK(ctx, status);
        }

        // Initialize the AddN op on the CPU
        TF_OperationDescription* add_n_desc =
            TF_NewOperation(graph, "AddN", "DmlVariantAddN");
        TF_SetDevice(add_n_desc, "/device:CPU");

        std::vector<TF_Output> placeholder_outputs;
        placeholder_outputs.reserve(num_inputs);

        for (int i = 0; i < num_inputs; ++i)
        {
            placeholder_outputs.push_back(TF_Output{placeholder_ops_[i], 0});
        }

        TF_AddInputList(
            add_n_desc,
            placeholder_outputs.data(),
            placeholder_outputs.size());

        TF_SetAttrType(add_n_desc, "T", dtype);

        add_n_op_ = TF_FinishOperation(add_n_desc, status.raw());
        OP_REQUIRES_OK(ctx, status);

        // Create a new session that will be executed on the CPU
        TF_SessionOptions* opts = TF_NewSessionOptions();
        absl::Cleanup session_opts_cleanup = [opts]
        { TF_DeleteSessionOptions(opts); };

        sess_ = TF_NewSession(graph, opts, status.raw());
        OP_REQUIRES_OK(ctx, status);
    }

    ~DmlAddNVariantKernel() override
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
        std::vector<TF_Output> feeds;
        feeds.reserve(placeholder_ops_.size());

        std::vector<TF_Tensor*> feedValues;
        feedValues.reserve(placeholder_ops_.size());

        std::vector<Tensor> inputs;
        inputs.reserve(placeholder_ops_.size());

        for (int i = 0; i < placeholder_ops_.size(); ++i)
        {
            Tensor input = ctx->input(i);
            feeds.push_back(TF_Output{placeholder_ops_[i], 0});
            feedValues.push_back(input.raw());
            inputs.push_back(std::move(input));
        }

        TF_Output fetches[] = {TF_Output{add_n_op_, 0}};
        TF_Tensor* fetchValues[] = {nullptr};

        Status status;
        TF_SessionRun(
            sess_,
            nullptr,
            feeds.data(),
            feedValues.data(),
            placeholder_ops_.size(),
            fetches,
            fetchValues,
            1,
            nullptr,
            0,
            nullptr,
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        Tensor host_output(fetchValues[0]);
        OP_REQUIRES_OK(ctx, ctx->set_output(0, host_output));
    }

  private:
    std::vector<TF_Operation*> placeholder_ops_;
    TF_Operation* add_n_op_ = nullptr;
    TF_Session* sess_ = nullptr;
};

void RegisterKernels_AddN()
{
    using K = KernelDefinition<
        ops::AddN,
        DmlKernelWrapper<DmlAddNKernel, GetOutputShapeAsInputShapeHelper>>;

    constexpr auto T = ops::AddN::Attribute::T;
    K::WithTypeConstraint<T, TF_FLOAT>::Register();
    K::WithTypeConstraint<T, TF_HALF>::Register();
    K::WithTypeConstraint<T, TF_INT64>::Register();
    K::WithTypeConstraint<T, TF_UINT32>::Register();

    using VariantK = KernelDefinition<ops::AddN, DmlAddNVariantKernel>::
        WithHostMemoryArguments<ops::AddN::Argument::sum>;
    VariantK::WithTypeConstraint<T, TF_VARIANT>::Register();
}

} // namespace tfdml