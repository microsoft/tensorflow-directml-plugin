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
#include "tensorflow/c/eager/c_api.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

class PackInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
        }

        int axis;
    };

    PackInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(attr)
    {
        std::vector<Tensor> values;
        for (int i = 0; i < ctx->num_inputs(); ++i)
        {
            auto input = ctx->input(i);
            values.push_back(std::move(input));
        }

        CHECK(values.size() > 0);
        input_shape_ = values[0].shape();

        int output_dims = input_shape_.dims() + 1;
        positive_axis_ =
            attr_->axis < 0 ? attr_->axis + output_dims : attr_->axis;

        OP_REQUIRES(
            ctx,
            0 <= positive_axis_ && positive_axis_ < output_dims,
            errors::InvalidArgument(
                "axis = ",
                attr_->axis,
                " not in [",
                -output_dims,
                ", ",
                output_dims,
                ")"));

        input_count_ = values.size();
        const TensorShape& first_input_shape = values[0].shape();

        // Verify that all input shapes match
        for (uint32_t i = 1; i < values.size(); i++)
        {
            const TensorShape& input_shape = values[i].shape();

            OP_REQUIRES(
                ctx,
                first_input_shape.IsSameSize(input_shape),
                errors::InvalidArgument(
                    "Shapes of all inputs must match: values[0].shape = ",
                    first_input_shape.DebugString(),
                    " != values[",
                    i,
                    "].shape = ",
                    input_shape.DebugString()));
        }
    }

    int GetAxis() const { return positive_axis_; }
    int GetInputCount() const { return input_count_; }
    const TensorShape& GetInputShape() const { return input_shape_; }

  private:
    const std::shared_ptr<const Attributes> attr_;
    int positive_axis_;
    int input_count_;
    TensorShape input_shape_;
};

using InitHelper = PackInitHelper;

class PackShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        int axis = init_helper->GetAxis();
        int input_count = init_helper->GetInputCount();
        const TensorShape& input_shape = init_helper->GetInputShape();

        TensorShape output_shape(input_shape);
        output_shape.InsertDim(axis, input_count);

        return {std::move(output_shape)};
    }
};

class DmlPackKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper;

    explicit DmlPackKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        assert(ctx->GetInputCount() > 0);
        assert(ctx->GetOutputCount() == 1);

        TensorShape output_shape = ctx->GetOutputTensorShape(0);
        int axis = init_helper->GetAxis();

        // We can collapse all dimensions to the left together and all
        // dimensions to the right together. This allows us to send tensors with
        // an "unlimited" number of dimensions to DirectML
        int left_dim_size = 1;
        int right_dim_size = 1;

        for (int i = 0; i < axis; ++i)
        {
            left_dim_size *= output_shape.dim_size(i);
        }

        for (int i = axis + 1; i < output_shape.dims(); ++i)
        {
            right_dim_size *= output_shape.dim_size(i);
        }

        TensorShape input_shape({left_dim_size, 1, right_dim_size});

        DmlKernelTensors tensors;

        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i)
        {
            DmlTensorInfo input_info;
            input_info.kernel_index = i;
            input_info.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                input_shape,
                input_shape);

            tensors.inputs.push_back(std::move(input_info));
        }

        int axis_size = output_shape.dim_size(axis);
        output_shape = TensorShape({left_dim_size, axis_size, right_dim_size});

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);
        tensors.outputs = {output};

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());

        std::vector<dml::Expression> input_tensors;
        input_tensors.reserve(inputs.size());

        for (int i = 0; i < inputs.size(); ++i)
        {
            input_tensors.push_back(dml::InputTensor(scope, i, inputs[i]));
        }

        auto result = dml::Join(input_tensors, kNchwDimensionCount - 2);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class DmlPackCpuKernel : public OpKernel
{
  public:
    explicit DmlPackCpuKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        TFE_ContextOptions* context_options = TFE_NewContextOptions();
        auto context_options_cleanup = absl::MakeCleanup(
            [context_options] { TFE_DeleteContextOptions(context_options); });

        Status status;
        eager_context_ = TFE_NewContext(context_options, status.raw());
        OP_REQUIRES_OK(ctx, status);

        pack_op_ = TFE_NewOp(eager_context_, "Pack", status.raw());
        OP_REQUIRES_OK(ctx, status);

        int axis;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
        TFE_OpSetAttrInt(pack_op_, "axis", axis);
    }

    ~DmlPackCpuKernel() override
    {
        TFE_DeleteOp(pack_op_);
        TFE_DeleteContext(eager_context_);
    }

    void Compute(OpKernelContext* ctx)
    {
        Status status;
        TFE_OpSetDevice(pack_op_, "/device:CPU", status.raw());
        OP_REQUIRES_OK(ctx, status);

        std::vector<TFE_TensorHandle*> input_handles;
        auto input_handles_cleanup = absl::MakeCleanup(
            [&input_handles]
            {
                for (TFE_TensorHandle* input_handle : input_handles)
                {
                    TFE_DeleteTensorHandle(input_handle);
                }
            });

        for (int i = 0; i < ctx->num_inputs(); ++i)
        {
            const Tensor& input_tensor = ctx->input(i);
            TFE_TensorHandle* input_handle =
                TFE_NewTensorHandle(input_tensor.raw(), status.raw());
            OP_REQUIRES_OK(ctx, status);
            input_handles.push_back(input_handle);
        }

        TFE_OpAddInputList(
            pack_op_,
            input_handles.data(),
            input_handles.size(),
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        TFE_TensorHandle* output_handle = nullptr;
        TFE_TensorHandle** output_handle_ptr = &output_handle;
        OP_REQUIRES_OK(ctx, status);
        auto output_handle_cleanup =
            absl::MakeCleanup([output_handle_ptr]
                              { TFE_DeleteTensorHandle(*output_handle_ptr); });

        int num_retvals = 1;
        TFE_Execute(pack_op_, &output_handle, &num_retvals, status.raw());
        OP_REQUIRES_OK(ctx, status);

        TF_Tensor* output =
            TFE_TensorHandleResolve(output_handle, status.raw());
        OP_REQUIRES_OK(ctx, status);

        OP_REQUIRES_OK(ctx, ctx->set_output(0, Tensor(output)));
    }

  private:
    TFE_Context* eager_context_ = nullptr;
    TFE_Op* pack_op_ = nullptr;
};

void RegisterPack()
{
    using K = KernelDefinition<
        ops::Pack,
        DmlKernelWrapper<DmlPackKernel, PackShapeHelper>>;

    RegisterWithTypes<
        K,
        ops::Pack::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT64,
        TF_INT16,
        TF_BOOL>();
}

void RegisterPackCpu()
{
    KernelDefinition<ops::Pack, DmlPackCpuKernel>::WithHostMemoryArguments<
        ops::Pack::Argument::values,
        ops::Pack::Argument::output>::
        WithTypeConstraint<ops::Pack::Attribute::T, TF_INT32>::Register();
}

void RegisterKernels_Pack()
{
    RegisterPack();
    RegisterPackCpu();
}

} // namespace tfdml