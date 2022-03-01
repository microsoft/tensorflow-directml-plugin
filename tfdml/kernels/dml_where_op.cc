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
#include <Windows.h>

namespace tfdml
{

class DummyInitializationHelper : public InitializationHelper
{
};

class DmlWhereHelper : public DmlKernel
{
  public:
    using DmlKernel::Compute;

    explicit DmlWhereHelper(
        OpKernelContext* ctx,
        const TensorShape& output_count_shape,
        const TensorShape& output_coordinates_shape)
    {
        Tensor input_tensor = ctx->input(0);
        auto input_desc = DmlTensorDesc::Create(
            input_tensor.dtype(),
            input_tensor.shape(),
            input_tensor.shape());

        auto output_count_desc = DmlTensorDesc::Create(
            TF_UINT32,
            output_count_shape,
            output_count_shape);

        auto output_coordinates_desc = DmlTensorDesc::Create(
            TF_INT64,
            output_coordinates_shape,
            output_coordinates_shape);

        DmlTensorInfo input_info = {};
        input_info.kernel_index = 0;
        input_info.desc = input_desc;

        DmlTensorInfo output_count_info = {};
        output_count_info.kernel_index = 0;
        output_count_info.desc = output_count_desc;

        DmlTensorInfo output_coordinates_info = {};
        output_coordinates_info.kernel_index = 1;
        output_coordinates_info.desc = output_coordinates_desc;

        DmlKernelTensors tensors = {};
        tensors.inputs = {input_info};
        tensors.outputs = {output_count_info, output_coordinates_info};

        const DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(dml_device->GetDmlDevice());
        const auto input = dml::InputTensor(scope, 0, inputs[0]);
        auto nonzero_coordinates_result = dml::NonZeroCoordinates(
            input,
            static_cast<uint32_t>(input_tensor.dims()));
        auto num_nonzero_coordinates = nonzero_coordinates_result.count;
        auto nonzero_coordinates = nonzero_coordinates_result.coordinates;

        nonzero_coordinates =
            dml::Cast(nonzero_coordinates, DML_TENSOR_DATA_TYPE_INT64);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(
                DML_EXECUTION_FLAG_NONE,
                {num_nonzero_coordinates, nonzero_coordinates});

        auto init_helper = std::make_shared<DummyInitializationHelper>();

        status_ = Initialize(
            ctx->raw(),
            std::move(tensors),
            compiled_op.Get(),
            std::move(init_helper),
            dml_device->GetDmlDevice(),
            dml_device->GetDeviceContext(),
            "Where");
    }

    StatusOr<DmlGpuEvent> Compute(
        OpKernelContext* ctx,
        const Tensor& input_tensor,
        const Tensor& output_count_tensor,
        const Tensor& output_coordinates_tensor) const
    {
        const DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());

        D3D12BufferRegion input_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(input_tensor);

        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            input_buffer.GetBufferBinding(),
        };

        D3D12BufferRegion output_count_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(
                output_count_tensor);

        D3D12BufferRegion output_coordinates_buffer =
            dml_device->GetDeviceContext()->GetBufferForTensor(
                output_coordinates_tensor);

        // Bind the first input as the output, to take advantage of in-place
        // execution
        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            output_count_buffer.GetBufferBinding(),
            output_coordinates_buffer.GetBufferBinding(),
        };

        return DmlKernel::Compute(
            ctx->raw(),
            dml_device->GetDmlDevice(),
            dml_device->GetDeviceContext(),
            input_bindings,
            output_bindings);
    }

    const Status& GetStatus() const { return status_; }

  private:
    Status status_;
};

class DmlWhereKernel : public OpKernel
{
  public:
    explicit DmlWhereKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* ctx)
    {
        Tensor input_tensor = ctx->input(0);
        OP_REQUIRES(
            ctx,
            input_tensor.dims() <= 8,
            errors::InvalidArgument(
                "DML doesn't support more than 8D for Where, but ",
                input_tensor.dims(),
                " dimensions were provided."));

        int input_dims = input_tensor.dims();

        TensorShape output_count_shape;
        for (int i = 0; i < input_dims; ++i)
        {
            output_count_shape.AddDim(1);
        }

        TensorShape output_coordinates_shape;
        for (int i = 0; i < input_dims - 2; ++i)
        {
            output_coordinates_shape.AddDim(1);
        }

        output_coordinates_shape.AddDim(ctx->input(0).NumElements());
        output_coordinates_shape.AddDim(input_dims);

        Tensor output_count_tensor;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                TF_UINT32,
                output_count_shape,
                &output_count_tensor,
                false));

        Tensor output_coordinates_tensor;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                TF_INT64,
                output_coordinates_shape,
                &output_coordinates_tensor,
                false));

        DmlWhereHelper where_helper(
            ctx,
            output_count_shape,
            output_coordinates_shape);
        OP_REQUIRES_OK(ctx, where_helper.GetStatus());

        StatusOr<DmlGpuEvent> status_or_event = where_helper.Compute(
            ctx,
            input_tensor,
            output_count_tensor,
            output_coordinates_tensor);
        OP_REQUIRES_OK(ctx, status_or_event.status());

        // Copy the number of nonzero coordinates back to the CPU to be able to
        // allocate the real output shape
        Tensor output_count_tensor_cpu;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                TF_UINT32,
                output_count_shape,
                &output_count_tensor_cpu,
                true));

        DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());

        OP_REQUIRES_OK(
            ctx,
            dml_device->GetDeviceContext()->CopyDeviceTensorToCPU(
                dml_device,
                &output_count_tensor,
                &output_count_tensor_cpu));

        uint32_t num_nonzero_elements =
            output_count_tensor_cpu.base<uint32_t>()[0];

        // Allocate output with its compressed shape
        TensorShape output_shape({num_nonzero_elements, input_dims});

        StatusOr<Tensor> status_or_output =
            ctx->allocate_output(0, output_shape);
        OP_REQUIRES_OK(ctx, status_or_output.status());

        dml_device->CopyTensorInSameDevice(
            &output_coordinates_tensor,
            &status_or_output.ValueOrDie());
    }
};

void RegisterKernels_Where()
{
    using K = KernelDefinition<ops::Where, DmlWhereKernel>;

    RegisterWithTypes<
        K,
        ops::Where::Attribute::T,
        TF_FLOAT,
        TF_INT8,
        TF_UINT8,
        TF_INT64,
        TF_BOOL>();
}

} // namespace tfdml