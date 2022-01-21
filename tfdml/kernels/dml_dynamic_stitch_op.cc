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

#if _WIN32
#define NOMINMAX
#endif

#include "tfdml/kernels/pch.h"

namespace tfdml
{

static bool SameExtraShape(
    const Tensor& data0,
    const Tensor& indices0,
    const Tensor& data1,
    const Tensor& indices1)
{
    const int extra0 = data0.dims() - indices0.dims();
    const int extra1 = data1.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++)
    {
        if (data0.dim_size(indices0.dims() + i) !=
            data1.dim_size(indices1.dims() + i))
        {
            return false;
        }
    }
    return true;
}

class DmlDynamicStitchKernel : public OpKernel
{
  public:
    explicit DmlDynamicStitchKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_inputs_));
    }

    void Compute(OpKernelContext* ctx)
    {
        // Find maximum index in the indices vectors
        std::vector<Tensor> indices_inputs;
        for (int i = 0; i < num_inputs_; ++i)
        {
            auto input = ctx->input(i);
            indices_inputs.push_back(std::move(input));
        }

        int32_t max_index = -1;

        for (const Tensor& indices : indices_inputs)
        {
            if (indices.NumElements() > 0)
            {
                max_index = std::max(
                    *std::max_element(
                        indices.base<int32_t>(),
                        indices.base<int32_t>() + indices.NumElements()),
                    max_index);
            }
        }

        std::vector<Tensor> data_inputs;
        for (int i = num_inputs_; i < num_inputs_ * 2; ++i)
        {
            auto input = ctx->input(i);
            data_inputs.push_back(std::move(input));
        }

        const Tensor& data0 = (data_inputs)[0];
        const Tensor& indices0 = (indices_inputs)[0];
        for (int input_num = 0; input_num < indices_inputs.size(); input_num++)
        {
            const Tensor& indices = (indices_inputs)[input_num];
            const Tensor& data = (data_inputs)[input_num];
            OP_REQUIRES(
                ctx,
                TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
                errors::InvalidArgument(
                    "data[",
                    input_num,
                    "].shape = ",
                    data.shape().DebugString(),
                    " does not start with indices[",
                    input_num,
                    "].shape = ",
                    indices.shape().DebugString()));
            OP_REQUIRES(
                ctx,
                input_num == 0 ||
                    SameExtraShape(data0, indices0, data, indices),
                errors::InvalidArgument(
                    "Need data[0].shape[",
                    indices0.dims(),
                    ":] = data[",
                    input_num,
                    "].shape[",
                    indices.dims(),
                    ":], got data[0].shape = ",
                    data0.shape().DebugString(),
                    ", data[",
                    input_num,
                    "].shape = ",
                    data.shape().DebugString(),
                    ", indices[0].shape = ",
                    indices0.shape().DebugString(),
                    ", indices[",
                    input_num,
                    "].shape = ",
                    indices.shape().DebugString()));
        }

        int first_dim_size = max_index + 1;
        TensorShape output_shape({first_dim_size});
        for (int d = indices0.dims(); d < data0.dims(); d++)
        {
            output_shape.AddDim(data0.dim_size(d));
        }

        StatusOr<Tensor> status_or_output_tensor =
            ctx->allocate_output(0, output_shape);
        OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

        if (output_shape.num_elements() == 0)
        {
            return;
        }

        DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
        auto* device_context = device->GetDeviceContext();

        const uint64_t data_type_size =
            DataTypeSize(ctx->expected_output_dtype(0));

        const uint64_t byte_stride = output_shape.num_elements() /
                                     output_shape.dim_size(0) * data_type_size;

        std::vector<D3D12BufferRegion> input_buffers;
        input_buffers.reserve(data_inputs.size());

        for (const Tensor& data_tensor : data_inputs)
        {
            if (data_tensor.NumElements() == 0)
            {
                input_buffers.push_back({});
                continue;
            }

            D3D12BufferRegion input_buffer =
                device_context->GetBufferForTensor(data_tensor);

            input_buffers.push_back(std::move(input_buffer));
        }

        D3D12BufferRegion output_buffer = device_context->GetBufferForTensor(
            status_or_output_tensor.ValueOrDie());

        assert(indices_inputs.size() == data_inputs.size());
        for (int tensor_idx = 0; tensor_idx < indices_inputs.size();
             ++tensor_idx)
        {
            const Tensor& indices_tensor = indices_inputs[tensor_idx];
            const Tensor& data_tensor = data_inputs[tensor_idx];

            const D3D12BufferRegion& input_buffer = input_buffers[tensor_idx];

            if (!input_buffer)
            {
                assert(indices_tensor.NumElements() == 0);
                continue;
            }

            const auto& indices = indices_tensor.base<int32_t>();
            for (int i = 0; i < indices_tensor.NumElements(); ++i)
            {
                int32_t output_idx = indices[i];

                const uint64_t src_offset = byte_stride * i;
                const uint64_t dst_offset = byte_stride * output_idx;

                device_context->CopyBufferToBuffer(
                    output_buffer.Subregion(dst_offset),
                    input_buffer.Subregion(src_offset, byte_stride));
            }
        }
    }

  private:
    int32_t num_inputs_;
};

void RegisterDynamicStitch()
{
    using K = KernelDefinition<ops::DynamicStitch, DmlDynamicStitchKernel>::
        WithHostMemoryArgument<ops::DynamicStitch::Argument::indices>;

    RegisterWithTypes<
        K,
        ops::DynamicStitch::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();
}

void RegisterParallelDynamicStitch()
{
    using K =
        KernelDefinition<ops::ParallelDynamicStitch, DmlDynamicStitchKernel>::
            WithHostMemoryArgument<
                ops::ParallelDynamicStitch::Argument::indices>;

    RegisterWithTypes<
        K,
        ops::ParallelDynamicStitch::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_INT32,
        TF_INT64>();
}

void RegisterKernels_DynamicStitch()
{
    RegisterDynamicStitch();
    RegisterParallelDynamicStitch();
}

} // namespace tfdml