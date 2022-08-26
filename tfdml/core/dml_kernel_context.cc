/* Copyright (c) Microsoft Corporation.

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

#include "tfdml/core/dml_kernel_context.h"

#include "tfdml/core/dml_device.h"
#include "tfdml/core/dml_event_queue.h"
#include "tfdml/core/dml_execution_context.h"
#include "tfdml/core/dml_upload_heap.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/tensor.h"

using Microsoft::WRL::ComPtr;

namespace tfdml
{

//
// DmlKernelConstruction
//

DmlKernelConstruction::DmlKernelConstruction(
    const DmlDevice* device,
    OpKernelContext* op_ctx,
    absl::Span<const TensorShape> output_shapes,
    std::shared_ptr<const InitializationHelper> init_helper)
    : device_(device),
      op_ctx_(op_ctx),
      output_shapes_(output_shapes),
      init_helper_(init_helper)
{
}

IDMLDevice* DmlKernelConstruction::GetDmlDevice() const
{
    return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelConstruction::GetD3D12Device() const
{
    return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelConstruction::GetOpKernelContext() const
{
    return op_ctx_;
}

DMLDeviceContext* DmlKernelConstruction::GetDmlDeviceContext() const
{
    return device_->GetDeviceContext();
}

std::shared_ptr<const InitializationHelper> DmlKernelConstruction::
    GetInitializationHelper() const
{
    return init_helper_;
}

TF_DataType DmlKernelConstruction::GetInputDataType(uint32_t index) const
{
    return op_ctx_->input_dtype(index);
}

TensorShape DmlKernelConstruction::GetInputTensorShape(uint32_t index) const
{
    return op_ctx_->input(index).shape();
}

Tensor DmlKernelConstruction::GetConstantInputTensor(uint32_t index) const
{
    CHECK(op_ctx_->input_memory_type(index) == HOST_MEMORY);
    CHECK(op_ctx_->input_dtype(index) != TF_RESOURCE);

    return op_ctx_->input(index);
}

TF_DataType DmlKernelConstruction::GetOutputDataType(uint32_t index) const
{
    return op_ctx_->expected_output_dtype(index);
}

const TensorShape& DmlKernelConstruction::GetOutputTensorShape(
    uint32_t index) const
{
    return output_shapes_[index];
}

//
// DmlKernelContext
//

DmlKernelContext::DmlKernelContext(
    const DmlDevice* device,
    OpKernelContext* op_ctx,
    const InitializationHelper* init_helper,
    absl::Span<const TensorShape> output_shapes,
    absl::Span<const absl::optional<uint32_t>> output_refs_forwarding)
    : device_(device),
      op_ctx_(op_ctx),
      init_helper_(init_helper)
{
    assert(output_shapes.size() == op_ctx_->num_outputs());

    // Allocate output tensors
    output_tensors_.reserve(output_shapes.size());
    for (int i = 0; i < static_cast<int>(output_shapes.size()); ++i)
    {
        if (i < output_refs_forwarding.size() &&
            output_refs_forwarding[i].has_value())
        {
            constexpr bool lock_held = true;
            constexpr bool is_variant = false;
            op_ctx->forward_ref_input_to_ref_output(
                *output_refs_forwarding[i],
                i);

            Tensor output_tensor;
            OP_REQUIRES_OK(
                op_ctx_,
                op_ctx->GetInputTensorFromVariable(
                    i,
                    lock_held,
                    is_variant,
                    &output_tensor));
            output_tensors_.push_back(std::move(output_tensor));
        }
        else
        {
            absl::InlinedVector<int, 4> candidate_input_indices(
                op_ctx_->num_inputs());
            std::iota(
                candidate_input_indices.begin(),
                candidate_input_indices.end(),
                0);

            int forwarded_input_index = -1;
            auto status_or_tensor = op_ctx_->forward_input_or_allocate_output(
                candidate_input_indices,
                i,
                output_shapes[i],
                &forwarded_input_index);
            OP_REQUIRES_OK(op_ctx_, status_or_tensor.status());
            output_tensors_.push_back(status_or_tensor.ConsumeValueOrDie());

            // TODO: Remove this after testing is done
            if (forwarded_input_index != -1)
            {
                printf("*********FORWARDED INPUT: %d\n", forwarded_input_index);
            }
        }
    }
}

IDMLDevice* DmlKernelContext::GetDmlDevice() const
{
    return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelContext::GetD3D12Device() const
{
    return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelContext::GetOpKernelContext() const
{
    return op_ctx_;
}

DMLDeviceContext* DmlKernelContext::GetDmlDeviceContext() const
{
    return device_->GetDeviceContext();
}

} // namespace tfdml