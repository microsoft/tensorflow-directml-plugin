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

#include "dml_device.h"

#include "dml_adapter_impl.h"
#include "dml_bfc_allocator.h"
#include "dml_common.h"
#include "dml_device_context.h"
#include "dml_device_state.h"
#include "dml_event_queue.h"
#include "dml_kernel_manager.h"
#include "dml_readback_heap.h"
#include "dml_tracing.h"
#include "dml_upload_heap.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/runtime_adapter/allocator.h"
#include "tfdml/runtime_adapter/tensor.h"

namespace tfdml
{

DmlDevice::DmlDevice(const DmlDeviceState* state, uint32_t device_ordinal)
    : state_(state),
      device_ordinal_(device_ordinal)
{
    device_context_ = absl::make_unique<DMLDeviceContext>(
        state_->execution_context.get(),
        state_->event_queue.get(),
        state_->upload_heap.get(),
        state_->readback_heap.get(),
        state_->dml_allocator.get(),
        state_->descriptor_allocator.get());
}

Status DmlDevice::Sync()
{
    TF_VLog(2, "DirectML device: performing GPU sync.");

    auto start_time = std::chrono::high_resolution_clock::now();

    auto status_or_event = state_->execution_context->Flush();
    TF_RETURN_IF_ERROR(status_or_event.status());
    status_or_event.ConsumeValueOrDie().WaitForSignal();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> wait_seconds = end_time - start_time;
    TF_VLog(
        2,
        "DirectML device: GPU sync took %lf ms.",
        wait_seconds.count() * 1e3);

    // Take the opportunity to free some memory if needed
    state_->kernel_manager->ReleaseCompletedReferences();
    return Status::OK();
}

Status DmlDevice::CopyCPUTensorToDevice(
    const Tensor* cpu_tensor,
    Tensor* device_tensor)
{
    return device_context_->CopyCPUTensorToDevice(
        this,
        cpu_tensor,
        device_tensor);
}

Status DmlDevice::CopyDeviceTensorToCPU(
    const Tensor* device_tensor,
    Tensor* cpu_tensor)
{
    return device_context_->CopyDeviceTensorToCPU(
        this,
        device_tensor,
        cpu_tensor);
}

Status DmlDevice::CopyDeviceTensorsToCPU(
    absl::Span<const Tensor> device_tensors,
    absl::Span<Tensor> cpu_tensors)
{
    return device_context_->CopyDeviceTensorsToCPU(
        this,
        device_tensors,
        cpu_tensors);
}

void DmlDevice::CopyTensorInSameDevice(
    const Tensor* input_tensor,
    Tensor* output_tensor)
{
    // Forward to the device context where the real implementation lives
    device_context_->CopyTensorInSameDevice(this, input_tensor, output_tensor);
}

ID3D12Device* DmlDevice::GetD3D12Device() const
{
    return state_->d3d_device.Get();
}

IDMLDevice* DmlDevice::GetDmlDevice() const { return state_->dml_device.Get(); }

DmlAllocator* DmlDevice::GetAllocator() const
{
    return state_->dml_allocator.get();
}

DmlDescriptorAllocator* DmlDevice::GetDescriptorAllocator() const
{
    return state_->descriptor_allocator.get();
}

DmlKernelManager* DmlDevice::GetKernelManager() const
{
    return state_->kernel_manager.get();
}

DmlExecutionContext* DmlDevice::GetExecutionContext() const
{
    return state_->execution_context.get();
}

DmlUploadHeap* DmlDevice::GetUploadHeap() const
{
    return state_->upload_heap.get();
}

DmlReadbackHeap* DmlDevice::GetReadbackHeap() const
{
    return state_->readback_heap.get();
}

DmlEventQueue* DmlDevice::GetEventQueue() const
{
    return state_->event_queue.get();
}

DMLDeviceContext* DmlDevice::GetDeviceContext() const
{
    return device_context_.get();
}

absl::optional<uint32_t> DmlDevice::TryLogKernelComputeStart(
    const absl::string_view type,
    const absl::string_view name) const
{
    return DmlTracing::Instance().TryLogKernelComputeStart(
        device_ordinal_,
        type,
        name);
}

void DmlDevice::LogKernelComputeEnd(uint32_t event_id) const
{
    DmlTracing::Instance().LogKernelComputeEnd(device_ordinal_, event_id);
}

} // namespace tfdml