/* Copyright (c) Microsoft Corporation.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "dml_device_context.h"

#include "dml_bfc_allocator.h"
#include "dml_util.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "tfdml/core/dml_device.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

Status DMLDeviceContext::CopyCPUTensorToDevice(
    DmlDevice* device,
    const Tensor* cpu_tensor,
    Tensor* device_tensor) const
{
    size_t total_bytes = cpu_tensor->TotalBytes();
    if (total_bytes == 0)
    {
        return Status::OK();
    }

    const void* src_data = cpu_tensor->base<void>();

    D3D12BufferRegion dst = GetBufferForTensor(*device_tensor);

    auto byte_span = absl::Span<const uint8_t>(
        static_cast<const uint8_t*>(src_data),
        total_bytes);

    StatusOr<DmlGpuEvent> status_or_event =
        upload_heap_->BeginUploadToGpu(dst, byte_span);

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
    return status_or_event.status();
}

void DMLDeviceContext::CopyTensorInSameDevice(
    DmlDevice* device,
    const Tensor* input_tensor,
    Tensor* output_tensor) const
{
    auto total_bytes = static_cast<uint64_t>(output_tensor->TotalBytes());
    if (total_bytes == 0)
    {
        return;
    }

    D3D12BufferRegion src = GetBufferForTensor(*input_tensor);
    D3D12BufferRegion dst = GetBufferForTensor(*output_tensor);

    (void)execution_context_->CopyBufferRegion(
        dst,
        src.Subregion(0, total_bytes));

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
}

Status DMLDeviceContext::CopyDeviceTensorToCPU(
    DmlDevice* device,
    const Tensor* device_tensor,
    Tensor* cpu_tensor)
{
    size_t total_bytes = cpu_tensor->TotalBytes();
    if (total_bytes == 0)
    {
        return Status::OK();
    }

    D3D12BufferRegion src = GetBufferForTensor(*device_tensor);

    void* dst_data = cpu_tensor->base<void>();

    // Performs a blocking call to synchronize and read back data from the GPU
    // into the destination buffer
    auto byte_span =
        absl::Span<uint8_t>(static_cast<uint8_t*>(dst_data), total_bytes);

    StatusOr<DmlGpuEvent> status_or_event =
        readback_heap_->ReadbackFromGpu(byte_span, src);

    if (!status_or_event.ok())
    {
        return status_or_event.status();
    }

    TF_RETURN_IF_ERROR(device->Sync());
    status_or_event.ConsumeValueOrDie().WaitForSignal();
    return Status::OK();
}

Status DMLDeviceContext::CopyCPUMemoryToDevice(
    DmlDevice* device,
    const void* cpu_memory,
    SP_DeviceMemoryBase* device_memory,
    uint64_t size_in_bytes) const
{
    if (size_in_bytes == 0)
    {
        return Status::OK();
    }

    D3D12BufferRegion dst =
        device->GetDeviceContext()->GetBufferForDeviceMemory(
            device_memory,
            size_in_bytes);

    auto byte_span = absl::Span<const uint8_t>(
        static_cast<const uint8_t*>(cpu_memory),
        size_in_bytes);

    StatusOr<DmlGpuEvent> status_or_event =
        upload_heap_->BeginUploadToGpu(dst, byte_span);

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
    return status_or_event.status();
}

StatusOr<DmlGpuEvent> DMLDeviceContext::CopyDeviceMemoryToCPU(
    DmlDevice* device,
    const SP_DeviceMemoryBase* device_memory,
    void* cpu_memory,
    uint64_t size_in_bytes)
{
    if (size_in_bytes == 0)
    {
        return Status::OK();
    }

    D3D12BufferRegion src =
        device->GetDeviceContext()->GetBufferForDeviceMemory(
            device_memory,
            size_in_bytes);

    // Performs a blocking call to synchronize and read back data from the GPU
    // into the destination buffer
    auto byte_span =
        absl::Span<uint8_t>(static_cast<uint8_t*>(cpu_memory), size_in_bytes);

    StatusOr<DmlGpuEvent> status_or_event =
        readback_heap_->ReadbackFromGpu(byte_span, src);

    return status_or_event;
}

void DMLDeviceContext::CopyMemoryInSameDevice(
    DmlDevice* device,
    const SP_DeviceMemoryBase* input_memory,
    const SP_DeviceMemoryBase* output_memory,
    uint64_t size_in_bytes) const
{
    if (size_in_bytes == 0)
    {
        return;
    }

    D3D12BufferRegion src =
        device->GetDeviceContext()->GetBufferForDeviceMemory(
            input_memory,
            size_in_bytes);

    D3D12BufferRegion dst =
        device->GetDeviceContext()->GetBufferForDeviceMemory(
            output_memory,
            size_in_bytes);

    (void)device->GetExecutionContext()->CopyBufferRegion(dst, src);

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
}

DmlGpuEvent DMLDeviceContext::BindAndInitializeOperator(
    IDMLOperatorInitializer* initializer,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* heap_for_binding_table,
    _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding)
{
    // Bind the temporary resource
    if (temporary_resource_binding)
    {
        DML_BINDING_DESC temporary_binding_desc = {
            DML_BINDING_TYPE_BUFFER,
            temporary_resource_binding};
        binding_table->BindTemporaryResource(&temporary_binding_desc);
    }

    // Bind the persistent resource
    if (persistent_resource_binding)
    {
        DML_BINDING_DESC persistent_binding_desc = {
            DML_BINDING_TYPE_BUFFER,
            persistent_resource_binding};
        binding_table->BindOutputs(1, &persistent_binding_desc);
    }

    return execution_context_->InitializeOperator(
        initializer,
        std::move(binding_table),
        heap_for_binding_table);
}

DmlGpuEvent DMLDeviceContext::BindAndExecuteOperator(
    IDMLCompiledOperator* op,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* heap_for_binding_table,
    _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> input_bindings,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> output_bindings)
{
    // Bind the temporary resource
    DML_BINDING_DESC temporary_binding_desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (temporary_resource_binding)
    {
        temporary_binding_desc = {
            DML_BINDING_TYPE_BUFFER,
            temporary_resource_binding};
    }
    binding_table->BindTemporaryResource(&temporary_binding_desc);

    // Bind the persistent resource
    DML_BINDING_DESC persistent_binding_desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (persistent_resource_binding)
    {
        persistent_binding_desc = {
            DML_BINDING_TYPE_BUFFER,
            persistent_resource_binding};
    }
    binding_table->BindPersistentResource(&persistent_binding_desc);

    // Set up the input bindings
    absl::InlinedVector<DML_BINDING_DESC, 8> input_binding_descs;
    for (const auto& binding : input_bindings)
    {
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
        if (binding)
        {
            desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
        }

        input_binding_descs.push_back(desc);
    }
    binding_table->BindInputs(
        static_cast<UINT>(input_binding_descs.size()),
        input_binding_descs.data());

    // Set up the output bindings
    absl::InlinedVector<DML_BINDING_DESC, 4> output_binding_descs;
    for (const auto& binding : output_bindings)
    {
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
        if (binding)
        {
            desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
        }

        output_binding_descs.push_back(desc);
    }
    binding_table->BindOutputs(
        static_cast<UINT>(output_binding_descs.size()),
        output_binding_descs.data());

    return execution_context_->ExecuteOperator(
        op,
        std::move(binding_table),
        heap_for_binding_table);
}

DmlGpuEvent DMLDeviceContext::InsertUavBarrier() const
{
    return execution_context_->UavBarrier();
}

DmlGpuEvent DMLDeviceContext::GetCurrentCompletionEvent() const
{
    return execution_context_->GetCurrentCompletionEvent();
}

void DMLDeviceContext::EnqueueCallbackForGpuEvent(
    DmlGpuEvent gpu_event,
    std::function<void()> callback) const
{
    event_queue_->Enqueue(std::move(gpu_event), std::move(callback));
}

DmlBuffer DMLDeviceContext::AllocateDefaultBuffer(
    TF_OpKernelContext* op_kernel_context,
    uint64_t num_bytes) const
{
    return DmlBuffer(op_kernel_context, allocator_, num_bytes);
}

D3D12BufferRegion GetBufferForOpaqueData(
    DmlAllocator* allocator,
    const void* opaque_data,
    uint64_t unaligned_size_in_bytes)
{
    // DML always requires at least 4 byte alignment in all cases, so both the
    // offset and size must certainly be divisible by 4.
    constexpr uint64_t DML_ALIGNMENT = 4;

    // The offset and size of the region must be aligned to DirectML's
    // requirement. Each tensor has two sizes:
    //
    // - TotalBytes: num_elements * sizeof_element. This may be too small if the
    // tensor has elements smaller than 4 bytes (e.g. 3x float16 is 6 bytes, but
    // DML needs an 8 byte region).
    //
    // - AllocatedBytes: the size of allocation backing the tensor. This is
    // often larger than TotalBytes since the smallest DML allocation size is
    // 256 bytes.
    //
    // While AllocatedBytes is guaranteed to meet DML's requirement, tensor
    // buffers may be offset within an individual allocation (see
    // Tensor::Slice). Using AllocatedBytes directly can result in a region that
    // extends beyond the bounds of the allocation. Instead we round the total
    // bytes up to an aligned value, which should always fit within the
    // allocated bytes.
    uint64_t size_in_bytes =
        (1 + (unaligned_size_in_bytes - 1) / DML_ALIGNMENT) * DML_ALIGNMENT;

    auto region = allocator->CreateBufferRegion(opaque_data, size_in_bytes);

    // DML always requires at least 4 byte alignment in all cases, so both the
    // offset and size must certainly be divisible by 4
    assert(region.Offset() % DML_ALIGNMENT == 0);
    assert(region.SizeInBytes() % DML_ALIGNMENT == 0);

    return region;
}

D3D12BufferRegion DMLDeviceContext::GetBufferForTensor(
    const Tensor& tensor) const
{
    const void* p = tensor.tensor_data().data();
    return GetBufferForOpaqueData(allocator_, p, tensor.TotalBytes());
}

D3D12BufferRegion DMLDeviceContext::GetBufferForTensor(
    const TF_Tensor* tensor) const
{
    const void* p = TF_TensorData(tensor);
    size_t total_bytes = TF_TensorByteSize(tensor);
    return GetBufferForOpaqueData(allocator_, p, total_bytes);
}

D3D12BufferRegion DMLDeviceContext::GetBufferForDeviceMemory(
    const SP_DeviceMemoryBase* data,
    uint64_t size_in_bytes)
{
    return GetBufferForOpaqueData(allocator_, data->opaque, size_in_bytes);
}

DescriptorAllocation DMLDeviceContext::AllocateDescriptors(
    size_t size_in_descriptors) const
{
    return descriptor_allocator_->Alloc(size_in_descriptors);
}

DmlGpuEvent DMLDeviceContext::CopyBufferToBuffer(
    const D3D12BufferRegion& dst,
    const D3D12BufferRegion& src) const
{
    return execution_context_->CopyBufferRegion(dst, src);
}

StatusOr<DmlGpuEvent> DMLDeviceContext::CopyHostToBuffer(
    const D3D12BufferRegion& dst,
    absl::Span<const uint8_t> src) const
{
    return upload_heap_->BeginUploadToGpu(dst, src);
}

// Fills dst region with zeroes.
DmlGpuEvent DMLDeviceContext::ZeroBuffer(const D3D12BufferRegion& dst) const
{
    uint8_t pattern[] = {0};
    return FillBufferWithPattern(dst, pattern);
}

// Fills dst region with a repeating byte pattern.
DmlGpuEvent DMLDeviceContext::FillBufferWithPattern(
    const D3D12BufferRegion& dst,
    absl::Span<const uint8_t> value) const
{
    return execution_context_->FillBufferWithPattern(dst, value);
}

} // namespace tfdml