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

#include <stdio.h>

#if !_WIN32
#include <sys/sysinfo.h>
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "absl/cleanup/cleanup.h"
#include "plugin_version.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tfdml/core/dml_bfc_allocator.h"
#include "tfdml/core/dml_common.h"
#include "tfdml/core/dml_device.h"
#include "tfdml/core/dml_device_cache.h"
#include "tfdml/core/dml_device_context.h"
#include "tfdml/core/dml_device_manager.h"
#include "tfdml/core/dml_tagged_pointer.h"
#include "tfdml/core/dml_tracing.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/runtime_adapter/stream.h"

using namespace tfdml;

void plugin_get_device_count(
    const SP_Platform* platform,
    int* device_count,
    TF_Status* status)
{
    *device_count = tfdml::DmlDeviceCache::Instance().GetAdapterCount();
    TF_SetStatus(status, TF_OK, "");
}

void plugin_create_device(
    const SP_Platform* platform,
    SE_CreateDeviceParams* params,
    TF_Status* const status)
{
    auto& device_cache = DmlDeviceCache::Instance();
    const auto* device_state =
        device_cache.GetOrCreateDeviceState(params->ordinal);

    params->device->ordinal = params->ordinal;
    params->device->device_vendor =
        GetVendorName(device_state->adapter->VendorID());
    params->device->hardware_name = device_state->adapter->Name().c_str();
    params->device->struct_size = SP_DEVICE_STRUCT_SIZE;

    DmlDevice* dml_device = new DmlDevice(device_state, params->ordinal);
    params->device->device_handle = dml_device;
    Status insert_device_status =
        DmlDeviceManager::Instance().InsertDevice(params->ordinal, dml_device);

    if (!insert_device_status.ok())
    {
        TF_SetStatus(
            status,
            insert_device_status.code(),
            insert_device_status.error_message());
        return;
    }

#ifdef DIRECTML_ENABLE_TELEMETRY
    DmlTracing::Instance().LogDeviceCreationTelemetry(
        device_state->adapter->AdapterLuid(),
        params->ordinal);
#endif

    TF_SetStatus(status, TF_OK, "");
}

void plugin_destroy_device(const SP_Platform* platform, SP_Device* device)
{
    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);
    delete dml_device;
    device->device_handle = nullptr;
    device->ordinal = -1;
}

void plugin_create_device_fns(
    const SP_Platform* platform,
    SE_CreateDeviceFnsParams* params,
    TF_Status* status)
{
    TF_SetStatus(status, TF_OK, "");
    params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void plugin_destroy_device_fns(
    const SP_Platform* platform,
    SP_DeviceFns* device_fns)
{
}

/*StreamExecutor Backend Impl*/
void plugin_allocate(
    const SP_Device* device,
    uint64_t size,
    int64_t memory_space,
    SP_DeviceMemoryBase* mem)
{
    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);
    DmlAllocator* allocator = dml_device->GetAllocator();

    mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    mem->opaque = allocator->Alloc(device->ordinal, size);
    mem->size = size;
    mem->payload = 12345;
}

void plugin_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem)
{
    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);
    DmlAllocator* allocator = dml_device->GetAllocator();
    allocator->Free(mem->opaque, mem->size);

    mem->opaque = nullptr;
    mem->size = 0;
}

void* plugin_host_memory_allocate(const SP_Device* device, uint64_t size)
{
#if _WIN32
    void* ptr = _aligned_malloc(size, 64);
#else
    void* ptr = aligned_alloc(64, size);
#endif
    return ptr;
}

void plugin_host_memory_deallocate(const SP_Device* device, void* mem)
{
#if _WIN32
    _aligned_free(mem);
#else
    free(mem);
#endif
}

TF_Bool plugin_get_allocator_stats(
    const SP_Device* device,
    SP_AllocatorStats* stats)
{
    // TODO: Return allocator information
    // TFDML 40858257
    return false;
}

TF_Bool plugin_device_memory_usage(
    const SP_Device* device,
    int64_t* free,
    int64_t* total)
{
    auto& device_cache = DmlDeviceCache::Instance();
    const DmlAdapter& adapter = device_cache.GetAdapter(device->ordinal);

    uint64_t total_gpu_memory = adapter.GetTotalDedicatedMemory();

    if (adapter.IsUmaAdapter())
    {
        total_gpu_memory += adapter.GetTotalSharedMemory();
    }

    *free = adapter.QueryAvailableLocalMemory();
    *total = total_gpu_memory;

    return true;
}

// We use streams to store the device pointer and make it accessible to the
// kernels.
void plugin_create_stream(
    const SP_Device* device,
    SP_Stream* stream,
    TF_Status* status)
{
    *stream = new SP_Stream_st(device->device_handle);
    TF_SetStatus(status, TF_OK, "");
}

// Destroys SP_Stream and deallocates any underlying resources.
void plugin_destroy_stream(const SP_Device* device, SP_Stream stream)
{
    delete stream;
}

void plugin_create_stream_dependency(
    const SP_Device* device,
    SP_Stream dependent,
    SP_Stream other,
    TF_Status* status)
{
}

// Without blocking the device, retrieve the current stream status.
void plugin_get_stream_status(
    const SP_Device* device,
    SP_Stream stream,
    TF_Status* status)
{
}

void plugin_create_event(
    const SP_Device* device,
    SP_Event* event,
    TF_Status* status)
{
}

// Destroy SE_Event and perform any platform-specific deallocation and
// cleanup of an event.
void plugin_destroy_event(const SP_Device* device, SP_Event event) {}

// Requests the current status of the event from the underlying platform.
SE_EventStatus plugin_get_event_status(const SP_Device* device, SP_Event event)
{
    return SE_EVENT_COMPLETE;
}

// Inserts the specified event at the end of the specified stream.
void plugin_record_event(
    const SP_Device* device,
    SP_Stream stream,
    SP_Event event,
    TF_Status* status)
{
}

// Wait for the specified event at the end of the specified stream.
void plugin_wait_for_event(
    const SP_Device* const device,
    SP_Stream stream,
    SP_Event event,
    TF_Status* const status)
{
}

/*** TIMER CALLBACKS ***/
// Creates SP_Timer. Allocates timer resources on the underlying platform
// and initializes its internals, setting `timer` output variable. Sets
// values in `timer_fns` struct.
void plugin_create_timer(
    const SP_Device* device,
    SP_Timer* timer,
    TF_Status* status)
{
}

// Destroy timer and deallocates timer resources on the underlying platform.
void plugin_destroy_timer(const SP_Device* device, SP_Timer timer) {}

// Records a start event for an interval timer.
void plugin_start_timer(
    const SP_Device* device,
    SP_Stream stream,
    SP_Timer timer,
    TF_Status* status)
{
}

// Records a stop event for an interval timer.
void plugin_stop_timer(
    const SP_Device* device,
    SP_Stream stream,
    SP_Timer timer,
    TF_Status* status)
{
}

/*** MEMCPY CALLBACKS ***/
// Enqueues a memcpy operation onto stream, with a host destination location
// `host_dst` and a device memory source, with target size `size`.
void plugin_memcpy_dtoh(
    const SP_Device* device,
    SP_Stream stream,
    void* host_dst,
    const SP_DeviceMemoryBase* device_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    StatusOr<DmlGpuEvent> copy_status_or_event =
        dml_device->GetDeviceContext()
            ->CopyDeviceMemoryToCPU(dml_device, device_src, host_dst, size);

    Status copy_status = copy_status_or_event.status();

    if (!copy_status.ok())
    {
        TF_SetStatus(status, copy_status.code(), copy_status.error_message());
        return;
    }

    Status sync_status = dml_device->Sync();

    if (!sync_status.ok())
    {
        TF_SetStatus(status, sync_status.code(), sync_status.error_message());
        return;
    }

    // Even though this function doesn't say that it should block the caller,
    // plugin_synchronize_all_activity is called before and therefore we need a
    // way to tell the caller that all data has been downloaded back to the CPU.
    // For this reason, we avoid implementing plugin_synchronize_all_activity
    // and just sync here instead.
    copy_status_or_event.ConsumeValueOrDie().WaitForSignal();
    TF_SetStatus(status, TF_OK, "");
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a host memory source, with target size `size`.
void plugin_memcpy_htod(
    const SP_Device* device,
    SP_Stream stream,
    SP_DeviceMemoryBase* device_dst,
    const void* host_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    Status copy_status = dml_device->GetDeviceContext()->CopyCPUMemoryToDevice(
        dml_device,
        host_src,
        device_dst,
        size);

    TF_SetStatus(status, copy_status.code(), copy_status.error_message());
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a device memory source, with target size `size`.
void plugin_memcpy_dtod(
    const SP_Device* device,
    SP_Stream stream,
    SP_DeviceMemoryBase* device_dst,
    const SP_DeviceMemoryBase* device_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    TaggedPointer device_dst_tagged_pointer =
        TaggedPointer::Unpack(device_dst->opaque);
    TaggedPointer device_src_tagged_pointer =
        TaggedPointer::Unpack(device_src->opaque);

    DmlDevice* dml_device_dst = DmlDeviceManager::Instance().GetDevice(
        device_dst_tagged_pointer.device_id);

    if (device_dst_tagged_pointer.device_id ==
        device_src_tagged_pointer.device_id)
    {
        dml_device_dst->GetDeviceContext()->CopyMemoryInSameDevice(
            dml_device_dst,
            device_src,
            device_dst,
            size);
    }
    else
    {
        // TODO: Remove this workaround once better support for cross-device
        // copies exists
        // https://github.com/tensorflow/tensorflow/issues/53446
        DmlDevice* dml_device_src = DmlDeviceManager::Instance().GetDevice(
            device_src_tagged_pointer.device_id);

        // TODO: Investigate potential performance improvements (e.g. using
        // D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER and avoiding the CPU entirely)
        // If the memory is on different devices, first do a GPU->CPU copy and
        // then a CPU->GPU copy
        void* cpu_memory = plugin_host_memory_allocate(device, size);
        auto cpu_memory_cleanup = absl::MakeCleanup(
            [device, cpu_memory]
            { plugin_host_memory_deallocate(device, cpu_memory); });

        auto status_or_event =
            dml_device_src->GetDeviceContext()->CopyDeviceMemoryToCPU(
                dml_device_src,
                device_src,
                cpu_memory,
                size);

        if (!status_or_event.ok())
        {
            TF_SetStatus(
                status,
                status_or_event.status().code(),
                status_or_event.status().error_message());
            return;
        }

        status_or_event.ValueOrDie().WaitForSignal();

        auto cpu_gpu_copy_status =
            dml_device_dst->GetDeviceContext()->CopyCPUMemoryToDevice(
                dml_device_dst,
                cpu_memory,
                device_dst,
                size);

        if (!cpu_gpu_copy_status.ok())
        {
            TF_SetStatus(
                status,
                cpu_gpu_copy_status.code(),
                cpu_gpu_copy_status.error_message());
            return;
        }
    }

    TF_SetStatus(status, TF_OK, "");
}

// Blocks the caller while a data segment of the given size is
// copied from the device source to the host destination.
void plugin_sync_memcpy_dtoh(
    const SP_Device* device,
    void* host_dst,
    const SP_DeviceMemoryBase* device_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    StatusOr<DmlGpuEvent> copy_status_or_event =
        dml_device->GetDeviceContext()
            ->CopyDeviceMemoryToCPU(dml_device, device_src, host_dst, size);

    Status copy_status = copy_status_or_event.status();

    if (!copy_status.ok())
    {
        TF_SetStatus(status, copy_status.code(), copy_status.error_message());
        return;
    }

    Status sync_status = dml_device->Sync();

    if (!sync_status.ok())
    {
        TF_SetStatus(status, sync_status.code(), sync_status.error_message());
        return;
    }

    copy_status_or_event.ConsumeValueOrDie().WaitForSignal();
    TF_SetStatus(status, TF_OK, "");
}

// Blocks the caller while a data segment of the given size is
// copied from the host source to the device destination.
void plugin_sync_memcpy_htod(
    const SP_Device* device,
    SP_DeviceMemoryBase* device_dst,
    const void* host_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    Status copy_status = dml_device->GetDeviceContext()->CopyCPUMemoryToDevice(
        dml_device,
        host_src,
        device_dst,
        size);

    if (!copy_status.ok())
    {
        TF_SetStatus(status, copy_status.code(), copy_status.error_message());
        return;
    }

    Status sync_status = dml_device->Sync();
    TF_SetStatus(status, sync_status.code(), sync_status.error_message());
}

// Blocks the caller while a data segment of the given size is copied from the
// device source to the device destination.
void plugin_sync_memcpy_dtod(
    const SP_Device* device,
    SP_DeviceMemoryBase* device_dst,
    const SP_DeviceMemoryBase* device_src,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    dml_device->GetDeviceContext()
        ->CopyMemoryInSameDevice(dml_device, device_src, device_dst, size);

    TF_SetStatus(status, TF_OK, "");
}

// Causes the host code to synchronously wait for the event to complete.
void plugin_block_host_for_event(
    const SP_Device* device,
    SP_Event event,
    TF_Status* status)
{
}

void plugin_block_host_until_done(
    const SP_Device* device,
    SP_Stream stream,
    TF_Status* status)
{
}

// Zero out `size` bytes starting at the location.
void plugin_mem_zero(
    const SP_Device* device,
    SP_Stream stream,
    SP_DeviceMemoryBase* location,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    D3D12BufferRegion dst =
        dml_device->GetDeviceContext()->GetBufferForDeviceMemory(
            location,
            size);

    (void)dml_device->GetDeviceContext()->ZeroBuffer(dst);

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
    TF_SetStatus(status, TF_OK, "");
}

template <typename T>
void plugin_memset(
    const SP_Device* device,
    SP_Stream stream,
    SP_DeviceMemoryBase* location,
    T pattern,
    uint64_t size,
    TF_Status* status)
{
    if (size == 0)
    {
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    DmlDevice* dml_device = static_cast<DmlDevice*>(device->device_handle);

    D3D12BufferRegion dst =
        dml_device->GetDeviceContext()->GetBufferForDeviceMemory(
            location,
            size);

    auto pattern_bytes = absl::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(&pattern),
        sizeof(pattern));

    (void)dml_device->GetDeviceContext()->FillBufferWithPattern(
        dst,
        pattern_bytes);

    // Immediately signal completion even though we haven't actually kicked off
    // the GPU, or waited for it to complete. This is because from the
    // framework's point of view, there's no way for it to observe this state
    // (except when copying a tensor back to CPU, at which point we correctly
    // flush and queue a callback)
    TF_SetStatus(status, TF_OK, "");
}

// Synchronizes all activity occurring in the StreamExecutor's context (most
// likely a whole device).
void plugin_synchronize_all_activity(const SP_Device* device, TF_Status* status)
{
    // We only need to synchronize when copying back to the CPU
}

// Enqueues on a stream a user-specified function to be run on the host.
// `callback_arg` should be passed as the first argument to `callback_fn`.
TF_Bool plugin_host_callback(
    const SP_Device* device,
    SP_Stream stream,
    SE_StatusCallbackFn callback_fn,
    void* callback_arg)
{
    return TF_OK;
}

/*Timer Backer Impl*/
uint64_t nanoseconds(SP_Timer timer) { return timer->timer_handle; }

void plugin_create_timer_fns(
    const SP_Platform* platform,
    SP_TimerFns* timer_fns,
    TF_Status* const status)
{
    timer_fns->nanoseconds = nanoseconds;
}

void plugin_destroy_timer_fns(
    const SP_Platform* platform,
    SP_TimerFns* timer_fns)
{
}

void plugin_create_stream_executor(
    const SP_Platform* platform,
    SE_CreateStreamExecutorParams* params,
    TF_Status* const status)
{
    params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
    params->stream_executor->allocate = plugin_allocate;
    params->stream_executor->deallocate = plugin_deallocate;
    params->stream_executor->host_memory_allocate = plugin_host_memory_allocate;
    params->stream_executor->host_memory_deallocate =
        plugin_host_memory_deallocate;
    params->stream_executor->get_allocator_stats = plugin_get_allocator_stats;
    params->stream_executor->device_memory_usage = plugin_device_memory_usage;

    params->stream_executor->create_stream = plugin_create_stream;
    params->stream_executor->destroy_stream = plugin_destroy_stream;
    params->stream_executor->create_stream_dependency =
        plugin_create_stream_dependency;
    params->stream_executor->get_stream_status = plugin_get_stream_status;
    params->stream_executor->create_event = plugin_create_event;
    params->stream_executor->destroy_event = plugin_destroy_event;
    params->stream_executor->get_event_status = plugin_get_event_status;
    params->stream_executor->record_event = plugin_record_event;
    params->stream_executor->wait_for_event = plugin_wait_for_event;
    params->stream_executor->create_timer = plugin_create_timer;
    params->stream_executor->destroy_timer = plugin_destroy_timer;
    params->stream_executor->start_timer = plugin_start_timer;
    params->stream_executor->stop_timer = plugin_stop_timer;

    params->stream_executor->memcpy_dtoh = plugin_memcpy_dtoh;
    params->stream_executor->memcpy_htod = plugin_memcpy_htod;
    params->stream_executor->memcpy_dtod = plugin_memcpy_dtod;
    params->stream_executor->sync_memcpy_dtoh = plugin_sync_memcpy_dtoh;
    params->stream_executor->sync_memcpy_htod = plugin_sync_memcpy_htod;
    params->stream_executor->sync_memcpy_dtod = plugin_sync_memcpy_dtod;
    params->stream_executor->block_host_until_done =
        plugin_block_host_until_done;
    params->stream_executor->block_host_for_event = plugin_block_host_for_event;

    params->stream_executor->synchronize_all_activity =
        plugin_synchronize_all_activity;

    params->stream_executor->mem_zero = plugin_mem_zero;
    params->stream_executor->memset = plugin_memset<uint8_t>;
    params->stream_executor->memset32 = plugin_memset<uint32_t>;

    params->stream_executor->host_callback = plugin_host_callback;
}

void plugin_destroy_stream_executor(
    const SP_Platform* platform,
    SP_StreamExecutor* stream_executor)
{
}

void plugin_destroy_platform(SP_Platform* const platform) {}
void plugin_destroy_platform_fns(SP_PlatformFns* const platform_fns) {}

extern "C"
{
    void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status)
    {
        params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
        params->platform->name = "DML";
        params->platform->type = "GPU";
        params->platform->supports_unified_memory = false;
        params->platform->force_memory_growth = true;
        params->platform->use_bfc_allocator = true;
        params->major_version = DML_MAJOR_VERSION;
        params->minor_version = DML_MINOR_VERSION;
        params->patch_version = DML_PATCH_VERSION;
        params->platform_fns->get_device_count = plugin_get_device_count;
        params->platform_fns->create_device = plugin_create_device;
        params->platform_fns->destroy_device = plugin_destroy_device;
        params->platform_fns->create_device_fns = plugin_create_device_fns;
        params->platform_fns->destroy_device_fns = plugin_destroy_device_fns;
        params->platform_fns->create_stream_executor =
            plugin_create_stream_executor;
        params->platform_fns->destroy_stream_executor =
            plugin_destroy_stream_executor;
        params->platform_fns->create_timer_fns = plugin_create_timer_fns;
        params->platform_fns->destroy_timer_fns = plugin_destroy_timer_fns;
        params->destroy_platform = plugin_destroy_platform;
        params->destroy_platform_fns = plugin_destroy_platform_fns;
    }
}
