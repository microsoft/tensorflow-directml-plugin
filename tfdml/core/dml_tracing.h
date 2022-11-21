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

#pragma once

#include "dml_common.h"
#include "tfdml/core/dml_adapter.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tfdml/runtime_adapter/xplane_builder.h"

// DirectML tracer that emits the following types of events:
// - ETW: for analysis in GPUView and WPA
// - PIX: for analysis in PIX timing and GPU captures
// - TF Profiler: for analysis in TensorBoard
class DmlTracing
{
  public:
    // GUID to associate a PIX event name with an IDMLObject (there is an
    // existing SetName method, but no public GetName method).
    constexpr static GUID kPixEventNameId = {
        0x191d7d5,
        0x9fe1,
        0x47cc,
        {0xb6, 0x46, 0xf6, 0x78, 0xb6, 0x33, 0x2f, 0x96},
    };

    enum TraceLevel
    {
        None = 0,
        Standard = 1,
        Verbose = 2,
    };

    enum class MemcpyType
    {
        D2H,
        H2D,
        D2D
    };

    // RAII helper to track a memcpy call on the CPU timeline.
    class MemcpyEventScope
    {
      public:
        MemcpyEventScope(
            uint32_t device_id,
            MemcpyType memory_type,
            uint64_t total_bytes)
            : device_id_(device_id)
        {
            device_event_id_ = DmlTracing::Instance().TryLogMemcpyStart(
                device_id,
                memory_type,
                total_bytes);
        }

        ~MemcpyEventScope()
        {
            if (device_event_id_)
            {
                DmlTracing::Instance().LogMemcpyEnd(
                    device_id_,
                    *device_event_id_);
            }
        }

      private:
        // This event will be null if the TF profiler isn't active when the
        // scope is constructed.
        absl::optional<uint32_t> device_event_id_;
        uint32_t device_id_;
    };

  private:
    DmlTracing();
    ~DmlTracing();

    // The trace_profiler_level_ only applies to events that occur while the TF
    // profiler is active (within a TF profiler session), so the default value
    // is higher than for PIX/ETW. PIX/ETW events may be emitted outside of
    // profiler sessions and require environment variables to be set to collect
    // data.
    TraceLevel trace_pix_level_ = None;
    TraceLevel trace_etw_level_ = None;
    TraceLevel trace_profiler_level_ = Standard;

    // Tracks a DML kernel compute call on the CPU timeline.
    struct KernelComputeEvent
    {
        std::string op_type;
        std::string op_name;
        int64_t start_timestamp_ns;
        int64_t end_timestamp_ns;
    };

    struct MemcpyEvent
    {
        MemcpyType memcpy_type;
        uint64_t size;
        int64_t start_timestamp_ns;
        int64_t end_timestamp_ns;
    };

    // Data collected for the TF profiler.
    struct DeviceEvents
    {
        std::vector<KernelComputeEvent> kernel_compute_events;
        std::vector<MemcpyEvent> memcpy_events;

        inline void Clear()
        {
            kernel_compute_events.clear();
            memcpy_events.clear();
        }
    };
    std::vector<DeviceEvents> device_events_;
    tensorflow::profiler::XSpace xspace_;
    bool xspace_dirty_ = true;
    int64_t profiler_start_timestamp_ns_ = 0;
    std::mutex mutex_;
    bool profiler_active_ = false;

  public:
    static DmlTracing& Instance();

    void StartProfiler();
    void StopProfiler();

    // CPU timeline
    void LogExecutionContextCopyBufferRegion();
    void LogExecutionContextFillBufferWithPattern();
    void LogExecutionContextFlush();

    absl::optional<uint32_t> TryLogKernelComputeStart(
        uint32_t device_ordinal,
        const absl::string_view op_type,
        const absl::string_view op_name);
    void LogKernelComputeEnd(uint32_t device_id, uint32_t event_id);

    absl::optional<uint32_t> TryLogMemcpyStart(
        uint32_t device_ordinal,
        MemcpyType memcpy_type,
        uint64_t data_size);
    void LogMemcpyEnd(uint32_t device_id, uint32_t event_id);

    // GPU timeline
    void LogExecuteOperatorStart(
        IDMLCompiledOperator* op,
        ID3D12GraphicsCommandList* command_list);
    void LogExecuteOperatorEnd(ID3D12GraphicsCommandList* command_list);

    // Lazily converts internal events into a profiler XSpace. Repeated calls
    // to this function will return the same XSpace.
    const tensorflow::profiler::XSpace& GetXSpace();

    void LogKernelComputeTelemetry(const char* kernel_name);

    void LogDeviceCreationTelemetry(
        const LUID& adapter_luid,
        uint32_t priority);
};