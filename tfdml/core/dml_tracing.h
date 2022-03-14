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

#pragma once

#include "dml_common.h"
#include "tfdml/core/dml_adapter.h"

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

    // Pair of IDs for the device and event within that device.
    struct ProfilerEventId
    {
        size_t device_id;
        size_t event_id;
    };

    // RAII helper to track a DML kernel compute call on the CPU timeline.
    class KernelComputeEventScope
    {
        // This event will be null if the TF profiler isn't active when the
        // scope is constructed.
        std::optional<ProfilerEventId> device_event_id;

      public:
        KernelComputeEventScope(
            uint32_t device_id,
            const std::string_view type,
            const std::string_view name)
        {
            device_event_id = DmlTracing::Instance().TryLogKernelComputeStart(
                device_id,
                type,
                name);
        }

        ~KernelComputeEventScope()
        {
            if (device_event_id)
            {
                DmlTracing::Instance().LogKernelComputeEnd(*device_event_id);
            }
        }
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

    // Data collected for the TF profiler.
    struct DeviceEvents
    {
        std::vector<KernelComputeEvent> kernel_compute_events;

        inline void Clear() { kernel_compute_events.clear(); }
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
    std::optional<ProfilerEventId> TryLogKernelComputeStart(
        uint32_t device_ordinal,
        const std::string_view op_type,
        const std::string_view op_name);
    void LogKernelComputeEnd(const ProfilerEventId& id);

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
        const char* adapterName,
        uint32_t vendor_id,
        uint32_t device_id,
        const tfdml::DriverVersion& driver_version,
        bool compute_only,
        uint32_t priority);
};