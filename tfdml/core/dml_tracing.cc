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

#if _WIN32
#define NOMINMAX
// clang-format off
#include <Windows.h>
#include <TraceLoggingProvider.h>
#include <evntrace.h>
// clang-format on

#include "absl/time/clock.h"

#define USE_PIX
#include <d3d12.h>

#include "WinPixEventRuntime/pix3.h"
#include "tfdml/core/dml_adapter.h"
#include "tfdml/core/dml_device_cache.h"
#include "tfdml/core/dml_dso_loader.h"
#include "tfdml/runtime_adapter/env.h"
#include "third_party/microsofttelemetry.h"

#pragma comment(lib, "advapi32.lib")

typedef HRESULT(WINAPI* PIXBeginEventOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList,
    UINT64 color,
    PCSTR formatString);
typedef HRESULT(WINAPI* PIXEndEventOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList);
typedef HRESULT(WINAPI* PIXSetMarkerOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList,
    UINT64 color,
    PCSTR formatString);

static decltype(PIXGetThreadInfo)* g_pixGetThreadInfo = nullptr;
static decltype(PIXEventsReplaceBlock)* g_pixEventsReplaceBlock = nullptr;
static PIXBeginEventOnCommandListFn g_pixBeginEventOnCommandList = nullptr;
static PIXEndEventOnCommandListFn g_pixEndEventOnCommandList = nullptr;
static PIXSetMarkerOnCommandListFn g_pixSetMarkerOnCommandList = nullptr;

void BeginEventOnCommandList(
    ID3D12GraphicsCommandList* command_list,
    UINT64 color,
    PCSTR format_string)
{
    if (g_pixBeginEventOnCommandList)
    {
        g_pixBeginEventOnCommandList(command_list, color, format_string);
    }
}

void EndEventOnCommandList(ID3D12GraphicsCommandList* command_list)
{
    if (g_pixEndEventOnCommandList)
    {
        g_pixEndEventOnCommandList(command_list);
    }
}

void SetMarkerOnCommandList(
    ID3D12GraphicsCommandList* command_list,
    UINT64 color,
    PCSTR format_string)
{
    if (g_pixSetMarkerOnCommandList)
    {
        g_pixSetMarkerOnCommandList(command_list, color, format_string);
    }
}

extern "C" PIXEventsThreadInfo* PIXGetThreadInfo() noexcept
{
    if (!g_pixGetThreadInfo)
    {
        return nullptr;
    }
    return g_pixGetThreadInfo();
}

extern "C" UINT64 WINAPI PIXEventsReplaceBlock(
    PIXEventsThreadInfo* threadInfo,
    bool getEarliestTime) noexcept
{
    if (!g_pixEventsReplaceBlock)
    {
        return 0;
    }
    return g_pixEventsReplaceBlock(threadInfo, getEarliestTime);
}

#else
// No-op macros for WSL
#define TRACELOGGING_DECLARE_PROVIDER(...)
#define TRACELOGGING_DEFINE_PROVIDER(...)
#define TraceLoggingRegister(...)
#define TraceLoggingUnregister(...)
#define TraceLoggingWrite(...)
#define TraceLoggingValue(...)
#define TraceLoggingString(...)
#define TraceLoggingOpcode(...)
#define EVENT_TRACE_TYPE_START
#define EVENT_TRACE_TYPE_STOP

#define PIXBeginEvent(...)
#define PIXEndEvent(...)
#define PIXSetMarker(...)
#define PIX_COLOR(...)
#define BeginEventOnCommandList(...)
#define EndEventOnCommandList(...)
#define SetMarkerOnCommandList(...)
#endif

#include "dml_tracing.h"
#include "tfdml/core/dml_device_cache.h"
#include "tfdml/runtime_adapter/env_var.h"
#include "tfdml/runtime_adapter/status.h"

TRACELOGGING_DECLARE_PROVIDER(g_providerHandle);

#ifdef DIRECTML_ENABLE_TELEMETRY
#define TfdmlTraceLoggingOptionGroup(guid) TraceLoggingOptionGroup(guid)
#else
#define TfdmlTraceLoggingOptionGroup(guid)
#endif

// {2D181A67-62EC-40FD-8CEB-DA6891614877}
TRACELOGGING_DEFINE_PROVIDER(
    g_providerHandle,
    "Microsoft.Windows.AI.MachineLearning.Dml.TensorFlowDirectMLPlugin",
    (0x2D181A67,
     0x62EC,
     0x40FD,
     0x8C,
     0xEB,
     0xDA,
     0x68,
     0x91,
     0x61,
     0x48,
     0x77),
    TfdmlTraceLoggingOptionGroup(DIRECTML_TELEMETRY_PROVIDER_GROUP_GUID));

// {D113B493-BBA2-4993-8608-D706A73B91CE}
static constexpr GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce}};

// Overrides the default value of a tracing level using an environment variable
// (if set).
void MaybeOverrideTraceLevelFromEnvVar(
    const char* name,
    DmlTracing::TraceLevel& level)
{
    int64_t trace_level = 0;
    tfdml::Status s = tfdml::ReadInt64FromEnvVar(name, level, &trace_level);
    if (!s.ok() || (trace_level != DmlTracing::None &&
                    trace_level != DmlTracing::Standard &&
                    trace_level != DmlTracing::Verbose))
    {
        TF_Log(
            TF_WARNING,
            "The '%s' environment variable, if defined, may only have one of "
            "the following values: %d, %d, or %d.",
            name,
            DmlTracing::None,
            DmlTracing::Standard,
            DmlTracing::Verbose);
    }
    level = static_cast<DmlTracing::TraceLevel>(trace_level);
}

DmlTracing::DmlTracing()
{
    TraceLoggingRegister(g_providerHandle);

    MaybeOverrideTraceLevelFromEnvVar(
        "TF_DIRECTML_TRACE_PIX_LEVEL",
        trace_pix_level_);
    MaybeOverrideTraceLevelFromEnvVar(
        "TF_DIRECTML_TRACE_ETW_LEVEL",
        trace_etw_level_);
    MaybeOverrideTraceLevelFromEnvVar(
        "TF_DIRECTML_TRACE_PROFILER_LEVEL",
        trace_profiler_level_);

    device_events_.resize(tfdml::DmlDeviceCache::Instance().GetAdapterCount());

#if _WIN32
    if (trace_pix_level_ > TraceLevel::None)
    {
        auto pix_handle_or = tfdml::DmlCachedDsoLoader::GetPixDsoHandle();
        if (pix_handle_or.ok())
        {
            tfdml::env::GetSymbolFromLibrary(
                pix_handle_or.ValueOrDie(),
                "PIXGetThreadInfo",
                reinterpret_cast<void**>(&g_pixGetThreadInfo));
            tfdml::env::GetSymbolFromLibrary(
                pix_handle_or.ValueOrDie(),
                "PIXEventsReplaceBlock",
                reinterpret_cast<void**>(&g_pixEventsReplaceBlock));
            tfdml::env::GetSymbolFromLibrary(
                pix_handle_or.ValueOrDie(),
                "PIXBeginEventOnCommandList",
                reinterpret_cast<void**>(&g_pixBeginEventOnCommandList));
            tfdml::env::GetSymbolFromLibrary(
                pix_handle_or.ValueOrDie(),
                "PIXEndEventOnCommandList",
                reinterpret_cast<void**>(&g_pixEndEventOnCommandList));
            tfdml::env::GetSymbolFromLibrary(
                pix_handle_or.ValueOrDie(),
                "PIXSetMarkerOnCommandList",
                reinterpret_cast<void**>(&g_pixSetMarkerOnCommandList));
        }
    }
#endif // _WIN32
}

DmlTracing::~DmlTracing() { TraceLoggingUnregister(g_providerHandle); }

/*static*/ DmlTracing& DmlTracing::Instance()
{
    static DmlTracing traceLogger;
    return traceLogger;
}

void DmlTracing::StartProfiler()
{
    // The core TF runtime should not be calling start when the profiler is
    // already active.
    assert(!profiler_active_);
    profiler_active_ = true;

    // Reset previously collected events for the TF profiler.
    profiler_start_timestamp_ns_ = absl::GetCurrentTimeNanos();
    xspace_dirty_ = true;
    for (auto& device_events : device_events_)
    {
        device_events.Clear();
    }

    if (trace_etw_level_ >= TraceLevel::Standard)
    {
        TraceLoggingWrite(
            g_providerHandle,
            "ProfilerSession",
            TraceLoggingOpcode(EVENT_TRACE_TYPE_START));
    }

    if (trace_pix_level_ >= TraceLevel::Standard)
    {
        PIXBeginEvent(PIX_COLOR(255, 0, 0), "ProfilerSession");
    }

    // If attached to PIX, marks the start of a "frame" of GPU work.
    auto& device_cache = tfdml::DmlDeviceCache::Instance();
    for (uint32_t i = 0; i < device_cache.GetAdapterCount(); ++i)
    {
        const auto* state = device_cache.GetOrCreateDeviceState(i);

        if (state->sharing_contract)
        {
            state->sharing_contract->BeginCapturableWork(
                PIX_EVAL_CAPTURABLE_WORK_GUID);
        }
    }
}

void DmlTracing::StopProfiler()
{
    // Marks the end of the profiling region using both ETW (GPUView/WPA) and
    // PIX events.
    if (trace_etw_level_ >= TraceLevel::Standard)
    {
        TraceLoggingWrite(
            g_providerHandle,
            "ProfilerSession",
            TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP));
    }

    if (trace_pix_level_ >= TraceLevel::Standard)
    {
        PIXEndEvent();
    }

    // If attached to PIX, marks the end of a "frame" of GPU work.
    auto& device_cache = tfdml::DmlDeviceCache::Instance();
    for (uint32_t i = 0; i < device_cache.GetAdapterCount(); ++i)
    {
        const auto* state = device_cache.GetOrCreateDeviceState(i);

        if (state->sharing_contract)
        {
            state->sharing_contract->EndCapturableWork(
                PIX_EVAL_CAPTURABLE_WORK_GUID);
        }
    }

    profiler_active_ = false;
}

void DmlTracing::LogExecutionContextCopyBufferRegion()
{
    if (trace_etw_level_ >= TraceLevel::Verbose)
    {
        TraceLoggingWrite(g_providerHandle, "ExecutionContextCopyBufferRegion");
    }
}

void DmlTracing::LogExecutionContextFillBufferWithPattern()
{
    if (trace_etw_level_ >= TraceLevel::Verbose)
    {
        TraceLoggingWrite(
            g_providerHandle,
            "ExecutionContextFillBufferWithPattern");
    }
}

void DmlTracing::LogExecutionContextFlush()
{
    if (trace_etw_level_ >= TraceLevel::Verbose)
    {
        TraceLoggingWrite(g_providerHandle, "ExecutionContextFlush");
    }

    if (trace_pix_level_ >= TraceLevel::Verbose)
    {
        PIXSetMarker(0, "EC Flush");
    }
}

std::optional<DmlTracing::ProfilerEventId> DmlTracing::TryLogKernelComputeStart(
    uint32_t device_ordinal,
    const std::string_view op_type,
    const std::string_view op_name)
{
    std::optional<ProfilerEventId> profiler_event_id;
    if (profiler_active_ && trace_profiler_level_ >= TraceLevel::Standard)
    {
        auto timestamp = absl::GetCurrentTimeNanos();

        // Locking here is not ideal and can be avoided with TLS.
        std::unique_lock<std::mutex> lock(mutex_);
        auto& events = device_events_[device_ordinal].kernel_compute_events;
        profiler_event_id = {device_ordinal, events.size()};
        events.push_back(KernelComputeEvent{
            op_type.data(),
            op_name.data(),
            timestamp,
            timestamp});
        lock.unlock();
    }

    return profiler_event_id;
}

void DmlTracing::LogKernelComputeEnd(const ProfilerEventId& id)
{
    if (profiler_active_ && trace_profiler_level_ >= TraceLevel::Standard)
    {
        // Locking here is not ideal and can be avoided with TLS.
        std::unique_lock<std::mutex> lock(mutex_);
        auto& event =
            device_events_[id.device_id].kernel_compute_events[id.event_id];
        event.end_timestamp_ns = absl::GetCurrentTimeNanos();
        lock.unlock();
    }
}

void DmlTracing::LogExecuteOperatorStart(
    IDMLCompiledOperator* op,
    ID3D12GraphicsCommandList* command_list)
{
#if _WIN32
    if (trace_pix_level_ >= TraceLevel::Verbose)
    {
        std::vector<char> eventName(100);
        UINT data_size = (UINT)(eventName.size() * sizeof(char));
        op->GetPrivateData(kPixEventNameId, &data_size, eventName.data());
        BeginEventOnCommandList(
            command_list,
            PIX_COLOR(128, 255, 128),
            eventName.data());
    }
#endif
}

void DmlTracing::LogExecuteOperatorEnd(ID3D12GraphicsCommandList* command_list)
{
    if (trace_pix_level_ >= TraceLevel::Verbose)
    {
        EndEventOnCommandList(command_list);
    }
}

const tensorflow::profiler::XSpace& DmlTracing::GetXSpace()
{
    using namespace tensorflow::profiler;

    if (!xspace_dirty_)
    {
        return xspace_;
    }

    xspace_.Clear();

    auto& device_cache = tfdml::DmlDeviceCache::Instance();

    for (uint32_t i = 0; i < device_events_.size(); i++)
    {
        auto& device_events = device_events_[i];
        if (device_events.kernel_compute_events.empty())
        {
            continue;
        }

        auto& adapter = device_cache.GetAdapter(i);

        auto xplane = xspace_.add_planes();
        XPlaneBuilder plane(xplane);
        plane.SetId(i);

        // A few undocumented "rules" for naming the planes in our pluggable
        // profiler:
        //
        // - The plane name MUST start with /device:GPU, /device:TPU, or
        // /device:CUSTOM for the plane to be included in the trace_viewer tool.
        // See `ConvertXSpaceToTraceEvents` (xplane_to_trace_events.cc) in the
        // TF core runtime.
        // - The plane name MUST start with /device:GPU: for the plane to be
        // included in the tensorflow_stats tool. See `ConvertXSpaceToOpStats`
        // (xplane_to_op_stats.cc) in TF core runtime.
        plane.SetName(
            absl::StrCat("/device:GPU:", i, " (DirectML) - ", adapter.Name()));

        auto line = plane.GetOrCreateLine(0);
        line.SetName("Kernels (CPU Timeline)");

        plane.ForEachLine(
            [&](XLineBuilder line)
            { line.SetTimestampNs(profiler_start_timestamp_ns_); });

        for (auto& kernel_event : device_events.kernel_compute_events)
        {
            // WARNING: The pluggable profiler interface doesn't guarantee
            // events from the plugin will be reflected in all the various
            // tools. This logic may change in the future, but for now any
            // events tagged with the "tf_op" stat and named <op_name>:<op_type>
            // (e.g. "MyMatrixMultiply:MatMul") will be parsed correctly.
            auto event_name =
                absl::StrCat(kernel_event.op_name, ":", kernel_event.op_type);

            auto event_metadata = plane.GetOrCreateEventMetadata(event_name);
            event_metadata->set_display_name(kernel_event.op_type);
            auto event = line.AddEvent(*event_metadata);
            event.SetTimestampNs(kernel_event.start_timestamp_ns);
            event.SetEndTimestampNs(kernel_event.end_timestamp_ns);
            event.AddStatValue(
                *plane.GetOrCreateStatMetadata("tf_op"),
                *plane.GetOrCreateStatMetadata(event_name));
        }
    }

    xspace_dirty_ = false;
    return xspace_;
}

void DmlTracing::LogKernelComputeTelemetry(const char* kernel_name)
{
#ifdef DIRECTML_ENABLE_TELEMETRY
    TraceLoggingWrite(
        g_providerHandle,
        "KernelCompute",
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
        TraceLoggingString(kernel_name, "kernelName"));
#endif
}

void DmlTracing::LogDeviceCreationTelemetry(
    const char* adapterName,
    uint32_t vendor_id,
    uint32_t device_id,
    const LUID& adapter_luid,
    const tfdml::DriverVersion& driver_version,
    bool compute_only,
    uint32_t priority)
{
#ifdef DIRECTML_ENABLE_TELEMETRY
    TraceLoggingWrite(
        g_providerHandle,
        "DeviceCreation",
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
        TraceLoggingString(adapterName, "adapterName"),
        TraceLoggingUInt32(vendor_id, "vendorId"),
        TraceLoggingUInt32(device_id, "adapterDeviceId"),
        TraceLoggingUInt32(adapter_luid.LowPart, "adapterLuidLowPart"),
        TraceLoggingUInt32(adapter_luid.HighPart, "adapterLuidHighPart"),
        TraceLoggingUInt16(driver_version.parts.a, "driverVersionA"),
        TraceLoggingUInt16(driver_version.parts.b, "driverVersionB"),
        TraceLoggingUInt16(driver_version.parts.c, "driverVersionC"),
        TraceLoggingUInt16(driver_version.parts.d, "driverVersionD"),
        TraceLoggingBool(compute_only, "isComputeOnly"),
        TraceLoggingUInt32(priority, "priority"));
#endif
}
