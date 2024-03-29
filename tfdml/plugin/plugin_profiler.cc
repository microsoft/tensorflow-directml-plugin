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

#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tfdml/core/dml_tracing.h"
#include "tfdml/runtime_adapter/macros.h"

void profiler_start(const TP_Profiler* profiler, TF_Status* status)
{
    DmlTracing::Instance().StartProfiler();
}

void profiler_stop(const TP_Profiler* profiler, TF_Status* status)
{
    DmlTracing::Instance().StopProfiler();
}

void profiler_collect_data_xspace(
    const TP_Profiler* profiler,
    uint8_t* buffer,
    size_t* size_in_bytes,
    TF_Status* status)
{
    auto& xspace = DmlTracing::Instance().GetXSpace();

    // The first call to this function is to query the XSpace size in bytes, so
    // the buffer pointer will be null.
    if (buffer == nullptr)
    {
        *size_in_bytes = xspace.ByteSizeLong();
        TF_SetStatus(status, TF_OK, "");
        return;
    }

    // The second call to this function occurs after an appropriately sized
    // buffer is reserved for the XSpace.
    bool success = xspace.SerializeToArray(buffer, *size_in_bytes);
    if (success)
    {
        TF_SetStatus(status, TF_OK, "");
    }
    else
    {
        TF_SetStatus(
            status,
            TF_FAILED_PRECONDITION,
            "Failed to serialize XSpace into buffer.");
    }
}

void profiler_destroy_profiler(TP_Profiler* profiler)
{
    // Nothing to do here
}

void profiler_destroy_profiler_fns(TP_ProfilerFns* profiler_fns)
{
    // Nothing to do here
}

TFDML_EXPORT void TF_InitProfiler(
    TF_ProfilerRegistrationParams* params,
    TF_Status* status)
{
    params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
    params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;

    params->profiler->device_type = "GPU";

    params->profiler_fns->start = profiler_start;
    params->profiler_fns->stop = profiler_stop;
    params->profiler_fns->collect_data_xspace = profiler_collect_data_xspace;
    params->destroy_profiler = profiler_destroy_profiler;
    params->destroy_profiler_fns = profiler_destroy_profiler_fns;
}