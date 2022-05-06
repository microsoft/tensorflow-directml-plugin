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

#include "tfdml/core/dml_error_handling.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/c/logging.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{
namespace dml_util
{

[[noreturn]] void HandleFailedHr(
    HRESULT hr,
    const char* expression,
    const char* file,
    int line)
{
    assert(FAILED(hr));

    // Detect device removal and print a diagnostic
    switch (hr)
    {
    case DXGI_ERROR_DEVICE_REMOVED:
    case DXGI_ERROR_DEVICE_HUNG:
    case DXGI_ERROR_DEVICE_RESET:
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR: {
        absl::string_view reason = StringifyDeviceRemovedReason(hr);
        TF_Log(
            TF_ERROR,
            "The DirectML device has encountered an unrecoverable error "
            "(%.*s). "
            "This is most often caused by a timeout occurring on the GPU. "
            "Please visit https://aka.ms/tfdmltimeout for more information "
            "and troubleshooting steps.",
            reason.length(),
            reason.data());
    }
    break;
    }

    // Emit a generic error message and exit
    LogFatal(absl::StrCat(
                 "HRESULT failed with 0x",
                 absl::Hex(hr, absl::kSpacePad8),
                 ": ",
                 expression)
                 .c_str());

    // Should never get here
    assert(false);
}

bool HrIsOutOfMemory(HRESULT hr)
{
    // E_OUTOFMEMORY has a different value depending on whether _WIN32 is
    // defined when building winerror.h, so we check both potential values here
    return hr == 0x80000002 || hr == 0x8007000e;
}

absl::string_view StringifyDeviceRemovedReason(HRESULT reason)
{
    switch (reason)
    {
    case DXGI_ERROR_DEVICE_HUNG: return "DXGI_ERROR_DEVICE_HUNG";
    case DXGI_ERROR_DEVICE_REMOVED: return "DXGI_ERROR_DEVICE_REMOVED";
    case DXGI_ERROR_DEVICE_RESET: return "DXGI_ERROR_DEVICE_RESET";
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
        return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
    case DXGI_ERROR_INVALID_CALL: return "DXGI_ERROR_INVALID_CALL";
    case S_OK: return "S_OK";
    default: return "UNKNOWN";
    }
}

} // namespace dml_util
} // namespace tfdml