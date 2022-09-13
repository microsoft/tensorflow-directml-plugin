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
#include <stdlib.h>

#include "DirectMLConfig.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/logging.h"
#include "tfdml/runtime_adapter/env.h"
#include "tfdml/runtime_adapter/path.h"
#include "tfdml/runtime_adapter/statusor.h"

#if _WIN32
#include <Windows.h>
#include <pathcch.h>

#include "tfdml/runtime_adapter/wide_char.h"

#ifdef _DEBUG
// #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
    #define DBG_NEW new
    // Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
    // allocations to be of _CLIENT_BLOCK type
#else
    #define DBG_NEW new
#endif

#pragma comment(lib, "Pathcch.lib") // For PathCchRemoveFileSpec
#endif

namespace tfdml
{

namespace
{
std::string GetDirectMLPath()
{
    const char* path = getenv("TF_DIRECTML_PATH");
    return (path != nullptr ? path : "");
}

#if _WIN32
std::string GetModuleDirectory()
{
    HMODULE tensorflowHmodule = nullptr;
    BOOL getHandleResult = GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT |
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        reinterpret_cast<LPCWSTR>(&GetModuleDirectory),
        &tensorflowHmodule);
    CHECK(getHandleResult == TRUE);

    // Safe const_cast because of explicit bounds checking and contiguous memory
    // in C++11 and later.
    std::wstring wpath(MAX_PATH, '\0');
    DWORD filePathSize = GetModuleFileNameW(
        tensorflowHmodule,
        const_cast<wchar_t*>(wpath.data()),
        static_cast<DWORD>(wpath.size()));

    // Stop searching if the path is 2^16 characters long to avoid allocating an
    // absurd amount of memory. Where DID you install python?
    while ((GetLastError() == ERROR_INSUFFICIENT_BUFFER) &&
           (wpath.size() < 65536))
    {
        wpath.resize(wpath.size() * 2);
        filePathSize = GetModuleFileNameW(
            tensorflowHmodule,
            const_cast<wchar_t*>(wpath.data()),
            static_cast<DWORD>(wpath.size()));
    }
    CHECK(filePathSize != 0);

    // Strip TF library filename from the path and truncate the buffer.
    // PathCchRemoveFileSpec may return S_FALSE if nothing was removed, but
    // this indicates an error (module path should be a filename, not a dir).
    CHECK(
        PathCchRemoveFileSpec(
            const_cast<wchar_t*>(wpath.data()),
            wpath.size()) == S_OK);
    wpath.resize(wcslen(wpath.c_str()));

    return WideCharToUtf8(wpath);
}
#endif

StatusOr<void*> GetDsoHandle(
    const std::string& name,
    const std::string& version,
    const std::string& search_path = "")
{
    auto filename = env::FormatLibraryFileName(name, version);
    if (!search_path.empty())
    {
        filename = JoinPath(search_path, filename);
    }

    void* dso_handle = nullptr;
    Status status = env::LoadDynamicLibrary(filename.c_str(), &dso_handle);

    if (status.ok())
    {
        TF_Log(
            TF_INFO,
            "Successfully opened dynamic library %s",
            filename.c_str());
        return dso_handle;
    }

    auto message = absl::StrCat(
        "Could not load dynamic library '",
        filename,
        "'; dlerror: ",
        status.error_message());
#if !defined(PLATFORM_WINDOWS)
    if (const char* ld_library_path = getenv("LD_LIBRARY_PATH"))
    {
        message += absl::StrCat("; LD_LIBRARY_PATH: ", ld_library_path);
    }
#endif
    TF_Log(TF_WARNING, message.c_str());
    return Status(TF_FAILED_PRECONDITION, message);
}
} // namespace

namespace DmlDsoLoader
{

StatusOr<void*> GetDirectMLLibraryHandle(const std::string& basename)
{
    auto path = GetDirectMLPath();

    // Bundled DirectML libraries have a mangled name to avoid collision:
    //
    // Original Name  | Mangled Name
    // ---------------|-------------
    // directml.dll   | directml.<sha>.dll
    // libdirectml.so | libdirectml.<sha>.so
    //
    // We use the original name if TF_DIRECTML_PATH is set.
    // We use the mangled name if TF_DIRECTML_PATH isn't set (most cases).
    std::string name = basename;
    if (path.empty())
    {
        name += std::string(".") + DIRECTML_SOURCE_VERSION;

        // Look for DML under the same directory as the core tensorflow module.
        // This check isn't required for WSL since the RPATH of the tensorflow
        // .so file is modified.
#if _WIN32
        path = GetModuleDirectory();

        // The DirectML DLL can't be located next to the pluggable device
        // library because TensorFlow would try to load it, so we place it in a
        // "directml" folder that we append to the search path
        path = JoinPath(path, "directml");
#endif
    }

    return GetDsoHandle(name, "", path);
}

StatusOr<void*> GetDirectMLDsoHandle()
{
    return GetDirectMLLibraryHandle("directml");
}

StatusOr<void*> GetDirectMLDebugDsoHandle()
{
    return GetDirectMLLibraryHandle("directml.debug");
}

StatusOr<void*> GetD3d12DsoHandle() { return GetDsoHandle("d3d12", ""); }

StatusOr<void*> GetDxgiDsoHandle()
{
#if _WIN32
    return GetDsoHandle("dxgi", "");
#else
    return Status(TF_UNIMPLEMENTED, "DXGI is not supported in WSL");
#endif
}

StatusOr<void*> GetDxCoreDsoHandle() { return GetDsoHandle("dxcore", ""); }

StatusOr<void*> GetPixDsoHandle()
{
#if _WIN32
    // The WinPixEventRuntime DLL can't be located next to the pluggable device
    // library because TensorFlow would try to load it, so we place it in a
    // "directml" folder that we append to the search path
    auto path = GetModuleDirectory();
    path = JoinPath(path, "directml");
    return GetDsoHandle("WinPixEventRuntime", "", path);
#else
    return Status(TF_UNIMPLEMENTED, "PIX events are not supported in WSL");
#endif
}

StatusOr<void*> GetKernel32DsoHandle()
{
#if _WIN32
    return GetDsoHandle("Kernel32", "");
#else
    return Status(
        TF_UNIMPLEMENTED,
        "Kernel32.dll is only available on Windows");
#endif
}

} // namespace DmlDsoLoader

namespace DmlCachedDsoLoader
{
StatusOr<void*> GetDirectMLDsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetDirectMLDsoHandle());
    return *result;
}

StatusOr<void*> GetDirectMLDebugDsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetDirectMLDebugDsoHandle());
    return *result;
}

StatusOr<void*> GetD3d12DsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetD3d12DsoHandle());
    return *result;
}

StatusOr<void*> GetDxgiDsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetDxgiDsoHandle());
    return *result;
}

StatusOr<void*> GetDxCoreDsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetDxCoreDsoHandle());
    return *result;
}

StatusOr<void*> GetPixDsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetPixDsoHandle());
    return *result;
}

StatusOr<void*> GetKernel32DsoHandle()
{
    static auto result = DBG_NEW auto(DmlDsoLoader::GetKernel32DsoHandle());
    return *result;
}

} // namespace DmlCachedDsoLoader
} // namespace tfdml
