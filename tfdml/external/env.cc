/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "env.h"

#if _WIN32
#include <Windows.h>

#include "tfdml/external/wide_char.h"
#else
#include <dlfcn.h>
#endif

namespace tfdml
{
namespace env
{

std::string FormatLibraryFileName(
    const std::string& name,
    const std::string& version)
{
    std::string filename;
#if _WIN32
    if (version.empty())
    {
        filename = name + ".dll";
    }
    else
    {
        filename = name + version + ".dll";
    }
#else
    if (version.empty())
    {
        filename = "lib" + name + ".so";
    }
    else
    {
        filename = "lib" + name + ".so" + "." + version;
    }
#endif

    return filename;
}

Status GetSymbolFromLibrary(
    void* handle,
    const char* symbol_name,
    void** symbol)
{
#if _WIN32
    FARPROC found_symbol;

    found_symbol = GetProcAddress((HMODULE)handle, symbol_name);
    if (found_symbol == NULL)
    {
        return errors::NotFound(std::string(symbol_name) + " not found");
    }
    *symbol = (void**)found_symbol;
#else
    *symbol = dlsym(handle, symbol_name);
    if (!*symbol)
    {
        return errors::NotFound(dlerror());
    }
#endif
    return Status::OK();
}

Status LoadDynamicLibrary(const char* library_filename, void** handle)
{
#if _WIN32
    std::string file_name = library_filename;
    std::replace(file_name.begin(), file_name.end(), '/', '\\');

    std::wstring ws_file_name(Utf8ToWideChar(file_name));

    HMODULE hModule = LoadLibraryExW(
        ws_file_name.c_str(),
        NULL,
        LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!hModule)
    {
        return errors::NotFound(file_name + " not found");
    }
    *handle = hModule;
#else
    *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
    if (!*handle)
    {
        return errors::NotFound(dlerror());
    }
#endif
    return Status::OK();
}

} // namespace env

} // namespace tfdml
