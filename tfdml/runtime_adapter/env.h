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

#pragma once

#include <string>

#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
namespace env
{
std::string FormatLibraryFileName(
    const std::string& name,
    const std::string& version);

Status GetSymbolFromLibrary(
    void* handle,
    const char* symbol_name,
    void** symbol);

Status LoadDynamicLibrary(const char* library_filename, void** handle);
} // namespace env
} // namespace tfdml
