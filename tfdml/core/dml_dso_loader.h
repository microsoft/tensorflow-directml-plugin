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

// Common DSO loading functionality: exposes callables that dlopen DSOs
// in either the runfiles directories

#pragma once

#include "tfdml/runtime_adapter/statusor.h"

namespace tfdml
{
namespace DmlDsoLoader
{
// The following methods either load the DSO of interest and return a dlopen
// handle or error status.
StatusOr<void*> GetDirectMLDsoHandle();
StatusOr<void*> GetDirectMLDebugDsoHandle();
StatusOr<void*> GetD3d12DsoHandle();
StatusOr<void*> GetDxgiDsoHandle();
StatusOr<void*> GetDxCoreDsoHandle();
StatusOr<void*> GetPixDsoHandle();
StatusOr<void*> GetKernel32DsoHandle();
} // namespace DmlDsoLoader

// Wrapper around the DmlDsoLoader that prevents us from dlopen'ing any of the
// DSOs more than once.
namespace DmlCachedDsoLoader
{
// Cached versions of the corresponding DmlDsoLoader methods above.
StatusOr<void*> GetDirectMLDsoHandle();
StatusOr<void*> GetDirectMLDebugDsoHandle();
StatusOr<void*> GetD3d12DsoHandle();
StatusOr<void*> GetDxgiDsoHandle();
StatusOr<void*> GetDxCoreDsoHandle();
StatusOr<void*> GetPixDsoHandle();
StatusOr<void*> GetKernel32DsoHandle();
} // namespace DmlCachedDsoLoader
} // namespace tfdml
