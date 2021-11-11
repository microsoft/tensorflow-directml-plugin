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

#include "dml_bfc_allocator.h"

#include "dml_common.h"
#include "dml_heap_allocator.h"
#include "tfdml/core/util/env_var.h"

namespace tfdml
{

DmlAllocator::DmlAllocator(
    D3D12HeapAllocator* heap_allocator,
    const std::string& name)
    : heap_allocator_(heap_allocator)
{
}

D3D12BufferRegion DmlAllocator::CreateBufferRegion(
    const void* ptr,
    uint64_t size_in_bytes)
{
    return heap_allocator_->CreateBufferRegion(ptr, size_in_bytes);
}

void* DmlAllocator::Alloc(size_t num_bytes)
{
    void* p = heap_allocator_->Alloc(num_bytes);
    return p;
}

void DmlAllocator::Free(void* ptr, size_t num_bytes)
{
    heap_allocator_->Free(ptr, num_bytes);
}

} // namespace tfdml
