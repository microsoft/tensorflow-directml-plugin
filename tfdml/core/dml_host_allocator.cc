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

#include "tfdml/core/dml_host_allocator.h"
#include "absl/synchronization/mutex.h"

namespace tfdml
{
void* DmlHostAllocator::Alloc(uint64_t size)
{
#if _WIN32
    void* memory = _aligned_malloc(size, 64);
#else
    void* memory = aligned_alloc(64, size);
#endif

    absl::MutexLock lock(&mutex_);

    uint64_t allocation_id =
        available_ids_.empty() ? ++max_allocation_id_ : available_ids_.back();

    allocations_by_id_[allocation_id] = memory;

    if (!available_ids_.empty())
    {
        available_ids_.pop_back();
    }

    // The MSB is a bit representing whether the allocation is on the host
    // or the GPU, and the 63 remaining bytes are the id of the allocation.
    uint64_t ptr = ((uint64_t)1 << 63) | max_allocation_id_;
    return reinterpret_cast<void*>(ptr);
}

void* DmlHostAllocator::GetMemory(const void* ptr) const
{
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    assert(ptr_val & ((uint64_t)1 << 63));
    uint64_t allocation_id = ptr_val & (((uint64_t)1 << 63) - 1);

    absl::ReaderMutexLock lock(&mutex_);
    return allocations_by_id_.at(allocation_id);
}

void DmlHostAllocator::Free(const void* ptr)
{
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    assert(ptr_val & ((uint64_t)1 << 63));

    uint64_t allocation_id = ptr_val & (((uint64_t)1 << 63) - 1);

    auto it = allocations_by_id_.find(allocation_id);
    void* memory = nullptr;

    {
        absl::MutexLock lock(&mutex_);
        available_ids_.push_back(allocation_id);
        memory = it->second;
        allocations_by_id_.erase(it);
    }

#if _WIN32
    _aligned_free(memory);
#else
    free(memory);
#endif
}
} // namespace tfdml