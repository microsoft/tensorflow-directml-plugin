/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "allocator.h"

#include <atomic>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace tfdml
{

std::string AllocatorStats::DebugString() const
{
    return absl::StrFormat(
        "Limit:        %20lld\n"
        "InUse:        %20lld\n"
        "MaxInUse:     %20lld\n"
        "NumAllocs:    %20lld\n"
        "MaxAllocSize: %20lld\n",
        this->bytes_limit ? *this->bytes_limit : 0,
        this->bytes_in_use,
        this->peak_bytes_in_use,
        this->num_allocs,
        this->largest_alloc_size);
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {}

std::string AllocatorAttributes::DebugString() const
{
    return absl::StrCat(
        "AllocatorAttributes(on_host=",
        on_host(),
        " nic_compatible=",
        nic_compatible(),
        " gpu_compatible=",
        gpu_compatible(),
        ")");
}

SubAllocator::SubAllocator(
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors),
      free_visitors_(free_visitors)
{
}

void SubAllocator::VisitAlloc(void* ptr, int index, size_t num_bytes)
{
    for (const auto& v : alloc_visitors_)
    {
        v(ptr, index, num_bytes);
    }
}

void SubAllocator::VisitFree(void* ptr, int index, size_t num_bytes)
{
    // Although we don't guarantee any order of visitor application, strive
    // to apply free visitors in reverse order of alloc visitors.
    for (int i = free_visitors_.size() - 1; i >= 0; --i)
    {
        free_visitors_[i](ptr, index, num_bytes);
    }
}
} // namespace tfdml
