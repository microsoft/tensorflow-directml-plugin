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

#pragma once

#include <climits>
#include <cstdint>

namespace tfdml
{

// D3D12HeapAllocator and D3D12DescriptorHeapAllocator encode the allocation ID
// into the high bits of the pointers it returns, while the low bits are used as
// an offset into the allocation. Note that since the layout of bitfields is
// implementation-defined, you can't just cast a void* into a TaggedPointer: it
// must be done using masks and shifts.
struct TaggedPointer
{
    static constexpr uint64_t kDeviceIDBits = 4;
    static constexpr uint64_t kAllocationIDBits = 20;
    static constexpr uint64_t kOffsetBits = 40;

    uint64_t device_id : kDeviceIDBits;
    uint64_t allocation_id : kAllocationIDBits;
    uint64_t offset : kOffsetBits;

    static void* Pack(
        uint32_t device_id,
        uint32_t allocation_id,
        uint64_t offset);
    static TaggedPointer Unpack(const void* ptr);
};

static_assert(
    sizeof(TaggedPointer) == sizeof(void*),
    "DML requires a 64-bit architecture");
static_assert(
    TaggedPointer::kDeviceIDBits + TaggedPointer::kAllocationIDBits +
            TaggedPointer::kOffsetBits ==
        sizeof(void*) * CHAR_BIT,
    "DML requires a 64-bit architecture");

} // namespace tfdml
