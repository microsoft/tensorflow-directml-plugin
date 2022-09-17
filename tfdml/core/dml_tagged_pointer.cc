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

#include "tfdml/core/dml_tagged_pointer.h"
#include <cassert>

namespace tfdml
{
/*static*/ TaggedPointer TaggedPointer::Unpack(const void* ptr)
{
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);

    static constexpr uint64_t kAllocationIDMask =
        (1ull << kAllocationIDBits) - 1;
    static constexpr uint64_t kOffsetMask = (1ull << kOffsetBits) - 1;

    TaggedPointer tagged_ptr;
    tagged_ptr.device_id = (ptr_val >> (kAllocationIDBits + kOffsetBits));
    tagged_ptr.allocation_id = (ptr_val >> kOffsetBits) & kAllocationIDMask;
    tagged_ptr.offset = (ptr_val & kOffsetMask);

    return tagged_ptr;
}

/*static*/ void* TaggedPointer::Pack(
    uint32_t device_id,
    uint32_t allocation_id,
    uint64_t offset)
{
    assert(device_id < (1ull << kDeviceIDBits));
    assert(allocation_id < (1ull << kAllocationIDBits));
    assert(offset < (1ull << kOffsetBits));

    // Store the device ID in the upper bits of the pointer, followed by the
    // allocation id and the offset in the lower bits
    uint64_t ptr = ((uint64_t)device_id << (kAllocationIDBits + kOffsetBits)) |
                   ((uint64_t)allocation_id << kOffsetBits) | offset;

    return reinterpret_cast<void*>(ptr);
}
} // namespace tfdml
