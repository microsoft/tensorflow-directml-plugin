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

#include "absl/container/flat_hash_map.h"
#include "dml_common.h"

namespace tfdml
{

// A contiguous range of descriptors, at some offset into the specified heap.
struct D3D12DescriptorHandles
{
    ID3D12DescriptorHeap* heap;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
};

class D3D12DescriptorHeapAllocator
{
  public:
    D3D12DescriptorHeapAllocator(
        ID3D12Device* device,
        D3D12_DESCRIPTOR_HEAP_TYPE type,
        D3D12_DESCRIPTOR_HEAP_FLAGS flags,
        uint32_t device_id);

    D3D12DescriptorHandles GetDescriptorHandles(const void* ptr) const;

    void* Alloc(uint64_t size_in_descriptors);
    void Free(void* ptr, uint64_t size_in_descriptors);

  private:
    mutable std::mutex mutex_;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    const D3D12_DESCRIPTOR_HEAP_TYPE heap_type_;
    const D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags_;
    const uint32_t handle_increment_;

    // The largest allocation ID we've returned so far (or 0 if we've never done
    // so). Note that our allocation IDs start at 1 (not 0) to ensure that it
    // isn't possible for a valid allocation to have a pointer value of
    // 0x00000000.
    uint32_t current_allocation_id_ = 0;

    // A list of unused allocation IDs. This is for re-use of IDs once they get
    // freed. We only bump the max_allocation_id_ once there are no more free
    // IDs.
    std::vector<uint32_t> free_allocation_ids_;

    struct Allocation
    {
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
    };

    absl::flat_hash_map<uint32_t, Allocation> allocations_by_id_;

    // Retrieves a free allocation ID, or nullopt if no more IDs are available.
    absl::optional<uint32_t> TryReserveAllocationID();

    // Releases an allocation ID back to the pool of IDs.
    void ReleaseAllocationID(uint32_t id);

  private:
    const uint32_t device_id_;
    friend class D3D12BufferRegion;
};

} // namespace tfdml
