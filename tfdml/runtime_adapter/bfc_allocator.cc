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

#include "bfc_allocator.h"

#include <atomic>

#include "tensorflow/c/logging.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/numbers.h"

namespace tfdml
{

constexpr BFCAllocator::ChunkHandle BFCAllocator::kInvalidChunkHandle;

BFCAllocator::BFCAllocator(
    std::unique_ptr<SubAllocator> sub_allocator,
    size_t total_memory,
    const std::string& name,
    const Options& opts)
    : opts_(opts),
      coalesce_regions_(sub_allocator->SupportsCoalescing()),
      sub_allocator_(std::move(sub_allocator)),
      name_(name),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1)
{
    if (opts.allow_growth)
    {
        // 2MiB smallest initial allocation, unless total memory available
        // is less.
        curr_region_allocation_bytes_ =
            RoundedBytes(std::min(total_memory, size_t{2 << 20}));
    }
    else
    {
        curr_region_allocation_bytes_ = RoundedBytes(total_memory);
    }

    curr_region_allocation_bytes_ =
        std::min(curr_region_allocation_bytes_, opts.max_allocation_size_bytes);

    // Allocate the requested amount of memory.
    memory_limit_ = total_memory;
    stats_.bytes_limit = static_cast<int64_t>(total_memory);

    // Create a bunch of bins of various good sizes.

    // We create bins to fit all possible ranges that cover the
    // memory_limit_ starting from allocations up to 256 bytes to
    // allocations up to (and including) the memory limit.
    TF_VLog(1, "Creating new BFCAllocator named: %s", name.c_str());
    for (BinNum b = 0; b < kNumBins; b++)
    {
        size_t bin_size = BinNumToSize(b);
        TF_VLog(
            1,
            "Creating bin of max chunk size %s",
            strings::HumanReadableNumBytes(bin_size).c_str());
        new (BinFromIndex(b)) Bin(this, bin_size);
        CHECK(BinForSize(bin_size) == BinFromIndex(b));
        CHECK(BinForSize(bin_size + 255) == BinFromIndex(b));
        CHECK(BinForSize(bin_size * 2 - 1) == BinFromIndex(b));
        if (b + 1 < kNumBins)
        {
            CHECK(BinForSize(bin_size * 2) != BinFromIndex(b));
        }
    }
}

BFCAllocator::~BFCAllocator()
{
    // Return memory back.
    TF_VLog(
        2,
        "Number of regions allocated: %llu",
        region_manager_.regions().size());
    for (const auto& region : region_manager_.regions())
    {
        sub_allocator_->Free(region.ptr(), region.memory_size());
    }

    for (BinNum b = 0; b < kNumBins; b++)
    {
        BinFromIndex(b)->~Bin();
    }
}

BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h)
{
    assert(h >= 0);
    assert(h < static_cast<int>(chunks_.size()));
    return &(chunks_[h]);
}

const BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) const
{
    assert(h >= 0);
    assert(h < static_cast<int>(chunks_.size()));
    return &(chunks_[h]);
}

bool BFCAllocator::Extend(size_t alignment, size_t rounded_bytes)
{
    size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
    // Rounds available_bytes down to the nearest multiple of
    // kMinAllocationSize.
    available_bytes =
        (available_bytes / kMinAllocationSize) * kMinAllocationSize;

    // Do we have enough space to handle the client's request?
    // If not, fail immediately.
    if (rounded_bytes > available_bytes)
    {
        return false;
    }

    // If curr_region_allocation_bytes_ is not enough to satisfy the
    // allocation, keep multiplying by a power of two until that is
    // sufficient.
    bool increased_allocation = false;
    while (rounded_bytes > curr_region_allocation_bytes_)
    {
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
    }

    curr_region_allocation_bytes_ = std::min(
        curr_region_allocation_bytes_,
        opts_.max_allocation_size_bytes);

    // Try allocating.
    size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
    size_t bytes_received;
    void* mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
    if (mem_addr == nullptr && !started_backpedal_)
    {
        // Only backpedal once.
        started_backpedal_ = true;

        static constexpr float kBackpedalFactor = 0.9;

        // Try allocating less memory.
        while (mem_addr == nullptr)
        {
            bytes = RoundedBytes(bytes * kBackpedalFactor);
            if (bytes < rounded_bytes) break;
            mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
        }
    }

    if (mem_addr == nullptr)
    {
        return false;
    }

    if (!increased_allocation)
    {
        // Increase the region size of the next required allocation.
        curr_region_allocation_bytes_ *= 2;

        curr_region_allocation_bytes_ = std::min(
            curr_region_allocation_bytes_,
            opts_.max_allocation_size_bytes);
    }

    TF_VLog(
        1,
        "Extending allocation by %s bytes.",
        strings::HumanReadableNumBytes(bytes).c_str());

    total_region_allocated_bytes_ += bytes;
    TF_VLog(
        1,
        "Total allocated bytes: %s",
        strings::HumanReadableNumBytes(total_region_allocated_bytes_).c_str());

    TF_VLog(
        1,
        "Allocated memory at %p to %p",
        mem_addr,
        static_cast<void*>(static_cast<char*>(mem_addr) + bytes));

    AllocationRegion* maybe_extended_region = nullptr;
    if (coalesce_regions_)
    {
        maybe_extended_region = region_manager_.AddOrExtendAllocationRegion(
            mem_addr,
            bytes_received);
    }
    else
    {
        region_manager_.AddAllocationRegion(mem_addr, bytes_received);
    }

    // Create one large chunk for the whole memory space that will
    // be chunked later.
    ChunkHandle h = AllocateChunk();
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    c->ptr = mem_addr;
    c->size = bytes_received;
    c->allocation_id = -1;
    c->prev = kInvalidChunkHandle;
    c->next = kInvalidChunkHandle;
    c->freed_at_count = 0;

    region_manager_.set_handle(c->ptr, h);

    // If the region was extended, then there exists a previous chunk that
    // should be linked to the new chunk.
    if (maybe_extended_region != nullptr)
    {
        ChunkHandle prev =
            maybe_extended_region->get_handle(maybe_extended_region->ptr());
        BFCAllocator::Chunk* prev_chunk = ChunkFromHandle(prev);
        // Find the last recorded chunk in the extended region.
        while (prev_chunk->next != kInvalidChunkHandle)
        {
            prev = prev_chunk->next;
            prev_chunk = ChunkFromHandle(prev);
        }
        c->prev = prev;
        prev_chunk->next = h;
    }

    // Maybe merge adjacent chunks and insert the chunk into the right bin.
    InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

    return true;
}

BFCAllocator::ChunkHandle BFCAllocator::AllocateChunk()
{
    if (free_chunks_list_ != kInvalidChunkHandle)
    {
        ChunkHandle h = free_chunks_list_;
        Chunk* c = ChunkFromHandle(h);
        free_chunks_list_ = c->next;
        return h;
    }
    else
    {
        ChunkHandle h = chunks_.size();
        chunks_.resize(h + 1);
        return h;
    }
}

void BFCAllocator::DeallocateChunk(ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    c->allocation_id = -1;
    c->bin_num = kInvalidBinNum;
    c->next = free_chunks_list_;
    free_chunks_list_ = h;
}

void* BFCAllocator::AllocateRawInternalWithRetry(
    size_t unused_alignment,
    size_t num_bytes,
    const AllocationAttributes& allocation_attr)
{
    // Fast path: Try once to allocate without getting the retry_helper_
    // involved
    uint64_t freed_by_count = 0;
    if (allocation_attr.freed_by_func != nullptr)
    {
        freed_by_count = (*allocation_attr.freed_by_func)();
    }
    void* r = AllocateRawInternal(unused_alignment, num_bytes, freed_by_count);
    if (r != nullptr)
    {
        return r;
    }
    else
    {
        static const int64_t kMaxMillisToWait = 10000; // 10 seconds
        r = retry_helper_.AllocateRaw(
            [this, &allocation_attr](size_t a, size_t nb, bool v)
            {
                uint64_t freed_by_count = 0;
                if (allocation_attr.freed_by_func != nullptr)
                {
                    freed_by_count = (*allocation_attr.freed_by_func)();
                }
                return AllocateRawInternal(a, nb, freed_by_count);
            },
            kMaxMillisToWait,
            unused_alignment,
            num_bytes);
        return r;
    }
}

void* BFCAllocator::AllocateRaw(
    size_t unused_alignment,
    size_t num_bytes,
    const AllocationAttributes& allocation_attr)
{
    TF_VLog(1, "AllocateRaw %s %llu", Name().c_str(), num_bytes);
    void* result = [&]
    {
        if (!opts_.allow_retry_on_failure || !allocation_attr.retry_on_failure)
        {
            // If we have globally disabled retry-on-failure and fail to
            // allocate an "important" alloc, we want to print a log, because
            // the program may be about to fail due to OOM.
            //
            // Bit of a hack: We deem "important" allocs as those which are
            // retryable. In TF, *non*-retryable allocations are usually those
            // which we can tolerate failing.  For example, we allocate
            // convolution scratch memory as non-retryable; if it fails, we'll
            // just use a fallback algorithm that uses no scratch.
            static std::atomic<int32_t> log_counter{0};
            constexpr int kMaxFailureLogs = 10;

            uint64_t freed_by_count = 0;
            if (allocation_attr.freed_by_func != nullptr)
            {
                freed_by_count = (*allocation_attr.freed_by_func)();
            }
            void* res = AllocateRawInternal(
                unused_alignment,
                num_bytes,
                freed_by_count);
            return res;
        }
        else
        {
            return AllocateRawInternalWithRetry(
                unused_alignment,
                num_bytes,
                allocation_attr);
        }
    }();
    TF_VLog(3, "AllocateRaw %s  %llu %p", Name().c_str(), num_bytes, result);
    return result;
}

// static
size_t BFCAllocator::RoundedBytes(size_t bytes)
{
    size_t rounded_bytes =
        (kMinAllocationSize *
         ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
    assert(size_t{0} == rounded_bytes % kMinAllocationSize);
    return rounded_bytes;
}

bool BFCAllocator::DeallocateFreeRegions(size_t rounded_bytes)
    EXCLUSIVE_LOCKS_REQUIRED(lock_)
{
    // Do nothing if garbage collection is off.
    if (!opts_.garbage_collection)
    {
        return false;
    }

    // Searching for free regions.
    absl::flat_hash_set<void*> free_region_ptrs;
    size_t total_free_bytes = 0;
    for (const AllocationRegion& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        bool any_use = false;
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            if (c->in_use())
            {
                any_use = true;
                break;
            }
            h = c->next;
        }

        if (!any_use)
        {
            TF_VLog(2, "Found free region with ptr = %p", region.ptr());
            free_region_ptrs.insert(region.ptr());
            total_free_bytes += region.memory_size();
        }
    }

    if (total_free_bytes == 0)
    {
        return false;
    }

    // Rough estimation to check whether deallocation can help.
    size_t available_bytes =
        memory_limit_ - total_region_allocated_bytes_ + total_free_bytes;
    if (rounded_bytes > available_bytes)
    {
        return false;
    }

    TF_Log(
        TF_WARNING,
        "Garbage collection: deallocate free memory regions (i.e., "
        "allocations) so that we can re-allocate a larger region to avoid OOM "
        "due to memory fragmentation. If you see this message frequently, you "
        "are running near the threshold of the available device memory and "
        "re-allocation may incur great performance overhead. You may try "
        "smaller batch sizes to observe the performance impact. Set "
        "TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to disable this "
        "feature.");

    // Deallocate free regions.
    DeallocateRegions(free_region_ptrs);

    return true;
}

void BFCAllocator::DeallocateRegions(
    const absl::flat_hash_set<void*>& region_ptrs)
    EXCLUSIVE_LOCKS_REQUIRED(lock_)
{
    // Explicitly remove the const qualifier as some compilers disallow passing
    // const_iterator to std::vector::erase(), which is used in
    // RemoveAllocationRegion().
    auto regions =
        const_cast<std::vector<AllocationRegion>*>(&region_manager_.regions());
    auto it = regions->begin();
    while (it != regions->end())
    {
        if (!region_ptrs.contains(it->ptr()))
        {
            ++it;
            continue;
        }

        TF_VLog(2, "Deallocate region with ptr = %p", it->ptr());
        // Remove all chunk registrations from Bins.
        ChunkHandle h = region_manager_.get_handle(it->ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            if (c->bin_num != kInvalidBinNum)
            {
                RemoveFreeChunkFromBin(h);
            }
            auto h_to_delete = h;
            h = c->next;
            DeleteChunk(h_to_delete);
        }

        // Deallocate the memory.
        sub_allocator_->Free(it->ptr(), it->memory_size());
        total_region_allocated_bytes_ -= it->memory_size();
        it = region_manager_.RemoveAllocationRegion(it);
    }
}

void* BFCAllocator::AllocateRawInternal(
    size_t unused_alignment,
    size_t num_bytes,
    uint64_t freed_before)
{
    if (num_bytes == 0)
    {
        TF_VLog(2, "tried to allocate 0 bytes");
        return nullptr;
    }

    if (num_bytes > opts_.max_allocation_size_bytes)
    {
        TF_VLog(
            2,
            "requested allocation of %llu exceeds maximum supported allocation "
            "size of %llu",
            num_bytes,
            opts_.max_allocation_size_bytes);
        return nullptr;
    }

    // First, always allocate memory of at least kMinAllocationSize
    // bytes, and always allocate multiples of kMinAllocationSize bytes
    // so all memory addresses are nicely byte aligned.
    size_t rounded_bytes = RoundedBytes(num_bytes);

    // The BFC allocator tries to find the best fit first.
    BinNum bin_num = BinNumForSize(rounded_bytes);

    absl::MutexLock l(&lock_);
    if (!timestamped_chunks_.empty())
    {
        // Merge timestamped chunks whose counts have become safe for general
        // use.
        MergeTimestampedChunks(0);
    }
    void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr)
    {
        return ptr;
    }

    // Try to extend
    if (Extend(unused_alignment, rounded_bytes))
    {
        ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
        if (ptr != nullptr)
        {
            return ptr;
        }
    }

    if ((freed_before == 0) && (!timestamped_chunks_.empty()))
    {
        // We're unable to satisfy an allocation request without a specific
        // timestamp requirement.  Rather than fail, try merging any held-out
        // timestamped chunks more aggressively until a free chunk of the
        // necessary size is formed.
        if (MergeTimestampedChunks(rounded_bytes))
        {
            ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
            if (ptr != nullptr)
            {
                return ptr;
            }
        }
    }

    // Reaching this point means that no chunks can satisfy the request. Also,
    // the unallocated bytes cannot satisfy the request. Before giving up, let's
    // try deallocating free regions so that suballocator can combine them with
    // the unallocated bytes and form a larger region.
    if (DeallocateFreeRegions(rounded_bytes) &&
        Extend(unused_alignment, rounded_bytes))
    {
        ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
        if (ptr != nullptr)
        {
            return ptr;
        }
    }

    return nullptr;
}

int64_t BFCAllocator::LargestFreeChunk()
{
    for (int i = kNumBins - 1; i >= 0; i--)
    {
        if (!BinFromIndex(i)->free_chunks.empty())
        {
            return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())
                ->size;
        }
    }
    return 0;
}

double BFCAllocator::GetFragmentation()
{
    int64_t bytes_available =
        total_region_allocated_bytes_ - stats_.bytes_in_use;
    assert(bytes_available > 0);
    return static_cast<double>(bytes_available - LargestFreeChunk()) /
           bytes_available;
}

void* BFCAllocator::FindChunkPtr(
    BinNum bin_num,
    size_t rounded_bytes,
    size_t num_bytes,
    uint64_t freed_before)
{
    // First identify the first bin that could satisfy rounded_bytes.
    for (; bin_num < kNumBins; bin_num++)
    {
        // Start searching from the first bin for the smallest chunk that fits
        // rounded_bytes.
        Bin* b = BinFromIndex(bin_num);
        for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
             ++citer)
        {
            const BFCAllocator::ChunkHandle h = (*citer);
            BFCAllocator::Chunk* chunk = ChunkFromHandle(h);
            assert(!chunk->in_use());
            if (freed_before > 0 && freed_before < chunk->freed_at_count)
            {
                continue;
            }
            if (chunk->size >= rounded_bytes)
            {
                // We found an existing chunk that fits us that wasn't in use,
                // so remove it from the free bin structure prior to using.
                RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

                // If we can break the size of the chunk into two reasonably
                // large pieces, do don't waste more than
                // max_internal_fragmentation_bytes on padding. If this
                // threshold is not set by the user, then use 128MB as the
                // default.
                const int64_t max_internal_fragmentation_bytes =
                    (opts_.fragmentation_fraction > 0.0)
                        ? opts_.fragmentation_fraction * memory_limit_
                        : 128 << 20;

                if (chunk->size >= rounded_bytes * 2 ||
                    static_cast<int64_t>(chunk->size) - rounded_bytes >=
                        max_internal_fragmentation_bytes)
                {
                    SplitChunk(h, rounded_bytes);
                    chunk = ChunkFromHandle(
                        h); // Update chunk pointer in case it moved
                }

                // The requested size of the returned chunk is what the user
                // has allocated.
                chunk->requested_size = num_bytes;
                // Assign a unique id and increment the id counter, marking the
                // chunk as being in use.
                chunk->allocation_id = next_allocation_id_++;

                // Update stats.
                ++stats_.num_allocs;
                stats_.bytes_in_use += chunk->size;
                if (stats_.bytes_in_use > stats_.peak_bytes_in_use)
                {
                    TF_VLog(
                        2,
                        "New Peak memory usage of %lld bytes for %s",
                        stats_.bytes_in_use,
                        Name().c_str());
                }
                stats_.peak_bytes_in_use =
                    std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
                stats_.largest_alloc_size = std::max<std::size_t>(
                    stats_.largest_alloc_size,
                    chunk->size);

                TF_VLog(4, "Returning: %p", chunk->ptr);
                TF_VLog(4, "A: %s", RenderOccupancy().c_str());
                return chunk->ptr;
            }
        }
    }

    return nullptr;
}

void BFCAllocator::SplitChunk(BFCAllocator::ChunkHandle h, size_t num_bytes)
{
    // Allocate the new chunk before we do any ChunkFromHandle
    ChunkHandle h_new_chunk = AllocateChunk();

    Chunk* c = ChunkFromHandle(h);
    CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

    // Create a new chunk starting num_bytes after c
    BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
    new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
    region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

    // Set the new sizes of the chunks.
    new_chunk->size = c->size - num_bytes;
    c->size = num_bytes;

    // The new chunk is not in use.
    new_chunk->allocation_id = -1;

    // It inherits the freed time.
    new_chunk->freed_at_count = c->freed_at_count;

    // Maintain the pointers.
    // c <-> c_neighbor becomes
    // c <-> new_chunk <-> c_neighbor
    BFCAllocator::ChunkHandle h_neighbor = c->next;
    new_chunk->prev = h;
    new_chunk->next = h_neighbor;
    c->next = h_new_chunk;
    if (h_neighbor != kInvalidChunkHandle)
    {
        Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
        c_neighbor->prev = h_new_chunk;
    }

    // Add the newly free chunk to the free bin.
    InsertFreeChunkIntoBin(h_new_chunk);
}

void BFCAllocator::DeallocateRaw(void* ptr)
{
    TF_VLog(
        1,
        "DeallocateRaw %s %llu",
        Name().c_str(),
        (ptr ? RequestedSize(ptr) : 0));
    DeallocateRawInternal(ptr);
    retry_helper_.NotifyDealloc();
}

void BFCAllocator::DeallocateRawInternal(void* ptr)
{
    if (ptr == nullptr)
    {
        TF_VLog(2, "tried to deallocate nullptr");
        return;
    }
    absl::MutexLock l(&lock_);

    // Find the chunk from the ptr.
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle);
    // Record chunk information before it's freed.
    Chunk* chunk = ChunkFromHandle(h);
    void* chunk_ptr = chunk->ptr;
    int64_t req_bytes = chunk->requested_size;
    int64_t alloc_bytes = chunk->size;

    MarkFree(h);
    InsertFreeChunkIntoBin(TryToCoalesce(h, false));

    TF_VLog(4, "F: %s", RenderOccupancy().c_str());
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void BFCAllocator::Merge(
    BFCAllocator::ChunkHandle h1,
    BFCAllocator::ChunkHandle h2)
{
    Chunk* c1 = ChunkFromHandle(h1);
    Chunk* c2 = ChunkFromHandle(h2);
    // We can only merge chunks that are not in use.
    CHECK(!c1->in_use() && !c2->in_use());

    // c1's prev doesn't change, still points to the same ptr, and is
    // still not in use.

    // Fix up neighbor pointers
    //
    // c1 <-> c2 <-> c3 should become
    // c1 <-> c3

    BFCAllocator::ChunkHandle h3 = c2->next;
    c1->next = h3;
    CHECK(c2->prev == h1);
    if (h3 != kInvalidChunkHandle)
    {
        BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
        c3->prev = h1;
    }

    // Set the new size
    c1->size += c2->size;

    // Pick latest free time.
    c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

    DeleteChunk(h2);
}

void BFCAllocator::DeleteChunk(ChunkHandle h)
{
    // Delete h and cleanup all state
    Chunk* c = ChunkFromHandle(h);
    //  VLOG(4) << "Removing: " << c->ptr;
    region_manager_.erase(c->ptr);
    DeallocateChunk(h);
}

void BFCAllocator::InsertFreeChunkIntoBin(BFCAllocator::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));
    BinNum bin_num = BinNumForSize(c->size);
    Bin* new_bin = BinFromIndex(bin_num);
    c->bin_num = bin_num;
    new_bin->free_chunks.insert(h);
}

void BFCAllocator::RemoveFreeChunkIterFromBin(
    BFCAllocator::Bin::FreeChunkSet* free_chunks,
    const BFCAllocator::Bin::FreeChunkSet::iterator& citer)
{
    ChunkHandle h = *citer;
    Chunk* c = ChunkFromHandle(h);
    CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
    free_chunks->erase(citer);
    c->bin_num = kInvalidBinNum;
}

void BFCAllocator::RemoveFreeChunkFromBin(BFCAllocator::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
    CHECK(BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0);
    c->bin_num = kInvalidBinNum;
}

void BFCAllocator::MarkFree(BFCAllocator::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

    // Mark the chunk as no longer in use.
    c->allocation_id = -1;

    // Updates the stats.
    stats_.bytes_in_use -= c->size;
}

BFCAllocator::ChunkHandle BFCAllocator::TryToCoalesce(
    ChunkHandle h,
    bool ignore_freed_at)
{
    Chunk* c = ChunkFromHandle(h);
    if ((!ignore_freed_at) && c->freed_at_count > 0) return h;
    ChunkHandle coalesced_chunk = h;

    // If the next chunk is free, merge it into c and delete it.
    if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use())
    {
        Chunk* n = ChunkFromHandle(c->next);
        if ((n->freed_at_count == 0) || ignore_freed_at)
        {
            TF_VLog(4, "Merging c->next %p with c %p", n->ptr, c->ptr);
            RemoveFreeChunkFromBin(c->next);
            Merge(h, c->next);
        }
    }

    // If the previous chunk is free, merge c into it and delete c.
    if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use())
    {
        Chunk* n = ChunkFromHandle(c->prev);
        if ((n->freed_at_count == 0) || ignore_freed_at)
        {
            TF_VLog(4, "Merging c %p into c->prev %p", c->ptr, n->ptr);
            coalesced_chunk = c->prev;
            RemoveFreeChunkFromBin(c->prev);
            Merge(c->prev, h);
        }
    }

    return coalesced_chunk;
}

void BFCAllocator::SetSafeFrontier(uint64_t count)
{
    uint64_t current = safe_frontier_.load(std::memory_order_relaxed);
    while (count > current)
    {
        if (safe_frontier_.compare_exchange_strong(current, count))
        {
            retry_helper_.NotifyDealloc();
            return;
        }
        else
        {
            current = safe_frontier_.load(std::memory_order_relaxed);
        }
    }
}

bool BFCAllocator::MergeTimestampedChunks(size_t required_bytes)
{
    TF_VLog(
        1,
        "MergeTimestampedChunks queue_len=%llu required_bytes=",
        timestamped_chunks_.size(),
        required_bytes);
    bool satisfied = (required_bytes == 0);
    std::vector<void*> to_merge;
    std::deque<ChunkHandle> new_ts_queue;
    while (!timestamped_chunks_.empty())
    {
        ChunkHandle h = timestamped_chunks_.front();
        timestamped_chunks_.pop_front();
        assert(h != kInvalidChunkHandle);
        Chunk* c = ChunkFromHandle(h);
        // It's possible this chunk has already been merged so refetch and
        // retest the handle.
        h = region_manager_.get_handle(c->ptr);
        if (h == kInvalidChunkHandle)
        {
            continue;
        }
        if (c->in_use() || (c->bin_num == kInvalidBinNum))
        {
            // This chunk has already been reallocated.
            continue;
        }
        if (c->freed_at_count == 0)
        {
            to_merge.push_back(c->ptr);
            continue;
        }
        // Chunk should be free and assigned to a bin.
        assert(c->bin_num != kInvalidBinNum);
        if (c->freed_at_count < safe_frontier_)
        {
            c->freed_at_count = 0;
            to_merge.push_back(c->ptr);
        }
        else if (required_bytes > 0)
        {
            to_merge.push_back(c->ptr);
        }
        else
        {
            new_ts_queue.push_back(h);
        }
    }
    assert(timestamped_chunks_.empty());
    std::swap(timestamped_chunks_, new_ts_queue);

    // At this point all candidate chunks have been moved from
    // timestamped_chunks_ to to_merge.  If this is a standard merge
    // (required_bytes == 0) then merge them all, otherwise merge just until a
    // Chunk of the required size is produced.
    for (int ci = 0, end = to_merge.size(); ci < end; ++ci)
    {
        void* ptr = to_merge[ci];
        // It's possible that the Chunk associated with this memory location got
        // merged and deallocated in a prior iteration so refetch the handle and
        // retest.
        ChunkHandle h = region_manager_.get_handle(ptr);
        if (h == kInvalidChunkHandle) continue;
        if (required_bytes == 0 || !satisfied)
        {
            Chunk* c = ChunkFromHandle(h);
            assert(c->bin_num != kInvalidBinNum);
            assert(!c->in_use());
            RemoveFreeChunkFromBin(h);
            ChunkHandle new_h = TryToCoalesce(h, (required_bytes > 0));
            InsertFreeChunkIntoBin(new_h);
            if (required_bytes > 0)
            {
                c = ChunkFromHandle(new_h);
                if (new_h != h && c->freed_at_count > 0)
                {
                    timestamped_chunks_.push_back(new_h);
                }
                if (c->size >= required_bytes)
                {
                    satisfied = true;
                }
            }
        }
        else
        {
            // We were force merging Chunks with unsafe timestamps, but managed
            // to create a satisfying Chunk so just requeue the rest.
            timestamped_chunks_.push_back(h);
        }
    }
    return satisfied;
}

bool BFCAllocator::TracksAllocationSizes() const { return true; }

size_t BFCAllocator::RequestedSize(const void* ptr) const
{
    CHECK(ptr);
    absl::MutexLock l(&lock_);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle);
    const BFCAllocator::Chunk* c = ChunkFromHandle(h);
    return c->requested_size;
}

size_t BFCAllocator::AllocatedSize(const void* ptr) const
{
    absl::MutexLock l(&lock_);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle);
    const BFCAllocator::Chunk* c = ChunkFromHandle(h);
    return c->size;
}

int64_t BFCAllocator::AllocationId(const void* ptr) const
{
    absl::MutexLock l(&lock_);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle);
    const BFCAllocator::Chunk* c = ChunkFromHandle(h);
    return c->allocation_id;
}

namespace
{

void RenderRegion(
    char* rendered,
    const size_t resolution,
    const size_t total_render_size,
    const size_t offset,
    const void* base_ptr,
    const void* ptr,
    const size_t size,
    const char c)
{
    const char* base_ptr_c = static_cast<const char*>(base_ptr);
    const char* ptr_c = static_cast<const char*>(ptr);

    size_t start_location =
        ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
    CHECK(start_location >= 0);
    CHECK(start_location < resolution);
    size_t end_location =
        ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) /
        total_render_size;
    CHECK(end_location >= 0);
    CHECK(end_location < resolution);

    for (size_t i = start_location; i <= end_location; ++i)
    {
        rendered[i] = c;
    }
}

} // namespace

std::string BFCAllocator::RenderOccupancy()
{
    // Make a buffer for the ASCII-art representation.
    const size_t resolution = 100;
    char rendered[resolution];

    // Compute the total region size to render over
    size_t total_region_size = 0;
    for (const auto& region : region_manager_.regions())
    {
        total_region_size += region.memory_size();
    }

    if (total_region_size == 0)
    {
        return "<allocator contains no memory>";
    }

    // Start out with everything empty
    RenderRegion(
        rendered,
        resolution,
        total_region_size,
        0,
        nullptr,
        nullptr,
        total_region_size,
        '_');

    size_t region_offset = 0;
    for (const auto& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        // Then render each chunk left to right.
        while (h != kInvalidChunkHandle)
        {
            Chunk* c = ChunkFromHandle(h);
            if (c->in_use())
            {
                // Render the wasted space
                size_t wasted = c->size - c->requested_size;
                if (wasted > 0)
                {
                    RenderRegion(
                        rendered,
                        resolution,
                        total_region_size,
                        region_offset + c->requested_size,
                        region.ptr(),
                        c->ptr,
                        wasted,
                        'x');
                }
                // Then the occupied space
                RenderRegion(
                    rendered,
                    resolution,
                    total_region_size,
                    region_offset,
                    region.ptr(),
                    c->ptr,
                    c->requested_size,
                    '*');
            }
            h = c->next;
        }
        region_offset += region.memory_size();
    }

    return std::string(rendered, resolution);
}

absl::optional<AllocatorStats> BFCAllocator::GetStats()
{
    absl::MutexLock l(&lock_);
    return stats_;
}

bool BFCAllocator::ClearStats()
{
    absl::MutexLock l(&lock_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
    return true;
}

std::array<BFCAllocator::BinDebugInfo, BFCAllocator::kNumBins> BFCAllocator::
    get_bin_debug_info()
{
    std::array<BinDebugInfo, kNumBins> bin_infos;
    for (const auto& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            BinNum bin_num = BinNumForSize(c->size);
            BinDebugInfo& bin_info = bin_infos[bin_num];
            bin_info.total_bytes_in_bin += c->size;
            bin_info.total_chunks_in_bin++;
            if (c->in_use())
            {
                bin_info.total_bytes_in_use += c->size;
                bin_info.total_requested_bytes_in_use += c->requested_size;
                bin_info.total_chunks_in_use++;
            }
            else
            {
                Bin* bin = BinFromIndex(bin_num);
                CHECK(bin->free_chunks.count(h) == 1);
                CHECK(c->bin_num == bin_num);
            }
            h = c->next;
        }
    }
    return bin_infos;
}

AllocatorMemoryType BFCAllocator::GetMemoryType() const
{
    return sub_allocator_->GetMemoryType();
}

} // namespace tfdml