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

#include "allocator_retry.h"

namespace tfdml
{

static void WaitForMilliseconds(
    std::unique_lock<std::mutex>* mu,
    std::condition_variable* cv,
    int64_t ms)
{
    cv->wait_for(*mu, std::chrono::milliseconds(ms));
}

AllocatorRetry::AllocatorRetry() {}

void* AllocatorRetry::AllocateRaw(
    std::function<
        void*(size_t alignment, size_t num_bytes, bool verbose_failure)>
        alloc_func,
    int max_millis_to_wait,
    size_t alignment,
    size_t num_bytes)
{
    if (num_bytes == 0)
    {
        return nullptr;
    }
    uint64_t deadline_micros = 0;
    bool first = true;
    void* ptr = nullptr;
    while (ptr == nullptr)
    {
        ptr = alloc_func(alignment, num_bytes, false);
        if (ptr == nullptr)
        {
            uint64_t now =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
            if (first)
            {
                deadline_micros = now + max_millis_to_wait * 1000;
                first = false;
            }
            if (now < deadline_micros)
            {
                std::unique_lock<std::mutex> l(mu_);
                WaitForMilliseconds(
                    &l,
                    &memory_returned_,
                    (deadline_micros - now) / 1000);
            }
            else
            {
                return alloc_func(alignment, num_bytes, true);
            }
        }
    }
    return ptr;
}

} // namespace tfdml
