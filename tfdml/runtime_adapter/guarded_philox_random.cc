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

#include "tfdml/runtime_adapter/guarded_philox_random.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/op_kernel_construction.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

static std::mt19937_64* InitRngWithRandomSeed()
{
    std::random_device device("/dev/urandom");
    return new std::mt19937_64(device());
}

static uint64_t New64()
{
    static std::mt19937_64* rng = InitRngWithRandomSeed();
    static std::mutex mu;
    std::unique_lock<std::mutex> l(mu);
    return (*rng)();
}

Status GuardedPhiloxRandom::Init(OpKernelConstruction* context)
{
    CHECK(!initialized_);
    // Grab seed Attrs.
    int64_t seed, seed2;
    auto status = context->GetAttr("seed", &seed);
    if (!status.ok()) return status;
    status = context->GetAttr("seed2", &seed2);
    if (!status.ok()) return status;

    // Initialize with the given seeds
    if (seed == 0 && seed2 == 0)
    {
        // If both seeds are unspecified, use completely random seeds.
        seed = New64();
        seed2 = New64();
    }
    std::unique_lock<std::mutex> lock(mu_);
    generator_ = random::PhiloxRandom(seed, seed2);
    initialized_ = true;
    return Status::OK();
}

random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64_t samples)
{
    CHECK(initialized_);
    std::unique_lock<std::mutex> lock(mu_);
    auto local = generator_;
    generator_.Skip(samples);
    return local;
}

} // namespace tfdml
