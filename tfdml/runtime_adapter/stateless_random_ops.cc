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

#include "tfdml/runtime_adapter/philox_random.h"
#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/tensor.h"

namespace tfdml
{

Status GenerateKey(
    const Tensor& seed,
    random::PhiloxRandom::Key* out_key,
    random::PhiloxRandom::ResultType* out_counter)
{
    // Grab the two seeds
    uint64_t seed0;
    uint64_t seed1;
    if (seed.dtype() == TF_INT32)
    {
        const auto seed_vals = seed.base<int32_t>();
        seed0 = seed_vals[0];
        seed1 = seed_vals[1];
    }
    else if (seed.dtype() == TF_INT64)
    {
        const auto seed_vals = seed.base<int64_t>();
        seed0 = seed_vals[0];
        seed1 = seed_vals[1];
    }
    else
    {
        return errors::InvalidArgument(
            "Invalid seed type: ",
            DataTypeString(seed.dtype()));
    }

    // Scramble the seeds so that the user doesn't need to worry about which
    // part of the seed needs to be strong.
    (*out_key)[0] = 0x3ec8f720;
    (*out_key)[1] = 0x02461e29;
    (*out_counter)[0] = static_cast<uint32_t>(seed0);
    (*out_counter)[1] = static_cast<uint32_t>(seed0 >> 32);
    (*out_counter)[2] = static_cast<uint32_t>(seed1);
    (*out_counter)[3] = static_cast<uint32_t>(seed1 >> 32);
    const auto mix = random::PhiloxRandom(*out_counter, *out_key)();
    (*out_key)[0] = mix[0];
    (*out_key)[1] = mix[1];
    (*out_counter)[0] = (*out_counter)[1] = 0;
    (*out_counter)[2] = mix[2];
    (*out_counter)[3] = mix[3];
    return Status::OK();
}

} // namespace tfdml
