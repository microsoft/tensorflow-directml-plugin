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

#include "absl/container/inlined_vector.h"
#include "tfdml/core/dml_device.h"
#include "tfdml/core/dml_tagged_pointer.h"

namespace tfdml
{
class DmlDeviceManager
{
  public:
    static DmlDeviceManager& Instance();
    Status InsertDevice(uint32_t device_id, DmlDevice* dml_device);
    DmlDevice* GetDevice(uint32_t device_id) const;

  private:
    static constexpr uint64_t kNumMaxDevices = TaggedPointer::kDeviceIDBits
                                               << 1;

    DmlDeviceManager() = default;
    absl::InlinedVector<DmlDevice*, kNumMaxDevices> devices_;
};
} // namespace tfdml