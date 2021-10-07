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

#include "tfdml/core/util/resource_mgr.h"
#include "tfdml/core/util/status.h"

namespace tfdml {
class Tensor;

class Device {
 public:
  Device();
  virtual ~Device();

  virtual Status CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                       Tensor* device_tensor) = 0;

  // Returns the resource manager associated w/ this device.
  virtual ResourceMgr* resource_manager() { return rmgr_; }

 private:
  ResourceMgr* rmgr_;
};
}  // namespace tfdml
