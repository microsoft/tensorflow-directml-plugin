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

#include "tfdml/core/dml_device_manager.h"

namespace tfdml
{
/*static*/ DmlDeviceManager& DmlDeviceManager::Instance()
{
    static DmlDeviceManager device_manager;
    return device_manager;
}

Status DmlDeviceManager::InsertDevice(uint32_t device_id, DmlDevice* dml_device)
{
    if (device_id >= kNumMaxDevices)
    {
        return errors::InvalidArgument(
            "DML doesn't support more than ",
            kNumMaxDevices,
            " devices at the moment. Use the DML_VISIBLE_DEVICES environment "
            "variable to reduce the number of visible devices (e.g. "
            "DML_VISIBLE_DEVICES=\"0,1,2\" to show only the first 3 devices).");
    }

    if (device_id >= devices_.size())
    {
        devices_.resize(device_id + 1, nullptr);
    }

    devices_[device_id] = dml_device;
    return Status::OK();
}

DmlDevice* DmlDeviceManager::GetDevice(uint32_t device_id) const
{
    return devices_[device_id];
}
} // namespace tfdml