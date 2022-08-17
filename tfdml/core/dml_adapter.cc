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

#include "tfdml/core/dml_adapter.h"
#include "tfdml/core/dml_adapter_impl.h"

namespace tfdml
{

DmlAdapter::DmlAdapter(const DmlAdapterImpl& impl)
    : impl_(std::make_shared<DmlAdapterImpl>(impl))
{
}

DmlAdapter::~DmlAdapter() = default;

DriverVersion DmlAdapter::DriverVersion() const
{
    return impl_->DriverVersion();
}

VendorID DmlAdapter::VendorID() const { return impl_->VendorID(); }
uint32_t DmlAdapter::DeviceID() const { return impl_->DeviceID(); }
const std::string& DmlAdapter::Name() const { return impl_->Name(); }
bool DmlAdapter::IsComputeOnly() const { return impl_->IsComputeOnly(); }
const LUID& DmlAdapter::AdapterLuid() const { return impl_->AdapterLuid(); }

uint64_t DmlAdapter::GetTotalDedicatedMemory() const
{
    return impl_->GetTotalDedicatedMemory();
}

uint64_t DmlAdapter::GetTotalSharedMemory() const
{
    return impl_->GetTotalSharedMemory();
}

uint64_t DmlAdapter::QueryAvailableLocalMemory() const
{
    return impl_->QueryAvailableLocalMemory();
}

bool DmlAdapter::IsUmaAdapter() const { return impl_->IsUmaAdapter(); }

const char* GetVendorName(VendorID id)
{
    switch (id)
    {
    case VendorID::kAmd: return "AMD";
    case VendorID::kNvidia: return "NVIDIA";
    case VendorID::kMicrosoft: return "Microsoft";
    case VendorID::kQualcomm: return "Qualcomm";
    case VendorID::kIntel: return "Intel";
    default: return "Unknown";
    }
}

std::vector<DmlAdapter> EnumerateAdapters()
{
    bool allow_warp_adapters = false;
    const char* allow_warp_adapters_string =
        std::getenv("TF_DIRECTML_ALLOW_WARP_ADAPTERS");

    if (allow_warp_adapters_string != nullptr)
    {
        if (strcmp("false", allow_warp_adapters_string) == 0)
        {
            allow_warp_adapters = false;
        }
        else if (strcmp("true", allow_warp_adapters_string) == 0)
        {
            allow_warp_adapters = true;
        }
        else
        {
            TF_Log(
                TF_ERROR,
                "The TF_DIRECTML_ALLOW_WARP_ADAPTERS environment variable "
                "is "
                "set but could not be parsed: \"%s\". Valid values are "
                "\"true\" "
                "or \"false\". Using default value of \"false\".",
                allow_warp_adapters_string);
        }
    }

    auto impls = EnumerateAdapterImpls(allow_warp_adapters);

    std::vector<DmlAdapter> adapters;
    adapters.reserve(impls.size());

    for (auto&& impl : impls)
    {
        adapters.push_back(DmlAdapter(std::move(impl)));
    }

    return adapters;
}

} // namespace tfdml