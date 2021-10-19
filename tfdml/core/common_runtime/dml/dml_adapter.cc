/* Copyright (c) Microsoft Corporation.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tfdml/core/common_runtime/dml/dml_adapter.h"

#include "tfdml/core/common_runtime/dml/dml_adapter_impl.h"

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
    auto impls = EnumerateAdapterImpls();

    std::vector<DmlAdapter> adapters;
    adapters.reserve(impls.size());

    for (auto&& impl : impls)
    {
        adapters.push_back(DmlAdapter(std::move(impl)));
    }

    return adapters;
}

} // namespace tfdml