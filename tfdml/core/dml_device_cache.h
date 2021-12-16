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

#pragma once

#include <mutex>
#include <unordered_map>

#include "tfdml/core/dml_adapter.h"
#include "tfdml/core/dml_device_state.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

// Maintains a static cache of device singletons, one per adapter. This class is
// thread-safe.
class DmlDeviceCache
{
  public:
    static DmlDeviceCache& Instance();
    uint32_t GetAdapterCount() const;

    // It is a little odd that we require GPUOptions and memory_limit here, as
    // those can vary per TF device instance - they're not process-global. We
    // handle this by using the options and memory limit that are provided to
    // the first device created on this adapter. If subsequent devices are
    // created on the same adapter but with different options/memory_limit, they
    // are ignored. This is unusual, but matches the behavior of the CUDA
    // device.
    const DmlDeviceState* GetOrCreateDeviceState(uint32_t adapter_index);
    const DmlAdapter& GetAdapter(uint32_t adapter_index) const;

  private:
    DmlDeviceCache();

    mutable std::mutex mutex_;
    std::vector<DmlAdapter> adapters_;
    std::vector<std::unique_ptr<DmlDeviceState>> device_states_;
};

} // namespace tfdml