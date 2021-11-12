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

#include "tfdml/runtime_adapter/resource_mgr.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
class Tensor;

class Device
{
  public:
    Device();
    virtual ~Device();

    virtual Status CopyCPUTensorToDevice(
        const Tensor* cpu_tensor,
        Tensor* device_tensor) = 0;

    // Returns the resource manager associated w/ this device.
    virtual ResourceMgr* resource_manager() { return rmgr_; }

  private:
    ResourceMgr* rmgr_;
};
} // namespace tfdml
