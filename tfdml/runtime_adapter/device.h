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

#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
class Tensor;

class Device
{
  public:
    virtual ~Device() = default;

    virtual Status CopyCPUTensorToDevice(
        const Tensor* cpu_tensor,
        Tensor* device_tensor) = 0;

    virtual Status CopyDeviceTensorToCPU(
        const Tensor* device_tensor,
        Tensor* cpu_tensor) = 0;

    virtual void CopyTensorInSameDevice(
        const Tensor* input_tensor,
        Tensor* output_tensor) = 0;
};
} // namespace tfdml
