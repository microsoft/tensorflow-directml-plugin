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

#include "tfdml/runtime_adapter/device.h"

class IDMLDevice;
class ID3D12Device;

namespace tfdml
{

class Tensor;
class DmlAdapter;
class DmlAllocator;
class DmlDescriptorAllocator;
class DmlKernelManager;
class DmlExecutionContext;
class DmlUploadHeap;
class DmlReadbackHeap;
class DmlEventQueue;
class DMLDeviceContext;
struct DmlDeviceState;

class DmlDevice : public Device
{
  public:
    DmlDevice(const DmlDeviceState* state, uint32_t device_ordinal);

    ID3D12Device* GetD3D12Device() const;
    IDMLDevice* GetDmlDevice() const;
    DmlAllocator* GetAllocator() const;
    DmlDescriptorAllocator* GetDescriptorAllocator() const;
    DmlKernelManager* GetKernelManager() const;
    DmlExecutionContext* GetExecutionContext() const;
    DmlUploadHeap* GetUploadHeap() const;
    DmlReadbackHeap* GetReadbackHeap() const;
    DmlEventQueue* GetEventQueue() const;
    DMLDeviceContext* GetDeviceContext() const;
    Status Sync();
    inline uint32_t GetDeviceOrdinal() const { return device_ordinal_; }

    void CopyTensorInSameDevice(
        const Tensor* input_tensor,
        Tensor* output_tensor) final;

    Status CopyCPUTensorToDevice(
        const Tensor* cpu_tensor,
        Tensor* device_tensor) final;

    Status CopyDeviceTensorToCPU(
        const Tensor* device_tensor,
        Tensor* cpu_tensor) final;

  private:
    const DmlDeviceState* state_; // Weak; owned by the device factory
    std::unique_ptr<DMLDeviceContext> device_context_;
    uint32_t device_ordinal_;
};

} // namespace tfdml