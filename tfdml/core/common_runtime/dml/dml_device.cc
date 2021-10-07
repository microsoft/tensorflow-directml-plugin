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

#include "dml_device.h"

#include "dml_adapter_impl.h"
#include "dml_bfc_allocator.h"
#include "dml_common.h"
#include "dml_device_context.h"
#include "dml_device_state.h"
#include "dml_event_queue.h"
#include "dml_kernel_manager.h"
#include "dml_readback_heap.h"
#include "dml_tracing.h"
#include "dml_upload_heap.h"
#include "tfdml/core/common_runtime/dml/dml_util.h"
#include "tfdml/core/util/allocator.h"
#include "tfdml/core/util/tensor.h"

// {D113B493-BBA2-4993-8608-D706A73B91CE}
static const GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce}};

namespace tfdml {

DmlDevice::DmlDevice(const DmlDeviceState* state) : state_(state) {
  device_context_ = absl::make_unique<DMLDeviceContext>(
      state_->execution_context.get(), state_->event_queue.get(),
      state_->upload_heap.get(), state_->readback_heap.get(),
      state_->dml_allocator.get(), state_->descriptor_allocator.get());
}

Status DmlDevice::Sync() {
  TF_VLog(2, "DirectML device: performing GPU sync.");

  auto start_time = std::chrono::high_resolution_clock::now();

  auto status_or_event = state_->execution_context->Flush();
  TF_RETURN_IF_ERROR(status_or_event.status());
  status_or_event.ConsumeValueOrDie().WaitForSignal();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> wait_seconds = end_time - start_time;
  TF_VLog(2, "DirectML device: GPU sync took %lf ms.",
          wait_seconds.count() * 1e3);

  // Take the opportunity to free some memory if needed
  state_->kernel_manager->ReleaseCompletedReferences();
  return Status::OK();
}

void DmlDevice::DebugOnSessionRunStart() {
  DmlTracing::Instance().LogSessionRunStart();
  if (state_->sharing_contract) {
    state_->sharing_contract->BeginCapturableWork(
        PIX_EVAL_CAPTURABLE_WORK_GUID);
  }
}

void DmlDevice::DebugOnSessionRunEnd() {
  DmlTracing::Instance().LogSessionRunEnd();
  if (state_->sharing_contract) {
    state_->sharing_contract->EndCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
  }
}

Status DmlDevice::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                        Tensor* device_tensor) {
  return device_context_->CopyCPUTensorToDevice(this, cpu_tensor,
                                                device_tensor);
}

void DmlDevice::CopyTensorInSameDevice(const Tensor* input_tensor,
                                       Tensor* output_tensor) {
  // Forward to the device context where the real implementation lives
  device_context_->CopyTensorInSameDevice(this, input_tensor, output_tensor);
}

ID3D12Device* DmlDevice::GetD3D12Device() const {
  return state_->d3d_device.Get();
}

IDMLDevice* DmlDevice::GetDmlDevice() const { return state_->dml_device.Get(); }

DmlAllocator* DmlDevice::GetAllocator() const {
  return state_->dml_allocator.get();
}

DmlDescriptorAllocator* DmlDevice::GetDescriptorAllocator() const {
  return state_->descriptor_allocator.get();
}

DmlKernelManager* DmlDevice::GetKernelManager() const {
  return state_->kernel_manager.get();
}

DmlExecutionContext* DmlDevice::GetExecutionContext() const {
  return state_->execution_context.get();
}

DmlUploadHeap* DmlDevice::GetUploadHeap() const {
  return state_->upload_heap.get();
}

DmlReadbackHeap* DmlDevice::GetReadbackHeap() const {
  return state_->readback_heap.get();
}

DmlEventQueue* DmlDevice::GetEventQueue() const {
  return state_->event_queue.get();
}

DMLDeviceContext* DmlDevice::GetDeviceContext() const {
  return device_context_.get();
}

}  // namespace tfdml