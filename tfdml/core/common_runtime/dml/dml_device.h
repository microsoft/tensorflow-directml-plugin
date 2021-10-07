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

#include "tfdml/core/util/device.h"

class IDMLDevice;
class ID3D12Device;

namespace tfdml {

class Tensor;
class DmlAllocator;
class DmlDescriptorAllocator;
class DmlKernelManager;
class DmlExecutionContext;
class DmlUploadHeap;
class DmlReadbackHeap;
class DmlEventQueue;
class DMLDeviceContext;
struct DmlDeviceState;

class DmlDevice : public Device {
 public:
  DmlDevice(const DmlDeviceState* state);

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

  void CopyTensorInSameDevice(const Tensor* input_tensor,
                              Tensor* output_tensor);

  Status CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                               Tensor* device_tensor) final;

  // TODO: Make them override if/when we implement it as part of TensorFlow's
  // Device class. Check if we can hook them to the proflier API.
  void DebugOnSessionRunStart();
  void DebugOnSessionRunEnd();

 private:
  const DmlDeviceState* state_;  // Weak; owned by the device factory
  std::unique_ptr<DMLDeviceContext> device_context_;
};

}  // namespace tfdml