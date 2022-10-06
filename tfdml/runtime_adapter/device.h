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

#include "absl/types/optional.h"
#include "absl/types/span.h"
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

    virtual Status CopyDeviceTensorsToCPU(
        absl::Span<const Tensor> device_tensors,
        absl::Span<Tensor> cpu_tensors) = 0;

    virtual void CopyTensorInSameDevice(
        const Tensor* input_tensor,
        Tensor* output_tensor) = 0;

    virtual absl::optional<uint32_t> TryLogKernelComputeStart(
        const absl::string_view type,
        const absl::string_view name) const = 0;

    virtual void LogKernelComputeEnd(uint32_t event_id) const = 0;
};
} // namespace tfdml
