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

#include "tfdml/runtime_adapter/variable_lock.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tfdml/runtime_adapter/device.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "tfdml/runtime_adapter/stream.h"

namespace tfdml
{

VariableLock::VariableLock(OpKernelContext* ctx) : ctx_(ctx) {}

VariableLock::VariableLock(
    OpKernelContext* ctx,
    bool exclusive_lock,
    absl::Span<const int> input_indices)
    : ctx_(ctx)
{
    if (exclusive_lock)
    {
        LockUnique(input_indices);
    }
    else
    {
        LockShared(input_indices);
    }
}

VariableLock::VariableLock(VariableLock&& other)
    : lock_holder_(other.lock_holder_),
      ctx_(other.ctx_)
{
    other.lock_holder_ = nullptr;
}

VariableLock::~VariableLock() { Unlock(); }

// This copy will only be done for sparse tensors
static void CopyTensorInSameDevice(
    TF_OpKernelContext* ctx,
    TF_Tensor* source,
    TF_Tensor* dest)
{
    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    Tensor input_tensor(source);
    Tensor output_tensor(dest);
    device->CopyTensorInSameDevice(&input_tensor, &output_tensor);
}

void VariableLock::LockShared(absl::Span<const int> input_indices)
{
    constexpr bool unique_lock = false;
    constexpr bool sparse = false;
    Status status;

    TF_MaybeLockVariableInputMutexesInOrder(
        ctx_->raw(),
        unique_lock,
        sparse,
        input_indices.data(),
        input_indices.size(),
        CopyTensorInSameDevice,
        &lock_holder_,
        status.raw());
}
void VariableLock::LockUnique(absl::Span<const int> input_indices)
{
    constexpr bool unique_lock = true;
    constexpr bool sparse = false;
    Status status;

    TF_MaybeLockVariableInputMutexesInOrder(
        ctx_->raw(),
        unique_lock,
        sparse,
        input_indices.data(),
        input_indices.size(),
        CopyTensorInSameDevice,
        &lock_holder_,
        status.raw());
}
void VariableLock::Unlock()
{
    if (lock_holder_)
    {
        TF_ReleaseVariableInputLockHolder(lock_holder_);
        lock_holder_ = nullptr;
    }
}

} // namespace tfdml