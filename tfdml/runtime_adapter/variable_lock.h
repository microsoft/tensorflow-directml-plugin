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

#include "absl/types/span.h"

struct TF_VariableInputLockHolder;

namespace tfdml
{

class OpKernelContext;

class VariableLock
{
  public:
    VariableLock(OpKernelContext* ctx);
    VariableLock(
        OpKernelContext* ctx,
        bool exclusive_lock,
        absl::Span<const int> input_indices);
    VariableLock(VariableLock&& other);
    ~VariableLock();
    void LockShared(absl::Span<const int> input_indices);
    void LockUnique(absl::Span<const int> input_indices);
    void Unlock();

  private:
    TF_VariableInputLockHolder* lock_holder_ = nullptr;
    OpKernelContext* ctx_;
};

} // namespace tfdml