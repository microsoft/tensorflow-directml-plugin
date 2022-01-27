/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/runtime_adapter/training_op_helpers.h"
#include "tfdml/runtime_adapter/device.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"

namespace tfdml
{
// This is for use with ResourceVariables to ensure *tensor has a
// reference count of 1 before you update it.
// REQUIRES: If you pass in variable->tensor(), *variable->mu() must be held.
Status PrepareToUpdateVariable(
    OpKernelContext* ctx,
    Tensor* tensor,
    bool copy_on_read_mode)
{
    if (copy_on_read_mode)
    {
        // Tensor's buffer is in use by some read, so we need to copy before
        // updating.
        Tensor tmp;
        if (tensor->dtype() == TF_VARIANT)
        {
            LogFatal("TF_VARIANT is not supported yet.");
        }
        else
        {
            constexpr bool on_host = false;
            TF_RETURN_IF_ERROR(ctx->allocate_temp(
                tensor->dtype(),
                tensor->shape(),
                &tmp,
                on_host));

            ctx->device()->CopyTensorInSameDevice(tensor, &tmp);
        }
        *tensor = tmp;
    }
    return Status::OK();
}
} // namespace tfdml