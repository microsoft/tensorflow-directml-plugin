/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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

#include "tfdml/kernels/pch.h"

namespace tfdml
{
class DmlSnapshotOp : public OpKernel
{
  public:
    explicit DmlSnapshotOp(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

    void Compute(OpKernelContext* context)
    {
        const Tensor& input = context->input(0);
        // Try to use buffer forwarding to avoid an explicit copy.
        int candidate_input_indices[] = {0};
        StatusOr<Tensor> status_or_output =
            context->forward_input_or_allocate_output(
                candidate_input_indices,
                0,
                input.shape());

        OP_REQUIRES_OK(context, status_or_output.status());
        if (!status_or_output.ValueOrDie().SharesBufferWith(input))
        {
            context->device()->CopyTensorInSameDevice(
                &input,
                &status_or_output.ValueOrDie());
        }
    }
};

void RegisterKernels_Snapshot()
{
    using K = KernelDefinition<ops::Snapshot, DmlSnapshotOp>;

    RegisterWithTypes<
        K,
        ops::Snapshot::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT64,
        TF_INT32,
        TF_UINT16,
        TF_INT16,
        TF_UINT8,
        TF_INT8>();
}

} // namespace tfdml