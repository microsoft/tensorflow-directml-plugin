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

#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "attribute.h"
#include "node_def.h"
#include "tfdml/runtime_adapter/device.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "types.h"

namespace tfdml
{
class OpKernel
{
  public:
    OpKernel(std::shared_ptr<const NodeDef> node_def)
        : node_def_(std::move(node_def))
    {
    }

    virtual ~OpKernel() = default;

    std::shared_ptr<const NodeDef> node_def() const { return node_def_; }

    const absl::string_view type_string() const
    {
        return node_def_->GetOpTypeName();
    }
    const absl::string_view name() const { return node_def_->GetOpName(); }

    MemoryType input_memory_type(int index) const
    {
        return node_def_->GetInputTensorMemoryType(index);
    }

    MemoryType output_memory_type(int index) const
    {
        return node_def_->GetOutputTensorMemoryType(index);
    }

    void Compute(OpKernelContext* ctx)
    {
        auto profiler_event_id = ctx->device()->TryLogKernelComputeStart(
            ctx->op_kernel().type_string(),
            ctx->op_kernel().name());

        auto profiler_cleanup = absl::MakeCleanup(
            [ctx, &profiler_event_id]
            {
                if (profiler_event_id)
                {
                    ctx->device()->LogKernelComputeEnd(*profiler_event_id);
                }
            });

        ComputeImpl(ctx);
    }

  private:
    virtual void ComputeImpl(OpKernelContext* raw_ctx) = 0;

    std::shared_ptr<const NodeDef> node_def_;
};
} // namespace tfdml
