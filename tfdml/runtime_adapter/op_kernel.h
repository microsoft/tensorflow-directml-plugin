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

#include "absl/types/span.h"
#include "attribute.h"
#include "node_def.h"
#include "types.h"
#include "absl/strings/string_view.h"

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

  private:
    std::shared_ptr<const NodeDef> node_def_;
};
} // namespace tfdml
