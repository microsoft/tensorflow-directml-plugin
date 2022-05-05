/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/optimizer/op_types.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tfdml/optimizer/graph_view.h"

constexpr char kAttrT[] = "T";

namespace tfdml
{
bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormGrad" ||
           node.op() == "FusedBatchNormGradV2" ||
           node.op() == "FusedBatchNormGradV3";
}

bool IsConv2D(const tensorflow::NodeDef& node) { return node.op() == "Conv2D"; }

bool IsMerge(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Merge" || op == "RefMerge" || op == "_XlaMerge";
}
bool IsPad(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Pad" || op == "PadV2";
}
bool IsTranspose(const tensorflow::NodeDef& node)
{
    return node.op() == "Transpose";
}

bool IsConstant(const tensorflow::NodeDef& node)
{
    return node.op() == "Const";
}

bool IsNextIteration(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "NextIteration" || op == "RefNextIteration";
}

} // namespace tfdml