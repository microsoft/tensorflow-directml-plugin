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
bool IsBiasAdd(const tensorflow::NodeDef& node)
{
    return node.op() == "BiasAdd" || node.op() == "BiasAddV1";
}

bool IsCast(const tensorflow::NodeDef& node)
{
    return node.op() == "Cast";
}

bool IsConstant(const tensorflow::NodeDef& node)
{
    return node.op() == "Const";
}

bool IsConv2D(const tensorflow::NodeDef& node) { return node.op() == "Conv2D"; }

bool IsElu(const tensorflow::NodeDef& node) { return node.op() == "Elu"; }

bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormGrad" ||
           node.op() == "FusedBatchNormGradV2" ||
           node.op() == "FusedBatchNormGradV3";
}

bool IsLeakyRelu(const tensorflow::NodeDef& node)
{
    return node.op() == "LeakyRelu";
}

bool IsMerge(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Merge" || op == "RefMerge" || op == "_XlaMerge";
}

bool IsPlaceholder(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Placeholder" || op == "PlaceholderV2" ||
           op == "PlaceholderWithDefault";
}

bool IsNextIteration(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "NextIteration" || op == "RefNextIteration";
}

bool IsPad(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Pad" || op == "PadV2";
}

bool IsRelu(const tensorflow::NodeDef& node) { return node.op() == "Relu"; }

bool IsRelu6(const tensorflow::NodeDef& node) { return node.op() == "Relu6"; }

bool IsSymbolicGradient(const tensorflow::NodeDef& node)
{
    return node.op() == "SymbolicGradient";
}

bool IsTranspose(const tensorflow::NodeDef& node)
{
    return node.op() == "Transpose";
}

} // namespace tfdml