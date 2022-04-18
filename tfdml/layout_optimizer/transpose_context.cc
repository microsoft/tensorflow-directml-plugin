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

#include "tfdml/layout_optimizer/transpose_context.h"
#include "absl/container/flat_hash_map.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/perm_utils.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

Status TransposeContext::InitializeTransposeContext(
    bool assume_valid_feeds,
    const GrapplerItem& item,
    TransposeContext* context)
{
    assert(context != nullptr);
    context->graph_properties = absl::make_unique<GraphProperties>(item);
    TF_RETURN_IF_ERROR(
        context->graph_properties->InferStatically(assume_valid_feeds));
    TF_RETURN_IF_ERROR(
        context->graph_properties->AnnotateOutputShapes(&context->graph));
    Status status;
    context->graph_view =
        absl::make_unique<MutableGraphView>(&context->graph, &status);
    TF_RETURN_IF_ERROR(status);
    context->num_nodes = context->graph.node_size();
    const auto& nodes_to_preserve = item.NodesToPreserve();
    context->nodes_to_preserve = absl::flat_hash_set<std::string>(
        nodes_to_preserve.begin(),
        nodes_to_preserve.end());
    return Status::OK();
}

Status TransposeContext::InitializeTransposeContext(
    const GrapplerItem& item,
    TransposeContext* context)
{
    return InitializeTransposeContext(false, item, context);
}

// Sets data formats to convert from and to for specified device type.
void TransposeContext::AssignDeviceAndDataFormats(
    absl::string_view target_device,
    absl::string_view src_format,
    absl::string_view dst_format)
{
    this->target_device = std::string(target_device);
    this->src_format = std::string(src_format);
    this->dst_format = std::string(dst_format);
    this->src_dim_indices = GetDimensionIndices(src_format);
    this->dst_dim_indices = GetDimensionIndices(dst_format);
    this->src_to_dst = GetPermutation(this->src_dim_indices, dst_format);
    this->dst_to_src = GetPermutation(this->dst_dim_indices, src_format);
}
} // namespace tfdml