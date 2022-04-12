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

#pragma once

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/graph_properties.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

class GrapplerItem;
class MutableGraphView;

// TransposeContext owns all data members. Must initialize GraphProperties,
// FrameView, GraphDef and MutableGraphView with the same graph. NodeDef
// pointers in FrameView, GraphDef and MutableGraphView must point to nodes in
// the same GraphDef instance.
struct TransposeContext
{
    // Initializes TransposeContext with given GrapplerItem. Because
    // initializing FrameMap and GraphProperties may return error, we initialize
    // TransposeContext outside constructor.
    static Status InitializeTransposeContext(
        bool assume_valid_feeds,
        const GrapplerItem& item,
        TransposeContext* context);

    static Status InitializeTransposeContext(
        const GrapplerItem& item,
        TransposeContext* context);

    // Sets data formats to convert from and to for specified device type.
    void AssignDeviceAndDataFormats(
        absl::string_view target_device,
        absl::string_view src_format,
        absl::string_view dst_format);

    tensorflow::GraphDef graph;
    // Number of nodes in the original graph. As new nodes are appended to the
    // end of the graph, all new nodes should have a node index greater than or
    // equal to this.
    int num_nodes;
    absl::flat_hash_set<std::string> nodes_to_preserve;
    std::unique_ptr<GraphProperties> graph_properties;
    std::unique_ptr<MutableGraphView> graph_view;

    std::string target_device;
    std::string src_format;
    std::string dst_format;
    absl::flat_hash_map<char, int> src_dim_indices;
    absl::flat_hash_map<char, int> dst_dim_indices;
    std::vector<int> src_to_dst;
    std::vector<int> dst_to_src;

    std::string enforced_layout;
};
} // namespace tfdml
