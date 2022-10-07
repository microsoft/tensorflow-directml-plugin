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

#include "tfdml/optimizer/cast_remover.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/graph_properties.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/tensor_proto_util.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status CastRemover::Optimize(
    const GrapplerItem& item,
    tensorflow::GraphDef* output)
{
    *output = item.graph;
    Status status;
    auto graph_view = absl::make_unique<MutableGraphView>(output, &status);
    TF_RETURN_IF_ERROR(status);

    Mutation* mutation = graph_view->GetMutationBuilder();
    const auto& nodes_to_preserve = item.NodesToPreserve();

    for (int i = 0; i < graph_view->NumNodes(); ++i)
    {
        MutableNodeView* node = graph_view->GetNode(i);

/*
        const std::string& device_name = node->node()->device();
        std::string device;
        std::string task;
        DeviceNameUtils::SplitDeviceName(device_name, &task, &device);

        printf(
            "*************Op (%s): %s (%s)\n",
            node->node()->op().c_str(),
            node->node()->name().c_str(),
            device.c_str());
*/

        if (!DeviceNameUtils::IsOnDml(*node->node()))
        {
            continue;
        }

        if (!IsCast(*node->node()))
        {
            continue;
        }

        if (nodes_to_preserve.find(node->GetName()) != nodes_to_preserve.end())
        {
            continue;
        }

        auto* attrs = node->node()->mutable_attr();

        auto src_type_iter = attrs->find("SrcT");
        if (src_type_iter == attrs->end() || !src_type_iter->second.has_type())
        {
            continue;
        }

        auto dst_type_iter = attrs->find("DstT");
        if (dst_type_iter == attrs->end() || !dst_type_iter->second.has_type())
        {
            continue;
        }

        // Only remove the Cast nodes that have the same src and dst types
        if (src_type_iter->second.type() != dst_type_iter->second.type())
        {
            continue;
        }

        // Get the node that feeds into Cast
        TensorId fanin_tensor_id(
            node->GetRegularFanin(0).node_view()->GetName(),
            node->GetRegularFanin(0).index());

        // Get all the nodes that Cast feeds into
        auto fanouts = node->GetRegularFanout(0);

        for (auto& fanout : fanouts)
        {
            // Link all the outputs of Cast to its input
            mutation->AddOrUpdateRegularFanin(
                fanout.node_view(),
                fanout.index(),
                fanin_tensor_id);
        }
    }

    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
}
} // namespace tfdml
