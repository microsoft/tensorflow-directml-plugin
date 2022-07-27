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

#include "tfdml/optimizer/device_placement_logger.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/graph_properties.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/utils.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status DevicePlacementLogger::Optimize(
    const GrapplerItem& item,
    tensorflow::GraphDef* output)
{
    *output = item.graph;
    Status status;
    auto graph_view = absl::make_unique<MutableGraphView>(output, &status);
    TF_RETURN_IF_ERROR(status);

    Mutation* mutation = graph_view->GetMutationBuilder();

    for (int i = 0; i < graph_view->NumNodes(); ++i)
    {
        MutableNodeView* node = graph_view->GetNode(i);

#ifdef DIRECTML_ENABLE_TELEMETRY
        if (IsOnDml(*node->node()))
        {
            DmlTracing::Instance().LogGraphDevicePlacement(
                node->node()->op().c_str(),
                "GPU");
        }
        else if (IsOnCpu(*node->node()))
        {
            DmlTracing::Instance().LogGraphDevicePlacement(
                node->node()->op().c_str(),
                "CPU");
        }
#endif
    }

    return Status::OK();
}
} // namespace tfdml
