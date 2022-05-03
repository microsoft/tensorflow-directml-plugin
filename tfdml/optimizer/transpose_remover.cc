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

#include "tfdml/optimizer/transpose_remover.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/graph_properties.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/node_utils.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/perm_utils.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

static bool IsOnDml(const tensorflow::NodeDef& node_def)
{
    const std::string& device_name = node_def.device();
    std::string device;
    std::string task;
    return DeviceNameUtils::SplitDeviceName(device_name, &task, &device) &&
           absl::StrContains(
               absl::AsciiStrToLower(device),
               absl::AsciiStrToLower("GPU"));
}

static bool IsLayoutTranspose(
    const MutableNodeView* transpose_node,
    absl::Span<const int> expected_perm)
{
    if (!IsTranspose(*transpose_node->node()))
    {
        return false;
    }

    const auto& perm_fanin = transpose_node->GetRegularFanin(1);
    auto* perm_node = perm_fanin.node_view();

    if (!IsConstant(*perm_node->node()))
    {
        return false;
    }

    auto* const_attrs = perm_node->node()->mutable_attr();
    auto perm_iter = const_attrs->find("value");

    if (perm_iter == const_attrs->end() || !perm_iter->second.has_tensor())
    {
        return false;
    }

    auto perm_tensor = perm_iter->second.tensor();

    if (!perm_tensor.has_tensor_shape() ||
        perm_tensor.tensor_shape().dim_size() != 1 ||
        perm_tensor.tensor_shape().dim(0).size() != expected_perm.size())
    {
        return false;
    }

    switch (perm_tensor.dtype())
    {
    case ::tensorflow::DT_INT32: {
        if (perm_tensor.tensor_content().size() !=
            expected_perm.size() * sizeof(int32_t))
        {
            return false;
        }

        const auto* perm_values = reinterpret_cast<const int32_t*>(
            perm_tensor.tensor_content().c_str());

        for (int i = 0; i < expected_perm.size(); ++i)
        {
            if (perm_values[i] != expected_perm[i])
            {
                return false;
            }
        }
        break;
    }
    case ::tensorflow::DT_INT64: {
        if (perm_tensor.tensor_content().size() !=
            expected_perm.size() * sizeof(int64_t))
        {
            return false;
        }

        const auto* perm_values = reinterpret_cast<const int64_t*>(
            perm_tensor.tensor_content().c_str());

        for (int i = 0; i < expected_perm.size(); ++i)
        {
            if (perm_values[i] != expected_perm[i])
            {
                return false;
            }
        }
        break;
    }
    default:
        // Other datatypes are not supported for perm in Transpose
        return false;
    }

    return true;
}

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status TransposeRemover::Optimize(
    const GrapplerItem& item,
    tensorflow::GraphDef* output)
{
    *output = item.graph;
    Status status;
    auto graph_view = absl::make_unique<MutableGraphView>(output, &status);
    TF_RETURN_IF_ERROR(status);

    Mutation* mutation = graph_view->GetMutationBuilder();

    const auto& nodes_to_preserve = item.NodesToPreserve();

    auto src_dim_indices = GetDimensionIndices("NHWC");
    auto src_to_dst = GetPermutation(src_dim_indices, "NCHW");

    auto dst_dim_indices = GetDimensionIndices("NCHW");
    auto dst_to_src = GetPermutation(dst_dim_indices, "NHWC");

    for (int i = 0; i < graph_view->NumNodes(); ++i)
    {
        MutableNodeView* fused_batch_norm_grad_node = graph_view->GetNode(i);

        if (!IsOnDml(*fused_batch_norm_grad_node->node()))
        {
            continue;
        }

        if (!IsFusedBatchNormGrad(*fused_batch_norm_grad_node->node()))
        {
            continue;
        }

        auto* fused_batch_norm_grad_attrs =
            fused_batch_norm_grad_node->node()->mutable_attr();
        auto data_format_iter =
            fused_batch_norm_grad_attrs->find("data_format");

        if (data_format_iter == fused_batch_norm_grad_attrs->end() ||
            !data_format_iter->second.has_s() ||
            data_format_iter->second.s() != "NHWC")
        {
            continue;
        }

        if (fused_batch_norm_grad_node->GetRegularFanins().size() < 2)
        {
            continue;
        }

        const auto& fanin_0 = fused_batch_norm_grad_node->GetRegularFanin(0);
        auto* transpose_node_0 = fanin_0.node_view();

        const auto& fanin_1 = fused_batch_norm_grad_node->GetRegularFanin(1);
        auto* transpose_node_1 = fanin_1.node_view();

        // FusedBatchNormGrad has 2 inputs that could have been transposed
        if (!IsLayoutTranspose(transpose_node_0, dst_to_src) ||
            !IsLayoutTranspose(transpose_node_1, dst_to_src))
        {
            continue;
        }

        if (fused_batch_norm_grad_node->GetRegularFanouts().size() < 1)
        {
            continue;
        }

        const auto& fanouts = fused_batch_norm_grad_node->GetRegularFanout(0);

        bool can_convert_outputs = true;

        for (auto& fanout : fanouts)
        {
            auto* output_transpose_node = fanout.node_view();

            if (!IsLayoutTranspose(output_transpose_node, src_to_dst))
            {
                can_convert_outputs = false;
                break;
            }
        }

        if (!can_convert_outputs)
        {
            continue;
        }

        bool should_preserve_output = false;

        // Remove the fanout transposes and connect their outputs to
        // FusedBatchNormGrad
        for (auto& fanout : fanouts)
        {
            auto* output_transpose_node = fanout.node_view();

            if (nodes_to_preserve.find(output_transpose_node->GetName()) !=
                nodes_to_preserve.end())
            {
                should_preserve_output = true;
                break;
            }

            auto& tenspose_fanouts = output_transpose_node->GetRegularFanout(0);
            for (auto& transpose_fanout : tenspose_fanouts)
            {
                TensorId fanout_tensor_id(
                    fused_batch_norm_grad_node->GetName(),
                    0);

                mutation->AddOrUpdateRegularFanin(
                    transpose_fanout.node_view(),
                    transpose_fanout.index(),
                    fanout_tensor_id);
            }
        }

        // If one of the outputs cannot be removed, we do nothing
        if (should_preserve_output)
        {
            continue;
        }

        // Remove the first transpose node and connect its input to
        // FusedBatchNormGrad
        TensorId fanin_0_tensor_id(
            transpose_node_0->GetRegularFanin(0).node_view()->GetName(),
            transpose_node_0->GetRegularFanin(0).index());

        mutation->AddOrUpdateRegularFanin(
            fused_batch_norm_grad_node,
            0,
            fanin_0_tensor_id);

        // Remove the second transpose node and connect its input to
        // FusedBatchNormGrad
        TensorId fanin_1_tensor_id(
            transpose_node_1->GetRegularFanin(0).node_view()->GetName(),
            transpose_node_1->GetRegularFanin(0).index());

        mutation->AddOrUpdateRegularFanin(
            fused_batch_norm_grad_node,
            1,
            fanin_1_tensor_id);

        // Finally, change the format of FusedBatchNormGrad
        tensorflow::AttrValue nchw_data_format;
        nchw_data_format.set_s("NCHW");
        mutation->AddOrUpdateNodeAttr(
            fused_batch_norm_grad_node,
            "data_format",
            std::move(nchw_data_format));
    }

    TF_RETURN_IF_ERROR(mutation->Apply());

    return Status::OK();
}
} // namespace tfdml
