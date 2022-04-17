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

#include "tfdml/layout_optimizer/generic_layout_optimizer.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/layout_optimizer/transpose_context.h"
#include "tfdml/layout_optimizer/transposer.h"
#include "tfdml/layout_optimizer/transposer_factory.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

constexpr char kAttrValue[] = "value";

static Status ExpandLayoutSensitiveOp(
    TransposeContext* context,
    TransposerFactory* transposer_factory)
{
    const int num_nodes = context->num_nodes;
    for (int i = 0; i < num_nodes; ++i)
    {
        auto* node_view = context->graph_view->GetNode(i);
        auto* node_def = node_view->node();
        if (IsLayoutSensitiveOp(*node_def))
        {
            std::shared_ptr<Transposer> transposer =
                transposer_factory->GetTransposer(*node_def);
            if (transposer == nullptr)
            {
                return errors::NotFound(absl::StrCat(
                    "DML Layout sensitive operation should have a transposer. "
                    "Node: ",
                    node_def->DebugString()));
            }
            TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
        }
    }
    return Status::OK();
}

static Status ExpandLayoutAgnosticOp(
    TransposeContext* context,
    TransposerFactory* transposer_factory)
{
    const int num_nodes = context->num_nodes;
    for (int i = 0; i < num_nodes; ++i)
    {
        auto* node_view = context->graph_view->GetNode(i);
        auto* node_def = node_view->node();
        if (IsLayoutAgnosticOp(*node_def))
        {
            const auto& transposer =
                transposer_factory->GetTransposer(*node_def);
            if (transposer == nullptr)
            {
                return errors::NotFound(absl::StrCat(
                    "DML Layout agnostic operation should have a transposer. "
                    "Node: ",
                    node_def->DebugString()));
            }
            TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
        }
    }
    return Status::OK();
}

inline bool IsCancellableDataFormatNodePair(
    const MutableNodeView& fanout_transpose,
    const MutableNodeView& fanin_transpose)
{
    if (!IsDataFormatOp(fanout_transpose) || !IsDataFormatOp(fanin_transpose))
    {
        return false;
    }

    auto src_dst_match =
        [](const MutableNodeView& src, const MutableNodeView& dst)
    {
        const auto* src_format = src.GetAttr(kAttrSrcFormat);
        if (src_format == nullptr)
        {
            return false;
        }
        const auto* dst_format = dst.GetAttr(kAttrDstFormat);
        if (dst_format == nullptr)
        {
            return false;
        }
        return src_format->s() == dst_format->s();
    };

    // If src_format node A is equal to dst_format of node B and dst_format of
    // node A is equal to src_format of node B, then they are cancellable.
    return src_dst_match(fanin_transpose, fanout_transpose) &&
           src_dst_match(fanout_transpose, fanin_transpose);
}

static bool GetValueAttrFromConstInputNode(
    const MutableNodeView& node,
    const std::function<bool(const tensorflow::NodeDef&)>& predicate,
    int index,
    tensorflow::TensorProto* tensor)
{
    if (!predicate(*node.node()))
    {
        return false;
    }
    const auto& regular_fanin = node.GetRegularFanin(index);
    auto* regular_fanin_node = regular_fanin.node_view();
    if (!IsConstant(*regular_fanin_node->node()))
    {
        return false;
    }
    const auto* value_attr = regular_fanin_node->GetAttr(kAttrValue);
    if (value_attr == nullptr ||
        value_attr->tensor().dtype() != tensorflow::DT_INT32)
    {
        return false;
    }
    *tensor = value_attr->tensor();

    return true;
}

inline bool IsCancellableConstPermTransposeNodePair(
    const MutableNodeView& fanout_transpose,
    const MutableNodeView& fanin_transpose)
{
    tensorflow::TensorProto fanout_tensor;
    if (!GetValueAttrFromConstInputNode(
            fanout_transpose,
            IsTranspose,
            1,
            &fanout_tensor))
    {
        return false;
    }
    tensorflow::TensorProto fanin_tensor;
    if (!GetValueAttrFromConstInputNode(
            fanin_transpose,
            IsTranspose,
            1,
            &fanin_tensor))
    {
        return false;
    }

    if (fanout_tensor.int_val_size() != fanin_tensor.int_val_size())
    {
        return false;
    }

    // Using dst->src to permute on src->dst will result in
    // seq(0, ..., num_elements - 1) if they are cancellable.
    for (int i = 0; i < fanout_tensor.int_val_size(); ++i)
    {
        if (fanout_tensor.int_val(fanin_tensor.int_val(i)) != i)
        {
            return false;
        }
    }
    return true;
}

inline bool IsCancellableNodePair(
    const MutableNodeView& fanout_transpose,
    const MutableNodeView& fanin_transpose)
{
    return IsCancellableConstPermTransposeNodePair(
               fanout_transpose,
               fanin_transpose) ||
           IsCancellableDataFormatNodePair(fanout_transpose, fanin_transpose);
}

static Status EraseCancellableNodes(TransposeContext* context)
{
    const int original_num_nodes = context->num_nodes;
    MutableGraphView* graph_view = context->graph_view.get();
    Mutation* mutation = graph_view->GetMutationBuilder();
    const int num_nodes = graph_view->NumNodes();

    for (int i = original_num_nodes; i < num_nodes; ++i)
    {
        auto* node = graph_view->GetNode(i);
        if (node->NumRegularFanins() < 1)
        {
            continue;
        }
        const auto& regular_fanin_0 = node->GetRegularFanin(0);
        auto* fanin_node = regular_fanin_0.node_view();
        if (fanin_node->node_index() < original_num_nodes)
        {
            continue;
        }
        if (!IsCancellableNodePair(*node, *fanin_node))
        {
            continue;
        }
        const auto& fanin_to_forward = fanin_node->GetRegularFanin(0);
        TensorId fanin_id_to_forward(
            fanin_to_forward.node_view()->GetName(),
            fanin_to_forward.index());
        for (const auto& regular_fanout : node->GetRegularFanout(0))
        {
            mutation->AddOrUpdateRegularFanin(
                regular_fanout.node_view(),
                regular_fanout.index(),
                fanin_id_to_forward);
        }
        mutation->RemoveNode(node);
        if (node->NumRegularFanins() > 1)
        {
            mutation->RemoveNode(node->GetRegularFanin(1).node_view());
        }
        mutation->RemoveNode(fanin_node);
        if (fanin_node->NumRegularFanins() > 1)
        {
            mutation->RemoveNode(fanin_node->GetRegularFanin(1).node_view());
        }
    }
    return mutation->Apply();
}

struct MutableNodeViewFormatter
{
    void operator()(std::string* out, MutableNodeView* node_view) const
    {
        absl::StrAppend(out, node_view->node()->name());
    }
};

// TODO: This is a temporary workaround for a graph pattern in Resnet models. We
// should be able to push down transpose nodes across Pad and many other ops,
// and then rely on cancellation to remove them.
//
// From: Transpose[NHWC->NCHW] -> Pad[paddings] -> Transpose[NCHW->NHWC]
// To:   Pad[Permute(paddings)]
static Status EraseCancellableNodesAroundPad(TransposeContext* context)
{
    MutableGraphView* graph_view = context->graph_view.get();
    Mutation* mutation = graph_view->GetMutationBuilder();

    absl::flat_hash_set<MutableNodeView*> cancelled_transposes;

    const int num_nodes = graph_view->NumNodes();
    for (int i = 0; i < num_nodes; ++i)
    {
        // Transpose node after Pad.
        auto* transpose_after = graph_view->GetNode(i);
        if (!IsTranspose(*transpose_after->node())) continue;

        // This transpose was already cancelled in previous loop iteration.
        if (cancelled_transposes.contains(transpose_after)) continue;

        // Pad node.
        const auto& transpose_after_fanin = transpose_after->GetRegularFanin(0);
        auto* pad = transpose_after_fanin.node_view();
        if (!IsPad(*pad->node())) continue;

        // Transpose node before Pad.
        const auto& pad_fanin_0 = pad->GetRegularFanin(0);
        auto* transpose_before = pad_fanin_0.node_view();
        if (!IsTranspose(*transpose_before->node())) continue;

        // Transpose before output used once by the Pad node.
        if (transpose_before->NumRegularFanouts() != 1) continue;

        // Transposes are cancellable.
        if (!IsCancellableConstPermTransposeNodePair(
                *transpose_after,
                *transpose_before))
            continue;

        // Paddings are known constant values.
        tensorflow::TensorProto paddings_t;
        if (!GetValueAttrFromConstInputNode(*pad, IsPad, 1, &paddings_t))
            continue;

        // Paddings value used once by the pad node only.
        const auto& pad_fanin_1 = pad->GetRegularFanin(1);
        auto* paddings = pad_fanin_1.node_view();
        if (paddings->NumRegularFanouts() != 1) continue;

        // Get permutation after the padding.
        tensorflow::TensorProto permute_t;
        if (!GetValueAttrFromConstInputNode(
                *transpose_after,
                IsTranspose,
                1,
                &permute_t))
            continue;

        // Pad output might be used multiple times by different Transpose nodes.
        // If they all have identical permutation, we can cancel all of them.
        std::vector<MutableNodeView*> pad_fanout_transposes;
        pad_fanout_transposes.emplace_back(transpose_after);

        bool pad_has_unsupported_fanout = false;
        for (auto& fanout : pad->GetRegularFanout(0))
        {
            auto* extra_transpose = fanout.node_view();
            if (extra_transpose == transpose_after) continue;

            // Check that fanout is a Transpose identical to the
            // transpose_after.
            tensorflow::TensorProto extra_permute_t;
            if (!GetValueAttrFromConstInputNode(
                    *extra_transpose,
                    IsTranspose,
                    1,
                    &extra_permute_t) ||
                extra_permute_t.tensor_content() != permute_t.tensor_content())
            {
                pad_has_unsupported_fanout = true;
                break;
            }

            pad_fanout_transposes.emplace_back(extra_transpose);
        }
        if (pad_has_unsupported_fanout) continue;

        TF_VLog(
            0,
            "Cancel Transpose nodes around Pad: transpose_before=%s pad=%s "
            "transpose_after=%s",
            transpose_before->node()->name().c_str(),
            pad->node()->name().c_str(),
            absl::StrJoin(
                pad_fanout_transposes,
                ",",
                MutableNodeViewFormatter()));

        // Permute paddings in place according to permutation in second
        // transpose.
        auto permutation_s = absl::Span<const int32_t>(
            reinterpret_cast<const int32_t*>(permute_t.tensor_content().data()),
            permute_t.int_val_size());
        auto paddings_s = absl::Span<int32_t>(
            reinterpret_cast<int32_t*>(
                paddings_t.mutable_tensor_content()->data()),
            paddings_t.int_val_size());
        TF_RETURN_IF_ERROR(PermuteDouble(
            absl::StrCat("paddings in ", pad->GetName()),
            permutation_s,
            &paddings_s));

        // Update paddings constant value with a permuted tensor.
        tensorflow::AttrValue permuted_paddings_tensor;
        *permuted_paddings_tensor.mutable_tensor()->mutable_tensor_content() =
            paddings_t.tensor_content();
        mutation->AddOrUpdateNodeAttr(
            paddings,
            "value",
            permuted_paddings_tensor);

        // Transform Transpose nodes into Identity nodes.
        const auto transpose_to_identity =
            [&cancelled_transposes,
             &mutation](MutableNodeView* transpose) -> void
        {
            mutation->UpdateNodeOp(transpose, "Identity");
            mutation->RemoveNodeAttr(transpose, "Tperm");
            mutation->RemoveRegularFanin(transpose, 1);
            cancelled_transposes.insert(transpose);
        };

        transpose_to_identity(transpose_before);
        absl::c_for_each(pad_fanout_transposes, transpose_to_identity);
    }

    return mutation->Apply();
}

static Status EraseOutputShapeAttrs(TransposeContext* context)
{
    MutableGraphView* graph_view = context->graph_view.get();
    Mutation* mutation = graph_view->GetMutationBuilder();
    const int num_nodes = graph_view->NumNodes();
    for (int i = 0; i < num_nodes; ++i)
    {
        auto* node = graph_view->GetNode(i);
        if (IsArg(*node->node()))
        {
            continue;
        }
        mutation->RemoveNodeAttr(node, kAttrOutputShape);
        TF_RETURN_IF_ERROR(mutation->Apply());
    }
    return Status::OK();
}

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status GenericLayoutOptimizer::Optimize(
    const GrapplerItem& item,
    tensorflow::GraphDef* output)
{
    const bool is_aggressive = true;

    TransposeContext context;

    TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
        /*assume_valid_feeds=*/is_aggressive,
        item,
        &context));

    context.AssignDeviceAndDataFormats("GPU", "NHWC", "NCHW");

    TransposerFactory transposer_factory;
    TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
    if (context.graph.node_size() > context.num_nodes || is_aggressive)
    {
        TF_RETURN_IF_ERROR(
            ExpandLayoutAgnosticOp(&context, &transposer_factory));
        TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
        TF_RETURN_IF_ERROR(EraseCancellableNodesAroundPad(&context));
        TF_RETURN_IF_ERROR(
            context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));
    }
    TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

    *output = context.graph;
    return Status::OK();
}
} // namespace tfdml
