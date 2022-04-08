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

#include "tfdml/optimizer/generic_layout_optimizer.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/transpose_context.h"
#include "tfdml/optimizer/transposer.h"
#include "tfdml/optimizer/transposer_factory.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

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
                    "Layout sensitive operation should have a transposer. "
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
                    "Layout agnostic operation should have a transposer. "
                    "Node: ",
                    node_def->DebugString()));
            }
            TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
        }
    }
    return Status::OK();
}

static Status EraseCancellableNodes(TransposeContext* context)
{
    // TODO: Implement Me
    return Status::OK();
}

// TODO: This is a temporary workaround for a graph pattern in Resnet models. We
// should be able to push down transpose nodes across Pad and many other ops,
// and then rely on cancellation to remove them.
//
// From: Transpose[NHWC->NCHW] -> Pad[paddings] -> Transpose[NCHW->NHWC]
// To:   Pad[Permute(paddings)]
static Status EraseCancellableNodesAroundPad(TransposeContext* context)
{
    // TODO: Implement me
    return Status::OK();
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

    context.AssignDeviceAndDataFormats("DML", "NHWC", "NCHW");

    TransposerFactory transposer_factory;
    TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
    if (context.graph.node_size() > context.num_nodes || is_aggressive)
    {
        TF_RETURN_IF_ERROR(
            ExpandLayoutAgnosticOp(&context, &transposer_factory));
        TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
        TF_RETURN_IF_ERROR(EraseCancellableNodesAroundPad(&context));
        // TODO(lyandy): Remove sorting once other optimizers are migrated to
        // using `utils::GraphView`.
        TF_RETURN_IF_ERROR(
            context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));
    }
    TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

    *output = context.graph;
    return Status::OK();
}
} // namespace tfdml
