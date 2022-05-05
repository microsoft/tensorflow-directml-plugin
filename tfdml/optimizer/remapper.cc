/* Copyright (c) Microsoft Corporation.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tfdml/optimizer/remapper.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tfdml/optimizer/graph_properties.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/tensor_proto_util.h"
#include "tfdml/optimizer/utils.h"
#include "tfdml/runtime_adapter/padding.h"
#include "tfdml/runtime_adapter/tensor_format.h"
#include "tfdml/runtime_adapter/tensor_shape_utils.h"

namespace tfdml
{

constexpr char kDataFormat[] = "data_format";
constexpr char kPadding[] = "padding";
constexpr char kDilations[] = "dilations";

struct RemapperContext
{
    static Status InitializeRemapperContext(
        const GrapplerItem& item,
        RemapperContext* context);

    tensorflow::GraphDef graph;

    int num_nodes;
    absl::flat_hash_set<std::string> nodes_to_preserve;
    std::unique_ptr<GraphProperties> graph_properties;
    std::unique_ptr<MutableGraphView> graph_view;
    bool inferred_graph_properties;
};

Status RemapperContext::InitializeRemapperContext(
    const GrapplerItem& item,
    RemapperContext* context)
{
    assert(context != nullptr);
    context->graph_properties = absl::make_unique<GraphProperties>(item);
    TF_RETURN_IF_ERROR(context->graph_properties->InferStatically(true));
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

// Pad node followed by a Conv2D.
struct PadWithConv2D
{
    PadWithConv2D() = default;
    int pad = kMissingIndex;
    int conv_2d = kMissingIndex;
    int32_t new_padding_values[8] = {0};
};

bool IsInPreserveSet(
    const RemapperContext* ctx,
    const tensorflow::NodeDef* node)
{
    return ctx->nodes_to_preserve.count(node->name()) > 0;
}

bool HaveSameDataType(
    const tensorflow::NodeDef* lhs,
    const tensorflow::NodeDef* rhs,
    const std::string& type_attr = "T")
{
    tensorflow::DataType lhs_attr = GetDataTypeFromAttr(*lhs, type_attr);
    tensorflow::DataType rhs_attr = GetDataTypeFromAttr(*rhs, type_attr);

    return lhs_attr != tensorflow::DT_INVALID &&
           rhs_attr != tensorflow::DT_INVALID && lhs_attr == rhs_attr;
}

bool IsSupportedActivation(const tensorflow::NodeDef& node)
{
    return IsRelu(node) || IsRelu6(node) || IsElu(node) || IsLeakyRelu(node);
}

inline bool HasControlFaninOrFanout(const MutableNodeView& node_view)
{
    return node_view.NumControllingFanins() > 0 ||
           node_view.NumControlledFanouts() > 0;
}

// Returns true if at most one fanout reads output at port 0 (output used once).
inline bool HasAtMostOneFanoutAtPort0(const MutableNodeView& node_view)
{
    return node_view.GetRegularFanout(0).size() <= 1;
}

bool FindPadWithConv2D(
    const RemapperContext* ctx,
    int node_index,
    PadWithConv2D* matched)
{
    const auto* conv_node_view = ctx->graph_view->GetNode(node_index);
    const auto* conv_node_def = conv_node_view->node();

    // Root of the pattern must be a Conv2D.
    if (!IsConv2D(*conv_node_def)) return false;

    // Forward controls for patterns with control dependencies.
    if (HasControlFaninOrFanout(*conv_node_view)) return false;

    // Input to the Conv2D must be a Pad.
    if (conv_node_view->NumRegularFanins() < 1) return false;
    const auto& regular_fanin_0 = conv_node_view->GetRegularFanin(0);
    const auto* pad_node_view = regular_fanin_0.node_view();
    const auto* pad_node_def = pad_node_view->node();

    if (!IsPad(*pad_node_def)) return false;
    if (!HaveSameDataType(conv_node_def, pad_node_def)) return false;
    if (HasControlFaninOrFanout(*pad_node_view)) return false;
    if (!HasAtMostOneFanoutAtPort0(*pad_node_view)) return false;
    if (IsInPreserveSet(ctx, pad_node_def)) return false;

    const auto& paddings_fanin = pad_node_view->GetRegularFanin(1);
    const auto* paddings_node_view = paddings_fanin.node_view();
    const auto* paddings_node_def = paddings_node_view->node();

    if (!IsConstant(*paddings_node_def)) return false;

    TensorFormat data_format;
    bool valid_data_format = FormatFromString(
        conv_node_def->attr().at(kDataFormat).s(),
        &data_format);

    if (!valid_data_format) return false;
    const std::vector<tensorflow::OpInfo::TensorProperties>& pad_props =
        ctx->graph_properties->GetInputProperties(pad_node_def->name());
    const auto& pad_op = pad_node_def->op();

    if (pad_op == "PadV2")
    {
        if (pad_props.size() != 3) return false;

        // Make sure that the padding value is 0
        tensorflow::DataType dtype = pad_props[2].value().dtype();
        const auto& constant_values_fanin = pad_node_view->GetRegularFanin(2);
        const auto* constant_values_node_view =
            constant_values_fanin.node_view();
        const auto* constant_values_node_def =
            constant_values_node_view->node();

        if (!IsConstant(*constant_values_node_def)) return false;

        auto val_tensor = constant_values_node_def->attr().at("value").tensor();

        const auto* value_attr = constant_values_node_view->GetAttr("value");
        if (value_attr == nullptr) return false;

        float float_val;
        switch (val_tensor.dtype())
        {
        case tensorflow::DT_HALF:
            if (val_tensor.half_val_size() == 0)
            {
                float_val = value_attr->f();
            }
            else
            {
                float_val = static_cast<float>(
                    GetTensorElement<Eigen::half>(val_tensor, 0));
            }
            break;
        case tensorflow::DT_FLOAT:
            if (val_tensor.float_val_size() == 0)
            {
                float_val = value_attr->f();
            }
            else
            {
                float_val = GetTensorElement<float>(val_tensor, 0);
            }
            break;
        default: return false;
        }

        if (float_val != 0.0f) return false;
    }
    else if (pad_op != "Pad")
    {
        return false;
    }

    int n_index = GetTensorDimIndex(data_format, 'N', 4);
    int h_index = GetTensorDimIndex(data_format, 'H', 4);
    int w_index = GetTensorDimIndex(data_format, 'W', 4);
    int c_index = GetTensorDimIndex(data_format, 'C', 4);

    if (pad_props.size() < 2) return false;
    if (!pad_props[1].has_shape()) return false;
    if (pad_props[1].shape().dim_size() != 2) return false;
    if (pad_props[1].shape().dim(0).size() != 4) return false;
    if (pad_props[1].shape().dim(1).size() != 2) return false;

    const auto* value_attr = paddings_node_view->GetAttr("value");
    if (value_attr == nullptr) return false;

    auto val_tensor = paddings_node_def->attr().at("value").tensor();

    // Make sure that the paddings are known and that the batch and depth
    // paddings are 0
    switch (val_tensor.dtype())
    {
    case tensorflow::DT_INT32: {
        for (int i = 0; i < 4; ++i)
        {
            matched->new_padding_values[i * 2] =
                GetTensorElement<int32_t>(val_tensor, i * 2);
            matched->new_padding_values[i * 2 + 1] =
                GetTensorElement<int32_t>(val_tensor, i * 2 + 1);
        }
    }
    break;
    case tensorflow::DT_INT64: {
        for (int i = 0; i < 4; ++i)
        {
            matched->new_padding_values[i * 2] =
                GetTensorElement<int64_t>(val_tensor, i * 2);
            matched->new_padding_values[i * 2 + 1] =
                GetTensorElement<int64_t>(val_tensor, i * 2 + 1);
        }
    }
    break;
    default: return false;
    }

    // Conv2D doesn't support padding for non-spatial dimensions
    if (matched->new_padding_values[n_index * 2] != 0) return false;
    if (matched->new_padding_values[n_index * 2 + 1] != 0) return false;
    if (matched->new_padding_values[c_index * 2] != 0) return false;
    if (matched->new_padding_values[c_index * 2 + 1] != 0) return false;

    Padding padding_type;

    auto padding_t = conv_node_def->attr().at(kPadding).tensor();

    const auto* p_value_attr = conv_node_view->GetAttr(kPadding);
    if (p_value_attr == nullptr) return false;

    if (p_value_attr->s() == "VALID")
    {
        padding_type = Padding::VALID;
    }
    else if (p_value_attr->s() == "EXPLICIT")
    {
        padding_type = Padding::EXPLICIT;
    }
    else if (p_value_attr->s() == "SAME")
    {
        padding_type = Padding::SAME;
    }
    else
        return false;

    if (padding_type == Padding::EXPLICIT)
    {
        auto paddings = conv_node_def->attr().at("explicit_paddings").list();
        if (paddings.i_size() != 8) return false;
        // We add Conv2D's explicit padding to Pad's padding
        for (int i = 0; i < 8; ++i)
        {
            matched->new_padding_values[i] += paddings.i(i);
        }
    }
    else if (padding_type == Padding::SAME)
    {
        const auto& conv_props =
            ctx->graph_properties->GetInputProperties(conv_node_def->name());
        if (conv_props.size() != 2) return false;
        if (!conv_props[1].has_shape()) return false;
        auto filter_shape = conv_props[1].shape();
        if (filter_shape.dim_size() != 4) return false;

        // Make sure that all filter dimensions are statically known
        bool known_filter_shape = std::all_of(
            filter_shape.dim().begin(),
            filter_shape.dim().end(),
            [](const tensorflow::TensorShapeProto::Dim& dim)
            { return dim.size() >= 0; });

        if (!known_filter_shape)
        {
            // The filter shape is not known, so check if it's a Placeholder op
            // that has a valid shape attribute instead
            if (conv_node_view->NumRegularFanins() < 2) return false;
            const auto& conv_fanin_1 = conv_node_view->GetRegularFanin(1);
            const auto* filter_node_view = conv_fanin_1.node_view();
            const auto* filter_node_def = filter_node_view->node();

            if (!IsPlaceholder(*filter_node_def)) return false;

            auto shape_iterator = filter_node_def->attr().find("shape");
            if (shape_iterator == filter_node_def->attr().end()) return false;

            filter_shape = shape_iterator->second.shape();
            known_filter_shape = std::all_of(
                filter_shape.dim().begin(),
                filter_shape.dim().end(),
                [](const tensorflow::TensorShapeProto::Dim& dim)
                { return dim.size() >= 0; });
            if (!known_filter_shape) return false;
        }

        auto dilations_attr = conv_node_def->attr().at(kDilations).list();
        if (dilations_attr.i_size() != 4) return false;
        const int64_t dilation_h = dilations_attr.i(h_index);
        const int64_t dilation_w = dilations_attr.i(w_index);

        // The filter shape is in HWCN format. After simplifying the padding
        // equations from GetWindowedOutputSizeVerboseV2, the input and strides
        // terms disappear and all we need to compute the padding are the
        // dilations and filter shape.
        const int64_t padding_needed_h =
            std::max<int64_t>(0, (filter_shape.dim(0).size() - 1) * dilation_h);
        const int64_t padding_needed_w =
            std::max<int64_t>(0, (filter_shape.dim(1).size() - 1) * dilation_w);
        const int64_t padding_before_h = padding_needed_h / 2;
        const int64_t padding_after_h = padding_needed_h - padding_before_h;
        const int64_t padding_before_w = padding_needed_w / 2;
        const int64_t padding_after_w = padding_needed_w - padding_before_w;

        // We add Conv2D's computed padding to Pad's padding
        matched->new_padding_values[h_index * 2] += padding_before_h;
        matched->new_padding_values[h_index * 2 + 1] += padding_after_h;
        matched->new_padding_values[w_index * 2] += padding_before_w;
        matched->new_padding_values[w_index * 2 + 1] += padding_after_w;
    }
    else if (padding_type != Padding::VALID)
    {
        return false;
    }

    // We successfully found a Pad+Conv2D pattern.
    matched->pad = pad_node_view->node_index();
    matched->conv_2d = node_index;
    return true;
}

void CopyConv2DAttributes(
    const tensorflow::NodeDef& conv2d,
    tensorflow::NodeDef* fused_conv2d,
    const tensorflow::NodeDef* activation = nullptr)
{
    assert(IsConv2D(conv2d));

    auto* attr = fused_conv2d->mutable_attr();
    auto& src_attr = conv2d.attr();

    (*attr)["T"] = src_attr.at("T");
    (*attr)["strides"] = src_attr.at("strides");
    (*attr)["padding"] = src_attr.at("padding");
    (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
    (*attr)["dilations"] = src_attr.at("dilations");
    (*attr)["data_format"] = src_attr.at("data_format");
    (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
    // Copy LeakyRelu's attr alpha to FusedConv2D's attr leakyrelu_alpha
    if (activation != nullptr && IsLeakyRelu(*activation))
    {
        auto& activation_attr = activation->attr();
        (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
    }
}

static void FuseConv2DExplicitPaddings(
    const PadWithConv2D& matched,
    tensorflow::NodeDef& fused_op)
{
    fused_op.mutable_attr()->at("padding").set_s("EXPLICIT");

    auto* paddings =
        (*fused_op.mutable_attr())["explicit_paddings"].mutable_list();
    paddings->Clear();

    for (int i = 0; i < 8; ++i)
    {
        paddings->add_i(matched.new_padding_values[i]);
    }
}

Status AddFusedContractionNode(
    RemapperContext* ctx,
    const PadWithConv2D& matched,
    std::vector<bool>* invalidated_nodes,
    std::vector<bool>* nodes_to_delete)
{
    const tensorflow::GraphDef* graph = ctx->graph_view->graph();
    const tensorflow::NodeDef& pad = graph->node(matched.pad);
    const tensorflow::NodeDef& conv_2d = graph->node(matched.conv_2d);

    tensorflow::NodeDef fused_op;
    fused_op.set_name(conv_2d.name());
    fused_op.set_device(conv_2d.device());
    fused_op.add_input(pad.input(0));     // 0: input
    fused_op.add_input(conv_2d.input(1)); // 1: filter
    fused_op.set_op(conv_2d.op());
    CopyConv2DAttributes(conv_2d, &fused_op);

    FuseConv2DExplicitPaddings(matched, fused_op);

    Mutation* mutation = ctx->graph_view->GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());

    (*nodes_to_delete)[matched.pad] = true;
    (*invalidated_nodes)[matched.conv_2d] = true;

    return Status::OK();
}

// Check if a node is a candidate to one of the patterns that require inferred
// shapes:
//   (1) Fusing Pad into Conv2D
bool RequiresInferredShapes(const RemapperContext& ctx, int node_index)
{
    // Candidate for a FusedBatchNorm splitting.
    const auto* node_view = ctx.graph_view->GetNode(node_index);
    const auto* node_def = node_view->node();

    const auto is_pad_conv2d_fusion_candidate = [&]() -> bool
    {
        const auto* current_node_view = node_view;

        if (IsSupportedActivation(*current_node_view->node()))
        {
            // Pad + Conv2D + BiasAdd + Activation
            if (current_node_view->NumRegularFanins() < 1) return false;

            const auto& fanin_0 = current_node_view->GetRegularFanin(0);
            current_node_view = fanin_0.node_view();
        }

        if (IsBiasAdd(*current_node_view->node()))
        {
            // Pad + Conv2D + BiasAdd
            if (current_node_view->NumRegularFanins() < 1) return false;

            const auto& fanin_0 = current_node_view->GetRegularFanin(0);
            current_node_view = fanin_0.node_view();
        }

        if (IsConv2D(*current_node_view->node()))
        {
            // Pad + Conv2D
            if (current_node_view->NumRegularFanins() < 1) return false;

            const auto& fanin_0 = current_node_view->GetRegularFanin(0);
            current_node_view = fanin_0.node_view();

            return IsPad(*current_node_view->node());
        }

        return false;
    };

    return is_pad_conv2d_fusion_candidate();
}

Status Remapper::Optimize(
    const GrapplerItem& item,
    tensorflow::GraphDef* optimized_graph)
{
    RemapperContext ctx;
    TF_RETURN_IF_ERROR(RemapperContext::InitializeRemapperContext(item, &ctx));
    // Processing graph in reverse-topological sorted order allows to remap
    // longer chains of dependent ops in one pass.
    TF_RETURN_IF_ERROR(
        ctx.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));

    const int num_nodes = item.graph.node_size();
    // Skip nodes that were invalidated by a remapper, e.g. do not process
    // BiasAdd and Activation nodes that were fused into a Conv2D node.
    std::vector<bool> invalidated_nodes(num_nodes);
    std::vector<bool> nodes_to_delete(num_nodes);

    // _Fused{...} kernels do not have registered gradient function, so we must
    // not perform rewrite if the graph will be differentiated later.
    bool allow_non_differentiable_rewrites = true;

    for (int i = num_nodes - 1; i >= 0; --i)
    {
        // Check if node was invalidated by one of the previous remaps.
        if (invalidated_nodes[i] || nodes_to_delete[i])
        {
            continue;
        }

        // Infer properties lazily in case they are not needed.
        if (!ctx.inferred_graph_properties && RequiresInferredShapes(ctx, i))
        {
            const bool assume_valid_feeds = true;
            TF_RETURN_IF_ERROR(ctx.graph_properties->InferStatically(
                assume_valid_feeds,
                /*aggressive_shape_inference=*/false,
                /*include_input_tensor_values=*/true,
                /*include_output_tensor_values=*/false));
            ctx.inferred_graph_properties = true;
        }

        // Infer properties lazily in case they are not needed.
        if (!ctx.inferred_graph_properties && RequiresInferredShapes(ctx, i))
        {
            const bool assume_valid_feeds = true;
            TF_RETURN_IF_ERROR(ctx.graph_properties->InferStatically(
                assume_valid_feeds,
                /*aggressive_shape_inference=*/false,
                /*include_input_tensor_values=*/true,
                /*include_output_tensor_values=*/false));
            ctx.inferred_graph_properties = true;
        }

        // Remap Pad + Conv2D into Conv2D with explicit padding
        PadWithConv2D pad_with_conv_2d;
        if (allow_non_differentiable_rewrites &&
            FindPadWithConv2D(&ctx, i, &pad_with_conv_2d))
        {
            TF_RETURN_IF_ERROR(AddFusedContractionNode(
                &ctx,
                pad_with_conv_2d,
                &invalidated_nodes,
                &nodes_to_delete));
            continue;
        }
    }

    // Remove invalidated nodes.
    Mutation* mutation = ctx.graph_view->GetMutationBuilder();
    for (int i = 0; i < num_nodes; ++i)
    {
        if (nodes_to_delete[i])
        {
            mutation->RemoveNode(ctx.graph_view->GetNode(i));
        }
    }
    TF_RETURN_IF_ERROR(mutation->Apply());

    *optimized_graph = std::move(ctx.graph);

    return Status::OK();
}

} // namespace tfdml
