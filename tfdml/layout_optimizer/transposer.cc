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

#include "tfdml/layout_optimizer/transposer.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tfdml/layout_optimizer/scoped_data_format_upgrader.h"
#include "tfdml/layout_optimizer/transpose_context.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/node_utils.h"
#include "tfdml/optimizer/op_registry.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{
constexpr char kAttrDataFormat[] = "data_format";
constexpr char kAttrKSize[] = "ksize";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrDilations[] = "dilations";
constexpr char kAttrExplicitPaddings[] = "explicit_paddings";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kAttrValue[] = "value";
constexpr char kAttrNumSplit[] = "num_split";
constexpr char kAttrIsTraining[] = "is_training";
constexpr char kAttrNumOuts[] = "num_outs";
constexpr char kAttrKeepDims[] = "keep_dims";
constexpr char kAttrSqueezeDims[] = "squeeze_dims";
constexpr char kAttrN[] = "N";
constexpr char kAttrT[] = "T";
constexpr int kRank = 4;
constexpr int kUnknownRank = -1;
constexpr int kInvalidRank = -2;

static bool AttrDataFormatMatch(
    const MutableNodeView& node,
    absl::string_view src_data_format,
    bool* missing)
{
    const auto* attr = node.GetAttr(kAttrDataFormat);
    if (attr != nullptr)
    {
        return attr->s() == src_data_format;
    }
    *missing = true;
    return false;
}

static bool AttrDataFormatMatch(
    const MutableNodeView& node,
    absl::string_view src_data_format)
{
    bool missing = false;
    return AttrDataFormatMatch(node, src_data_format, &missing);
}

// Permutes elements according to permutation and replaces the original values.
// Permutation and values must have same size.
template <typename T>
static Status PermuteSingle(
    absl::string_view location,
    absl::Span<const int> permutation,
    T* values)
{
    assert(values != nullptr);
    int permutation_size = permutation.size();
    if (values->size() != permutation_size)
    {
        return errors::InvalidArgument(absl::StrCat(
            "Size of values ",
            values->size(),
            " does not match size of permutation ",
            permutation_size,
            " @ ",
            location));
    }
    typedef typename T::value_type V;
    std::vector<V> elements(values->begin(), values->end());
    int index = 0;
    for (V& element : *values)
    {
        element = elements[permutation[index++]];
    }
    return Status::OK();
}

bool Transposer::ShouldProcess(
    const TransposeContext& context,
    const MutableNodeView& node) const
{
    const auto* node_def = node.node();
    const std::string& device_name = node_def->device();
    std::string device;
    std::string task;
    const bool is_on_target_device =
        DeviceNameUtils::SplitDeviceName(device_name, &task, &device) &&
        absl::StrContains(
            absl::AsciiStrToLower(device),
            absl::AsciiStrToLower(context.target_device));

    // Only checks data format for layout sensitive op.
    const bool data_format_match =
        !IsLayoutSensitiveOp(*node_def) ||
        AttrDataFormatMatch(node, context.src_format);

    // Only transposes floating point nodes.
    const bool is_integer_conv2d = IsNonFloatingConv2D(node);

    return is_on_target_device && data_format_match && !is_integer_conv2d &&
           !context.nodes_to_preserve.contains(node_def->name()) &&
           !(node.NumRegularFanouts() == 0 && node.NumControlledFanouts() == 0);
}

Status LayoutSensitiveOpTransposer::UpdateNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    Mutation* mutation = context->graph_view->GetMutationBuilder();
    tensorflow::AttrValue data_format_attr;
    data_format_attr.set_s(context->dst_format);
    mutation->AddOrUpdateNodeAttr(node, kAttrDataFormat, data_format_attr);

    auto permute_attr =
        [&context, &node, &mutation](absl::string_view attr_name)
    {
        const auto* attr = node->GetAttr(attr_name);
        if (attr != nullptr)
        {
            tensorflow::AttrValue attr_copy(*attr);
            TF_RETURN_IF_ERROR(PermuteSingle(
                absl::StrCat(attr_name, " attribute in", node->GetName()),
                context->src_to_dst,
                attr_copy.mutable_list()->mutable_i()));
            mutation->AddOrUpdateNodeAttr(node, attr_name, attr_copy);
        }
        return Status::OK();
    };

    // Update attrs.
    TF_RETURN_IF_ERROR(permute_attr(kAttrStrides));
    TF_RETURN_IF_ERROR(permute_attr(kAttrKSize));
    TF_RETURN_IF_ERROR(permute_attr(kAttrDilations));

    const auto* explicit_paddings_attr = node->GetAttr(kAttrExplicitPaddings);
    if (explicit_paddings_attr != nullptr &&
        explicit_paddings_attr->has_list() &&
        explicit_paddings_attr->list().i_size() > 0)
    {
        tensorflow::AttrValue explicit_paddings_attr_copy(
            *explicit_paddings_attr);
        TF_RETURN_IF_ERROR(PermuteDouble(
            absl::StrCat("explicit_paddings attribute in", node->GetName()),
            context->src_to_dst,
            explicit_paddings_attr_copy.mutable_list()->mutable_i()));
        mutation->AddOrUpdateNodeAttr(
            node,
            kAttrExplicitPaddings,
            explicit_paddings_attr_copy);
    }

    return Status::OK();
}

Status DefaultLayoutSensitiveOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsDefaultLayoutSensitiveOp(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

std::string Transposer::GetFaninNameFormat(
    absl::string_view node_name,
    int port,
    absl::string_view src_format,
    absl::string_view dst_format)
{
    return absl::StrCat(
        node_name,
        "-",
        port,
        "-$0",
        src_format,
        "To",
        dst_format,
        "-",
        kOptimizedSuffix);
}

std::string Transposer::GetFanoutNameFormat(
    absl::string_view node_name,
    int port,
    int index,
    absl::string_view src_format,
    absl::string_view dst_format)
{
    return absl::StrCat(
        node_name,
        "-",
        port,
        "-",
        index,
        "-$0",
        dst_format,
        "To",
        src_format,
        "-",
        kOptimizedSuffix);
}

Status Transposer::UpdateFaninEdgesWithOp(
    TransposeContext* context,
    absl::Span<const int> dst_ports,
    MutableNodeView* dst_node,
    absl::string_view op)
{
    const bool is_in_frame = true;
    for (int dst_port : dst_ports)
    {
        auto& fanin_port = dst_node->GetRegularFanin(dst_port);
        auto* fanin_node_view = fanin_port.node_view();

        TF_RETURN_IF_ERROR(UpdateEdge(
            context,
            GetFaninNameFormat(
                dst_node->GetName(),
                dst_port,
                context->src_format,
                context->dst_format),
            op,
            /*input_shape=*/nullptr,
            /*is_in_frame=*/is_in_frame,
            /*is_src_format_to_dst_format=*/true,
            fanin_port.index(),
            dst_port,
            fanin_node_view,
            dst_node));
    }
    return Status::OK();
}

struct ComparatorByNodeNameAndIndex
{
    bool operator()(
        const MutableFaninView& node1,
        const MutableFaninView& node2) const
    {
        auto* node1_view = node1.node_view();
        auto* node2_view = node2.node_view();
        auto name_compare =
            node1_view->GetName().compare(node2_view->GetName());
        if (name_compare == 0)
        {
            return node1.index() < node2.index();
        }
        return name_compare < 0;
    }
};

Status Transposer::UpdateFanoutEdgesWithOp(
    TransposeContext* context,
    absl::Span<const int> src_ports,
    MutableNodeView* src_node,
    absl::string_view op)
{
    // Update attr _output_shapes for output ports.
    const auto* output_shape_attr = src_node->GetAttr(kAttrOutputShape);
    tensorflow::AttrValue shape_attr_copy;
    if (op == kOpTranspose && output_shape_attr != nullptr)
    {
        shape_attr_copy = *output_shape_attr;
        for (int port : src_ports)
        {
            auto* shape = shape_attr_copy.mutable_list()->mutable_shape(port);
            if (shape->unknown_rank()) continue;
            TF_RETURN_IF_ERROR(PermuteSingle(
                absl::StrCat(
                    "output shape attribute at port ",
                    port,
                    " in",
                    src_node->GetName()),
                context->src_to_dst,
                shape->mutable_dim()));
        }
        context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
            src_node,
            kAttrOutputShape,
            shape_attr_copy);
    }

    const bool is_in_frame = true;
    // We might modify the output set in the loop. Make a copy first.
    // Use a set with custom comparator to order output nodes by node name,
    // so that we can keep transposer name deterministic.
    for (int src_port : src_ports)
    {
        const auto& fanouts_src_port = src_node->GetRegularFanout(src_port);
        std::vector<MutableFaninView> sorted_fanouts(
            fanouts_src_port.begin(),
            fanouts_src_port.end());
        std::sort(
            sorted_fanouts.begin(),
            sorted_fanouts.end(),
            ComparatorByNodeNameAndIndex());
        int num_downstream_transposers = 0;
        for (const auto& fanout : sorted_fanouts)
        {
            TF_RETURN_IF_ERROR(UpdateEdge(
                context,
                GetFanoutNameFormat(
                    src_node->GetName(),
                    src_port,
                    num_downstream_transposers++,
                    context->src_format,
                    context->dst_format),
                op,
                &shape_attr_copy,
                /*is_in_frame=*/is_in_frame,
                /*is_src_format_to_dst_format=*/false,
                src_port,
                fanout.index(),
                src_node,
                fanout.node_view()));
        }
    }
    return Status::OK();
}

int Transposer::GetFanoutPortRank(const MutableNodeView& node, int port) const
{
    const auto* output_shape_attr = node.GetAttr(kAttrOutputShape);
    if (output_shape_attr == nullptr ||
        output_shape_attr->list().shape_size() <= port)
    {
        return kInvalidRank;
    }
    const auto& shape = output_shape_attr->list().shape(port);
    if (shape.unknown_rank())
    {
        return kUnknownRank;
    }
    return shape.dim_size();
}

Status Transposer::CreateDataFormatNode(
    TransposeContext* context,
    absl::string_view node_name,
    absl::string_view op,
    absl::string_view device,
    const tensorflow::DataType& data_type,
    bool is_fanin_on_host,
    bool is_src_format_to_dst_format,
    MutationNewNode* added_node)
{
    auto* graph_view = context->graph_view.get();
    assert(!graph_view->HasNode(node_name));

    // Create the node
    tensorflow::NodeDef node;
    node.set_name(std::string(node_name));

    node.set_device(std::string(device));
    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    node.mutable_attr()->insert({"T", attr_data_type});

    std::string op_name;

    // The inputs of a DataFormat op could be in host memory for ops such as
    // Reshape. In such cases, run the kernel on the host too.
    if (is_fanin_on_host)
    {
        // TODO: Remove once TensorFlow core implements host for DEVICE_DEFAULT
        // https://github.com/tensorflow/tensorflow/pull/55558
        if (op == kOpDataFormatVecPermute)
        {
            // Set up parameters of node.
            node.set_op("DmlDataFormatVecPermuteHost");
        }
        else if (op == kOpDataFormatDimMap)
        {
            // Set up parameters of node.
            node.set_op("DmlDataFormatDimMapHost");
        }
        else
        {
            return errors::InvalidArgument(
                "Unsupported data format op '",
                op,
                "'");
        }
        // tensorflow::AttrValue attr_kernel;
        // attr_kernel.set_s("host");
        // node.mutable_attr()->insert({"_kernel", attr_kernel});
    }
    else
    {
        // Set up parameters of node.
        node.set_op(std::string(op));
    }

    tensorflow::AttrValue src_format;
    src_format.set_s(
        is_src_format_to_dst_format ? context->src_format
                                    : context->dst_format);
    node.mutable_attr()->insert({kAttrSrcFormat, src_format});
    tensorflow::AttrValue dst_format;
    dst_format.set_s(
        is_src_format_to_dst_format ? context->dst_format
                                    : context->src_format);
    node.mutable_attr()->insert({kAttrDstFormat, dst_format});

    // Add place holder for 1st input field.
    node.add_input("");

    Status status;
    *added_node =
        graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
    return status;
}

Status Transposer::CreateConstPermNode(
    TransposeContext* context,
    absl::string_view node_name,
    absl::string_view device,
    absl::Span<const int> permutation,
    absl::string_view control_node_name,
    MutationNewNode* added_node)
{
    auto* graph_view = context->graph_view.get();
    assert(!graph_view->HasNode(node_name));

    tensorflow::NodeDef node;
    node.set_name(std::string(node_name));
    node.set_op(kOpConst);
    node.set_device(std::string(device));

    if (!control_node_name.empty())
    {
        node.add_input(std::string(control_node_name));
    }

    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(tensorflow::DT_INT32);
    node.mutable_attr()->insert({"dtype", attr_data_type});

    tensorflow::AttrValue attr_tensor;
    tensorflow::TensorProto* tensor = attr_tensor.mutable_tensor();
    tensor->set_dtype(tensorflow::DT_INT32);
    tensor->mutable_tensor_shape()->add_dim()->set_size(permutation.size());

    for (int i = 0; i < permutation.size(); i++)
    {
        tensor->add_int_val(permutation[i]);
    }

    node.mutable_attr()->insert({"value", attr_tensor});

    Status status;
    *added_node =
        graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
    return status;
}

Status Transposer::CreateTransposeNode(
    TransposeContext* context,
    absl::string_view name_format,
    const tensorflow::DataType& data_type,
    absl::string_view device,
    tensorflow::TensorShapeProto fanin_shape,
    absl::Span<const int> permutation,
    absl::string_view control_node_name,
    MutationNewNode* added_node,
    std::string* transpose_node_name)
{
    const std::string node_name = absl::Substitute(name_format, kOpTranspose);
    auto* graph_view = context->graph_view.get();
    assert(!graph_view->HasNode(node_name));
    *transpose_node_name = node_name;

    tensorflow::NodeDef node;
    node.set_name(node_name);
    node.set_op(kOpTranspose);
    node.set_device(std::string(device));

    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    node.mutable_attr()->insert({"T", attr_data_type});

    tensorflow::AttrValue attr_data_type_perm;
    attr_data_type_perm.set_type(tensorflow::DT_INT32);
    node.mutable_attr()->insert({"Tperm", attr_data_type_perm});

    if (!fanin_shape.unknown_rank())
    {
        TF_RETURN_IF_ERROR(PermuteSingle(
            absl::StrCat("fanin shape in", node.name()),
            permutation,
            fanin_shape.mutable_dim()));
        tensorflow::AttrValue attr_output_shape;
        *attr_output_shape.mutable_list()->add_shape() = fanin_shape;
        node.mutable_attr()->insert({kAttrOutputShape, attr_output_shape});
    }

    // Create Const Node
    MutationNewNode const_perm_added_node;
    const std::string const_perm_node_name =
        absl::Substitute(name_format, "PermConst");
    TF_RETURN_IF_ERROR(CreateConstPermNode(
        context,
        const_perm_node_name,
        device,
        permutation,
        control_node_name,
        &const_perm_added_node));
    // Add place holder for 1st input.
    node.add_input("");
    // Connect const_perm_node to 2nd input of transpose_node.
    node.add_input(const_perm_node_name);

    Status status;
    *added_node =
        graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
    return status;
}

Status Transposer::UpdateEdge(
    TransposeContext* context,
    absl::string_view name_format,
    absl::string_view op,
    const tensorflow::AttrValue* input_shape,
    bool is_in_frame,
    bool is_src_format_to_dst_format,
    const int src_port,
    const int dst_port,
    MutableNodeView* src_node,
    MutableNodeView* dst_node)
{
    assert(src_node != nullptr);
    assert(dst_node != nullptr);
    auto* src_node_def = src_node->node();
    auto* dst_node_def = dst_node->node();

    const std::string device =
        (is_src_format_to_dst_format ? *dst_node_def : *src_node_def).device();
    tensorflow::DataType data_type =
        is_src_format_to_dst_format
            ? context->graph_properties
                  ->GetInputProperties(dst_node->GetName())[dst_port]
                  .dtype()
            : context->graph_properties
                  ->GetOutputProperties(src_node->GetName())[src_port]
                  .dtype();

    MutationNewNode added_node;
    std::string added_node_name;
    if (op == kOpTranspose)
    {
        tensorflow::TensorShapeProto input_shape_proto;
        input_shape_proto.set_unknown_rank(true);
        if (input_shape != nullptr)
        {
            input_shape_proto = input_shape->list().shape(src_port);
        }
        else
        {
            const auto* src_node_shape_attr =
                src_node->GetAttr(kAttrOutputShape);
            if (src_node_shape_attr != nullptr)
            {
                input_shape_proto = src_node_shape_attr->list().shape(src_port);
            }
        }
        const std::string control_node_name =
            is_in_frame ? AsControlDependency(src_node_def->name()) : "";
        const std::vector<int>& permutation = is_src_format_to_dst_format
                                                  ? context->src_to_dst
                                                  : context->dst_to_src;
        TF_RETURN_IF_ERROR(CreateTransposeNode(
            context,
            name_format,
            data_type,
            device,
            input_shape_proto,
            permutation,
            control_node_name,
            &added_node,
            &added_node_name));
    }
    else if (op == kOpDataFormatVecPermute || op == kOpDataFormatDimMap)
    {
        DeviceNameUtils::ParsedName parsed_name;
        bool is_fanin_on_host = DeviceNameUtils::ParseFullName(
                                    src_node_def->device(),
                                    &parsed_name) &&
                                parsed_name.type != "CPU" &&
                                IsHostMemory(*src_node_def, src_port);
        const std::string node_name = absl::Substitute(name_format, op);
        TF_RETURN_IF_ERROR(CreateDataFormatNode(
            context,
            node_name,
            op,
            device,
            data_type,
            is_fanin_on_host,
            is_src_format_to_dst_format,
            &added_node));
        added_node_name = node_name;
    }
    else
    {
        return errors::InvalidArgument(absl::StrCat(
            "Unsupported op \"",
            op,
            "\". Supported ops are Transpose, "
            "DataFormatVecPerm, DataFormatDimMap."));
    }

    // Connect src_node to 1st input of added_node.
    Mutation* mutation = context->graph_view->GetMutationBuilder();
    mutation->AddOrUpdateRegularFanin(
        added_node,
        0,
        {src_node->GetName(), src_port});

    // Connect output of added_node to dst_node:dst_port.
    mutation->AddOrUpdateRegularFanin(dst_node, dst_port, {added_node_name, 0});

    return Status::OK();
}

bool GetValueAttrFromConstInputNode(
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

inline bool IsValidDataFormatNode(
    const MutableNodeView& node,
    absl::string_view src_format,
    absl::string_view dst_format)
{
    if (!IsDataFormatOp(node))
    {
        return false;
    }
    const auto* src_format_attr = node.GetAttr(kAttrSrcFormat);
    if (src_format_attr == nullptr || src_format_attr->s() != src_format)
    {
        return false;
    }
    const auto* dst_format_attr = node.GetAttr(kAttrDstFormat);
    if (dst_format_attr == nullptr || dst_format_attr->s() != dst_format)
    {
        return false;
    }
    return true;
}

inline bool IsValidConstPermTransposeNode(
    const MutableNodeView& node,
    absl::Span<const int> permutation)
{
    tensorflow::TensorProto tensor;
    if (!GetValueAttrFromConstInputNode(node, IsTranspose, 1, &tensor))
    {
        return false;
    }
    const int permutation_size = permutation.size();

    if (tensor.int_val_size() != permutation_size)
    {
        return false;
    }

    for (int i = 0; i < permutation_size; i++)
    {
        if (permutation[i] != tensor.int_val(i))
        {
            return false;
        }
    }
    return true;
}

inline bool IsLayoutOptimizerAddedDstToSrcTransform(
    const TransposeContext& context,
    const MutableNodeView& node)
{
    return node.node_index() >= context.num_nodes &&
           (IsValidConstPermTransposeNode(node, context.dst_to_src) ||
            IsValidDataFormatNode(
                node,
                context.dst_format,
                context.src_format));
}

static std::vector<int> GetRegularFaninPorts(const MutableNodeView& node)
{
    const int num_regular_fanins = node.NumRegularFanins();
    std::vector<int> values(num_regular_fanins);
    std::iota(values.begin(), values.end(), 0);
    return values;
}

static std::vector<int> GetConcatDataFaninPorts(const MutableNodeView& node)
{
    const auto* n_attr = node.GetAttr(kAttrN);
    const int n = n_attr != nullptr ? n_attr->i() : 0;
    const int start = (node.GetOp() == "Concat") ? 1 : 0;
    const int end = start + n;
    std::vector<int> values(end - start);
    std::iota(values.begin(), values.end(), start);
    return values;
}

static std::vector<int> GetDataFaninPorts(const MutableNodeView& node)
{
    const auto* node_def = node.node();
    if (IsAvgPoolGrad(*node_def) || IsSplit(*node_def))
    {
        return {1};
    }
    if (IsStridedSliceGrad(*node_def))
    {
        return {4};
    }
    if (IsBinaryOp(*node_def) || IsUnaryGrad(*node_def))
    {
        return {0, 1};
    }
    if (IsTernaryOp(*node_def) || IsSelect(*node_def) ||
        IsMaxPoolGrad(*node_def) || IsMaxPoolGradV2(*node_def) ||
        IsMaxPoolGradGradV1(*node_def) || IsMaxPoolGradGradV2(*node_def))
    {
        return {0, 1, 2};
    }
    if (IsShapeN(*node_def) || IsIdentityN(*node_def) || IsAddN(*node_def) ||
        IsMerge(*node_def))
    {
        return GetRegularFaninPorts(node);
    }
    if (IsConcat(*node_def))
    {
        return GetConcatDataFaninPorts(node);
    }
    if (node.NumRegularFanins() > 0)
    {
        return {0};
    }
    return {};
}

bool LayoutAgnosticOpTransposer::IsAfterDstToSrcTransform(
    const TransposeContext& context,
    const MutableNodeView& node) const
{
    std::deque<MutableNodeView*> queue;
    absl::flat_hash_set<MutableNodeView*> visited_nodes;
    auto data_node_pos = GetDataFaninPorts(node);
    for (const int pos : data_node_pos)
    {
        const auto& fanin = node.GetRegularFanin(pos);
        auto* fanin_node = fanin.node_view();
        queue.push_back(fanin_node);
        visited_nodes.insert(fanin_node);
    }
    // The code will exit this while loop in one iteration in most cases, as the
    // graph is already topologically sorted.
    while (!queue.empty())
    {
        MutableNodeView* current_node = queue.front();
        queue.pop_front();
        if (IsLayoutOptimizerAddedDstToSrcTransform(context, *current_node))
        {
            return true;
        }
        // We only continue searching if the path is connected through
        // format-agnostic nodes.
        if (IsLayoutAgnosticOp(*current_node->node()))
        {
            auto current_node_pos = GetDataFaninPorts(*current_node);
            for (const auto& pos : current_node_pos)
            {
                const auto& fanin = current_node->GetRegularFanin(pos);
                auto* fanin_node = fanin.node_view();
                if (visited_nodes.insert(fanin_node).second)
                {
                    queue.push_back(fanin_node);
                }
            }
        }
    }
    return false;
}

inline bool IsLayoutOptimizerAddedDstToSrcTranspose(
    const TransposeContext& context,
    const MutableNodeView& node)
{
    return node.node_index() >= context.num_nodes &&
           IsValidConstPermTransposeNode(node, context.dst_to_src);
}

bool Transposer::IsFanoutPortRankN(const MutableNodeView& node, int port, int n)
    const
{
    return GetFanoutPortRank(node, port) == n;
}

std::vector<int> LayoutAgnosticOpTransposer::GetVariadicNDFaninPorts(
    const TransposeContext& context,
    const MutableNodeView& node,
    int rank) const
{
    std::vector<int> ports;
    const int num_regular_fanins = node.NumRegularFanins();
    ports.reserve(num_regular_fanins);
    for (int i = 0; i < num_regular_fanins; ++i)
    {
        const auto& regular_fanin = node.GetRegularFanin(i);
        auto* regular_fanin_node = regular_fanin.node_view();
        int regular_fanin_port = regular_fanin.index();
        if ((IsFanoutPortRankN(
                *regular_fanin_node,
                regular_fanin_port,
                rank)) &&
            ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
              IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
             IsLayoutOptimizerAddedDstToSrcTranspose(
                 context,
                 *regular_fanin_node)))
        {
            ports.push_back(i);
        }
    }
    return ports;
}

int Transposer::GetFaninPortRank(const MutableNodeView& node, int port) const
{
    if (port < node.NumRegularFanins() && port >= 0)
    {
        const auto& regular_fanin = node.GetRegularFanin(port);
        return GetFanoutPortRank(
            *regular_fanin.node_view(),
            regular_fanin.index());
    }
    return kInvalidRank;
}

bool Transposer::IsFaninPortRankN(const MutableNodeView& node, int port, int n)
    const
{
    return GetFaninPortRank(node, port) == n;
}

bool BinaryOpTransposer::IsNDOperateWithMD(
    const MutableNodeView& node,
    int n,
    int m)
{
    return IsFaninPortRankN(node, 0, n) && IsFaninPortRankN(node, 1, m);
}

bool BinaryOpTransposer::IsFaninShapeSupported(
    const MutableNodeView& node,
    int rank)
{
    return (
        IsNDOperateWithMD(node, rank, 0) || IsNDOperateWithMD(node, rank, 1) ||
        IsNDOperateWithMD(node, rank, rank) ||
        IsNDOperateWithMD(node, 0, rank) || IsNDOperateWithMD(node, 1, rank));
}

std::vector<int> BinaryOpTransposer::GetNDDataFaninPorts(
    const MutableNodeView& node,
    int rank)
{
    std::vector<int> values;
    if (IsFaninPortRankN(node, 0, rank))
    {
        values.push_back(0);
    }
    if (IsFaninPortRankN(node, 1, rank))
    {
        values.push_back(1);
    }
    return values;
}

Status BinaryOpTransposer::AddNodeReshape(
    Mutation* mutation,
    absl::string_view node_name,
    absl::string_view node_device,
    absl::string_view input_name,
    absl::string_view shape_const_node_name,
    const tensorflow::DataType& data_type)
{
    tensorflow::NodeDef new_node;
    new_node.set_name(std::string(node_name));
    new_node.add_input(std::string(input_name));
    new_node.add_input(std::string(shape_const_node_name));
    new_node.set_op(kReshape);
    new_node.set_device(std::string(node_device));

    tensorflow::AttrValue attr_type_indices;
    attr_type_indices.set_type(tensorflow::DT_INT32);
    new_node.mutable_attr()->insert({"Tshape", attr_type_indices});

    tensorflow::AttrValue attr_type_params;
    attr_type_params.set_type(data_type);
    new_node.mutable_attr()->insert({"T", attr_type_params});

    Status status;
    mutation->AddNode(std::move(new_node), &status);
    return status;
}

Status BinaryOpTransposer::AddNodeShapeConst(
    Mutation* mutation,
    absl::string_view node_name,
    absl::string_view node_device,
    bool node_in_frame,
    int num_channels,
    absl::string_view depended_node,
    int rank)
{
    tensorflow::NodeDef new_node;
    new_node.set_name(std::string(node_name));
    new_node.set_op(kOpConst);
    new_node.set_device(std::string(node_device));
    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(tensorflow::DT_INT32);
    new_node.mutable_attr()->insert({"dtype", attr_data_type});

    std::vector<int> shape(rank, 1);
    shape[1] = num_channels;

    tensorflow::AttrValue attr_tensor;
    tensorflow::TensorProto* tensor = attr_tensor.mutable_tensor();
    tensor->set_dtype(tensorflow::DT_INT32);
    tensor->mutable_tensor_shape()->add_dim()->set_size(rank);

    for (int i = 0; i < static_cast<int>(shape.size()); i++)
    {
        tensor->add_int_val(shape[i]);
    }

    new_node.mutable_attr()->insert({"value", attr_tensor});
    if (node_in_frame)
    {
        // This is to ensure the transpose node and the const node are in the
        // same frame.
        new_node.add_input(AsControlDependency(std::string(depended_node)));
    }

    Status status;
    mutation->AddNode(std::move(new_node), &status);
    return status;
}

static std::string TensorIdToString(const TensorId& tensor_id)
{
    return tensor_id.index() == 0 ? std::string(tensor_id.node())
                                  : tensor_id.ToString();
}

std::string Transposer::GetReshapeNodeNameFormat(
    absl::string_view node_name,
    int index,
    absl::string_view src_format,
    absl::string_view dst_format)
{
    return absl::StrCat(
        node_name,
        "-",
        index,
        "-",
        kReshape,
        src_format,
        "To",
        dst_format);
}

std::string Transposer::LayoutOptimizerNode(absl::string_view node_name)
{
    return absl::StrCat(node_name, "-", kOptimizedSuffix);
}

std::string Transposer::GetShapeConstNodeNameFormat(
    absl::string_view node_name,
    int index)
{
    return absl::StrCat(node_name, "-", index, "-", kReshapeConst);
}

Status BinaryOpTransposer::MaybeReshapeVectorFanin(
    TransposeContext* context,
    MutableNodeView* node,
    int rank)
{
    int vector_index = -1;
    if (IsNDOperateWithMD(*node, rank, 1))
    {
        vector_index = 1;
    }
    else if (IsNDOperateWithMD(*node, 1, rank))
    {
        vector_index = 0;
    }
    if (vector_index != -1)
    {
        const std::string& node_name = node->GetName();
        const std::string& node_device = node->GetDevice();
        std::string reshape_node_name =
            LayoutOptimizerNode(GetReshapeNodeNameFormat(
                node_name,
                vector_index,
                context->src_format,
                context->dst_format));
        std::string shape_const_node_name = LayoutOptimizerNode(
            GetShapeConstNodeNameFormat(node_name, vector_index));
        const auto& fanin = node->GetRegularFanin(vector_index);
        auto* fanin_node = fanin.node_view();
        const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
        if (output_shape_attr == nullptr)
        {
            return errors::InvalidArgument(
                "Missing attribute ",
                kAttrOutputShape);
        }
        int vector_size =
            output_shape_attr->list().shape(fanin.index()).dim(0).size();
        Mutation* mutation = context->graph_view->GetMutationBuilder();
        TF_RETURN_IF_ERROR(AddNodeShapeConst(
            mutation,
            shape_const_node_name,
            node_device,
            true,
            vector_size,
            fanin_node->GetName(),
            rank));
        const auto* t_attr = node->GetAttr(kAttrT);
        if (t_attr == nullptr)
        {
            return errors::InvalidArgument("Missing attribute ", kAttrT);
        }
        TF_RETURN_IF_ERROR(AddNodeReshape(
            mutation,
            reshape_node_name,
            node_device,
            TensorIdToString({fanin_node->GetName(), fanin.index()}),
            shape_const_node_name,
            t_attr->type()));
        mutation->AddOrUpdateRegularFanin(
            node,
            vector_index,
            {reshape_node_name, 0});
    }
    return Status::OK();
}

Status BinaryOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsBinaryOp(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsFaninShapeSupported(*node, rank) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        GetNDDataFaninPorts(*node, rank),
        node,
        kOpTranspose));
    TF_RETURN_IF_ERROR(MaybeReshapeVectorFanin(context, node, rank));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status BiasAddGradTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsBiasAddGrad(*node->node()));
    const int rank = GetFaninPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    // BiasAddGrad itself only needs NCHW/NHWC to determine whether C dim is the
    // second or the last dim. Therefore, we use the original 4D data format in
    // the context to update the node. For the input tensor, the corresponding
    // 4D or 5D data format is needed.
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    // No need to update output shape, as it is always of shape 1-D with size
    // the feature dimension of `out_backprop`, regardless of whether NCHW or
    // NHWC is used.
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropInputTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(
        IsConv2DBackpropInput(*node->node()) ||
        IsDepthwiseConv2dNativeBackpropInput(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4))
    {
        return Status::OK();
    }

    const auto& fanin = node->GetRegularFanin(0);
    auto* fanin_node = fanin.node_view();
    const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
    if (output_shape_attr == nullptr)
    {
        TF_VLog(
            3,
            "Cannot compute the shape of %s because it is missing attribute %s",
            fanin_node->GetName().c_str(),
            kAttrOutputShape);
        return Status::OK();
    }
    tensorflow::TensorShapeProto fanin_shape =
        output_shape_attr->list().shape(fanin.index());
    if (fanin_shape.dim_size() != 1)
    {
        TF_VLog(3, "%s is not a vector.", fanin_node->GetName().c_str());
        return Status::OK();
    }

    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {2}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(
        IsConv2DBackpropFilter(*node->node()) ||
        IsDepthwiseConv2dNativeBackpropFilter(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 2}, node, kOpTranspose));
    // No need to update output shape, as it is always of shape
    // [filter_height, filter_width, in_channels, out_channels], regardless of
    // whether NCHW or NHWC is used.
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status DefaultLayoutAgnosticOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsDefaultLayoutAgnosticOp(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsShape(*node->node()));
    const int rank = GetFaninPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool MergeTransposer::IsEveryFaninAfterDstToSrcTransform(
    const TransposeContext& context,
    const MutableNodeView& node) const
{
    for (const auto& regular_fanin : node.GetRegularFanins())
    {
        auto* regular_fanin_node = regular_fanin.node_view();
        if (IsFanoutPortRankN(*regular_fanin_node, regular_fanin.index(), 4) &&
            ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
              IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
             IsLayoutOptimizerAddedDstToSrcTranspose(
                 context,
                 *regular_fanin_node)))
        {
            continue;
        }
        return false;
    }
    return true;
}

Status MergeTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsMerge(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsEveryFaninAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        GetDataFaninPorts(*node),
        node,
        kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

static std::vector<int> GetDataFanoutPorts(const MutableNodeView& node)
{
    const auto* node_def = node.node();
    if (IsIdentityN(*node_def) || IsShape(*node_def) || IsShapeN(*node_def))
    {
        return GetDataFaninPorts(node);
    }
    if (IsSplit(*node_def) || IsSplitV(*node_def))
    {
        const auto* num_split_attr = node.GetAttr(kAttrNumSplit);
        if (num_split_attr == nullptr)
        {
            return {0};
        }
        std::vector<int> values(num_split_attr->i());
        std::iota(values.begin(), values.end(), 0);
        return values;
    }
    if (IsSwitch(*node_def))
    {
        const auto* num_outs_attr = node.GetAttr(kAttrNumOuts);
        const int num_outs = num_outs_attr != nullptr ? num_outs_attr->i() : 2;
        std::vector<int> values(num_outs);
        std::iota(values.begin(), values.end(), 0);
        return values;
    }
    return {0};
}

bool Transposer::IsFanoutPortsRankN(
    const MutableNodeView& node,
    absl::Span<const int> ports,
    int n) const
{
    for (const auto& port : ports)
    {
        if (!IsFanoutPortRankN(node, port, n))
        {
            return false;
        }
    }
    return true;
}

Status SplitTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSplit(*node->node()));
    const auto ports = GetDataFanoutPorts(*node);
    if (!ShouldProcess(*context, *node) ||
        !IsFanoutPortsRankN(*node, ports, 4) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatDimMap));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status ReverseV2Transposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsReverseV2(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool SqueezeTransposer::IsInputConvertible(
    const TransposeContext& context,
    const MutableNodeView& node) const
{
    const auto& regular_fanin_0 = node.GetRegularFanin(0);
    auto* regular_fanin_0_node = regular_fanin_0.node_view();
    const auto* output_shape_attr =
        regular_fanin_0_node->GetAttr(kAttrOutputShape);
    if (output_shape_attr != nullptr)
    {
        auto& shape = output_shape_attr->list().shape(regular_fanin_0.index());
        if (shape.dim_size() != kRank)
        {
            return false;
        }
        const int height_dim = context.src_dim_indices.at('H');
        const int width_dim = context.src_dim_indices.at('W');
        if (shape.dim(height_dim).size() == 1 &&
            shape.dim(width_dim).size() == 1)
        {
            return true;
        }
    }
    return false;
}

static std::vector<int> GetDimensionIndicesFromLabel(
    const absl::flat_hash_map<char, int>& dim_indices,
    absl::Span<const char> labels)
{
    std::vector<int> indices;
    indices.reserve(labels.size());
    for (const auto& label : labels)
    {
        indices.push_back(dim_indices.at(label));
    }
    return indices;
}

bool SqueezeTransposer::IsAlongAxis(
    const tensorflow::AttrValue& attr,
    absl::Span<const int> axis,
    int rank) const
{
    const auto& list = attr.list();
    // If list is empty, Squeeze op will squeeze all dimensions of size 1.
    int axis_size = axis.size();
    if (list.i_size() == 0)
    {
        return true;
    }
    else if (list.i_size() != axis_size)
    {
        return false;
    }
    for (int i = 0; i < axis_size; ++i)
    {
        int local_axis = list.i(i);
        if (local_axis < 0)
        {
            local_axis += rank;
        }
        bool along_axis = false;
        for (int dim : axis)
        {
            if (local_axis == dim)
            {
                along_axis = true;
                break;
            }
        }
        if (!along_axis)
        {
            return false;
        }
    }
    return true;
}

bool SqueezeTransposer::IsDimsSupported(
    const TransposeContext& context,
    const MutableNodeView& node) const
{
    auto indices = [&context](absl::Span<const char> labels)
    { return GetDimensionIndicesFromLabel(context.src_dim_indices, labels); };
    const auto* squeeze_dims_attr = node.GetAttr(kAttrSqueezeDims);
    if (squeeze_dims_attr == nullptr)
    {
        return false;
    }
    return (IsFanoutPortRankN(node, 0, 2) &&
            IsAlongAxis(*squeeze_dims_attr, indices({'H', 'W'}), kRank)) ||
           (IsFanoutPortRankN(node, 0, 1) &&
            IsAlongAxis(*squeeze_dims_attr, indices({'N', 'H', 'W'}), kRank));
}

Status SqueezeTransposer::UpdateSqueezeDims(
    TransposeContext* context,
    MutableNodeView* node)
{
    const auto* squeeze_dims_attr = node->GetAttr(kAttrSqueezeDims);
    if (squeeze_dims_attr == nullptr)
    {
        return errors::InvalidArgument("Missing attribute ", kAttrSqueezeDims);
    }
    const int num_input_dims = context->src_format.length();
    const int min_squeeze_dim = -num_input_dims;
    std::vector<int> squeeze_dims_mapped;
    const int squeeze_dims_size = squeeze_dims_attr->list().i_size();
    squeeze_dims_mapped.reserve(squeeze_dims_size);
    for (int i = 0; i < squeeze_dims_size; ++i)
    {
        int dim = squeeze_dims_attr->list().i(i);
        if (dim < min_squeeze_dim || dim >= num_input_dims)
        {
            return errors::InvalidArgument(
                "Attribute '",
                kAttrSqueezeDims,
                "' contains out of range index '",
                dim,
                "', index must be between [",
                min_squeeze_dim,
                ", ",
                num_input_dims,
                ")");
        }
        if (dim < 0)
        {
            dim += num_input_dims;
        }
        squeeze_dims_mapped.push_back(context->dst_to_src[dim]);
    }
    std::sort(squeeze_dims_mapped.begin(), squeeze_dims_mapped.end());
    tensorflow::AttrValue squeeze_dims;
    squeeze_dims.mutable_list()->mutable_i()->Reserve(squeeze_dims_size);
    for (const auto& dim : squeeze_dims_mapped)
    {
        squeeze_dims.mutable_list()->mutable_i()->Add(dim);
    }
    context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
        node,
        kAttrSqueezeDims,
        squeeze_dims);
    return Status::OK();
}

Status SqueezeTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSqueeze(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsDimsSupported(*context, *node) ||
        !IsInputConvertible(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(UpdateSqueezeDims(context, node));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool Transposer::IsFaninPortDimsNIfConst(
    const MutableNodeView& node,
    int port,
    absl::Span<const int> dims) const
{
    if (port < node.NumRegularFanins() && port >= 0)
    {
        const auto& regular_fanin = node.GetRegularFanin(port);
        const auto* fanin_node_view = regular_fanin.node_view();
        if (!IsConstant(*fanin_node_view->node()))
        {
            return true;
        }
        // If fanin is a Const, check tensor to see if dimensions match.
        const auto* value_attr = fanin_node_view->GetAttr(kAttrValue);
        if (value_attr == nullptr)
        {
            return false;
        }
        const int dims_size = dims.size();
        if (value_attr->tensor().tensor_shape().dim_size() != dims_size)
        {
            return false;
        }
        for (int i = 0; i < dims_size; ++i)
        {
            if (value_attr->tensor().tensor_shape().dim(i).size() != dims[i])
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

Status PadTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(
        IsMirrorPad(*node->node()) || IsMirrorPadGrad(*node->node()) ||
        IsPad(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsFaninPortDimsNIfConst(*node, 1, {4, 2}) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeNTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsShapeN(*node->node()));
    // ShapeN requires all input tensors to have the same dimensions. Therefore,
    // we simply use the 0th fanin port.
    const int rank = GetFaninPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    const auto ports = GetVariadicNDFaninPorts(*context, *node, rank);
    if (!ShouldProcess(*context, *node) || ports.empty())
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, ports, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, ports, node, kOpDataFormatVecPermute));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool Transposer::IsFaninPortsDimsNIfConst(
    const MutableNodeView& node,
    absl::Span<const int> ports,
    absl::Span<const int> dims) const
{
    for (const auto& port : ports)
    {
        if (!IsFaninPortDimsNIfConst(node, port, dims))
        {
            return false;
        }
    }
    return true;
}

Status SliceTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSlice(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsFaninPortsDimsNIfConst(*node, {1, 2}, {rank}) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool StridedSliceTransposer::IsMaskZero(
    const MutableNodeView& node,
    absl::string_view mask)
{
    const auto* mask_attr = node.GetAttr(mask);
    if (mask_attr != nullptr)
    {
        return mask_attr->i() == 0;
    }
    return true;
}

bool StridedSliceTransposer::HasOnlyBeginEndMask(const MutableNodeView& node)
{
    return IsMaskZero(node, "ellipsis_mask") &&
           IsMaskZero(node, "new_axis_mask") &&
           IsMaskZero(node, "shrink_axis_mask");
}

Status StridedSliceTransposer::PermuteMask(
    TransposeContext* context,
    MutableNodeView* node,
    absl::string_view mask)
{
    // Computers the permutation of the masks based on the src and dst format.
    // For example:
    // src_format = NHWC
    // dst_format = NCHW
    // src_to_dst permutation = [0, 3, 1, 2].
    // mask : 0010 [Note the bit positions correspond to indexes i.e this is in
    // reverse order of the src format (CWHN)] result : 0100 (WHCN)
    const auto* mask_attr = node->GetAttr(mask);
    const int mask_i = mask_attr != nullptr ? mask_attr->i() : 0;
    if (mask_i < 0 || mask_i > 15)
    {
        return errors::InvalidArgument("invalid mask value: ", mask_i);
    }
    int result = 0;
    for (int i = 0, end = context->src_to_dst.size(); i < end; i++)
    {
        const int final_pos = context->src_to_dst[i];
        const int position_mask = 1 << final_pos;
        const int bit_i = (mask_i & position_mask) >> final_pos;
        result |= bit_i << i;
    }
    tensorflow::AttrValue new_mask_attr;
    new_mask_attr.set_i(result);
    context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
        node,
        mask,
        new_mask_attr);
    return Status::OK();
}

Status StridedSliceTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsStridedSlice(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsFaninPortsDimsNIfConst(*node, {1, 2, 3}, {4}) ||
        !HasOnlyBeginEndMask(*node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(PermuteMask(context, node, "begin_mask"));
    TF_RETURN_IF_ERROR(PermuteMask(context, node, "end_mask"));
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        {1, 2, 3},
        node,
        kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status ConcatOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsConcat(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        GetConcatDataFaninPorts(*node),
        node,
        kOpTranspose));
    int axis_node = 0;
    if (node->GetOp() == "ConcatV2")
    {
        const auto* n_attr = node->GetAttr(kAttrN);
        if (n_attr != nullptr)
        {
            axis_node = n_attr->i();
        }
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        {axis_node},
        node,
        kOpDataFormatDimMap));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status FillOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsFill(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsFaninPortDimsNIfConst(*node, 0, {4}) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status SwitchTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSwitch(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(
        context,
        GetDataFanoutPorts(*node),
        node,
        kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool ReduceTransposer::KeepDims(const MutableNodeView& node)
{
    const auto* keep_dims_attr = node.GetAttr(kAttrKeepDims);
    if (keep_dims_attr != nullptr)
    {
        return keep_dims_attr->b();
    }
    return false;
}

bool ReduceTransposer::IsAlongAxis(
    const tensorflow::TensorProto& tensor,
    absl::Span<const int> axis,
    int rank)
{
    const int axis_size = axis.size();
    if (tensor.tensor_shape().dim_size() != 1 ||
        tensor.tensor_shape().dim(0).size() != axis_size)
    {
        return false;
    }
    for (int i = 0; i < axis_size; ++i)
    {
        int local_axis = 0;
        if (tensor.dtype() == tensorflow::DT_INT32)
        {
            local_axis = tensor.int_val(i);
        }
        else
        {
            local_axis = tensor.int64_val(i);
        }
        if (local_axis < 0)
        {
            local_axis += rank;
        }
        bool along_axis = false;
        for (int dim : axis)
        {
            if (local_axis == dim)
            {
                along_axis = true;
                break;
            }
        }
        if (!along_axis)
        {
            return false;
        }
    }
    return true;
}

bool ReduceTransposer::IsReduceAxisSupported(
    const TransposeContext& context,
    const MutableNodeView& node,
    int rank)
{
    if (KeepDims(node))
    {
        return true;
    }
    const auto& regular_fanin_1 = node.GetRegularFanin(1);
    auto* axis_node = regular_fanin_1.node_view();
    if (!IsConstant(*axis_node->node()))
    {
        return false;
    }
    const auto* value_attr = axis_node->GetAttr(kAttrValue);
    if (value_attr == nullptr)
    {
        return false;
    }
    const tensorflow::TensorProto& tensor = value_attr->tensor();
    auto indices = [&context](absl::Span<const char> labels)
    { return GetDimensionIndicesFromLabel(context.src_dim_indices, labels); };
    if (rank == 5)
    {
        return IsAlongAxis(tensor, indices({'N', 'D', 'H', 'W', 'C'}), 5) ||
               IsAlongAxis(tensor, indices({'D', 'H', 'W', 'C'}), 5) ||
               IsAlongAxis(tensor, indices({'N', 'D', 'H', 'W'}), 5) ||
               IsAlongAxis(tensor, indices({'D', 'H', 'W'}), 5) ||
               IsAlongAxis(tensor, indices({'C'}), 5);
    }
    assert(rank == 4);
    return IsAlongAxis(tensor, indices({'N', 'H', 'W', 'C'}), 4) ||
           IsAlongAxis(tensor, indices({'H', 'W', 'C'}), 4) ||
           IsAlongAxis(tensor, indices({'N', 'H', 'W'}), 4) ||
           IsAlongAxis(tensor, indices({'H', 'W'}), 4) ||
           IsAlongAxis(tensor, indices({'C'}), 4);
}

Status ReduceTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    eigen_assert(IsReduceOp(*node->node()));
    const int rank = GetFaninPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsReduceAxisSupported(*context, *node, rank) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
    if (KeepDims(*node))
    {
        TF_RETURN_IF_ERROR(
            UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    }
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status UnaryGradTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsUnaryGrad(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status TileTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsTile(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsFaninPortDimsNIfConst(*node, 1, {4}) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status TernaryOpTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsTernaryOp(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitVTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSplitV(*node->node()));
    const auto ports = GetDataFanoutPorts(*node);
    if (!ShouldProcess(*context, *node) ||
        !IsFanoutPortsRankN(*node, ports, 4) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {2}, node, kOpDataFormatDimMap));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool SelectTransposer::IsFaninScalarVector4D(
    const MutableNodeView& fanin,
    int port)
{
    return IsFanoutPortRankN(fanin, port, 0) ||
           IsFanoutPortRankN(fanin, port, 1) ||
           IsFanoutPortRankN(fanin, port, 4);
}

std::vector<int> SelectTransposer::GetFaninPorts(
    const MutableNodeView& fanin,
    int port)
{
    // Input 0 could be a scalar, a vector with size matching the first
    // dimension of input 1 and 2, or must have the same shape as input 1 and 2.
    if (IsFanoutPortRankN(fanin, port, 4))
    {
        return {0, 1, 2};
    }
    return {1, 2};
}

Status SelectTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsSelect(*node->node()));
    const auto& regular_fanin_0 = node->GetRegularFanin(0);
    auto* regular_fanin_0_node = regular_fanin_0.node_view();
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
        !IsFaninScalarVector4D(
            *regular_fanin_0_node,
            regular_fanin_0.index()) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        GetFaninPorts(*regular_fanin_0_node, regular_fanin_0.index()),
        node,
        kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status IdentityNTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsIdentityN(*node->node()));
    const auto ports_4d = GetVariadicNDFaninPorts(*context, *node, 4);

    // Temporarily upgrade the context to obtain the number of 5D fanin ports.
    std::vector<int> ports_5d;
    {
        ScopedDataFormatUpgrader data_format_upgrader(context, 5);
        ports_5d = GetVariadicNDFaninPorts(*context, *node, 5);
    }

    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }

    if (!ports_4d.empty())
    {
        TF_RETURN_IF_ERROR(
            UpdateFaninEdgesWithOp(context, ports_4d, node, kOpTranspose));
        TF_RETURN_IF_ERROR(
            UpdateFanoutEdgesWithOp(context, ports_4d, node, kOpTranspose));
    }

    if (!ports_5d.empty())
    {
        ScopedDataFormatUpgrader data_format_upgrader(context, 5);
        TF_RETURN_IF_ERROR(
            UpdateFaninEdgesWithOp(context, ports_5d, node, kOpTranspose));
        TF_RETURN_IF_ERROR(
            UpdateFanoutEdgesWithOp(context, ports_5d, node, kOpTranspose));
    }
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status AddNTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsAddN(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node) ||
        !IsAfterDstToSrcTransform(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
        context,
        GetDataFaninPorts(*node),
        node,
        kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradV2Transposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(
        IsMaxPoolGradV2(*node->node()) || IsMaxPoolGradGradV2(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {3, 4}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsMaxPoolGrad(*node->node()) || IsMaxPoolGradGradV1(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolV2Transposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsMaxPoolV2(*node->node()));
    // We check data_input's shape instead, because the shape inference of
    // MaxPoolV2 is not able to infer the shape when ksize or strides is not
    // constant.
    const auto& data_fanin = node->GetRegularFanin(0);
    auto* data_fanin_node = data_fanin.node_view();
    if (!ShouldProcess(*context, *node) ||
        !IsFanoutPortRankN(*data_fanin_node, data_fanin.index(), 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

bool FusedBatchNormGradTransposer::IsTraining(const MutableNodeView& node) const
{
    const auto* is_training_attr = node.GetAttr(kAttrIsTraining);
    if (is_training_attr != nullptr)
    {
        return is_training_attr->b();
    }
    return false;
}

Status FusedBatchNormGradTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsFusedBatchNormGrad(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }

    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status FusedBatchNormExTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsFusedBatchNormEx(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    if (node->NumRegularFanins() == 6)
    {
        TF_RETURN_IF_ERROR(
            UpdateFaninEdgesWithOp(context, {0, 5}, node, kOpTranspose));
    }
    else
    {
        TF_RETURN_IF_ERROR(
            UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    }
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsConv3DBackpropFilterV2(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 2}, node, kOpTranspose));
    // No need to update output shape, as it is always of shape
    // [filter_height, filter_width, in_channels, out_channels], regardless of
    // whether NCHW or NHWC is used.
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DBackpropInputTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsConv3DBackpropInputV2(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {2}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsConv3D(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 5)
    {
        return Status::OK();
    }
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    if (!ShouldProcess(*context, *node))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status BiasAddTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    // This TransposeNode allows for BiasAdd but not BiasAddV1, since BiasAdd
    // supports different data format.
    assert(IsBiasAddV2(*node->node()));
    const int rank = GetFanoutPortRank(*node, 0);
    if (rank != 4 && rank != 5)
    {
        return Status::OK();
    }
    if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, rank))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    // BiasAdd itself only needs NCHW/NHWC to determine whether C dim is the
    // second or the last dim. Therefore, we use the original 4D data format in
    // the context to update the node. For the input/output tensor, the
    // corresponding 4D or 5D data format is needed.
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    ScopedDataFormatUpgrader data_format_upgrader(context, rank);
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

Status AvgPoolGradTransposer::TransposeNode(
    TransposeContext* context,
    MutableNodeView* node)
{
    assert(IsAvgPoolGrad(*node->node()));
    if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 1, 4))
    {
        return Status::OK();
    }
    TF_VLog(
        3,
        "GenericLayoutOptimizer: transforming node '%s' with op '%s' from data "
        "format '%s' to '%s'",
        node->GetName().c_str(),
        node->GetOp().c_str(),
        context->src_format.c_str(),
        context->dst_format.c_str());
    TF_RETURN_IF_ERROR(UpdateNode(context, node));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
    return context->graph_view->GetMutationBuilder()->Apply();
}

} // namespace tfdml