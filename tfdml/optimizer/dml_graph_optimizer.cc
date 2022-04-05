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

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

static Status GraphDefToBuffer(const tensorflow::GraphDef& in, TF_Buffer* out)
{
    if (out->data != nullptr)
    {
        return errors::InvalidArgument(
            "Passing non-empty TF_Buffer is invalid.");
    }
    const size_t proto_size = in.ByteSizeLong();
    void* buf = malloc(proto_size);
    if (buf == nullptr)
    {
        return errors::ResourceExhausted(
            "Failed to allocate memory to serialize message of type '",
            in.GetTypeName(),
            "' and size ",
            proto_size);
    }
    if (!in.SerializeWithCachedSizesToArray(static_cast<uint8_t*>(buf)))
    {
        free(buf);
        return errors::InvalidArgument(
            "Unable to serialize ",
            in.GetTypeName(),
            " protocol buffer, perhaps the serialized size (",
            proto_size,
            " bytes) is too large?");
    }
    out->data = buf;
    out->length = proto_size;
    out->data_deallocator = [](void* data, size_t length) { free(data); };
    return Status::OK();
}

template <typename T>
static Status ParseBuffer(const TF_Buffer* in, T* out)
{
    if (in == nullptr || !out->ParseFromArray(in->data, in->length))
    {
        return errors::InvalidArgument(
            "Unparseable ",
            out->GetTypeName(),
            " proto");
    }
    return Status::OK();
}

static bool IsDefaultLayoutSensitiveOp(const tensorflow::NodeDef& node)
{
    const std::set<std::string> default_layout_sensitive_ops = {
        "AvgPool",
        "BiasAdd",
        "Conv2D",
        "DepthwiseConv2dNative",
        "DepthToSpace",
        "FusedBatchNorm",
        "FusedBatchNormV2",
        "FusedBatchNormV3",
        "FusedConv2DBiasActivation",
        "_FusedConv2D",
        "MaxPool",
        "SpaceToDepth"};
    return default_layout_sensitive_ops.find(node.op()) !=
           default_layout_sensitive_ops.end();
}

static bool IsAvgPoolGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "AvgPoolGrad";
}

static bool IsBiasAddGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "BiasAddGrad";
}

static bool IsConv2DBackpropFilter(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv2DBackpropFilter";
}

static bool IsConv2DBackpropInput(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv2DBackpropInput";
}

static bool IsDepthwiseConv2dNativeBackpropFilter(
    const tensorflow::NodeDef& node)
{
    return node.op() == "DepthwiseConv2dNativeBackpropFilter";
}

static bool IsDepthwiseConv2dNativeBackpropInput(
    const tensorflow::NodeDef& node)
{
    return node.op() == "DepthwiseConv2dNativeBackpropInput";
}

static bool IsFusedBatchNormEx(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormEx";
}

static bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormGrad";
}

static bool IsMaxPoolV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolV2";
}

static bool IsMaxPoolGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGrad";
}

static bool IsMaxPoolGradV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradV2";
}

static bool IsMaxPoolGradGradV1(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradGrad";
}

static bool IsMaxPoolGradGradV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradGradV2";
}

static bool IsLayoutSensitiveOp(const tensorflow::NodeDef& node)
{
    return IsDefaultLayoutSensitiveOp(node) || IsAvgPoolGrad(node) ||
           IsBiasAddGrad(node) || IsConv2DBackpropFilter(node) ||
           IsConv2DBackpropInput(node) ||
           IsDepthwiseConv2dNativeBackpropFilter(node) ||
           IsDepthwiseConv2dNativeBackpropInput(node) ||
           IsFusedBatchNormEx(node) || IsFusedBatchNormGrad(node) ||
           IsMaxPoolV2(node) || IsMaxPoolGrad(node) || IsMaxPoolGradV2(node) ||
           IsMaxPoolGradGradV1(node) || IsMaxPoolGradGradV2(node);
}

static bool AttrDataFormatMatch(
    const tensorflow::NodeDef& node,
    absl::string_view src_data_format)
{
    auto iter = node.attr().find("data_format");
    return iter != node.attr().end() && iter->second.s() == src_data_format;
}

static std::vector<tensorflow::OpInfo::TensorProperties> GetInputProperties(
    TF_GraphProperties* graph_props,
    const tensorflow::NodeDef& node,
    TF_Status* raw_status)
{
    int num_inputs;
    TF_GetInputPropertiesListSize(
        graph_props,
        node.op().c_str(),
        &num_inputs,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return {};
    }

    std::vector<TF_Buffer*> tensor_props_buffer(num_inputs);

    for (int i = 0; i < num_inputs; i++)
    {
        tensor_props_buffer[i] = TF_NewBuffer();
    }

    absl::Cleanup props_cleanup = [&tensor_props_buffer, num_inputs]
    {
        for (int i = 0; i < num_inputs; i++)
        {
            TF_DeleteBuffer(tensor_props_buffer[i]);
        }
    };

    TF_GetInputPropertiesList(
        graph_props,
        node.op().c_str(),
        tensor_props_buffer.data(),
        num_inputs,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return {};
    }

    std::vector<tensorflow::OpInfo::TensorProperties> input_properties;
    input_properties.reserve(num_inputs);

    for (int i = 0; i < num_inputs; ++i)
    {
        tensorflow::OpInfo::TensorProperties tensor_props;
        Status status = ParseBuffer(tensor_props_buffer[i], &tensor_props);

        if (!status.ok())
        {
            TF_SetStatus(raw_status, status.code(), status.error_message());
            return {};
        }

        input_properties.push_back(std::move(tensor_props));
    }

    TF_SetStatus(raw_status, TF_OK, "");
    return input_properties;
}

static std::vector<tensorflow::OpInfo::TensorProperties> GetOutputProperties(
    TF_GraphProperties* graph_props,
    const tensorflow::NodeDef& node,
    TF_Status* raw_status)
{
    int num_outputs;
    TF_GetOutputPropertiesListSize(
        graph_props,
        node.op().c_str(),
        &num_outputs,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return {};
    }

    std::vector<TF_Buffer*> tensor_props_buffer(num_outputs);

    for (int i = 0; i < num_outputs; i++)
    {
        tensor_props_buffer[i] = TF_NewBuffer();
    }

    absl::Cleanup props_cleanup = [&tensor_props_buffer, num_outputs]
    {
        for (int i = 0; i < num_outputs; i++)
        {
            TF_DeleteBuffer(tensor_props_buffer[i]);
        }
    };

    TF_GetOutputPropertiesList(
        graph_props,
        node.op().c_str(),
        tensor_props_buffer.data(),
        num_outputs,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return {};
    }

    std::vector<tensorflow::OpInfo::TensorProperties> output_properties;
    output_properties.reserve(num_outputs);

    for (int i = 0; i < num_outputs; ++i)
    {
        tensorflow::OpInfo::TensorProperties tensor_props;
        Status status = ParseBuffer(tensor_props_buffer[i], &tensor_props);

        if (!status.ok())
        {
            TF_SetStatus(raw_status, status.code(), status.error_message());
            return {};
        }

        output_properties.push_back(std::move(tensor_props));
    }

    TF_SetStatus(raw_status, TF_OK, "");
    return output_properties;
}

static bool IsFanoutPortRankN(tensorflow::NodeDef& node, int port, int n)
{
    auto output_shape_attr_iter = node.attr().find("_output_shapes");
    if (output_shape_attr_iter == node.attr().end() ||
        output_shape_attr_iter->second.list().shape_size() <= port)
    {
        return false;
    }

    const auto& shape = output_shape_attr_iter->second.list().shape(port);
    return !shape.unknown_rank() && shape.dim_size() == n;
}

template <typename T>
static Status PermuteSingle(
    absl::string_view location,
    absl::Span<const int> permutation,
    T* values)
{
    assert(values != nullptr);
    if (values->size() != permutation.size())
    {
        return errors::InvalidArgument(absl::StrCat(
            "Size of values ",
            values->size(),
            " does not match size of permutation ",
            permutation.size(),
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

template <typename T>
static Status PermuteDouble(
    absl::string_view location,
    absl::Span<const int> permutation,
    T* values)
{
    assert(values != nullptr);
    if (values->size() != permutation.size() * 2)
    {
        return errors::InvalidArgument(absl::StrCat(
            "Size of values ",
            values->size(),
            " does not match twice the size of permutation ",
            permutation.size(),
            " @ ",
            location));
    }
    typedef typename T::value_type V;
    std::vector<V> elements(values->begin(), values->end());
    for (int i = 0; i < values->size(); i = i + 2)
    {
        const int permutation_index = permutation[i / 2];
        (*values)[i] = elements[permutation_index * 2];
        (*values)[i + 1] = elements[permutation_index * 2 + 1];
    }
    return Status::OK();
}

static absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format)
{
    const int size = data_format.size();
    absl::flat_hash_map<char, int> index;
    index.reserve(size);
    for (int i = 0; i < size; i++)
    {
        index[data_format[i]] = i;
    }
    return index;
}

static std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format)
{
    // Generate permutation for transformation between src and dst format.
    // Example:
    // src = NWHC, dst = NCWH
    // index = { N:0 W:1 H:2 C:3 }
    // permutation = [0, 3, 1, 2]
    assert(src_dim_indices.size() == dst_format.size());
    std::vector<int> permutation;
    const int size = dst_format.size();
    permutation.reserve(size);
    for (int i = 0; i < size; i++)
    {
        permutation.push_back(src_dim_indices.at(dst_format[i]));
    }
    return permutation;
}

static void AddOrUpdateNodeAttr(
    tensorflow::NodeDef& node,
    const char* attr_name,
    const tensorflow::AttrValue& attr_value)
{
    auto attr_iter = node.mutable_attr()->find(attr_name);
    if (attr_iter == node.mutable_attr()->end())
    {
        node.mutable_attr()->insert({attr_name, attr_value});
    }
    else
    {
        attr_iter->second = attr_value;
    }
}

std::string GetFaninNameFormat(
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
        "DmlLayoutOptimizer");
}

std::string GetFanoutNameFormat(
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
        "DmlLayoutOptimizer");
}

static std::string AsControlDependency(const std::string& node_name)
{
    CHECK(!node_name.empty());
    return (!node_name.empty() && node_name[0] == '^')
               ? node_name
               : absl::StrCat("^", node_name);
}

void CreateConstPermNode(
    tensorflow::GraphDef& graph,
    absl::string_view node_name,
    absl::string_view device,
    absl::Span<const int> permutation,
    absl::string_view control_node_name)
{
    tensorflow::NodeDef* node = graph.add_node();
    node->set_name(std::string(node_name));
    node->set_op("Const");
    node->set_device(std::string(device));

    if (!control_node_name.empty())
    {
        node->add_input(std::string(control_node_name));
    }

    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(tensorflow::DT_INT32);
    node->mutable_attr()->insert({"dtype", attr_data_type});

    tensorflow::AttrValue attr_tensor;
    tensorflow::TensorProto tensor;
    tensor.set_dtype(tensorflow::DT_INT32);
    tensor.mutable_tensor_shape()->add_dim()->set_size(4);
    for (int i = 0; i < permutation.size(); i++)
    {
        tensor.mutable_tensor_content()->append(
            reinterpret_cast<const char*>(&permutation[i]),
            sizeof(int32_t));
    }
    node->mutable_attr()->insert({"value", attr_tensor});
}

static Status CreateTransposeNode(
    tensorflow::GraphDef& graph,
    absl::string_view name_format,
    tensorflow::DataType data_type,
    absl::string_view device,
    tensorflow::TensorShapeProto fanin_shape,
    absl::Span<const int> permutation,
    absl::string_view control_node_name,
    tensorflow::NodeDef* added_node,
    std::string* transpose_node_name)
{
    const std::string node_name = absl::Substitute(name_format, "Transpose");
    *transpose_node_name = node_name;

    added_node = graph.add_node();
    added_node->set_name(std::string(node_name));
    added_node->set_op("Transpose");
    added_node->set_device(std::string(device));

    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    added_node->mutable_attr()->insert({"T", attr_data_type});

    tensorflow::AttrValue attr_data_type_perm;
    attr_data_type_perm.set_type(tensorflow::DT_INT32);
    added_node->mutable_attr()->insert({"Tperm", attr_data_type_perm});

    if (!fanin_shape.unknown_rank())
    {
        TF_RETURN_IF_ERROR(PermuteSingle(
            absl::StrCat("fanin shape in", node_name),
            permutation,
            fanin_shape.mutable_dim()));
        tensorflow::AttrValue attr_output_shape;
        *attr_output_shape.mutable_list()->add_shape() = fanin_shape;
        added_node->mutable_attr()->insert(
            {"_output_shapes", attr_output_shape});
    }

    // Create Const Node
    const std::string const_perm_node_name =
        absl::Substitute(name_format, "PermConst");
    CreateConstPermNode(
        graph,
        const_perm_node_name,
        device,
        permutation,
        control_node_name);
    // Add place holder for 1st input.
    added_node->add_input("");
    // Connect const_perm_node to 2nd input of transpose_node.
    added_node->add_input(const_perm_node_name);
    return Status::OK();
}

static Status UpdateEdge(
    tensorflow::GraphDef& graph,
    absl::string_view name_format,
    absl::string_view op,
    const tensorflow::AttrValue* input_shape,
    bool is_in_frame,
    bool is_src_format_to_dst_format,
    int src_port,
    int dst_port,
    tensorflow::NodeDef& src_node,
    tensorflow::NodeDef& dst_node,
    tensorflow::DataType data_type,
    const std::vector<int>& permutation)
{
    tensorflow::NodeDef* added_node = nullptr;
    std::string added_node_name;
    if (op == "Transpose")
    {
        tensorflow::TensorShapeProto input_shape_proto;
        input_shape_proto.set_unknown_rank(true);
        if (input_shape != nullptr)
        {
            input_shape_proto = input_shape->list().shape(src_port);
        }
        else
        {
            auto src_node_shape_attr_iter =
                src_node.attr().find("_output_shapes");
            if (src_node_shape_attr_iter != src_node.attr().end())
            {
                input_shape_proto =
                    src_node_shape_attr_iter->second.list().shape(src_port);
            }
        }

        const std::string control_node_name =
            is_in_frame ? AsControlDependency(src_node.name()) : "";

        TF_RETURN_IF_ERROR(CreateTransposeNode(
            graph,
            name_format,
            data_type,
            "DML",
            input_shape_proto,
            permutation,
            control_node_name,
            added_node,
            &added_node_name));
    }
    else if (op == "DataFormatVecPermute" || op == "DataFormatDimMap")
    {
        // TODO: Support DataFormatVecPermute and DataFormatDimMap
        return errors::InvalidArgument(absl::StrCat(
            "Unsupported op \"",
            op,
            "\". Supported ops are Transpose."));
    }
    else
    {
        return errors::InvalidArgument(absl::StrCat(
            "Unsupported op \"",
            op,
            "\". Supported ops are Transpose"));
    }

    // Connect src_node to 1st input of added_node.
    added_node->set_input(0, src_node.name());

    // Connect output of added_node to dst_node:dst_port.
    dst_node.set_input(dst_port, added_node->name());

    return Status::OK();
}

Status UpdateFaninEdgesWithOp(
    tensorflow::GraphDef& graph,
    const std::unordered_map<std::string, tensorflow::NodeDef*>& nodes,
    absl::Span<const tensorflow::OpInfo::TensorProperties> dst_node_input_props,
    absl::Span<const int> dst_ports,
    tensorflow::NodeDef& dst_node,
    absl::string_view op,
    absl::string_view src_format,
    absl::string_view dst_format,
    const std::vector<int>& src_to_dst)
{
    const bool is_in_frame = true;
    for (int dst_port : dst_ports)
    {
        const std::string& input_node_name = dst_node.input(dst_port);
        tensorflow::NodeDef* src_node = nodes.at(input_node_name);
        const int src_port = 0;

        tensorflow::DataType data_type = dst_node_input_props[dst_port].dtype();

        TF_RETURN_IF_ERROR(UpdateEdge(
            graph,
            GetFaninNameFormat(
                dst_node.name(),
                dst_port,
                src_format,
                dst_format),
            op,
            /*input_shape=*/nullptr,
            /*is_in_frame=*/is_in_frame,
            /*is_src_format_to_dst_format=*/true,
            src_port,
            dst_port,
            *src_node,
            dst_node,
            data_type,
            src_to_dst));
    }
    return Status::OK();
}

void OptimizeGraph(
    void* optimizer,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status)
{
    int num_preserved_nodes;
    size_t preserved_nodes_size;

    TF_GraphProperties* graph_props = TF_NewGraphProperties(grappler_item);
    absl::Cleanup props_cleanup = [graph_props]
    { TF_DeleteGraphProperties(graph_props); };

    TF_GetNodesToPreserveListSize(
        grappler_item,
        &num_preserved_nodes,
        &preserved_nodes_size,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return;
    }

    std::vector<char*> preserved_node_names(num_preserved_nodes);
    std::vector<size_t> preserved_node_name_lengths(num_preserved_nodes);
    std::vector<char> preserved_node_name_storage(preserved_nodes_size);

    TF_GetNodesToPreserveList(
        grappler_item,
        preserved_node_names.data(),
        preserved_node_name_lengths.data(),
        num_preserved_nodes,
        preserved_node_name_storage.data(),
        preserved_node_name_storage.size(),
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return;
    }

    std::set<std::string> preserved_nodes(
        preserved_node_names.begin(),
        preserved_node_names.end());

    tensorflow::GraphDef input_graph_def;
    Status status = ParseBuffer(input_graph_buffer, &input_graph_def);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    constexpr char* src_format = "NHWC";
    constexpr char* dst_format = "NCHW";
    const auto src_dim_indices = GetDimensionIndices(src_format);
    const auto dst_dim_indices = GetDimensionIndices(dst_format);
    const auto src_to_dst = GetPermutation(src_dim_indices, dst_format);
    const auto dst_to_src = GetPermutation(dst_dim_indices, src_format);

    const int input_graph_size = input_graph_def.node_size();

    std::unordered_map<std::string, tensorflow::NodeDef*> nodes;

    for (int i = 0; i < input_graph_size; ++i)
    {
        tensorflow::NodeDef* node = input_graph_def.mutable_node(i);
        nodes[node->name()] = node;
    }

    for (int i = 0; i < input_graph_size; ++i)
    {
        tensorflow::NodeDef* node = input_graph_def.mutable_node(i);

        if (node->device() != "DML")
        {
            continue;
        }

        if (preserved_nodes.find(node->op()) != preserved_nodes.end())
        {
            continue;
        }

        const bool data_format_match = !IsLayoutSensitiveOp(*node) ||
                                       AttrDataFormatMatch(*node, src_format);

        if (data_format_match)
        {
            if (IsDefaultLayoutSensitiveOp(*node))
            {
                auto input_props =
                    GetInputProperties(graph_props, *node, raw_status);

                if (TF_GetCode(raw_status) != TF_OK)
                {
                    return;
                }

                auto output_props =
                    GetOutputProperties(graph_props, *node, raw_status);

                if (TF_GetCode(raw_status) != TF_OK)
                {
                    return;
                }

                if (!IsFanoutPortRankN(*node, 0, 4))
                {
                    continue;
                }

                printf(
                    "**********TRANSPOSING OPERATOR: %s\n",
                    node->op().c_str());

                tensorflow::AttrValue data_format_attr;
                data_format_attr.set_s(dst_format);
                AddOrUpdateNodeAttr(*node, "data_format", data_format_attr);

                auto permute_attr = [&node, &src_to_dst](const char* attr_name)
                {
                    auto iter = node->attr().find(attr_name);

                    if (iter != node->attr().end())
                    {
                        tensorflow::AttrValue attr_copy(iter->second);
                        TF_RETURN_IF_ERROR(PermuteSingle(
                            absl::StrCat(
                                attr_name,
                                " attribute in ",
                                node->name()),
                            src_to_dst,
                            attr_copy.mutable_list()->mutable_i()));

                        AddOrUpdateNodeAttr(*node, attr_name, attr_copy);
                    }

                    return Status::OK();
                };

                status = permute_attr("strides");
                if (!status.ok())
                {
                    TF_SetStatus(
                        raw_status,
                        status.code(),
                        status.error_message());
                    return;
                }

                status = permute_attr("ksize");
                if (!status.ok())
                {
                    TF_SetStatus(
                        raw_status,
                        status.code(),
                        status.error_message());
                    return;
                }

                status = permute_attr("dilations");
                if (!status.ok())
                {
                    TF_SetStatus(
                        raw_status,
                        status.code(),
                        status.error_message());
                    return;
                }

                auto explicit_paddings_iter =
                    node->attr().find("explicit_paddings");

                if (explicit_paddings_iter != node->attr().end() &&

                    explicit_paddings_iter->second.has_list() &&
                    explicit_paddings_iter->second.list().i_size() > 0)
                {
                    tensorflow::AttrValue explicit_paddings_attr_copy(
                        explicit_paddings_iter->second);
                    status = PermuteDouble(
                        absl::StrCat(
                            "explicit_paddings attribute in ",
                            node->name()),
                        src_to_dst,
                        explicit_paddings_attr_copy.mutable_list()
                            ->mutable_i());

                    if (!status.ok())
                    {
                        TF_SetStatus(
                            raw_status,
                            status.code(),
                            status.error_message());
                        return;
                    }

                    AddOrUpdateNodeAttr(
                        *node,
                        "explicit_paddings",
                        explicit_paddings_attr_copy);
                }
            }
            else
            {
                // TODO: Handle other cases here
            }
        }
    }

    status = GraphDefToBuffer(input_graph_def, output_graph_buffer);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    TF_SetStatus(raw_status, TF_OK, "");
}
} // namespace tfdml
