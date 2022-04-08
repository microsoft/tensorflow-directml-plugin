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

#include "tfdml/optimizer/transposer.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/op_registry.h"
#include "tfdml/optimizer/op_types.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/optimizer/scoped_data_format_upgrader.h"
#include "tfdml/optimizer/transpose_context.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{
constexpr char kAttrDataFormat[] = "data_format";
constexpr char kAttrKSize[] = "ksize";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrDilations[] = "dilations";
constexpr char kAttrExplicitPaddings[] = "explicit_paddings";
constexpr char kOpTranspose[] = "Transpose";
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

// Permutes two elements at a time according to permutation and replaces the
// original values. Values must be twice the size of permutation.
template <typename T>
Status PermuteDouble(
    absl::string_view location,
    absl::Span<const int> permutation,
    T* values)
{
    assert(values != nullptr);
    int permutation_size = permutation.size();
    if (values->size() != permutation_size * 2)
    {
        return errors::InvalidArgument(absl::StrCat(
            "Size of values ",
            values->size(),
            " does not match twice the size of permutation ",
            permutation_size,
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

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType
{
  public:
    DeviceType(const char* type) // NOLINT(runtime/explicit)
        : type_(type)
    {
    }

    explicit DeviceType(absl::string_view type)
        : type_(type.data(), type.size())
    {
    }

    const char* type() const { return type_.c_str(); }
    const std::string& type_string() const { return type_; }

    bool operator<(const DeviceType& other) const
    {
        return type_ < other.type_;
    }
    bool operator==(const DeviceType& other) const
    {
        return type_ == other.type_;
    };
    bool operator!=(const DeviceType& other) const { return !(*this == other); }

  private:
    std::string type_;
};

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

    // Set up parameters of node.
    node.set_op(std::string(op));
    node.set_device(std::string(device));
    tensorflow::AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    node.mutable_attr()->insert({"T", attr_data_type});

    // The inputs of a DataFormat op could be in host memory for ops such as
    // Reshape. In such cases, run the kernel on the host too.
    if (is_fanin_on_host)
    {
        tensorflow::AttrValue attr_kernel;
        attr_kernel.set_s("host");
        node.mutable_attr()->insert({"_kernel", attr_kernel});
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

static const std::string& GetKernelLabelAttr(
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& node_attrs)
{
    static const std::string& kKernelAttr = *new std::string("_kernel");
    static const std::string& kEmptyString = *new std::string("");

    // NOTE: We inline the implementation of `GetNodeAttrString()` here in order
    // to use the `AttrSlice::FindByString()` overload, which does a more
    // efficient map lookup (instead of a linear scan) when the attribute name
    // is already a `const string&`.
    auto attr_value_iter = node_attrs.find(kKernelAttr);
    if (attr_value_iter == node_attrs.end() ||
        attr_value_iter->second.value_case() != tensorflow::AttrValue::kS)
        return kEmptyString;
    else
        return attr_value_iter->second.s();
}

enum
{
    kDataTypeRefOffset = 100
};
inline bool IsRefType(tensorflow::DataType dtype)
{
    return dtype > static_cast<tensorflow::DataType>(kDataTypeRefOffset);
}

static Status AttrValueHasType(
    const tensorflow::AttrValue& attr_value,
    absl::string_view type)
{
    int num_set = 0;

#define VALIDATE_FIELD(name, type_string, oneof_case)                          \
    do                                                                         \
    {                                                                          \
        if (attr_value.has_list())                                             \
        {                                                                      \
            if (attr_value.list().name##_size() > 0)                           \
            {                                                                  \
                if (type != "list(" type_string ")")                           \
                {                                                              \
                    return errors::InvalidArgument(                            \
                        "AttrValue had value with type 'list(" type_string     \
                        ")' when '",                                           \
                        type,                                                  \
                        "' expected");                                         \
                }                                                              \
                ++num_set;                                                     \
            }                                                                  \
        }                                                                      \
        else if (attr_value.value_case() == tensorflow::AttrValue::oneof_case) \
        {                                                                      \
            if (type != type_string)                                           \
            {                                                                  \
                return errors::InvalidArgument(                                \
                    "AttrValue had value with type '" type_string "' when '",  \
                    type,                                                      \
                    "' expected");                                             \
            }                                                                  \
            ++num_set;                                                         \
        }                                                                      \
    } while (false)

    VALIDATE_FIELD(s, "string", kS);
    VALIDATE_FIELD(i, "int", kI);
    VALIDATE_FIELD(f, "float", kF);
    VALIDATE_FIELD(b, "bool", kB);
    VALIDATE_FIELD(type, "type", kType);
    VALIDATE_FIELD(shape, "shape", kShape);
    VALIDATE_FIELD(tensor, "tensor", kTensor);
    VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

    if (attr_value.value_case() == tensorflow::AttrValue::kPlaceholder)
    {
        return errors::InvalidArgument(
            "AttrValue had value with unexpected type 'placeholder'");
    }

    // If the attr type is 'list', we expect attr_value.has_list() to be
    // true.  However, proto3's attr_value.has_list() can be false when
    // set to an empty list for GraphDef versions <= 4. So we simply
    // check if has_list is false and some other field in attr_value is
    // set to flag the error.  This test can be made more strict once
    // support for GraphDef versions <= 4 is dropped.
    if (absl::StartsWith(type, "list(") && !attr_value.has_list())
    {
        if (num_set)
        {
            return errors::InvalidArgument(
                "AttrValue missing value with expected type '",
                type,
                "'");
        }
        else
        {
            // Indicate that we have a list, but an empty one.
            ++num_set;
        }
    }

    // Okay to have an empty list, but not to be missing a non-list value.
    if (num_set == 0 && !absl::StartsWith(type, "list("))
    {
        return errors::InvalidArgument(
            "AttrValue missing value with expected type '",
            type,
            "'");
    }

    // Ref types and DT_INVALID are illegal, and DataTypes must
    // be a valid enum type.
    if (type == "type")
    {
        if (!tensorflow::DataType_IsValid(attr_value.type()))
        {
            return errors::InvalidArgument(
                "AttrValue has invalid DataType enum: ",
                attr_value.type());
        }
        if (IsRefType(attr_value.type()))
        {
            return errors::InvalidArgument(
                "AttrValue must not have reference type value of ",
                tensorflow::DataType_Name(attr_value.type()));
        }
        if (attr_value.type() == tensorflow::DT_INVALID)
        {
            return errors::InvalidArgument("AttrValue has invalid DataType");
        }
    }
    else if (type == "list(type)")
    {
        for (auto as_int : attr_value.list().type())
        {
            const tensorflow::DataType dtype =
                static_cast<tensorflow::DataType>(as_int);
            if (!tensorflow::DataType_IsValid(dtype))
            {
                return errors::InvalidArgument(
                    "AttrValue has invalid DataType enum: ",
                    as_int);
            }
            if (IsRefType(dtype))
            {
                return errors::InvalidArgument(
                    "AttrValue must not have reference type value of ",
                    tensorflow::DataType_Name(dtype));
            }
            if (dtype == tensorflow::DT_INVALID)
            {
                return errors::InvalidArgument(
                    "AttrValue contains invalid DataType");
            }
        }
    }

    return Status::OK();
}

static bool InTypeList(
    tensorflow::DataType dt,
    const tensorflow::AttrValue& type_list)
{
    for (int in_list : type_list.list().type())
    {
        if (dt == in_list) return true;
    }
    return false;
}

static Status KernelAttrsMatch(
    const tensorflow::KernelDef& kernel_def,
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attrs,
    bool* match)
{
    *match = false;
    for (const auto& constraint : kernel_def.constraint())
    {
        auto constraint_value_case = tensorflow::AttrValue::VALUE_NOT_SET;
        int value_type_num = 0;
        if (constraint.allowed_values().list().type_size() > 0)
        {
            constraint_value_case = tensorflow::AttrValue::kType;
            value_type_num++;
        }
        if (constraint.allowed_values().list().s_size() > 0)
        {
            constraint_value_case = tensorflow::AttrValue::kS;
            value_type_num++;
        }
        if (constraint.allowed_values().list().i_size() > 0)
        {
            constraint_value_case = tensorflow::AttrValue::kI;
            value_type_num++;
        }
        if (constraint.allowed_values().list().b_size() > 0)
        {
            constraint_value_case = tensorflow::AttrValue::kB;
            value_type_num++;
        }

        if (value_type_num == 0)
        {
            return errors::Unimplemented(
                "KernelDef '",
                kernel_def.ShortDebugString(),
                " has constraint on attr '",
                constraint.name(),
                "' with unsupported type: ",
                constraint.allowed_values().DebugString());
        }
        if (value_type_num > 1)
        {
            return errors::InvalidArgument(
                "KernelDef '",
                kernel_def.ShortDebugString(),
                " has constraint on attr '",
                constraint.name(),
                "' with more than one value type: ",
                constraint.allowed_values().DebugString());
        }

        auto attr_value_iter = attrs.find(constraint.name());

        if (attr_value_iter == attrs.end())
        {
            return errors::InvalidArgument(
                "OpKernel '",
                kernel_def.op(),
                "' has constraint on attr '",
                constraint.name(),
                "' not in NodeDef. KernelDef: '",
                kernel_def.ShortDebugString(),
                "'");
        }

#define RETURN_IF_ATTR_NOT_FOUND(n, oneof_case, type_str)                      \
    do                                                                         \
    {                                                                          \
        if (constraint_value_case == tensorflow::AttrValue::oneof_case)        \
        {                                                                      \
            Status s = AttrValueHasType(attr_value_iter->second, type_str);    \
            if (!s.ok())                                                       \
            {                                                                  \
                return errors::InvalidArgument(                                \
                    "KernelDef '",                                             \
                    kernel_def.ShortDebugString(),                             \
                    "' has constraint on attr '",                              \
                    constraint.name(),                                         \
                    "' that has value '",                                      \
                    attr_value_iter->second.DebugString(),                     \
                    "' that does not have the same type in NodeDef.");         \
            }                                                                  \
            bool found = false;                                                \
            for (auto& value : constraint.allowed_values().list().n())         \
            {                                                                  \
                if (value == attr_value_iter->second.n())                      \
                {                                                              \
                    found = true;                                              \
                    break;                                                     \
                }                                                              \
            }                                                                  \
            if (!found)                                                        \
            {                                                                  \
                return Status::OK();                                           \
            }                                                                  \
        }                                                                      \
    } while (false)

        RETURN_IF_ATTR_NOT_FOUND(s, kS, "string");
        RETURN_IF_ATTR_NOT_FOUND(i, kI, "int");
        RETURN_IF_ATTR_NOT_FOUND(b, kB, "bool");

#undef RETURN_IF_ATTR_NOT_FOUND

        if (constraint_value_case != tensorflow::AttrValue::kType)
        {
            continue;
        }

        if (attr_value_iter->second.type() != tensorflow::DT_INVALID)
        {
            if (!InTypeList(
                    attr_value_iter->second.type(),
                    constraint.allowed_values()))
            {
                return Status::OK();
            }
        }
        else
        {
            if (!AttrValueHasType(attr_value_iter->second, "list(type)").ok())
            {
                return errors::InvalidArgument(
                    "KernelDef '",
                    kernel_def.ShortDebugString(),
                    "' has constraint on attr '",
                    constraint.name(),
                    "' that has value '",
                    attr_value_iter->second.DebugString(),
                    "' that does not have type 'type' or 'list(type)' in "
                    "NodeDef.");
            }

            for (int t : attr_value_iter->second.list().type())
            {
                if (!InTypeList(
                        static_cast<tensorflow::DataType>(t),
                        constraint.allowed_values()))
                {
                    return Status::OK();
                }
            }
        }
    }
    *match = true;
    return Status::OK();
}

static Status FindKernelRegistration(
    const DeviceType& device_type,
    absl::string_view node_name,
    bool has_experimental_debug_info,
    const tensorflow::NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    absl::string_view node_op,
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& node_attrs,
    tensorflow::KernelDef* def,
    bool* was_attr_mismatch)
{
    *was_attr_mismatch = false;

    const std::string& label = GetKernelLabelAttr(node_attrs);

    Status status;
    TF_Buffer* kernel_list_buffer = TF_GetRegisteredKernelsForOp(
        std::string(node_op).c_str(),
        status.raw());
    TF_RETURN_IF_ERROR(status);

    tensorflow::KernelList kernel_list;
    TF_RETURN_IF_ERROR(ParseBuffer(kernel_list_buffer, &kernel_list));

    const tensorflow::KernelDef* prev_kernel_def = nullptr;

    for (int i = 0; i < kernel_list.kernel_size(); ++i)
    {
        const tensorflow::KernelDef& kernel_def = kernel_list.kernel(i);
        if (kernel_def.device_type() != device_type.type_string())
        {
            continue;
        }

        if (kernel_def.label() != label)
        {
            continue;
        }

        // If there is a kernel registered for the op and device_type,
        // check that the attrs match.
        bool match;
        TF_RETURN_IF_ERROR(KernelAttrsMatch(kernel_def, node_attrs, &match));
        if (match)
        {
            if (prev_kernel_def != nullptr)
            {
                if (prev_kernel_def->priority() == kernel_def.priority())
                {
                    return errors::InvalidArgument(
                        "Multiple OpKernel registrations match NodeDef at the "
                        "same "
                        "priority '",
                        node_name,
                        "': '",
                        prev_kernel_def->ShortDebugString(),
                        "' and '",
                        kernel_def.ShortDebugString(),
                        "'");
                }
                else if (prev_kernel_def->priority() > kernel_def.priority())
                {
                    continue;
                }
            }
            prev_kernel_def = &kernel_list.kernel(i);
        }
        else
        {
            *was_attr_mismatch = true;
        }
    }

    *def = *prev_kernel_def;

    return Status::OK();
}

static Status FindKernelDef(
    const DeviceType& device_type,
    absl::string_view node_name,
    bool has_experimental_debug_info,
    const tensorflow::NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    absl::string_view node_op,
    absl::string_view node_device,
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& node_attrs,
    tensorflow::KernelDef* def)
{
    def = nullptr;

    tensorflow::KernelDef kernel_def;
    bool was_attr_mismatch;
    TF_RETURN_IF_ERROR(FindKernelRegistration(
        device_type,
        node_name,
        has_experimental_debug_info,
        experimental_debug_info,
        node_op,
        node_attrs,
        def,
        &was_attr_mismatch));
    if (!def)
    {
        const std::string& device_str = device_type.type_string();
        if (was_attr_mismatch)
        {
            return errors::NotFound(
                "No registered '",
                node_op,
                "' OpKernel for ",
                device_str,
                " devices compatible with node ",
                node_name,
                " (OpKernel was found, but attributes didn't match) ",
                "Requested Attributes for ",
                node_name);
        }

        return errors::NotFound(
            "No registered '",
            node_op,
            "' OpKernel for ",
            device_str,
            " devices compatible with node ",
            node_name);
    }

    return Status::OK();
}

static Status FindKernelDef(
    const DeviceType& device_type,
    const tensorflow::NodeDef& node_def,
    tensorflow::KernelDef* def)
{
    return FindKernelDef(
        device_type,
        node_def.name(),
        node_def.has_experimental_debug_info(),
        node_def.experimental_debug_info(),
        node_def.op(),
        node_def.device(),
        node_def.attr(),
        def);
}

static bool IsHostMemory(const tensorflow::NodeDef& node, int output_port)
{
    DeviceNameUtils::ParsedName parsed_name;
    if (DeviceNameUtils::ParseFullName(node.device(), &parsed_name))
    {
        DeviceType device_type(parsed_name.type);
        tensorflow::KernelDef kernel_def;
        Status s = FindKernelDef(device_type, node, &kernel_def);

        if (!s.ok())
        {
            return true;
        }

        tensorflow::OpDef op_def;
        s = OpRegistry::Instance().LookUpOpDef(node.op().c_str(), &op_def);

        if (!s.ok() || output_port >= op_def.output_arg_size())
        {
            return true;
        }

        for (int i = 0; i < kernel_def.host_memory_arg_size(); ++i)
        {
            if (kernel_def.host_memory_arg(i) ==
                op_def.output_arg(output_port).name())
            {
                return true;
            }
        }
    }
    return false;
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
    std::string tensor_bytes(permutation.size() * DataTypeSize(TF_INT32), '\0');

    for (int i = 0; i < permutation.size(); i++)
    {
        reinterpret_cast<int32_t*>(tensor_bytes.data())[i] = permutation[i];
    }

    *tensor->mutable_tensor_content() = std::move(tensor_bytes);
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

    // TODO(lyandy): Minimize device parsing/fetching.
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

} // namespace tfdml