/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/optimizer/node_utils.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tfdml/optimizer/device_name_utils.h"
#include "tfdml/optimizer/device_type.h"
#include "tfdml/optimizer/op_registry.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

enum
{
    kDataTypeRefOffset = 100
};

static inline bool IsRefType(tensorflow::DataType dtype)
{
    return dtype > static_cast<tensorflow::DataType>(kDataTypeRefOffset);
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

bool IsHostMemory(const tensorflow::NodeDef& node, int output_port)
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

} // namespace tfdml