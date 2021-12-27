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

#pragma once

#include "absl/types/span.h"

namespace tfdml
{

enum class ArgumentType
{
    Input,
    Output
};

struct ArgumentDesc
{
    enum class TensorCount
    {
        Single,           // argument maps to a single tensor
        SequenceAttrInt,  // argument maps to a list of tensors with the same
                          // type
        SequenceAttrList, // argument maps to a list of tensors with different
                          // types
    };

    const char* name;
    TensorCount tensor_count;
    const char* sequence_attr_name = nullptr;
};

enum class AttributeType
{
    Type,
    Int,
    Float,
    Bool,
    String,
    Shape,
    Func,
    Tensor,
    ListType,
    ListInt,
    ListFloat,
    ListBool,
    ListString,
    ListShape,
    ListFunc,
    ListTensor,
};

struct AttributeDesc
{
    const char* name;
    AttributeType type;
};

// Helper to safely convert an OpDef::Argument or OpDef::Attribute to an integer
// index into their respective arrays. All OpDef enum classes must have an
// underlying type of int, but this helper is preferred over static_cast in case
// the codegen changes or a handwritten custom DML op def uses the wrong type by
// mistake.
template <typename E>
constexpr int ConvertOpDefEnumToIndex(E enum_value)
{
    auto index = std::underlying_type_t<E>(enum_value);
    static_assert(
        std::is_same_v<decltype(index), int>,
        "OpDef enums should have an underlying type of int");
    return index;
}

// Helper to fetch an operator's argument desc by enum value.
template <typename OpDef>
constexpr const ArgumentDesc& GetArgumentDesc(typename OpDef::Argument arg)
{
    return OpDef::argument_descs[ConvertOpDefEnumToIndex(arg)];
}

// Helper to fetch an operator's attribute desc by enum value.
template <typename OpDef>
constexpr const AttributeDesc& GetAttributeDesc(typename OpDef::Attribute attr)
{
    return OpDef::attribute_descs[ConvertOpDefEnumToIndex(attr)];
}

// Op definitions list input args followed by output args.
template <typename OpDef>
constexpr ArgumentType GetArgumentType(typename OpDef::Argument arg)
{
    auto arg_index = ConvertOpDefEnumToIndex(arg);
    return arg_index < OpDef::input_arg_count ? ArgumentType::Input
                                              : ArgumentType::Output;
}

template <typename OpDef>
constexpr absl::Span<const ArgumentDesc> GetInputArgumentDescs()
{
    static_assert(OpDef::argument_descs.size() >= OpDef::input_arg_count);
    return {OpDef::argument_descs.data(), OpDef::input_arg_count};
}

template <typename OpDef>
constexpr absl::Span<const ArgumentDesc> GetOutputArgumentDescs()
{
    static_assert(
        OpDef::argument_descs.size() ==
        OpDef::input_arg_count + OpDef::output_arg_count);
    return {
        OpDef::argument_descs.data() + OpDef::input_arg_count,
        OpDef::output_arg_count};
}

} // namespace tfdml

// clang-format off
#include <array>
#include "op_defs_core.h"
#include "op_defs_dml.h"
// clang-format on