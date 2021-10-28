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

struct AttributeDesc
{
    const char* name;
};

// Helper to fetch an operator's argument desc by enum value.
template <typename OpDef>
constexpr const ArgumentDesc& GetArgumentDesc(typename OpDef::Argument arg)
{
    auto arg_index = std::underlying_type_t<typename OpDef::Argument>(arg);
    static_assert(
        std::is_same_v<decltype(arg_index), int>,
        "OpDef::Argument enum should have an underlying type of int");

    return OpDef::argument_descs[arg_index];
}

// Helper to fetch an operator's argument desc by enum value.
template <typename OpDef>
constexpr const AttributeDesc& GetAttributeDesc(typename OpDef::Attribute attr)
{
    auto attr_index = std::underlying_type_t<typename OpDef::Attribute>(attr);
    static_assert(
        std::is_same_v<decltype(attr_index), int>,
        "OpDef::Argument enum should have an underlying type of int");

    return OpDef::attribute_descs[attr_index];
}

// Op definitions list input args followed by output args.
template <typename OpDef>
constexpr uint32_t GetArgumentType(typename OpDef::Argument arg)
{
    auto arg_index = std::underlying_type_t<typename OpDef::Argument>(arg);
    static_assert(
        std::is_same_v<decltype(arg_index), int>,
        "Op::Argument enum should have an underlying type of int");

    return arg_index < OpDef::input_arg_count ? ArgumentType::Input
                                              : ArgumentType::Output;
}

} // namespace tfdml

#include "op_defs_core.h"
#include "op_defs_dml.h"