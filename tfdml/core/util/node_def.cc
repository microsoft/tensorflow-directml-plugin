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

#include "tfdml/core/util/node_def.h"

namespace tfdml
{

NodeDef NodeDef::Create(
    TF_OpKernelConstruction* ctx,
    std::string_view op_type_name,
    absl::Span<const ArgumentDesc> input_arg_descs,
    absl::Span<const ArgumentDesc> output_arg_descs,
    absl::Span<const AttributeDesc> attribute_descs)
{
    TF_StringView name = TF_OpKernelConstruction_GetName(ctx);

    NodeDef node = {};
    node.op_name = std::string_view{name.data, name.len};
    node.op_type_string = op_type_name;

    // uint32_t arg_index = 0;
    // uint32_t tensor_index = 0;

    // uint32_t tensor_offsets[Op::argument_descs.size()];
    // uint32_t tensor_counts[Op::argument_descs.size()];

    // // Init all input tensors to DEVICE_MEMORY
    // for (; arg_index < Op::input_arg_count; arg_index++)
    // {
    //     uint32_t arg_tensor_count = 1; // TODO: unless attr based
    //     tensor_counts[arg_index] = arg_tensor_count;
    //     tensor_offsets[arg_index] = tensor_index;
    //     tensor_index += arg_tensor_count;
    // }
    // node.input_tensor_memory_types.resize(
    //     tensor_index,
    //     MemoryType::DEVICE_MEMORY);

    // // Init all output tensors to DEVICE_MEMORY
    // for (; arg_index < Op::input_arg_count + Op::output_arg_count;
    //         arg_index++)
    // {
    //     uint32_t arg_tensor_count = 1; // TODO: unless attr based
    //     tensor_offsets[arg_index] = tensor_index;
    //     tensor_counts[arg_index] = arg_tensor_count;
    //     tensor_index += arg_tensor_count;
    // }
    // node.output_tensor_memory_types.resize(
    //     tensor_index - node.input_tensor_memory_types.size(),
    //     MemoryType::DEVICE_MEMORY);

    // // Set input/output tensors that live in HOST_MEMORY
    // for (auto arg : std::initializer_list<Op::Argument>{HostArguments...})
    // {
    //     auto memory_types_start =
    //         (GetArgumentType<Op>(arg) == ArgumentType::Input)
    //             ? node.input_tensor_memory_types.begin()
    //             : node.output_tensor_memory_types.begin();

    //     uint32_t host_arg_index = ConvertOpDefEnumToIndex(arg);
    //     std::fill_n(
    //         memory_types_start + tensor_offsets[host_arg_index],
    //         tensor_counts[host_arg_index],
    //         MemoryType::HOST_MEMORY);
    // }

    return node;
}

} // namespace tfdml
