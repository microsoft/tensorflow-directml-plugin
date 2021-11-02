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

#include "absl/container/inlined_vector.h"
#include "tensorflow/c/kernels.h"
#include "tfdml/core/util/op_defs.h"
#include "tfdml/core/util/types.h"

namespace tfdml
{

class NodeDef
{
  public:
    inline std::string_view GetOpName() const { return op_name; }

    inline std::string_view GetOpTypeName() const { return op_type_name; }

    inline MemoryType GetInputTensorMemoryType(
        uint32_t input_tensor_index) const
    {
        return tensor_memory_types[input_tensor_index];
    }

    inline MemoryType GetOutputTensorMemoryType(
        uint32_t output_tensor_index) const
    {
        return tensor_memory_types[output_tensor_index + input_tensor_count];
    }

    // Combines data from an OpDef with runtime info to build a NodeDef.
    template <typename Op, typename Op::Argument... HostArguments>
    static NodeDef Create(TF_OpKernelConstruction* ctx)
    {
        TF_StringView name = TF_OpKernelConstruction_GetName(ctx);

        NodeDef node = {};
        node.op_name = std::string_view{name.data, name.len};
        node.op_type_name = Op::name;

        // Calculate mapping from arguments to tensors. This has to be done at
        // runtime because some args have a variable number of tensors specified
        // by an attribute value.
        uint32_t total_tensor_count = 0;
        std::array<uint32_t, Op::argument_descs.size()> arg_tensor_offsets = {};
        std::array<uint32_t, Op::argument_descs.size()> arg_tensor_counts = {};

        for (uint32_t arg_index = 0; arg_index < Op::argument_descs.size();
             arg_index++)
        {
            uint32_t arg_tensor_count = 1; // TODO: unless attr based
            arg_tensor_counts[arg_index] = arg_tensor_count;
            arg_tensor_offsets[arg_index] = total_tensor_count;
            total_tensor_count += arg_tensor_count;
            if (arg_index < Op::input_arg_count)
            {
                node.input_tensor_count += arg_tensor_count;
            }
        }

        // Init all tensors to DEVICE_MEMORY.
        node.tensor_memory_types.resize(
            total_tensor_count,
            MemoryType::DEVICE_MEMORY);

        // Set tensors belonging to arguments in HostArguments to
        // HOST_MEMORY.
        for (auto arg : std::initializer_list<Op::Argument>{HostArguments...})
        {
            auto arg_index = ConvertOpDefEnumToIndex(arg);
            std::fill_n(
                node.tensor_memory_types.begin() +
                    arg_tensor_offsets[arg_index],
                arg_tensor_counts[arg_index],
                MemoryType::HOST_MEMORY);
        }

        return node;
    }

  private:
    std::string_view op_name;
    std::string_view op_type_name;

    // Memory types are stored as a contiguous list of inputs followed by
    // outputs.
    absl::InlinedVector<MemoryType, 8> tensor_memory_types;
    uint32_t input_tensor_count;
};

} // namespace tfdml
