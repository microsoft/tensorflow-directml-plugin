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

#include <string>

#include "absl/types/span.h"
#include "tfdml/core/util/attribute.h"
#include "tfdml/core/util/node_def.h"
#include "tfdml/core/util/resource_mgr.h"
#include "tfdml/core/util/types.h"

namespace tfdml
{
class OpKernel
{
  public:
    OpKernel(NodeDef node_def) : node_def_(std::move(node_def)) {}

    virtual ~OpKernel() = default;

    const NodeDef& node_def() const { return node_def_; }

    const std::string_view type_string() const
    {
        return node_def_.op_type_string;
    }
    const std::string_view name() const { return node_def_.op_name; }

    MemoryType input_memory_type(int index) const
    {
        return node_def_.input_tensor_memory_types[index];
    }

    MemoryType output_memory_type(int index) const
    {
        return node_def_.output_tensor_memory_types[index];
    }

  private:
    const NodeDef node_def_;
};
} // namespace tfdml
