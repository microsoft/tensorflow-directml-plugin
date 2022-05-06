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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tfdml/runtime_adapter/status.h"

struct TF_GraphProperties;

namespace tfdml
{
class GrapplerItem;

class GraphProperties
{
  public:
    GraphProperties(const GrapplerItem& item);
    ~GraphProperties();

    Status InferStatically(
        bool assume_valid_feeds,
        bool aggressive_shape_inference,
        bool include_input_tensor_values,
        bool include_output_tensor_values);
    Status InferStatically(
        bool assume_valid_feeds,
        bool aggressive_shape_inference,
        bool include_tensor_values);
    Status InferStatically(bool assume_valid_feeds);

    const std::vector<tensorflow::OpInfo::TensorProperties>& GetInputProperties(
        const std::string& node_name) const
    {
        auto it = input_properties_.find(node_name);
        if (it != input_properties_.end())
        {
            return it->second;
        }
        return missing_properties_;
    }

    const std::vector<tensorflow::OpInfo::TensorProperties>&
    GetOutputProperties(const std::string& node_name) const
    {
        auto it = output_properties_.find(node_name);
        if (it != output_properties_.end())
        {
            return it->second;
        }
        return missing_properties_;
    }

  private:
    TF_GraphProperties* graph_props_;

    const GrapplerItem& item_;

    absl::flat_hash_map<
        std::string,
        std::vector<tensorflow::OpInfo::TensorProperties>>
        input_properties_;

    absl::flat_hash_map<
        std::string,
        std::vector<tensorflow::OpInfo::TensorProperties>>
        output_properties_;

    const std::vector<tensorflow::OpInfo::TensorProperties> missing_properties_;
};
} // namespace tfdml