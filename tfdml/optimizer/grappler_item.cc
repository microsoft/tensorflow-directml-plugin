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

#include "tfdml/optimizer/grappler_item.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
GrapplerItem::GrapplerItem(
    const TF_GrapplerItem* grappler_item,
    tensorflow::GraphDef graph)
    : grappler_item_(grappler_item),
      graph(std::move(graph))
{
}

absl::flat_hash_set<std::string> GrapplerItem::NodesToPreserve() const
{
    int num_preserved_nodes;
    size_t preserved_nodes_size;

    Status status;
    TF_GetNodesToPreserveListSize(
        grappler_item_,
        &num_preserved_nodes,
        &preserved_nodes_size,
        status.raw());
    CHECK(status.ok());

    std::vector<char*> preserved_node_names(num_preserved_nodes);
    std::vector<size_t> preserved_node_name_lengths(num_preserved_nodes);
    std::vector<char> preserved_node_name_storage(preserved_nodes_size);

    TF_GetNodesToPreserveList(
        grappler_item_,
        preserved_node_names.data(),
        preserved_node_name_lengths.data(),
        num_preserved_nodes,
        preserved_node_name_storage.data(),
        preserved_node_name_storage.size(),
        status.raw());
    CHECK(status.ok());

    absl::flat_hash_set<std::string> preserved_nodes;
    for (int i = 0; i < num_preserved_nodes; ++i)
    {
        preserved_nodes.insert(std::string(
            preserved_node_names[i],
            preserved_node_name_lengths[i]));
    }
    return preserved_nodes;
}

const TF_GrapplerItem* GrapplerItem::raw() const { return grappler_item_; }
} // namespace tfdml
