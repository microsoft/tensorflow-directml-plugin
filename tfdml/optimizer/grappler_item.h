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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/graph.pb.h"

struct TF_GrapplerItem;

namespace tfdml
{
struct OptimizationOptions
{
    // Is it allowed to add nodes to the graph that do not have registered
    // gradient function.
    bool allow_non_differentiable_rewrites = true;
};

struct GrapplerItem
{
    GrapplerItem(
        const TF_GrapplerItem* grappler_item,
        OptimizationOptions optimization_options,
        tensorflow::GraphDef graph);

    absl::flat_hash_set<std::string> NodesToPreserve() const;
    const TF_GrapplerItem* raw() const;
    tensorflow::GraphDef graph;
    OptimizationOptions& optimization_options();
    OptimizationOptions optimization_options_;

  private:
    const TF_GrapplerItem* const grappler_item_;
};
} // namespace tfdml
