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

#include "tfdml/optimizer/graph_optimizer.h"
#include "tfdml/runtime_adapter/status.h"

namespace tensorflow
{
class GraphDef;
}

namespace tfdml
{
// Optimize TF computations by remapping subgraphs/nodes onto other subgraphs or
// nodes to decrease the amount of operations needed to perform a computation.
class Remapper : public GraphOptimizer
{
  public:
    ~Remapper() override {}
    Status Optimize(
        const GrapplerItem& item,
        tensorflow::GraphDef* optimized_graph) override;
};
} // namespace tfdml