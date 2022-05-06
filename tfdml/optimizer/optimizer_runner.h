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

#include "tfdml/runtime_adapter/status.h"

struct TF_GrapplerItem;

namespace tensorflow
{
class GraphDef;
}

namespace tfdml
{
Status RunOptimizer(
    void* optimizer,
    const tensorflow::GraphDef& input_graph_def,
    const TF_GrapplerItem* grappler_item,
    tensorflow::GraphDef& output_graph_def);
} // namespace tfdml
