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

#include "tfdml/optimizer/optimizer_runner.h"
#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/graph_optimizer.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/op_registry.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
Status RunOptimizer(
    void* optimizer,
    const tensorflow::GraphDef& input_graph_def,
    const TF_GrapplerItem* grappler_item,
    tensorflow::GraphDef& output_graph_def)
{
    // TODO: Remove the copy once the API takes a const TF_Buffer*
    // https://github.com/tensorflow/tensorflow/issues/55226
    TF_Buffer* input_graph_copy = TF_NewBuffer();
    absl::Cleanup input_graph_copy_cleanup = [input_graph_copy]
    { TF_DeleteBuffer(input_graph_copy); };
    TF_RETURN_IF_ERROR(GraphDefToBuffer(input_graph_def, input_graph_copy));

    Status status;
    TF_FunctionLibraryDefinition* f_lib =
        TF_NewFunctionLibraryDefinition(input_graph_copy, status.raw());
    TF_RETURN_IF_ERROR(status);

    absl::Cleanup f_lib_cleanup = [f_lib]
    { TF_DeleteFunctionLibraryDefinition(f_lib); };

    OpRegistry::Instance().Initialize(f_lib);
    GrapplerItem grappler_item_wrapper(grappler_item, input_graph_def);

    TF_RETURN_IF_ERROR(static_cast<GraphOptimizer*>(optimizer)->Optimize(
        grappler_item_wrapper,
        &output_graph_def));

    return Status::OK();
}
} // namespace tfdml
