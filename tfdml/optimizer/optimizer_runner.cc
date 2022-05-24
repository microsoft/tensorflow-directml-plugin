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
#include "tfdml/optimizer/map_utils.h"
#include "tfdml/optimizer/op_registry.h"
#include "tfdml/optimizer/op_types.h"
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
    auto input_graph_copy_cleanup = absl::MakeCleanup([input_graph_copy]
    { TF_DeleteBuffer(input_graph_copy); });
    TF_RETURN_IF_ERROR(GraphDefToBuffer(input_graph_def, input_graph_copy));

    Status status;
    TF_FunctionLibraryDefinition* f_lib =
        TF_NewFunctionLibraryDefinition(input_graph_copy, status.raw());
    TF_RETURN_IF_ERROR(status);

    auto f_lib_cleanup = absl::MakeCleanup([f_lib]
    { TF_DeleteFunctionLibraryDefinition(f_lib); });

    OpRegistry op_reg;
    op_reg.Initialize(f_lib);
    OptimizationOptions op_options;

    using NodeDefs = google::protobuf::RepeatedPtrField<tensorflow::NodeDef>;

    // Find functions for which we might need to compute a gradient at runtime.
    absl::flat_hash_set<std::string> differentiable_functions;

    const auto find_differentiable_functions =
        [&](const NodeDefs& nodes) -> void
    {
        for (const tensorflow::NodeDef& node : nodes)
        {
            if (IsSymbolicGradient(node))
            {
                const auto* f_attr = FindOrNull(node.attr(), "f");
                if (f_attr)
                {
                    op_options.allow_non_differentiable_rewrites = false;
                    break;
                }
            }
        }
    };

    // SymbolicGradient nodes inside the main graph.
    find_differentiable_functions(input_graph_def.node());
    // SymbolicGradient nodes inside the function library.
    tensorflow::OpDef op_def;
    Status lookup_status = op_reg.LookUpOpDef("SymbolicGradient", &op_def);
    if (lookup_status.ok())
        op_options.allow_non_differentiable_rewrites = false;

    GrapplerItem grappler_item_wrapper(
        grappler_item,
        op_options,
        input_graph_def);

    TF_RETURN_IF_ERROR(static_cast<GraphOptimizer*>(optimizer)->Optimize(
        grappler_item_wrapper,
        &output_graph_def));

    return Status::OK();
}
} // namespace tfdml
