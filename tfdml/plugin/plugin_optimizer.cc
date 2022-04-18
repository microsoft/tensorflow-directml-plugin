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

#include "plugin_version.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/data_format_ops_converter.h"
#include "tfdml/optimizer/optimizer_runner.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/optimizer/transpose_remover.h"

namespace tfdml
{
static void* CreateOptimizer()
{
    return new std::vector<GraphOptimizer*>{
        // TODO: Remove DataFormatOpsConverter when the data format ops get
        // their DEVICE_DEFAULT registration
        // https://github.com/tensorflow/tensorflow/pull/55558
        new DataFormatOpsConverter(),
        new TransposeRemover(),
    };
}

static void OptimizeGraph(
    void* optimizers,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status)
{
    tensorflow::GraphDef output_graph_def;
    Status status = ParseBuffer(input_graph_buffer, &output_graph_def);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    auto cast_optimizers =
        static_cast<std::vector<GraphOptimizer*>*>(optimizers);

    for (GraphOptimizer* optimizer : *cast_optimizers)
    {
        tensorflow::GraphDef input_graph_def = std::move(output_graph_def);

        Status status = RunOptimizer(
            optimizer,
            input_graph_def,
            grappler_item,
            output_graph_def);

        if (!status.ok())
        {
            TF_SetStatus(raw_status, status.code(), status.error_message());
            return;
        }
    }

    status = GraphDefToBuffer(output_graph_def, output_graph_buffer);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    TF_SetStatus(raw_status, TF_OK, "");
}

void DeleteOptimizer(void* optimizers)
{
    auto cast_optimizers =
        static_cast<std::vector<GraphOptimizer*>*>(optimizers);

    for (GraphOptimizer* optimizer : *cast_optimizers)
    {
        delete optimizer;
    }
}

} // namespace tfdml

// TODO: Re-enable the warning once the header has been fixed
// https://github.com/tensorflow/tensorflow/pull/55579
#pragma warning(push)
#pragma warning(disable : 4273)

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status)
{
    params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
    params->device_type = "GPU";
    params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
    params->major_version = DML_MAJOR_VERSION;
    params->minor_version = DML_MINOR_VERSION;
    params->patch_version = DML_PATCH_VERSION;
    params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
    params->optimizer->create_func = tfdml::CreateOptimizer;
    params->optimizer->optimize_func = tfdml::OptimizeGraph;
    params->optimizer->destroy_func = tfdml::DeleteOptimizer;
}

#pragma warning(pop)
