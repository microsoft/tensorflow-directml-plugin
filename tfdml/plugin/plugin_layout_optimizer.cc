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
#include "tfdml/layout_optimizer/generic_layout_optimizer.h"
#include "tfdml/optimizer/optimizer_runner.h"

namespace tfdml
{
static void* CreateOptimizer() { return new GenericLayoutOptimizer(); }

static void OptimizeGraph(
    void* optimizer,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status)
{
    RunOptimizer(
        optimizer,
        input_graph_buffer,
        grappler_item,
        output_graph_buffer,
        raw_status);
}

void DeleteOptimizer(void* optimizer)
{
    delete static_cast<GenericLayoutOptimizer*>(optimizer);
}

} // namespace tfdml

// TODO: Re-enable the warning once the header has been fixed
// https://github.com/tensorflow/tensorflow/pull/55579
#pragma warning(push)
#pragma warning(disable : 4273)

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status)
{
    params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
    params->device_type = "DML";
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

void TF_InitKernel() {}
