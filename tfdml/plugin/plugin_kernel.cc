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

#include "tensorflow/c/kernels.h"

namespace tfdml
{
void RegisterKernels_AddN();
void RegisterKernels_AssignVariableOp();
void RegisterKernels_BiasAdd();
void RegisterKernels_Concat();
void RegisterKernels_Conv();
void RegisterKernels_Gather();
void RegisterKernels_GatherNd();
void RegisterKernels_MatMul();
} // namespace tfdml

void TF_InitKernel()
{
    // NOTE: we could add logic here to conditionally register kernels based on
    // D3D12 adapter capabilities (for example).

    tfdml::RegisterKernels_AddN();

    // TODO: Enable when TF_RESOURCE deserialization across ABI is enabled
    // https://github.com/tensorflow/tensorflow/issues/53531
    // tfdml::RegisterKernels_AssignVariableOp();

    tfdml::RegisterKernels_BiasAdd();
    tfdml::RegisterKernels_Concat();
    tfdml::RegisterKernels_Conv();
    tfdml::RegisterKernels_Gather();
    tfdml::RegisterKernels_GatherNd();
    tfdml::RegisterKernels_MatMul();
}
