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
#include "tfdml/core/util/kernel_def_builder.h"

extern "C" void RegisterKernels_Concat();

void TF_InitKernel()
{
    // do kernel registration here... gives us a chance to detect device
    // capabilities and conditionally register support

    RegisterKernels_Concat();
}
