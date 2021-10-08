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

#include "op_kernel_construction.h"

#include "tensorflow/c/kernels.h"

namespace tfdml {
OpKernelConstruction::OpKernelConstruction(TF_OpKernelConstruction* context)
    : context_(context) {}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  TF_VLog(1, "OP_REQUIRES failed at %s:%d : %s", file, line, s.error_message());
  status_.Update(s);
  TF_OpKernelConstruction_Failure(context_, status_.raw());
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  TF_Log(TF_WARNING, "OP_REQUIRES failed at %s:%d : %s", file, line,
         s.error_message());
  status_.Update(s);
  TF_OpKernelConstruction_Failure(context_, status_.raw());
}

}  // namespace tfdml
