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

#include "tfdml/optimizer/op_registry.h"
#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
void OpRegistry::Initialize(TF_FunctionLibraryDefinition* function_lib_def)
{
    function_lib_def_ = function_lib_def;
}

Status OpRegistry::LookUpOpDef(const char* op_name, tensorflow::OpDef* op_def)
{
    if (function_lib_def_ == nullptr)
    {
        return errors::InvalidArgument("OpRegistry::Initialize must be called "
                                       "once before OpRegistry::LookUpOpDef.");
    }

    Status status;
    TF_Buffer* op_buf = TF_NewBuffer();
    absl::Cleanup buf_cleanup = [op_buf] { TF_DeleteBuffer(op_buf); };

    TF_LookUpOpDef(function_lib_def_, op_name, op_buf, status.raw());
    TF_RETURN_IF_ERROR(status);

    return ParseBuffer(op_buf, op_def);
}
} // namespace tfdml