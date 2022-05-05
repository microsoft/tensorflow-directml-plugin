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

struct TF_FunctionLibraryDefinition;

namespace tensorflow
{
class OpDef;
}

namespace tfdml
{
class OpRegistry
{
  public:
    OpRegistry();
    void Initialize(TF_FunctionLibraryDefinition* function_lib_def);
    Status LookUpOpDef(const char* op_name, tensorflow::OpDef* op_def);

  private:
    TF_FunctionLibraryDefinition* function_lib_def_ = nullptr;
};
} // namespace tfdml