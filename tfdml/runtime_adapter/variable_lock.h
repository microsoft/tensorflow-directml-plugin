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

#include "absl/types/span.h"

struct TF_VariableInputLockHolder;

namespace tfdml
{

class OpKernelContext;

class VariableLock
{
  public:
    VariableLock(OpKernelContext* ctx);
    void LockShared(absl::Span<const int> input_indices);
    void LockUnique(absl::Span<const int> input_indices);
    void Unlock();

  private:
    TF_VariableInputLockHolder* lock_holder_ = nullptr;
    OpKernelContext* ctx_;
};

} // namespace tfdml