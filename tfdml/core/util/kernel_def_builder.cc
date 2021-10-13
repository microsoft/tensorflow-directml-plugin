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

#include "kernel_def_builder.h"

#include "tensorflow/c/kernels.h"

namespace tfdml {

KernelDefBuilder& KernelDefBuilder::Device(const char* device_type) {
  device_type_ = device_type;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::HostMemory(const char* arg_name) {
  host_memory_arg_names_.push_back(arg_name);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Priority(int32_t priority) {
  priority_ = priority;
  return *this;
}

}  // namespace tfdml
