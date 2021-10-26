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

#include <string>

#include "absl/types/span.h"
#include "tfdml/core/util/attribute.h"
#include "tfdml/core/util/resource_mgr.h"
#include "tfdml/core/util/types.h"

namespace tfdml
{
class OpKernel
{
  public:
    OpKernel(const char* op_type_string, const char* op_name)
        : op_type_string_(op_type_string),
          op_name_(op_name)
    {
    }

    virtual ~OpKernel() = default;

    const std::string& type_string() const { return op_type_string_; }
    const std::string& name() const { return op_name_; }

    virtual MemoryType input_memory_type(int index) const
    {
        LogFatal("input_memory_type should only be called by DML kernels that "
                 "inherit "
                 "directly from DmlKernelWrapperBase.");
    }

    virtual MemoryType output_memory_type(int index) const
    {
        LogFatal("output_memory_type should only be called by DML kernels that "
                 "inherit "
                 "directly from DmlKernelWrapperBase.");
    }

  private:
    const std::string op_type_string_;
    const std::string op_name_;
};
} // namespace tfdml
