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
#include "tfdml/external/status.h"
#include "tfdml/external/statusor.h"
#include "tfdml/external/tensor.h"
#include "tfdml/core/util/types.h"

struct TF_OpKernelContext;

namespace tfdml
{
class Device;
class OpKernel;
class ResourceMgr;

class OpKernelContext
{
  public:
    OpKernelContext(TF_OpKernelContext* context, OpKernel* op_kernel);
    Tensor input(int input_index);
    int num_inputs() const;
    int num_outputs() const;
    void CtxFailure(const char* file, int line, const Status& s);
    void CtxFailureWithWarning(const char* file, int line, const Status& s);
    TF_DataType input_dtype(int index);
    TF_DataType expected_output_dtype(int index);
    StatusOr<Tensor> allocate_output(int index, const TensorShape& shape);
    const Status& status() const;
    Device* device() const;
    Status allocate_temp(
        TF_DataType dtype,
        const TensorShape& shape,
        Tensor* tensor,
        bool on_host = true);
    MemoryType input_memory_type(int index) const;
    MemoryType output_memory_type(int index) const;
    ResourceMgr* resource_manager() const;
    Status set_output(int index, const Tensor& tensor);
    const OpKernel& op_kernel() const;

  private:
    TF_OpKernelContext* const context_;
    Status status_;
    Device* device_;
    OpKernel* const op_kernel_;
};
} // namespace tfdml
