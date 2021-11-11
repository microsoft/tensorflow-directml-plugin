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

#include "tfdml/core/dml_kernel_context.h"

#include "tfdml/core/dml_device.h"
#include "tfdml/core/dml_event_queue.h"
#include "tfdml/core/dml_execution_context.h"
#include "tfdml/core/dml_upload_heap.h"
#include "tfdml/core/dml_util.h"
#include "tfdml/external/status.h"
#include "tfdml/external/tensor.h"

using Microsoft::WRL::ComPtr;

namespace tfdml
{

//
// DmlKernelConstruction
//

DmlKernelConstruction::DmlKernelConstruction(
    const DmlDevice* device,
    OpKernelContext* op_ctx,
    absl::Span<const TensorShape> output_shapes,
    std::shared_ptr<const InitializationHelper> init_helper)
    : device_(device),
      op_ctx_(op_ctx),
      output_shapes_(output_shapes),
      init_helper_(init_helper)
{
}

IDMLDevice* DmlKernelConstruction::GetDmlDevice() const
{
    return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelConstruction::GetD3D12Device() const
{
    return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelConstruction::GetOpKernelContext() const
{
    return op_ctx_;
}

DMLDeviceContext* DmlKernelConstruction::GetDmlDeviceContext() const
{
    return device_->GetDeviceContext();
}

std::shared_ptr<const InitializationHelper> DmlKernelConstruction::
    GetInitializationHelper() const
{
    return init_helper_;
}

TF_DataType DmlKernelConstruction::GetInputDataType(uint32_t index) const
{
    return op_ctx_->input_dtype(index);
}

TensorShape DmlKernelConstruction::GetInputTensorShape(uint32_t index) const
{
    return op_ctx_->input(index).shape();
}

Tensor DmlKernelConstruction::GetConstantInputTensor(uint32_t index) const
{
    CHECK(op_ctx_->input_memory_type(index) == HOST_MEMORY);
    CHECK(op_ctx_->input_dtype(index) != TF_RESOURCE);

    return op_ctx_->input(index);
}

TF_DataType DmlKernelConstruction::GetOutputDataType(uint32_t index) const
{
    return op_ctx_->expected_output_dtype(index);
}

const TensorShape& DmlKernelConstruction::GetOutputTensorShape(
    uint32_t index) const
{
    return output_shapes_[index];
}

//
// DmlKernelContext
//

DmlKernelContext::DmlKernelContext(
    const DmlDevice* device,
    OpKernelContext* op_ctx,
    const InitializationHelper* init_helper,
    absl::Span<const TensorShape> output_shapes,
    absl::Span<const absl::optional<uint32_t>> output_refs_forwarding)
    : device_(device),
      op_ctx_(op_ctx),
      init_helper_(init_helper)
{
    assert(output_shapes.size() == op_ctx_->num_outputs());

    // Allocate output tensors
    output_tensors_.reserve(output_shapes.size());
    for (int i = 0; i < static_cast<int>(output_shapes.size()); ++i)
    {
        auto status_or_tensor = op_ctx_->allocate_output(i, output_shapes[i]);
        OP_REQUIRES_OK(op_ctx_, status_or_tensor.status());
        output_tensors_.push_back(status_or_tensor.ConsumeValueOrDie());
    }
}

IDMLDevice* DmlKernelContext::GetDmlDevice() const
{
    return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelContext::GetD3D12Device() const
{
    return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelContext::GetOpKernelContext() const
{
    return op_ctx_;
}

DMLDeviceContext* DmlKernelContext::GetDmlDeviceContext() const
{
    return device_->GetDeviceContext();
}

} // namespace tfdml
