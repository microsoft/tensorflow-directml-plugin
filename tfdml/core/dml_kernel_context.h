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

#include "tensorflow/c/kernels.h"
#include "tfdml/core/dml_buffer.h"
#include "tfdml/core/dml_buffer_region.h"
#include "tfdml/core/dml_common.h"
#include "tfdml/core/dml_descriptor_bfc_allocator.h"
#include "tfdml/core/dml_device_context.h"
#include "tfdml/core/dml_gpu_event.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "tfdml/runtime_adapter/statusor.h"
#include "tfdml/runtime_adapter/tensor.h"

class IDMLDevice;
class ID3D12Device;

namespace tfdml
{

class DmlDevice;
class ShapeHelper;
class InitializationHelper;

// Context supplied to a DML kernel during construction.
class DmlKernelConstruction
{
  public:
    DmlKernelConstruction(
        const DmlDevice* device,
        OpKernelContext* op_ctx,
        absl::Span<const TensorShape> output_shapes,
        std::shared_ptr<const InitializationHelper> init_helper);

    IDMLDevice* GetDmlDevice() const;
    ID3D12Device* GetD3D12Device() const;
    OpKernelContext* GetOpKernelContext() const;
    DMLDeviceContext* GetDmlDeviceContext() const;
    std::shared_ptr<const InitializationHelper> GetInitializationHelper() const;

    // Input tensors
    uint32_t GetInputCount() const { return op_ctx_->num_inputs(); }
    TF_DataType GetInputDataType(uint32_t index) const;
    TensorShape GetInputTensorShape(uint32_t index) const;

    // Retrieves a constant CPU input tensor.
    //
    // Constant CPU input tensors are those which are declared as residing in
    // host memory during kernel registration (by specifying
    // .HostMemory("input_name") during REGISTER_KERNEL_BUILDER). Constant CPU
    // input tensors are available during kernel construction, their contents
    // are guaranteed not to change for the lifetime of the kernel, and their
    // contents are guaranteed to reside in CPU memory. This is useful when an
    // operator defines an input as dynamic but the kernel expects it to be
    // static (like an attribute would be).
    //
    // For example, the ClipByValue operator defines its clip_value_min and
    // clip_value_max inputs as scalar tensors - however DirectML expects it to
    // be provided as part of the operator desc. In this case, the kernel can
    // mark those inputs with .HostMemory which makes them constant CPU inputs,
    // and therefore available during kernel construction.
    //
    // Although the contents of the returned tensor are guaranteed not to change
    // during the lifetime of the kernel, the memory backing the tensor itself
    // is not guaranteed to outlive the DmlKernelConstruction. For this reason,
    // kernels should not store references to the returned tensor which persist
    // beyond the kernel constructor.
    Tensor GetConstantInputTensor(uint32_t index) const;

    // Output tensors

    uint32_t GetOutputCount() const { return op_ctx_->num_outputs(); }
    TF_DataType GetOutputDataType(uint32_t index) const;
    const TensorShape& GetOutputTensorShape(uint32_t index) const;

    // See OpKernelConstruction::GetAttr
    template <typename T> Status GetAttr(const char* attr_name, T* value) const;

    template <>
    Status GetAttr<int32_t>(const char* attr_name, int32_t* value) const
    {
        Status status;
        TF_OpKernelConstruction_GetAttrInt32(
            ctx_,
            attr_name,
            value,
            status.raw());
        return status;
    }

    template <>
    Status GetAttr<int64_t>(const char* attr_name, int64_t* value) const
    {
        Status status;
        TF_OpKernelConstruction_GetAttrInt64(
            ctx_,
            attr_name,
            value,
            status.raw());
        return status;
    }

  private:
    const DmlDevice* device_;
    OpKernelContext* op_ctx_;
    absl::Span<const TensorShape> output_shapes_;
    std::shared_ptr<const InitializationHelper> init_helper_;
    TF_OpKernelConstruction* ctx_;
};

// Context supplied to a DML kernel during execution.
class DmlKernelContext
{
  public:
    DmlKernelContext(
        const DmlDevice* device,
        OpKernelContext* op_ctx,
        const InitializationHelper* init_helper,
        absl::Span<const TensorShape> output_shapes,
        absl::Span<const absl::optional<uint32_t>> output_refs_forwarding = {});

    IDMLDevice* GetDmlDevice() const;
    ID3D12Device* GetD3D12Device() const;
    OpKernelContext* GetOpKernelContext() const;
    DMLDeviceContext* GetDmlDeviceContext() const;

    template <typename T> const T* GetInitializationHelper() const
    {
        return static_cast<const T*>(init_helper_);
    }

    Tensor GetInputTensor(int index) const { return op_ctx_->input(index); }
    uint32_t GetInputCount() const { return op_ctx_->num_inputs(); }

    Tensor& GetOutputTensor(int index) { return output_tensors_[index]; }
    uint32_t GetOutputCount() const { return op_ctx_->num_outputs(); }

  private:
    const DmlDevice* device_;
    OpKernelContext* op_ctx_;
    const InitializationHelper* init_helper_;

    // These output tensors are owned by the framework, because they're
    // allocated using OpKernelContext::allocate_output()
    absl::InlinedVector<Tensor, 4> output_tensors_;
};

} // namespace tfdml
