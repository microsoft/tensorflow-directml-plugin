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

#include "op_kernel_context.h"

#include "tensorflow/c/env.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_tensor.h"
#include "tfdml/runtime_adapter/device.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/op_kernel.h"
#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/stream.h"

namespace tfdml
{
OpKernelContext::OpKernelContext(
    TF_OpKernelContext* context,
    OpKernel* op_kernel)
    : context_(context),
      op_kernel_(op_kernel)
{
    Status status;
    SP_Stream stream = TF_GetStream(context, status.raw());
    CHECK(status.ok());

    device_ = static_cast<Device*>(stream->stream_handle);
}

Tensor OpKernelContext::input(int input_index)
{
    assert(input_index < TF_NumInputs(context_));
    TF_Tensor* tensor = nullptr;
    Status status;
    TF_GetInput(context_, input_index, &tensor, status.raw());

    CHECK(status.ok());

    return Tensor(tensor);
}

int OpKernelContext::num_inputs() const
{
    assert(context_ != nullptr);
    return TF_NumInputs(context_);
}

int OpKernelContext::num_outputs() const
{
    assert(context_ != nullptr);
    return TF_NumOutputs(context_);
}

void OpKernelContext::CtxFailure(const char* file, int line, const Status& s)
{
    TF_VLog(
        1,
        "OP_REQUIRES failed at %s:%d : %s",
        file,
        line,
        s.error_message());
    status_.Update(s);
    TF_OpKernelContext_Failure(context_, status_.raw());
}

void OpKernelContext::CtxFailureWithWarning(
    const char* file,
    int line,
    const Status& s)
{
    TF_Log(
        TF_WARNING,
        "OP_REQUIRES failed at %s:%d : %s",
        file,
        line,
        s.error_message());
    status_.Update(s);
    TF_OpKernelContext_Failure(context_, status_.raw());
}

StatusOr<Tensor> OpKernelContext::allocate_output(
    int index,
    const TensorShape& shape)
{
    TF_DataType dtype = TF_ExpectedOutputDataType(context_, index);
    size_t dtype_size = TF_DataTypeSize(dtype);
    size_t size_in_bytes = dtype_size * shape.num_elements();

    Status status;
    TF_Tensor* raw_tensor = TF_AllocateOutput(
        context_,
        index,
        dtype,
        shape.data(),
        shape.dims(),
        size_in_bytes,
        status.raw());

    if (!status.ok())
    {
        return status;
    }

    return Tensor(raw_tensor);
}

// TODO: Make candidate_input_indices constant once the API has been fixed
// https://github.com/tensorflow/tensorflow/pull/54139
StatusOr<Tensor> OpKernelContext::forward_input_or_allocate_output(
    absl::Span<int> candidate_input_indices,
    int output_index,
    const TensorShape& output_shape,
    int* forwarded_input)
{
    Status status;
    TF_Tensor* raw_tensor = TF_ForwardInputOrAllocateOutput(
        context_,
        candidate_input_indices.data(),
        candidate_input_indices.size(),
        output_index,
        output_shape.data(),
        output_shape.dims(),
        forwarded_input,
        status.raw());

    if (!status.ok())
    {
        return status;
    }

    return Tensor(raw_tensor);
}

void OpKernelContext::forward_ref_input_to_ref_output(
    int input_index,
    int output_index)
{
    TF_OpKernelContext_ForwardRefInputToRefOutput(
        context_,
        input_index,
        output_index);
}

TF_DataType OpKernelContext::input_dtype(int index)
{
    TF_Tensor* raw_tensor = nullptr;

    Status status;
    TF_GetInput(context_, index, &raw_tensor, status.raw());
    CHECK(status.ok());

    return Tensor(raw_tensor).dtype();
}

TF_DataType OpKernelContext::expected_output_dtype(int index)
{
    return TF_ExpectedOutputDataType(context_, index);
}

Status OpKernelContext::allocate_temp(
    TF_DataType dtype,
    const TensorShape& shape,
    Tensor* tensor,
    bool on_host)
{
    assert(tensor != nullptr);
    TF_AllocatorAttributes alloc_attributes;
    alloc_attributes.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
    alloc_attributes.on_host = on_host;

    Status status;
    TF_Tensor* raw_tensor = TF_AllocateTemp(
        context_,
        dtype,
        shape.data(),
        shape.dims(),
        &alloc_attributes,
        status.raw());

    if (status.ok())
    {
        *tensor = Tensor(raw_tensor);
    }

    return status;
}

const Status& OpKernelContext::status() const { return status_; }

Device* OpKernelContext::device() const { return device_; }

const OpKernel& OpKernelContext::op_kernel() const { return *op_kernel_; }

MemoryType OpKernelContext::input_memory_type(int index) const
{
    return op_kernel_->input_memory_type(index);
}

MemoryType OpKernelContext::output_memory_type(int index) const
{
    return op_kernel_->output_memory_type(index);
}

Status OpKernelContext::set_output(int index, const Tensor& tensor)
{
    Status status;
    TF_SetOutput(context_, index, tensor.raw(), status.raw());
    return status;
}

static void CopyTensorInSameDevice(
    TF_OpKernelContext* ctx,
    TF_Tensor* source,
    TF_Tensor* dest)
{
    Status status;
    SP_Stream stream = TF_GetStream(ctx, status.raw());
    CHECK(status.ok());

    Device* device = static_cast<Device*>(stream->stream_handle);
    Tensor input_tensor(source);
    Tensor output_tensor(dest);
    device->CopyTensorInSameDevice(&input_tensor, &output_tensor);
}

Status OpKernelContext::AssignVariable(int var_index, int value_index)
{
    Status status;

    TF_AssignVariable(
        context_,
        var_index,
        value_index,
        CopyTensorInSameDevice,
        status.raw());

    return status;
}

Status OpKernelContext::AssignUpdateVariable(
    int var_index,
    int value_index,
    void (*updateFunc)(
        TF_OpKernelContext* ctx,
        TF_Tensor* tensor,
        TF_Tensor* value,
        int Op))
{
    assert(var_index < num_inputs());
    assert(value_index < num_inputs());
    Status status;

    TF_AssignUpdateVariable(
        context_,
        var_index,
        value_index,
        0,
        input(value_index).dtype() == TF_VARIANT,
        CopyTensorInSameDevice,
        updateFunc,
        status.raw());
    return status;
}

Status OpKernelContext::AddNVariant(void (*binary_add_func)(
    TF_OpKernelContext* ctx,
    const TF_Tensor* a,
    const TF_Tensor* b,
    TF_Tensor* out))
{
    Status status;
    TF_AddNVariant(context_, binary_add_func, status.raw());
    return status;
}

Status OpKernelContext::GetInputTensorFromVariable(
    int var_index,
    bool lock_held,
    bool is_variant,
    Tensor* tensor)
{
    assert(var_index < num_inputs());
    constexpr bool sparse = false;
    static constexpr int64_t empty_sizes[1] = {0};
    TF_Tensor* raw_tensor = TF_AllocateTensor(TF_FLOAT, empty_sizes, 1, 0);

    Status status;
    TF_GetInputTensorFromVariable(
        context_,
        var_index,
        lock_held,
        is_variant,
        sparse,
        CopyTensorInSameDevice,
        &raw_tensor,
        status.raw());

    if (!status.ok())
    {
        return status;
    }

    *tensor = Tensor(raw_tensor);

    return Status::OK();
}

TF_OpKernelContext* OpKernelContext::raw() const { return context_; }

} // namespace tfdml
