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

#include "tfdml/runtime_adapter/variable_lock.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tfdml/runtime_adapter/device.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "tfdml/runtime_adapter/stream.h"

namespace tfdml
{

VariableLock::VariableLock(OpKernelContext* ctx) : ctx_(ctx) {}

// This copy will only be done for sparse tensors
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

void VariableLock::LockShared(absl::Span<const int> input_indices)
{
    constexpr bool unique_lock = false;
    constexpr bool sparse = false;
    Status status;

#ifdef _WIN32
    void* tf_handle =
        TF_LoadSharedLibrary("_pywrap_tensorflow_internal.pyd", status.raw());

    CHECK(status.ok());

    void* lock_variables_function = TF_GetSymbolFromLibrary(
        tf_handle,
        "TF_MaybeLockVariableInputMutexesInOrder",
        status.raw());

    CHECK(status.ok());

    using TF_MaybeLockVariableInputMutexesInOrderFunction =
        decltype(TF_MaybeLockVariableInputMutexesInOrder);

    auto* TF_MaybeLockVariableInputMutexesInOrder =
        static_cast<TF_MaybeLockVariableInputMutexesInOrderFunction*>(
            lock_variables_function);
#endif

    TF_MaybeLockVariableInputMutexesInOrder(
        ctx_->raw(),
        unique_lock,
        sparse,
        input_indices.data(),
        input_indices.size(),
        CopyTensorInSameDevice,
        &lock_holder_,
        status.raw());
}
void VariableLock::LockUnique(absl::Span<const int> input_indices)
{
    constexpr bool unique_lock = true;
    constexpr bool sparse = false;
    Status status;

#ifdef _WIN32
    void* tf_handle =
        TF_LoadSharedLibrary("_pywrap_tensorflow_internal.pyd", status.raw());

    CHECK(status.ok());

    void* lock_variables_function = TF_GetSymbolFromLibrary(
        tf_handle,
        "TF_MaybeLockVariableInputMutexesInOrder",
        status.raw());

    CHECK(status.ok());

    using TF_MaybeLockVariableInputMutexesInOrderFunction =
        decltype(TF_MaybeLockVariableInputMutexesInOrder);

    auto* TF_MaybeLockVariableInputMutexesInOrder =
        static_cast<TF_MaybeLockVariableInputMutexesInOrderFunction*>(
            lock_variables_function);
#endif

    TF_MaybeLockVariableInputMutexesInOrder(
        ctx_->raw(),
        unique_lock,
        sparse,
        input_indices.data(),
        input_indices.size(),
        CopyTensorInSameDevice,
        &lock_holder_,
        status.raw());
}
void VariableLock::Unlock()
{
    if (lock_holder_)
    {
#ifdef _WIN32
        Status status;
        void* tf_handle =
            TF_LoadSharedLibrary("_pywrap_tensorflow_internal.pyd", status.raw());

        CHECK(status.ok());

        void* unlock_variables_function = TF_GetSymbolFromLibrary(
            tf_handle,
            "TF_ReleaseVariableInputLockHolder",
            status.raw());

        CHECK(status.ok());

        using TF_ReleaseVariableInputLockHolderFunction =
            decltype(TF_ReleaseVariableInputLockHolder);

        auto* TF_ReleaseVariableInputLockHolder =
            static_cast<TF_ReleaseVariableInputLockHolderFunction*>(
                unlock_variables_function);
#endif

        TF_ReleaseVariableInputLockHolder(lock_holder_);
        lock_holder_ = nullptr;
    }
}

} // namespace tfdml