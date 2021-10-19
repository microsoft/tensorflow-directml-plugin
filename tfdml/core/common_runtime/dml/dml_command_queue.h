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

#include <deque>

#include "dml_common.h"
#include "dml_gpu_event.h"

namespace tfdml
{

// Manages a D3D12 command queue and provides a waitable fence which is signaled
// with a monotonically increasing value once each execute completes on the GPU.
class DmlCommandQueue
{
  public:
    // Creates a CommandQueue object that wraps an existing D3D12 queue.
    DmlCommandQueue(ID3D12CommandQueue* existing_queue);

    D3D12_COMMAND_LIST_TYPE GetType() const { return type_; }
    Microsoft::WRL::ComPtr<ID3D12Fence> GetFence() const { return fence_; }
    uint64_t GetLastFenceValue() const { return last_fence_value_; }

    void ExecuteCommandLists(absl::Span<ID3D12CommandList*> command_lists);

    // Returns an event that will become signaled when everything submitted to
    // the queue thus far has completed execution on the GPU.
    DmlGpuEvent GetCurrentCompletionEvent();

    // Returns an event that will become signaled after the next
    // ExecuteCommandLists call.
    DmlGpuEvent GetNextCompletionEvent();

  private:
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    D3D12_COMMAND_LIST_TYPE type_;

    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    uint64_t last_fence_value_ = 0;
};

} // namespace tfdml
