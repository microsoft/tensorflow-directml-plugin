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

#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
Status GraphDefToBuffer(const tensorflow::GraphDef& in, TF_Buffer* out)
{
    if (out->data != nullptr)
    {
        return errors::InvalidArgument(
            "Passing non-empty TF_Buffer is invalid.");
    }
    const size_t proto_size = in.ByteSizeLong();
    void* buf = malloc(proto_size);
    if (buf == nullptr)
    {
        return errors::ResourceExhausted(
            "Failed to allocate memory to serialize message of type '",
            in.GetTypeName(),
            "' and size ",
            proto_size);
    }
    if (!in.SerializeWithCachedSizesToArray(static_cast<uint8_t*>(buf)))
    {
        free(buf);
        return errors::InvalidArgument(
            "Unable to serialize ",
            in.GetTypeName(),
            " protocol buffer, perhaps the serialized size (",
            proto_size,
            " bytes) is too large?");
    }
    out->data = buf;
    out->length = proto_size;
    out->data_deallocator = [](void* data, size_t length) { free(data); };
    return Status::OK();
}
} // namespace tfdml