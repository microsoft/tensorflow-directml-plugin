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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tfdml/optimizer/generic_layout_optimizer.h"
#include "tfdml/optimizer/graph_optimizer.h"
#include "tfdml/optimizer/graph_view.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/optimizer/transpose_context.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{

static Status GraphDefToBuffer(const tensorflow::GraphDef& in, TF_Buffer* out)
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

void OptimizeGraph(
    void* optimizer,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status)
{
    tensorflow::GraphDef input_graph_def;
    Status status = ParseBuffer(input_graph_buffer, &input_graph_def);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    tensorflow::GraphDef output_graph_def;
    GrapplerItem grappler_item_wrapper(grappler_item, input_graph_def);

    std::array<std::unique_ptr<GraphOptimizer>, 1> optimizers = {
        std::make_unique<GenericLayoutOptimizer>(),
    };

    for (auto& optimizer : optimizers)
    {
        status = optimizer->Optimize(grappler_item_wrapper, &output_graph_def);

        if (!status.ok())
        {
            TF_SetStatus(raw_status, status.code(), status.error_message());
            return;
        }
    }

    status = GraphDefToBuffer(output_graph_def, output_graph_buffer);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    TF_SetStatus(raw_status, TF_OK, "");
}
} // namespace tfdml
