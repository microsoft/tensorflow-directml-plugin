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

#include "tensorflow/c/env.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
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

static Status BufferToGraphDef(const TF_Buffer* in, tensorflow::GraphDef* out)
{
    if (in == nullptr || !out->ParseFromArray(in->data, in->length))
    {
        return errors::InvalidArgument(
            "Unparseable ",
            out->GetTypeName(),
            " proto");
    }
    return Status::OK();
}

void OptimizeGraph(
    void* optimizer,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status)
{
    printf("*****************************\n");
    printf("*****************************\n");
    printf("*****************************\n");
    printf("*****************************\n");
    printf("OPTIMIZING OPTIMIZER\n");

    int num_preserved_nodes;
    size_t preserved_nodes_size;

    TF_GetNodesToPreserveListSize(
        grappler_item,
        &num_preserved_nodes,
        &preserved_nodes_size,
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return;
    }

    std::vector<char*> preserved_node_names(num_preserved_nodes);
    std::vector<size_t> preserved_node_name_lengths(num_preserved_nodes);
    std::vector<char> preserved_node_name_storage(preserved_nodes_size);

    TF_GetNodesToPreserveList(
        grappler_item,
        preserved_node_names.data(),
        preserved_node_name_lengths.data(),
        num_preserved_nodes,
        preserved_node_name_storage.data(),
        preserved_node_name_storage.size(),
        raw_status);

    if (TF_GetCode(raw_status) != TF_OK)
    {
        return;
    }

    std::unordered_set<std::string> preserved_nodes;

    for (char* node_name : preserved_node_names)
    {
        printf("*******************\n");
        printf("*******************\n");
        printf("*******************\n");
        printf("*******************\n");
        printf("PRESERVED NODE NAME: %s\n", node_name);
        preserved_nodes.insert(node_name);
    }

    tensorflow::GraphDef input_graph_def;
    Status status = BufferToGraphDef(input_graph_buffer, &input_graph_def);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    for (int i = 0; i < input_graph_def.node_size(); ++i)
    {
        tensorflow::NodeDef node = input_graph_def.node(i);

        if (preserved_nodes.count(node.name()))
        {
            printf("*************************\n");
            printf("*************************\n");
            printf("*************************\n");
            printf("*************************\n");
            printf(
                "Found a node to preserve (%s), skipping...\n",
                node.name().c_str());
        }
        else
        {
            printf("*************************\n");
            printf("*************************\n");
            printf("*************************\n");
            printf("*************************\n");
            printf("Node found: %s\n", node.name().c_str());
        }
    }

    status = GraphDefToBuffer(input_graph_def, output_graph_buffer);

    if (!status.ok())
    {
        TF_SetStatus(raw_status, status.code(), status.error_message());
        return;
    }

    TF_SetStatus(raw_status, TF_OK, "");
}
} // namespace tfdml
