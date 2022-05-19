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

#include "tfdml/optimizer/graph_properties.h"
#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tfdml/optimizer/grappler_item.h"
#include "tfdml/optimizer/proto_buffer_helpers.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
static void NormalizeShapeForOutput(tensorflow::TensorShapeProto* shape)
{
    for (int i = 0; i < shape->dim_size(); i++)
    {
        if (shape->dim(i).size() < -1)
        {
            TF_VLog(
                2,
                "Normalizing dimension: %d from %lld to -1",
                i,
                shape->dim(i).size());
            shape->mutable_dim(i)->set_size(-1);
        }
    }
}

GraphProperties::GraphProperties(const GrapplerItem& item)
    : item_(item),
      graph_props_(TF_NewGraphProperties(item.raw()))
{
}

GraphProperties::~GraphProperties() { TF_DeleteGraphProperties(graph_props_); }

Status GraphProperties::InferStatically(
    bool assume_valid_feeds,
    bool aggressive_shape_inference,
    bool include_input_tensor_values,
    bool include_output_tensor_values)
{
    Status status;
    TF_InferStatically(
        graph_props_,
        assume_valid_feeds,
        aggressive_shape_inference,
        include_input_tensor_values,
        include_output_tensor_values,
        status.raw());
    TF_RETURN_IF_ERROR(status);

    for (int i = 0; i < item_.graph.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = item_.graph.node(i);

        int num_inputs;
        TF_GetInputPropertiesListSize(
            graph_props_,
            node.name().c_str(),
            &num_inputs,
            status.raw());
        TF_RETURN_IF_ERROR(status);

        int num_outputs;
        TF_GetOutputPropertiesListSize(
            graph_props_,
            node.name().c_str(),
            &num_outputs,
            status.raw());
        TF_RETURN_IF_ERROR(status);

        std::vector<TF_Buffer*> input_tensor_props_buffer(num_inputs);
        for (int i = 0; i < num_inputs; i++)
        {
            input_tensor_props_buffer[i] = TF_NewBuffer();
        }

        std::vector<TF_Buffer*> output_tensor_props_buffer(num_outputs);
        for (int i = 0; i < num_outputs; i++)
        {
            output_tensor_props_buffer[i] = TF_NewBuffer();
        }

        absl::Cleanup props_cleanup = [&input_tensor_props_buffer,
                                       &output_tensor_props_buffer,
                                       num_inputs,
                                       num_outputs]
        {
            for (int i = 0; i < num_inputs; i++)
            {
                TF_DeleteBuffer(input_tensor_props_buffer[i]);
            }

            for (int i = 0; i < num_outputs; i++)
            {
                TF_DeleteBuffer(output_tensor_props_buffer[i]);
            }
        };

        TF_GetInputPropertiesList(
            graph_props_,
            node.name().c_str(),
            input_tensor_props_buffer.data(),
            num_inputs,
            status.raw());
        TF_RETURN_IF_ERROR(status);

        std::vector<tensorflow::OpInfo::TensorProperties>& input_properties =
            input_properties_[node.name()];
        input_properties.reserve(num_inputs);

        for (int i = 0; i < num_inputs; ++i)
        {
            tensorflow::OpInfo::TensorProperties tensor_props;
            TF_RETURN_IF_ERROR(
                ParseBuffer(input_tensor_props_buffer[i], &tensor_props));
            input_properties.push_back(std::move(tensor_props));
        }

        TF_GetOutputPropertiesList(
            graph_props_,
            node.name().c_str(),
            output_tensor_props_buffer.data(),
            num_outputs,
            status.raw());
        TF_RETURN_IF_ERROR(status);

        std::vector<tensorflow::OpInfo::TensorProperties>& output_properties =
            output_properties_[node.name()];
        output_properties.reserve(num_outputs);

        for (int i = 0; i < num_outputs; ++i)
        {
            tensorflow::OpInfo::TensorProperties tensor_props;
            TF_RETURN_IF_ERROR(
                ParseBuffer(output_tensor_props_buffer[i], &tensor_props));
            output_properties.push_back(std::move(tensor_props));
        }
    }

    return status;
}
Status GraphProperties::InferStatically(
    bool assume_valid_feeds,
    bool aggressive_shape_inference,
    bool include_tensor_values)
{
    return InferStatically(
        assume_valid_feeds,
        /*aggressive_shape_inference=*/aggressive_shape_inference,
        /*include_input_tensor_values=*/include_tensor_values,
        /*include_output_tensor_values=*/include_tensor_values);
}
Status GraphProperties::InferStatically(bool assume_valid_feeds)
{
    return InferStatically(
        assume_valid_feeds,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/true);
}
} // namespace tfdml