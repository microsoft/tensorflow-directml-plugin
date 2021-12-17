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

struct TF_Buffer;
struct TF_GrapplerItem;
struct TF_Status;

namespace tfdml
{
void OptimizeGraph(
    void* optimizer,
    const TF_Buffer* input_graph_buffer,
    const TF_GrapplerItem* grappler_item,
    TF_Buffer* output_graph_buffer,
    TF_Status* raw_status);
} // namespace tfdml
