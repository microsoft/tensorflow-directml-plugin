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

#include "tensorflow/c/c_api.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
template <typename T>
static Status ParseBuffer(const TF_Buffer* in, T* out)
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
} // namespace tfdml
