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

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tfdml/runtime_adapter/types.h"

namespace tfdml
{
using AttributeValue = absl::optional<absl::variant<
    TF_DataType,
    int64_t,
    float,
    bool,
    std::string,
    std::vector<TF_DataType>,
    std::vector<int64_t>,
    std::vector<float>,
    std::vector<bool>,
    std::vector<std::string>>>;

} // namespace tfdml