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
#include "absl/types/span.h"
#include "absl/types/variant.h"

using PrimitiveAttribute =
    absl::variant<int32_t, int64_t, float, bool, std::string>;

using Attribute = absl::
    variant<PrimitiveAttribute, absl::InlinedVector<PrimitiveAttribute, 4>>;

using NameAttributePair = std::pair<std::string, Attribute>;

// TODO: Remove this when/if the following PR gets merged
// https://github.com/tensorflow/tensorflow/pull/52157
struct BaseAttributes
{
    virtual ~BaseAttributes() = default;
    virtual absl::Span<const NameAttributePair> GetNamedAttributes() const = 0;
};
