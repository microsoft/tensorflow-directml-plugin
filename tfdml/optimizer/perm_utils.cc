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

#include "tfdml/optimizer/perm_utils.h"
#include "absl/container/flat_hash_map.h"
#include <vector>

namespace tfdml
{
absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format)
{
    const int size = data_format.size();
    absl::flat_hash_map<char, int> index;
    index.reserve(size);
    for (int i = 0; i < size; i++)
    {
        index[data_format[i]] = i;
    }
    return index;
}

std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format)
{
    // Generate permutation for transformation between src and dst format.
    // Example:
    // src = NWHC, dst = NCWH
    // index = { N:0 W:1 H:2 C:3 }
    // permutation = [0, 3, 1, 2]
    assert(src_dim_indices.size() == dst_format.size());
    std::vector<int> permutation;
    const int size = dst_format.size();
    permutation.reserve(size);
    for (int i = 0; i < size; i++)
    {
        permutation.push_back(src_dim_indices.at(dst_format[i]));
    }
    return permutation;
}
} // namespace tfdml
