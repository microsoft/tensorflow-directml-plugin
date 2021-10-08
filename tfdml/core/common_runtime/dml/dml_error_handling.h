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

#include "absl/strings/string_view.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include "winadapter.h"
#endif

namespace tfdml {
namespace dml_util {
[[noreturn]] void HandleFailedHr(HRESULT hr, const char* expression,
                                 const char* file, int line);

bool HrIsOutOfMemory(HRESULT hr);
absl::string_view StringifyDeviceRemovedReason(HRESULT reason);

}  // namespace dml_util
}  // namespace tfdml

#define DML_CHECK_SUCCEEDED(x)                                      \
  do {                                                              \
    HRESULT _hr = (x);                                              \
    if (FAILED(_hr)) {                                              \
      tfdml::dml_util::HandleFailedHr(_hr, #x, __FILE__, __LINE__); \
    }                                                               \
  } while (0)
