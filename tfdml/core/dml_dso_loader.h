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

// Common DSO loading functionality: exposes callables that dlopen DSOs
// in either the runfiles directories

#pragma once

#include "tfdml/core/util/statusor.h"

namespace tfdml
{
namespace DmlDsoLoader
{
// The following methods either load the DSO of interest and return a dlopen
// handle or error status.
StatusOr<void*> GetDirectMLDsoHandle();
StatusOr<void*> GetDirectMLDebugDsoHandle();
StatusOr<void*> GetD3d12DsoHandle();
StatusOr<void*> GetDxgiDsoHandle();
StatusOr<void*> GetDxCoreDsoHandle();
StatusOr<void*> GetPixDsoHandle();
StatusOr<void*> GetKernel32DsoHandle();
} // namespace DmlDsoLoader

// Wrapper around the DmlDsoLoader that prevents us from dlopen'ing any of the
// DSOs more than once.
namespace DmlCachedDsoLoader
{
// Cached versions of the corresponding DmlDsoLoader methods above.
StatusOr<void*> GetDirectMLDsoHandle();
StatusOr<void*> GetDirectMLDebugDsoHandle();
StatusOr<void*> GetD3d12DsoHandle();
StatusOr<void*> GetDxgiDsoHandle();
StatusOr<void*> GetDxCoreDsoHandle();
StatusOr<void*> GetPixDsoHandle();
StatusOr<void*> GetKernel32DsoHandle();
} // namespace DmlCachedDsoLoader
} // namespace tfdml
