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

// This file's sole purpose is to initialize the GUIDs declared using the
// DEFINE_GUID macro. This file is used instead of dxguids.cpp in the
// DirectX-Headers repository for two reasons:
// 1. DXGI IIDs aren't defined in DirectX-Headers
// 2. DirectML IIDs aren't defined in DirectX-Headers

#define INITGUID

// clang-format off
#ifndef _WIN32
#include "winadapter.h"
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include "DirectML.h"
#include "dxguids.h"
#include "dml_guids.h"
#else
#include <dxgi1_6.h>
#include <directx/d3d12.h>
#endif
// clang-format on
