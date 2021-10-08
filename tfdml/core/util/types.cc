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

#include "types.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/c/logging.h"

namespace tfdml {
std::string DataTypeString(TF_DataType dtype) {
  switch (dtype) {
    case TF_FLOAT:
      return "float";
    case TF_DOUBLE:
      return "double";
    case TF_INT32:
      return "int32";
    case TF_UINT32:
      return "uint32";
    case TF_UINT8:
      return "uint8";
    case TF_UINT16:
      return "uint16";
    case TF_INT16:
      return "int16";
    case TF_INT8:
      return "int8";
    case TF_STRING:
      return "string";
    case TF_COMPLEX64:
      return "complex64";
    case TF_COMPLEX128:
      return "complex128";
    case TF_INT64:
      return "int64";
    case TF_UINT64:
      return "uint64";
    case TF_BOOL:
      return "bool";
    case TF_QINT8:
      return "qint8";
    case TF_QUINT8:
      return "quint8";
    case TF_QUINT16:
      return "quint16";
    case TF_QINT16:
      return "qint16";
    case TF_QINT32:
      return "qint32";
    case TF_BFLOAT16:
      return "bfloat16";
    case TF_HALF:
      return "half";
    case TF_RESOURCE:
      return "resource";
    case TF_VARIANT:
      return "variant";
    default:
      TF_Log(TF_ERROR, "Unrecognized DataType enum value %d",
             static_cast<int>(dtype));
      return absl::StrCat("unknown dtype enum (", dtype, ")");
  }
}
}  //  namespace tfdml
