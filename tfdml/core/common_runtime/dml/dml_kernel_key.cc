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

#include "tfdml/core/common_runtime/dml/dml_kernel_key.h"

#include "tfdml/core/util/tensor.h"

namespace tfdml {

DmlInputTensorKey DmlInputTensorKey::Clone() const {
  DmlInputTensorKey clone = {};

  // If the input is a CPU initializer, we need to deep copy its content
  // since it uniquely identifies it. Otherwise, only the shape and datatype
  // need to be part of the signature.
  if (this->is_constant_cpu_input) {
    clone.tensor = absl::get<Tensor>(this->tensor).DeepCopy();
  } else {
    clone.tensor = absl::get<TensorShapeAndType>(this->tensor);
  }

  clone.is_constant_cpu_input = this->is_constant_cpu_input;
  return clone;
}

DmlKernelKey DmlKernelKey::Clone() const {
  DmlKernelKey clone = {};
  clone.op_type_name = this->op_type_name;
  clone.attributes = this->attributes;

  for (const auto& input : this->input_tensors) {
    clone.input_tensors.push_back(input.Clone());
  }

  return clone;
}

bool DmlKernelKey::operator==(const DmlKernelKey& other) const {
  if (this->op_type_name != other.op_type_name) {
    return false;
  }

  // The named attributes should always be in the same order since they're
  // returned from the same kernel function that builds them in a deterministic
  // way
  if (this->attributes->GetNamedAttributes() !=
      other.attributes->GetNamedAttributes()) {
    return false;
  }

  if (this->input_tensors != other.input_tensors) {
    return false;
  }

  return true;
}

bool DmlInputTensorKey::operator==(const DmlInputTensorKey& other) const {
  if (this->is_constant_cpu_input != other.is_constant_cpu_input) {
    return false;
  }

  // Compare the tensors
  if (is_constant_cpu_input) {
    const auto& tensor0 = absl::get<Tensor>(this->tensor);
    const auto& tensor1 = absl::get<Tensor>(other.tensor);

    if (tensor0.shape() != tensor1.shape()) {
      return false;
    }

    if (tensor0.dtype() != tensor1.dtype()) {
      return false;
    }

    // If this is a constant CPU input, the tensor contents also form part of
    // the key, so we need to compare those too
    if (this->is_constant_cpu_input) {
      auto data_0 = tensor0.tensor_data();
      auto data_1 = tensor1.tensor_data();
      if (data_0.size() != data_1.size()) {
        return false;
      }

      if (memcmp(data_0.data(), data_1.data(), data_0.size())) {
        return false;
      }
    }
  } else {
    const auto& tensor0 = absl::get<TensorShapeAndType>(this->tensor);
    const auto& tensor1 = absl::get<TensorShapeAndType>(other.tensor);

    if (tensor0.shape != tensor1.shape) {
      return false;
    }

    if (tensor0.dtype != tensor1.dtype) {
      return false;
    }
  }

  return true;
}

}  // namespace tfdml