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

#include "absl/container/flat_hash_map.h"
#include "absl/types/variant.h"
#include "tensorflow/c/tf_datatype.h"
#include "tfdml/core/common_runtime/dml/dml_common.h"
#include "tfdml/core/util/attribute.h"
#include "tfdml/core/util/tensor.h"

namespace tfdml
{

struct TensorShapeAndType
{
    TensorShape shape;
    TF_DataType dtype;

    template <typename H>
    friend H AbslHashValue(H h, const TensorShapeAndType& shape_and_type)
    {
        auto result = H::combine(
            std::move(h),
            shape_and_type.shape,
            shape_and_type.dtype);
        return result;
    }
};

// Used to identify/hash an input tensor for a DML kernel. A DML kernel may
// choose to register certain input tensors as requiring to be stored in host
// memory. (This is achieved by using .HostMemory("input_name") during
// REGISTER_KERNEL_BUILDER). Such tensors are known as constant CPU input
// tensors, otherwise they're just regular input tensors.
//
// Since constant CPU inputs are provided during construction of DML kernels,
// the contents of the tensor (as well as its shape and type) forms part of the
// signature that uniquely identifies a DML kernel instance. Otherwise, just the
// shape and data type form part of the key.
struct DmlInputTensorKey
{
    // If is_constant_cpu_input is false, this stores just the
    // TensorShape and type. Otherwise, for constant CPU inputs, this
    // stores the entire tensor (i.e. the shape/dtype as well as the data
    // itself.)
    absl::variant<Tensor, TensorShapeAndType> tensor;
    bool is_constant_cpu_input;

    DmlInputTensorKey Clone() const; // Performs a deep copy
    bool operator==(const DmlInputTensorKey& other) const;

    template <typename H>
    friend H AbslHashValue(H h, const DmlInputTensorKey& input_tensor_key)
    {
        auto result = H::combine(std::move(h), input_tensor_key.tensor);
        return result;
    }
};

// Uniquely identifies a DML kernel instance. This is used for caching of
// kernels, since DML kernels are immutable once constructed.
struct DmlKernelKey
{
    std::string op_type_name; // e.g. "Conv2D"
    std::shared_ptr<const BaseAttributes> attributes;
    absl::InlinedVector<DmlInputTensorKey, 6> input_tensors;

    DmlKernelKey Clone() const; // Performs a deep copy
    bool operator==(const DmlKernelKey& other) const;

    template <typename H>
    friend H AbslHashValue(H h, const DmlKernelKey& kernel_key)
    {
        if (kernel_key.attributes)
        {
            return H::combine(
                std::move(h),
                kernel_key.op_type_name,
                kernel_key.attributes->GetNamedAttributes(),
                kernel_key.input_tensors);
        }

        return H::combine(
            std::move(h),
            kernel_key.op_type_name,
            kernel_key.input_tensors);
    }
};

} // namespace tfdml
