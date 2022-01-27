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

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/c/tf_datatype.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/tensor_shape.h"
#include "tfdml/runtime_adapter/tensor_types.h"
#include "tfdml/runtime_adapter/types.h"

struct TF_Tensor;

namespace tfdml
{

class Tensor
{
  public:
    Tensor();
    Tensor(TF_DataType data_type);
    Tensor(TF_Tensor* tensor);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);
    int64_t AllocatedBytes() const;
    absl::string_view tensor_data() const;
    TF_DataType dtype() const;
    TensorShape shape() const;
    int64_t NumElements() const;
    Tensor DeepCopy() const;
    int64_t TotalBytes() const;
    int64_t dims() const;
    int64_t dim_size(int64_t dim_index) const;
    void* raw_data() const;
    bool IsInitialized() const;
    TF_Tensor* raw() const;
    bool CopyFrom(const Tensor& other, const TensorShape& shape);

    template <typename T, size_t NDIMS>
    typename TTypes<T, NDIMS>::Tensor tensor();

    template <typename T, size_t NDIMS>
    typename TTypes<T, NDIMS>::ConstTensor tensor() const;

    template <typename T>
    T* base()
    {
        return reinterpret_cast<T*>(raw_data());
    }

    template <typename T>
    const T* base() const
    {
        return reinterpret_cast<T*>(raw_data());
    }

    std::string DebugString() const;

    template <typename H>
    friend H AbslHashValue(H h, const Tensor& tensor)
    {
        auto result = H::combine(
            std::move(h),
            tensor.shape(),
            tensor.dtype(),
            tensor.tensor_data());
        return result;
    }

    template <typename T>
    typename TTypes<T>::ConstMatrix matrix() const
    {
        return tensor<T, 2>();
    }

    bool IsAligned() const
    {
#if EIGEN_MAX_ALIGN_BYTES == 0
        return true;
#else
        const void* ptr = base<void>();
        return dtype() == TF_STRING ||
               (reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0);
#endif
    }

    bool IsSameSize(const Tensor& other) const;

  private:
    static TF_Tensor* shallow_copy(const Tensor& other);

    std::shared_ptr<TF_Tensor> tensor_;
    TensorShape shape_;
};

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor()
{
    CHECK(IsAligned());
    CHECK(dtype() == DataTypeToEnum<T>());
    return typename TTypes<T, NDIMS>::Tensor(
        base<T>(),
        shape().AsEigenDSizes<NDIMS>());
}
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::tensor() const
{
    CHECK(IsAligned());
    CHECK(dtype() == DataTypeToEnum<T>());
    return typename TTypes<T, NDIMS>::ConstTensor(
        base<const T>(),
        shape().AsEigenDSizes<NDIMS>());
}

} // namespace tfdml
