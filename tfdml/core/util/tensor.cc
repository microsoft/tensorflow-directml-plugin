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

#include "tensor.h"

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tstring.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tfdml/core/util/macros.h"
#include "tfdml/core/util/status.h"

namespace tfdml {
static void DeleteTensor(TF_Tensor* tensor) {
  if (tensor) {
    TF_DeleteTensor(tensor);
  }
}

static std::shared_ptr<TF_Tensor> MakeTensor(TF_Tensor* tensor) {
  return std::shared_ptr<TF_Tensor>(tensor, DeleteTensor);
}

static TensorShape MakeShape(TF_Tensor* tensor) {
  assert(tensor != nullptr);
  int num_dims = TF_NumDims(tensor);
  absl::InlinedVector<int64_t, 5> dim_sizes;
  dim_sizes.reserve(num_dims);

  for (int dim_index = 0; dim_index < num_dims; ++dim_index) {
    int64_t dim_size = TF_Dim(tensor, dim_index);
    dim_sizes.push_back(dim_size);
  }

  return TensorShape(std::move(dim_sizes));
}

static TF_Tensor* init_empty_tensor() {
  static constexpr int64_t empty_sizes[1] = {0};
  return TF_AllocateTensor(TF_FLOAT, empty_sizes, 1, 0);
}

TF_Tensor* Tensor::shallow_copy(const Tensor& other) {
  if (other.dtype() == TF_RESOURCE) {
  }
  TF_Tensor* copy_tensor = nullptr;

  int num_dims = TF_NumDims(other.tensor_.get());

  std::vector<int64_t> sizes(num_dims);

  for (int i = 0; i < num_dims; ++i) {
    sizes[i] = TF_Dim(other.tensor_.get(), i);
  }

  if (other.dtype() == TF_RESOURCE) {
    copy_tensor = TF_AllocateTensor(TF_RESOURCE, sizes.data(), num_dims,
                                    other.TotalBytes());
  } else {
    copy_tensor = init_empty_tensor();

    Status status;
    TF_TensorBitcastFrom(other.tensor_.get(), other.dtype(), copy_tensor,
                         other.shape().data(), other.shape().dims(),
                         status.raw());

    if (!status.ok()) {
      LogFatal(status.error_message());
    }
  }

  return copy_tensor;
}

Tensor::Tensor() : Tensor(TF_FLOAT) {}

Tensor::Tensor(TF_DataType data_type) {
  tensor_ = MakeTensor(init_empty_tensor());
  shape_ = MakeShape(tensor_.get());
}

Tensor::Tensor(TF_Tensor* tensor) {
  assert(tensor != nullptr);
  tensor_ = MakeTensor(tensor);
  shape_ = MakeShape(tensor);

  if (dtype() == TF_RESOURCE) {
    auto serialized_view = tensor_data();
    std::string serialized_data(serialized_view.data(), serialized_view.size());

    resource_handle_ = std::make_shared<tensorflow::ResourceHandleProto>();
    CHECK(resource_handle_->ParseFromString(serialized_data));
  }
}

Tensor::Tensor(const Tensor& other)
    : tensor_(MakeTensor(shallow_copy(other))),
      shape_(other.shape()),
      resource_handle_(other.resource_handle_) {}

Tensor& Tensor::operator=(const Tensor& other) {
  tensor_ = MakeTensor(shallow_copy(other));
  shape_ = other.shape();
  resource_handle_ = other.resource_handle_;
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
  tensor_ = std::move(other.tensor_);
  shape_ = std::move(other.shape_);
  resource_handle_ = std::move(other.resource_handle_);
  other.tensor_ = nullptr;
  other.resource_handle_ = nullptr;
  return *this;
}

bool Tensor::IsInitialized() const {
  return (raw_data() != nullptr) || NumElements() == 0;
}

int64_t Tensor::AllocatedBytes() const {
  assert(tensor_ != nullptr);
  return TF_TensorByteSize(tensor_.get());
}

TF_DataType Tensor::dtype() const {
  assert(tensor_ != nullptr);
  return TF_TensorType(tensor_.get());
}

absl::string_view Tensor::tensor_data() const {
  assert(tensor_ != nullptr);
  size_t tensor_size = TF_TensorByteSize(tensor_.get());
  char* tensor_data = reinterpret_cast<char*>(raw_data());
  return absl::string_view(tensor_data, tensor_size);
}

const TensorShape& Tensor::shape() const { return shape_; }

int64_t Tensor::dims() const { return shape_.dims(); }

int64_t Tensor::dim_size(int64_t dim_index) const {
  return shape_.dim_size(dim_index);
}

int64_t Tensor::NumElements() const {
  assert(tensor_ != nullptr);
  return TF_TensorElementCount(tensor_.get());
}

int64_t Tensor::TotalBytes() const {
  assert(tensor_ != nullptr);
  CHECK(dtype() != TF_STRING);
  return TF_DataTypeSize(dtype()) * NumElements();
}

std::string Tensor::DebugString() const {
  int num_dims = TF_NumDims(tensor_.get());

  // A TF_Tensor cannot have an unknown rank.
  CHECK(num_dims >= 0);
  std::string s = "[";
  for (int i = 0; i < num_dims; ++i) {
    if (i > 0) {
      absl::StrAppend(&s, ",");
    }

    int64_t dim = TF_Dim(tensor_.get(), i);
    // A TF_Tensor cannot have an unknown dimension.
    CHECK(dim >= 0);
    absl::StrAppend(&s, dim);
  }
  absl::StrAppend(&s, "]");
  return s;
}

void* Tensor::raw_data() const { return TF_TensorData(tensor_.get()); }

Tensor Tensor::DeepCopy() const {
  assert(tensor_ != nullptr);
  TF_Tensor* tensor = TF_AllocateTensor(dtype(), shape_.data(), shape_.dims(),
                                        AllocatedBytes());

  absl::string_view input_data = tensor_data();
  void* output_data = TF_TensorData(tensor);
  memcpy(reinterpret_cast<char*>(output_data), input_data.data(),
         input_data.size());

  return Tensor(tensor);
}

const TF_Tensor* Tensor::raw() const { return tensor_.get(); }

}  // namespace tfdml
