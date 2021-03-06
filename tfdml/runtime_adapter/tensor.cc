/* Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

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
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/status.h"

namespace tfdml
{
static void DeleteTensor(TF_Tensor* tensor)
{
    if (tensor)
    {
        TF_DeleteTensor(tensor);
    }
}

static std::shared_ptr<TF_Tensor> MakeTensor(TF_Tensor* tensor)
{
    return std::shared_ptr<TF_Tensor>(tensor, DeleteTensor);
}

static TensorShape MakeShape(TF_Tensor* tensor)
{
    assert(tensor != nullptr);
    int num_dims = TF_NumDims(tensor);
    absl::InlinedVector<int64_t, 5> dim_sizes;
    dim_sizes.reserve(num_dims);

    for (int dim_index = 0; dim_index < num_dims; ++dim_index)
    {
        int64_t dim_size = TF_Dim(tensor, dim_index);
        dim_sizes.push_back(dim_size);
    }

    return TensorShape(std::move(dim_sizes));
}

static TF_Tensor* init_empty_tensor()
{
    static constexpr int64_t empty_sizes[1] = {0};
    return TF_AllocateTensor(TF_FLOAT, empty_sizes, 1, 0);
}

TF_Tensor* Tensor::shallow_copy(const Tensor& other)
{
    TF_Tensor* copy_tensor = init_empty_tensor();

    Status status;
    TF_TensorBitcastFrom(
        other.tensor_.get(),
        other.dtype(),
        copy_tensor,
        other.shape().data(),
        other.shape().dims(),
        status.raw());

    if (!status.ok())
    {
        LogFatal(status.error_message());
    }

    return copy_tensor;
}

bool Tensor::CopyFrom(const Tensor& other, const TensorShape& shape)
{
    if (other.NumElements() != shape.num_elements()) return false;

    auto new_tensor = MakeTensor(init_empty_tensor());

    Status status;
    TF_TensorBitcastFrom(
        other.tensor_.get(),
        other.dtype(),
        new_tensor.get(),
        shape.data(),
        shape.dims(),
        status.raw());

    if (!status.ok())
    {
        return false;
    }

    tensor_ = std::move(new_tensor);

    return true;
}

bool Tensor::SharesBufferWith(const Tensor& other) const
{
    return raw_data() == other.raw_data();
}

Tensor::Tensor() : Tensor(TF_FLOAT) {}

Tensor::Tensor(TF_DataType data_type)
{
    tensor_ = MakeTensor(init_empty_tensor());
    shape_ = MakeShape(tensor_.get());
}

Tensor::Tensor(TF_Tensor* tensor)
{
    assert(tensor != nullptr);
    tensor_ = MakeTensor(tensor);
    shape_ = MakeShape(tensor);
}

Tensor::Tensor(const Tensor& other)
    : tensor_(MakeTensor(shallow_copy(other))),
      shape_(other.shape())
{
}

Tensor::Tensor(Tensor&& other)
    : tensor_(std::move(other.tensor_)),
      shape_(std::move(other.shape_))
{
    other.tensor_ = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    tensor_ = MakeTensor(shallow_copy(other));
    shape_ = other.shape();
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    tensor_ = std::move(other.tensor_);
    shape_ = std::move(other.shape_);
    other.tensor_ = nullptr;
    return *this;
}

bool Tensor::IsInitialized() const
{
    return (raw_data() != nullptr) || NumElements() == 0;
}

int64_t Tensor::AllocatedBytes() const
{
    assert(tensor_ != nullptr);
    return TF_TensorByteSize(tensor_.get());
}

TF_DataType Tensor::dtype() const
{
    assert(tensor_ != nullptr);
    return TF_TensorType(tensor_.get());
}

absl::string_view Tensor::tensor_data() const
{
    assert(tensor_ != nullptr);
    size_t tensor_size = TF_TensorByteSize(tensor_.get());
    char* tensor_data = reinterpret_cast<char*>(raw_data());
    return absl::string_view(tensor_data, tensor_size);
}

TensorShape Tensor::shape() const { return shape_; }

int64_t Tensor::dims() const { return shape_.dims(); }

int64_t Tensor::dim_size(int64_t dim_index) const
{
    return shape_.dim_size(dim_index);
}

int64_t Tensor::NumElements() const
{
    assert(tensor_ != nullptr);
    return TF_TensorElementCount(tensor_.get());
}

int64_t Tensor::TotalBytes() const
{
    assert(tensor_ != nullptr);
    CHECK(dtype() != TF_STRING);
    return TF_DataTypeSize(dtype()) * NumElements();
}

std::string Tensor::DebugString() const
{
    int num_dims = TF_NumDims(tensor_.get());

    // A TF_Tensor cannot have an unknown rank.
    CHECK(num_dims >= 0);
    std::string s = "[";
    for (int i = 0; i < num_dims; ++i)
    {
        if (i > 0)
        {
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

Tensor Tensor::DeepCopy() const
{
    assert(tensor_ != nullptr);
    TF_Tensor* tensor = TF_AllocateTensor(
        dtype(),
        shape_.data(),
        shape_.dims(),
        AllocatedBytes());

    absl::string_view input_data = tensor_data();
    void* output_data = TF_TensorData(tensor);
    memcpy(
        reinterpret_cast<char*>(output_data),
        input_data.data(),
        input_data.size());

    return Tensor(tensor);
}

TF_Tensor* Tensor::raw() const { return tensor_.get(); }

bool Tensor::IsSameSize(const Tensor& other) const
{
    return shape_.IsSameSize(other.shape());
}

} // namespace tfdml
