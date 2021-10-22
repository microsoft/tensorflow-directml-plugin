/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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

#include "tensor_shape.h"

#include <numeric>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tfdml
{
TensorShape::TensorShape(std::initializer_list<int64_t> dim_sizes)
    : TensorShape(absl::Span<const int64_t>(dim_sizes))
{
}

TensorShape::TensorShape(absl::Span<const int64_t> dim_sizes)
    : dim_sizes_(dim_sizes.begin(), dim_sizes.end())
{
    UpdateNumElements();
}

TensorShape::TensorShape(absl::InlinedVector<int64_t, 5>&& dim_sizes)
    : dim_sizes_(std::move(dim_sizes))
{
    UpdateNumElements();
}

TensorShape::TensorShape(const tensorflow::TensorShapeProto& proto)
{
    dim_sizes_.reserve(proto.dim_size());

    for (const auto& d : proto.dim())
    {
        dim_sizes_.push_back(d.size());
        num_elements_ *= d.size();
    }
}

bool operator==(const TensorShape& a, const TensorShape& b)
{
    return a.dim_sizes_ == b.dim_sizes_;
}

bool operator!=(const TensorShape& a, const TensorShape& b)
{
    return a.dim_sizes_ != b.dim_sizes_;
}

void TensorShape::AddDim(int64_t dim_size)
{
    dim_sizes_.push_back(dim_size);
    num_elements_ *= dim_size;
}

void TensorShape::InsertDim(int index, int64_t dim_size)
{
    assert(index <= dim_sizes_.size());
    dim_sizes_.insert(dim_sizes_.begin() + index, dim_size);
    num_elements_ *= dim_size;
}

void TensorShape::RemoveLastDims(int num_dims)
{
    assert(num_dims <= dim_sizes_.size());
    dim_sizes_.resize(dim_sizes_.size() - num_dims);
}

int64_t TensorShape::dim_size(int dim_index) const
{
    assert(dim_index < dim_sizes_.size());
    return dim_sizes_[dim_index];
}

int64_t TensorShape::dims() const { return dim_sizes_.size(); }

void TensorShape::set_dim(int dim_index, int64_t dim)
{
    assert(dim_index < dim_sizes_.size());

    if (dim_sizes_[dim_index] == dim)
    {
        return;
    }

    if (dim_sizes_[dim_index] == 0)
    {
        dim_sizes_[dim_index] = dim;
        UpdateNumElements();
    }
    else
    {
        num_elements_ /= dim_sizes_[dim_index];
        num_elements_ *= dim;
        dim_sizes_[dim_index] = dim;
    }
}

absl::InlinedVector<int64_t, 4> TensorShape::dim_sizes() const {
  absl::InlinedVector<int64_t, 4> result;
  for (auto dim_size : dim_sizes_) {
    result.push_back(dim_size);
  }
  return result;
}

int64_t TensorShape::num_elements() const { return num_elements_; }

int64_t* TensorShape::data() { return dim_sizes_.data(); }
const int64_t* TensorShape::data() const { return dim_sizes_.data(); }

std::string TensorShape::DebugString() const
{
    std::string s = "[";

    for (int i = 0; i < dim_sizes_.size(); ++i)
    {
        if (i > 0)
        {
            absl::StrAppend(&s, ",");
        }

        if (dim_sizes_[i] < 0)
        {
            absl::StrAppend(&s, "?");
        }
        else
        {
            absl::StrAppend(&s, dim_sizes_[i]);
        }
    }
    absl::StrAppend(&s, "]");
    return s;
}

void TensorShape::UpdateNumElements()
{
    num_elements_ = std::accumulate(
        dim_sizes_.begin(),
        dim_sizes_.end(),
        1LL,
        std::multiplies<int64_t>());
}

} // namespace tfdml
