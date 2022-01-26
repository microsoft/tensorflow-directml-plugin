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

#pragma once

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tfdml
{
class TensorShape
{
  public:
    TensorShape();
    TensorShape(std::initializer_list<int64_t> dim_sizes);
    TensorShape(absl::Span<const int64_t> dim_sizes);
    TensorShape(absl::InlinedVector<int64_t, 5>&& dim_sizes);

    // Maximum number of dimensions in a tensor.
    // It's 254 because 255 = kUnknownRank is used to represent unknown rank.
    static constexpr int MaxDimensions() { return 254; }

    void AddDim(int64_t dim_size);
    void InsertDim(int index, int64_t dim_size);
    void RemoveLastDims(int num_dims);
    int64_t dim_size(int dim_index) const;
    int64_t dims() const;
    absl::InlinedVector<int64_t, 4> dim_sizes() const;
    int64_t num_elements() const;
    int64_t* data();
    const int64_t* data() const;
    std::string DebugString() const;
    void set_dim(int dim_index, int64_t dim);
    bool IsSameSize(const TensorShape& other) const;

    friend bool operator==(const TensorShape& a, const TensorShape& b);
    friend bool operator!=(const TensorShape& a, const TensorShape& b);

    template <typename H>
    friend H AbslHashValue(H h, const TensorShape& shape)
    {
        auto result = H::combine(std::move(h), shape.dim_sizes_);
        return result;
    }

    template <int NDIMS, typename IndexType = Eigen::DenseIndex>
    Eigen::DSizes<IndexType, NDIMS> AsEigenDSizes() const;

    // Same as `AsEigenDSizes()` but allows for `NDIMS > dims()` -- in
    // which case we pad the rest of the sizes with 1.
    template <int NDIMS, typename IndexType = Eigen::DenseIndex>
    Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const;

  private:
    absl::InlinedVector<int64_t, 5> dim_sizes_;
    int64_t num_elements_;

    void UpdateNumElements();
};

bool operator==(const TensorShape& a, const TensorShape& b);
bool operator!=(const TensorShape& a, const TensorShape& b);

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizes() const
{
    assert(NDIMS == TensorShape::dims());
    return AsEigenDSizesWithPadding<NDIMS, IndexType>();
}

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizesWithPadding() const
{
    assert(NDIMS >= TensorShape::dims());
    static_assert(NDIMS <= TensorShape::MaxDimensions(), "Too many dimensions");
    Eigen::DSizes<IndexType, NDIMS> dsizes;
    for (int d = 0; d < dims(); d++)
    {
        dsizes[d] = static_cast<IndexType>(dim_size(d));
    }
    for (int d = dims(); d < NDIMS; d++)
    {
        dsizes[d] = 1;
    }
    return dsizes;
}

} // namespace tfdml
