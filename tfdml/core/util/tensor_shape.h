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

namespace tensorflow
{
class TensorShapeProto;
}

namespace tfdml
{
class TensorShape
{
  public:
    TensorShape() = default;
    TensorShape(const tensorflow::TensorShapeProto& proto);
    TensorShape(std::initializer_list<int64_t> dim_sizes);
    TensorShape(absl::Span<const int64_t> dim_sizes);
    TensorShape(absl::InlinedVector<int64_t, 5>&& dim_sizes);

    void AddDim(int64_t dim_size);
    void InsertDim(int index, int64_t dim_size);
    void RemoveLastDims(int num_dims);
    int64_t dim_size(int dim_index) const;
    int64_t dims() const;
    int64_t num_elements() const;
    int64_t* data();
    const int64_t* data() const;
    std::string DebugString() const;
    void set_dim(int dim_index, int64_t dim);

    friend bool operator==(const TensorShape& a, const TensorShape& b);
    friend bool operator!=(const TensorShape& a, const TensorShape& b);

    template <typename H> friend H AbslHashValue(H h, const TensorShape& shape)
    {
        auto result = H::combine(std::move(h), shape.dim_sizes_);
        return result;
    }

  private:
    absl::InlinedVector<int64_t, 5> dim_sizes_;
    int64_t num_elements_;

    void UpdateNumElements();
};

bool operator==(const TensorShape& a, const TensorShape& b);
bool operator!=(const TensorShape& a, const TensorShape& b);

} // namespace tfdml
