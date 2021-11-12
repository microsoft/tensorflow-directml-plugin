/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensor_shape_utils.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_datatype.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/tensor.h"

namespace tfdml
{
TensorShape TensorShapeUtils::MakeShape(const Tensor& tensor)
{
    const TF_DataType dtype = tensor.dtype();
    int64_t num_elements = tensor.NumElements();
    absl::string_view raw_data = tensor.tensor_data();

    if (dtype == TF_INT32)
    {
        const int32_t* int32_data =
            reinterpret_cast<const int32_t*>(raw_data.data());
        absl::InlinedVector<int64_t, 5> dim_sizes;
        dim_sizes.reserve(num_elements);

        for (int i = 0; i < num_elements; ++i)
        {
            dim_sizes.push_back(int32_data[i]);
        }

        return TensorShape(std::move(dim_sizes));
    }

    CHECK(dtype == TF_INT64);
    const int64_t* int64_data =
        reinterpret_cast<const int64_t*>(raw_data.data());
    return TensorShape(absl::Span<const int64_t>(int64_data, num_elements));
}

bool TensorShapeUtils::IsScalar(const TensorShape& shape)
{
    return shape.dims() == 0;
}

bool TensorShapeUtils::IsVector(const TensorShape& shape)
{
    return shape.dims() == 1;
}

bool TensorShapeUtils::IsVectorOrHigher(const TensorShape& shape)
{
    return shape.dims() >= 1;
}

bool TensorShapeUtils::IsMatrix(const TensorShape& shape)
{
    return shape.dims() == 2;
}
} // namespace tfdml
