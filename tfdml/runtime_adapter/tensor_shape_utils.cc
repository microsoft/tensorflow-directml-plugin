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

// Multiply two nonnegative int64's, returning negative for overflow
// If any of the arguments is negative, return negative too.
int64_t MultiplyWithoutOverflow(int64_t x, int64_t y)
{
    if (x < 0)
        return -1;
    if (y < 0)
        return -1;
    if (x == 0)
        return 0;

    // Multiply in uint64 rather than int64 since signed overflow is undefined.
    // Negative values will wrap around to large unsigned values in the casts
    // (see section 4.7 [conv.integral] of the C++14 standard).
    const uint64_t ux = x;
    const uint64_t uy = y;
    const uint64_t uxy = ux * uy;

    // Check if we overflow uint64
    if ((ux | uy) >> 32 != 0)
    {
        // Otherwise, detect overflow using a division
        if (uxy / ux != uy)
            return -1;
    }

    // Cast back to signed. A negative value will signal an error.
    return static_cast<int64_t>(uxy);
}

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

Status TensorShapeUtils::MakeShape(const Tensor& tensor, TensorShape* out)
{
    if (!TensorShapeUtils::IsVector(tensor.shape()))
    {
        return errors::InvalidArgument(
            "shape must be a vector of {int32,int64}, got shape ",
            tensor.shape().DebugString());
    }

    if (tensor.dtype() != TF_INT32 && tensor.dtype() != TF_INT64)
    {
        return errors::InvalidArgument(
            "shape must be a vector of {int32,int64}.");
    }

    int64_t num_dims = tensor.NumElements();

    if (num_dims > TensorShape::MaxDimensions())
    {
        return errors::InvalidArgument("Too many dimensions");
    }

    if (num_dims < 0)
    {
        return errors::InvalidArgument(
            "Negative number of dimensions ",
            num_dims);
    }

    absl::string_view raw_data = tensor.tensor_data();

    for (int64_t i = 0; i < num_dims; ++i)
    {
        int64_t dim;

        if (tensor.dtype() == TF_INT32)
        {
            const int32_t* int32_data =
                reinterpret_cast<const int32_t*>(raw_data.data());
            dim = int32_data[i];
        }
        else
        {
            assert(tensor.dtype() == TF_INT64);
            const int64_t* int64_data =
                reinterpret_cast<const int64_t*>(raw_data.data());
            dim = int64_data[i];
        }

        if (dim < 0)
        {
            return errors::InvalidArgument("Dimension ", dim, " must be >= 0");
        }

        int64_t new_num_elements;
        if (out->num_elements() < 0)
        {
            new_num_elements = -1;
        }
        else
        {
            new_num_elements =
                MultiplyWithoutOverflow(out->num_elements(), dim);
            if (new_num_elements < 0)
            {
                TensorShape bad_shape;
                for (int64_t j = 0; j < num_dims; ++j)
                {
                    if (tensor.dtype() == TF_INT32)
                    {
                        const int32_t* int32_data =
                            reinterpret_cast<const int32_t*>(raw_data.data());
                        bad_shape.AddDim(int32_data[j]);
                    }
                    else
                    {
                        assert(tensor.dtype() == TF_INT64);
                        const int64_t* int64_data =
                            reinterpret_cast<const int64_t*>(raw_data.data());
                        bad_shape.AddDim(int64_data[j]);
                    }
                }
                return errors::InvalidArgument(
                    "Shape ",
                    bad_shape.DebugString(),
                    " would have more than 2**63 - 1 elements");
            }
        }
        out->AddDim(dim);
    }

    return Status::OK();
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
