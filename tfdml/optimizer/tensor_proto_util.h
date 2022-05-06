/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.pb.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"

namespace tfdml
{
#define RETURN_TENSOR_VALUE(TENSOR, DTYPE)                                     \
    do                                                                         \
    {                                                                          \
        if (TENSOR.version_number() == 0 && TENSOR.DTYPE##_val_size() == 1)    \
        {                                                                      \
            return TENSOR.DTYPE##_val(0);                                      \
        }                                                                      \
        assert(TENSOR.DTYPE##_val_size() == GetNumElements(TENSOR));           \
        return TENSOR.DTYPE##_val(elem_index);                                 \
    } while (0)

int GetNumElements(const tensorflow::TensorProto& tensor);

template <typename T>
T GetTensorElement(const tensorflow::TensorProto& tensor, int elem_index)
{
    assert(elem_index < GetNumElements(tensor));

    if (!tensor.tensor_content().empty())
    {
        assert(
            tensor.tensor_content().size() / sizeof(T) ==
            GetNumElements(tensor));

        const T* values =
            reinterpret_cast<const T*>(tensor.tensor_content().c_str());

        return values[elem_index];
    }

    if constexpr (std::is_same_v<T, Eigen::half>)
    {
        assert(tensor.dtype() == tensorflow::DT_HALF);
        if (tensor.version_number() == 0 && tensor.half_val_size() == 1)
        {
            auto half_val = tensor.half_val(0);
            return *reinterpret_cast<Eigen::half*>(&half_val);
        }
        assert(tensor.half_val_size() == GetNumElements(tensor));
        auto half_val = tensor.half_val(elem_index);
        return *reinterpret_cast<Eigen::half*>(&half_val);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        assert(tensor.dtype() == tensorflow::DT_FLOAT);
        RETURN_TENSOR_VALUE(tensor, float);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        assert(tensor.dtype() == tensorflow::DT_DOUBLE);
        RETURN_TENSOR_VALUE(tensor, double);
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        assert(tensor.dtype() == tensorflow::DT_UINT32);
        RETURN_TENSOR_VALUE(tensor, uint32);
    }
    else if constexpr (std::is_same_v<T, uint64_t>)
    {
        assert(tensor.dtype() == tensorflow::DT_UINT64);
        RETURN_TENSOR_VALUE(tensor, uint64);
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        assert(tensor.dtype() == tensorflow::DT_INT64);
        RETURN_TENSOR_VALUE(tensor, int64);
    }
    else
    {
        // DT_INT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8.
        static_assert(
            std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> ||
            std::is_same_v<T, uint16_t> || std::is_same_v<T, int8_t> ||
            std::is_same_v<T, uint8_t>);

        assert(
            tensor.dtype() == tensorflow::DT_INT32 ||
            tensor.dtype() == tensorflow::DT_INT16 ||
            tensor.dtype() == tensorflow::DT_UINT16 ||
            tensor.dtype() == tensorflow::DT_INT8 ||
            tensor.dtype() == tensorflow::DT_UINT8);

        RETURN_TENSOR_VALUE(tensor, int);
    }
}

} // namespace tfdml