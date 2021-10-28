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

#include "tensorflow/c/tf_datatype.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tfdml
{
enum MemoryType
{
    DEVICE_MEMORY = 0,
    HOST_MEMORY = 1,
};

template <TF_DataType T> struct EnumToDataType
{
};

template <> struct EnumToDataType<TF_INT64>
{
    using T = int64_t;
};
template <> struct EnumToDataType<TF_INT32>
{
    using T = int32_t;
};
template <> struct EnumToDataType<TF_INT16>
{
    using T = int16_t;
};
template <> struct EnumToDataType<TF_INT8>
{
    using T = int8_t;
};
template <> struct EnumToDataType<TF_UINT64>
{
    using T = uint64_t;
};
template <> struct EnumToDataType<TF_UINT32>
{
    using T = uint32_t;
};
template <> struct EnumToDataType<TF_UINT16>
{
    using T = uint16_t;
};
template <> struct EnumToDataType<TF_UINT8>
{
    using T = uint8_t;
};
template <> struct EnumToDataType<TF_BOOL>
{
    using T = bool;
};
template <> struct EnumToDataType<TF_FLOAT>
{
    using T = float;
};
template <> struct EnumToDataType<TF_DOUBLE>
{
    using T = double;
};
template <> struct EnumToDataType<TF_HALF>
{
    using T = Eigen::half;
};
template <> struct EnumToDataType<TF_BFLOAT16>
{
    using T = Eigen::bfloat16;
};
template <> struct EnumToDataType<TF_QINT8>
{
    using T = Eigen::QInt8;
};
template <> struct EnumToDataType<TF_QINT16>
{
    using T = Eigen::QInt16;
};
template <> struct EnumToDataType<TF_QUINT16>
{
    using T = Eigen::QUInt16;
};
template <> struct EnumToDataType<TF_QINT32>
{
    using T = Eigen::QInt32;
};
template <> struct EnumToDataType<TF_COMPLEX64>
{
    using T = std::complex<float>;
};
template <> struct EnumToDataType<TF_COMPLEX128>
{
    using T = std::complex<double>;
};

template <typename T> TF_DataType DataTypeToEnum();

template <> inline TF_DataType DataTypeToEnum<int64_t>() { return TF_INT64; }
template <> inline TF_DataType DataTypeToEnum<int32_t>() { return TF_INT32; }
template <> inline TF_DataType DataTypeToEnum<int16_t>() { return TF_INT16; }
template <> inline TF_DataType DataTypeToEnum<int8_t>() { return TF_INT8; }
template <> inline TF_DataType DataTypeToEnum<uint64_t>() { return TF_UINT64; }
template <> inline TF_DataType DataTypeToEnum<uint32_t>() { return TF_UINT32; }
template <> inline TF_DataType DataTypeToEnum<uint16_t>() { return TF_UINT16; }
template <> inline TF_DataType DataTypeToEnum<uint8_t>() { return TF_UINT8; }
template <> inline TF_DataType DataTypeToEnum<bool>() { return TF_BOOL; }
template <> inline TF_DataType DataTypeToEnum<float>() { return TF_FLOAT; }
template <> inline TF_DataType DataTypeToEnum<double>() { return TF_DOUBLE; }
template <> inline TF_DataType DataTypeToEnum<Eigen::half>() { return TF_HALF; }
template <> inline TF_DataType DataTypeToEnum<Eigen::bfloat16>()
{
    return TF_BFLOAT16;
}
template <> inline TF_DataType DataTypeToEnum<Eigen::QInt8>()
{
    return TF_QINT8;
}
template <> inline TF_DataType DataTypeToEnum<Eigen::QInt16>()
{
    return TF_QINT16;
}
template <> inline TF_DataType DataTypeToEnum<Eigen::QUInt16>()
{
    return TF_QUINT16;
}
template <> inline TF_DataType DataTypeToEnum<Eigen::QInt32>()
{
    return TF_QINT32;
}
template <> inline TF_DataType DataTypeToEnum<std::complex<float>>()
{
    return TF_COMPLEX64;
}
template <> inline TF_DataType DataTypeToEnum<std::complex<double>>()
{
    return TF_COMPLEX128;
}

std::string DataTypeString(TF_DataType dtype);

} // namespace tfdml
