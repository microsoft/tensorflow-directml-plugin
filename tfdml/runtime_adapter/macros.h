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

#include "tensorflow/c/logging.h"
#include "types.h"

// Compiler attributes
#if defined(__GNUC__)
// Compiler supports GCC-style attributes
#define TFDML_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define TFDML_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
// Non-GCC equivalents
#define TFDML_ATTRIBUTE_NORETURN
#endif

// TF_Log(TF_FATAL, ...) doesn't tell the compiler that it doesn't return, so
// it's inconvient to use in the 'default' block of switch statements or in
// functions that should return a value
template <typename... TArgs>
TFDML_ATTRIBUTE_NORETURN void LogFatal(const char* fmt, TArgs... args)
{
    TF_Log(TF_FATAL, fmt, args...);
    abort();
}

#define CHECK(condition)                                                       \
    if (!(condition))                                                          \
    {                                                                          \
        LogFatal("Check failed: " #condition);                                 \
    }

#define TF_CHECK_OK(val) CHECK(val.ok())

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                                                \
    do                                                                         \
    {                                                                          \
        ::tfdml::Status _status = (__VA_ARGS__);                               \
        if (!_status.ok()) return _status;                                     \
    } while (0)

// The TF_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression TF_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
#define TF_ARRAYSIZE(a)                                                        \
    ((sizeof(a) / sizeof(*(a))) /                                              \
     static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#define OP_REQUIRES(CTX, EXP, STATUS)                                          \
    do                                                                         \
    {                                                                          \
        if (!(EXP))                                                            \
        {                                                                      \
            (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));                   \
            return;                                                            \
        }                                                                      \
    } while (0)

#define OP_REQUIRES_OK(CTX, ...)                                               \
    do                                                                         \
    {                                                                          \
        ::tfdml::Status _s(__VA_ARGS__);                                       \
        if (!_s.ok())                                                          \
        {                                                                      \
            (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);              \
            return;                                                            \
        }                                                                      \
    } while (0)

#define TF_CALL_int64(m) m(int64_t)
#define TF_CALL_int32(m) m(int32_t)
#define TF_CALL_int16(m) m(int16_t)
#define TF_CALL_int8(m) m(int8_t)
#define TF_CALL_uint64(m) m(uint64_t)
#define TF_CALL_uint32(m) m(uint32_t)
#define TF_CALL_uint16(m) m(uint16_t)
#define TF_CALL_uint8(m) m(uint8_t)
#define TF_CALL_bool(m) m(bool)
#define TF_CALL_double(m) m(double)
#define TF_CALL_float(m) m(float)
#define TF_CALL_half(m) m(Eigen::half)
#define TF_CALL_complex64(m) m(std::complex<float>)
#define TF_CALL_complex128(m) m(std::complex<double>)
