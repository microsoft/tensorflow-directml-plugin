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

#pragma once

#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/tensor_shape.h"

namespace tfdml
{
class Tensor;

class TensorShapeUtils
{
  public:
    // Makes a shape from a tensor. The datatype of the tensor must be int32 or
    // int64
    static TensorShape MakeShape(const Tensor& tensor);
    static Status MakeShape(const Tensor& tensor, TensorShape* out);

    static bool IsScalar(const TensorShape& shape);
    static bool IsVector(const TensorShape& shape);
    static bool IsVectorOrHigher(const TensorShape& shape);
    static bool IsMatrix(const TensorShape& shape);
    static bool IsMatrixOrHigher(const TensorShape& shape);
    static bool StartsWith(const TensorShape& shape, const TensorShape& prefix);
};

int64_t MultiplyWithoutOverflow(int64_t x, int64_t y);

} // namespace tfdml