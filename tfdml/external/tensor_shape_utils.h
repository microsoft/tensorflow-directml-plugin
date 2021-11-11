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

#include "tfdml/core/util/tensor_shape.h"

namespace tfdml
{
class Tensor;

class TensorShapeUtils
{
  public:
    // Makes a shape from a tensor. The datatype of the tensor must be int32 or
    // int64
    static TensorShape MakeShape(const Tensor& tensor);

    static bool IsScalar(const TensorShape& shape);
    static bool IsVector(const TensorShape& shape);
    static bool IsVectorOrHigher(const TensorShape& shape);
    static bool IsMatrix(const TensorShape& shape);
};
} // namespace tfdml