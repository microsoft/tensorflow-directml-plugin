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

#include "tfdml/optimizer/tensor_proto_util.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tfdml
{
int GetNumElements(const tensorflow::TensorProto& tensor)
{
    assert(tensor.has_tensor_shape());
    const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

    int64_t num_elements = 1;
    for (int i = 0; i < shape.dim_size(); ++i)
    {
        num_elements *= shape.dim(i).size();
    }

    return num_elements;
}
} // namespace tfdml