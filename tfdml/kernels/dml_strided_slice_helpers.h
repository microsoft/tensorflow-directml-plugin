/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/tensor.h"

namespace tfdml
{
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor,
    const Tensor* end_tensor,
    const Tensor& strides_tensor,
    const TensorShape& input_shape,
    int32_t begin_mask_spec,
    int32_t end_mask_spec,
    const int32_t ellipsis_mask,
    int32_t new_axis_mask,
    int32_t shrink_axis_mask,
    TensorShape* processing_shape,
    TensorShape* final_shape,
    bool* is_identity,
    bool* is_simple_slice,
    bool* slice_dim0,
    absl::InlinedVector<int64_t, 4>* begin,
    absl::InlinedVector<int64_t, 4>* end,
    absl::InlinedVector<int64_t, 4>* strides);
}