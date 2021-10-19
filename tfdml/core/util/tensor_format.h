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

namespace tfdml
{
enum TensorFormat
{
    // FORMAT_NHWC is the default format in TensorFlow.
    FORMAT_NHWC = 0,

    // FORMAT_NCHW often improves performance on GPUs.
    FORMAT_NCHW = 1,

    // NCHW_VECT_C is the most performant tensor format for cudnn6's quantized
    // int8 convolution and fused convolution. It is laid out in the same order
    // as NCHW, except that the size of the Channels dimension is divided by 4,
    // and a new dimension of size 4 is appended, which packs 4 adjacent channel
    // activations for the same pixel into an int32. Thus an NCHW format tensor
    // with dimensions [N, C, H, W] would have dimensions [N, C/4, H, W, 4] in
    // NCHW_VECT_C format.
    // A pre-condition of this format is that C must be a multiple of 4.
    FORMAT_NCHW_VECT_C = 2,

    // Similar to NHWC, but the size of the W dimension is divided by 4, and a
    // new dimension of size 4 is appended, which packs 4 adjacent activations
    // in the width dimension.
    FORMAT_NHWC_VECT_W = 3,

    // Note: although the current code in this file assumes VECT_C and VECT_W
    // enums imply int8x4 vectors, this should not be relied upon.
    // In the future we may change the meaning of these enums to include vectors
    // of other types such as int16x2, with op implementations automatically
    // determining which format is implied based on the datatype.

    // FORMAT_HWNC is for TPUs.
    FORMAT_HWNC = 4,

    // FORMAT_HWCN is for TPUs.
    FORMAT_HWCN = 5,
};
} // namespace tfdml
