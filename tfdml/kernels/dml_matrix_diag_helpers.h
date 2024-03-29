/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing perMatrixDiagPartmissions
and limitations under the License.
==============================================================================*/

namespace dml
{
dml::Expression MatrixDiag(
    dml::Graph& scope,
    dml::Expression diag,
    int32_t k_min,
    int32_t k_max,
    float padding_value,
    int64_t out_height,
    int64_t out_width,
    bool align_sup_left,
    bool align_sub_left);

dml::Expression MatrixDiagPart(
    dml::Graph& scope,
    dml::Expression m,
    int32_t k0,
    int32_t k1,
    float padding_value,
    uint32_t out_height,
    uint32_t out_width,
    bool align_sup_left,
    bool align_sub_left);
} // namespace dml
