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

#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow
{
class NodeDef;
} // namespace tensorflow

namespace tfdml
{
class MutableNodeView;

constexpr char kOpDataFormatVecPermute[] = "DataFormatVecPermute";
constexpr char kOpDataFormatDimMap[] = "DataFormatDimMap";

bool IsBiasAdd(const tensorflow::NodeDef& node);
bool IsConstant(const tensorflow::NodeDef& node);
bool IsConv2D(const tensorflow::NodeDef& node);
bool IsElu(const tensorflow::NodeDef& node);
bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node);
bool IsLeakyRelu(const tensorflow::NodeDef& node);
bool IsMerge(const tensorflow::NodeDef& node);
bool IsNextIteration(const tensorflow::NodeDef& node);
bool IsPad(const tensorflow::NodeDef& node);
bool IsPlaceholder(const tensorflow::NodeDef& node);
bool IsRelu(const tensorflow::NodeDef& node);
bool IsRelu6(const tensorflow::NodeDef& node);
bool IsSymbolicGradient(const tensorflow::NodeDef& node);
bool IsTranspose(const tensorflow::NodeDef& node);

} // namespace tfdml
