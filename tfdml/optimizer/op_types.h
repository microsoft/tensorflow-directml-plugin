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

constexpr char kAttrOutputShape[] = "_output_shapes";
constexpr char kOpDataFormatVecPermute[] = "DataFormatVecPermute";
constexpr char kOpDataFormatDimMap[] = "DataFormatDimMap";
constexpr char kAttrSrcFormat[] = "src_format";
constexpr char kAttrDstFormat[] = "dst_format";
constexpr char kOpConst[] = "Const";
constexpr char kOptimizedSuffix[] = "LayoutOptimizer";

bool IsDefaultLayoutSensitiveOp(const tensorflow::NodeDef& node);
bool IsDefaultLayoutAgnosticOp(const tensorflow::NodeDef& node);
bool IsAvgPoolGrad(const tensorflow::NodeDef& node);
bool IsBiasAddGrad(const tensorflow::NodeDef& node);
bool IsConv2DBackpropFilter(const tensorflow::NodeDef& node);
bool IsConv2DBackpropInput(const tensorflow::NodeDef& node);
bool IsDepthwiseConv2dNativeBackpropFilter(const tensorflow::NodeDef& node);
bool IsDepthwiseConv2dNativeBackpropInput(const tensorflow::NodeDef& node);
bool IsFusedBatchNormEx(const tensorflow::NodeDef& node);
bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node);
bool IsMaxPoolV2(const tensorflow::NodeDef& node);
bool IsMaxPoolGrad(const tensorflow::NodeDef& node);
bool IsMaxPoolGradV2(const tensorflow::NodeDef& node);
bool IsMaxPoolGradGradV1(const tensorflow::NodeDef& node);
bool IsMaxPoolGradGradV2(const tensorflow::NodeDef& node);
bool IsConv2D(const tensorflow::NodeDef& node);
bool IsConv2DBackpropInput(const tensorflow::NodeDef& node);
bool IsAddN(const tensorflow::NodeDef& node);
bool IsAdd(const tensorflow::NodeDef& node);
bool IsApproximateEqual(const tensorflow::NodeDef& node);
bool IsEqual(const tensorflow::NodeDef& node);
bool IsGreater(const tensorflow::NodeDef& node);
bool IsGreaterEqual(const tensorflow::NodeDef& node);
bool IsLess(const tensorflow::NodeDef& node);
bool IsLessEqual(const tensorflow::NodeDef& node);
bool IsNotEqual(const tensorflow::NodeDef& node);
bool IsComplex(const tensorflow::NodeDef& node);
bool IsDiv(const tensorflow::NodeDef& node);
bool IsFloorDiv(const tensorflow::NodeDef& node);
bool IsIgamma(const tensorflow::NodeDef& node);
bool IsIgammac(const tensorflow::NodeDef& node);
bool IsLogicalAnd(const tensorflow::NodeDef& node);
bool IsLogicalOr(const tensorflow::NodeDef& node);
bool IsMaximum(const tensorflow::NodeDef& node);
bool IsMinimum(const tensorflow::NodeDef& node);
bool IsMod(const tensorflow::NodeDef& node);
bool IsMul(const tensorflow::NodeDef& node);
bool IsPolygamma(const tensorflow::NodeDef& node);
bool IsPow(const tensorflow::NodeDef& node);
bool IsRealDiv(const tensorflow::NodeDef& node);
bool IsSquaredDifference(const tensorflow::NodeDef& node);
bool IsSub(const tensorflow::NodeDef& node);
bool IsTruncateDiv(const tensorflow::NodeDef& node);
bool IsTruncateMod(const tensorflow::NodeDef& node);
bool IsZeta(const tensorflow::NodeDef& node);
bool IsIdentityN(const tensorflow::NodeDef& node);
bool IsMerge(const tensorflow::NodeDef& node);
bool IsMirrorPad(const tensorflow::NodeDef& node);
bool IsMirrorPadGrad(const tensorflow::NodeDef& node);
bool IsPad(const tensorflow::NodeDef& node);
bool IsSelect(const tensorflow::NodeDef& node);
bool IsSwitch(const tensorflow::NodeDef& node);
bool IsBetainc(const tensorflow::NodeDef& node);
bool IsEluGrad(const tensorflow::NodeDef& node);
bool IsInvGrad(const tensorflow::NodeDef& node);
bool IsLeakyReluGrad(const tensorflow::NodeDef& node);
bool IsReciprocalGrad(const tensorflow::NodeDef& node);
bool IsRelu6Grad(const tensorflow::NodeDef& node);
bool IsReluGrad(const tensorflow::NodeDef& node);
bool IsRsqrtGrad(const tensorflow::NodeDef& node);
bool IsSeluGrad(const tensorflow::NodeDef& node);
bool IsSigmoidGrad(const tensorflow::NodeDef& node);
bool IsSoftplusGrad(const tensorflow::NodeDef& node);
bool IsSoftsignGrad(const tensorflow::NodeDef& node);
bool IsSqrtGrad(const tensorflow::NodeDef& node);
bool IsTanhGrad(const tensorflow::NodeDef& node);
bool IsConcat(const tensorflow::NodeDef& node);
bool IsReverseV2(const tensorflow::NodeDef& node);
bool IsTile(const tensorflow::NodeDef& node);
bool IsShape(const tensorflow::NodeDef& node);
bool IsShapeN(const tensorflow::NodeDef& node);
bool IsFill(const tensorflow::NodeDef& node);
bool IsSlice(const tensorflow::NodeDef& node);
bool IsSplit(const tensorflow::NodeDef& node);
bool IsSqueeze(const tensorflow::NodeDef& node);
bool IsSplitV(const tensorflow::NodeDef& node);
bool IsStridedSlice(const tensorflow::NodeDef& node);
bool IsSum(const tensorflow::NodeDef& node);
bool IsMean(const tensorflow::NodeDef& node);
bool IsProd(const tensorflow::NodeDef& node);
bool IsMax(const tensorflow::NodeDef& node);
bool IsMin(const tensorflow::NodeDef& node);
bool IsAll(const tensorflow::NodeDef& node);
bool IsAny(const tensorflow::NodeDef& node);
bool IsReduceOp(const tensorflow::NodeDef& node);
bool IsFloatingDataType(tensorflow::DataType dtype);
bool IsNonFloatingConv2D(const MutableNodeView& node);
bool IsAtan2(const tensorflow::NodeDef& node);
bool IsComparisonOp(const tensorflow::NodeDef& node);
bool IsBinaryOp(const tensorflow::NodeDef& node);
bool IsTernaryOp(const tensorflow::NodeDef& node);
bool IsUnaryGrad(const tensorflow::NodeDef& node);
bool IsLayoutSensitiveOp(const tensorflow::NodeDef& node);
bool IsLayoutAgnosticOp(const tensorflow::NodeDef& node);
bool IsDataFormatOp(const MutableNodeView& node);
bool IsTranspose(const tensorflow::NodeDef& node);
bool IsArg(const tensorflow::NodeDef& node);
} // namespace tfdml
