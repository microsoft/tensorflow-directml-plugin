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

#include "tfdml/optimizer/op_types.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tfdml/optimizer/graph_view.h"

constexpr char kAttrT[] = "T";

namespace tfdml
{
bool IsDefaultLayoutSensitiveOp(const tensorflow::NodeDef& node)
{
    const std::set<std::string> default_layout_sensitive_ops = {
        "AvgPool",
        "BiasAdd",
        "Conv2D",
        "DepthwiseConv2dNative",
        "DepthToSpace",
        "FusedBatchNorm",
        "FusedBatchNormV2",
        "FusedBatchNormV3",
        "FusedConv2DBiasActivation",
        "_FusedConv2D",
        "MaxPool",
        "SpaceToDepth"};
    return default_layout_sensitive_ops.find(node.op()) !=
           default_layout_sensitive_ops.end();
}

bool IsDefaultLayoutAgnosticOp(const tensorflow::NodeDef& node)
{
    static absl::flat_hash_set<std::string>* agnostic_nodes =
        new absl::flat_hash_set<std::string>(
            {"Abs",
             "Acos",
             "Acosh",
             "Angle",
             "Asin",
             "Asinh",
             "Atan",
             "Atanh",
             "Bitcast",
             "Cast",
             "Ceil",
             "CheckNumerics",
             "ComplexAbs",
             "Conj",
             "Cos",
             "Cosh",
             "Digamma",
             "Elu",
             "Enter",
             "Erf",
             "Erfc",
             "Exit",
             "Exp",
             "Expm1",
             "FakeQuantWithMinMaxVars",
             "FakeQuantWithMinMaxArgs",
             "Floor",
             "GuaranteeConst",
             "Identity",
             "Imag",
             "Inv",
             "IsFinite",
             "IsInf",
             "IsNan",
             "LeakyRelu",
             "Lgamma",
             "Log",
             "LogicalNot",
             "Log1p",
             "Neg",
             "NextIteration",
             "OnesLike",
             "PreventGradient",
             "QuantizeAndDequantizeV2",
             "QuantizeAndDequantizeV3",
             "QuantizeAndDequantizeV4",
             "Real",
             "Reciprocal",
             "Relu",
             "Relu6",
             "Rint",
             "Selu",
             "Sigmoid",
             "Sign",
             "Sin",
             "Sinh",
             "Snapshot",
             "Softplus",
             "Round",
             "Rsqrt",
             "Sqrt",
             "Square",
             "StopGradient",
             "Tan",
             "Tanh",
             "ZerosLike"});
    return agnostic_nodes->find(node.op()) != agnostic_nodes->end();
}

bool IsAvgPoolGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "AvgPoolGrad";
}

bool IsBiasAddGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "BiasAddGrad";
}

bool IsConv2DBackpropFilter(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv2DBackpropFilter";
}

bool IsConv2DBackpropInput(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv2DBackpropInput";
}

bool IsDepthwiseConv2dNativeBackpropFilter(const tensorflow::NodeDef& node)
{
    return node.op() == "DepthwiseConv2dNativeBackpropFilter";
}

bool IsDepthwiseConv2dNativeBackpropInput(const tensorflow::NodeDef& node)
{
    return node.op() == "DepthwiseConv2dNativeBackpropInput";
}

bool IsFusedBatchNormEx(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormEx";
}

bool IsFusedBatchNormGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "FusedBatchNormGrad" ||
           node.op() == "FusedBatchNormGradV2" ||
           node.op() == "FusedBatchNormGradV3";
}

bool IsMaxPoolV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolV2";
}

bool IsMaxPoolGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGrad";
}

bool IsMaxPoolGradV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradV2";
}

bool IsMaxPoolGradGradV1(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradGrad";
}

bool IsMaxPoolGradGradV2(const tensorflow::NodeDef& node)
{
    return node.op() == "MaxPoolGradGradV2";
}

bool IsConv2D(const tensorflow::NodeDef& node) { return node.op() == "Conv2D"; }

bool IsAddN(const tensorflow::NodeDef& node) { return node.op() == "AddN"; }

bool IsAdd(const tensorflow::NodeDef& node)
{
    if (node.op() == "AddV2")
    {
        return true;
    }
    if (node.op() == "Add")
    {
        tensorflow::DataType type = node.attr().at("T").type();
        return type != tensorflow::DT_STRING;
    }
    return false;
}

bool IsApproximateEqual(const tensorflow::NodeDef& node)
{
    return node.op() == "ApproximateEqual";
}

bool IsEqual(const tensorflow::NodeDef& node) { return node.op() == "Equal"; }

bool IsGreater(const tensorflow::NodeDef& node)
{
    return node.op() == "Greater";
}

bool IsGreaterEqual(const tensorflow::NodeDef& node)
{
    return node.op() == "GreaterEqual";
}

bool IsLess(const tensorflow::NodeDef& node) { return node.op() == "Less"; }

bool IsLessEqual(const tensorflow::NodeDef& node)
{
    return node.op() == "LessEqual";
}

bool IsNotEqual(const tensorflow::NodeDef& node)
{
    return node.op() == "NotEqual";
}

bool IsComplex(const tensorflow::NodeDef& node)
{
    return node.op() == "Complex";
}

bool IsDiv(const tensorflow::NodeDef& node) { return node.op() == "Div"; }

bool IsFloorDiv(const tensorflow::NodeDef& node)
{
    return node.op() == "FloorDiv";
}

bool IsIgamma(const tensorflow::NodeDef& node) { return node.op() == "Igamma"; }

bool IsIgammac(const tensorflow::NodeDef& node)
{
    return node.op() == "Igammac";
}

bool IsLogicalAnd(const tensorflow::NodeDef& node)
{
    return node.op() == "LogicalAnd";
}

bool IsLogicalOr(const tensorflow::NodeDef& node)
{
    return node.op() == "LogicalOr";
}

bool IsMaximum(const tensorflow::NodeDef& node)
{
    return node.op() == "Maximum";
}

bool IsMinimum(const tensorflow::NodeDef& node)
{
    return node.op() == "Minimum";
}

bool IsMod(const tensorflow::NodeDef& node) { return node.op() == "Mod"; }

bool IsMul(const tensorflow::NodeDef& node) { return node.op() == "Mul"; }

bool IsPolygamma(const tensorflow::NodeDef& node)
{
    return node.op() == "Polygamma";
}

bool IsPow(const tensorflow::NodeDef& node) { return node.op() == "Pow"; }

bool IsRealDiv(const tensorflow::NodeDef& node)
{
    return node.op() == "RealDiv";
}

bool IsSquaredDifference(const tensorflow::NodeDef& node)
{
    return node.op() == "SquaredDifference";
}

bool IsSub(const tensorflow::NodeDef& node) { return node.op() == "Sub"; }

bool IsTruncateDiv(const tensorflow::NodeDef& node)
{
    return node.op() == "TruncateDiv";
}

bool IsTruncateMod(const tensorflow::NodeDef& node)
{
    return node.op() == "TruncateMod";
}

bool IsZeta(const tensorflow::NodeDef& node) { return node.op() == "Zeta"; }

bool IsIdentityN(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "IdentityN";
}
bool IsMerge(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Merge" || op == "RefMerge" || op == "_XlaMerge";
}
bool IsMirrorPad(const tensorflow::NodeDef& node)
{
    return node.op() == "MirrorPad";
}
bool IsMirrorPadGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "MirrorPadGrad";
}
bool IsPad(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "Pad" || op == "PadV2";
}
bool IsSelect(const tensorflow::NodeDef& node)
{
    return node.op() == "Select" || node.op() == "SelectV2";
}
bool IsSwitch(const tensorflow::NodeDef& node)
{
    const auto& op = node.op();
    return op == "_SwitchN" || op == "Switch" || op == "RefSwitch";
}
bool IsBetainc(const tensorflow::NodeDef& node)
{
    return node.op() == "Betainc";
}
bool IsEluGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "EluGrad";
}
bool IsInvGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "InvGrad";
}
bool IsLeakyReluGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "LeakyReluGrad";
}
bool IsReciprocalGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "ReciprocalGrad";
}
bool IsRelu6Grad(const tensorflow::NodeDef& node)
{
    return node.op() == "Relu6Grad";
}
bool IsReluGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "ReluGrad";
}
bool IsRsqrtGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "RsqrtGrad";
}
bool IsSeluGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "SeluGrad";
}
bool IsSigmoidGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "SigmoidGrad";
}
bool IsSoftplusGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "SoftplusGrad";
}
bool IsSoftsignGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "SoftsignGrad";
}
bool IsSqrtGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "SqrtGrad";
}
bool IsTanhGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "TanhGrad";
}
bool IsConcat(const tensorflow::NodeDef& node)
{
    return node.op() == "Concat" || node.op() == "ConcatV2";
}
bool IsReverseV2(const tensorflow::NodeDef& node)
{
    return node.op() == "ReverseV2";
}
bool IsTile(const tensorflow::NodeDef& node) { return node.op() == "Tile"; }
bool IsShape(const tensorflow::NodeDef& node) { return node.op() == "Shape"; }
bool IsShapeN(const tensorflow::NodeDef& node) { return node.op() == "ShapeN"; }
bool IsFill(const tensorflow::NodeDef& node) { return node.op() == "Fill"; }
bool IsSlice(const tensorflow::NodeDef& node) { return node.op() == "Slice"; }
bool IsSplit(const tensorflow::NodeDef& node) { return node.op() == "Split"; }
bool IsSqueeze(const tensorflow::NodeDef& node)
{
    return node.op() == "Squeeze";
}
bool IsSplitV(const tensorflow::NodeDef& node) { return node.op() == "SplitV"; }
bool IsStridedSlice(const tensorflow::NodeDef& node)
{
    return node.op() == "StridedSlice";
}

bool IsSum(const tensorflow::NodeDef& node) { return node.op() == "Sum"; }
bool IsMean(const tensorflow::NodeDef& node) { return node.op() == "Mean"; }
bool IsProd(const tensorflow::NodeDef& node) { return node.op() == "Prod"; }
bool IsMax(const tensorflow::NodeDef& node) { return node.op() == "Max"; }
bool IsMin(const tensorflow::NodeDef& node) { return node.op() == "Min"; }
bool IsAll(const tensorflow::NodeDef& node) { return node.op() == "All"; }
bool IsAny(const tensorflow::NodeDef& node) { return node.op() == "Any"; }

bool IsReduceOp(const tensorflow::NodeDef& node)
{
    return IsSum(node) || IsMean(node) || IsProd(node) || IsMax(node) ||
           IsMin(node) || IsAll(node) || IsAny(node);
}

bool IsFloatingDataType(tensorflow::DataType dtype)
{
    return dtype == tensorflow::DT_HALF || dtype == tensorflow::DT_FLOAT ||
           dtype == tensorflow::DT_DOUBLE || dtype == tensorflow::DT_BFLOAT16;
}

bool IsNonFloatingConv2D(const MutableNodeView& node)
{
    if (IsConv2D(*node.node()) || IsConv2DBackpropInput(*node.node()))
    {
        const auto* attr = node.GetAttr(kAttrT);
        if (attr != nullptr)
        {
            return !IsFloatingDataType(attr->type());
        }
    }
    return false;
}

bool IsAtan2(const tensorflow::NodeDef& node) { return node.op() == "Atan2"; }

bool IsComparisonOp(const tensorflow::NodeDef& node)
{
    bool is_compare = IsApproximateEqual(node) || IsEqual(node) ||
                      IsGreater(node) || IsGreaterEqual(node) || IsLess(node) ||
                      IsLessEqual(node) || IsNotEqual(node);
    return is_compare;
}

bool IsBinaryOp(const tensorflow::NodeDef& node)
{
    bool is_binary = IsAdd(node) || IsAtan2(node) || IsComparisonOp(node) ||
                     IsComplex(node) || IsDiv(node) || IsFloorDiv(node) ||
                     IsIgamma(node) || IsIgammac(node) || IsLogicalAnd(node) ||
                     IsLogicalOr(node) || IsMaximum(node) || IsMinimum(node) ||
                     IsMod(node) || IsMul(node) || IsPolygamma(node) ||
                     IsPow(node) || IsRealDiv(node) ||
                     IsSquaredDifference(node) || IsSub(node) ||
                     IsTruncateDiv(node) || IsTruncateMod(node) || IsZeta(node);
    return is_binary;
}

bool IsTernaryOp(const tensorflow::NodeDef& node) { return IsBetainc(node); }

bool IsUnaryGrad(const tensorflow::NodeDef& node)
{
    bool is_unary_grad =
        IsEluGrad(node) || IsInvGrad(node) || IsLeakyReluGrad(node) ||
        IsReciprocalGrad(node) || IsRelu6Grad(node) || IsReluGrad(node) ||
        IsRsqrtGrad(node) || IsSeluGrad(node) || IsSigmoidGrad(node) ||
        IsSoftplusGrad(node) || IsSoftsignGrad(node) || IsSqrtGrad(node) ||
        IsTanhGrad(node);
    return is_unary_grad;
}

bool IsLayoutSensitiveOp(const tensorflow::NodeDef& node)
{
    return IsDefaultLayoutSensitiveOp(node) || IsAvgPoolGrad(node) ||
           IsBiasAddGrad(node) || IsConv2DBackpropFilter(node) ||
           IsConv2DBackpropInput(node) ||
           IsDepthwiseConv2dNativeBackpropFilter(node) ||
           IsDepthwiseConv2dNativeBackpropInput(node) ||
           IsFusedBatchNormEx(node) || IsFusedBatchNormGrad(node) ||
           IsMaxPoolV2(node) || IsMaxPoolGrad(node) || IsMaxPoolGradV2(node) ||
           IsMaxPoolGradGradV1(node) || IsMaxPoolGradGradV2(node);
}

bool IsLayoutAgnosticOp(const tensorflow::NodeDef& node)
{
    return IsDefaultLayoutAgnosticOp(node) || IsAddN(node) ||
           IsBinaryOp(node) || IsIdentityN(node) || IsMerge(node) ||
           IsMirrorPad(node) || IsMirrorPadGrad(node) || IsPad(node) ||
           IsSelect(node) || IsSwitch(node) || IsTernaryOp(node) ||
           IsUnaryGrad(node) || IsConcat(node) || IsReverseV2(node) ||
           IsTile(node) || IsShape(node) || IsShapeN(node) || IsFill(node) ||
           IsSlice(node) || IsSplit(node) || IsSqueeze(node) ||
           IsSplitV(node) || IsStridedSlice(node) || IsReduceOp(node);
}

bool IsDataFormatOp(const MutableNodeView& node)
{
    const std::string& op = node.GetOp();
    return op == kOpDataFormatDimMap || op == kOpDataFormatVecPermute;
}

bool IsTranspose(const tensorflow::NodeDef& node)
{
    return node.op() == "Transpose";
}

bool IsArg(const tensorflow::NodeDef& node)
{
    return node.op() == "_Arg" || node.op() == "_DeviceArg";
}

bool IsConstant(const tensorflow::NodeDef& node)
{
    return node.op() == "Const";
}

bool IsStridedSliceGrad(const tensorflow::NodeDef& node)
{
    return node.op() == "StridedSliceGrad";
}

bool IsConv3DBackpropFilterV2(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv3DBackpropFilterV2";
}

bool IsConv3DBackpropInputV2(const tensorflow::NodeDef& node)
{
    return node.op() == "Conv3DBackpropInputV2";
}

bool IsConv3D(const tensorflow::NodeDef& node) { return node.op() == "Conv3D"; }

bool IsBiasAddV2(const tensorflow::NodeDef& node)
{
    return node.op() == "BiasAdd";
}

} // namespace tfdml