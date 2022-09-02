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

#include "tfdml/kernels/pch.h"

namespace tfdml
{

template <typename T>
class RangeInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    RangeInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        const Tensor& start_in = ctx->input(0);
        const Tensor& limit_in = ctx->input(1);
        const Tensor& delta_in = ctx->input(2);
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(start_in.shape()) ||
                (TensorShapeUtils::IsVector(start_in.shape()) &&
                 start_in.shape().dim_size(0) == 1),
            errors::InvalidArgument(
                "start must be a scalar, not shape ",
                start_in.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(limit_in.shape()) ||
                (TensorShapeUtils::IsVector(limit_in.shape()) &&
                 limit_in.shape().dim_size(0) == 1),
            errors::InvalidArgument(
                "limit must be a scalar, not shape ",
                limit_in.shape().DebugString()));
        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(delta_in.shape()) ||
                (TensorShapeUtils::IsVector(delta_in.shape()) &&
                 delta_in.shape().dim_size(0) == 1),
            errors::InvalidArgument(
                "delta must be a scalar, not shape ",
                delta_in.shape().DebugString()));
        const T start = start_in.base<T>()[0];
        const T limit = limit_in.base<T>()[0];
        const T delta = delta_in.base<T>()[0];
        OP_REQUIRES(
            ctx,
            delta != 0,
            errors::InvalidArgument("Requires delta != 0: ", delta));
        if (delta > 0)
        {
            OP_REQUIRES(
                ctx,
                start <= limit,
                errors::InvalidArgument(
                    "Requires start <= limit when delta > 0: ",
                    start,
                    "/",
                    limit));
        }
        else
        {
            OP_REQUIRES(
                ctx,
                start >= limit,
                errors::InvalidArgument(
                    "Requires start >= limit when delta < 0: ",
                    start,
                    "/",
                    limit));
        }
        int64_t size;
        if (std::is_integral<T>::value)
        {
            size = Eigen::divup(
                Eigen::numext::abs(limit - start),
                Eigen::numext::abs(delta));
        }
        else
        {
            auto size_auto = Eigen::numext::ceil(
                Eigen::numext::abs((limit - start) / delta));
            OP_REQUIRES(
                ctx,
                size_auto <= std::numeric_limits<int64_t>::max(),
                errors::InvalidArgument(
                    "Requires ((limit - start) / delta) <= ",
                    std::numeric_limits<int64_t>::max()));
            size = static_cast<int64_t>(size_auto);
        }

        OP_REQUIRES_OK(ctx, output_shape_.AddDimWithStatus(size));

        start_ = start;
        delta_ = delta;
    }

    const TensorShape& GetOutputShape() const { return output_shape_; }
    T GetStart() const { return start_; }
    T GetDelta() const { return delta_; }

  private:
    TensorShape output_shape_;
    T start_;
    T delta_;
};

template <typename T>
class RangeShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto* init_helper =
            static_cast<const RangeInitHelper<T>*>(initialization_helper);

        return {init_helper->GetOutputShape()};
    }
};

template <typename T>
class DmlRangeKernel : public DmlKernel
{
  public:
    using InitHelper = RangeInitHelper<T>;

    explicit DmlRangeKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        DmlKernelParams params;
        params.kernel_input_indices = {
            absl::nullopt,
            absl::nullopt,
            absl::nullopt,
        };

        DmlTensorInfo output_tensor = {};
        output_tensor.desc =
            CreateTensorDescFromOutput(ctx, 0, ctx->GetOutputTensorShape(0));
        output_tensor.kernel_index = 0;

        DmlKernelTensors tensors = {};
        tensors.outputs = {output_tensor};

        auto scope = dml::Graph(ctx->GetDmlDevice());

        auto start = dml::ScalarUnion(
            init_helper->GetStart(),
            output_tensor.desc.GetDmlDataType());

        auto delta = dml::ScalarUnion(
            init_helper->GetDelta(),
            output_tensor.desc.GetDmlDataType());

        auto result = dml::FillValueSequence(
            scope,
            NarrowTensorShape(ctx->GetOutputTensorShape(0)),
            output_tensor.desc.GetDmlDataType(),
            start,
            delta);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename T>
using K = typename KernelDefinition<
    ops::Range,
    DmlKernelWrapper<DmlRangeKernel<T>, RangeShapeHelper<T>>>::
    template WithHostMemoryArguments<
        ops::Range::Argument::start,
        ops::Range::Argument::limit,
        ops::Range::Argument::delta>::
        template WithTypeConstraint<
            ops::Range::Attribute::Tidx,
            DataTypeToEnum<T>()>;

void RegisterKernels_Range()
{
    K<float>::Register();
    K<int64_t>::Register();
}

} // namespace tfdml