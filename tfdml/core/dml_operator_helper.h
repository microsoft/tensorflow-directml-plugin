/* Copyright (c) Microsoft Corporation.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "tfdml/core/dml_common.h"
#include "tfdml/runtime_adapter/attribute.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "tfdml/runtime_adapter/tensor.h"
#include "tfdml/runtime_adapter/tensor_shape_utils.h"

namespace tfdml
{

class OpKernelConstruction;

class InitializationHelper;

class ShapeHelper
{
  public:
    virtual std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const = 0;
};

class InitializationHelper
{
  public:
    struct EmptyAttributes
    {
        explicit EmptyAttributes(OpKernelConstruction* ctx) {}
    };

    // By default, a kernel is considered a no-op if any of its input or output
    // tensors are empty.
    virtual bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const
    {
        for (int i = 0; i < ctx->num_inputs(); ++i)
        {
            if (ctx->input(i).NumElements() == 0)
            {
                return true;
            }
        }

        for (const auto& output_shape : output_shapes)
        {
            if (output_shape.num_elements() == 0)
            {
                return true;
            }
        }

        return false;
    }

    virtual absl::optional<int> GetForwardableInputIndex(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes,
        int outputIndex) const
    {
        return {};
    }

    virtual ~InitializationHelper() = default;
};

class NoOpInitializationHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;
    NoOpInitializationHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
    }
};

template <typename TIndex, int dims_tensor_index>
class GetOutputShapeFromDimsTensorHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const
    {
        TensorShape shape =
            TensorShapeUtils::MakeShape(ctx->input(dims_tensor_index));
        return {std::move(shape)};
    }
};

class NoOutputShapeHelper : public ShapeHelper
{
  public:
    inline std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        return {};
    };
};

class ScalarOutputShapeHelper : public ShapeHelper
{
  public:
    inline std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        return {{}};
    };
};

template <int input_tensor_index>
class GetOutputShapeFromInputShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        return {ctx->input(input_tensor_index).shape()};
    }
};

using GetOutputShapeAsInputShapeHelper = GetOutputShapeFromInputShapeHelper<0>;

template <int input_tensor_index>
class GetOutputShapeFromRefInputShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        constexpr bool lock_held = false;
        constexpr bool is_variant = false;
        Tensor input_tensor;

        Status status = ctx->GetInputTensorFromVariable(
            input_tensor_index,
            lock_held,
            is_variant,
            &input_tensor);
        CHECK(status.ok());

        return {input_tensor.shape()};
    }
};

using GetOutputShapeAsRefInputShapeHelper =
    GetOutputShapeFromRefInputShapeHelper<0>;

class BroadcastedOutputShapeInitHelper : public InitializationHelper
{
  public:
    using Attributes = EmptyAttributes;

    BroadcastedOutputShapeInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr);
    const TensorShape& GetBroadcastedShape() const
    {
        return broadcasted_shape_;
    }

  private:
    TensorShape broadcasted_shape_;
};

class GetBroadcastedOutputShapeHelper : public ShapeHelper
{
  public:
    using InitHelper = BroadcastedOutputShapeInitHelper;

    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override;
};

class BatchNormShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override;
};

class BatchNormGradShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override;
};

class SparseXentShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        const Tensor logits = ctx->input(0);
        const Tensor labels = ctx->input(1);
        // logits must be 2-D
        CHECK(TensorShapeUtils::IsMatrix(logits.shape()));
        // labels must be 1-D
        CHECK(TensorShapeUtils::IsVector(labels.shape()));
        // logits and labels must have the same first dimension
        CHECK(logits.dim_size(0) == labels.dim_size(0));
        // Must have at least one class
        CHECK(logits.dim_size(1) > 0);

        return {labels.shape(), logits.shape()};
    }
};

template <typename T>
absl::InlinedVector<T, 5> IntTensorToVec(const Tensor& tensor)
{
    absl::InlinedVector<T, 5> out;

    if (tensor.dtype() == TF_INT32)
    {
        auto int32_values = tensor.base<int32_t>();

        for (int64_t i = 0; i < tensor.NumElements(); ++i)
        {
            out.push_back(static_cast<T>(int32_values[i]));
        }
    }
    else
    {
        assert(tensor.dtype() == TF_INT64);
        auto int64_values = tensor.base<int64_t>();

        for (int64_t i = 0; i < tensor.NumElements(); ++i)
        {
            out.push_back(static_cast<T>(int64_values[i]));
        }
    }

    return out;
}

TensorShape BroadcastTensorShapes(absl::Span<const TensorShape> shapes);
TensorShape ComputeFlatOuterDims(const TensorShape& orig, int64_t num_out_dims);

} // namespace tfdml