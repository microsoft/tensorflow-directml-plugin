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

#include "absl/cleanup/cleanup.h"
#include "tfdml/kernels/dml_strided_slice_helpers.h"
#include "tfdml/kernels/pch.h"
#include "tfdml/runtime_adapter/ops_util.h"
#include "tfdml/runtime_adapter/prefetch.h"
#include "tfdml/runtime_adapter/register_types_traits.h"
#include "tfdml/runtime_adapter/variable_lock.h"

namespace tfdml
{

template <typename Device>
struct MaybeWith32BitIndexingImpl
{
    template <typename Func, typename... Args>
    void operator()(Func func, Args&&... args) const
    {
        func(std::forward<Args>(args)...);
    }
};

template <typename Device, typename Func, typename... Args>
void MaybeWith32BitIndexing(Func func, Args&&... args)
{
    return MaybeWith32BitIndexingImpl<Device>()(
        func,
        std::forward<Args>(args)...);
}

template <typename Device, typename T, int NDIMS>
struct Slice
{
    void operator()(
        const Device& d,
        typename TTypes<T, NDIMS>::Tensor output,
        typename TTypes<T, NDIMS>::ConstTensor input,
        const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_indices,
        const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_sizes)
    {
        MaybeWith32BitIndexing<Device>(
            [&](auto output32,
                auto input32,
                auto slice_indices32,
                auto slice_sizes32) {
                output32.device(d) =
                    input32.slice(slice_indices32, slice_sizes32);
            },
            output,
            input,
            slice_indices,
            slice_sizes);
    }
};

template <typename T>
struct MemCpyFunctor
{
    // Returns true if the copy was made with memcpy, false otherwise.
    bool Copy(
        const Tensor& input,
        const absl::InlinedVector<int64_t, 4>& begin,
        const absl::InlinedVector<int64_t, 4>& end,
        Tensor* result)
    {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>()))
        {
            auto in = input.tensor<T, 2>();
            auto output = result->tensor<T, 2>();
            for (int row_in = begin[0], row_out = 0; row_in < end[0];
                 ++row_in, ++row_out)
            {
                if (row_in + 1 < end[0])
                {
                    port::prefetch<port::PREFETCH_HINT_T0>(
                        &output(row_in + 1, 0));
                    port::prefetch<port::PREFETCH_HINT_T0>(
                        &in(row_in + 1, begin[1]));
                }
                memcpy(
                    &output(row_out, 0),
                    &in(row_in, begin[1]),
                    (end[1] - begin[1]) * sizeof(T));
            }
            return true;
        }
        return false;
    }
};

template <typename Device, typename T, int NDIM>
static void HandleStridedSliceCase(
    OpKernelContext* context,
    const absl::Span<const int64_t>& begin,
    const absl::Span<const int64_t>& end,
    const absl::Span<const int64_t>& strides,
    const TensorShape& processing_shape,
    bool is_simple_slice,
    Tensor* result)
{
    typedef typename proxy_type<Device, T>::type Proxy;

    absl::InlinedVector<int64_t, 4> processing_dims =
        processing_shape.dim_sizes();
    if (is_simple_slice)
    {
        Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
        Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes_di;
        for (int i = 0; i < NDIM; ++i)
        {
            begin_di[i] = begin[i];
            sizes_di[i] = end[i] - begin[i];
        }

        const Tensor& input = context->input(0);

        Slice<Device, Proxy, NDIM>()(
            context->eigen_device<Device>(),
            result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
            input.bit_casted_tensor<Proxy, NDIM>(),
            begin_di,
            sizes_di);
    }
    else
    {
        Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
        Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
        Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
        for (int i = 0; i < NDIM; ++i)
        {
            begin_di[i] = begin[i];
            end_di[i] = end[i];
            strides_di[i] = strides[i];
        }

        const Tensor& input = context->input(0);

        functor::StridedSlice<Device, Proxy, NDIM>()(
            context->eigen_device<Device>(),
            result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
            input.bit_casted_tensor<Proxy, NDIM>(),
            begin_di,
            end_di,
            strides_di);
    }
}

template <typename Device, typename T>
class StridedSliceCpuOp : public OpKernel
{
  public:
    explicit StridedSliceCpuOp(
        OpKernelConstruction* context,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
        OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
        OP_REQUIRES_OK(
            context,
            context->GetAttr("ellipsis_mask", &ellipsis_mask));
        OP_REQUIRES_OK(
            context,
            context->GetAttr("new_axis_mask", &new_axis_mask));
        OP_REQUIRES_OK(
            context,
            context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
    }

    void Compute(OpKernelContext* context)
    {
        TensorShape processing_shape, final_shape;
        bool is_identity = true;
        bool slice_dim0 = true;
        bool is_simple_slice = true;
        absl::InlinedVector<int64_t, 4> begin;
        absl::InlinedVector<int64_t, 4> end;
        absl::InlinedVector<int64_t, 4> strides;

        const Tensor& input = context->input(0);
        const Tensor& begin_tensor = context->input(1);
        const Tensor& end_tensor = context->input(2);
        const Tensor& strides_tensor = context->input(3);

        OP_REQUIRES_OK(
            context,
            ValidateStridedSliceOp(
                &begin_tensor,
                &end_tensor,
                strides_tensor,
                input.shape(),
                begin_mask,
                end_mask,
                ellipsis_mask,
                new_axis_mask,
                shrink_axis_mask,
                &processing_shape,
                &final_shape,
                &is_identity,
                &is_simple_slice,
                &slice_dim0,
                &begin,
                &end,
                &strides));

        // Optimization #1, slice is a no-op plus reshape
        if (is_identity)
        {
            VLOG(1) << "Strided slice identity ";
            Tensor tmp;
            OP_REQUIRES(
                context,
                tmp.CopyFrom(input, final_shape),
                errors::Internal("Copy failed"));
            context->set_output(0, tmp);
            return;
        }

        StatusOr<Tensor> status_or_output =
            context->allocate_output(0, final_shape);
        OP_REQUIRES_OK(context, status_or_output.status());
        Tensor& result = status_or_output.ValueOrDie();

        // Optimization #2, slice is memory contiguous (only occurs in dim
        // 0)
        if (slice_dim0 &&
            IsDim0SliceAligned<T>(input.shape(), begin[0], end[0]))
        {
            OP_REQUIRES(
                context,
                input.dims() >= 1,
                errors::InvalidArgument(
                    "Input must have rank at least 1, got: ",
                    input.dims()));
            // Otherwise, is_identity should be true.
            VLOG(1) << "Strided slice dim 0: " << input.shape().DebugString();
            // To tolerate begin[0] > end[0] (a 0-output slice), we
            // min(begin, end).

            int64_t dim0_size = input.dim_size(0);
            int64_t start = std::min(begin[0], end[0]);
            int64_t limit = end[0];
            const int64_t elems_per_dim0 = input.NumElements() / dim0_size;
            const int64_t delta = start * elems_per_dim0;
            dim0_size = limit - start;
            const int64_t num_elems = dim0_size * elems_per_dim0;
            memcpy(result.base<T>(), input.base<T>() + delta, num_elems);
            return;
        }

        const int input_dims = input.dims();
        const int processing_dims = processing_shape.dims();

        if (processing_shape.num_elements() > 0)
        {
            // Optimization #3, slice has stride 1 in all dimensions
            // Optimization #3A, slice has only two dimensions
            // TODO(aselle): Here we are restricting to processing_shape and
            // final_shape being 2D. This isn't strictly necessary, but I
            // don't want to blow up code gen size, because to shape<> you
            // need static NDIM and T
            if (is_simple_slice &&
                std::is_same<Device, Eigen::ThreadPoolDevice>::value &&
                input_dims == 2 && processing_shape.dims() == 2 &&
                final_shape.dims() == 2 && new_axis_mask == 0)
            {
                MemCpyFunctor<T> functor;
                if (functor.Copy(input, begin, end, &result))
                {
                    return;
                }
            }

#define HANDLE_DIM(NDIM)                                                       \
    if (processing_dims == NDIM)                                               \
    {                                                                          \
        HandleStridedSliceCase<Device, T, NDIM>(                               \
            context,                                                           \
            begin,                                                             \
            end,                                                               \
            strides,                                                           \
            processing_shape,                                                  \
            is_simple_slice,                                                   \
            &result);                                                          \
        return;                                                                \
    }

            HANDLE_DIM(1);
            HANDLE_DIM(2);
            HANDLE_DIM(3);
            HANDLE_DIM(4);
            HANDLE_DIM(5);
            HANDLE_DIM(6);
            HANDLE_DIM(7);
            HANDLE_DIM(8);

#undef HANDLE_DIM

            OP_REQUIRES(
                context,
                false,
                errors::Unimplemented(
                    "Unhandled input dimensions ",
                    input_dims));
        }
    }

  private:
    int32_t begin_mask, end_mask;
    int32_t ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

void RegisterStridedSliceCpu()
{
    KernelDefinition<
        ops::StridedSlice,
        StridedSliceCpuOp<Eigen::ThreadPoolDevice, int32_t>>::
        WithHostMemoryArguments<
            ops::StridedSlice::Argument::input,
            ops::StridedSlice::Argument::begin,
            ops::StridedSlice::Argument::end,
            ops::StridedSlice::Argument::strides,
            ops::StridedSlice::Argument::output>::
            WithTypeConstraint<ops::StridedSlice::Attribute::T, TF_INT32>::
                Register();
}

void RegisterKernels_StridedSliceCpu() { RegisterStridedSliceCpu(); }
} // namespace tfdml