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
#include "tfdml/kernels/pch.h"

namespace tfdml
{
namespace
{
/// Constants
constexpr int32_t kShrinkAxis = -1, kNewAxis = -2;

// Sparse slicing specification
// if one does foo[3:5, ..., -3], this will have 3 length tensors
struct StridedSliceSparseSpec
{
    int64_t dims;
    int32_t num_add_axis_after_ellipsis;
    const Tensor* begin_tensor;
    const Tensor* end_tensor;
    const Tensor& strides_tensor;
    const int32_t begin_mask, end_mask;
    int32_t ellipsis_mask;
    const int32_t new_axis_mask, shrink_axis_mask;
};

// Dense slicing specification
// all ellipses and newaxis' are expanded out. So if
// foo[3:5, ..., -3] where foo is 10 dimensional,
// each inlinedVector will have 10 entries whereas the
// sparse had 3 length tensors.
struct StridedSliceDenseSpec
{
    const int64_t dims;
    int32_t begin_mask;
    int32_t end_mask;
    bool begin_valid;
    bool end_valid;
    absl::InlinedVector<int64_t, 4>& begin;
    absl::InlinedVector<int64_t, 4>& end;
    absl::InlinedVector<int64_t, 4>& strides;
    // This vector helps construct the final shape of the slice.
    // The final tensor is reduced in rank whenever a single index e.g. foo[3]
    // is called for. The final tensor increases in rank with tf.newaxis
    // entries. If an index in this array is positive, the size of the dimension
    // is obtained from canonical end-begin. Otherwise, if it is a kNewAxis,
    // it will be 1. A shrunk dimension is skipped.
    absl::InlinedVector<int32_t, 4> final_shape_gather_indices;
    // The dense indexed shrink mask is which processing dimensions
    // should be shrunk. For example, if foo.shape = (10,10,10,10)
    // foo[3, ..., 5] has sparse_shrink_axis_mask of 0x5 and
    // dense_shrink_axis_mask of 0x9, yielding a final shape (10,10).
    int32_t shrink_axis_mask;
};

} // namespace

template <class T>
static Status BuildDenseSpec(
    const StridedSliceSparseSpec& sparse,
    StridedSliceDenseSpec* dense)
{
    // Build expanded begin, end, strides, begin_mask, end_mask
    // to remove any ellipsis
    dense->begin.resize(dense->dims);
    dense->end.resize(dense->dims);
    dense->strides.resize(dense->dims);
    // What indices to get the final shape from.
    dense->begin_mask = 0;
    dense->end_mask = 0;
    dense->shrink_axis_mask = 0;
    {
        int full_index = 0;

        auto strides_flat =
            static_cast<const T* const>(sparse.strides_tensor.raw_data());
        dense->begin_valid = sparse.begin_tensor != nullptr;
        dense->end_valid = sparse.end_tensor != nullptr;

        auto begin_flat =
            sparse.begin_tensor != nullptr
                ? static_cast<const T* const>(sparse.begin_tensor->raw_data())
                : nullptr;
        auto end_flat =
            sparse.end_tensor != nullptr
                ? static_cast<const T* const>(sparse.end_tensor->raw_data())
                : nullptr;

        for (int i = 0; i < sparse.dims; i++)
        {
            if ((1 << i) & sparse.ellipsis_mask)
            {
                // Expand the ellipsis into the appropriate indices
                // NOTE: this only works because we guaranteed one ellipsis
                int32_t next_index = std::min(
                    dense->dims - (sparse.dims - i) + 1 +
                        sparse.num_add_axis_after_ellipsis,
                    dense->dims);
                for (; full_index < next_index; full_index++)
                {
                    // new_axis' aren't real axis so you have to skip
                    dense->begin[full_index] = dense->end[full_index] = 0;
                    dense->strides[full_index] = 1;
                    dense->begin_mask |= (1 << full_index);
                    dense->end_mask |= (1 << full_index);
                    dense->final_shape_gather_indices.push_back(full_index);
                }
            }
            else if ((1 << i) & sparse.new_axis_mask)
            {
                dense->final_shape_gather_indices.push_back(kNewAxis);
            }
            else
            {
                if (full_index == dense->begin.size())
                {
                    return errors::InvalidArgument(
                        "Index out of range using input dim ",
                        full_index,
                        "; input has only ",
                        dense->dims,
                        " dims");
                }

                // Gather slicing spec into appropriate index
                if (begin_flat != nullptr)
                {
                    dense->begin[full_index] = begin_flat[i];
                }
                if (end_flat != nullptr)
                {
                    dense->end[full_index] = end_flat[i];
                }
                dense->strides[full_index] = strides_flat[i];
                if (sparse.begin_mask & (1 << i))
                {
                    dense->begin_mask |= (1 << full_index);
                }
                if (sparse.end_mask & (1 << i))
                {
                    dense->end_mask |= (1 << full_index);
                }
                // If shrink, record where to get the dimensionality from (i.e.
                // new_axis creates a fake 1 size dimension. Also remember
                // shrink axis (now in dense form) so we can ignore dense->end
                // below.
                if (sparse.shrink_axis_mask & (1 << i))
                {
                    dense->final_shape_gather_indices.push_back(kShrinkAxis);
                    dense->shrink_axis_mask |= (1 << full_index);
                }
                else
                {
                    dense->final_shape_gather_indices.push_back(full_index);
                }
                full_index++;
            }
        }
    }
    return Status::OK();
}

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
    absl::InlinedVector<int64_t, 4>* strides)
{
    const bool begin_is_wrong =
        begin_tensor != nullptr &&
        !(TensorShapeUtils::IsVector(begin_tensor->shape()) &&
          begin_tensor->NumElements() == strides_tensor.NumElements() &&
          begin_tensor->NumElements() < 32 /* using 32 bit masks */);
    const bool end_is_wrong =
        end_tensor != nullptr &&
        !(TensorShapeUtils::IsVector(end_tensor->shape()) &&
          end_tensor->NumElements() == strides_tensor.NumElements());
    if (begin_is_wrong || end_is_wrong ||
        !TensorShapeUtils::IsVector(strides_tensor.shape()))
    {
        if (begin_tensor != nullptr && end_tensor != nullptr)
        {
            return errors::InvalidArgument(
                "Expected begin, end, and strides to be 1D equal size "
                "tensors, ",
                "but got shapes ",
                begin_tensor->shape().DebugString(),
                ", ",
                end_tensor->shape().DebugString(),
                ", and ",
                strides_tensor.shape().DebugString(),
                " instead.");
        }
        else
        {
            return errors::InvalidArgument(
                "Expected begin, end, and strides to be 1D equal size "
                "tensors, ",
                "but got shape ",
                strides_tensor.shape().DebugString(),
                " for strides.");
        }
    }
    // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
    // i.e. there exists only no more than one ellipsis
    if (ellipsis_mask && ((ellipsis_mask & (ellipsis_mask - 1)) != 0))
    {
        return errors::InvalidArgument(
            "Multiple ellipses in slice spec not allowed");
    }

    // Step 1: Account for ellipsis and new axis
    //
    // Check for ellipses and count how many non-newaxis' there are after
    // TODO(aselle): Convert this to do a fast log2 followed by iteration
    //               counting ones in next guys
    bool ellipsis_seen = false;

    StridedSliceSparseSpec sparse_spec = {
        strides_tensor.NumElements(),
        0,
        begin_tensor,
        end_tensor,
        strides_tensor,
        begin_mask_spec,
        end_mask_spec,
        ellipsis_mask,
        new_axis_mask,
        shrink_axis_mask};

    for (int32_t i = 0; i < sparse_spec.dims; i++)
    {
        if (ellipsis_seen && ((1 << i) & new_axis_mask) != 0)
        {
            sparse_spec.num_add_axis_after_ellipsis++;
        }
        if ((1 << i) & ellipsis_mask)
        {
            ellipsis_seen = true;
        }
    }
    // If no ellipsis insert one at the end
    if (!ellipsis_seen)
    {
        sparse_spec.ellipsis_mask |= (1 << sparse_spec.dims);
        sparse_spec.dims++; // this effects loop iteration below
    }

    // Step 2: Make a sparse spec into a full index spec
    //
    // The sparse spec does not correspond to the number of dimensions
    // Make a dense spec that corresponds to the number of dimensions
    //
    // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
    // we need to produce the missing begin_mask for the first two
    // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
    // we achieve begin_mask=6, end_mask=7
    StridedSliceDenseSpec dense_spec = {
        input_shape.dims(),
        0 /* begin_mask */,
        0 /* end_mask */,
        false /* begin_valid */,
        false /* end_valid */,
        *begin,
        *end,
        *strides};

    if (strides_tensor.dtype() == TF_INT32)
    {
        TF_RETURN_IF_ERROR(BuildDenseSpec<int32_t>(sparse_spec, &dense_spec));
    }
    else if (strides_tensor.dtype() == TF_INT64)
    {
        TF_RETURN_IF_ERROR(BuildDenseSpec<int64_t>(sparse_spec, &dense_spec));
    }
    else
    {
        LogFatal("begin must be either int32_t or int64_t");
    }

    // Step 3: Make implicit ranges (non-zero begin_masks and end_masks)
    // explicit
    //         and bounds check!
    *is_identity = true;
    *slice_dim0 = true;
    *is_simple_slice = true;
    processing_shape->Clear();
    for (int i = 0; i < input_shape.dims(); ++i)
    {
        int64_t& begin_i = (*begin)[i];
        int64_t& end_i = (*end)[i];
        int64_t& stride_i = (*strides)[i];
        int64_t dim_i = input_shape.dim_size(i);
        if (stride_i == 0)
        {
            return errors::InvalidArgument("strides[", i, "] must be non-zero");
        }
        bool shrink_i = (dense_spec.shrink_axis_mask & (1 << i));
        if (dim_i == -1)
        {
            processing_shape->AddDim(shrink_i ? 1 : -1);
            continue;
        }

        const std::array<int64_t, 2> masks = {
            {dense_spec.begin_mask & (1 << i), dense_spec.end_mask & (1 << i)}};
        const std::array<int64_t, 2> valid_range = {
            {stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

        auto canonical = [stride_i, dim_i, masks, valid_range](int64_t x, int c)
        {
            if (masks[c])
            {
                return stride_i > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
            }
            else
            {
                int64_t x_fwd =
                    x < 0 ? dim_i + x : x; // make negative indices positive
                return x_fwd < valid_range[0]   ? valid_range[0]
                       : x_fwd > valid_range[1] ? valid_range[1]
                                                : x_fwd;
            }
        };
        if (shrink_i && stride_i <= 0)
        {
            return errors::InvalidArgument(
                "only stride 1 allowed on non-range indexing.");
        }
        (*is_simple_slice) &= stride_i == 1;

        const bool begin_and_end_masked = (dense_spec.begin_mask & (1 << i)) &&
                                          (dense_spec.end_mask & (1 << i));
        if (dense_spec.begin_valid && dense_spec.end_valid)
        {
            if (shrink_i)
            {
                // If we are shrinking, the end index is now possibly incorrect.
                // In particular foo[-1] produces sparse_begin = -1, sparse_end
                // = 0. and canonical puts these to n-1 and 0, which implies a
                // degenerate interval. Fortunately, it is now safe to re-create
                // end as begin+1.
                int64_t x_fwd = begin_i < 0 ? dim_i + begin_i : begin_i;
                begin_i = x_fwd;
                end_i = begin_i + 1;
                if (x_fwd < 0 || x_fwd >= dim_i)
                {
                    return errors::InvalidArgument(
                        "slice index ",
                        begin_i,
                        " of dimension ",
                        i,
                        " out of bounds.");
                }
            }
            else
            {
                begin_i = canonical(begin_i, 0);
                end_i = canonical(end_i, 1);
            }
            // Update optimization values
            bool take_all_in_dimension =
                stride_i == 1 && begin_i == 0 && end_i == dim_i;
            (*is_identity) &= take_all_in_dimension;
            (*slice_dim0) &= (i == 0 && stride_i == 1) || take_all_in_dimension;
        }
        else
        {
            (*is_identity) &= stride_i == 1 && begin_and_end_masked;
            (*slice_dim0) &= (i == 0 && stride_i == 1) || begin_and_end_masked;
        }
        // Compute the processing shape (the intermediate Eigen will produce)
        int64_t interval_length;
        bool known_interval = false;
        if (dense_spec.begin_valid && dense_spec.end_valid)
        {
            interval_length = end_i - begin_i;
            known_interval = true;
        }
        else if (shrink_i)
        {
            // The dimension is still known as 1 for the processing_shape, but
            // will be discarded for the final shape.
            interval_length = 1;
            known_interval = true;
        }
        else if (begin_and_end_masked)
        {
            // Even if we don't have values for begin or end, we do know that
            // this dimension covers the whole interval. If we have shape
            // information for this dimension, that tells us the interval
            // length.
            if (dim_i >= 0)
            {
                if (stride_i < 0)
                {
                    interval_length = -dim_i;
                }
                else
                {
                    interval_length = dim_i;
                }
                known_interval = true;
            }
        }
        if (known_interval)
        {
            int64_t size_i;
            // Hold zero if the interval is degenerate, otherwise account for
            // remainder
            if (interval_length == 0 ||
                ((interval_length < 0) != (stride_i < 0)))
            {
                size_i = 0;
            }
            else
            {
                size_i = interval_length / stride_i +
                         (interval_length % stride_i != 0 ? 1 : 0);
            }
            processing_shape->AddDim(size_i);
        }
        else
        {
            processing_shape->AddDim(-1);
        }
    }

    // Step 4: Compute the final shape
    //
    // new_axis will increase dimension by 1 (with a one-size dimension)
    // slices like foo[3,...] will reduce dimension by 1.
    // This cannot be done earlier, because it depends on Step 3.
    final_shape->Clear();
    for (auto gather_index : dense_spec.final_shape_gather_indices)
    {
        if (gather_index >= 0)
        {
            final_shape->AddDim(processing_shape->dim_size(gather_index));
        }
        else if (gather_index == kNewAxis)
        {
            final_shape->AddDim(1);
        }
    }
    return Status::OK();
}

struct SimplifiedSlice
{
    dml::TensorDesc::Dimensions input_sizes;
    dml::TensorDesc::Dimensions input_strides;
    dml::TensorDesc::Dimensions output_sizes;
    dml::SmallVector<uint32_t, 5> window_offset;
    dml::SmallVector<uint32_t, 5> window_sizes;
    dml::SmallVector<int32_t, 5> window_strides;
};

template <typename T>
void ShiftDim(T& vec, int shift_amount, uint32_t dim_count)
{
    std::rotate(vec.begin(), vec.begin() + shift_amount, vec.end());
    vec.resize(dim_count);
}

// This helper may simplify an N-dimensional slice to a lower rank slice by
// coalescing dimensions that meet the following criteria:
// - Dimensions with size 1 are always coalesced.
// - Adjacent dimensions that are fully included in the slice are always
// coalesced.
// - A higher-order dimension that is partially included in the slice, and has
// no offset/stride, will be
//   merged with lower-order dimensions that are fully included in the slice.
static absl::optional<SimplifiedSlice> SimplifySlice(
    const TensorShape& input_shape,
    const absl::InlinedVector<int64_t, 4>& canonical_begins,
    const absl::InlinedVector<int64_t, 4>& canonical_ends,
    const absl::InlinedVector<int64_t, 4>& strides,
    uint32_t min_output_size = 4,
    uint32_t max_output_size = 8)
{
    assert(input_shape.dims() == canonical_begins.size());
    assert(input_shape.dims() == canonical_ends.size());
    assert(input_shape.dims() == strides.size());
    assert(max_output_size > 0);

    SimplifiedSlice desc = {};
    desc.input_sizes.resize(max_output_size, 1);
    desc.input_strides.resize(max_output_size, 1);
    desc.output_sizes.resize(max_output_size, 1);
    desc.window_offset.resize(max_output_size, 0);
    desc.window_sizes.resize(max_output_size, 1);
    desc.window_strides.resize(max_output_size, 1);

    int current_dim = max_output_size - 1;

    // Insertion becomes a no-op if the shape cannot be simplified into the
    // requested max_output_size.
    auto InsertDim = [&](uint32_t input_size,
                         uint32_t input_stride,
                         uint32_t output_size,
                         uint32_t window_offset,
                         uint32_t window_size,
                         int32_t window_stride)
    {
        if (current_dim >= 0)
        {
            desc.input_sizes[current_dim] = input_size;
            desc.input_strides[current_dim] = input_stride;
            desc.output_sizes[current_dim] = output_size;
            desc.window_offset[current_dim] = window_offset;
            desc.window_sizes[current_dim] = window_size;
            desc.window_strides[current_dim] = window_stride;
        }
        current_dim--;
    };

    uint32_t coalesced = 1;
    uint32_t total_stride = 1;

    for (int i = input_shape.dims() - 1; i >= 0; i--)
    {
        const uint32_t input_size = input_shape.dim_size(i);
        const int32_t window_stride = static_cast<int32_t>(strides[i]);

        // Here, begin and end contain the canonical values. This means that
        // they cannot be negative when strides are positive. When strides are
        // negative, end can only be positive or -1. See the
        // ValidateStridedSliceOp function in strided_slice_op.cc for reference.
        const int64_t begin = canonical_begins[i];
        const int64_t end = canonical_ends[i];
        CHECK(end >= -1);

        uint32_t window_offset, window_size, output_size;
        if (window_stride > 0)
        {
            window_offset = begin;
            window_size = end - begin;
            output_size = 1 + (window_size - 1) / window_stride;
        }
        else
        {
            window_offset = end + 1; // +1 to convert exclusive to inclusive
            window_size = begin - end;
            output_size = 1 + (window_size - 1) / -window_stride;
        }

        if (input_size == output_size && window_stride > 0)
        {
            // The dimension can be collapsed, since all of its elements are
            // included in the slice. However, coalescing can only be performed
            // if the elements are read in order (i.e. stride is positive).
            coalesced *= input_size;
        }
        else
        {
            if (begin == 0 && window_stride == 1 && coalesced > 1)
            {
                // The current dim is merged with all previously collapsed
                // dims.This is only possible because slicing of the current dim
                // emits elements adjacent to the previously collapsed dims.
                // Some of the tail elements in the current dim won't be
                // included in the slice, but they can be skipped by padding the
                // input strides to account for the extra physical elements.
                InsertDim(
                    /*inputSize    */ coalesced * input_size,
                    /*inputStride  */ total_stride,
                    /*outputSize   */ coalesced * output_size,
                    /*windowOffset */ 0,
                    /*windowSize   */ coalesced * output_size,
                    /*windowStride */ 1);
                total_stride *= coalesced * input_size;
            }
            else
            {
                // The current dim cannot be merged at all, so (up to) two dims
                // are inserted: the previously collapsed dims, if any, and a
                // separate dim for the non-contiguous current dim.
                if (coalesced > 1)
                {
                    InsertDim(
                        /*inputSize    */ coalesced,
                        /*inputStride  */ total_stride,
                        /*outputSize   */ coalesced,
                        /*windowOffset */ 0,
                        /*windowSize   */ coalesced,
                        /*windowStride */ 1);
                    total_stride *= coalesced;
                }
                InsertDim(
                    /*inputSize    */ input_size,
                    /*inputStride  */ total_stride,
                    /*outputSize   */ output_size,
                    /*windowOffset */ window_offset,
                    /*windowSize   */ window_size,
                    /*windowStride */ window_stride);
                total_stride *= input_size;
            }
            coalesced = 1;
        }
    }

    if (coalesced > 1)
    {
        InsertDim(
            /*inputSize    */ coalesced,
            /*inputStride  */ total_stride,
            /*outputSize   */ coalesced,
            /*windowOffset */ 0,
            /*windowSize   */ coalesced,
            /*windowStride */ 1);
        total_stride *= coalesced;
    }

    // current_dim is the index of the next dim to write; if it's -1, then all
    // max_output_size dims have been filled (0 dims remain). Anything larger
    // than -1 indicates padding.
    int dims_remaining = current_dim + 1;
    if (dims_remaining < 0)
    {
        return absl::nullopt;
    }
    else
    {
        for (int i = current_dim; i >= 0; i--)
        {
            desc.input_strides[current_dim--] = total_stride;
        }

        // DML is (in general) faster with fewer dims, so shift values left if
        // there are leading padding dims. No need to use 5D shader if 4D is
        // possible.
        int max_shift = max_output_size - min_output_size;
        int shift_amount = std::min<int>(max_shift, dims_remaining);
        uint32_t dim_count = max_output_size - shift_amount;

        ShiftDim(desc.input_sizes, shift_amount, dim_count);
        ShiftDim(desc.input_strides, shift_amount, dim_count);
        ShiftDim(desc.output_sizes, shift_amount, dim_count);
        ShiftDim(desc.window_offset, shift_amount, dim_count);
        ShiftDim(desc.window_sizes, shift_amount, dim_count);
        ShiftDim(desc.window_strides, shift_amount, dim_count);
    }

    return desc;
}

class StridedSliceInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask));
        }

        int32_t begin_mask, end_mask;
        int32_t ellipsis_mask, new_axis_mask, shrink_axis_mask;
    };

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        const bool is_grad_op = ctx->num_inputs() == 5;

        if (is_grad_op)
        {
            // For StridedSliceGrad, the last input is the input gradient
            return ctx->input(4).NumElements() == 0 ||
                   output_shapes[0].num_elements() == 0;
        }

        // StridedSlice is only a no-op if the first input or its output is
        // empty
        return ctx->input(0).NumElements() == 0 ||
               output_shapes[0].num_elements() == 0;
    }

    StridedSliceInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        TensorShape processing_shape;
        bool slice_dim0 = true;
        bool is_simple_slice = true;
        absl::InlinedVector<int64_t, 4> begin;
        absl::InlinedVector<int64_t, 4> end;
        absl::InlinedVector<int64_t, 4> strides;

        // StridedSliceGrad has a 5th tensor for dy.
        bool is_grad_op = ctx->num_inputs() == 5;

        // StridedSliceGrad stores shape in a 1D host tensor.
        TensorShape input_shape;
        if (is_grad_op)
        {
            const Tensor& input_shape_tensor = ctx->input(0);
            OP_REQUIRES(
                ctx,
                input_shape_tensor.dims() == 1,
                errors::InvalidArgument(
                    "shape must be 1-D, got shape.shape = ",
                    input_shape_tensor.shape().DebugString()));
            if (input_shape_tensor.dtype() == TF_INT32)
            {
                OP_REQUIRES_OK(
                    ctx,
                    TensorShapeUtils::MakeShape(
                        input_shape_tensor,
                        &input_shape));
            }
            else if (input_shape_tensor.dtype() == TF_INT64)
            {
                OP_REQUIRES_OK(
                    ctx,
                    TensorShapeUtils::MakeShape(
                        input_shape_tensor,
                        &input_shape));
            }
            else
            {
                LogFatal("shape must have type int32_t or int64_t.");
            }
        }
        else
        {
            input_shape = ctx->input(0).shape();
        }

        OP_REQUIRES_OK(
            ctx,
            ValidateStridedSliceOp(
                &ctx->input(1),
                &ctx->input(2),
                ctx->input(3),
                input_shape,
                attr->begin_mask,
                attr->end_mask,
                attr->ellipsis_mask,
                attr->new_axis_mask,
                attr->shrink_axis_mask,
                &processing_shape,
                &output_shape_,
                &is_identity_,
                &is_simple_slice,
                &slice_dim0,
                &begin,
                &end,
                &strides));

        // Check to make sure dy is consistent with the original slice.
        if (is_grad_op)
        {
            TensorShape dy_shape = ctx->input(4).shape();
            OP_REQUIRES(
                ctx,
                output_shape_ == dy_shape,
                errors::InvalidArgument(
                    "shape of dy was ",
                    dy_shape.DebugString(),
                    " instead of ",
                    output_shape_.DebugString()));
            output_shape_ = input_shape;
        }

        // Attempt to simplify the slice into a lower-rank slice.
        simple_slice_ = SimplifySlice(input_shape, begin, end, strides);
        if (!simple_slice_)
        {
            OP_REQUIRES(
                ctx,
                simple_slice_,
                errors::InvalidArgument(
                    "DML only support slicing up to 8D inputs, "
                    "but received ",
                    input_shape.dims()));
        }
    }

    const TensorShape& GetOutputShape() const { return output_shape_; }
    const bool IsIdentity() const { return is_identity_; }

    const absl::optional<SimplifiedSlice>& GetSimplifiedSlice() const
    {
        return simple_slice_;
    }

  private:
    TensorShape output_shape_;
    absl::optional<SimplifiedSlice> simple_slice_;
    bool is_identity_;
};

using InitHelper = StridedSliceInitHelper;

class StridedSliceShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        return {init_helper->GetOutputShape()};
    }
};

class DmlStridedSliceKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper;

    explicit DmlStridedSliceKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 4);
        CHECK(ctx->GetOutputCount() == 1);

        auto simple_slice = init_helper->GetSimplifiedSlice();
        auto dtype_tf = ctx->GetInputDataType(0);
        const DML_TENSOR_DATA_TYPE dtype_dml =
            GetDmlDataTypeFromTfDataType(dtype_tf);

        // TODO #24881131: 64-bit data support should be revisited
        // TFDML #24881131
        uint64_t end_padding_in_bytes = 0;
        dml::TensorDesc::Dimensions output_strides(
            simple_slice->output_sizes.size());
        uint32_t stride = 1;
        for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--)
        {
            output_strides[i] = stride;
            stride *= simple_slice->output_sizes[i];
        }
        if (Is64BitIntegerType(dtype_tf))
        {
            for (auto& stride : simple_slice->input_strides)
            {
                stride *= 2;
            }
            for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--)
            {
                output_strides[i] *= 2;
            }
            end_padding_in_bytes = sizeof(uint32_t);
        }

        DmlTensorInfo input;
        input.kernel_index = 0;
        input.desc = DmlTensorDesc{
            dtype_dml,
            simple_slice->input_sizes,
            simple_slice->input_strides,
            0,
            end_padding_in_bytes};

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc{
            dtype_dml,
            simple_slice->output_sizes,
            output_strides,
            0,
            end_padding_in_bytes};

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto result = dml::InputTensor(scope, 0, inputs[0]);

        if (init_helper->IsIdentity())
        {
            result = dml::Identity(result);
        }
        else
        {
            result = dml::Slice(
                result,
                simple_slice->window_offset,
                simple_slice->window_sizes,
                simple_slice->window_strides);
        }

        // TFDML #24881131
        if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

// ----------------------------------------
// StridedSliceGrad
// ----------------------------------------

class DmlStridedSliceGradKernel : public DmlKernel
{
  public:
    using InitHelper = tfdml::InitHelper;

    explicit DmlStridedSliceGradKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        CHECK(ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 1);

        auto simple_slice = init_helper->GetSimplifiedSlice();
        auto dtype_tf = ctx->GetInputDataType(4);
        const DML_TENSOR_DATA_TYPE dtype_dml =
            GetDmlDataTypeFromTfDataType(dtype_tf);

        // TODO #24881131: 64-bit data support should be revisited
        // TFDML #24881131
        uint64_t end_padding_in_bytes = 0;
        dml::TensorDesc::Dimensions output_strides(
            simple_slice->output_sizes.size());
        uint32_t stride = 1;
        for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--)
        {
            output_strides[i] = stride;
            stride *= simple_slice->output_sizes[i];
        }
        if (Is64BitIntegerType(dtype_tf))
        {
            for (auto& stride : simple_slice->input_strides)
            {
                stride *= 2;
            }
            for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--)
            {
                output_strides[i] *= 2;
            }
            end_padding_in_bytes = sizeof(uint32_t);
        }

        DmlTensorInfo input;
        input.kernel_index = 4;
        input.desc = DmlTensorDesc{
            dtype_dml,
            simple_slice->output_sizes,
            output_strides,
            0,
            end_padding_in_bytes};

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc{
            dtype_dml,
            simple_slice->input_sizes,
            simple_slice->input_strides,
            0,
            end_padding_in_bytes};

        DmlKernelTensors tensors;
        tensors.inputs = {input};
        tensors.outputs = {output};

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto result = dml::InputTensor(scope, 0, inputs[0]);

        if (init_helper->IsIdentity())
        {
            result = dml::Identity(result);
        }
        else
        {
            result = dml::SliceGrad(
                result,
                simple_slice->input_sizes,
                simple_slice->window_offset,
                simple_slice->window_sizes,
                simple_slice->window_strides);
        }

        // TFDML #24881131
        if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0)))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class StridedSliceAssignInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask));
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask));
        }

        int32_t begin_mask, end_mask;
        int32_t ellipsis_mask, new_axis_mask, shrink_axis_mask;
    };

    bool IsNoOpKernel(
        OpKernelContext* ctx,
        absl::Span<const TensorShape> output_shapes) const override
    {
        if (!output_shapes.empty() && output_shapes[0].num_elements() == 0)
        {
            return true;
        }

        if (ctx->input(4).NumElements() == 0)
        {
            return true;
        }

        absl::Cleanup lock_cleanup = [this] { Unlock(); };
        const Tensor input_tensor = GetInputTensor(ctx);

        if (input_tensor.NumElements() == 0)
        {
            return true;
        }

        return false;
    }

    StridedSliceAssignInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
    {
        if (ctx->input(0).dtype() == TF_RESOURCE)
        {
            OP_REQUIRES_OK(
                ctx,
                LookupResource(ctx, HandleFromInput(ctx, 0), &input_resource_));
            input_resource_->mu()->lock_shared();
            locked_ = true;
        }

        const Tensor input = GetInputTensor(ctx);

        TensorShape processing_shape;
        bool slice_dim0 = true;
        bool is_simple_slice = true;
        absl::InlinedVector<int64_t, 4> begin;
        absl::InlinedVector<int64_t, 4> end;
        absl::InlinedVector<int64_t, 4> strides;

        TensorShape input_shape = input.shape();
        TensorShape final_shape;

        OP_REQUIRES_OK(
            ctx,
            ValidateStridedSliceOp(
                &ctx->input(1),
                &ctx->input(2),
                ctx->input(3),
                input_shape,
                attr->begin_mask,
                attr->end_mask,
                attr->ellipsis_mask,
                attr->new_axis_mask,
                attr->shrink_axis_mask,
                &processing_shape,
                &final_shape,
                &is_identity_,
                &is_simple_slice,
                &slice_dim0,
                &begin,
                &end,
                &strides));

        if (processing_shape.num_elements())
        {
            TensorShape values_shape = ctx->input(4).shape();
            OP_REQUIRES(
                ctx,
                final_shape == values_shape,
                errors::Unimplemented(
                    "sliced l-value shape ",
                    final_shape.DebugString(),
                    " does not match r-value shape ",
                    values_shape.DebugString(),
                    ". Automatic broadcasting not ",
                    "yet implemented."));
        }

        // Attempt to simplify the slice into a lower-rank slice.
        simple_slice_ = SimplifySlice(input_shape, begin, end, strides);
        if (!simple_slice_)
        {
            OP_REQUIRES(
                ctx,
                simple_slice_,
                errors::InvalidArgument(
                    "DML only support slicing up to 8D inputs, "
                    "but received ",
                    input_shape.dims()));
        }
    }

    Tensor GetInputTensor(OpKernelContext* ctx) const
    {
        return ctx->input(0).dtype() == TF_RESOURCE ? *input_resource_->tensor()
                                                    : ctx->input(0);
    }

    void Unlock() const
    {
        if (input_resource_ && locked_)
        {
            input_resource_->mu()->unlock_shared();
            locked_ = false;
        }
    }

    const absl::optional<SimplifiedSlice>& GetSimplifiedSlice() const
    {
        return simple_slice_;
    }

    const bool IsIdentity() const { return is_identity_; }

  private:
    absl::optional<SimplifiedSlice> simple_slice_;
    RefCountPtr<Var> input_resource_;
    mutable bool locked_ = false;
    bool is_identity_;
};

class DmlStridedSliceAssignKernel : public DmlKernel
{
  public:
    using InitHelper = StridedSliceAssignInitHelper;

    explicit DmlStridedSliceAssignKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        const Tensor input =
            init_helper->GetInputTensor(ctx->GetOpKernelContext());
        const TensorShape& input_shape = input.shape();
        const TensorShape& updates_shape = ctx->GetInputTensorShape(4);

        auto simple_slice = init_helper->GetSimplifiedSlice();
        auto dtype_tf = ctx->GetInputDataType(4);

        dml::TensorDimensions collapsed_input_sizes = {
            1,
            1,
            1,
            static_cast<uint32_t>(input_shape.num_elements()),
        };

        dml::TensorDimensions collapsed_updates_sizes = {
            1,
            1,
            1,
            static_cast<uint32_t>(updates_shape.num_elements()),
        };

        DmlTensorInfo updates;
        updates.kernel_index = 4;
        updates.desc = DmlTensorDesc::Create(
            dtype_tf,
            collapsed_updates_sizes,
            collapsed_updates_sizes);

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            dtype_tf,
            collapsed_input_sizes,
            collapsed_input_sizes);

        DmlKernelTensors tensors;
        tensors.inputs = {updates};

        if (!init_helper->IsIdentity())
        {
            DmlTensorInfo original_input;
            original_input.kernel_index = 0;
            original_input.desc = DmlTensorDesc::Create(
                dtype_tf,
                collapsed_input_sizes,
                collapsed_input_sizes);

            tensors.inputs.push_back(original_input);
        }

        tensors.outputs = {output};

        if (input.dtype() != TF_RESOURCE)
        {
            // The input ref and the output ref must refer to the same memory
            tensors.output_refs_forwarding = {0};
        }

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto updates_tensor = dml::InputTensor(scope, 0, inputs[0]);

        dml::Expression result;

        if (init_helper->IsIdentity())
        {
            result = dml::Identity(updates_tensor);
        }
        else
        {
            auto original_input_tensor = dml::InputTensor(scope, 1, inputs[1]);

            auto indices_start =
                dml::ScalarUnion(0, DML_TENSOR_DATA_TYPE_UINT32);
            auto indices_delta =
                dml::ScalarUnion(1, DML_TENSOR_DATA_TYPE_UINT32);

            auto indices = dml::FillValueSequence(
                scope,
                simple_slice->input_sizes,
                DML_TENSOR_DATA_TYPE_UINT32,
                indices_start,
                indices_delta);

            auto sliced_indices = dml::Slice(
                indices,
                simple_slice->window_offset,
                simple_slice->window_sizes,
                simple_slice->window_strides);

            sliced_indices =
                dml::Reinterpret(sliced_indices, collapsed_updates_sizes, {});

            result = dml::ScatterElements(
                original_input_tensor,
                sliced_indices,
                updates_tensor,
                3);
        }

        // TFDML #24881131
        if (Is64BitSignedIntegerType(ctx->GetInputDataType(4)))
        {
            result = dml::ConvertInt32ToInt64(result);
        }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override
    {
        auto init_helper = ctx->GetInitializationHelper<InitHelper>();
        absl::Cleanup lock_cleanup = [init_helper] { init_helper->Unlock(); };

        const Tensor input_tensor =
            init_helper->GetInputTensor(ctx->GetOpKernelContext());

        // Identity can be done in-place
        if (init_helper->IsIdentity())
        {
            D3D12BufferRegion input_buffer =
                ctx->GetDmlDeviceContext()->GetBufferForTensor(
                    ctx->GetInputTensor(4));

            D3D12BufferRegion output_buffer =
                ctx->GetDmlDeviceContext()->GetBufferForTensor(input_tensor);

            absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
                input_buffer.GetBufferBinding(),
            };

            absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
                output_buffer.GetBufferBinding(),
            };

            return DmlKernel::Compute(ctx, input_bindings, output_bindings);
        }

        // Create input buffers
        D3D12BufferRegion input_buffers[] = {
            ctx->GetDmlDeviceContext()->GetBufferForTensor(
                ctx->GetInputTensor(4)),
            ctx->GetDmlDeviceContext()->GetBufferForTensor(input_tensor),
        };

        // Create input bindings
        absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
            input_buffers[0].GetBufferBinding(),
            input_buffers[1].GetBufferBinding(),
        };

        DmlBuffer output_buffer =
            ctx->GetDmlDeviceContext()->AllocateDefaultBuffer(
                ctx->GetOpKernelContext(),
                input_buffers[1].SizeInBytes());

        absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
            output_buffer.GetBufferBinding(),
        };

        auto status_or_event =
            DmlKernel::Compute(ctx, input_bindings, output_bindings);
        if (!status_or_event.ok())
        {
            return status_or_event;
        }

        ctx->GetDmlDeviceContext()->CopyBufferToBuffer(
            input_buffers[1],
            output_buffer.Region());

        return ctx->GetDmlDeviceContext()->InsertUavBarrier();
    }
};

void RegisterStridedSlice()
{
    using K = KernelDefinition<
        ops::StridedSlice,
        DmlKernelWrapper<DmlStridedSliceKernel, StridedSliceShapeHelper>>::
        WithHostMemoryArguments<
            ops::StridedSlice::Argument::begin,
            ops::StridedSlice::Argument::end,
            ops::StridedSlice::Argument::strides>;

    RegisterWithTypes<
        K,
        ops::StridedSlice::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_UINT8,
        TF_UINT32,
        TF_INT64>();
}

void RegisterStridedSliceGrad()
{
    using K = KernelDefinition<
        ops::StridedSliceGrad,
        DmlKernelWrapper<DmlStridedSliceGradKernel, StridedSliceShapeHelper>>::
        WithHostMemoryArguments<
            ops::StridedSliceGrad::Argument::begin,
            ops::StridedSliceGrad::Argument::shape,
            ops::StridedSliceGrad::Argument::end,
            ops::StridedSliceGrad::Argument::strides>;

    RegisterWithTypes<
        K,
        ops::StridedSliceGrad::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_UINT8,
        TF_UINT32,
        TF_INT64>();
}

void RegisterStridedSliceAssign()
{
    using K = KernelDefinition<
        ops::StridedSliceAssign,
        DmlKernelWrapper<
            DmlStridedSliceAssignKernel,
            GetOutputShapeAsInputShapeHelper>>::
        WithHostMemoryArguments<
            ops::StridedSliceAssign::Argument::begin,
            ops::StridedSliceAssign::Argument::end,
            ops::StridedSliceAssign::Argument::strides>;

    RegisterWithTypes<
        K,
        ops::StridedSliceAssign::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_UINT8,
        TF_UINT32,
        TF_INT64>();
}

void RegisterResourceStridedSliceAssign()
{
    using K = KernelDefinition<
        ops::ResourceStridedSliceAssign,
        DmlKernelWrapper<
            DmlStridedSliceAssignKernel,
            NoOutputShapeHelper,
            DmlKernelCachePolicy::Never>>::
        WithHostMemoryArguments<
            ops::ResourceStridedSliceAssign::Argument::ref,
            ops::ResourceStridedSliceAssign::Argument::begin,
            ops::ResourceStridedSliceAssign::Argument::end,
            ops::ResourceStridedSliceAssign::Argument::strides>;

    RegisterWithTypes<
        K,
        ops::ResourceStridedSliceAssign::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_BOOL,
        TF_INT8,
        TF_UINT8,
        TF_UINT32,
        TF_INT64>();
}

void RegisterKernels_StridedSlice()
{
    RegisterStridedSlice();
    RegisterStridedSliceGrad();
    RegisterStridedSliceAssign();

    // TODO: Enable when TF_RESOURCE deserialization across ABI is enabled
    // https://github.com/tensorflow/tensorflow/issues/53531
    // RegisterResourceStridedSliceAssign();
}
} // namespace tfdml