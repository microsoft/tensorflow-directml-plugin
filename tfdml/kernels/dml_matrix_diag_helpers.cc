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

#define NOMINMAX
#include "tfdml/kernels/dml_matrix_diag_helpers.h"
#include "tfdml/kernels/pch.h"
#include <numeric>

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
    bool align_sub_left)
{
    assert(k_min <= k_max);

    dml::TensorDesc::Dimensions diag_shape = diag.GetOutputDesc().sizes;
    uint32_t diag_depth = diag_shape[diag_shape.size() - 2];
    uint32_t diag_width = diag_shape[diag_shape.size() - 1];

    std::vector<int32_t> k_range;
    std::generate_n(
        std::back_inserter(k_range),
        k_max - k_min + 1,
        [val = k_min]() mutable { return std::abs(val++); });

    int32_t min_k2zero = *std::min_element(k_range.begin(), k_range.end());
    int32_t max_diag_len = min_k2zero + diag_width;
    int64_t k_btm = 1 - out_height;

    // Get K lengths
    uint32_t rwcl_min = std::min(out_height, out_width);
    uint32_t rwcl_gap = std::abs(out_height - out_width);

    // Build k_lens by inserting series in the following order:
    //   Left bottom
    //   Middle
    //   Top right
    // The result will be a vector that looks like [1 2 3 4 1 1 1 4 3 2 1]
    std::vector<int> k_lens_cpu(rwcl_min - 1);
    std::iota(k_lens_cpu.begin(), k_lens_cpu.end(), 1);
    k_lens_cpu.resize(k_lens_cpu.size() + rwcl_gap + 1, rwcl_min);
    std::generate_n(
        std::back_inserter(k_lens_cpu),
        rwcl_min - 1,
        [val = rwcl_min - 1]() mutable { return val--; });

    // MatrixDiag and MatrixDiagV2 align both the superdiagonal and the
    // subdiagonal to the left
    int k_sup_stt = std::max(0, k_min) - k_btm;
    int k_sup_end = std::max<int>(k_max + 1 - k_btm, k_sup_stt);
    int k_sub_stt = k_min - k_btm;
    int k_sub_end = std::max<int>(std::min(0, k_max + 1) - k_btm, k_sub_stt);

    std::vector<int> all_k_lens(
        k_lens_cpu.begin() + k_sub_stt,
        k_lens_cpu.begin() + k_sub_end);

    std::copy(
        k_lens_cpu.begin() + k_sup_stt,
        k_lens_cpu.begin() + k_sup_end,
        std::back_inserter(all_k_lens));

    int max_k_len = *std::max_element(all_k_lens.begin(), all_k_lens.end());
    int top_k_len = all_k_lens.back();
    int btm_k_len = all_k_lens.front();

    uint32_t diag_elem_count = std::accumulate(
        diag_shape.begin(),
        diag_shape.end(),
        1u,
        std::multiplies<uint32_t>());

    dml::TensorDesc::Dimensions diag_rev_shape(
        {1u, 1u, diag_elem_count / diag_width, diag_width});
    auto reshaped_diag = dml::Reinterpret(diag, diag_rev_shape, {});

    uint32_t k_sub_len = std::max(0, k_sub_end - k_sub_stt);
    uint32_t k_sup_len = std::max(0, k_sup_end - k_sup_stt);

    dml::Expression k_lens;
    if ((k_sub_len != 0 && align_sub_left) ||
        (k_sup_len != 0 && !align_sup_left))
    {
        auto left_btm =
            dml::Sequence<int32_t>(scope, 1, 1, {1, 1, 1, rwcl_min - 1});

        auto right_top = dml::Sequence<int32_t>(
            scope,
            rwcl_min - 1,
            -1,
            {1, 1, 1, rwcl_min - 1});

        auto klen_mid = dml::ScalarTensor<int32_t>(
            scope,
            rwcl_min,
            {1, 1, 1, rwcl_gap + 1});

        k_lens = dml::Join({left_btm, klen_mid, right_top}, 3);
    }

    dml::Expression sup_rev_len_1;
    dml::Expression sup_rev_len_2;
    if (k_sup_len != 0)
    {
        if (align_sup_left)
        {
            sup_rev_len_1 =
                dml::ScalarTensor<uint32_t>(scope, 1, {1, 1, 1, k_sup_len});

            sup_rev_len_2 = sup_rev_len_1;
        }
        else
        {
            sup_rev_len_1 = dml::ScalarTensor<uint32_t>(
                scope,
                diag_width,
                {1, 1, 1, k_sup_len});

            sup_rev_len_2 = dml::Slice(
                k_lens,
                {0, 0, 0, static_cast<uint32_t>(k_sup_stt)},
                {1, 1, 1, k_sup_len},
                {1, 1, 1, 1});
            sup_rev_len_2 =
                dml::Reinterpret(sup_rev_len_2, DML_TENSOR_DATA_TYPE_UINT32);
        }
    }

    dml::Expression sub_rev_len_1;
    dml::Expression sub_rev_len_2;
    if (k_sub_len != 0)
    {
        if (align_sub_left)
        {
            sub_rev_len_1 = dml::Slice(
                k_lens,
                {0, 0, 0, static_cast<uint32_t>(k_sub_stt)},
                {1, 1, 1, k_sub_len},
                {1, 1, 1, 1});
            sub_rev_len_1 =
                dml::Reinterpret(sub_rev_len_1, DML_TENSOR_DATA_TYPE_UINT32);

            sub_rev_len_2 = dml::ScalarTensor<uint32_t>(
                scope,
                diag_width,
                {1, 1, 1, k_sub_len});
        }
        else
        {
            sub_rev_len_1 =
                dml::ScalarTensor<uint32_t>(scope, 1, {1, 1, 1, k_sub_len});
            sub_rev_len_2 = sub_rev_len_1;
        }
    }

    // Build cnt_rev_len_1 and cnt_rev_len_2
    dml::Expression cnt_rev_len_1;
    dml::Expression cnt_rev_len_2;
    if (k_sub_len == 0)
    {
        cnt_rev_len_1 = sup_rev_len_1;
        cnt_rev_len_2 = sup_rev_len_2;
    }
    else if (k_sup_len == 0)
    {
        cnt_rev_len_1 = sub_rev_len_1;
        cnt_rev_len_2 = sub_rev_len_2;
    }
    else
    {
        cnt_rev_len_1 = dml::Join({sub_rev_len_1, sup_rev_len_1}, 3);
        cnt_rev_len_2 = dml::Join({sub_rev_len_2, sup_rev_len_2}, 3);
    }

    auto cnt_rev_len_length = cnt_rev_len_1.GetOutputDesc().sizes.back();
    auto exp_rev_len_1 = cnt_rev_len_1;
    auto exp_rev_len_2 = cnt_rev_len_2;

    // Build exp_rev_len_1
    if (cnt_rev_len_length > 1)
    {
        auto exp_rev_len_seqs = dml::ScalarTensor<uint32_t>(
            scope,
            cnt_rev_len_length,
            {1, 1, 1, 1});

        exp_rev_len_1 =
            dml::ReverseSubsequences(exp_rev_len_1, exp_rev_len_seqs, 3);

        exp_rev_len_2 =
            dml::ReverseSubsequences(exp_rev_len_2, exp_rev_len_seqs, 3);
    }

    // Broadcast exp_rev_len_1 to match reshaped_diag
    dml::TensorDesc::Dimensions rev_shape({
        1,
        1,
        diag_elem_count / diag_width / cnt_rev_len_length,
        cnt_rev_len_length,
    });

    dml::TensorDesc::Dimensions rev_strides({0, 0, 0, 1});

    dml::TensorDesc::Dimensions reshaped_rev_sizes({
        1,
        1,
        diag_elem_count / diag_width,
        1,
    });

    // Broadcast and reshape exp_rev_len_1, which specifies the length to
    // reverse each row of each batch
    exp_rev_len_1 = dml::Reinterpret(exp_rev_len_1, rev_shape, rev_strides);
    exp_rev_len_1 = dml::Identity(exp_rev_len_1);
    exp_rev_len_1 = dml::Reinterpret(exp_rev_len_1, reshaped_rev_sizes, {});

    auto reversed_diag_1 =
        dml::ReverseSubsequences(reshaped_diag, exp_rev_len_1, 3);

    // Broadcast and reshape exp_rev_len_2, which specifies the length to
    // reverse each row of each batch
    exp_rev_len_2 = dml::Reinterpret(exp_rev_len_2, rev_shape, rev_strides);
    exp_rev_len_2 = dml::Identity(exp_rev_len_2);
    exp_rev_len_2 = dml::Reinterpret(exp_rev_len_2, reshaped_rev_sizes, {});

    auto sorted_diag =
        dml::ReverseSubsequences(reversed_diag_1, exp_rev_len_2, 3);

    // DML only supports until 5D for Identity, but we can coalesce together the
    // dimensions that don't need to be transposed
    uint32_t head_shape_elem_count = std::accumulate(
        diag_shape.begin(),
        diag_shape.end() - 2,
        1u,
        std::multiplies<uint32_t>());

    dml::TensorDesc::Dimensions tran_diag_sizes({
        head_shape_elem_count,
        1,
        diag_width,
        diag_depth,
    });

    dml::TensorDesc::Dimensions tran_diag_strides({
        diag_width * diag_depth,
        diag_width * diag_depth,
        1,
        diag_width,
    });

    auto tran_diag =
        dml::Reinterpret(sorted_diag, tran_diag_sizes, tran_diag_strides);

    tran_diag = dml::Identity(tran_diag);

    // Make the diagonal
    uint32_t width = tran_diag_sizes.back();
    uint32_t height = diag_elem_count / width;
    auto reshaped_tran_diag =
        dml::Reinterpret(tran_diag, {1, 1, height, width}, {});

    uint32_t top_pad = max_k_len - top_k_len;
    uint32_t btm_pad = max_k_len - btm_k_len;
    uint32_t left_pad = top_pad;
    uint32_t right_pad = btm_pad + diag_width;

    auto diag_pad = dml::Padding(
        reshaped_tran_diag,
        DML_PADDING_MODE_CONSTANT,
        padding_value,
        {0, 0, 0, left_pad},
        {0, 0, 0, right_pad});

    int diag_pad_width = width + left_pad + right_pad;

    dml::TensorDesc::Dimensions exp_shape({
        1,
        1,
        head_shape_elem_count,
        diag_width,
    });

    dml::TensorDesc::Dimensions exp_shape_reshaped(
        {1, 1, head_shape_elem_count * diag_width, 1});

    int32_t rg_from = (static_cast<int32_t>(left_pad) * 2) -
                      static_cast<int32_t>(diag_width) + 1;
    auto rg = dml::Sequence<int32_t>(scope, rg_from, 1, {1, 1, 1, diag_width});
    rg = dml::ActivationRelu(rg);

    auto expanded_range = dml::Reinterpret(
        rg,
        DML_TENSOR_DATA_TYPE_UINT32,
        exp_shape,
        dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    expanded_range = dml::Identity(expanded_range);

    auto reshaped_range =
        dml::Reinterpret(expanded_range, exp_shape_reshaped, {});

    auto pad_left = dml::ReverseSubsequences(diag_pad, reshaped_range, 3);
    auto pad_left_shape = pad_left.GetOutputDesc().sizes;
    pad_left_shape[3] = diag_pad_width - left_pad;

    if (left_pad > 0)
    {
        pad_left = dml::Slice(
            pad_left,
            {0, 0, 0, left_pad},
            pad_left_shape,
            {1, 1, 1, 1});
    }

    uint32_t pad_left_depth = pad_left_shape[2];
    uint32_t pad_left_width = pad_left_shape[3];

    auto pad_full_length = dml::ScalarTensor<uint32_t>(
        scope,
        pad_left_width,
        {1, 1, pad_left_depth, 1});

    auto rev = dml::ReverseSubsequences(pad_left, pad_full_length, 3);

    auto rg2 = dml::Sequence<int32_t>(
        scope,
        right_pad + btm_pad,
        -1,
        {1, 1, 1, diag_width});

    auto expanded_range2 = dml::Reinterpret(
        rg2,
        exp_shape,
        dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    expanded_range2 = dml::Identity(expanded_range2);

    auto reshaped_range2 = dml::Reinterpret(
        expanded_range2,
        DML_TENSOR_DATA_TYPE_UINT32,
        exp_shape_reshaped,
        {});

    auto raw_pad_right = dml::ReverseSubsequences(rev, reshaped_range2, 3);
    auto raw_pad_right_shape = raw_pad_right.GetOutputDesc().sizes;
    int raw_pad_right_width = raw_pad_right_shape.back();

    auto sliced_raw_pad_right = raw_pad_right;

    if (btm_pad > 0)
    {
        raw_pad_right_shape.back() = raw_pad_right_width - btm_pad;

        sliced_raw_pad_right = dml::Slice(
            raw_pad_right,
            {0, 0, 0, btm_pad},
            raw_pad_right_shape,
            {1, 1, 1, 1});
    }

    // Build all_width
    auto all_width = dml::ScalarTensor<uint32_t>(
        scope,
        raw_pad_right_width - btm_pad,
        exp_shape_reshaped);

    auto pad_right =
        dml::ReverseSubsequences(sliced_raw_pad_right, all_width, 3);

    // Diagonalize
    auto rg3 = dml::Sequence<uint32_t>(
        scope,
        diag_depth - btm_pad,
        1,
        {1, 1, 1, diag_width});

    auto expanded_range3 = dml::Reinterpret(
        rg3,
        exp_shape,
        dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    expanded_range3 = dml::Identity(expanded_range3);

    auto reshaped_range3 =
        dml::Reinterpret(expanded_range3, exp_shape_reshaped, {});

    auto rev2 = dml::ReverseSubsequences(pad_right, reshaped_range3, 3);

    int k_max_idx = k_max - k_btm;
    int k_max_len = k_lens_cpu[k_max_idx];
    int k_gap = std::abs(k_max) - min_k2zero;
    int diagonalize_width = k_max_len + k_gap;

    auto sliced_rev2_shape = rev2.GetOutputDesc().sizes;
    auto new_diag = rev2;

    if (diagonalize_width != sliced_rev2_shape.back())
    {
        sliced_rev2_shape.back() = diagonalize_width;
        new_diag =
            dml::Slice(new_diag, {0, 0, 0, 0}, sliced_rev2_shape, {1, 1, 1, 1});
    }

    int new_diag_elem_count = std::accumulate(
        sliced_rev2_shape.begin(),
        sliced_rev2_shape.end(),
        1u,
        std::multiplies<uint32_t>());

    uint32_t new_depth = diag_width;
    uint32_t new_width =
        new_diag_elem_count / head_shape_elem_count / diag_width;

    dml::TensorDesc::Dimensions new_diag_shape =
        {1, head_shape_elem_count, new_depth, new_width};

    new_diag = dml::Reinterpret(new_diag, new_diag_shape, {});

    // Finally, pad to output shape
    uint32_t pad_row = out_height - new_depth;
    uint32_t pad_col = out_width - new_width;
    uint32_t pad_top = std::max(0, -k_max);
    uint32_t pad_lft = std::max(0, k_min);
    uint32_t pad_btm = pad_row - pad_top;
    uint32_t pad_rht = pad_col - pad_lft;
    auto result = dml::Padding(
        new_diag,
        DML_PADDING_MODE_CONSTANT,
        padding_value,
        {0, 0, pad_top, pad_lft},
        {0, 0, pad_btm, pad_rht});

    return result;
}

static dml::Expression RightAlign(
    int32_t maxsize,
    int32_t stride,
    dml::Expression sizes,
    dml::Expression indices,
    dml::Expression starts,
    dml::Expression maxval)
{
    auto op1 = maxsize - sizes;
    auto op2 = op1 * stride;
    auto op3 = indices - op2;
    auto op4 = op3 < starts;
    auto op5 = dml::If(op4, maxval, op3);
    return op5;
}

dml::Expression MatrixDiagPart(
    dml::Graph& scope,
    dml::Expression m,
    int32_t k0,
    int32_t k1,
    float padding_value,
    uint32_t out_height,
    uint32_t out_width,
    bool align_sup_left,
    bool align_sub_left)
{
    dml::TensorDesc::Dimensions input_shape = m.GetOutputDesc().sizes;
    uint32_t xlen = input_shape[input_shape.size() - 1];
    uint32_t ylen = input_shape[input_shape.size() - 2];
    uint32_t elem_count = std::accumulate(
        input_shape.begin(),
        input_shape.end(),
        1u,
        std::multiplies<uint32_t>());
    uint32_t leading_dims_size = elem_count / xlen / ylen;

    int32_t xlenp = xlen + 1;
    int32_t stride = xlenp + 1;
    int32_t xmax_0 = xlen * xlenp;
    int32_t xmax_1 = xmax_0 + xlenp;
    int32_t xmax = xmax_1 - 1;

    int32_t ymax_0 = xlenp * ylen;
    int32_t ymax = ymax_0 - 1;

    auto m_padded = dml::Padding(
        m,
        DML_PADDING_MODE_CONSTANT,
        padding_value,
        {0, 0, 0, 0},
        {0, 0, 1, 1});

    auto m2 = dml::Reinterpret(
        m_padded,
        {1, 1, leading_dims_size, (ylen + 1) * (xlen + 1)},
        {});

    uint32_t minxy = std::min(xlen, ylen);

    dml::Expression diags_indices;

    int32_t xstart_0 = k0;
    int32_t xstart_1 = std::max(0, xstart_0);
    int32_t xstart_2 = xstart_1;
    int32_t xstart_3 = xstart_2 - 1;
    int32_t xdiag_size = k1 - xstart_3;
    // TODO (pavignol): Remove if we don't need anymore
    std::vector<int32_t> xstart_4;
    std::generate_n(
        std::back_inserter(xstart_4),
        xdiag_size,
        [val = k1]() mutable { return val--; });
    const std::vector<int32_t>& xstart = xstart_4;

    int32_t ystart_0 = k1;
    int32_t ystart_1 = std::min(-1, ystart_0);
    int32_t ystart_2 = ystart_1;
    int32_t ystart_3 = k0 - 1;
    int32_t ydiag_size = ystart_2 - ystart_3;
    // TODO (pavignol): Remove if we don't need anymore
    std::vector<int32_t> ystart_4;
    std::generate_n(
        std::back_inserter(ystart_4),
        ydiag_size,
        [val = ystart_2]() mutable { return val--; });
    const std::vector<int32_t>& ystart = ystart_4;

    std::vector<int32_t> xsize(xstart.size());
    for (int i = 0; i < xsize.size(); ++i)
    {
        int32_t xsize_0 = xlen - xstart[i];
        int32_t xsize_1 = xsize_0;
        int32_t xsize_2 = std::min<int32_t>(xsize_1, minxy);
        xsize[i] = xsize_2;
    }

    std::vector<int32_t> ysize(ystart.size());
    for (int i = 0; i < ysize.size(); ++i)
    {
        int32_t ysize_0 = ylen + ystart[i];
        int32_t ysize_1 = ysize_0;
        int32_t ysize_2 = std::min<int32_t>(ysize_1, minxy);
        ysize[i] = ysize_2;
    }

    int32_t maxsize = INT_MIN;
    if (xdiag_size > 0)
    {
        maxsize = *std::max_element(xsize.begin(), xsize.end());
    }

    if (ydiag_size > 0)
    {
        maxsize =
            std::max(maxsize, *std::max_element(ysize.begin(), ysize.end()));
    }

    int32_t maxsize_0 = maxsize;
    int32_t maxsize_scalar = maxsize_0;

    auto diagdistances = dml::Sequence<int32_t>(
        scope,
        0,
        stride,
        {1, 1, 1, static_cast<uint32_t>(maxsize_scalar)});

    dml::Expression minxy_gpu;
    if ((xdiag_size > 0 && !align_sup_left) ||
        (ydiag_size > 0 && !align_sub_left))
    {
        minxy_gpu = dml::ScalarTensor<int32_t>(scope, minxy, {1, 1, 1, 1});
    }

    dml::Expression ymax_gpu;
    if ((xdiag_size > 0 && !align_sup_left) || ydiag_size > 0)
    {
        ymax_gpu = dml::ScalarTensor<int32_t>(scope, ymax, {1, 1, 1, 1});
    }

    // Starting indices for super diagonals
    dml::Expression xdiags;
    if (xdiag_size > 0)
    {
        dml::TensorDesc::Dimensions broadcast_sizes({
            1,
            1,
            static_cast<uint32_t>(xdiag_size),
            static_cast<uint32_t>(maxsize_scalar),
        });

        auto xstart_4_gpu = dml::Sequence<int32_t>(
            scope,
            k1,
            -1,
            {1, 1, static_cast<uint32_t>(xdiag_size), 1});
        auto xstart_gpu = dml::Reinterpret(
            xstart_4_gpu,
            broadcast_sizes,
            dml::TensorDesc::Dimensions({0, 0, 1, 0}));
        diagdistances = dml::Reinterpret(
            diagdistances,
            broadcast_sizes,
            dml::TensorDesc::Dimensions({0, 0, 0, 1}));

        auto xdiags_0 = xstart_gpu + diagdistances;

        if (align_sup_left)
        {
            auto xmax_0_gpu = xstart_gpu * xlenp;
            auto xmax_gpu = xmax - xmax_0_gpu;
            auto xdiags_1 = xdiags_0;
            auto xdiags_2 = dml::Min(xdiags_1, xmax_gpu);
            xdiags = xdiags_2;
        }
        else
        {
            minxy_gpu = dml::Reinterpret(
                minxy_gpu,
                broadcast_sizes,
                dml::TensorStrides({0, 0, 0, 0}));
            ymax_gpu = dml::Reinterpret(
                ymax_gpu,
                broadcast_sizes,
                dml::TensorStrides({0, 0, 0, 0}));

            auto xsize_0 = xlen - xstart_gpu;
            auto xsize_1 = xsize_0;
            auto xsize_2 = dml::Min(xsize_1, minxy_gpu);
            auto xsize = xsize_2;
            xdiags = RightAlign(
                maxsize,
                stride,
                xsize,
                xdiags_0,
                xstart_gpu,
                ymax_gpu);
        }
        diags_indices = xdiags;
    }

    // Starting indices for sub diagonals
    dml::Expression ydiags;
    if (ydiag_size > 0)
    {
        dml::TensorDesc::Dimensions broadcast_sizes({
            1,
            1,
            static_cast<uint32_t>(ydiag_size),
            static_cast<uint32_t>(maxsize_scalar),
        });

        auto ystart_4_gpu = dml::Sequence<int32_t>(
            scope,
            ystart_2,
            -1,
            {1, 1, static_cast<uint32_t>(ydiag_size), 1});
        auto ystart_gpu = dml::Reinterpret(
            ystart_4_gpu,
            broadcast_sizes,
            dml::TensorDesc::Dimensions({0, 0, 1, 0}));
        diagdistances = dml::Reinterpret(
            diagdistances,
            broadcast_sizes,
            dml::TensorDesc::Dimensions({0, 0, 0, 1}));

        auto ydiags_0 = dml::Abs(ystart_gpu);
        auto ydiags_1 = ydiags_0 * xlenp;
        auto ydiags_2 = ydiags_1 + diagdistances;

        ymax_gpu = dml::Reinterpret(
            ymax_gpu,
            broadcast_sizes,
            dml::TensorStrides({0, 0, 0, 0}));

        if (align_sub_left)
        {
            auto ydiags_3 = ydiags_2;
            auto ydiags_4 = dml::Min(ydiags_3, ymax_gpu);
            ydiags = ydiags_4;
        }
        else
        {
            minxy_gpu = dml::Reinterpret(
                minxy_gpu,
                broadcast_sizes,
                dml::TensorStrides({0, 0, 0, 0}));

            auto ysize_0 = ylen + ystart_gpu;
            auto ysize_1 = ysize_0;
            auto ysize_2 = dml::Min(ysize_1, minxy_gpu);
            auto ysize = ysize_2;
            ydiags = RightAlign(
                maxsize,
                stride,
                ysize,
                ydiags_2,
                ydiags_1,
                ymax_gpu);
        }
        diags_indices = ydiags;
    }

    if (xdiag_size > 0 && ydiag_size > 0)
    {
        diags_indices = dml::Join({xdiags, ydiags}, 2);
    }

    uint32_t diags_indices_elem_count = std::accumulate(
        diags_indices.GetOutputDesc().sizes.begin(),
        diags_indices.GetOutputDesc().sizes.end(),
        1u,
        std::multiplies<uint32_t>());

    // Reshape into a single row and broadcast to all batches
    diags_indices = dml::Reinterpret(
        diags_indices,
        {1, 1, leading_dims_size, diags_indices_elem_count},
        dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    auto diags = dml::GatherElements(m2, diags_indices, 3);
    return diags;
}

} // namespace dml
