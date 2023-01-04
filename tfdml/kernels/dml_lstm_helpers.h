/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

namespace tfdml
{
class OpKernelContext;

inline dml::TensorDesc::Dimensions DimensionFromOffset(
    const Eigen::array<Eigen::DenseIndex, 2>& offset)
{
    return dml::TensorDesc::Dimensions{
        0,
        0,
        static_cast<uint32_t>(offset[0]),
        static_cast<uint32_t>(offset[1])};
};

inline dml::TensorDesc::Dimensions DimensionFromExtent(
    const Eigen::array<Eigen::DenseIndex, 2>& extent)
{
    return dml::TensorDesc::Dimensions{
        1,
        1,
        static_cast<uint32_t>(extent[0]),
        static_cast<uint32_t>(extent[1])};
};

enum GateLayout
{
    ICFO,
    IFCO
};

constexpr int gate_c_offset(GateLayout gate_layout, int cell_size)
{
    return (gate_layout == ICFO) ? cell_size : cell_size * 2;
}

constexpr int gate_f_offset(GateLayout gate_layout, int cell_size)
{
    return (gate_layout == ICFO) ? cell_size * 2 : cell_size;
}

namespace functor
{

struct LSTMBlockCell
{
    LSTMBlockCell(
        const int batch_size,
        const int input_size,
        const int cell_size)
        : batch_size_(batch_size),
          input_size_(input_size),
          cell_size_(cell_size)
    {
    }

    int batch_size() const { return batch_size_; }

    int input_size() const { return input_size_; }

    int cell_size() const { return cell_size_; }

    inline Eigen::array<Eigen::DenseIndex, 2> gates_i_offsets() const
    {
        return {0, 0};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> gates_c_offsets(
        const GateLayout gate_layout) const
    {
        return {0, gate_c_offset(gate_layout, cell_size_)};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> gates_f_offsets(
        const GateLayout gate_layout) const
    {
        return {0, gate_f_offset(gate_layout, cell_size_)};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> gates_o_offsets() const
    {
        return {0, cell_size_ * 3};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> cell_extents() const
    {
        return {batch_size_, cell_size_};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> xh_x_offsets() const
    {
        return {0, 0};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> xh_x_extents() const
    {
        return {batch_size_, input_size_};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> xh_h_offsets() const
    {
        return {0, input_size_};
    }

    inline Eigen::array<Eigen::DenseIndex, 2> xh_h_extents() const
    {
        return {batch_size_, cell_size_};
    }

  protected:
    const int batch_size_;
    const int input_size_;
    const int cell_size_;
};

} // namespace functor
} // namespace tfdml