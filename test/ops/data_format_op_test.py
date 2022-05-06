# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tests for tensorflow.raw_ops.DataFormatVecPermute and
tensorflow.raw_ops.DataFormatDimMap
"""

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class DataFormatOpTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatVecPermute1D(self):
    with self.cached_session():
      permuted = tf.raw_ops.DataFormatVecPermute(
          x=[1, 2, 3, 4], src_format='NHWC', dst_format='NCHW')
      self.assertAllEqual([1, 4, 2, 3], permuted)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatVecPermute2D(self):
    with self.cached_session():
      permuted = tf.raw_ops.DataFormatVecPermute(
          x=[[1, 2], [3, 4], [5, 6], [7, 8]],
          src_format='NHWC',
          dst_format='NCHW')
      self.assertAllEqual([[1, 2], [7, 8], [3, 4], [5, 6]], permuted)

  def testDataFormatVecPermuteBadRank(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"input must be a vector or 2D tensor, but got shape \[1,1,4\]"):
        tf.raw_ops.DataFormatVecPermute(
            x=[[[1, 2, 3, 4]]],
            src_format='NHWC',
            dst_format='NCHW')

  def testDataFormatVecPermuteBadShape1D(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"1D input must be of size 2 or 4, but got shape \[5\]"):
        tf.raw_ops.DataFormatVecPermute(
            x=[1, 2, 3, 4, 5],
            src_format='NHWC',
            dst_format='NCHW')

  def testDataFormatVecPermuteBadShape2D(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"First dimension of 2D input must be of size 2 or 4, but got shape "
          r"\[5,2\]"):
        tf.raw_ops.DataFormatVecPermute(
            x=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            src_format='NHWC',
            dst_format='NCHW')

  def testDataFormatVecPermuteBadShapeSecondDim(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"Second dimension of 2D input must be of size 2, but got shape "
          r"\[4,3\]"):
        tf.raw_ops.DataFormatVecPermute(
            x=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            src_format='NHWC',
            dst_format='NCHW')

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapLayout4D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 1, -4, 0, 2, -3, -1], src_format='NHWC', dst_format='NCHW')
      self.assertAllEqual([1, 3, 2, 0, 0, 3, 2, 1], mapped)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapLayout5D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 4, 1, -4, 0, 2, -5, -3, -1],
          src_format='NDHWC',
          dst_format='NCDHW')
      self.assertAllEqual([4, 4, 1, 2, 2, 0, 3, 0, 3, 1], mapped)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapNumbers4D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 1, -4, 0, 2, -3, -1], src_format='0123', dst_format='0312')
      self.assertAllEqual([1, 3, 2, 0, 0, 3, 2, 1], mapped)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapNumbers5D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 4, 1, -4, 0, 2, -5, -3, -1],
          src_format='01234',
          dst_format='04123')
      self.assertAllEqual([4, 4, 1, 2, 2, 0, 3, 0, 3, 1], mapped)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapRandom4D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 1, -4, 0, 2, -3, -1], src_format='h9J!', dst_format='h!9J')
      self.assertAllEqual([1, 3, 2, 0, 0, 3, 2, 1], mapped)

  @test_util.run_in_graph_and_eager_modes
  def testDataFormatDimMapRandom5D(self):
    with self.cached_session():
      mapped = tf.raw_ops.DataFormatDimMap(
          x=[3, -2, 4, 1, -4, 0, 2, -5, -3, -1],
          src_format='h9J!*',
          dst_format='h*9J!')
      self.assertAllEqual([4, 4, 1, 2, 2, 0, 3, 0, 3, 1], mapped)

  def testDataFormatDimMapBadSrcLength(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"Source format must be of length 4 or 5, received src_format = "
          r"012345"):
        tf.raw_ops.DataFormatDimMap(
            x=[3, -2, 1, -4, 0, 2, -3, -1],
            src_format='012345',
            dst_format='01234')

  def testDataFormatDimMapBadDstLength(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"Destination format must be of length 4 or 5, received dst_format = "
          r"012345"):
        tf.raw_ops.DataFormatDimMap(
            x=[3, -2, 1, -4, 0, 2, -3, -1],
            src_format='01234',
            dst_format='012345')

  def testDataFormatDimMapBadPerms(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"Destination and source format must determine a permutation, got "
          r"1234 and 5678"):
        tf.raw_ops.DataFormatDimMap(
            x=[3, -2, 1, -4, 0, 2, -3, -1],
            src_format='1234',
            dst_format='5678')


if __name__ == "__main__":
  test.main()