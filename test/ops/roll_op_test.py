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
"""Tests for tensorflow.ops.tf.roll."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import test
import dml_test_util

class RollTest(dml_test_util.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testRollBasic(self):
    with self.cached_session():
      t = np.array([0, 1, 2, 3, 4])
      roll = manip_ops.roll(t, shift=2, axis=0)
      np_roll = np.roll(t, 2, 0)
      self.assertAllEqual(np_roll, roll)

  @test_util.run_in_graph_and_eager_modes
  def testRollMultipleDims(self):
    with self.cached_session():
      t = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
      roll = manip_ops.roll(t, shift=[1, -2], axis=[0, 1])
      np_roll = np.roll(t, [1, -2], [0, 1])
      self.assertAllEqual(np_roll, roll)

  @test_util.run_in_graph_and_eager_modes
  def testRollSameDim(self):
    with self.cached_session():
      t = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
      roll = manip_ops.roll(t, shift=[2, -3], axis=[1, 1])
      np_roll = np.roll(t, [2, -3], [1, 1])
      self.assertAllEqual(np_roll, roll)

  def testRollBadAxis(self):
    with self.cached_session():
      t = np.array([0, 1, 2, 3, 4])
      with self.assertRaisesOpError(
          r"axis 1 is out of range"):
        manip_ops.roll(t, shift=2, axis=1)
      with self.assertRaisesOpError(
          r"axis must be a scalar or a 1-D vector. Found: \[2,1\]"):
        manip_ops.roll(t, shift=2, axis=[[1],[1]])

  def testRollBadInput(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"input must be 1-D or higher"):
        manip_ops.roll(0, shift=1, axis=0)

  def testRollBadShift(self):
    with self.cached_session():
      t = np.array([0, 1, 2, 3, 4])
      with self.assertRaisesOpError(
          r"shift must be a scalar or a 1-D vector. Found: \[2,1\]"):
        manip_ops.roll(t, shift=[[1],[1]], axis=1)

  def testRollShiftAxisMismatch(self):
    with self.cached_session():
      t = np.array([0, 1, 2, 3, 4])
      with self.assertRaisesOpError(
          r"shift and axis must have the same size"):
        manip_ops.roll(t, shift=2, axis=[1, 1])


if __name__ == "__main__":
  test.main()