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
"""Tests for tensorflow.ops.tf.fill."""

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class FillTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testFill(self):
    with self.cached_session():
      fill = array_ops.fill([2, 3], 1.0)
      self.assertAllEqual([[1, 1, 1], [1, 1, 1]], fill)

  def testFillBadShape(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"dims must be a vector, got shape \[2,2\]"):
        array_ops.fill([[2, 3], [2, 3]], 1.0)

  def testFillBadValue(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"value must be a scalar, got shape \[2\]"):
        array_ops.fill([2, 3], [1.0, 2.0])


if __name__ == "__main__":
  test.main()