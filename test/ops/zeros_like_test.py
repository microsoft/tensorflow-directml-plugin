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
"""Tests for tensorflow.ops.tf.zeros_like."""

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class ZerosLikeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testZerosLike(self):
    with self.cached_session():
      for dtype in [tf.float16, tf.float32, tf.int64]:
        const_input = tf.constant([[1, 1, 3], [4, 5, 6]], dtype=dtype)
        result = tf.raw_ops.ZerosLike(x=const_input)
        self.assertAllEqual([[0, 0, 0], [0, 0, 0]], result)


if __name__ == "__main__":
  test.main()