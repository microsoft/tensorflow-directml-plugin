# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensorflow.ops.numerics."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import numerics
from tensorflow.python.platform import test


class VerifyTensorAllFiniteTest(test.TestCase):

  def testVerifyTensorAllFiniteSucceeds(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    with test_util.use_gpu():
      t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
      t_verified = numerics.verify_tensor_all_finite(t,
                                                     "Input is not a number.")
      self.assertAllClose(x, self.evaluate(t_verified))

  def testVerifyTensorAllFiniteFails(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    my_msg = "Input is not a number."

    # Test NaN.
    x[0] = np.nan
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)

    # Test Inf.
    x[0] = np.inf
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)


if __name__ == "__main__":
  test.main()