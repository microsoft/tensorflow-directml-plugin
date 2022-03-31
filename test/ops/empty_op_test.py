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
"""Tests for tensorflow.ops.tf.empty."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes
import dml_test_util

class EmptyTest(dml_test_util.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testEmpty(self):
        
        def empty_like(x, init=None):
            x = ops.convert_to_tensor(x)
            return gen_array_ops.empty(array_ops.shape(x), x.dtype, init=init)

        for dtype in [
            dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64]:
            with self.cached_session(use_gpu=True):
                test_shapes = [(), (1,), (2, 3), (0, 2), (2, 3, 5), (2, 0, 5)]
                for shape in test_shapes:
                    val = gen_array_ops.empty(shape, dtype)
                    self.assertEqual(val.shape, shape)
                    self.assertDTypeEqual(val, dtype)
                    val = gen_array_ops.empty(shape, dtype, init=True)
                    self.assertEqual(val.shape, shape)
                    self.assertDTypeEqual(val, dtype)
                    self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))
                    val = empty_like(array_ops.zeros(shape, dtype))
                    self.assertEqual(val.shape, shape)
                    self.assertDTypeEqual(val, dtype)
                    val = empty_like(
                        array_ops.zeros(shape, dtype), init=True)
                    self.assertEqual(val.shape, shape)
                    self.assertDTypeEqual(val, dtype)
                    self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))


if __name__ == "__main__":
  test.main()