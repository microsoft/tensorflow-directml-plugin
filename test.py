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
"""Functional tests for segment reduction ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class HostToDeviceCopyTest(test.TestCase):
    def testGradientsGradientTape(self):
        np_x = (
            np.arange(1, 3)
            .reshape((1, 2))
            .astype(dtypes_lib.float32.as_numpy_dtype)
        )

        def callback(x):
            return math_ops.unsorted_segment_sum(x, np.array([0]), 3)

        with test_util.use_gpu():
            # pylint: disable=cell-var-from-loop

            gradient_tape_jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
                callback, [np_x], delta=1.0
            )
            # pylint: enable=cell-var-from-loop
            self.assertAllCloseAccordingToType(jacob_n, gradient_tape_jacob_t)


if __name__ == "__main__":
    test.main()
