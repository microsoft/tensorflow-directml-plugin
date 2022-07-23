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
"""Tests for tensorflow.ops.argmax_op."""
import functools

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMaxTest(test.TestCase):
    """ArgMaxTest"""

    def _test_arg(  # pylint:disable=too-many-arguments
        self,
        method,
        input_value,
        axis,
        expected_values,
        use_gpu=False,
        expected_err_re=None,
    ):
        with self.session(use_gpu=use_gpu):
            ans = method(input_value, axis=axis)
            if expected_err_re is None:
                tf_ans = self.evaluate(ans)
                # Defaults to int64 output.
                self.assertEqual(np.int64, tf_ans.dtype)
                self.assertAllEqual(tf_ans, expected_values)
                self.assertShapeEqual(expected_values, ans)
            else:
                with self.assertRaisesOpError(expected_err_re):
                    self.evaluate(ans)

    def _test_both_arg(  # pylint:disable=too-many-arguments
        self, method, input_value, axis, expected_values, expected_err_re=None
    ):
        self._test_arg(
            method, input_value, axis, expected_values, True, expected_err_re
        )
        # Compilation time is too large with XLA/CPU autojit.
        if not test_util.is_xla_enabled():
            self._test_arg(
                method, input_value, axis, expected_values, False, expected_err_re
            )

    def _test_basic(self, dtype):
        value = np.arange(200, dtype=np.float32).astype(dtype)
        np.random.shuffle(value)

        # Check that argmin and argmax match numpy along the primary axis
        self._test_both_arg(math_ops.argmax, value, 0, value.argmax())
        self._test_both_arg(math_ops.argmin, value, 0, value.argmin())

    def _test_tie_breaking(self, dtype):
        value = np.zeros(200, dtype=dtype)

        # Check that argmin and argmax match numpy along the primary axis for
        # breaking ties.
        self._test_both_arg(math_ops.argmax, value, 0, value.argmax())
        self._test_both_arg(math_ops.argmin, value, 0, value.argmin())

        # Check that argmin and argmax match numpy along axis=1 for
        # breaking ties.
        value = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=dtype)
        self._test_both_arg(math_ops.argmax, value, 1, value.argmax(axis=1))
        self._test_both_arg(math_ops.argmin, value, 1, value.argmin(axis=1))

    def _test_dim(self, dtype):
        shape = (3, 2, 4, 5, 6, 3, 7)
        value = np.arange(
            functools.reduce(lambda x, y: x * y, shape), dtype=np.float32
        ).astype(dtype)
        np.random.shuffle(value)
        value = value.reshape(shape)

        # Check that argmin and argmax match numpy along all axes
        for axis in range(-7, 7):
            self._test_both_arg(math_ops.argmax, value, axis, value.argmax(axis))
            self._test_both_arg(math_ops.argmin, value, axis, value.argmin(axis))

    def test_float(self):
        """test_float"""
        self._test_basic(np.float32)
        self._test_tie_breaking(np.float32)
        self._test_dim(np.float32)

    def test_float_int_32_output(self):
        """test_float_int_32_output"""
        value = np.asarray(100 * np.random.randn(200), dtype=np.float32)
        expected_values = value.argmax()
        with self.session():
            ans = math_ops.argmax(value, axis=0, output_type=dtypes.int32)
            tf_ans = self.evaluate(ans)
            self.assertEqual(np.int32, tf_ans.dtype)
            # The values are equal when comparing int32 to int64 because
            # the values don't have a range that exceeds 32-bit integers.
            self.assertAllEqual(tf_ans, expected_values)
        expected_values = value.argmin()
        with self.session():
            ans = math_ops.argmin(value, axis=0, output_type=dtypes.int32)
            tf_ans = self.evaluate(ans)
            self.assertEqual(np.int32, tf_ans.dtype)
            self.assertAllEqual(tf_ans, expected_values)

    def test_double(self):
        """test_double"""
        self._test_basic(np.float64)
        self._test_tie_breaking(np.float64)
        self._test_dim(np.float64)

    def test_int32(self):
        """test_int32"""
        self._test_basic(np.int32)
        self._test_tie_breaking(np.int32)
        self._test_dim(np.int32)

    def test_int_64(self):
        """test_int_64"""
        self._test_basic(np.int64)
        self._test_tie_breaking(np.int64)
        self._test_dim(np.int64)

    def test_bool(self):
        """test_bool"""
        self._test_basic(np.bool_)
        self._test_tie_breaking(np.bool_)
        self._test_dim(np.bool_)

    def test_empty(self):
        """test_empty"""
        with self.cached_session():
            for operator in math_ops.argmin, math_ops.argmax:
                with self.assertRaisesOpError(
                    r"Reduction axis 0 is empty in shape \[0\]"
                ):
                    operator([], 0).eval()

    @test_util.run_deprecated_v1
    def test_default_axis(self):
        """test_default_axis"""
        with self.cached_session():
            for operator in math_ops.argmin, math_ops.argmax:
                ans = operator([1]).eval()
                self.assertAllEqual(ans, 0)

    @test_util.run_deprecated_v1
    def test_output_empty(self):
        """test_output_empty"""
        with self.cached_session():
            for operator in math_ops.argmin, math_ops.argmax:
                ret = operator(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
                self.assertEqual(ret.shape, (1, 0))


if __name__ == "__main__":
    test.main()
