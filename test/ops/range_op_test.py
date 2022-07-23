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
"""Tests for tensorflow.raw_ops.Range."""

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class RangeTest(test.TestCase):
    @test_util.run_in_graph_and_eager_modes
    def testRange(self):
        with self.cached_session():
            for dtype in [tf.float32, tf.int64]:
                start = tf.constant(1, dtype=dtype)
                limit = tf.constant(20, dtype=dtype)
                delta = tf.constant(2, dtype=dtype)
                result = tf.raw_ops.Range(start=start, limit=limit, delta=delta)
                self.assertAllEqual([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], result)

    @test_util.run_in_graph_and_eager_modes
    def testRangeNegative(self):
        with self.cached_session():
            for dtype in [tf.float32, tf.int64]:
                start = tf.constant(19, dtype=dtype)
                limit = tf.constant(0, dtype=dtype)
                delta = tf.constant(-2, dtype=dtype)
                result = tf.raw_ops.Range(start=start, limit=limit, delta=delta)
                self.assertAllEqual([19, 17, 15, 13, 11, 9, 7, 5, 3, 1], result)

    def testRangeBadStartShape(self):
        with self.cached_session():
            with self.assertRaisesOpError(r"start must be a scalar, not shape \[1,1\]"):
                start = tf.constant([[1]], dtype=tf.float32)
                limit = tf.constant(20, dtype=tf.float32)
                delta = tf.constant(2, dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)

    def testRangeBadLimitShape(self):
        with self.cached_session():
            with self.assertRaisesOpError(r"limit must be a scalar, not shape \[1,1\]"):
                start = tf.constant(1, dtype=tf.float32)
                limit = tf.constant([[20]], dtype=tf.float32)
                delta = tf.constant(2, dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)

    def testRangeBadDeltaShape(self):
        with self.cached_session():
            with self.assertRaisesOpError(r"delta must be a scalar, not shape \[1,1\]"):
                start = tf.constant(1, dtype=tf.float32)
                limit = tf.constant(20, dtype=tf.float32)
                delta = tf.constant([[2]], dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)

    def testRangeZeroDelta(self):
        with self.cached_session():
            with self.assertRaisesOpError(r"Requires delta != 0: 0"):
                start = tf.constant(1, dtype=tf.float32)
                limit = tf.constant(20, dtype=tf.float32)
                delta = tf.constant(0, dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)

    def testRangePositiveDeltaWrongStart(self):
        with self.cached_session():
            with self.assertRaisesOpError(
                r"Requires start <= limit when delta > 0: 20/1"
            ):
                start = tf.constant(20, dtype=tf.float32)
                limit = tf.constant(1, dtype=tf.float32)
                delta = tf.constant(2, dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)

    def testRangeNegativeDeltaWrongLimit(self):
        with self.cached_session():
            with self.assertRaisesOpError(
                r"Requires start >= limit when delta < 0: 1/20"
            ):
                start = tf.constant(1, dtype=tf.float32)
                limit = tf.constant(20, dtype=tf.float32)
                delta = tf.constant(-2, dtype=tf.float32)
                tf.raw_ops.Range(start=start, limit=limit, delta=delta)


if __name__ == "__main__":
    test.main()
