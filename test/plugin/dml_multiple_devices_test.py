#!/usr/bin/env python
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

"""Tests computation and copies across multiple DML devices"""

from absl.testing import absltest
import tensorflow as tf


class VisibleDevicesTest(absltest.TestCase):
    """Tests computation and copies across multiple DML devices"""

    def test(self):
        """Tests computation and copies across multiple DML devices"""

        gpu_devices = tf.config.list_physical_devices("GPU")
        # pylint: disable=duplicate-code
        dml_devices = list(
            filter(
                lambda x: tf.config.experimental.get_device_details(x)["device_name"]
                == "DML",
                gpu_devices,
            )
        )
        # pylint: enable=duplicate-code

        if len(dml_devices) < 2:
            self.skipTest(
                f"This test requires more than 1 DirectML GPU, but only "
                f"{len(dml_devices)} devices were found."
            )

        with tf.device("GPU:0"):
            device1_tensor = tf.constant(1, dtype=tf.float32)

        with tf.device("GPU:1"):
            device2_tensor = tf.constant(2, dtype=tf.float32)

        actual = tf.math.add(device1_tensor, device2_tensor)
        expected = tf.constant(3, dtype=tf.float32)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    absltest.main()
