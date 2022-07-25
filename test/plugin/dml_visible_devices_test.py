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

"""Tests the DML device creation and visibility"""

import os
import tensorflow as tf
from absl.testing import absltest
from absl import flags

flags.DEFINE_string(
    "dml_visible_devices", "", "Value of DML_VISIBLE_DEVICES environment variable"
)


class VisibleDevicesTest(absltest.TestCase):
    """Tests the visibility of DML devices"""

    def test(self):
        """Tests the visibility of DML devices"""
        os.environ["DML_VISIBLE_DEVICES"] = flags.FLAGS.dml_visible_devices

        # See https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq
        # The value should be a comma-separated list of device IDs.
        # Any IDs appearing after -1 are invalid.
        valid_id_count = 0
        for device_id in flags.FLAGS.dml_visible_devices.split(","):
            if device_id == "-1":
                break
            valid_id_count += 1

        gpu_devices = tf.config.list_physical_devices("GPU")
        dml_devices = list(
            filter(
                lambda x: tf.config.experimental.get_device_details(x)["device_name"]
                == "DML",
                gpu_devices,
            )
        )

        # We can't guarantee the machine running this test has multiple
        # devices/adapters, but it must have at least one.
        if valid_id_count == 0:
            self.assertEmpty(dml_devices)
        else:
            self.assertBetween(len(dml_devices), 1, valid_id_count)


if __name__ == "__main__":
    absltest.main()
