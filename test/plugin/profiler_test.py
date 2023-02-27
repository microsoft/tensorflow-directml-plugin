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

"""Contains the tests for the DML profiler"""

import tempfile
import shutil
import gzip
import json
from pathlib import Path
from absl.testing import absltest
import tensorflow as tf
from tensorflow.core.profiler.protobuf import xplane_pb2


class ProfilerTest(absltest.TestCase):
    """Contains the tests for the DML profiler"""

    # Executes a trivial graph. The profiler output provides the input data for test
    # methods below.
    @classmethod
    def setUpClass(cls):
        tf.compat.v1.disable_eager_execution()

        a_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 1])
        a_values = [[1], [2], [3]]

        b_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 1])
        b_values = [[4], [6], [11]]

        c_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 3])
        c_values = [[2, 4, 5]]

        add_n = tf.raw_ops.AddN(inputs=[a_placeholder, b_placeholder], name="MyAdd")
        mat_mul = tf.raw_ops.MatMul(a=add_n, b=c_placeholder, name="MyMultiply")

        profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2, python_tracer_level=0, device_tracer_level=1
        )

        cls.root_log_dir = tempfile.mkdtemp()

        tf.profiler.experimental.start(
            logdir=cls.root_log_dir, options=profiler_options
        )
        with tf.compat.v1.Session() as session:
            print(
                session.run(
                    mat_mul,
                    feed_dict={
                        a_placeholder: a_values,
                        b_placeholder: b_values,
                        c_placeholder: c_values,
                    },
                )
            )
        tf.profiler.experimental.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root_log_dir)

    def _get_profiler_log_file_path(self, glob):
        profiles_dir = Path(self.__class__.root_log_dir) / "plugins" / "profile"
        last_profile_dir = [p for p in profiles_dir.iterdir() if p.is_dir()][-1]
        return list(Path(last_profile_dir).glob(glob))[-1]

    def test_x_plane_kernel_events(self):
        """Checks that xplane.pb contains expected kernel events"""
        file_path = self._get_profiler_log_file_path("*xplane.pb")
        xspace = xplane_pb2.XSpace()
        with open(file_path, "rb") as file:
            xspace.ParseFromString(file.read())

        for xplane in xspace.planes:
            if not xplane.name.startswith("/device:"):
                continue

            # DML planes are prefixed /device:GPU so that they get picked up
            # by tensorflow_stats and trace_viewer.
            self.assertRegex(xplane.name, r"/device:GPU:\d \(DirectML\) - .*")
            self.assertEqual(len(xplane.lines), 4)
            self.assertEqual(xplane.lines[0].name, "MemcpyH2D (CPU Timeline)")
            self.assertEqual(xplane.lines[1].name, "MemcpyD2D (CPU Timeline)")
            self.assertEqual(xplane.lines[2].name, "MemcpyD2H (CPU Timeline)")
            self.assertEqual(xplane.lines[3].name, "Kernels (CPU Timeline)")
            kernels_cpu_timeline = xplane.lines[3]

            # Ensure the AddN kernel is traced.
            self.assertEqual(len(kernels_cpu_timeline.events), 2)
            addn_event = kernels_cpu_timeline.events[0]
            addn_metadata = xplane.event_metadata[addn_event.metadata_id]
            self.assertEqual(addn_metadata.name, "MyAdd:AddN")
            self.assertEqual(addn_metadata.display_name, "AddN")
            self.assertGreater(len(addn_event.stats), 0)
            self.assertEqual(
                xplane.stat_metadata[addn_event.stats[0].metadata_id].name, "tf_op"
            )

            # Ensure the MatMul kernel is traced.
            matmul_event = kernels_cpu_timeline.events[1]
            matmul_metadata = xplane.event_metadata[matmul_event.metadata_id]
            self.assertEqual(matmul_metadata.name, "MyMultiply:MatMul")
            self.assertEqual(matmul_metadata.display_name, "MatMul")
            self.assertGreater(len(matmul_event.stats), 0)
            self.assertEqual(
                xplane.stat_metadata[matmul_event.stats[0].metadata_id].name, "tf_op"
            )


if __name__ == "__main__":
    absltest.main()
