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
from tensorboard_plugin_profile.protobuf import tf_stats_pb2


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
            self.assertEqual(len(xplane.lines), 3)
            self.assertEqual(xplane.lines[0].name, "MemcpyH2D (CPU Timeline)")
            self.assertEqual(xplane.lines[1].name, "MemcpyD2H (CPU Timeline)")
            self.assertEqual(xplane.lines[2].name, "Kernels (CPU Timeline)")
            kernels_cpu_timeline = xplane.lines[2]

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

    def test_trace_kernel_events(self):
        """Checks that trace.json contains expected device kernel events"""
        file_path = self._get_profiler_log_file_path("*trace.json.gz")
        with gzip.open(file_path) as file:
            json_trace = json.loads(file.read())

        pid_to_event = {}
        dml_kernel_events = 0

        for trace_event in json_trace["traceEvents"]:
            if not "ph" in trace_event:
                continue

            if trace_event["ph"] == "M":
                if trace_event["name"] == "process_name":
                    pid_to_event[trace_event["pid"]] = trace_event
            elif trace_event["ph"] == "X":
                # The MatMul and AddN events should be on a DirectML device
                if trace_event["name"] == "MatMul" or trace_event["name"] == "AddN":
                    proc_event = pid_to_event[trace_event["pid"]]
                    self.assertRegex(
                        proc_event["args"]["name"], r"/device:GPU:\d \(DirectML\) - .*"
                    )
                    dml_kernel_events += 1

        self.assertEqual(dml_kernel_events, 2)

    def test_tensorflow_stats(self):
        """Checks that device kernels appear in the tensorflow_stats.pb"""
        file_path = self._get_profiler_log_file_path("*tensorflow_stats.pb")
        database = tf_stats_pb2.TfStatsDatabase()
        with open(file_path, "rb") as file:
            database.ParseFromString(file.read())

        # Even though we're using DML devices we should see "GPU" here
        # because the XPlanes are prefixed '/device:GPU:# (DirectML) ...'
        self.assertEqual(database.device_type, "GPU")  # pylint:disable=no-member

        records = database.without_idle.tf_stats_record  # pylint:disable=no-member
        self.assertEqual(len(records), 3)

        # Record ordering might change.
        matched_records = 0
        for record in records:
            if record.op_type == "AddN":
                self.assertEqual(record.host_or_device, "Device")
                self.assertEqual(record.op_name, "MyAdd")
                matched_records += 1
            elif record.op_type == "MatMul":
                self.assertEqual(record.host_or_device, "Device")
                self.assertEqual(record.op_name, "MyMultiply")
                matched_records += 1
            elif record.op_type == "_Recv":
                self.assertEqual(record.host_or_device, "Host")
                self.assertEqual(record.op_name, "MyMultiply/_7")
                matched_records += 1

        self.assertEqual(matched_records, 3)


if __name__ == "__main__":
    absltest.main()
