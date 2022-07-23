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

import absl.testing.absltest as absltest
import tensorflow as tf
import tensorflow.core.profiler.protobuf.xplane_pb2 as xplane_pb2
import tensorboard_plugin_profile.protobuf.tf_stats_pb2 as tf_stats_pb2
import tempfile
import shutil
from pathlib import Path
import gzip
import json


class ProfilerTest(absltest.TestCase):

    # Executes a trivial graph. The profiler output provides the input data for test methods below.
    @classmethod
    def setUpClass(cls):
        tf.compat.v1.disable_eager_execution()

        a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 1])
        a_values = [[1], [2], [3]]

        b = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 1])
        b_values = [[4], [6], [11]]

        c = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 3])
        c_values = [[2, 4, 5]]

        y1 = tf.raw_ops.AddN(inputs=[a, b], name="MyAdd")
        y2 = tf.raw_ops.MatMul(a=y1, b=c, name="MyMultiply")

        profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2, python_tracer_level=0, device_tracer_level=1
        )

        cls.root_log_dir = tempfile.mkdtemp()

        tf.profiler.experimental.start(
            logdir=cls.root_log_dir, options=profiler_options
        )
        with tf.compat.v1.Session() as s:
            print(s.run(y2, feed_dict={a: a_values, b: b_values, c: c_values}))
        tf.profiler.experimental.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root_log_dir)

    def getProfilerLogFilePath(self, glob):
        profiles_dir = Path(self.__class__.root_log_dir) / "plugins" / "profile"
        last_profile_dir = [p for p in profiles_dir.iterdir() if p.is_dir()][-1]
        return list(Path(last_profile_dir).glob(glob))[-1]

    # Checks that xplane.pb contains expected kernel events.
    def testXPlaneKernelEvents(self):
        file_path = self.getProfilerLogFilePath("*xplane.pb")
        xspace = xplane_pb2.XSpace()
        with open(file_path, "rb") as f:
            xspace.ParseFromString(f.read())

        for xplane in xspace.planes:
            if not xplane.name.startswith("/device:"):
                continue

            # DML planes are prefixed /device:GPU so that they get picked up
            # by tensorflow_stats and trace_viewer.
            self.assertRegex(xplane.name, "/device:GPU:\d \(DirectML\) - .*")

            # DML profiler only emits a single line right now (CPU-timeline kernels).
            self.assertEqual(len(xplane.lines), 1)
            self.assertEqual(xplane.lines[0].name, "Kernels (CPU Timeline)")
            cpu_timeline = xplane.lines[0]

            # Ensure the AddN kernel is traced.
            self.assertEqual(len(cpu_timeline.events), 2)
            addn_event = cpu_timeline.events[0]
            addn_metadata = xplane.event_metadata[addn_event.metadata_id]
            self.assertEqual(addn_metadata.name, "MyAdd:AddN")
            self.assertEqual(addn_metadata.display_name, "AddN")
            self.assertGreater(len(addn_event.stats), 0)
            self.assertEqual(
                xplane.stat_metadata[addn_event.stats[0].metadata_id].name, "tf_op"
            )

            # Ensure the MatMul kernel is traced.
            matmul_event = cpu_timeline.events[1]
            matmul_metadata = xplane.event_metadata[matmul_event.metadata_id]
            self.assertEqual(matmul_metadata.name, "MyMultiply:MatMul")
            self.assertEqual(matmul_metadata.display_name, "MatMul")
            self.assertGreater(len(matmul_event.stats), 0)
            self.assertEqual(
                xplane.stat_metadata[matmul_event.stats[0].metadata_id].name, "tf_op"
            )

    # Checks that trace.json contains expected device kernel events.
    def testTraceKernelEvents(self):
        file_path = self.getProfilerLogFilePath("*trace.json.gz")
        with gzip.open(file_path) as f:
            json_trace = json.loads(f.read())

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
                        proc_event["args"]["name"], "/device:GPU:\d \(DirectML\) - .*"
                    )
                    dml_kernel_events += 1

        self.assertEqual(dml_kernel_events, 2)

    # Checks that device kernels appear in the tensorflow_stats.pb
    def testTensorFlowStats(self):
        file_path = self.getProfilerLogFilePath("*tensorflow_stats.pb")
        database = tf_stats_pb2.TfStatsDatabase()
        with open(file_path, "rb") as f:
            database.ParseFromString(f.read())

        # Even though we're using DML devices we should see "GPU" here
        # because the XPlanes are prefixed '/device:GPU:# (DirectML) ...'
        self.assertEqual(database.device_type, "GPU")

        records = database.without_idle.tf_stats_record
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
