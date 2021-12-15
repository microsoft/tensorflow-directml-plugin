import absl.testing.absltest as absltest
import tensorflow as tf
import tensorboard_plugin_profile.protobuf.tf_stats_pb2 as tf_stats
import tempfile
import shutil
from pathlib import Path

class ProfilerTest(absltest.TestCase):

    # Executes a trivial graph. The profiler output provides the input data for test methods below.
    @classmethod
    def setUpClass(cls):
        tf.compat.v1.disable_eager_execution()

        a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3,1])
        a_values = [[1],[2],[3]]

        b = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3,1])
        b_values = [[4],[6],[11]]

        c = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1,3])
        c_values = [[2,4,5]]

        y1 = tf.raw_ops.AddN(inputs=[a,b], name="MyAdd")
        y2 = tf.raw_ops.MatMul(a=y1, b=c, name="MyMultiply")

        profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2,
            python_tracer_level=0,
            device_tracer_level=1
        )

        cls.root_log_dir = tempfile.mkdtemp()

        tf.profiler.experimental.start(logdir=cls.root_log_dir, options=profiler_options)
        with tf.compat.v1.Session() as s:
            print(s.run(y2, feed_dict={a:a_values, b:b_values, c:c_values}))
        tf.profiler.experimental.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root_log_dir)

    def getProfilerLogFilePath(self, glob):
        profiles_dir = Path(self.__class__.root_log_dir) / "plugins" / "profile"
        last_profile_dir = [p for p in profiles_dir.iterdir() if p.is_dir()][-1]
        return list(Path(last_profile_dir).glob(glob))[-1]

    def testXPlane(self):
        # copy xspace.proto and deserialize
        print("testing kernel events")
        pass

    def testTraceJson(self):
        # copy xspace.proto and deserialize
        print("testing kernel events")
        pass

    def testTensorFlowStats(self):
        file_path = self.getProfilerLogFilePath("*tensorflow_stats.pb")
        database = tf_stats.TfStatsDatabase()
        with open(file_path, "rb") as f:
            database.ParseFromString(f.read())

        # Even though we're using DML devices we should see "GPU" here 
        # because the XPlanes are prefixed '/device:GPU:# (DirectML) ...'
        self.assertEqual(database.device_type, "GPU")

        records = database.without_idle.tf_stats_record
        self.assertEqual(len(records), 3)

        self.assertEqual(records[0].host_or_device, "Device")
        self.assertEqual(records[0].op_type, "AddN")
        self.assertEqual(records[0].op_name, "MyAdd")

        self.assertEqual(records[1].host_or_device, "Device")
        self.assertEqual(records[1].op_type, "MatMul")
        self.assertEqual(records[1].op_name, "MyMultiply")

        self.assertEqual(records[2].host_or_device, "Host")
        self.assertEqual(records[2].op_type, "_Recv")
        self.assertEqual(records[2].op_name, "MyMultiply/_7")


if __name__ == "__main__":
    absltest.main()