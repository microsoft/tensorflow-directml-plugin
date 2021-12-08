import time
import tensorflow as tf

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

tf.profiler.experimental.start(logdir="profiler_logs", options=profiler_options)
for i in range(1,3):
    with tf.profiler.experimental.Trace("Pre-Processing Event"):
        time.sleep(0.25)
        pass

tf.config.experimental
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as s:
    print(s.run(y2, feed_dict={a:a_values, b:b_values, c:c_values}))

for i in range(1,3):
    with tf.profiler.experimental.Trace("Post-Processing Event"):
        time.sleep(0.25)
        pass
tf.profiler.experimental.stop()