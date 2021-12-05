import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2,3])
a_values = [[1,2,3],[4,5,6]]

b = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3,1])
b_values = [[2],[3],[4]]

# 1 2 3 * 2 = 20
# 4 5 6   3   47
#         4
y = tf.raw_ops.MatMul(a=a, b=b)

profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3,
    python_tracer_level=0,
    device_tracer_level=1
)

tf.profiler.experimental.start(logdir="profiler_logs", options=profiler_options)
for i in range(1,3):
    with tf.profiler.experimental.Trace("Pre-Processing Event"):
        time.sleep(0.25)
        pass

with tf.compat.v1.Session() as s:
    print(s.run(y, feed_dict={a:a_values, b:b_values}))

for i in range(1,3):
    with tf.profiler.experimental.Trace("Pre-Processing Event"):
        time.sleep(0.25)
        pass
tf.profiler.experimental.stop()