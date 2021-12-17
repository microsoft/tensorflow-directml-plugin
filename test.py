import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3,1])
a_values = [[1],[2],[3]]

b = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3,1])
b_values = [[4],[6],[11]]

c = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1,3])
c_values = [[2,4,5]]

with tf.device("/device:DML:0"):
    y1 = tf.raw_ops.AddN(inputs=[a,b], name="MyAdd")
with tf.device("/device:DML:0"):
    y2 = tf.raw_ops.MatMul(a=y1, b=c, name="MyMultiply")

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as s:
    print(s.run(y2, feed_dict={a:a_values, b:b_values, c:c_values}))