import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

x1 = tf.constant(5)
x2 = tf.constant(6)

# 这行代码执行完后，并没有真正的计算，只是生成了一个新的tensor
result = tf.multiply(x1, x2)
print(result)

# run session的时候才开始计算
# sess = tf.Session()
# print(sess.run(result))

with tf.Session() as sess:
    output = sess.run(result)

print(output)