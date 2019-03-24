import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

style.use('ggplot')

x_data = np.float32(np.random.rand(2, 100))
print('x_data', x_data)
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    target_w = None
    target_b = None
    for step in range(0, 21):
        sess.run(train)
        target_w = sess.run(W)[0]
        target_b = sess.run(b)[0]

    print('W:', target_w, 'b:', target_b)
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x_data[0], ys=x_data[1], zs=y_data)
    
    # 曲面TODO
    # X, Y = np.meshgrid(x_data[0], x_data[1])
    # # y_data = np.dot(target_w, x_data) + target_b
    # y_data = target_w[0]*X + target_w[1]*Y + target_b
    # print(y_data)

    # ax.plot_surface(X=X, Y=Y, Z=y_data.reshape(X.shape))
    # plt.show()