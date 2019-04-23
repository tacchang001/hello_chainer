import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

import time

#
# データ生成
#
np.random.seed(seed=0)
xy_0 = np.random.multivariate_normal([2, 2], [[2, 0], [0, 2]], 50)
t_0 = np.zeros(len(xy_0))

xy_1 = np.random.multivariate_normal([7, 7], [[3, 0], [0, 3]], 50)
t_1 = np.ones(len(xy_1))

xy_all = np.vstack((xy_0, xy_1))
t_all = np.append(t_0, t_1).reshape(-1, 1)
train_data = np.random.permutation(np.hstack((xy_all, t_all)))

train_xy = train_data[:, [0, 1]]
train_t = train_data[:, 2].reshape(-1, 1)

#
# データを標準化(平均 0、標準偏差 1)
#
sc = StandardScaler()
train_xy = sc.fit_transform(train_xy)

#
# TensorFlow の各種定義
#
xy = tf.placeholder(tf.float32, [None, 2])
t = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

f = tf.matmul(xy, w) + b  # 線形変換
p = tf.sigmoid(f)  # シグモイド関数
loss = tf.reduce_mean(tf.square(p - t))  # 損失
# loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))

train_step = tf.train.AdamOptimizer().minimize(loss)

time_sta = time.perf_counter()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#
# 学習
#
for i in range(10000):
    sess.run(train_step, feed_dict={xy: train_xy, t: train_t})
    if i % 500 == 0:
        p_val, loss_val = sess.run([p, loss], feed_dict={xy: train_xy, t: train_t})
        acc = ((p_val > 0.5) == (train_t > 0.5)).sum() / len(p_val)
        print(loss_val, acc)

time_end = time.perf_counter()
tim = time_end - time_sta
print("time: {}".format(tim))

#
# 学習結果の重みとバイアスを取りだす
#
bias = sess.run(b)[0]
w_1, w_2 = sess.run(w)[:, 0]

plt.title('TensorFlow (loop:{0})'.format(i + 1))
plt.plot([min(train_xy[:, 0]), max(train_xy[:, 0])],
         list(map(lambda x: (-w_1 * x - bias) / w_2, [min(train_xy[:, 0]), max(train_xy[:, 0])])), c='green')
plt.scatter(train_xy[train_t.ravel() == 0, 0], train_xy[train_t.ravel() == 0, 1], c='red', marker='x', s=30,
            label='class 0')
plt.scatter(train_xy[train_t.ravel() == 1, 0], train_xy[train_t.ravel() == 1, 1], c='blue', marker='x', s=30,
            label='class 1')
plt.legend(loc='upper left')
plt.show()