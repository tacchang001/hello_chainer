import matplotlib.pyplot as plt
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers, Variable
from sklearn.preprocessing import StandardScaler

import time


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(2, 1),
        )

    def __call__(self, x):
        return F.sigmoid(self.l1(x))


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

train_xy = train_data[:, [0, 1]].astype(np.float32)
train_t = train_data[:, 2].reshape(-1, 1).astype(np.float32)

#
# データを標準化(平均 0、標準偏差 1)
#
sc = StandardScaler()
train_xy = sc.fit_transform(train_xy)

#
# モデル生成
#
model = Model()
optimizer = optimizers.Adam()
optimizer.setup(model)

time_sta = time.perf_counter()

#
# 学習
#
for i in range(10000):
    optimizer.reallocate_cleared_grads()
    p = model(Variable(train_xy))
    loss = F.mean_squared_error(p, Variable(train_t))
    loss.backward()  # 誤差逆伝播
    optimizer.update()
    if i % 500 == 0:
        acc = ((p.data > 0.5) == (train_t > 0.5)).sum() / len(p)
        print(loss.data, acc)

time_end = time.perf_counter()
tim = time_end - time_sta
print("time: {}".format(tim))

#
# 学習結果の重みとバイアスを取りだす
#
w_1, w_2 = model.l1.W.data[0]
bias = model.l1.b.data[0]

plt.title('Chainer (loop:{0})'.format(i + 1))
plt.plot([min(train_xy[:, 0]), max(train_xy[:, 0])],
         list(map(lambda x: (-w_1 * x - bias) / w_2, [min(train_xy[:, 0]), max(train_xy[:, 0])])))
plt.scatter(train_xy[train_t.ravel() == 0, 0], train_xy[train_t.ravel() == 0, 1], c='red', marker='x', s=30,
            label='train 0')
plt.scatter(train_xy[train_t.ravel() == 1, 0], train_xy[train_t.ravel() == 1, 1], c='blue', marker='x', s=30,
            label='train 1')
plt.legend(loc='upper left')
plt.show()
