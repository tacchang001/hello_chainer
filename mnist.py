# http://taichitary.hatenablog.com/entry/2017/02/16/215654

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import sys

plt.style.use("ggplot")

batchsize = 100
n_epoch = 20
n_units = 1000  # 中間層
pixel_size = 28
xp = cuda.cupy


# Chainerのクラス作成
class MNISTChain(Chain):
    def __init__(self):
        super(MNISTChain, self).__init__(
            l1=F.Linear(784, n_units),
            l2=F.Linear(n_units, n_units),
            l3=F.Linear(n_units, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(cuda.to_gpu(x_data)), Variable(cuda.to_gpu(y_data))
        h1 = F.dropout(F.relu(self.l1(x)), train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        y = self.l3(h2)

        # 交差エントロピー関数を誤差関数とする
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# MNISTの画像データDL
print("fetch MNIST dataset")
mnist = fetch_mldata('MNIST original', data_home=".")
# mnist.data : 70,000件の28x28=784次元ベクトルデータ
mnist.data = mnist.data.astype(xp.float32)
mnist.data /= 255  # 正規化

# mnist.target : 正解データ
mnist.target = mnist.target.astype(xp.int32)

# 学習用データN個，検証用データを残りの個数に設定
N = 60000
x_train, x_test = xp.split(mnist.data, [N])
y_train, y_test = xp.split(mnist.target, [N])
N_test = y_test.size

# modelを書く
model = MNISTChain()
# GPUの設定
cuda.get_device(0).use()
model.to_gpu()
# optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# train and show results
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# Learning loop
for epoch in range(1, n_epoch + 1):
    print("epoch", epoch)

    # training
    # Nこの順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    # 0~Nまでのデータをバッチサイズごとに使って学習
    for i in range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss, acc = model.forward(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer.update()

        train_loss.append(cuda.to_cpu(loss.data))
        train_acc.append(cuda.to_cpu(acc.data))
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # 訓練データの誤差と正解精度を表示
    print("train mean loss = {0}, accuracy = {1}".format(sum_loss / N, sum_accuracy / N))

    # evaluation
    # テストデータで誤差と正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]

        # 順伝播させて誤差と精度を算出
        loss, acc = model.forward(x_batch, y_batch, train=False)

        test_loss.append(cuda.to_cpu(loss.data))
        test_acc.append(cuda.to_cpu(acc.data))
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # テストデータの誤差と正解精度を表示
    print("test  mean loss = {0}, accuracy = {1}".format(sum_loss / N_test, sum_accuracy / N_test))

# 精度と誤差をグラフ描画
plt.figure(figsize=(8, 6))
print(train_acc)
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.legend(["train_acc", "test_acc"], loc=4)
plt.title("Accuracy of MNIST recognition.")
plt.plot()
plt.show()
