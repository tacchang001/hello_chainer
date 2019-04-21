# https://aripy.net/python/stockscience/chainer-01/
import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, datasets, optimizers
from chainer import report, training
from chainer.training import extensions
import chainer.cuda

start = dt.date(2016, 1, 1)
end = dt.date(2017, 9, 20)
df = web.DataReader('AMZN', "yahoo", start, end)

print(df.head(5))

plt.plot(df['Close'])
plt.show()

# 株価の標準化

scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'])

# 訓練データと教師データへの分割

x, t = [], []

N = len(df)
M = 25
for n in range(M, N):
    _x = df['Close'][n - M:n]
    _t = df['Close'][n]
    x.append(_x)
    t.append(_t)

x = np.array(x, dtype=np.float32)
t = np.array(t, dtype=np.float32).reshape(len(t), 1)

# 訓練：60%, 検証：40%で分割する
n_train = int(len(x) * 0.6)
dataset = list(zip(x, t))
train, test = chainer.datasets.split_dataset(dataset, n_train)


# ニューラルネットワークモデルを作成
class RNN(Chain):
    def __init__(self, n_units, n_output):
        super().__init__()
        with self.init_scope():
            self.l1 = L.LSTM(None, n_units)
            self.l2 = L.Linear(None, n_output)

    def reset_state(self):
        self.l1.reset_state()

    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss

    def predict(self, x):
        if train:
            h1 = F.dropout(self.l1(x), ratio=0.5)
        else:
            h1 = self.l1(x)
        return self.l2(h1)


# LSTMUpdaterを作る。
class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=None)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        batch = data_iter.__next__()
        x_batch, t_batch = chainer.dataset.concat_examples(batch, self.device)

        optimizer.target.reset_state()  # 追加
        optimizer.target.cleargrads()
        loss = optimizer.target(x_batch, t_batch)
        loss.backward()
        loss.unchain_backward()  # 追記
        optimizer.update()


# 乱数のシードを固定 (再現性の確保)
np.random.seed(1)

# モデルの宣言
model = RNN(30, 1)

# GPU対応
# chainer.cuda.get_device(0).use()
# model.to_gpu()

# Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Iterator
batchsize = 20
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# Updater &lt;- LSTM用にカスタマイズ
updater = LSTMUpdater(train_iter, optimizer, device=-1)

# Trainerとそのextensions
epoch = 3000
trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

# 評価データで評価
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

# 学習結果の途中を表示する
trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

# １エポックごとに、trainデータに対するlossと、testデータに対するlossを出力させる
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']),
               trigger=(1, 'epoch'))

trainer.run()
