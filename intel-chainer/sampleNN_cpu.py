# 元コード
# https://to-kei.net/python/machine-learning/chainer-neural-network/

import pandas as pd
import numpy as np

import chainer
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L

from chainer.training import extensions
from chainer.datasets import tuple_dataset

# データの読み込み
df = pd.read_csv("sampleNN.csv")

# 学習したモデルを出力するファイル名
resultFn = "sampleNN.model"

# 入力層のノードの数
inputNum = len(df.columns) - 1

# データの行数を取得
N = len(df)

# データの正規化、各列をその列の最大値で割ることで全て0~1の間にする
df.iloc[:, :-1] /= df.iloc[:, :-1].max()

# 学習に関する基本情報の定義
epoch = 400  # 学習回数
batch = 1  # バッチサイズ
hiddens = [inputNum, 800, 400, len(df.iloc[:, inputNum].unique())]  # 各層のノード数

# 学習、検証データの割合(単位：割)
trainSt = 0  # 学習用データの開始位置 0割目から〜
trainPro = 8  # 学習用データの終了位置　8割目まで
testPro = 10  # 検証用データの終了位置 8割目から10割目まで


# ニューラルネットワークの構築。
class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(hiddens[0], hiddens[1]),
            l2=L.Linear(hiddens[1], hiddens[2]),
            l3=L.Linear(hiddens[2], hiddens[3]),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        o = self.l3(h2)
        return o


def learning():
    # 学習用データと検証用データに分ける
    train_df = df.iloc[0:int(N * trainPro / 10), :]
    test_df = df.iloc[int(N * trainPro / 10):int(N * testPro / 10), :]

    # データの目的変数を落としてnumpy配列にする。
    train_data = np.array(train_df.iloc[:, :-1].astype(np.float32))
    test_data = np.array(test_df.iloc[:, :-1].astype(np.float32))

    # 目的変数もnumpy配列にする。
    train_target = np.array(train_df.iloc[:, inputNum]).astype(np.int32)
    test_target = np.array(test_df.iloc[:, inputNum]).astype(np.int32)

    # ランダムにデータを抽出してバッチ学習する設定
    train = tuple_dataset.TupleDataset(train_data, train_target)
    test = tuple_dataset.TupleDataset(test_data, test_target)
    train_iter = iterators.SerialIterator(train, batch_size=batch, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=batch, repeat=False, shuffle=False)

    # モデルを使う準備。オブジェクトを生成
    model = L.Classifier(MyChain())

    # 最適化手法の設定。今回はAdamを使ったが他にAdaGradやSGDなどがある。
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 学習データの割り当てを行う
    updater = training.StandardUpdater(train_iter, optimizer)

    # 学習回数を設定してtrainerの構築
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    # trainerの拡張をしておく
    trainer.extend(extensions.Evaluator(test_iter, model))  # 精度の確認
    # trainer.extend(extensions.LogReport())  # レポートを残す
    # trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))  # レポートの内容
    # trainer.extend(extensions.ProgressBar())  # プログレスバーの表示
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))  # モデルの保存

    # 学習の実行
    trainer.run()

    # モデルの保存
    serializers.save_npz(resultFn, model)


if __name__ == "__main__":
    learning()
    print("write to " + str(resultFn))
