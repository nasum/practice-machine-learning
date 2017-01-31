import create_array as ca
import sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain

args = sys.argv

train_0, test_0 = chainer.datasets.get_mnist(ndim=3)

train = ca.create_array(args[1],1)
test = ca.create_array(args[2],0)

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            # たたみ込み層
            conv1 = L.Convolution2D(3, 20, 5),
            conv2 = L.Convolution2D(20, 50, 5),
            # 多層パーセプトロン
            fc1 = L.Linear(800, 500),
            fc2 = L.Linear(500, 10),
        )
    def __call__(self, x, train = True):
        #ここでネットワークを作っている
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)), train = train)
        return self.fc2(h)

# パラメータ付き関数 誤差の計算をしてくれる
model = L.Classifier(Model())

optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

batchsize = 56

def cov(batch, batchsize):
    x = []
    t = []
    for j in range(batchsize):
        print(batch[j][0][0][0])
        x.append(batch[j][0])
        t.append(batch[j][1])
    return Variable(np.array(x)), Variable(np.array(t))

for n in range(20):
    for i in chainer.iterators.SerialIterator(train, batchsize, repeat = False):
        x, t = cov(i, batchsize)

        # 勾配を０に
        model.zerograds()

        # 損失を計算
        loss = model(x, t)

        # 逆伝搬
        loss.backward()

        # 最適化関数の更新
        optimizer.update()

    i = chainer.iterators.SerialIterator(test, batchsize, repeat = False).next()
    x,t = cov(i, batchsize)
    loss = model(x, t)
    print(n, loss.data)
