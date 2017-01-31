import create_array as ca
import sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain, training
from chainer.training import extensions

args = sys.argv

train_0, test_0 = chainer.datasets.get_mnist(ndim=3)

train = ca.create_array(args[1],1)
test = ca.create_array(args[2],0)

print(train_0.__class__)
print(train.__class__)

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

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'))

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.ProgressBar())

trainer.run()
