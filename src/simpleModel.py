#!/usr/bin/env python
# coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

model = L.Linear(1,1)
optimizer = optimizers.SGD()
optimizer.setup(model)

times = 50

x = Variable(np.array([[1],[4],[7]], dtype=np.float32))

t = Variable(np.array([[3],[12],[21]], dtype=np.float32))

for i in range(0, times):
    optimizer.zero_grads()

    y = model(x)

    print(y.data)

    loss = F.mean_squared_error(y, t)

    loss.backward()

    optimizer.update()


print("result")
x = Variable(np.array([[3],[4],[5]], dtype=np.float32))
y = model(x)
print(y.data)
