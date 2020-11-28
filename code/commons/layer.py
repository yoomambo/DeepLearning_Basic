# Layer 구현
from collections import OrderedDict
import numpy as np
from activation import ActivationFunction
import sys
import os
sys.path.append(os.pardir)


class ReLU(object):

    def __init__(self):
        """
        self.mask : 0보다 작은 지점을 mask 처리한다.
        """
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        return dout


class Sigmoid(object):

    def __init__(self, x):
        """
        docstring
        """
        self.out = out

    def forward(self, x):
        self.out = ActivationFunction.sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout*(1-self.out)*self.out
        return dx


class Affine(object):

    def __init__(self, W, b):
        """
        initialize params
        """
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        docstring
        """
        self.x = x
        result = np.dot(x, self.W) + self.b

    def backward(self, x, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.W.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftWithLoss(object):
    def __init__(self):

        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = ActivationFunction.softmax(x)
        self.loss = ActivationFunction.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
