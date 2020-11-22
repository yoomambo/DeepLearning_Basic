# Layer 구현
from collections import OrderedDict
import numpy as np
from activation import numerical_gradient, LossFunction, ActivationFunction
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


class SimpleNet(object):

    def __init__(self):
        """
        initialize weight
        """
        self.W = np.random.randn(2, 3)
        print(self.W)
        self.activation_object = ActivationFunction()
        self.loss_object = LossFunction()

    def predict(self, X):
        return np.dot(X, self.W)

    def loss(self, x, label):
        yhat = self.activation_object.softmax(self.predict(x))
        loss = self.loss_object.CEE(yhat, label)

        return loss


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
    def __init__(self, ):
        """
        docstring
        """
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = ActivationFunction.softmax(x)
        self.loss = LossFunction.CEE(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """
        docstring
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # weight initialize
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftWithLoss()

    def predict(self, ):
        """
        docstring
        """
        for layer in layers.values():
            x = layer.forward(x)

            return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        calculate accuracy
        """
        y = self.predict(x)
        y = np.argrmax(y, axis=1)
        if t.ndim != 1:
            np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy



if __name__ == "__main__":
    net = SimpleNet()
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    def f(W):
        """
        docstring
        """
        return net.loss(x, t)

    dw = numerical_gradient(net.loss(x, t), net.W)
    print(dw)
