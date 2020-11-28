# Optimizer 구현
import numpy as np
import sys
import os
sys.path.append(os.pardir)


class SGD(object):
    """
    Stochastic Gradient Descent

    Purpose
    ------
    기존방식은 update를 데이터 하나씩 업데이트한다.
    """

    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        """
        grads : backpropagation에 필요한 국소미분값
        """
        for key in params.keys():
            grads -= self.lr * grads[key]

class Momentum:
    """
    Define
    --------
    v <- αv - η(L/W)
    W <- W + v

    Purpose
    --------
    운동량을 가지는 GD
    """

    def __init__(self, lr=0.01, momentum_alpha=0.9):
        self.lr = lr
        self.momentum_alpha = momentum_alpha
        self.v = None

    def update(self, params, grads):
        if self.v = None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        for key, val in params.keys():
            self.v[key] = self.momentum_alpha * self.v[key] - self.lr*grads[key]
            self.params += self.v[key]

class Adagrad:
    """
    Define
    --------
    h <- h + (L/W) * (L/W)
    W <- W - η/h * (L/W)

    Purpose
    --------
    (L/W)가 커지면 η/h을 작게 만들어서 lr을 작게 줄이는 역할
    (L/W)가 작아지면 η/h을 크게 만들어서 lr을 크게 늘리는 역할
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h = None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key, val in params.keys():
            self.h[key] += self.h[key] * self.h[key]
            self.params -= self.lr/self.h[key] * grads[key]
