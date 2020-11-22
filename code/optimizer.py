# Optimizer 구현

import numpy as np

class SGD(object):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
            