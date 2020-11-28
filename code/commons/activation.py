# activation function 구현
import numpy as np
import sys
import os
sys.path.append(os.pardir)

class ActivationFunction:

    def identity(self, x):
        return x

    def step(self, x):
        y = x > 0
        return np.astype(np.int)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        # prevent overflow
        c = np.max(x)
        exp_a = np.exp(x - c)
        sum_exp_a = np.sum(exp_a)

        return exp_a/sum_exp_a

    def leaky_relu(self, x):
        """
        leaky_relu
        """
        return np.where(x > 0, test, test/2)

    def sum_square_error(self, output, correct_label):
        """
        divide two value : easy to calculate derivative
        """
        return np.sum((output-correct_label)**2)/2

    def cross_entropy_error(self, output, correct_label):
        """
        delta : prevent overflow
        """
        delta = 1e-7
        return -np.sum(correct_label*np.log(output + delta))
