# activation function 구현
import numpy as np
import sys
import os
sys.path.append(os.pardir)


def function_2(x):
    """
    docstring
    """
    return x[0]**2 + x[1]**2


def numerical_gradient(func, x):
    """
    function_2에 대한 gradient calculation
    """
    grad = np.zeros_like(x)
    h = 1e-4

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = func(x)

        # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val

    return grad


class ActivationFunction:
    # def step_function(self, x):
    #     if x > 0:
    #         return 1
    #     else:
    #         return 0

    def step_function(self, x):
        y = x > 0
        return np.astype(np.int)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # def relu(self, x):
    #     if x > 0:
    #         return x
    #     else:
    #         return 0

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


class LossFunction(object):

    # def __init__(self, x):

    def SSE(self, output, correct_label):
        """
        Sum of Square for Error
        """
        return np.sum((output-correct_label)**2)/2

    def CEE(self, output, correct_label):
        """
        Cross Entropy for Error
        """
        # prevent overflow
        delta = 1e-7
        return -np.sum(correct_label*np.log(output + delta))


if __name__ == "__main__":
    result = numerical_gradient(function_2, np.array([1.0, 2.0]))
    print(result)
