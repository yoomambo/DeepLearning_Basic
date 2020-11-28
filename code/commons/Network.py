# coding: utf-8
from commons.gradient import numerical_gradient
from commons.layer import *
from collections import OrderedDict
import numpy as np
import sys
import os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


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


class MultiLayerNet:
    """완전연결 다층 신경망
    Parameters
    ----------
    input_size :（MNIST : 784）

    hidden_size_list : hidden layer의 node 수 [20, 30, 40, 50...] cf) keras.layer와 비슷하게 생각하자

    output_size : （MNIST : 10）

    activation : 'relu' 혹은 'sigmoid'

    weight_init_std : weight의 std 지정,
        'None' : 0.01
        'relu' or 'he' : sqrt(2/n) 으로 설정, n은 앞층의 node 수
        'sigmoid' or 'xavier' : sqrt(1/n) 으로 설정

    weight_decay_lambda : regularization value
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation, weight_decay_lambda=0):

        self.params = {}

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.activation = activation
        self.weight_decay_lambda = weight_decay_lambda

        self.node_list = [input_size] + hidden_size_list + [output_size]

        self.__init_weight()
        self.layers = OrderedDict()

        for i in range(hidden_size_list):
            self.layers['Affine' + str(i+1)] = Affine(
                self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            elif self.activation.lower() == 'relu':
                self.layers[self.activation + str(i+1)] = ReLU(
                    self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            elif self.activation.lower() == 'sigmoid':
                self.layers[self.activation + str(i+1)] = Sigmoid(
                    self.params['W'+str(i+1)], self.params['b'+str(i+1)])

        self.layers['Affine' + str(len(hidden_size_list))] = Affine(self.params['W'+str(
            len(hidden_size_list)+1)], self.params['b'+str(len(hidden_size_list)+1)])
        self.layers['SoftmaxWithLoss'] = SoftWithLoss()

    def __init_weight(self):
        """
        initialize weight
        """
        for idx in range(len(self.node_list)):
            if idx == len(node_list)-1:
                break
            elif activation = None:
                scale = 100
            elif activation.lower in ('sigmoid', 'xavier'):
                scale = (np.sqrt(1/node_list[idx]))
            elif activation.lower in ('relu', 'he'):
                scale = (np.sqrt(2/node_list[idx]))

            self.params['W'+str(idx+1)] = scale * \
                np.random.randn(node_list[idx], node_list[idx+1])
            self.params['b' + str(idx+1)] = np.zeros(node_list[idx])

    def predict(self, x):
        """
        predict value
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        predict loss value
        """
        yhat = self.predict(x)
        # return regularization L2
        weight_decay = 0
        for idx in range(1, len(self.node_list)):
            W = self.params['W' + str(idx)]
            weight_decay += weight_decay_lambda * 0.5 * (np.sum(W**2))

        return self.layers['SoftmaxWithLoss'].forward(x, t) + weight_decay

    def accuracy(self, x, t):
        """
        caculate accuracy

        >>> A = np.array([[1,2,3,4],[5,6,7,8]])
        >>> np.argmax(A)         
        7
        >>> np.argmax(A, axis=1) 
        array([3, 3], dtype=int64)
        """
        yhat = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(yhat == t)/yhat.shape[0]
        
        return accuracy

    def numerical_gradient(self, x, t):
        """numerical gradient using derivative.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
            grads['W1']、grads['W2'] (weight each layer)
            grads['b1']、grads['b2'] (bias each layer)
        """
        def loss_W(W): return self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """Backpropagation.

        : Backpropagation을 좀 더 선호한다.
            1. 효율적인 계산
            2. 국소적인 부분을 알 수 있음.

        Parameters
        ----------
        x : input data
        t : label

        Returns
        -------
            grads['W1']、grads['W2'] (weight each layer)
            grads['b1']、grads['b2'] (bias each layer)
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + \
                self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
