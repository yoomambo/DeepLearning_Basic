import numpy as np
import sys
import os
sys.path.append(os.pardir)

def numerical_gradient(func, x):
    """
    x : numpy.array()
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

def numerical_gradient_2d(f, x):
    if X.ndim == 1:
        return numerical_gradient(f, x)
    else:
        # make zero array similar size to X
        grad = np.zeros_like(x)
        
        for idx, value in enumerate(x):
            grad[idx] = numerical_gradient(f, value)
        
        return grad
"""
[넘파이 행렬 iterator]

- 행렬 원소 접근에 관련된 내용입니다.

- 명시적 인덱싱, 슬라이싱 이외에, 행렬 모든 원소에 접근할 경우 이터레이터를 사용 가능합니다.

- 넘파이 iterator는, 다른 언어들의 이터레이터와 마찬가지로, next()메서드를 통해 데이터 값을 처음부터 끝까지 순차적으로 읽어들이는 방법을 제공합니다.
"""
def new_numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #이터레이터가 finished 위치가 아닐동안 반복
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        
        # f(x+h)
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 

        # f(x-h)
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        # 값 복원
        x[idx] = tmp_val
        #이터레이터를 다음 위치로 넘기기
        it.iternext()   
        
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):

    for i in range(step_num):
        grad = new_numerical_gradient(f, init_x)
        init_x -= (lr * grad)
    
    return x