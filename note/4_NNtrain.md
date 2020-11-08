# 신경망 학습 (Neural Network Training)

## 학습
    trained-data를 이용하여 weight들의 최적값을 획득하게 되는 행위를 말한다.

즉, _**기계학습, Machine Learning은 data를 활용해서 data들의 feature들을 추출, feature들의 pattern을 분석, 분석된 pattern을 학습하는 행위를 말한다.**_

<img src=../image/ML.png>

#### 그렇다면 Deep Learning은 Machine Learning과 무슨 차이점인가?

신경망은 2번 과정처럼 사람의 concept이 개입되지 않은 상태로 data를 있는 그대로 학습하는 것이다.
이러한 학습 방법을 end to end machine learning 이라고 한다.

> Deep Learning을 end to end machine learning 이라고 합니다.
> 데이터에서 목표한 결과를 사람의 개입 없이 얻는다는 뜻을 담고 있죠. (p110)

## 손실함수 (Loss Function)

_**신경망이 제대로된 학습을 했는지에 대한 수치를 알려주는 지표로 최적의 매개변수의 값을 탐색할 때 손실함수를 쓴다.**_    
손실함수는 max로 향할수도 있지만, 대부분 손실 함수의 값을 최소로 줄이는 것을 목표로한다.
손실함수는 직접 tuning 할 수 있지만, 오차제곱합, 교차 엔트로피 오차를 주로 사용한다.

### 왜 손실함수를 써야할까?

    손실함수의 목적은 신경망 모델의 높은 정확도를 이끌어내는 weight들을 찾는 것이다.

근데 왜 정확도라는 지표를 두고 손실함수를 이용해서 학습을 진행할까?? _**그 해답은 미분 (derivative)에 있다.**_

**정확도**라는 지표를 쓴다면? 100개의 test set 중에서 32개가 정답이여서 32%의 정확도라는 지표를 가지고 있다. 우리가 다음에 모델을 개선시키면 정확도는 32.011% 가 아닌, 34%, 35%로 _**불연속적인 미분의 값**_으로 학습할 것이다.

**손실함수**라는 지표를 이용한다면? 손실함수를 미분하여 연속적인 값이 나오게 되고, 연속적인 구간에서 우리가 weight들을 업데이트 할 수 있다.

<img src=../image/step_sigmoid.png width=70%>

이를 단편적으로 보여주는 것이 step function과 sigmoid 다.
계단함수는 한 순간에만 변화율이 나타나지만, sigmoid처럼 연속적인 함수들은 미분값이 연속적으로 변화한다.

_**기울기가 0이 되지 않아 신경망이 어느 지점에서나 올바르게 학습이 가능한 것이다.**_

### 1. 오차제곱합 (Sum of Square for Error, SSE)

$L = {1 \over 2}Σ_k(y_k-t_k)^2$

### 2. 교차 엔트로피 오차 (cross entropy error)

<img src=../image/cross_entropy.png width=50%>

교차 엔트로피 오차 방법은 $L = -Σt_k\log{y_k}$ 과 같다.
$log$는 자연로그 e이고, $y_k$는 신경망의 출력, $t_k$는 정답에 해당하는 레이블만 1이고 나머지는 0인 one-hot encoding이다. 따라서 실질적으로 정답을 맞춘 경우에만 Loss를 조절하는 단계이다.

**즉, $y_k$가 0에 가까워질수록 Loss가 커지고, $y_k$가 출력이 커질수록 Loss가 작아진다.**

나중에 이 부분을 code로 구현해보면 아래와 같다.

<code>

    def cross_entropy_error(y,t):
        delta = 1e-7
        return -np.sum(t*log(y + delta))

</code>

**여기에서 delta를 추가해준 이유는, log의 발산을 막기 위함이다.**

만약 신경망의 출력이 0이면 파이썬은 -inf 를 출력할 것이다. 따라서 발산을 막기위해 작은 숫자를 더해준다.

### 3. 미니배치 학습

$L =  - \frac{1}{N}\sum_{n}\sum_{k}{t_{nk}}\log{y_{nk}}$

Data가 $N$개 라면, $t_{nk}$는 n번째 데이터의 k번째 값, 정답 레이블을 의미한다.
마지막에 보면 $N$을 나누어 정규화하고 있다. $N$을 나눔으로써 _**평균손실함수**_ 를 구하는 것이다.

하지만, 모든 데이터를 대상으로 손실함수의 합을 구하려면 엄청나게 오래걸린다. 만약 data가 6만개가 있다고 가정하면, 이는 굉장한 낭비이다. 

많은 데이터를 한 번에 학습하면 각각의 데이터들이 가지는 정보량이 평균이라는 작업을 통해 손실된다.
따라서 data의 일부를 추려서 전체의 **근사치**로 이용한다. 
    
이를 _**미니배치라**_고 한다.

## 기울기 (gradient)


### 1. 정의
<img src=../image/gradient_func.png width=50%>

위와 같은 그림을 가지는 손실함수가 있다고 하자.
이 때, $(\frac{\partial{f}}{\partial x_0}, \frac{\partial{f}}{\partial x_1})$ 처럼 모든 변수의 편미분을 벡터로 정리한 것을 _**기울기**_라고 한다.

<img src=../image/gradient_graph.png width=50%>

편미분 값에 -부호를 붙여서 표현하면, 다음과 같은 그림대로 나타난다.
그림을 보면 _**기울기는 손실함수가 가장 낮은 곳으로 향하도록 방향을 잡아주는 수치이다.**_
_**즉, 기울기가 가르키는 쪽은 각 위치에서 함수의 출력값을 가장 크게 줄이는 방향이다.**_

<code>

    def _numerical_gradient(f, x): 
        h = 1e-4 # 0.0001 # 중심 차분을 써야 오류가 안 생김
        grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성 
        
        for idx in range(x.size): 
            tmp_val = x[idx] # f(x+h) 계산 
            
            x[idx] = float(tmp_val) + h 
            fxh1 = f(x) # f(x-h) 계산 
            
            x[idx] = tmp_val - h 
            fxh2 = f(x) 
            
            grad[idx] = (fxh1 - fxh2) / (2*h) 
            x[idx] = tmp_val # 값 복원 
            
        return grad

</code>

### 2. 손실함수에서의 기울기 의미 : 경사하강법

위의 그림처럼 기울어진 방향으로 가야 꼭 최소지점으로 가는 것은 아니나, **그 방향으로 가야 함수의 값을 줄 일 수 있다.**
경사법은 현 위치에서 기울어진 방향으로 일정한 간격을 두고 나아간다.
**이렇게 기울기를 이용해서 함수의 값을 줄여나가는 것을 경사법이라고 한다.**
경사법에는 경사하강법과 경사상강법 두 가지가 존재하는데, 경사하강법만 알아도 된다.

$x_0 = x_0 -\eta\frac{\partial{f}}{\partial x_0}$, $x_1 = x_1 -\eta\frac{\partial{f}}{\partial x_1}$

#### $\eta$는 learning rate, 학습률을 말한다.

<img src=../image/learning_rate.png width=50%>

1. learning rate가 일정 수준 크다면, 왼쪽처럼 발산하는 형태를 가진다.
2. learning rate가 너무 작다면, optimize하는데 너무 오래걸리고, 오른쪽 그림처럼 local minimum이 2개 이상 존재하는 함수에서는 local에 빠진다는 단점이 존재한다.

learning rate의 경우 hyperparameter로 사람이 직접 정해주어야 하는 변수를 말한다.
따라서 hyperparameter의 경우 잘 조절하면서 모델의 성능을 높여야한다.

### 3. 신경망에서의 기울기 의미 : 영향 정도

<img src=../image/NN_gradient.png width=50%>

<code>

    # coding: utf-8 
    import sys, os 
    sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정 
    import numpy as np 
    from common.functions import softmax, cross_entropy_error 
    from common.gradient import numerical_gradient 
    
    class simpleNet: 
        def __init__(self): 
            self.W = np.random.randn(2,3) # 정규분포로 초기화 
        
        def predict(self, x): 
            return np.dot(x, self.W) 
            
        def loss(self, x, t): 
            z = self.predict(x) 
            y = softmax(z) 
            loss = cross_entropy_error(y, t) 
            
            return loss 

    x = np.array([0.6, 0.9]) 
    t = np.array([0, 0, 1]) 
    
    net = simpleNet() 
    
    f = lambda w: net.loss(x, t) 
    dW = numerical_gradient(f, net.W) 
    
    print(dW)

    >>> import gradient_simplenet as gs 
    [[ 0.20712809 0.30619585 -0.51332394]
     [ 0.31069213 0.45929378 -0.76998591]] 
    
    >>> net = gs.simpleNet()
    >>> print(net.W)
    [[ 0.77872534 1.68368502 1.25014485] 
    [-1.29329159 0.30947396 -0.60716658]] 
    
    >>> import numpy as np 
    >>> x = np.array([0.6, 0.9]) 
    >>> p = net.predict(x) 
    >>> print(p) 
    [-0.69672723 1.28873757 0.20363699] 
    
    >>> np.argmax(p) 
    1 
    
    >>> t = np.array([0, 1, 0]) 
    >>> net.loss(x, t) 
    0.38878297065251444

    >>> def f(W):
            return net.loss(x, t) 
    >>> import gradient_2d as g2 
    >>> dW = g2.numerical_gradient(f, net.W) 
    >>> print(dW) 
    [[ 0.05585067 -0.19327121 0.13742053] 
     [ 0.08377601 -0.28990681 0.2061308 ]]

</code>

1. $\frac{\partial{f}}{\partial W_{23}}$은 0.206 정도 된다. 이는 h만큼 늘리면 손실함수는 0.206h 정도 된다는 것이다.
2. $\frac{\partial{f}}{\partial W_{22}}$은 -0.29 정도 된다. 이는 h만큼 늘리면 손실함수는 -0.29h 정도 된다는 것이다.
3. 손실함수를 줄인다는 관점에서는 $\frac{\partial{f}}{\partial W_{23}}$ 은 양의 방향으로 갱신한다. 미분 값의 크면 클수록 손실함수 갱신에 기여를 많이 한다는 뜻이기 때문이다.
4. 손실함수를 줄인다는 관점에서는 $W_{23}$은 음의 뱡향으로 갱신한다. 음의 방향으로 갱신해야 경사하강법에 의해 학습될 것이다.