# 신경망 (Neural Network)

## 1. 퍼셉트론 복습
퍼셉트론을 복습해보면, 여러 개의 signal을 받아서 하나의 출력을 내는 구조이다.
이해가 되지 않는다면, [복습하기](2_perceptron.md) 로 다시 공부하고 온다.

------------

## 2. 퍼셉트론 vs 신경망

<img src=../image/NN_structure.svg>

우선, 신경망은 퍼셉트론에 비해 2가지 차이점이 존재한다.
    
    1. 중간에 Hidden Layer가 존재한다.
    2. 여러 개의 Output이 존재한다.

------------

## 3. 활성화 함수 (Activation Function)
<img src=../image/activation_function_procedure.png width=30%>

Input signal를 받아서 Input signal의 총합을 Output signal로 변환하는 함수를 말한다.
통상적으로 $h(x)$, or $f$라고 표기한다.

### 3-1. 활성화 함수 종류
- sigmoid, tanh, ReLU, Leaky ReLU, Maxout, ELU
    <img src=../image/activation_function.png width=70%>
    - sigmoid : 보통 binary classification 할 때 주로 사용하며 퍼셉트론의 업그레이드 버젼
      - 문제점
        1. Gradient Vanishing : 미분해보면 $|(1-f(x))f(x)|<1$ 이기에 layer를 거칠수록 소실된다.
        2. output이 not zero centered : $\frac{\partial L}{\partial p} >0 , \frac{\partial L}{\partial w} >0$, 의존적이다.
        3. Exponential computation is expensive
    - tanh : $F(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, sigmoid와 비슷하지만 0 근처에서 경사가 sigmoid보다 좀 더 가파름
      - 문제점
        - exponential computation is expensive
    - ReLU : 가장 많이 쓰는 함수, 비선형성이 강함\
      - 문제점
        - X<0 인 지점은 모두 0, 정보 유실 가능성
    - Leaky-ReLU : ReLU가 input이 음수일 때 모두 0인 단점을 극복, 비선형성이 강함
    - Maxout : max인 부분을 골라서 output
    - ELU : ReLU와 비슷함.
- sigmoid, tanh, ReLU, ELU 미분 값 (역전파 때 중요)
    <img src=../image/activation_function_derivative.png width=70%>
- identity : 항등함수, _**Regression에 주로 사용한다.**_
    <img src=../image/identity_function.png width=40%>
- softmax : 모든 input signal을 받고 출력한다. _**classification에 주로 사용한다.**_
    <img src=../image/softmax.jpeg width=40%>
    - softmax 구현시 주의점
      - softmax는 지수함수로 이루어져 있기 때문에, 오버플로의 문제점이 존재. input signal 중 max값을 input에 다 빼준다
      - softmax를 이용하여 모델을 training한 후, predict할 때는 softmax는 computing power를 많이 잡아먹기에 생략
      - 어차피 softmax의 지수함수는 단조증가함수, input의 순위와 output의 순위가 동일하다.

### 3-2. 활성화 함수 써야하는 이유 : More Deeper
다중 클래스 분류의 경우 위 그림이 수학적으로 어떻게 정의되는지 보자.

$$ h = W_1x+b_1, o = W_2h+b_2, \hat y = softmax(o)$$

위 방법의 문제점은 은닉층을 $W=W_2W_1$ 과 $b=W_2b_1+b_2$ 를 사용해서 단일층 퍼셉트론(single layer perceptron) 식으로 재구성할 수 있기 때문에, 다층(mutilayer)이 아닌 단순한 단일층 퍼셉트론(single layer perceptron)이라는 점이다.

$$o=W_2h+b_2=W_2(W_1x+b_1)+b2=(W_2W_1)x+(W_2b_1+b_2)=W'x+b'$$

#### **즉, 비선형 Activation Function을 거치지 않는다면, 여러 개의 layer를 쌓아도 소용이 없다.** 그냥 퍼셉트론과 똑같은 상황이다.

#### _**이를 해결하는 방법은 모든 층의 다음에 max(x,0) (ReLU) 와 같은 비선형 함수 σ를 추가하면서 model이 복잡한 부분도 표현할 수 있게되는 것이다.**_

$$h_1 = \sigma(W_1x+b_1), h_2 = \sigma(W_2h_1+b_2), o = W_3h_2+b_3, \hat y = softmax(o)$$
 
이렇게 하면, 여러 개의 은닉층들을 쌓는 것도 가능하다. 즉, $h_1=σ(W_1x+b_1)$ 과 $h_2=σ(W_2h_1+b_2)$ 를 각각 연결해서 진짜 신경망을 만들 수 있다.