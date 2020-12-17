# Summary

## 1. Background

1. Ai vs ML vs DL
   1. Ai : 인간만이 할 수 있었던 지능이 필요한 문제들을 해결하는 지능형 프로그램이나 기계를 만드는 학문
   2. Machine Learning : 사전지식 없이 오로지 학습으로만 문제 해결 능력을 배우는 방법
   3. Deep Learning : 머신러닝 기법 중 하나이며 신경망을 닮아 Nerual Network 라고 한다.
2. Deep Learning 장 단점
   1. 장점
      1. 성능이 매우 뛰어나다
      2. 표현력이 좋다.
      3. 정보를 계층적으로 전달할 수 있다. (Backpropagation)
   2. 단점 & 극복
      1. 데이터를 많이 필요로 한다. → IOT, 저장 하드웨어의 발전, 인터넷 속도 발전
      2. 연산량이 많다. → 하드웨어 (CPU, GPU) 의 발전
      3. 복잡한 구조를 가져 해석하기 어렵다 → interpreatble 한 모델이 생기기 시작한다.
      4. 빈약한 이론 → 지금도 연구 중
3. 비선형 함수를 쓰는 이유
   1. Model을 Deep 하게 쌓아서 정보를 계층적으로 전달 및 복잡한 문제 해결
   2. Filter size를 줄여서 More Deep 하도록 쌓는다.

-----------

## 2. Perceptron & 3. Neural Network

1. OR + NAND → AND : XOR gate를 풀 수 있었다.
2. Activation Function
3. _**What if we use the sigmoid with MSE cost function for classification problem?**_ >> Due to a non-linearity of sigmoid, using the sigmoid with MSE provides a non-convex shape of cost function with respect to parameter W. Therefore, it is hard to optimize the cost function with existing optimizers, e.g., gradient descent.
4. ZigZag Effect
5. Sigmoid 단점
   1. Gradient Vanishing
   2. output이 not zero centered
   3. Exponential expensive

-----------

## 4. Train Model

1. Loss Function을 쓰는 이유
   1. Accuracy로 Model 성능 평가를 하게 되면, Classification의 경우 64%, 95%, 90.1% 이런식으로 숫자가 bulk하게 움직인다. 따라서 미세하게 이동할 수 없기 때문에 Loss Function의 Global Minimum or Maximum을 구하는 것이 어렵다. 
   2. 따라서 Loss Function의 Gradient Descent를 이용해서 Backpropagation을 이용하는 것이 바람직하다.
2. MSE or MAE
3. Cross Entropy Error
   1. 의의 : 단순히 classification을 성공했다고 끝나는 것이 아니라 Classification을 맞춘 확률의 정도를 검사해서, 맞게 분류했던 문제는 더 잘 분류하도록 만든다.
4. Mini-Batch 학습
   1. Batch에 들어온 모든 데이터의 Loss를 합해서 N분의 1로 나눈 평균값을 쓴다.
   2. 하나하나 없데이트하면 발산하는 경우도 많고, Computation이 효율적이지 않아 Approximate Value를 채택하는 것이다.
5. Loss Function에서 중요한 3가지
   1. 출발점이 어디야? : Initialization
   2. learning rate는 몇이야? : Learning rate
   3. 어느 방향으로 가느냐? Gradient

-----------

## 6. Optimizer

1. SGD
   1. ZigZag
   2. 비등방성함수 (방향에 따라 성질이 달라지는 함수) 에서는 부적합
2. Momentum
   1. $v_{t} = \rho v_{t-1} - \eta \triangledown L(w)$
   2. $w_{t+1} = w_t + v_{t+1}$
3. Nesterov Accelerated Gradient (NAG)
   1. $v_{t} = \rho v_{t-1} - \eta \triangledown L(w + \rho v_{t-1})$
   2. $w_{t+1} = w_t + v_{t+1}$
4. 초기 weight를 모두 동일하게하면 쌓는 의미가 없다.
5. He init
6. MSRA or He init
7. Hyperparamter 조절 방법
   1. epoch마다 lr을 0.1씩 곱해가자
   2. linear 하게 줄이자
   3. sqrt, Adagrad 처럼 큰 error를 내면 lr을 크게 하도록 한다.
   4. Coarse-to-fine 방법을 쓴다.
      1. 처음에는 Coarse 하게 범위를 넓게 설정하고 어떤 파라미터에서 error가 큰 폭으로 주는지 확인한다.
      2. hyperparameter 범위를 다시 설정하고 fine하게 튜닝한다.
   5. Monitor loss curve
      1. training 과 validation gap이 너무 크면? Overfitting
      2. training 과 validation gap이 너무 작으면? Model의 Capacity가 부족함
   6. grid vs Random
   7. Early Stop : 
8. Regularization 종류
   1. L1, L2, ElasticNet
   2. Dropout
   3. Mixup
   4. CutMix
   5. Transfer Learning
   6. Batch Normalization