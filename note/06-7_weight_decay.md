# Weight Decay

<img src="../image/weight_decay_2.png">

Overfitting 문제를 해결하기 위해서 여러가지 방법이 쓰일 수 있는데, 그 중 한가지가 Weight decay이다. 

Loss function이 작아지는 방향으로만 단순하게 학습을 진행하면 특정 가중치 값들이 커지면서 위 첫번째 그림처럼 오히려 결과가 나빠질 수 있다. 

Weight decay는 학습된 모델의 복잡도를 줄이기 위해서 학습 중 weight가 너무 큰 값을 가지지 않도록 Loss function에 Weight가 커질경우에 대한 패널티 항목을 집어넣는다. 

<img src="../image/weight_decay.png">

이 패널티 항목으로 많이 쓰이는 것이 L1 Regularization과 L2 Regularization이다. Weight decay 를 적용할 경우 위 두번째 그림처럼 Overfitting에서 벗어날 수 있다.
