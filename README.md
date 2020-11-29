# 밑바닥부터 시작하는 딥러닝 + DeepLearning_Basic

## Book & Lecture
![book](image/book.jpg)

1. 밑바닥부터 시작하는 딥러닝(DeepLearning From Scratch, OREILLY)
2. 2020-2 기계학습응용 by [Prof.김승룡 in korea.univ](https://www.youtube.com/playlist?list=PLCNc54m6eBRWz3tmkBPJAIdkSn6vFhtgR)

## Purpose

1. 밑바닥부터 시작하는 딥러닝 책으로 몰랐던 기초를 쌓는다.
    1. 현재 하고 있는 연구와의 연관성을 생각하면서 정리한다.
    2. 공부한 부분은 마크다운으로 정리하고 README.md에 링크 걸어둔다.
    3. 공부한 부분은 직접 Code로 구현한다.
2. 기계학습응용 수업을 들으면서 추가할 부분을 상세히 적는다.
   1. 밑바닥부터 시작하는 딥러닝 마크다운 파일에 추가한다.

이 4가지의 규칙을 가지고 책을 공부하도록 한다.

## Contents
- [0장 Background](note/00_Background.md)
- 1장 헬로 파이썬
    * 1.1 파이썬이란?
    * 1.2 파이썬 설치하기
    * 1.3 파이썬 인터프리터 
    * 1.4 파이썬 스크립트 파일 
    * [1.5 Numpy](note/01_6_numpy.md)
    * 1.6 Matplotlib
    * 1.7 정리 
- [2장 퍼셉트론](note/02_perceptron.md)
    * 2.1 퍼셉트론이란?
    * 2.2 단순한 논리 회로 
    * 2.3 퍼셉트론 구현하기 
    * 2.4 퍼셉트론의 한계 
    * 2.5 다층 퍼셉트론이 출동한다면
    * 2.6 NAND에서 컴퓨터까지 
    * 2.7 정리 
- [3장 신경망](note/03_neuralnetwork.md)
    * 3.1 퍼셉트론에서 신경망으로 
    * 3.2 활성화 함수 
    * 3.3 다차원 배열의 계산 
    * 3.4 3층 신경망 구현하기
    * 3.5 출력층 설계하기 
    * 3.6 손글씨 숫자 인식 
    * 3.7 정리 
- [4장 신경망 학습](note/04_NNtrain.md)
    * 4.1 데이터에서 학습한다!
    * 4.2 손실 함수 
    * 4.3 수치 미분 
    * 4.4 기울기 
    * 4.5 학습 알고리즘 구현하기 
    * 4.6 정리 
- [5장 오차역전파법](note/05_Backpropagation.md)
    * 5.1 계산 그래프
    * 5.2 연쇄법칙 
    * 5.3 역전파
    * 5.4 단순한 계층 구현하기 
    * 5.5 활성화 함수 계층 구현하기 
    * 5.6 Affine/Softmax 계층 구현하기 
    * 5.7 오차역전파법 구현하기 
    * 5.8 정리 
- 6장 학습 관련 기술들
    * [6.1 매개변수 갱신](note/06-1_Optimizer.md)
    * [6.2 가중치의 초깃값](note/06-2_Initialize.md)
    * [6.3 배치 정규화](note/06-3_Batch_normalization.md)
    * [6.4 바른 학습을 위해](note/06-3_Batch_normalization.md)
    * [6.5 적절한 하이퍼파라미터 값 찾기]((note/06-3_Batch_normalization.md))
    * 6.6 정리 
    * [6.7 weight decay](note/06-7_weight_decay.md)
- [7장 합성곱 신경망(CNN)](note/07_CNN.md)
    * 7.1 전체 구조 
    * 7.2 합성곱 계층 
    * 7.3 풀링 계층 
    * 7.4 합성곱/풀링 계층 구현하기 
    * 7.5 CNN 구현하기 
    * 7.6 CNN 시각화하기 
    * 7.7 대표적인 CNN 
    * 7.8 정리 
- 8장 딥러닝
    * [8.1 더 깊게](note/00_Background.md) 
    * 8.2 딥러닝의 초기 역사 
    * [8.3 더 빠르게(딥러닝 고속화)](note/08_DeepLearnig.md) 
      * cpu vs gpu
      * CUDA, cuDNN
      * im2col
      * GPU 병렬 연결
      * 연산 정밀도와 비트수 줄이기
        * 부동소수점
    * 8.4 딥러닝의 활용 
    * 8.5 딥러닝의 미래 
    * 8.6 정리 
- 9장 모르는거 정리 (복습하면 x 표시)
    - [x] XOR 한계 : p55
    - [x] XOR 극복 : p59
    - [x] 비선형 함수를 써야하는 이유 : p75
    - [x] 항등함수 정의 : p91
    - [x] softmax 정의시 프로그래밍 언어적 한계 : p93
    - [x] sottmax를 학습할 때만 쓰고, test할 때는 버리는 이유 : p95
    - [x] computer vision에서 feature selection하는 다양한 방법 : p109
    - [x] cross entropy error 구현하는데 프로그래밍 언어적 한계 : p115
    - [x] mini-batch의 정의
    - [x] 왜 Loss Function을 써야하는가? : p119
    - [x] 수치미분을 python으로 구현할 때 한계점 : p123
    - [x] Affine 정의 : p171
    - [ ] Affine 행렬 shape 계산 : p173
    - [ ] softmax backpropagation : p177, 291
    - [ ] TwolayerNet 구성하기 : p183
    - [x] SGD 정의 : p190, 191
    - [x] SGD 단점 : p192, 193
    - [x] Momentum 정의 : p195 
    - [x] Adagrad 정의 : p197
    - [x] weight initialize를 동일하게 하면 안되는 이유 : p202
    - [x] weight initialize 해야하는 이유 : p205
    - [x] Xavier, He 정의 : p206, 207
    - [x] Batch Normalization : p210, 211
    - [x] Fully connected 했을 때, 이미지를 잘 분류하지 못하는 이유 : p229
    - [x] CNN 의 weight : p231
    - [x] filter, kernel, window 정의 : p231
    - [x] input data의 channel수와 filter 수를 맞춰야한다 : p236
    - [x] CUDA, cuDNN : p275
    - [x] im2col : p275
    - [x] 연산정밀도 비트수 줄이기 : p277, 278


## Source Example (Reference)

https://github.com/WegraLee/deep-learning-from-scratch