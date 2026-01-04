---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 18차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# 딥러닝 입문: 신경망 기초

## 18차시 | Part III. 문제 중심 모델링 실습

**인공지능의 핵심, 신경망 이해하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **신경망**의 기본 구조를 이해한다
2. **뉴런과 층**의 개념을 설명한다
3. **딥러닝 vs 머신러닝**의 차이를 구분한다

---

# 딥러닝이란?

## Deep Learning

> 여러 층의 **신경망**으로 학습하는 머신러닝의 한 분야

```
  머신러닝
     │
     ├── 선형회귀
     ├── 의사결정트리
     ├── 랜덤포레스트
     └── 딥러닝 ← 여러 층의 신경망
            ├── MLP (다층 퍼셉트론)
            ├── CNN (이미지)
            └── RNN (시계열)
```

---

# 왜 딥러닝인가?

## 복잡한 패턴 학습

### 전통적 ML
- 특성 엔지니어링이 중요
- 사람이 특성을 설계

### 딥러닝
- 스스로 특성을 학습
- 복잡한 패턴 발견 가능
- 이미지, 텍스트, 음성에서 강력

> 데이터가 많고 복잡한 문제에 효과적!

---

# 생물학적 신경망

## 뇌의 뉴런

```
    수상돌기            세포체           축삭돌기
  (입력 받음)       (처리/판단)       (출력 전달)

      ○──┐             ┌─○─┐
      ○──┼───────●────●     ●───→ 다음 뉴런으로
      ○──┘
            신호 임계값 초과 시 활성화
```

- 여러 뉴런에서 신호 수신
- 임계값 초과 시 **활성화**
- 다음 뉴런으로 전달

---

# 인공 뉴런

## Artificial Neuron

```
  입력        가중치        합산        활성화 함수      출력

   x₁ ──→ ×w₁ ─┐
                │
   x₂ ──→ ×w₂ ─┼──→ Σ + b ──→ f(x) ──→ y
                │
   x₃ ──→ ×w₃ ─┘
```

### 수식
$$y = f(w_1x_1 + w_2x_2 + w_3x_3 + b)$$

> 입력 × 가중치 → 합산 → 활성화 함수 → 출력

---

# 가중치와 편향

## Weights & Bias

```python
# 입력 (온도, 습도, 속도)
x = [85, 50, 100]

# 가중치 (각 입력의 중요도) - 학습됨
w = [0.3, 0.2, 0.5]

# 편향 (기준점 조정) - 학습됨
b = 0.1

# 계산
z = 85*0.3 + 50*0.2 + 100*0.5 + 0.1
# z = 25.5 + 10 + 50 + 0.1 = 85.6
```

> **가중치** = 입력의 중요도 / **편향** = 기준점 조정

---

# 활성화 함수

## Activation Function

> 비선형성을 추가하는 함수

### 왜 필요한가?
- 활성화 함수 없으면 → 아무리 깊어도 선형 변환
- 비선형 함수 추가 → 복잡한 패턴 학습 가능

| 함수 | 범위 | 특징 |
|------|------|------|
| Sigmoid | 0~1 | 확률 출력에 적합 |
| **ReLU** | 0~∞ | 현재 가장 많이 사용 |
| Tanh | -1~1 | 중심이 0 |

---

# ReLU 함수

## Rectified Linear Unit

```python
def relu(x):
    return max(0, x)
```

```
       ↑ y
       │    /
       │   /
       │  /
───────┼─────→ x
       │

음수 → 0, 양수 → 그대로
```

> 단순하지만 강력! 현재 가장 많이 사용

---

# 신경망 층 (Layer)

## 뉴런들의 집합

```
  입력층        은닉층         출력층
 (Input)      (Hidden)      (Output)

   ○ ───────→ ○ ───────→ ○
   ○ ───────→ ○ ───────→
   ○ ───────→ ○
              ○

 3개 입력     4개 뉴런      1개 출력
```

- **입력층**: 데이터 받는 층
- **은닉층**: 특성 학습하는 층
- **출력층**: 결과 내는 층

---

# 심층 신경망 (DNN)

## Deep Neural Network

```
입력층      은닉층1      은닉층2      은닉층3      출력층

  ○ ─────→ ○ ─────→ ○ ─────→ ○ ─────→ ○
  ○ ─────→ ○ ─────→ ○ ─────→ ○
  ○ ─────→ ○ ─────→ ○ ─────→ ○
           ○         ○

         "Deep" = 층이 깊다
```

> 은닉층이 2개 이상이면 "Deep" Learning

---

# 학습 과정

## 순전파 & 역전파

### 1. 순전파 (Forward Propagation)
```
입력 → 계산 → 예측값 → 손실 계산
```

### 2. 역전파 (Backpropagation)
```
손실 → 기울기 계산 → 가중치 업데이트
```

```
    예측: 0.3
    실제: 1.0
    손실: 0.7 → 가중치 조정 필요!
```

---

# 손실 함수 (Loss Function)

## 예측 오차 측정

### 회귀 문제
```python
# MSE: Mean Squared Error
loss = mean((y_true - y_pred)**2)
```

### 분류 문제
```python
# Cross-Entropy
loss = -mean(y_true * log(y_pred))
```

> 손실이 작아지도록 가중치를 조정!

---

# 경사하강법 (Gradient Descent)

## 최적의 가중치 찾기

```
손실 ↑
      │   ●  시작점
      │    ╲
      │     ╲
      │      ●  경사 따라 내려가기
      │       ╲
      │        ●  최소점 (목표)
      └──────────→ 가중치
```

> 기울기(경사) 방향으로 조금씩 이동

---

# 딥러닝 vs 머신러닝

## 비교

| | 머신러닝 | 딥러닝 |
|--|---------|--------|
| 특성 | 사람이 설계 | 자동 학습 |
| 데이터 | 적어도 가능 | 많이 필요 |
| 연산 | CPU 가능 | GPU 권장 |
| 해석 | 상대적 쉬움 | 블랙박스 |
| 적합 | 테이블 데이터 | 이미지, 텍스트 |

### 제조 현장에서는?
- 테이블 데이터 → ML (RandomForest 등)
- 이미지 검사 → 딥러닝 (CNN)

---

# 이론 정리

## 딥러닝 핵심 개념

| 개념 | 설명 |
|------|------|
| 뉴런 | 입력 × 가중치 + 편향 → 활성화 |
| 층 (Layer) | 뉴런들의 집합 |
| 은닉층 | 특성을 학습하는 층 |
| 활성화 함수 | 비선형성 추가 (ReLU) |
| 역전파 | 가중치 업데이트 방법 |
| 경사하강법 | 손실 최소화 방향 탐색 |

---

# - 실습편 -

## 18차시

**신경망 구조 이해 실습**

---

# 실습 개요

## NumPy로 뉴런 이해하기

### 목표
- 뉴런 계산 구현
- 활성화 함수 이해
- 간단한 신경망 구조

### 실습 환경
```python
import numpy as np
import matplotlib.pyplot as plt
```

---

# 실습 1: 뉴런 계산

## 입력 × 가중치 + 편향

```python
# 입력 (온도, 습도, 속도)
inputs = np.array([85, 50, 100])

# 가중치
weights = np.array([0.3, 0.2, 0.5])

# 편향
bias = 0.1

# 뉴런 계산
z = np.dot(inputs, weights) + bias
print(f"계산 결과: {z}")  # 85.6
```

---

# 실습 2: 활성화 함수

## ReLU, Sigmoid

```python
# ReLU
def relu(x):
    return np.maximum(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 적용
z = 85.6
print(f"ReLU: {relu(z)}")      # 85.6
print(f"Sigmoid: {sigmoid(z)}")  # ≈ 1.0
```

---

# 실습 3: 활성화 함수 시각화

## 그래프

```python
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, relu(x), linewidth=2)
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, sigmoid(x), linewidth=2)
plt.title('Sigmoid')
plt.grid(True)

plt.show()
```

---

# 실습 4: 단일 뉴런 클래스

## Neuron Class

```python
class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return relu(z)

# 사용
neuron = Neuron(3)
output = neuron.forward([85, 50, 100])
print(f"뉴런 출력: {output}")
```

---

# 실습 5: 간단한 층

## Layer Class

```python
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        return relu(z)

# 3개 입력 → 4개 뉴런
layer = Layer(3, 4)
output = layer.forward([85, 50, 100])
print(f"층 출력: {output}")  # 4개 값
```

---

# 실습 6: 2층 신경망

## 순전파

```python
# 2층 신경망
layer1 = Layer(3, 4)   # 입력 → 은닉층
layer2 = Layer(4, 1)   # 은닉층 → 출력

# 순전파
inputs = np.array([85, 50, 100])
hidden = layer1.forward(inputs)  # 은닉층 출력
output = layer2.forward(hidden)  # 최종 출력

print(f"입력: {inputs}")
print(f"은닉층: {hidden}")
print(f"출력: {output}")
```

---

# 실습 7: 손실 계산

## MSE Loss

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 예측 vs 실제
y_pred = np.array([0.8, 0.3, 0.9])
y_true = np.array([1.0, 0.0, 1.0])

loss = mse_loss(y_true, y_pred)
print(f"손실: {loss:.4f}")
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] 뉴런: 입력 × 가중치 + 편향
- [ ] ReLU: max(0, x)
- [ ] Sigmoid: 1 / (1 + e^(-x))
- [ ] 층: 여러 뉴런의 집합
- [ ] 순전파: 입력 → 은닉층 → 출력

---

# 다음 차시 예고

## 19차시: 딥러닝 실습 - MLP로 품질 예측

### 학습 내용
- Keras로 신경망 구축
- MLP 모델 학습
- 제조 품질 예측 적용

> 실제 **딥러닝 프레임워크** 사용!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **뉴런**: 입력 × 가중치 + 편향 → 활성화
2. **활성화 함수**: 비선형성 (ReLU 많이 사용)
3. **층**: 뉴런들의 집합
4. **딥러닝**: 은닉층이 2개 이상
5. **역전파**: 손실 줄이는 방향으로 가중치 조정

---

# 감사합니다

## 18차시: 딥러닝 입문 - 신경망 기초

**인공지능의 핵심, 신경망을 이해했습니다!**
