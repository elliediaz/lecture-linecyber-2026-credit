# [20차시] 딥러닝 입문: 신경망 기초

## 학습 목표

| 번호 | 목표 |
|:----:|------|
| 1 | 인공 뉴런의 개념을 이해함 |
| 2 | 신경망 구조를 파악함 |
| 3 | 순전파와 역전파를 이해함 |

---

## 실습 데이터: sklearn digits 데이터셋

sklearn에서 제공하는 8x8 손글씨 숫자 데이터셋을 활용함

### 데이터 특성

| 항목 | 내용 |
|------|------|
| 샘플 수 | 1,797개 |
| 특성 수 | 64개 (8x8 픽셀) |
| 클래스 | 0-9 숫자 (10개) |
| 픽셀 값 | 0~16 범위 |

MNIST보다 작아서 CPU 환경에서도 학습이 빠름

---

## Part 1: 인공 뉴런의 개념

### 1.1 머신러닝 vs 딥러닝

| 구분 | 머신러닝 | 딥러닝 |
|------|----------|--------|
| 모델 | RF, SVM, LR | 신경망 |
| 특성 추출 | 수동 | 자동 |
| 데이터 양 | 적어도 가능 | 많아야 효과적 |
| 계산 자원 | CPU 가능 | GPU 권장 |
| 해석 | 비교적 쉬움 | 블랙박스 |

### 딥러닝이란?

- 깊은(Deep) 인공 신경망을 사용한 학습
- 여러 층(Layer)을 쌓아 복잡한 패턴을 학습함
- 층이 2개 이상이면 Deep이라고 함

---

### 1.2 생물학적 뉴런 vs 인공 뉴런

```
[생물학적 뉴런]
                  수상돌기
                    |
                    v
[입력 신호] -> [세포체] -> [축삭] -> [출력]
                    |
             역치 초과 시 발화

[인공 뉴런]
   x1 --w1--+
   x2 --w2--+
   x3 --w3--+---> Sigma ---> f(z) ---> y
      ...   |
   xn --wn--+
            +-- +b (bias)
```

---

### 1.3 인공 뉴런 (퍼셉트론) 수식

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
y = f(z)  (활성화 함수)
```

| 요소 | 기호 | 설명 |
|------|------|------|
| 입력 | x1, x2, ... | 특성 값 |
| 가중치 | w1, w2, ... | 입력의 중요도 |
| 편향 | b | 기준점 조정 |
| 합계 | z | 가중합 + 편향 |
| 활성화 함수 | f | 비선형 변환 |
| 출력 | y | 최종 결과 |

---

### 1.4 가중치의 의미

```python
# 온도와 압력으로 불량 예측
z = w1 * 온도 + w2 * 압력 + b

# w1 = 0.8, w2 = 0.2
# -> 온도가 4배 더 중요함!
```

학습 = 최적의 가중치 찾기
- 데이터를 통해 w, b를 조정함
- 예측 오차를 최소화함

---

### 1.5 실습 환경 설정

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[20차시] 딥러닝 입문: 신경망 기초")
print("=" * 60)
```

---

### 1.6 활성화 함수 구현

```python
def sigmoid(z):
    """시그모이드 함수: 출력 범위 0~1"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """시그모이드 미분: 역전파에서 사용"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU 함수: 음수는 0, 양수는 그대로"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU 미분: 양수면 1, 음수면 0"""
    return (z > 0).astype(float)

def tanh(z):
    """Tanh 함수: 출력 범위 -1~1"""
    return np.tanh(z)

def softmax(z):
    """Softmax 함수: 다중 클래스 분류용"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 안정성
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

### 1.7 활성화 함수의 필요성

#### 활성화 함수가 없으면?

```
z = w1*x1 + w2*x2 + b  (선형)
```

층을 아무리 쌓아도 선형 관계만 표현됨

#### 활성화 함수가 있으면?

```
y = f(w1*x1 + w2*x2 + b)  (비선형)
```

복잡한 패턴 학습이 가능해짐

---

### 1.8 주요 활성화 함수 비교

| 함수 | 수식 | 출력 범위 | 특징 |
|------|------|----------|------|
| **Sigmoid** | 1/(1+e^(-z)) | 0~1 | 이진 분류 출력층 |
| **ReLU** | max(0, z) | 0~무한 | 은닉층 표준 |
| **Tanh** | (e^z-e^(-z))/(e^z+e^(-z)) | -1~1 | 0 중심 |
| **Softmax** | e^zi/Sigma(e^zj) | 0~1 (합=1) | 다중 분류 출력층 |

---

### 1.9 활성화 함수 시각화

```python
z = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sigmoid
axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
axes[0, 0].plot(z, sigmoid_derivative(z), 'b--', linewidth=1.5, label='Sigmoid 미분')
axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
axes[0, 0].set_title('Sigmoid: 0~1 출력, 이진 분류 출력층')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ReLU
axes[0, 1].plot(z, relu(z), 'r-', linewidth=2, label='ReLU')
axes[0, 1].plot(z, relu_derivative(z), 'r--', linewidth=1.5, label='ReLU 미분')
axes[0, 1].set_title('ReLU: max(0, z), 은닉층 표준')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Tanh
axes[1, 0].plot(z, tanh(z), 'g-', linewidth=2, label='Tanh')
axes[1, 0].set_title('Tanh: -1~1 출력, 0 중심')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 비교
axes[1, 1].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
axes[1, 1].plot(z, relu(z)/5, 'r-', linewidth=2, label='ReLU/5')
axes[1, 1].plot(z, tanh(z), 'g-', linewidth=2, label='Tanh')
axes[1, 1].set_title('활성화 함수 비교')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 시각화 해설

- Sigmoid: S자 곡선, 0과 1 사이 출력
- ReLU: 0 이하는 0, 양수는 그대로 출력
- Tanh: Sigmoid와 유사하나 -1~1 범위

---

### 1.10 활성화 함수 선택 가이드

| 위치 | 권장 함수 | 이유 |
|------|----------|------|
| 은닉층 | ReLU (또는 Leaky ReLU) | 기울기 소실 완화, 빠른 계산 |
| 출력층 (이진 분류) | Sigmoid | 0~1 확률 출력 |
| 출력층 (다중 분류) | Softmax | 합이 1인 확률 분포 |
| 출력층 (회귀) | 없음 (Linear) | 연속 값 출력 |

---

## Part 2: 신경망 구조

### 2.1 신경망의 층 구조

```
  입력층        은닉층1       은닉층2        출력층
  +---+        +---+        +---+        +---+
  | o |------->| o |------->| o |------->| o |
  | o |------->| o |------->| o |------->| o |
  | o |------->| o |------->| o |------->+---+
  | o |------->| o |------->+---+
  +---+        +---+
   (4)          (4)          (3)          (2)
```

| 층 | 역할 |
|------|------|
| **입력층** | 특성 값을 받는 층, 뉴런 수 = 특성 수 |
| **은닉층** | 입력과 출력 사이의 층, 깊이 = 은닉층 개수 |
| **출력층** | 최종 예측 출력, 뉴런 수 = 클래스 수 (분류) 또는 1 (회귀) |

---

### 2.2 완전 연결 층 (Dense Layer)

```
  이전 층         다음 층
   o---------------o
    \             /
     \           /
      \---------o
       \       /
        \-----o
         \   /
          \-o
```

특징:
- 모든 뉴런이 서로 연결됨
- FC Layer, Dense Layer라고 불림
- 가장 기본적인 신경망 구조

---

### 2.3 파라미터 수 계산

```python
def count_parameters(layers):
    """
    신경망 파라미터 수 계산

    Args:
        layers: 각 층의 노드 수 리스트 [입력, 은닉1, 은닉2, ..., 출력]

    Returns:
        총 파라미터 수, 층별 파라미터 수
    """
    total_params = 0
    layer_params = []

    for i in range(len(layers) - 1):
        # 가중치: 이전 층 x 현재 층
        weights = layers[i] * layers[i+1]
        # 편향: 현재 층
        biases = layers[i+1]
        # 층별 총 파라미터
        params = weights + biases

        layer_params.append({
            'layer': f'층 {i+1} ({layers[i]}->{layers[i+1]})',
            'weights': weights,
            'biases': biases,
            'total': params
        })
        total_params += params

    return total_params, layer_params
```

---

```python
# 예시 1: digits 분류 (간단한 구조)
print("예시 1: 입력(64) -> 은닉(32) -> 출력(10)")
layers1 = [64, 32, 10]
total1, details1 = count_parameters(layers1)
for d in details1:
    print(f"  {d['layer']}: 가중치 {d['weights']}, 편향 {d['biases']}, 합계 {d['total']}")
print(f"  총 파라미터: {total1}개")

# 예시 2: 더 깊은 구조
print("\n예시 2: 입력(64) -> 은닉(128) -> 은닉(64) -> 출력(10)")
layers2 = [64, 128, 64, 10]
total2, details2 = count_parameters(layers2)
for d in details2:
    print(f"  {d['layer']}: 가중치 {d['weights']}, 편향 {d['biases']}, 합계 {d['total']}")
print(f"  총 파라미터: {total2}개")
```

#### 결과 해설

```
예시 1: 64 -> 32 -> 10
  층 1 (64->32): 가중치 2048, 편향 32, 합계 2080
  층 2 (32->10): 가중치 320, 편향 10, 합계 330
  총 파라미터: 2,410개

예시 2: 64 -> 128 -> 64 -> 10
  층 1 (64->128): 가중치 8192, 편향 128, 합계 8320
  층 2 (128->64): 가중치 8192, 편향 64, 합계 8256
  층 3 (64->10): 가중치 640, 편향 10, 합계 650
  총 파라미터: 17,226개
```

파라미터 수 = (이전 노드 x 현재 노드) + 현재 노드

---

### 2.4 MLP (Multi-Layer Perceptron)

정의:
- 여러 층으로 구성된 퍼셉트론
- 가장 기본적인 신경망

구조:
```
입력 -> [Dense + ReLU] x n -> [Dense] -> 출력
```

일반적인 MLP 구조:

| 문제 | 구조 |
|------|------|
| 분류 | 입력(특성수) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(클래스수, Softmax) |
| 회귀 | 입력(특성수) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear) |

---

### 2.5 데이터 로드

```python
# digits 데이터 로드 (8x8 손글씨 숫자)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"[Digits 데이터셋]")
print(f"  샘플 수: {X_digits.shape[0]}")
print(f"  특성 수: {X_digits.shape[1]} (8x8 픽셀)")
print(f"  클래스: {np.unique(y_digits)} (0-9 숫자)")
print(f"  데이터 범위: {X_digits.min():.1f} ~ {X_digits.max():.1f}")
```

---

```python
# 샘플 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y_digits[i]}')
    ax.axis('off')
plt.suptitle('Digits 샘플 이미지 (8x8 픽셀)', fontsize=14)
plt.tight_layout()
plt.show()
```

#### 시각화 해설

- 8x8 = 64 픽셀로 구성된 손글씨 숫자
- 0~16 범위의 그레이스케일 값
- 신경망 입력층 뉴런 수 = 64

---

## Part 3: 순전파와 역전파

### 3.1 신경망 학습 과정

```
+-----------------------------------------------+
| 1. 순전파: 입력 -> 출력 계산                    |
| 2. 손실 계산: 예측 vs 실제 비교                 |
| 3. 역전파: 기울기(gradient) 계산                |
| 4. 가중치 업데이트: w = w - lr * dL/dw         |
| 5. 반복 (epoch)                               |
+-----------------------------------------------+
```

---

### 3.2 순전파 (Forward Propagation)

```
입력 x -> 은닉층 -> 출력 y_hat

z1 = W1*x + b1
a1 = ReLU(z1)
z2 = W2*a1 + b2
y_hat = Softmax(z2)
```

의미:
- 입력을 받아 출력을 계산하는 과정
- 예측값을 생성함

---

### 3.3 손실 함수 (Loss Function)

예측과 실제의 차이를 수치화함

| 문제 | 손실 함수 | 수식 |
|------|----------|------|
| 회귀 | MSE | L = (1/n) * Sigma(y - y_hat)^2 |
| 이진 분류 | Binary Cross-Entropy | L = -Sigma(y*log(y_hat) + (1-y)*log(1-y_hat)) |
| 다중 분류 | Categorical Cross-Entropy | L = -Sigma(y*log(y_hat)) |

목표: L을 최소화하는 w, b 찾기

---

### 3.4 손실 함수 구현

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error (회귀용)"""
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-8):
    """Binary Cross-Entropy (이진 분류용)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-8):
    """Categorical Cross-Entropy (다중 분류용)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

---

### 3.5 역전파 (Backpropagation)

목적:
- 손실 함수의 기울기를 계산함
- 각 가중치가 손실에 얼마나 영향을 미치는지 파악함

과정:
```
출력 <- 은닉층 <- 입력

dL/dW2 = ... (출력층 기울기)
dL/dW1 = ... (은닉층 기울기, 연쇄법칙)
```

---

### 3.6 경사하강법 (Gradient Descent)

```
    손실 L
      |    \
      |     \
      |      o -> 현재 위치
      |       \
      |        \-> 기울기 방향
      |         \
      |----------\---- w
                  * 최저점
```

업데이트 규칙:
```
w = w - lr * (dL/dw)
```

---

### 3.7 학습 하이퍼파라미터

#### 학습률 (Learning Rate)

| 학습률 | 결과 |
|--------|------|
| 너무 작음 | 수렴이 느림 |
| 적절함 | 안정적 수렴 |
| 너무 큼 | 발산, 불안정 |

#### 에폭 (Epoch)

- 전체 데이터를 한 번 학습 = 1 epoch
- 너무 적으면: 학습 부족
- 너무 많으면: 과적합

#### 배치 크기 (Batch Size)

| 배치 크기 | 특징 |
|----------|------|
| 1 (SGD) | 빠른 업데이트, 불안정 |
| 32~256 | 일반적 선택 |
| 전체 (Batch GD) | 안정적, 느림 |

---

### 3.8 2층 신경망 클래스 구현

```python
class SimpleNeuralNetwork:
    """
    2층 신경망 (입력 -> 은닉 -> 출력)
    순전파, 역전파, 학습 구현
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate

        # 가중치 초기화 (He 초기화 for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 학습 기록
        self.loss_history = []
```

---

```python
    def forward(self, X):
        """순전파"""
        # 은닉층: z1 = XW1 + b1, a1 = ReLU(z1)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # 출력층: z2 = a1W2 + b2, a2 = Sigmoid(z2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """Binary Cross-Entropy 손실"""
        epsilon = 1e-8
        loss = -np.mean(
            y_true * np.log(y_pred + epsilon) +
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss
```

---

```python
    def backward(self, X, y_true):
        """역전파: 기울기 계산"""
        m = X.shape[0]  # 샘플 수

        # 출력층 기울기
        dz2 = self.a2 - y_true  # Sigmoid + BCE 미분 결과
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # 은닉층 기울기 (연쇄 법칙)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2):
        """경사하강법으로 가중치 업데이트"""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
```

---

```python
    def train(self, X, y, epochs=1000, verbose=True):
        """학습 루프"""
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)

            # 손실 계산
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # 역전파
            dW1, db1, dW2, db2 = self.backward(X, y)

            # 가중치 업데이트
            self.update(dW1, db1, dW2, db2)

            # 진행 상황 출력
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                accuracy = self.evaluate(X, y)
                print(f"  Epoch {epoch+1:4d}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")

        return self.loss_history

    def predict(self, X, threshold=0.5):
        """예측 (이진 분류)"""
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)

    def evaluate(self, X, y):
        """정확도 계산"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
```

---

### 3.9 XOR 문제 학습

XOR 문제는 선형 분리가 불가능하여 단일 퍼셉트론으로는 해결할 수 없음

```
(0, 0) -> 0
(0, 1) -> 1
(1, 0) -> 1
(1, 1) -> 0
```

신경망을 사용하면 비선형 결정 경계를 학습할 수 있음

---

```python
# XOR 데이터
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# 신경망 생성 및 학습
np.random.seed(42)
nn_xor = SimpleNeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.5
)

print("학습 시작...")
print(f"  구조: 입력(2) -> 은닉(4) -> 출력(1)")
print(f"  학습률: 0.5")
print(f"  에포크: 2000")

loss_history = nn_xor.train(X_xor, y_xor, epochs=2000, verbose=True)
```

---

```python
# 결과 확인
print("\n학습 결과:")
y_pred_prob = nn_xor.forward(X_xor)
y_pred = nn_xor.predict(X_xor)

for i in range(len(X_xor)):
    print(f"  입력: {X_xor[i]} -> 예측: {y_pred_prob[i, 0]:.4f} -> 분류: {y_pred[i, 0]} (정답: {y_xor[i, 0]})")

print(f"\n최종 정확도: {nn_xor.evaluate(X_xor, y_xor):.2%}")
```

#### 결과 예시

```
입력: [0 0] -> 예측: 0.0523 -> 분류: 0 (정답: 0)
입력: [0 1] -> 예측: 0.9412 -> 분류: 1 (정답: 1)
입력: [1 0] -> 예측: 0.9389 -> 분류: 1 (정답: 1)
입력: [1 1] -> 예측: 0.0612 -> 분류: 0 (정답: 0)

최종 정확도: 100.00%
```

---

### 3.10 학습 곡선 시각화

```python
# 손실 곡선 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('XOR 학습 손실 곡선')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 결정 경계 시각화
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
Z = nn_xor.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
plt.colorbar(label='Probability')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.ravel(),
            cmap='RdYlBu', s=200, edgecolors='black', linewidths=2)
plt.title('XOR 결정 경계')
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.show()
```

---

### 3.11 Digits 이진 분류 실습

```python
# 이진 분류 데이터 준비 (숫자 0 vs 나머지)
y_binary = (y_digits == 0).astype(int).reshape(-1, 1)

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"[데이터 분할]")
print(f"  학습: {len(X_train)}개, 테스트: {len(X_test)}개")
print(f"  학습 클래스 비율: {y_train.mean():.2%} (숫자 0)")
```

---

```python
# 신경망 생성 및 학습
np.random.seed(42)
nn_digits = SimpleNeuralNetwork(
    input_size=64,
    hidden_size=32,
    output_size=1,
    learning_rate=0.1
)

print("\n학습 시작...")
print(f"  구조: 입력(64) -> 은닉(32) -> 출력(1)")
loss_history_digits = nn_digits.train(X_train, y_train, epochs=500, verbose=True)

# 평가
train_acc = nn_digits.evaluate(X_train, y_train)
test_acc = nn_digits.evaluate(X_test, y_test)
print(f"\n학습 정확도: {train_acc:.2%}")
print(f"테스트 정확도: {test_acc:.2%}")
```

---

### 3.12 학습률 비교 실험

```python
learning_rates = [0.01, 0.1, 0.5, 1.0]
loss_histories = {}

print("학습률별 XOR 학습:")
for lr in learning_rates:
    np.random.seed(42)
    nn = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=4,
        output_size=1,
        learning_rate=lr
    )
    loss_histories[lr] = nn.train(X_xor, y_xor, epochs=500, verbose=False)
    final_acc = nn.evaluate(X_xor, y_xor)
    print(f"  lr={lr}: 최종 손실={loss_histories[lr][-1]:.4f}, 정확도={final_acc:.2%}")
```

#### 결과 해설

| 학습률 | 최종 손실 | 정확도 | 비고 |
|--------|----------|--------|------|
| 0.01 | 0.6931 | 50% | 너무 느림, 미수렴 |
| 0.1 | 0.2341 | 75% | 느린 수렴 |
| 0.5 | 0.0156 | 100% | 적절 |
| 1.0 | 0.0089 | 100% | 빠른 수렴 |

---

### 3.13 가중치 초기화 방법

```python
def initialize_weights(input_size, output_size, method='xavier'):
    """
    가중치 초기화 방법
    """
    if method == 'zero':
        # 절대 사용 금지! 모든 뉴런이 같은 값을 학습
        return np.zeros((input_size, output_size))

    elif method == 'random':
        # 간단한 랜덤 초기화
        return np.random.randn(input_size, output_size) * 0.01

    elif method == 'xavier':
        # Xavier (Glorot) 초기화: Sigmoid, Tanh에 적합
        return np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

    elif method == 'he':
        # He 초기화: ReLU에 적합
        return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
```

| 방법 | 적합한 활성화 함수 | 특징 |
|------|------------------|------|
| Zero | 사용 금지 | 모든 뉴런이 동일하게 학습됨 |
| Random | - | 단순하지만 불안정함 |
| Xavier | Sigmoid, Tanh | 분산 유지 |
| He | ReLU | ReLU에 최적화됨 |

---

## 20차시 핵심 정리

### 인공 뉴런

| 항목 | 내용 |
|------|------|
| 구조 | z = Sigma(wi*xi) + b, y = f(z) |
| 가중치(W) | 입력의 중요도 |
| 편향(b) | 활성화 기준점 |
| 활성화 함수 | 비선형성 추가 |

### 활성화 함수

| 함수 | 용도 |
|------|------|
| Sigmoid | 이진 분류 출력층, 0~1 |
| ReLU | 은닉층 표준, max(0, z) |
| Softmax | 다중 분류 출력층, 합=1 |

### 신경망 구조

| 항목 | 내용 |
|------|------|
| 입력층 | 특성 수만큼 뉴런 |
| 은닉층 | 패턴 학습, ReLU 활성화 |
| 출력층 | 문제 유형에 맞는 활성화 |
| 파라미터 수 | (이전 노드 x 현재 노드) + 편향 |

### 학습 과정

| 단계 | 내용 |
|------|------|
| 순전파 | 입력 -> 출력 계산 |
| 손실 계산 | 예측과 정답의 차이 |
| 역전파 | 손실의 기울기 계산 (연쇄 법칙) |
| 업데이트 | W = W - lr * gradient |

---

## 다음 차시 예고

### 21차시: 딥러닝 실습 - MLP 품질 예측

학습 내용:
- Keras/TensorFlow 사용
- Sequential 모델 구축
- 제조 품질 예측 실습
- 학습 곡선 분석

Keras를 사용하면 NumPy로 직접 구현한 것보다 훨씬 간편하게 신경망을 구축할 수 있음
