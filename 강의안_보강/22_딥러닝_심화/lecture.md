# 22차시: 딥러닝 심화

## 학습 목표

1. **CNN**(합성곱 신경망)의 구조와 원리를 이해함
2. **RNN**(순환 신경망)의 특징과 활용을 파악함
3. **고급 아키텍처**의 개념을 살펴봄

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | CNN (합성곱 신경망) |
| 대주제 2 | 10분 | RNN (순환 신경망) |
| 대주제 3 | 8분 | 고급 아키텍처 |
| 정리 | 2분 | 핵심 요약 |

---

## 지난 시간 복습

- **Keras로 MLP 구현**: Sequential, Dense, Dropout
- **학습 개선**: EarlyStopping, BatchNormalization
- **품질 예측**: 센서 데이터 -> 불량 여부 분류

**오늘**: MLP를 넘어서 다양한 딥러닝 아키텍처 탐구

---

# 대주제 1: CNN (합성곱 신경망)

## 1.1 왜 CNN인가?

**이미지 처리의 한계**:
- MLP로 이미지 처리 시 파라미터 폭발 발생
- 28x28 이미지 -> 784개 입력
- 첫 은닉층 128노드 -> 100,000개 파라미터!

**CNN의 해결책**:
- 지역적 패턴 학습 (필터)
- 파라미터 공유
- 위치 불변성

---

## 1.2 CNN 핵심 개념

```
[입력 이미지]
    |
[합성곱층 Conv2D] <- 필터로 특징 추출
    |
[풀링층 MaxPooling] <- 크기 축소
    |
[평탄화 Flatten]
    |
[완전연결층 Dense]
    |
[출력]
```

---

## 1.3 합성곱(Convolution) 연산

```
필터 (3x3):        입력 이미지 일부:
[1  0 -1]          [10 20 30]
[1  0 -1]    *     [40 50 60]  = 합성곱 결과
[1  0 -1]          [70 80 90]

계산:
1x10 + 0x20 + (-1)x30 +
1x40 + 0x50 + (-1)x60 +
1x70 + 0x80 + (-1)x90 = -60
```

---

## 실습 코드: 합성곱 연산 이해

```python
import numpy as np
import matplotlib.pyplot as plt

def convolution_2d(image, kernel):
    """
    2D 합성곱 연산 (단순 구현)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# 예시 이미지 (5x5)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# 수직 엣지 필터
vertical_edge = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

# 수평 엣지 필터
horizontal_edge = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=float)

print("입력 이미지 (5x5):")
print(image)

print("\n수직 엣지 필터:")
print(vertical_edge)

output_v = convolution_2d(image, vertical_edge)
print("\n수직 엣지 적용 결과:")
print(output_v)

output_h = convolution_2d(image, horizontal_edge)
print("\n수평 엣지 적용 결과:")
print(output_h)
```

---

## 1.4 필터의 역할

| 필터 유형 | 학습하는 패턴 |
|----------|--------------|
| 수평 엣지 | 가로 경계선 |
| 수직 엣지 | 세로 경계선 |
| 대각 엣지 | 대각선 경계 |
| 코너 | 모서리 패턴 |

**CNN은 필터 값을 자동으로 학습!**

---

## 1.5 특징 맵 (Feature Map)

```
입력 이미지 (28x28)
    |
필터 1 적용 -> 특징 맵 1 (수평 엣지)
필터 2 적용 -> 특징 맵 2 (수직 엣지)
필터 3 적용 -> 특징 맵 3 (대각 패턴)
...
32개 필터 -> 32개 특징 맵
```

---

## 1.6 풀링 (Pooling)

**목적**: 크기 축소, 계산량 감소, 위치 불변성

```
2x2 Max Pooling:

[1 3 | 2 4]      [3 | 4]
[5 6 | 7 8]  ->  [7 | 8]

4x4 -> 2x2로 축소
```

**최댓값을 선택** -> 가장 강한 특징 유지

---

## 실습 코드: MaxPooling 이해

```python
def max_pooling_2d(image, pool_size=2):
    """
    2D Max Pooling (단순 구현)
    """
    h, w = image.shape
    output_h = h // pool_size
    output_w = w // pool_size
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output

# 예시
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print("특징 맵 (4x4):")
print(feature_map)

pooled = max_pooling_2d(feature_map, pool_size=2)
print("\nMax Pooling (2x2) 결과:")
print(pooled)

print("\n설명:")
print("  - 왼쪽 상단 [1,3,5,6] -> max = 6")
print("  - 오른쪽 상단 [2,4,7,8] -> max = 8")
print("  - 크기: 4x4 -> 2x2 (1/4로 축소)")
```

---

## 1.7 Keras에서 CNN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential([
    # 합성곱 블록 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # 합성곱 블록 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 분류기
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## 1.8 Conv2D 파라미터

```python
Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
       padding='same', strides=(1, 1))
```

| 파라미터 | 의미 |
|---------|------|
| filters | 필터 개수 (출력 채널) |
| kernel_size | 필터 크기 (3x3) |
| padding | 'same': 크기 유지, 'valid': 줄어듦 |
| strides | 필터 이동 간격 |

---

## 실습 코드: CNN 모델 예시

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D,
    LSTM, GRU, SimpleRNN
)

# 이미지 분류 CNN
cnn_model = Sequential([
    # 합성곱 블록 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # 합성곱 블록 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 합성곱 블록 3
    Conv2D(64, (3, 3), activation='relu'),

    # 분류기
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

print("CNN 모델 구조:")
cnn_model.summary()

# 파라미터 계산 설명
print("\n파라미터 계산:")
print("  Conv2D(32, 3x3): 1 x 3 x 3 x 32 + 32 = 320")
print("  Conv2D(64, 3x3): 32 x 3 x 3 x 64 + 64 = 18,496")
print("  Conv2D(64, 3x3): 64 x 3 x 3 x 64 + 64 = 36,928")
print("  Dense(64): 576 x 64 + 64 = 36,928")
print("  Dense(10): 64 x 10 + 10 = 650")
```

---

## 1.9 CNN 활용 분야

| 분야 | 응용 |
|-----|------|
| 제조업 | **불량 이미지 검출**, 표면 결함 탐지 |
| 의료 | X-ray, MRI 분석, 세포 분류 |
| 자율주행 | 객체 인식, 차선 감지 |
| 보안 | 얼굴 인식, 지문 인식 |

---

## 1.10 제조업 CNN 예시

```python
# 불량 이미지 분류 모델
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])
```

---

## 1.11 CNN 아키텍처 진화

| 모델 | 년도 | 특징 |
|-----|-----|------|
| LeNet | 1998 | 최초의 CNN |
| AlexNet | 2012 | ImageNet 우승, 딥러닝 부흥 |
| VGG | 2014 | 깊은 구조 (16-19층) |
| ResNet | 2015 | Skip Connection |
| EfficientNet | 2019 | 효율적인 스케일링 |

---

## 1.12 전이학습 (Transfer Learning)

**아이디어**: 미리 학습된 모델 재활용

```python
from tensorflow.keras.applications import VGG16

# 사전 학습된 VGG16 로드
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))

# 가중치 동결
base_model.trainable = False

# 새로운 분류기 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')  # 불량 분류
])
```

---

## 1.13 전이학습 장점

| 항목 | 직접 학습 | 전이학습 |
|-----|----------|----------|
| 데이터 필요량 | 수만~수십만 | 수백~수천 |
| 학습 시간 | 수 시간~일 | 수 분~시간 |
| 성능 | 데이터 의존 | 높은 기본 성능 |

**적은 데이터로도 좋은 성능!**

---

# 대주제 2: RNN (순환 신경망)

## 2.1 왜 RNN인가?

**MLP의 한계**:
- 입력이 고정 크기
- 순서 정보 무시

**시계열/순차 데이터**:
- 센서 값의 **시간 순서** 중요
- 텍스트의 **단어 순서** 중요
- 이전 상태가 현재에 영향

---

## 2.2 RNN 핵심 개념

```
    t=0      t=1      t=2      t=3
     |        |        |        |
    [h0] -> [h1] -> [h2] -> [h3] -> 출력
     ^        ^        ^        ^
    x0       x1       x2       x3
```

**은닉 상태(h)가 다음 시점으로 전달**

---

## 2.3 RNN 수식

```
h_t = tanh(W_hh x h_{t-1} + W_xh x x_t + b)
y_t = W_hy x h_t
```

| 기호 | 의미 |
|-----|------|
| h_t | 현재 은닉 상태 |
| h_{t-1} | 이전 은닉 상태 |
| x_t | 현재 입력 |
| W | 가중치 행렬 |

---

## 실습 코드: RNN 구조 이해

```python
class SimpleRNNCell:
    """
    단순 RNN 셀 (NumPy 구현)
    """
    def __init__(self, input_size, hidden_size):
        # 가중치 초기화
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        """
        순전파
        h_t = tanh(Wxh * x + Whh * h_prev + bh)
        """
        h = np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h

    def process_sequence(self, X):
        """
        전체 시퀀스 처리
        X: (seq_len, input_size)
        """
        seq_len = X.shape[0]
        h = np.zeros((1, self.hidden_size))  # 초기 은닉 상태

        hidden_states = []
        for t in range(seq_len):
            x_t = X[t:t+1, :]  # (1, input_size)
            h = self.forward(x_t, h)
            hidden_states.append(h.copy())

        return np.vstack(hidden_states)

# RNN 예시
np.random.seed(42)
rnn_cell = SimpleRNNCell(input_size=3, hidden_size=4)

# 시퀀스 입력 (5 시점, 3 특성)
X_seq = np.random.randn(5, 3)
print("입력 시퀀스 (5 시점 x 3 특성):")
print(X_seq.round(2))

# 시퀀스 처리
hidden_states = rnn_cell.process_sequence(X_seq)
print("\n은닉 상태 (5 시점 x 4 은닉):")
print(hidden_states.round(3))

print("\n설명:")
print("  - 각 시점의 은닉 상태가 다음 시점으로 전달됨")
print("  - 마지막 은닉 상태에 전체 시퀀스 정보가 누적됨")
```

---

## 2.4 기본 RNN 문제점

**기울기 소실 (Vanishing Gradient)**:
- 긴 시퀀스에서 먼 과거 정보 손실
- 역전파 시 기울기가 점점 작아짐

**해결책**:
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

---

## 2.5 LSTM 구조

```
                    +------------+
    forget gate --> |   Cell     | <-- input gate
                    |   State    |
    output gate --> +-----+------+
                          |
                    [hidden state]
```

**게이트로 정보 흐름 제어**

---

## 2.6 LSTM 셀 개념

```
LSTM (Long Short-Term Memory) 구조:

+---------------------------------------------+
|                Cell State (C)               |
|        (장기 기억 - 컨베이어 벨트)           |
+---------------------------------------------+
       ^ forget    ^ input         | output
       |           |               v
+----------+ +-----------+ +---------------+
| Forget   | |  Input    | |    Output     |
|  Gate    | |   Gate    | |     Gate      |
| (버릴것) | | (저장할것)| |  (출력할것)   |
+----------+ +-----------+ +---------------+
       ^           ^               ^
       +-----------+---------------+
                   |
            h(t-1), x(t)

게이트 수식:
- Forget: f = sigmoid(Wf x [h, x] + bf)
- Input:  i = sigmoid(Wi x [h, x] + bi)
- Output: o = sigmoid(Wo x [h, x] + bo)
- Cell:   C = f*C + i*tanh(Wc x [h, x] + bc)
- Hidden: h = o*tanh(C)
```

---

## 2.7 LSTM 게이트

| 게이트 | 역할 |
|-------|------|
| Forget Gate | 이전 정보 중 버릴 것 결정 |
| Input Gate | 새 정보 중 저장할 것 결정 |
| Output Gate | 출력할 정보 결정 |

**셀 상태**: 장기 기억 저장소

---

## 2.8 Keras에서 LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # LSTM 층 (시퀀스 입력)
    LSTM(64, input_shape=(sequence_length, n_features),
         return_sequences=True),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    # 출력층
    Dense(1)  # 다음 값 예측
])

model.compile(optimizer='adam', loss='mse')
```

---

## 2.9 LSTM 파라미터

```python
LSTM(units=64, return_sequences=True, dropout=0.2)
```

| 파라미터 | 의미 |
|---------|------|
| units | 은닉 상태 크기 |
| return_sequences | True: 모든 시점 출력 |
| dropout | 입력 드롭아웃 비율 |
| recurrent_dropout | 순환 드롭아웃 비율 |

---

## 2.10 GRU (Gated Recurrent Unit)

**LSTM 간소화 버전**:
- 게이트 2개 (Reset, Update)
- 파라미터 적음, 빠른 학습
- LSTM과 비슷한 성능

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(64, input_shape=(seq_len, n_features)),
    Dense(1)
])
```

---

## 실습 코드: Keras RNN 모델 예시

```python
# 시계열 예측 LSTM
lstm_model = Sequential([
    # 첫 번째 LSTM 층 (시퀀스 출력)
    LSTM(64, input_shape=(24, 5), return_sequences=True),
    Dropout(0.2),

    # 두 번째 LSTM 층
    LSTM(32, return_sequences=False),
    Dropout(0.2),

    # 출력층
    Dense(16, activation='relu'),
    Dense(1)  # 다음 값 예측
])

print("LSTM 모델 구조:")
lstm_model.summary()

print("\n입력 형태 설명:")
print("  input_shape=(24, 5)")
print("  - 24: 시퀀스 길이 (예: 24시간)")
print("  - 5: 특성 수 (예: 온도, 압력, 속도, 습도, 진동)")

print("\nreturn_sequences 설명:")
print("  - True: 모든 시점의 출력 반환 (24, 64)")
print("  - False: 마지막 시점만 반환 (32,)")

# GRU 모델
gru_model = Sequential([
    GRU(64, input_shape=(24, 5), return_sequences=True),
    Dropout(0.2),
    GRU(32),
    Dense(1)
])

print("\nGRU 모델 구조:")
gru_model.summary()

print("\nLSTM vs GRU 파라미터 비교:")
lstm_params = lstm_model.count_params()
gru_params = gru_model.count_params()
print(f"  LSTM: {lstm_params:,} 파라미터")
print(f"  GRU:  {gru_params:,} 파라미터")
print(f"  차이: GRU가 {(lstm_params - gru_params):,} 적음")
```

---

## 2.11 RNN 활용 분야

| 분야 | 응용 |
|-----|------|
| 제조업 | **시계열 예측**, 설비 이상 탐지 |
| 금융 | 주가 예측, 거래 패턴 분석 |
| 자연어 | 번역, 챗봇, 감성 분석 |
| 음성 | 음성 인식, 음악 생성 |

---

## 2.12 제조업 LSTM 예시

```python
# 시계열 생산량 예측
model = Sequential([
    LSTM(128, input_shape=(24, 5),  # 24시간, 5개 특성
         return_sequences=True),
    Dropout(0.2),

    LSTM(64),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)  # 다음 시점 생산량
])

model.compile(optimizer='adam', loss='mse',
              metrics=['mae'])
```

---

## 2.13 시퀀스 데이터 생성

```python
def create_sequences(data, seq_length):
    """
    시계열 데이터를 시퀀스로 변환
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 예시 시계열 데이터
np.random.seed(42)
time_series = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)

print("원본 시계열 데이터:")
print(f"  길이: {len(time_series)}")
print(f"  처음 10개: {time_series[:10].round(2)}")

# 시퀀스 생성
seq_length = 10
X_seq, y_seq = create_sequences(time_series, seq_length)

print(f"\n시퀀스 생성 (seq_length={seq_length}):")
print(f"  X shape: {X_seq.shape}")
print(f"  y shape: {y_seq.shape}")

print("\n예시:")
print(f"  X[0] (과거 10개): {X_seq[0].round(2)}")
print(f"  y[0] (다음 1개): {y_seq[0]:.2f}")

# RNN 입력 형태로 변환
X_rnn = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
print(f"\nRNN 입력 형태: {X_rnn.shape}")
print("  (samples, timesteps, features)")
```

---

## 2.14 LSTM vs 전통 시계열

| 항목 | ARIMA | LSTM |
|-----|-------|------|
| 다변량 처리 | 어려움 | 쉬움 |
| 비선형 패턴 | 제한적 | 강함 |
| 해석 | 쉬움 | 어려움 |
| 데이터 필요량 | 적음 | 많음 |

---

# 대주제 3: 고급 아키텍처

## 3.1 딥러닝 아키텍처 발전

```
1998: LeNet (CNN 시작)
    |
2012: AlexNet (딥러닝 부흥)
    |
2014: VGG, GoogLeNet
    |
2015: ResNet (Skip Connection)
    |
2017: Transformer (Attention)
    |
2022: GPT-4, Diffusion Models
```

---

## 3.2 ResNet: Skip Connection

**문제**: 깊은 네트워크 학습 어려움

**해결**: 입력을 출력에 더하기

```
    x ------------------+
    |                   |
[Conv] -> [BN] -> [ReLU] -> [+] -> y
    |                    ^
[Conv] -> [BN] ----------+

y = F(x) + x  (Residual)
```

---

## 실습 코드: ResNet Skip Connection 개념

```python
print("""
ResNet의 핵심: Skip Connection (Residual Learning)

일반 네트워크:
    x -> [Conv] -> [BN] -> [ReLU] -> [Conv] -> [BN] -> y
    y = F(x)

ResNet:
    x -> [Conv] -> [BN] -> [ReLU] -> [Conv] -> [BN] -> (+) -> [ReLU] -> y
                                              ^
    x ---------------------------------------------+
    y = F(x) + x  (잔차 학습)

효과:
1. 기울기가 직접 전달 -> 깊은 네트워크 학습 가능
2. 152층까지도 학습 가능
3. 최소한 입력만큼은 출력 (성능 하한 보장)
""")

# Skip Connection 시뮬레이션
def residual_block(x, F):
    """잔차 블록 시뮬레이션"""
    return F + x  # y = F(x) + x

x = 10  # 입력
F = 2   # 변환 (배워야 할 것)

# 일반 네트워크: y = F(x) = 12를 직접 학습
# ResNet: F(x) = 2만 학습하면 y = 2 + 10 = 12

print("수치 예시:")
print(f"  입력 x: {x}")
print(f"  변환 F(x): {F}")
print(f"  출력 y = F(x) + x: {residual_block(x, F)}")
print("\n  -> 작은 변화만 학습하면 되므로 더 쉬움!")
```

---

## 3.3 ResNet 효과

| 네트워크 깊이 | 일반 CNN | ResNet |
|-------------|----------|--------|
| 20층 | 학습 가능 | 학습 가능 |
| 50층 | 어려움 | 학습 가능 |
| 152층 | 불가능 | 학습 가능 |

**기울기가 직접 전달 -> 깊은 네트워크 가능**

---

## 3.4 Attention 메커니즘

**아이디어**: 중요한 부분에 집중

```
입력: "나는 사과를 먹었다"

"먹었다"를 예측할 때:
- "나는" -> 낮은 attention
- "사과를" -> 높은 attention <-
```

**문맥에 따라 가중치 동적 결정**

---

## 실습 코드: Attention 메커니즘 개념

```python
def simple_attention(query, keys, values):
    """
    단순 Attention 예시

    query: 현재 질문/상태 (1, d)
    keys: 비교 대상들 (n, d)
    values: 가져올 값들 (n, d)
    """
    # 1. Score 계산 (내적)
    scores = query @ keys.T  # (1, n)

    # 2. Softmax로 가중치 변환
    weights = np.exp(scores) / np.sum(np.exp(scores))

    # 3. 가중 평균
    output = weights @ values  # (1, d)

    return output, weights

# 예시: 번역에서 Attention
# "나는 사과를 먹었다" -> "I ate an apple"
# "ate" 예측 시 어디에 집중?

np.random.seed(42)

# 단어 임베딩 (4차원)
words = {
    '나는': np.array([0.1, 0.2, 0.3, 0.1]),
    '사과를': np.array([0.8, 0.1, 0.2, 0.9]),
    '먹었다': np.array([0.2, 0.9, 0.8, 0.2])
}

keys = np.vstack(list(words.values()))  # (3, 4)
values = keys.copy()

# "ate"를 예측할 때의 query (동사 관련)
query = np.array([[0.3, 0.8, 0.7, 0.3]])  # 동사 성격

output, weights = simple_attention(query, keys, values)

print("Attention 예시 (번역):")
print("  입력: '나는 사과를 먹었다'")
print("  예측 대상: 'ate'")
print("\n  Attention 가중치:")
for word, w in zip(words.keys(), weights[0]):
    bar = '*' * int(w * 20)
    print(f"    {word}: {w:.3f} {bar}")

print("\n  -> '먹었다'에 가장 높은 attention!")
```

---

## 3.5 Transformer

**특징**:
- RNN 없이 Attention만으로 시퀀스 처리
- 병렬 처리 가능 -> 빠른 학습
- GPT, BERT의 기반

**구조**:
- Self-Attention
- Multi-Head Attention
- Feed-Forward Network

---

## 3.6 Transformer 아키텍처

```
[입력 임베딩]
     |
[Self-Attention]  <- 문맥 파악
     |
[Feed-Forward]
     |
(반복 N번)
     |
[출력]
```

---

## 3.7 대형 언어 모델 (LLM)

| 모델 | 파라미터 수 | 특징 |
|-----|-----------|------|
| BERT | 3억 | 양방향 문맥 |
| GPT-3 | 1,750억 | 생성 능력 |
| GPT-4 | 1조+ (추정) | 멀티모달 |
| LLaMA | 70억~700억 | 오픈소스 |

---

## 3.8 Diffusion Models

**이미지 생성 모델**:
- DALL-E 2
- Stable Diffusion
- Midjourney

**원리**:
1. 이미지에 노이즈 추가
2. 노이즈 제거 과정 학습
3. 노이즈 -> 이미지 생성

---

## 3.9 제조업 적용 가능성

| 기술 | 제조업 활용 |
|-----|------------|
| CNN | 불량 이미지 검출, 표면 검사 |
| RNN/LSTM | 시계열 예측, 이상 탐지 |
| Transformer | 설비 로그 분석, 문서 처리 |
| Diffusion | 합성 불량 이미지 생성 (데이터 증강) |

---

## 3.10 AutoML과 NAS

**AutoML**: 자동 머신러닝
- 하이퍼파라미터 자동 탐색
- 특성 엔지니어링 자동화

**NAS**: Neural Architecture Search
- 최적 아키텍처 자동 탐색
- EfficientNet이 NAS로 설계됨

---

## 3.11 Edge AI

**개념**: 기기에서 직접 AI 실행

```
[클라우드 AI]         [Edge AI]
센서 -> 클라우드 -> 결과    센서 -> 기기 -> 결과
    (지연 발생)             (즉시)
```

**제조업 적용**:
- 실시간 불량 검출
- 네트워크 없이 동작
- 데이터 보안

---

## 3.12 딥러닝 프레임워크 비교

| 프레임워크 | 특징 |
|-----------|------|
| TensorFlow/Keras | 산업 표준, 배포 용이 |
| PyTorch | 연구 표준, 직관적 |
| JAX | 고성능, 함수형 |
| ONNX | 프레임워크 간 호환 |

---

## 3.13 전이학습 예시 코드

```python
# 전이학습 코드 예시

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# 1. 사전 학습된 모델 로드
base_model = VGG16(
    weights='imagenet',      # ImageNet 가중치
    include_top=False,       # 분류기 제외
    input_shape=(224, 224, 3)
)

# 2. 가중치 동결 (학습 안 함)
base_model.trainable = False

# 3. 새 분류기 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])

# 4. 컴파일 및 학습
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 학습 (적은 데이터로 가능!)
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

print("전이학습 장점:")
print("  - 수백 장의 이미지로도 학습 가능")
print("  - 빠른 학습 (몇 분 이내)")
print("  - 높은 기본 성능 (ImageNet 지식 활용)")
```

---

## 3.14 학습 방향 제안

```
기초
+-- MLP (Dense)        <- 20차시에서 완료!
+-- CNN (이미지)       <- 필요시 심화
+-- RNN/LSTM (시계열)  <- 필요시 심화

심화
+-- Transformer (NLP)
+-- 전이학습
+-- 배포 (TensorFlow Serving)
```

---

## 3.15 아키텍처 비교

```
딥러닝 아키텍처 비교:

| 아키텍처    | 데이터 유형  | 핵심 연산        | 제조업 활용             |
|------------|-------------|-----------------|------------------------|
| MLP        | 정형        | 행렬 곱         | 품질 예측, 분류         |
| CNN        | 이미지      | 합성곱          | 불량 이미지 검출        |
| RNN/LSTM   | 시계열      | 순환 연결        | 생산량 예측, 이상 탐지   |
| Transformer| 시퀀스      | Self-Attention  | 로그 분석, 문서 처리    |

선택 기준:
1. 정형 데이터 (센서 값) -> MLP 또는 ML (RandomForest)
2. 이미지 데이터 -> CNN (전이학습 권장)
3. 시계열 데이터 -> LSTM/GRU
4. 텍스트 데이터 -> Transformer (BERT, GPT)
```

---

# 핵심 정리

## 아키텍처 선택 가이드

| 데이터 유형 | 권장 아키텍처 |
|-----------|--------------|
| 정형 데이터 (테이블) | MLP, ML |
| 이미지 | CNN |
| 시계열/순차 | LSTM, GRU |
| 텍스트 | Transformer, BERT |
| 다양한 입력 | 멀티모달 모델 |

---

## 오늘 배운 내용

1. **CNN (합성곱 신경망)**
   - 이미지 특징 자동 추출
   - Conv2D -> Pooling -> Dense
   - 전이학습으로 효율화

2. **RNN (순환 신경망)**
   - 시계열/순차 데이터 처리
   - LSTM, GRU로 장기 의존성 학습

3. **고급 아키텍처**
   - ResNet, Transformer, Diffusion
   - 제조업 적용 가능성

---

## 제조업 딥러닝 로드맵

```
1단계: 정형 데이터 (MLP)
   +-- 센서 데이터 품질 예측 (완료)

2단계: 이미지 데이터 (CNN)
   +-- 불량 이미지 분류

3단계: 시계열 데이터 (LSTM)
   +-- 생산량/품질 예측

4단계: 멀티모달
   +-- 센서 + 이미지 통합
```

---

## 다음 차시 예고

### [23차시] 모델 해석과 변수별 영향력 분석

- Feature Importance
- Permutation Importance
- SHAP 개념 소개
