# 23차시: 딥러닝 응용 - CNN 기반 이미지 불량 검출

## 학습 목표

1. **합성곱 신경망(CNN)**의 구조와 원리를 이해함
2. **Keras**로 CNN 모델을 구성함
3. **제품 외관 이미지** 기반 불량 분류를 수행함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 8분 | CNN 수학적 배경 |
| 대주제 2 | 10분 | Keras CNN 모델 구성 |
| 대주제 3 | 5분 | 이미지 불량 분류 실습 |
| 정리 | 2분 | 핵심 요약 |

---

## 지난 시간 복습

- **CNN 개념**: 합성곱층, 풀링층, 특징 맵
- **RNN/LSTM**: 시계열 데이터 처리
- **고급 아키텍처**: ResNet, Transformer

**오늘**: CNN을 활용한 실제 이미지 분류 실습

---

# 대주제 1: CNN 수학적 배경

## 1.1 CNN의 핵심 구조

CNN(Convolutional Neural Network)은 이미지 처리에 특화된 신경망임. MLP와 달리 공간적 구조를 유지하면서 특징을 추출함.

```
CNN 전체 구조:

[입력 이미지]
      |
      v
+------------------+
| 합성곱층 (Conv2D) | <- 필터로 특징 추출
+------------------+
      |
      v
+------------------+
| 풀링층 (Pooling)  | <- 공간 크기 축소
+------------------+
      |
      v
   (반복)
      |
      v
+------------------+
| 평탄화 (Flatten)  | <- 1차원 변환
+------------------+
      |
      v
+------------------+
| 완전연결 (Dense)  | <- 분류 수행
+------------------+
      |
      v
   [출력]
```

---

## 1.2 합성곱 연산의 수학적 정의

입력 이미지 $I$와 커널(필터) $K$의 2D 합성곱 연산:

$$
(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)
$$

| 기호 | 의미 |
|------|------|
| $I$ | 입력 이미지 (2D 행렬) |
| $K$ | 커널/필터 (작은 2D 행렬) |
| $(i,j)$ | 출력 특성 맵의 위치 |
| $(m,n)$ | 커널 내부의 상대 위치 |

**핵심 아이디어**: 커널을 이미지 위에서 슬라이딩하면서 지역적 패턴을 감지함

---

## 1.3 합성곱 연산 수동 계산

```python
import numpy as np

# 합성곱 연산 직접 구현
def convolution_2d_manual(image, kernel):
    """
    2D 합성곱 연산을 수동으로 계산함

    수식: (I * K)(i,j) = sum_m sum_n I(i+m, j+n) * K(m,n)
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    # 출력 크기 계산 (패딩 없음, 스트라이드 1)
    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # 현재 위치의 영역 추출
            region = image[i:i+ker_h, j:j+ker_w]
            # 요소별 곱셈 후 합산
            output[i, j] = np.sum(region * kernel)

    return output

# 예시: 3x3 입력, 2x2 커널
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

kernel = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

print("입력 이미지 (3x3):")
print(image)
print("\n커널 (2x2):")
print(kernel)

# 합성곱 연산 수행
output = convolution_2d_manual(image, kernel)
print("\n합성곱 결과 (2x2):")
print(output)

# 계산 과정 설명
print("\n계산 과정:")
print("  (0,0): 1*1 + 2*0 + 4*0 + 5*1 =", 1*1 + 2*0 + 4*0 + 5*1)
print("  (0,1): 2*1 + 3*0 + 5*0 + 6*1 =", 2*1 + 3*0 + 5*0 + 6*1)
print("  (1,0): 4*1 + 5*0 + 7*0 + 8*1 =", 4*1 + 5*0 + 7*0 + 8*1)
print("  (1,1): 5*1 + 6*0 + 8*0 + 9*1 =", 5*1 + 6*0 + 8*0 + 9*1)
```

---

## 1.4 특성 맵 크기 계산 공식

합성곱 연산 후 출력 크기를 계산하는 공식:

$$
O = \frac{W - K + 2P}{S} + 1
$$

| 기호 | 의미 | 설명 |
|------|------|------|
| $O$ | 출력 크기 | 특성 맵의 한 변 길이 |
| $W$ | 입력 크기 | 입력 이미지의 한 변 길이 |
| $K$ | 커널 크기 | 필터의 한 변 길이 |
| $P$ | 패딩 | 입력 주변에 추가하는 픽셀 수 |
| $S$ | 스트라이드 | 필터 이동 간격 |

---

## 1.5 출력 크기 계산 실습

```python
def calculate_output_size(W, K, P=0, S=1):
    """
    합성곱 출력 크기 계산

    공식: O = (W - K + 2P) / S + 1

    Parameters:
        W: 입력 크기
        K: 커널 크기
        P: 패딩 (기본값 0)
        S: 스트라이드 (기본값 1)

    Returns:
        O: 출력 크기
    """
    O = (W - K + 2*P) // S + 1
    return O

print("특성 맵 크기 계산 공식: O = (W - K + 2P) / S + 1")
print("=" * 50)

# 다양한 케이스 테스트
cases = [
    {"W": 28, "K": 3, "P": 0, "S": 1, "desc": "28x28 입력, 3x3 커널, 패딩 없음"},
    {"W": 28, "K": 3, "P": 1, "S": 1, "desc": "28x28 입력, 3x3 커널, same 패딩"},
    {"W": 28, "K": 3, "P": 0, "S": 2, "desc": "28x28 입력, 3x3 커널, 스트라이드 2"},
    {"W": 26, "K": 3, "P": 0, "S": 1, "desc": "26x26 입력, 3x3 커널"},
    {"W": 224, "K": 7, "P": 3, "S": 2, "desc": "224x224 입력, 7x7 커널 (ResNet 첫 층)"},
]

for case in cases:
    W, K, P, S = case["W"], case["K"], case["P"], case["S"]
    O = calculate_output_size(W, K, P, S)
    print(f"\n{case['desc']}:")
    print(f"  W={W}, K={K}, P={P}, S={S}")
    print(f"  O = ({W} - {K} + 2*{P}) / {S} + 1 = {O}")
```

---

## 1.6 패딩의 종류와 효과

```python
import numpy as np

def apply_padding(image, padding, mode='constant'):
    """
    이미지에 패딩 적용
    """
    return np.pad(image, padding, mode=mode, constant_values=0)

# 원본 이미지 (4x4)
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print("원본 이미지 (4x4):")
print(image)

# 패딩 1 적용
padded = apply_padding(image, padding=1)
print("\n패딩 1 적용 (6x6):")
print(padded.astype(int))

print("\n패딩 종류:")
print("  - 'valid': 패딩 없음, 출력 크기 감소")
print("  - 'same': 출력 크기 = 입력 크기 유지")
print("\nsame 패딩 계산:")
print("  3x3 커널의 경우: P = (K-1)/2 = (3-1)/2 = 1")
```

---

## 1.7 풀링 연산의 수학적 정의

Max Pooling 연산:

$$
y_{ij} = \max_{(m,n) \in R_{ij}} x_{mn}
$$

| 기호 | 의미 |
|------|------|
| $y_{ij}$ | 출력 위치 $(i,j)$의 값 |
| $R_{ij}$ | 풀링 영역 (예: 2x2) |
| $x_{mn}$ | 영역 내 입력값 |

**역할**:
- 공간 크기 축소 (계산량 감소)
- 위치 변화에 대한 강건성 (Translation Invariance)
- 가장 강한 특징 유지

---

## 1.8 풀링 연산 수동 계산

```python
def max_pooling_2d(image, pool_size=2, stride=2):
    """
    2x2 Max Pooling 연산

    수식: y_ij = max_{(m,n) in R_ij} x_mn
    """
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # 풀링 영역 추출
            region = image[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            # 최댓값 선택
            output[i, j] = np.max(region)

    return output

def avg_pooling_2d(image, pool_size=2, stride=2):
    """
    2x2 Average Pooling 연산
    """
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)

    return output

# 예시 특성 맵 (4x4)
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [2, 1, 3, 2]
], dtype=float)

print("입력 특성 맵 (4x4):")
print(feature_map.astype(int))

# Max Pooling
max_pooled = max_pooling_2d(feature_map)
print("\n2x2 Max Pooling 결과 (2x2):")
print(max_pooled.astype(int))

print("\n계산 과정:")
print("  좌상단 [1,3,5,6] -> max = 6")
print("  우상단 [2,4,7,8] -> max = 8")
print("  좌하단 [9,10,2,1] -> max = 10")
print("  우하단 [11,12,3,2] -> max = 12")

# Average Pooling
avg_pooled = avg_pooling_2d(feature_map)
print("\n2x2 Average Pooling 결과 (2x2):")
print(avg_pooled)
```

---

## 1.9 엣지 검출 필터 예시

```python
import matplotlib.pyplot as plt

# 다양한 엣지 검출 필터
filters = {
    "수직 엣지": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=float),

    "수평 엣지": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=float),

    "Sobel X": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float),

    "Sobel Y": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=float),

    "샤프닝": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=float),
}

print("CNN에서 사용하는 필터 예시:")
print("=" * 40)

for name, kernel in filters.items():
    print(f"\n{name} 필터:")
    print(kernel.astype(int))
```

---

## 1.10 필터 적용 시각화

```python
# 예시 이미지 생성 (흰색 사각형)
test_image = np.zeros((10, 10))
test_image[2:8, 2:8] = 1.0

print("테스트 이미지 (10x10, 중앙에 사각형):")
for row in test_image:
    print("".join(["#" if v > 0 else "." for v in row]))

# 필터 적용
for name, kernel in list(filters.items())[:2]:  # 수직, 수평 엣지만
    result = convolution_2d_manual(test_image, kernel)
    print(f"\n{name} 필터 적용 결과 (8x8):")
    print(np.round(result, 1))
```

---

## 1.11 CNN 파라미터 수 계산

```python
def calculate_conv_params(kernel_h, kernel_w, in_channels, out_channels):
    """
    Conv2D 층의 파라미터 수 계산

    공식: params = (K_h * K_w * C_in + 1) * C_out

    +1은 편향(bias)
    """
    weights = kernel_h * kernel_w * in_channels * out_channels
    biases = out_channels
    total = weights + biases
    return total

def calculate_dense_params(in_features, out_features):
    """
    Dense 층의 파라미터 수 계산

    공식: params = (n_in + 1) * n_out
    """
    weights = in_features * out_features
    biases = out_features
    return weights + biases

print("CNN 파라미터 수 계산")
print("=" * 50)

# 예시: Fashion-MNIST CNN
print("\n예시 CNN 구조 (Fashion-MNIST):")
print("-" * 50)

# 입력: 28x28x1
layers = [
    ("Conv2D(32, 3x3)", calculate_conv_params(3, 3, 1, 32)),
    ("Conv2D(64, 3x3)", calculate_conv_params(3, 3, 32, 64)),
    ("Conv2D(64, 3x3)", calculate_conv_params(3, 3, 64, 64)),
    ("Dense(64)", calculate_dense_params(576, 64)),  # 3x3x64 = 576
    ("Dense(10)", calculate_dense_params(64, 10)),
]

total_params = 0
for name, params in layers:
    print(f"  {name}: {params:,} 파라미터")
    total_params += params

print("-" * 50)
print(f"  총 파라미터: {total_params:,}")
```

---

# 대주제 2: Keras CNN 모델 구성

## 2.1 환경 설정 및 라이브러리 임포트

```python
import numpy as np
import matplotlib.pyplot as plt
import os

# TensorFlow 경고 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow 버전: {tf.__version__}")
print(f"Keras 버전: {keras.__version__}")
```

---

## 2.2 Fashion-MNIST 데이터셋 로드

Fashion-MNIST는 의류 이미지 분류 데이터셋으로, 제조업 외관 검사의 간소화된 예시로 활용 가능함.

```python
# 데이터 로드
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 클래스 이름 정의
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Fashion-MNIST 데이터셋 정보:")
print(f"  학습 데이터: {X_train.shape}")
print(f"  테스트 데이터: {X_test.shape}")
print(f"  클래스 수: {len(class_names)}")
print(f"  이미지 크기: {X_train.shape[1]}x{X_train.shape[2]} 픽셀")
print(f"  픽셀값 범위: {X_train.min()} ~ {X_train.max()}")

# 클래스별 샘플 수 확인
print("\n클래스별 학습 샘플 수:")
for i, name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"  {i}: {name}: {count:,}")
```

---

## 2.3 데이터 시각화

```python
# 샘플 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'{class_names[y_train[i]]}')
    ax.axis('off')

plt.suptitle('Fashion-MNIST 샘플 이미지', fontsize=14)
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# 클래스별 대표 이미지
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    # 해당 클래스의 첫 번째 이미지 찾기
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f'{i}: {class_names[i]}')
    ax.axis('off')

plt.suptitle('클래스별 대표 이미지', fontsize=14)
plt.tight_layout()
plt.savefig('fashion_mnist_classes.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2.4 데이터 전처리

CNN 입력을 위해 데이터 형태를 변환하고 정규화를 수행함.

```python
# 1. 정규화 (0~255 -> 0~1)
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

print("정규화 결과:")
print(f"  변환 전: {X_train.min()} ~ {X_train.max()}")
print(f"  변환 후: {X_train_normalized.min():.2f} ~ {X_train_normalized.max():.2f}")

# 2. 채널 차원 추가 (CNN 입력 형태: batch, height, width, channels)
X_train_cnn = X_train_normalized.reshape(-1, 28, 28, 1)
X_test_cnn = X_test_normalized.reshape(-1, 28, 28, 1)

print(f"\nCNN 입력 형태:")
print(f"  학습 데이터: {X_train_cnn.shape}")
print(f"  테스트 데이터: {X_test_cnn.shape}")
print(f"  형태 의미: (samples, height, width, channels)")
```

---

## 2.5 CNN 모델 아키텍처 설계

```
모델 구조 다이어그램:

입력 (28x28x1)
     |
     v
+------------------------+
| Conv2D(32, 3x3, relu)  |  -> (26x26x32)
| MaxPooling2D(2x2)      |  -> (13x13x32)
+------------------------+
     |
     v
+------------------------+
| Conv2D(64, 3x3, relu)  |  -> (11x11x64)
| MaxPooling2D(2x2)      |  -> (5x5x64)
+------------------------+
     |
     v
+------------------------+
| Conv2D(64, 3x3, relu)  |  -> (3x3x64)
+------------------------+
     |
     v
+------------------------+
| Flatten                |  -> (576,)
| Dense(64, relu)        |  -> (64,)
| Dropout(0.5)           |
| Dense(10, softmax)     |  -> (10,)
+------------------------+
     |
     v
출력 (10 클래스 확률)
```

---

## 2.6 기본 CNN 모델 생성

```python
def create_basic_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    기본 CNN 모델 생성

    구조:
    - Conv2D(32) -> MaxPooling
    - Conv2D(64) -> MaxPooling
    - Conv2D(64)
    - Flatten -> Dense(64) -> Dense(num_classes)
    """
    model = Sequential([
        # 첫 번째 합성곱 블록
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
               name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),

        # 두 번째 합성곱 블록
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),

        # 세 번째 합성곱 층
        Conv2D(64, (3, 3), activation='relu', name='conv3'),

        # 분류기
        Flatten(name='flatten'),
        Dense(64, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout'),
        Dense(num_classes, activation='softmax', name='output')
    ])

    return model

# 모델 생성
model_basic = create_basic_cnn()

# 모델 요약
print("기본 CNN 모델 구조:")
print("=" * 65)
model_basic.summary()
```

---

## 2.7 개선된 CNN 모델 (BatchNormalization 적용)

```python
def create_improved_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    개선된 CNN 모델 (BatchNormalization 포함)

    개선사항:
    - BatchNormalization: 학습 안정화
    - Dropout: 과적합 방지
    - 더 많은 필터
    """
    model = Sequential([
        # 첫 번째 합성곱 블록
        Conv2D(32, (3, 3), padding='same', activation='relu',
               input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 두 번째 합성곱 블록
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 분류기
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# 개선된 모델 생성
model_improved = create_improved_cnn()

print("\n개선된 CNN 모델 구조:")
print("=" * 65)
model_improved.summary()
```

---

## 2.8 모델 컴파일

```python
# 기본 모델 컴파일
model_basic.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # 정수 라벨 사용
    metrics=['accuracy']
)

print("모델 컴파일 설정:")
print("-" * 40)
print(f"  옵티마이저: Adam")
print(f"  손실 함수: sparse_categorical_crossentropy")
print(f"  평가 지표: accuracy")

print("\n손실 함수 선택 가이드:")
print("  - sparse_categorical_crossentropy: 정수 라벨 (0, 1, 2, ...)")
print("  - categorical_crossentropy: 원-핫 인코딩 라벨")
```

---

## 2.9 콜백 설정

```python
# 조기 종료 콜백
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 모델 체크포인트 콜백
checkpoint = ModelCheckpoint(
    'best_cnn_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

callbacks = [early_stop, checkpoint]

print("콜백 설정:")
print("-" * 40)
print("  EarlyStopping:")
print("    - monitor: val_loss")
print("    - patience: 5 (5 에포크 개선 없으면 종료)")
print("    - restore_best_weights: True")
print("\n  ModelCheckpoint:")
print("    - 최고 val_accuracy 모델 저장")
```

---

## 2.10 모델 학습

```python
# 모델 학습
print("모델 학습 시작...")
print("=" * 50)

history = model_basic.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n학습 완료!")
print(f"  실제 에포크 수: {len(history.history['loss'])}")
print(f"  최종 학습 정확도: {history.history['accuracy'][-1]:.4f}")
print(f"  최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")
```

---

## 2.11 학습 곡선 시각화

```python
def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    학습 곡선 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 손실 곡선
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 정확도 곡선
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# 학습 곡선 시각화
plot_learning_curves(history)

# 학습 곡선 해석
print("\n학습 곡선 해석 가이드:")
print("-" * 40)
print("  정상 학습: Train과 Val 모두 수렴")
print("  과적합: Train 감소, Val 증가/정체")
print("  과소적합: Train과 Val 모두 높은 손실")
```

---

## 2.12 테스트 데이터 평가

```python
# 테스트 데이터로 평가
test_loss, test_accuracy = model_basic.evaluate(X_test_cnn, y_test, verbose=0)

print("테스트 데이터 평가 결과:")
print("=" * 40)
print(f"  테스트 손실: {test_loss:.4f}")
print(f"  테스트 정확도: {test_accuracy:.2%}")

# 예측 수행
y_pred_prob = model_basic.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print(f"\n예측 결과:")
print(f"  총 테스트 샘플: {len(y_test):,}")
print(f"  정확한 예측: {np.sum(y_pred == y_test):,}")
print(f"  오분류: {np.sum(y_pred != y_test):,}")
```

---

## 2.13 분류 보고서

```python
# 분류 보고서 출력
print("분류 보고서:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))

# 클래스별 성능 분석
print("\n클래스별 성능 분석:")
print("-" * 40)
report = classification_report(y_test, y_pred, target_names=class_names,
                              output_dict=True)

# 가장 성능이 좋은/나쁜 클래스
performances = [(name, report[name]['f1-score']) for name in class_names]
performances.sort(key=lambda x: x[1], reverse=True)

print("\n성능 순위 (F1-score 기준):")
for rank, (name, f1) in enumerate(performances, 1):
    print(f"  {rank}. {name}: {f1:.3f}")
```

---

## 2.14 혼동 행렬 시각화

```python
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    혼동 행렬 시각화
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues')

    # 축 설정
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # 값 표시
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return cm

# 혼동 행렬 시각화
cm = plot_confusion_matrix(y_test, y_pred, class_names)

# 자주 혼동되는 클래스 쌍 분석
print("\n자주 혼동되는 클래스 쌍:")
print("-" * 40)

# 대각선 제외하고 가장 높은 값 찾기
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)

for _ in range(3):  # 상위 3개
    idx = np.unravel_index(np.argmax(cm_off_diag), cm_off_diag.shape)
    true_class = class_names[idx[0]]
    pred_class = class_names[idx[1]]
    count = cm_off_diag[idx]
    print(f"  {true_class} -> {pred_class}: {count}회")
    cm_off_diag[idx] = 0
```

---

## 2.15 오분류 샘플 분석

```python
def visualize_misclassifications(X, y_true, y_pred, class_names, n_samples=10):
    """
    오분류된 샘플 시각화
    """
    # 오분류 인덱스 찾기
    wrong_idx = np.where(y_true != y_pred)[0]

    # 랜덤하게 n_samples개 선택
    if len(wrong_idx) > n_samples:
        wrong_idx = np.random.choice(wrong_idx, n_samples, replace=False)

    # 시각화
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, (ax, idx) in enumerate(zip(axes.flat, wrong_idx)):
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        ax.axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig('misclassified_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

# 오분류 샘플 시각화
visualize_misclassifications(X_test, y_test, y_pred, class_names)

print("\n오분류 원인 분석:")
print("-" * 40)
print("  - 유사한 형태의 클래스 (Shirt vs T-shirt)")
print("  - 이미지 품질 문제 (흐릿함, 노이즈)")
print("  - 특이한 디자인/스타일")
```

---

# 대주제 3: 이미지 불량 분류 실습

## 3.1 제조업 시나리오 적용

Fashion-MNIST를 제조업 외관 검사 시나리오에 적용함.

```python
# 이진 분류로 변환: 의류 유형별 품질 분류 시뮬레이션
# 예: 신발류(5,7,9) vs 비신발류(나머지)로 분류

def create_binary_labels(y, positive_classes=[5, 7, 9]):
    """
    다중 클래스를 이진 클래스로 변환

    시나리오: 신발류 제품 검출
    - 양성(1): Sandal(5), Sneaker(7), Ankle boot(9)
    - 음성(0): 기타 의류
    """
    return np.isin(y, positive_classes).astype(int)

# 이진 라벨 생성
y_train_binary = create_binary_labels(y_train)
y_test_binary = create_binary_labels(y_test)

print("이진 분류 시나리오: 신발류 검출")
print("=" * 40)
print(f"  양성 클래스: Sandal, Sneaker, Ankle boot")
print(f"  음성 클래스: 기타 의류")
print(f"\n학습 데이터:")
print(f"  양성 (신발류): {np.sum(y_train_binary == 1):,}")
print(f"  음성 (기타): {np.sum(y_train_binary == 0):,}")
print(f"\n테스트 데이터:")
print(f"  양성 (신발류): {np.sum(y_test_binary == 1):,}")
print(f"  음성 (기타): {np.sum(y_test_binary == 0):,}")
```

---

## 3.2 이진 분류 CNN 모델

```python
def create_binary_cnn(input_shape=(28, 28, 1)):
    """
    이진 분류용 CNN 모델

    출력층: Dense(1, sigmoid) - 불량 확률
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # 이진 분류 손실 함수
        metrics=['accuracy']
    )

    return model

# 이진 분류 모델 생성 및 학습
model_binary = create_binary_cnn()

print("이진 분류 CNN 모델:")
model_binary.summary()

# 학습
history_binary = model_binary.fit(
    X_train_cnn, y_train_binary,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

# 평가
binary_loss, binary_acc = model_binary.evaluate(X_test_cnn, y_test_binary, verbose=0)
print(f"\n이진 분류 테스트 정확도: {binary_acc:.2%}")
```

---

## 3.3 불량 검출 시나리오 시뮬레이션

```python
# 불량 검출 시뮬레이션
# 특정 클래스를 "불량"으로 정의

def simulate_defect_detection(y_true, defect_class=6):
    """
    불량 검출 시뮬레이션

    시나리오: Shirt(6)를 불량 제품으로 가정
    - 불량(1): Shirt
    - 정상(0): 나머지
    """
    return (y_true == defect_class).astype(int)

# 불량 라벨 생성
y_train_defect = simulate_defect_detection(y_train, defect_class=6)
y_test_defect = simulate_defect_detection(y_test, defect_class=6)

print("불량 검출 시뮬레이션")
print("=" * 40)
print(f"  불량 클래스: Shirt (라벨 6)")
print(f"\n학습 데이터:")
print(f"  정상: {np.sum(y_train_defect == 0):,}")
print(f"  불량: {np.sum(y_train_defect == 1):,}")
print(f"  불량률: {np.mean(y_train_defect):.1%}")

# 불균형 데이터 문제 인식
print("\n주의: 불량 데이터 불균형!")
print("  - 실제 제조 환경에서는 불량률이 더 낮음 (1% 미만)")
print("  - 클래스 가중치 또는 오버샘플링 필요")
```

---

## 3.4 클래스 가중치 적용

```python
from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_defect),
    y=y_train_defect
)
class_weight_dict = dict(enumerate(class_weights))

print("클래스 가중치 (불균형 보정):")
print(f"  정상 (0): {class_weight_dict[0]:.3f}")
print(f"  불량 (1): {class_weight_dict[1]:.3f}")

# 불량 검출 모델 학습 (클래스 가중치 적용)
model_defect = create_binary_cnn()

history_defect = model_defect.fit(
    X_train_cnn, y_train_defect,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,  # 클래스 가중치 적용
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

# 평가
defect_loss, defect_acc = model_defect.evaluate(X_test_cnn, y_test_defect, verbose=0)
print(f"\n불량 검출 테스트 정확도: {defect_acc:.2%}")
```

---

## 3.5 불량 검출 성능 분석

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 예측
y_defect_pred_prob = model_defect.predict(X_test_cnn, verbose=0)
y_defect_pred = (y_defect_pred_prob > 0.5).astype(int).ravel()

# 성능 지표 계산
precision = precision_score(y_test_defect, y_defect_pred)
recall = recall_score(y_test_defect, y_defect_pred)
f1 = f1_score(y_test_defect, y_defect_pred)
auc = roc_auc_score(y_test_defect, y_defect_pred_prob)

print("불량 검출 성능 지표:")
print("=" * 40)
print(f"  정밀도 (Precision): {precision:.4f}")
print(f"  재현율 (Recall): {recall:.4f}")
print(f"  F1-score: {f1:.4f}")
print(f"  AUC-ROC: {auc:.4f}")

print("\n제조업 관점 해석:")
print("-" * 40)
print(f"  정밀도 {precision:.1%}: 불량 판정 중 실제 불량 비율")
print(f"  재현율 {recall:.1%}: 실제 불량 중 검출된 비율")
print("  -> 제조업에서는 재현율(불량 놓치지 않기)이 더 중요!")
```

---

## 3.6 특징 맵 시각화

```python
def visualize_feature_maps(model, image, layer_names=None):
    """
    CNN 특징 맵 시각화
    """
    # 합성곱 층 추출
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers
                      if 'conv' in layer.name]

    # 각 층의 출력을 얻는 모델 생성
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = keras.Model(inputs=model.input, outputs=outputs)

    # 특징 맵 계산
    feature_maps = feature_model.predict(image.reshape(1, 28, 28, 1), verbose=0)

    # 시각화
    for layer_name, fmap in zip(layer_names, feature_maps):
        n_features = min(fmap.shape[-1], 8)  # 최대 8개 필터만 표시

        fig, axes = plt.subplots(1, n_features, figsize=(15, 2))
        fig.suptitle(f'Layer: {layer_name}', fontsize=12)

        for i in range(n_features):
            axes[i].imshow(fmap[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'feature_map_{layer_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

# 샘플 이미지로 특징 맵 시각화
sample_image = X_test[0]
print(f"샘플 이미지 클래스: {class_names[y_test[0]]}")

plt.figure(figsize=(3, 3))
plt.imshow(sample_image, cmap='gray')
plt.title(f'Input: {class_names[y_test[0]]}')
plt.axis('off')
plt.savefig('sample_input.png', dpi=150, bbox_inches='tight')
plt.show()

# 특징 맵 시각화
visualize_feature_maps(model_basic, sample_image)
```

---

## 3.7 데이터 증강 기법

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 증강 생성기 설정
datagen = ImageDataGenerator(
    rotation_range=10,        # 회전 범위 (도)
    width_shift_range=0.1,    # 수평 이동 범위
    height_shift_range=0.1,   # 수직 이동 범위
    zoom_range=0.1,           # 확대/축소 범위
    horizontal_flip=True,     # 수평 반전
    fill_mode='nearest'       # 빈 공간 채우기 방식
)

print("데이터 증강 설정:")
print("=" * 40)
print("  - rotation_range: 10도")
print("  - width_shift_range: 10%")
print("  - height_shift_range: 10%")
print("  - zoom_range: 10%")
print("  - horizontal_flip: True")

# 증강된 이미지 예시 시각화
sample = X_train_cnn[0:1]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Data Augmentation Examples', fontsize=14)

# 원본 이미지
axes[0, 0].imshow(sample[0, :, :, 0], cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# 증강된 이미지들
for i, ax in enumerate(axes.flat[1:], 1):
    augmented = datagen.flow(sample, batch_size=1)
    aug_img = next(augmented)[0, :, :, 0]
    ax.imshow(aug_img, cmap='gray')
    ax.set_title(f'Augmented {i}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('data_augmentation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3.8 모델 저장 및 로드

```python
# 모델 저장
model_basic.save('fashion_mnist_cnn.keras')
print("모델 저장 완료: fashion_mnist_cnn.keras")

# 모델 로드
from tensorflow.keras.models import load_model

loaded_model = load_model('fashion_mnist_cnn.keras')
print("모델 로드 완료")

# 로드한 모델로 예측
sample_predictions = loaded_model.predict(X_test_cnn[:5], verbose=0)
predicted_classes = np.argmax(sample_predictions, axis=1)

print("\n로드된 모델 예측 결과:")
for i in range(5):
    actual = class_names[y_test[i]]
    predicted = class_names[predicted_classes[i]]
    confidence = sample_predictions[i, predicted_classes[i]]
    status = "O" if y_test[i] == predicted_classes[i] else "X"
    print(f"  [{status}] 실제: {actual:15} | 예측: {predicted:15} (확신도: {confidence:.2%})")
```

---

## 3.9 실시간 예측 함수

```python
def predict_single_image(model, image, class_names):
    """
    단일 이미지 예측

    Parameters:
        model: 학습된 CNN 모델
        image: 28x28 픽셀 이미지 (정규화 전)
        class_names: 클래스 이름 리스트

    Returns:
        예측 클래스, 확신도
    """
    # 전처리
    if image.max() > 1:
        image = image.astype('float32') / 255.0

    # 형태 변환
    image = image.reshape(1, 28, 28, 1)

    # 예측
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0, predicted_class]

    return class_names[predicted_class], confidence

# 예측 테스트
for i in range(5):
    predicted_class, confidence = predict_single_image(
        model_basic, X_test[i], class_names
    )
    actual_class = class_names[y_test[i]]
    print(f"샘플 {i+1}: 예측={predicted_class}, 실제={actual_class}, 확신도={confidence:.2%}")
```

---

## 3.10 전체 파이프라인 정리

```
제품 외관 검사 CNN 파이프라인:

1. 데이터 준비
   +-- 이미지 수집
   +-- 라벨링 (정상/불량)
   +-- 데이터 증강

2. 전처리
   +-- 크기 조정 (28x28 또는 224x224)
   +-- 정규화 (0~1)
   +-- 채널 차원 추가

3. 모델 구성
   +-- Conv2D + MaxPooling (특징 추출)
   +-- Flatten + Dense (분류)
   +-- Dropout (과적합 방지)

4. 학습
   +-- compile (optimizer, loss, metrics)
   +-- fit (callbacks 포함)
   +-- 검증 모니터링

5. 평가 및 배포
   +-- 테스트 정확도 확인
   +-- 혼동 행렬 분석
   +-- 모델 저장 (.keras)
```

---

# 핵심 정리

## 오늘 배운 내용

1. **CNN 수학적 배경**
   - 합성곱 연산: $(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)$
   - 특성 맵 크기: $O = \frac{W - K + 2P}{S} + 1$
   - Max Pooling: $y_{ij} = \max_{(m,n) \in R_{ij}} x_{mn}$

2. **Keras CNN 구현**
   - Conv2D, MaxPooling2D, Flatten, Dense
   - 다중 클래스: softmax + sparse_categorical_crossentropy
   - 이진 분류: sigmoid + binary_crossentropy

3. **이미지 불량 분류**
   - Fashion-MNIST 데이터셋 활용
   - 클래스 가중치로 불균형 처리
   - 특징 맵 시각화

---

## 핵심 코드

```python
# CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 컴파일 및 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_split=0.2,
          callbacks=[EarlyStopping(patience=5)])
```

---

## 체크리스트

- [ ] 합성곱 연산 수식 이해
- [ ] 특성 맵 크기 계산 공식 적용
- [ ] Conv2D, MaxPooling2D 층 구성
- [ ] 이미지 데이터 전처리 (정규화, reshape)
- [ ] 다중 클래스 / 이진 분류 설정
- [ ] 혼동 행렬로 성능 분석
- [ ] 모델 저장 및 로드

---

## 사용한 데이터셋

- **Fashion-MNIST**
  - 60,000 학습 이미지, 10,000 테스트 이미지
  - 28x28 픽셀 흑백 이미지
  - 10개 의류 카테고리

---

## 다음 차시 예고

### [24차시] 모델 저장과 실무 배포 준비

- 학습된 모델 저장/로드 방법
- 추론 최적화 기법
- 실무 배포 체크리스트
