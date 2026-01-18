---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [21차시] 딥러닝 실습: MLP로 품질 예측

## Keras로 신경망 구현하기

---

# 학습 목표

1. **Keras** 프레임워크의 기본 사용법을 익힌다
2. **MLP 모델**을 구현하고 학습시킨다
3. 학습 결과를 **분석하고 개선**한다

---

# 지난 시간 복습

- **인공 뉴런**: z = Σwx + b, y = f(z)
- **활성화 함수**: ReLU(은닉), Sigmoid/Softmax(출력)
- **순전파**: 입력 → 출력 계산
- **역전파**: 손실 → 기울기 계산 → 가중치 업데이트

**오늘**: Keras로 이 모든 것을 자동으로!

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | Keras 기초 |
| 대주제 2 | 10분 | MLP 모델 구현 |
| 대주제 3 | 8분 | 학습 및 개선 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## Keras 기초

---

# Keras란?

- **딥러닝 프레임워크**
- TensorFlow 위에서 동작
- **직관적인 API**: 복잡한 수학 없이 모델 구현
- 산업계에서 가장 많이 사용

```python
# 설치
pip install tensorflow

# 임포트
from tensorflow import keras
from tensorflow.keras import layers
```

---

# 왜 Keras인가?

| NumPy 직접 구현 | Keras 사용 |
|----------------|-----------|
| 순전파 직접 작성 | model.predict() |
| 역전파 직접 계산 | 자동 미분 |
| 경사하강법 구현 | 옵티마이저 선택 |
| 수십~수백 줄 코드 | **10줄 이내** |

---

# Keras 모델 구조

```
Sequential 모델
    │
    ├── Dense(64, activation='relu')    ← 은닉층 1
    ├── Dense(32, activation='relu')    ← 은닉층 2
    └── Dense(1, activation='sigmoid')  ← 출력층
```

층을 순서대로 쌓는 **Sequential** 모델

---

# Sequential 모델 생성

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 방법 1: 리스트로 전달
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 방법 2: add로 추가
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

---

# Dense 층 이해하기

```python
Dense(units=64, activation='relu', input_shape=(10,))
```

| 파라미터 | 의미 |
|---------|------|
| `units` | 노드(뉴런) 개수 |
| `activation` | 활성화 함수 |
| `input_shape` | 입력 형태 (첫 번째 층만) |

---

# 활성화 함수 선택

| 위치 | 문제 유형 | 활성화 함수 |
|------|----------|------------|
| 은닉층 | 모든 문제 | `'relu'` |
| 출력층 | 이진 분류 | `'sigmoid'` |
| 출력층 | 다중 분류 | `'softmax'` |
| 출력층 | 회귀 | `None` (linear) |

---

# 모델 컴파일

```python
model.compile(
    optimizer='adam',           # 옵티마이저
    loss='binary_crossentropy', # 손실 함수
    metrics=['accuracy']        # 평가 지표
)
```

**컴파일 = 학습 설정**

---

# 옵티마이저 선택

| 옵티마이저 | 특징 |
|-----------|------|
| `'sgd'` | 기본 경사하강법 |
| `'adam'` | **가장 많이 사용** (적응적 학습률) |
| `'rmsprop'` | RNN에 적합 |

**권장: 'adam'으로 시작**

---

# 손실 함수 선택

| 문제 유형 | 손실 함수 |
|----------|----------|
| 이진 분류 | `'binary_crossentropy'` |
| 다중 분류 | `'categorical_crossentropy'` |
| 다중 분류 (정수 라벨) | `'sparse_categorical_crossentropy'` |
| 회귀 | `'mse'` 또는 `'mae'` |

---

# 모델 요약

```python
model.summary()
```

출력:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                704
dense_1 (Dense)              (None, 32)                2080
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 2,817
```

---

# 파라미터 수 확인

```
입력(10) → Dense(64): 10×64 + 64 = 704
Dense(64) → Dense(32): 64×32 + 32 = 2,080
Dense(32) → Dense(1): 32×1 + 1 = 33

총: 2,817개
```

**파라미터 = (이전 노드 × 현재 노드) + 편향**

---

<!-- _class: lead -->
# 대주제 2
## MLP 모델 구현

---

# 실습 목표

**제조 공정 품질 예측**
- 입력: 온도, 압력, 속도, 습도 등 센서 데이터
- 출력: 불량 여부 (0: 정상, 1: 불량)
- 목표: 불량 제품 사전 예측

---

# 데이터 준비

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드
df = pd.read_csv('manufacturing_data.csv')

# 특성과 타겟 분리
X = df[['temperature', 'pressure', 'speed', 'humidity']].values
y = df['defect'].values

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

# 데이터 정규화의 중요성

| 정규화 전 | 정규화 후 |
|----------|----------|
| 온도: 150~250 | -1.5 ~ 1.5 |
| 압력: 30~70 | -1.5 ~ 1.5 |
| 스케일 불균형 | **스케일 통일** |
| 학습 불안정 | **학습 안정** |

**신경망은 정규화 필수!**

---

# 모델 설계

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    # 입력층 + 첫 번째 은닉층
    Dense(64, activation='relu', input_shape=(4,)),
    Dropout(0.3),  # 과적합 방지

    # 두 번째 은닉층
    Dense(32, activation='relu'),
    Dropout(0.2),

    # 출력층 (이진 분류)
    Dense(1, activation='sigmoid')
])
```

---

# Dropout 이해하기

```
학습 시:
[●] [○] [●] [●] [○] [●]  ← 30% 무작위 비활성화
 ↓   ↓   ↓   ↓   ↓   ↓
[●] [●] [●] [●] [●] [●]

예측 시:
모든 노드 활성화 (출력에 0.7 곱함)
```

**효과**: 특정 노드에 의존하지 않음 → 일반화 향상

---

# Dropout 사용법

```python
from tensorflow.keras.layers import Dropout

# Dropout(rate): rate 비율로 무작위 비활성화
Dense(64, activation='relu'),
Dropout(0.3),  # 30% 드롭아웃

Dense(32, activation='relu'),
Dropout(0.2),  # 20% 드롭아웃
```

**권장**: 은닉층 사이에 0.2~0.5

---

# 모델 컴파일

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
```

| 설정 | 값 | 이유 |
|-----|-----|------|
| optimizer | adam | 적응적 학습률 |
| loss | binary_crossentropy | 이진 분류 |
| metrics | accuracy, AUC | 평가 지표 |

---

# 모델 학습 (fit)

```python
history = model.fit(
    X_train, y_train,           # 학습 데이터
    epochs=100,                  # 반복 횟수
    batch_size=32,               # 배치 크기
    validation_split=0.2,        # 검증 데이터 비율
    verbose=1                    # 출력 레벨
)
```

---

# fit 파라미터

| 파라미터 | 의미 | 권장값 |
|---------|------|--------|
| epochs | 전체 데이터 반복 횟수 | 50~200 |
| batch_size | 한 번에 학습할 샘플 수 | 32~128 |
| validation_split | 검증 데이터 비율 | 0.2 |
| verbose | 0: 무출력, 1: 진행바, 2: 에포크당 한 줄 | 1 |

---

# 학습 출력 예시

```
Epoch 1/100
50/50 [==============================] - 1s 10ms/step
- loss: 0.6543 - accuracy: 0.6123 - val_loss: 0.5890 - val_accuracy: 0.6500

Epoch 50/100
50/50 [==============================] - 0s 5ms/step
- loss: 0.2345 - accuracy: 0.9012 - val_loss: 0.3012 - val_accuracy: 0.8750

Epoch 100/100
50/50 [==============================] - 0s 5ms/step
- loss: 0.1567 - accuracy: 0.9345 - val_loss: 0.3456 - val_accuracy: 0.8650
```

---

# history 객체

```python
# 학습 기록
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 손실 기록
history.history['loss']      # 학습 손실
history.history['val_loss']  # 검증 손실
```

**시각화에 활용**

---

# 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss')
axes[0].legend()

# 정확도
axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy')
axes[1].legend()
```

---

# 학습 곡선 해석

| 패턴 | 의미 | 조치 |
|-----|------|------|
| Train↓, Val↓ | 정상 학습 | 계속 |
| Train↓, Val→ | 과적합 시작 | 조기 종료 |
| Train↓, Val↑ | 과적합 | Dropout, 정규화 |
| Train→, Val→ | 학습 정체 | 학습률 조정 |

---

# 모델 평가

```python
# 테스트 데이터로 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {accuracy:.2%}")
```

출력:
```
13/13 [==============================] - 0s 2ms/step
- loss: 0.3234 - accuracy: 0.8750
테스트 정확도: 87.50%
```

---

# 예측하기

```python
# 확률 예측
y_pred_prob = model.predict(X_test)
print(y_pred_prob[:5])
# [[0.12], [0.87], [0.34], [0.95], [0.08]]

# 이진 분류 (임계값 0.5)
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred[:5])
# [[0], [1], [0], [1], [0]]
```

---

# 분류 보고서

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support
           0       0.90      0.85      0.87       300
           1       0.84      0.89      0.86       280
    accuracy                           0.87       580
   macro avg       0.87      0.87      0.87       580
```

---

<!-- _class: lead -->
# 대주제 3
## 학습 및 개선

---

# 과적합 문제

**증상**:
- 학습 정확도 ↑
- 검증 정확도 → 또는 ↓

**원인**:
- 모델이 너무 복잡
- 데이터가 부족

---

# 과적합 해결책

| 방법 | 구현 |
|-----|------|
| Dropout | `Dropout(0.3)` |
| 조기 종료 | `EarlyStopping` |
| L2 정규화 | `kernel_regularizer=l2(0.01)` |
| 데이터 증강 | 데이터 늘리기 |

---

# EarlyStopping 콜백

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',    # 모니터링 지표
    patience=10,           # 개선 없으면 10 에포크 후 중단
    restore_best_weights=True  # 최적 가중치 복원
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

---

# EarlyStopping 동작

```
Epoch 45: val_loss=0.2890 (최저)
Epoch 46: val_loss=0.2910
Epoch 47: val_loss=0.2950
...
Epoch 55: val_loss=0.3120 (patience=10 도달)
→ 학습 중단, Epoch 45 가중치 복원
```

**과적합 방지 + 학습 시간 절약**

---

# L2 정규화

```python
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.01),  # L2 정규화
          input_shape=(4,)),
    Dense(32, activation='relu',
          kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])
```

**가중치가 너무 커지는 것을 방지**

---

# ModelCheckpoint 콜백

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.keras',      # 저장 경로
    monitor='val_loss',
    save_best_only=True      # 최적 모델만 저장
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)
```

---

# 모델 저장 및 로드

```python
# 저장
model.save('quality_model.keras')

# 로드
from tensorflow.keras.models import load_model
loaded_model = load_model('quality_model.keras')

# 예측
y_pred = loaded_model.predict(X_new)
```

---

# 학습률 조정

```python
from tensorflow.keras.optimizers import Adam

# 기본 학습률 (0.001)
optimizer = Adam()

# 학습률 조정
optimizer = Adam(learning_rate=0.0001)  # 더 느리게
optimizer = Adam(learning_rate=0.01)    # 더 빠르게

model.compile(optimizer=optimizer, ...)
```

---

# ReduceLROnPlateau 콜백

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # 학습률을 절반으로
    patience=5,        # 5 에포크 개선 없으면
    min_lr=0.00001     # 최소 학습률
)
```

**학습 정체 시 자동으로 학습률 감소**

---

# 배치 정규화

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    BatchNormalization(),  # 배치 정규화
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])
```

**학습 안정화 + 빠른 수렴**

---

# 하이퍼파라미터 튜닝

| 파라미터 | 시도 범위 |
|---------|----------|
| 은닉층 수 | 1~3개 |
| 노드 수 | 32, 64, 128, 256 |
| Dropout | 0.1~0.5 |
| 학습률 | 0.0001~0.01 |
| 배치 크기 | 16, 32, 64, 128 |

---

# 완전한 학습 코드

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# 완전한 학습 코드 (계속)

```python
# 콜백
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss',
                    save_best_only=True)
]

# 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {accuracy:.2%}")
```

---

# 실습: 전체 흐름

```
1. 데이터 준비
   └─ 로드, 전처리, 분할, 정규화

2. 모델 설계
   └─ Sequential, Dense, Dropout

3. 컴파일
   └─ optimizer, loss, metrics

4. 학습
   └─ fit, callbacks

5. 평가
   └─ evaluate, predict
```

---

# ML vs DL 비교

| 항목 | RandomForest | MLP (Keras) |
|-----|-------------|-------------|
| 특성 엔지니어링 | 필요 | 자동 학습 |
| 학습 속도 | 빠름 | 느림 (GPU 필요) |
| 해석 가능성 | 높음 | 낮음 |
| 대용량 데이터 | 제한적 | 강점 |
| 이미지/텍스트 | 어려움 | 강점 |

---

# 언제 딥러닝을 쓸까?

**딥러닝이 유리한 경우**:
- 이미지, 자연어 데이터
- 복잡한 비선형 패턴
- 대용량 데이터 (수만 개 이상)

**ML이 유리한 경우**:
- 정형 데이터 (테이블)
- 데이터가 적음
- 해석이 중요함

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **Keras 기초**
   - Sequential 모델, Dense 층
   - compile: optimizer, loss, metrics

2. **MLP 구현**
   - 모델 설계 → 컴파일 → fit → evaluate
   - Dropout으로 과적합 방지

3. **학습 개선**
   - EarlyStopping, ModelCheckpoint
   - BatchNormalization, 학습률 조정

---

# 핵심 코드

```python
# 1. 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 2. 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. 학습
model.fit(X_train, y_train, epochs=100, validation_split=0.2,
          callbacks=[EarlyStopping(patience=10)])

# 4. 평가
model.evaluate(X_test, y_test)
```

---

# 체크리스트

- [ ] Keras Sequential 모델 생성
- [ ] Dense 층 추가 (활성화 함수 포함)
- [ ] Dropout으로 과적합 방지
- [ ] compile (optimizer, loss, metrics)
- [ ] fit (epochs, batch_size, validation_split)
- [ ] callbacks (EarlyStopping)
- [ ] evaluate로 테스트 평가
- [ ] 학습 곡선 시각화

---

# 다음 차시 예고

## [21차시] 딥러닝 심화

- CNN (합성곱 신경망)
- RNN (순환 신경망)
- 고급 아키텍처 개요

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습: Keras로 품질 예측 모델 구현하기
