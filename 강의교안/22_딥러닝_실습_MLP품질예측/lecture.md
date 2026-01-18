# 21차시: 딥러닝 실습 - MLP로 품질 예측

## 학습 목표

1. **Keras** 프레임워크의 기본 사용법을 익힘
2. **MLP 모델**을 구현하고 학습시킴
3. 학습 결과를 **분석하고 개선**함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | Keras 기초 |
| 대주제 2 | 10분 | MLP 모델 구현 |
| 대주제 3 | 8분 | 학습 및 개선 |
| 정리 | 2분 | 핵심 요약 |

---

## 지난 시간 복습

- **인공 뉴런**: z = Σwx + b, y = f(z)
- **활성화 함수**: ReLU(은닉), Sigmoid/Softmax(출력)
- **순전파**: 입력에서 출력으로 계산
- **역전파**: 손실에서 기울기 계산 후 가중치 업데이트

---

# 대주제 1: Keras 기초

## 1.1 Keras란?

Keras는 TensorFlow 위에서 동작하는 딥러닝 프레임워크임. 복잡한 수학 없이도 직관적인 API로 모델을 구현할 수 있으며, 산업계에서 가장 많이 사용됨.

**설치 및 임포트**

```python
# 설치
pip install tensorflow

# 임포트
from tensorflow import keras
from tensorflow.keras import layers
```

---

## 1.2 왜 Keras인가?

| NumPy 직접 구현 | Keras 사용 |
|----------------|-----------|
| 순전파 직접 작성 | model.predict() |
| 역전파 직접 계산 | 자동 미분 |
| 경사하강법 구현 | 옵티마이저 선택 |
| 수십~수백 줄 코드 | **10줄 이내** |

---

## 1.3 Keras 모델 구조

```
Sequential 모델
    │
    ├── Dense(64, activation='relu')    <- 은닉층 1
    ├── Dense(32, activation='relu')    <- 은닉층 2
    └── Dense(1, activation='sigmoid')  <- 출력층
```

층을 순서대로 쌓는 **Sequential** 모델 사용

---

## 1.4 Sequential 모델 생성

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

## 1.5 Dense 층 이해

```python
Dense(units=64, activation='relu', input_shape=(10,))
```

| 파라미터 | 의미 |
|---------|------|
| `units` | 노드(뉴런) 개수 |
| `activation` | 활성화 함수 |
| `input_shape` | 입력 형태 (첫 번째 층만 필요) |

---

## 1.6 활성화 함수 선택

| 위치 | 문제 유형 | 활성화 함수 |
|------|----------|------------|
| 은닉층 | 모든 문제 | `'relu'` |
| 출력층 | 이진 분류 | `'sigmoid'` |
| 출력층 | 다중 분류 | `'softmax'` |
| 출력층 | 회귀 | `None` (linear) |

---

## 1.7 모델 컴파일

```python
model.compile(
    optimizer='adam',           # 옵티마이저
    loss='binary_crossentropy', # 손실 함수
    metrics=['accuracy']        # 평가 지표
)
```

**컴파일 = 학습 설정**

---

## 1.8 옵티마이저 선택

| 옵티마이저 | 특징 |
|-----------|------|
| `'sgd'` | 기본 경사하강법 |
| `'adam'` | **가장 많이 사용** (적응적 학습률) |
| `'rmsprop'` | RNN에 적합 |

**권장: 'adam'으로 시작**

---

## 1.9 손실 함수 선택

| 문제 유형 | 손실 함수 |
|----------|----------|
| 이진 분류 | `'binary_crossentropy'` |
| 다중 분류 | `'categorical_crossentropy'` |
| 다중 분류 (정수 라벨) | `'sparse_categorical_crossentropy'` |
| 회귀 | `'mse'` 또는 `'mae'` |

---

## 1.10 모델 요약

```python
model.summary()
```

출력 예시:
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

## 1.11 파라미터 수 계산

```
입력(10) -> Dense(64): 10 x 64 + 64 = 704
Dense(64) -> Dense(32): 64 x 32 + 32 = 2,080
Dense(32) -> Dense(1): 32 x 1 + 1 = 33

총: 2,817개
```

**파라미터 = (이전 노드 x 현재 노드) + 편향**

---

## 실습 코드: Keras 기본 설정

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# TensorFlow/Keras 임포트
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 숨기기

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

print(f"TensorFlow 버전: {tf.__version__}")
```

---

# 대주제 2: MLP 모델 구현

## 2.1 실습 목표

**제조 공정 품질 예측** (Breast Cancer 데이터셋 활용)
- 입력: 세포 특성 데이터 (30개 특성)
- 출력: 양성/악성 분류
- 목표: 정확한 진단 예측

---

## 2.2 데이터 준비

```python
# Breast Cancer 데이터 로드
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# DataFrame으로 변환
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

print(f"총 샘플 수: {len(df)}")
print(f"특성 수: {len(cancer.feature_names)}")
print(f"클래스: {cancer.target_names} (0=악성, 1=양성)")
print(f"양성(benign) 비율: {y.mean():.2%}")
```

---

## 2.3 데이터 전처리

```python
# 정규화 (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"정규화 전 - mean radius 범위: {X[:, 0].min():.1f} ~ {X[:, 0].max():.1f}")
print(f"정규화 후 - mean radius 범위: {X_scaled[:, 0].min():.2f} ~ {X_scaled[:, 0].max():.2f}")

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 세트: {len(X_train)}개")
print(f"테스트 세트: {len(X_test)}개")
```

---

## 2.4 데이터 정규화의 중요성

| 정규화 전 | 정규화 후 |
|----------|----------|
| 온도: 150~250 | -1.5 ~ 1.5 |
| 압력: 30~70 | -1.5 ~ 1.5 |
| 스케일 불균형 | **스케일 통일** |
| 학습 불안정 | **학습 안정** |

**신경망은 정규화 필수!**

---

## 2.5 기본 MLP 모델 생성

```python
# 기본 모델 생성
model_basic = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 컴파일
model_basic.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("모델 구조:")
model_basic.summary()
```

---

## 2.6 Dropout 이해

```
학습 시:
[O] [X] [O] [O] [X] [O]  <- 30% 무작위 비활성화
 |   |   |   |   |   |
[O] [O] [O] [O] [O] [O]

예측 시:
모든 노드 활성화 (출력에 0.7 곱함)
```

**효과**: 특정 노드에 의존하지 않음 -> 일반화 향상

---

## 2.7 Dropout 사용법

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

## 2.8 개선된 모델 설계

```python
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# 개선된 모델
model = Sequential([
    # 첫 번째 은닉층
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    # 두 번째 은닉층
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # 세 번째 은닉층
    Dense(16, activation='relu'),
    Dropout(0.1),

    # 출력층
    Dense(1, activation='sigmoid')
])

# 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("개선된 모델 구조:")
model.summary()
```

---

## 2.9 모델 컴파일 설정

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

## 2.10 모델 학습 (fit)

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

## 2.11 fit 파라미터

| 파라미터 | 의미 | 권장값 |
|---------|------|--------|
| epochs | 전체 데이터 반복 횟수 | 50~200 |
| batch_size | 한 번에 학습할 샘플 수 | 32~128 |
| validation_split | 검증 데이터 비율 | 0.2 |
| verbose | 0: 무출력, 1: 진행바, 2: 에포크당 한 줄 | 1 |

---

## 2.12 학습 출력 예시

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

## 2.13 history 객체

```python
# 학습 기록
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 손실 기록
history.history['loss']      # 학습 손실
history.history['val_loss']  # 검증 손실
```

---

## 2.14 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실
axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 정확도
axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2.15 학습 곡선 해석

| 패턴 | 의미 | 조치 |
|-----|------|------|
| Train 감소, Val 감소 | 정상 학습 | 계속 |
| Train 감소, Val 정체 | 과적합 시작 | 조기 종료 |
| Train 감소, Val 증가 | 과적합 | Dropout, 정규화 |
| Train 정체, Val 정체 | 학습 정체 | 학습률 조정 |

---

## 2.16 모델 평가

```python
# 테스트 데이터로 평가
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.2%}")

# 예측
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# AUC 점수
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc_score:.4f}")
```

---

## 2.17 분류 보고서

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
                            target_names=['악성(Malignant)', '양성(Benign)']))
```

출력 예시:
```
              precision    recall  f1-score   support
악성(Malignant)       0.90      0.85      0.87       300
양성(Benign)          0.84      0.89      0.86       280
    accuracy                           0.87       580
   macro avg       0.87      0.87      0.87       580
```

---

## 2.18 혼동 행렬 시각화

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted: Malignant', 'Predicted: Benign'])
ax.set_yticklabels(['Actual: Malignant', 'Actual: Benign'])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
               fontsize=20, fontweight='bold')

ax.set_title('Confusion Matrix')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

# 대주제 3: 학습 및 개선

## 3.1 과적합 문제

**증상**:
- 학습 정확도 상승
- 검증 정확도 정체 또는 하락

**원인**:
- 모델이 너무 복잡
- 데이터가 부족

---

## 3.2 과적합 해결책

| 방법 | 구현 |
|-----|------|
| Dropout | `Dropout(0.3)` |
| 조기 종료 | `EarlyStopping` |
| L2 정규화 | `kernel_regularizer=l2(0.01)` |
| 데이터 증강 | 데이터 늘리기 |

---

## 3.3 콜백 설정

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# EarlyStopping: 검증 손실이 개선되지 않으면 조기 종료
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint: 최적 모델 저장
checkpoint = ModelCheckpoint(
    'best_cancer_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# ReduceLROnPlateau: 학습 정체 시 학습률 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]
```

---

## 3.4 EarlyStopping 동작

```
Epoch 45: val_loss=0.2890 (최저)
Epoch 46: val_loss=0.2910
Epoch 47: val_loss=0.2950
...
Epoch 55: val_loss=0.3120 (patience=10 도달)
-> 학습 중단, Epoch 45 가중치 복원
```

**과적합 방지 + 학습 시간 절약**

---

## 3.5 L2 정규화

```python
from tensorflow.keras.regularizers import l2

model_l2 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],),
          kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_l2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**가중치가 너무 커지는 것을 방지**

---

## 3.6 모델 학습 (콜백 적용)

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print(f"학습 완료!")
print(f"실제 에포크 수: {len(history.history['loss'])}")
print(f"최종 학습 손실: {history.history['loss'][-1]:.4f}")
print(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
```

---

## 3.7 모델 저장 및 로드

```python
# 저장
model.save('cancer_model.keras')
print("모델 저장 완료: cancer_model.keras")

# 로드
from tensorflow.keras.models import load_model
loaded_model = load_model('cancer_model.keras')
print("모델 로드 완료")

# 로드한 모델로 예측
y_pred_loaded = loaded_model.predict(X_test[:5], verbose=0)
for i in range(5):
    status = "양성(Benign)" if y_pred_loaded[i, 0] > 0.5 else "악성(Malignant)"
    actual = "양성(Benign)" if y_test[i] == 1 else "악성(Malignant)"
    print(f"샘플 {i+1}: 예측={y_pred_loaded[i, 0]:.4f} ({status}), 실제={actual}")
```

---

## 3.8 학습률 조정

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

## 3.9 배치 정규화

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

## 3.10 하이퍼파라미터 튜닝

| 파라미터 | 시도 범위 |
|---------|----------|
| 은닉층 수 | 1~3개 |
| 노드 수 | 32, 64, 128, 256 |
| Dropout | 0.1~0.5 |
| 학습률 | 0.0001~0.01 |
| 배치 크기 | 16, 32, 64, 128 |

---

## 3.11 하이퍼파라미터 실험

```python
# 다양한 구조 테스트
architectures = [
    {'hidden': [32], 'dropout': [0.2]},
    {'hidden': [64, 32], 'dropout': [0.3, 0.2]},
    {'hidden': [128, 64, 32], 'dropout': [0.3, 0.2, 0.1]},
]

results = []
print("다양한 구조 테스트:")

for i, arch in enumerate(architectures):
    model_exp = Sequential()
    model_exp.add(Dense(arch['hidden'][0], activation='relu',
                       input_shape=(X_train.shape[1],)))
    model_exp.add(Dropout(arch['dropout'][0]))

    for j in range(1, len(arch['hidden'])):
        model_exp.add(Dense(arch['hidden'][j], activation='relu'))
        model_exp.add(Dropout(arch['dropout'][j]))

    model_exp.add(Dense(1, activation='sigmoid'))

    model_exp.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])

    history_exp = model_exp.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    _, acc = model_exp.evaluate(X_test, y_test, verbose=0)
    results.append({
        'structure': str(arch['hidden']),
        'accuracy': acc
    })
    print(f"  구조 {arch['hidden']}: 정확도={acc:.2%}")

# 최적 구조
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\n최적 구조: {best_result['structure']}")
print(f"정확도: {best_result['accuracy']:.2%}")
```

---

## 3.12 완전한 학습 코드

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

# 콜백
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
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

## 3.13 ML vs DL 비교

| 항목 | RandomForest | MLP (Keras) |
|-----|-------------|-------------|
| 특성 엔지니어링 | 필요 | 자동 학습 |
| 학습 속도 | 빠름 | 느림 (GPU 필요) |
| 해석 가능성 | 높음 | 낮음 |
| 대용량 데이터 | 제한적 | 강점 |
| 이미지/텍스트 | 어려움 | 강점 |

---

## 3.14 ML vs DL 비교 실습

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)

print("모델 비교:")
print(f"  Logistic Regression: {lr_acc:.2%}")
print(f"  Random Forest: {rf_acc:.2%}")
print(f"  MLP (Keras): {accuracy:.2%}")
```

---

## 3.15 언제 딥러닝을 쓸까?

**딥러닝이 유리한 경우**:
- 이미지, 자연어 데이터
- 복잡한 비선형 패턴
- 대용량 데이터 (수만 개 이상)

**ML이 유리한 경우**:
- 정형 데이터 (테이블)
- 데이터가 적음
- 해석이 중요함

---

## 3.16 실습: 전체 흐름

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

# 핵심 정리

## 오늘 배운 내용

1. **Keras 기초**
   - Sequential 모델, Dense 층
   - compile: optimizer, loss, metrics

2. **MLP 구현**
   - 모델 설계 -> 컴파일 -> fit -> evaluate
   - Dropout으로 과적합 방지

3. **학습 개선**
   - EarlyStopping, ModelCheckpoint
   - BatchNormalization, 학습률 조정

---

## 핵심 코드

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

## 체크리스트

- [ ] Keras Sequential 모델 생성
- [ ] Dense 층 추가 (활성화 함수 포함)
- [ ] Dropout으로 과적합 방지
- [ ] compile (optimizer, loss, metrics)
- [ ] fit (epochs, batch_size, validation_split)
- [ ] callbacks (EarlyStopping)
- [ ] evaluate로 테스트 평가
- [ ] 학습 곡선 시각화

---

## 사용한 데이터셋

- **Breast Cancer Wisconsin (sklearn)**
  - 569개 샘플, 30개 특성
  - 이진 분류: 악성(Malignant) vs 양성(Benign)

---

## 다음 차시 예고

### [22차시] 딥러닝 심화

- CNN (합성곱 신경망)
- RNN (순환 신경망)
- 고급 아키텍처 개요
