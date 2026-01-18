# 실습과제 Q5: MLP 품질예측

**관련 차시**: 22차시 - 딥러닝 실습: MLP 품질예측
**난이도**: ★★★☆☆ (중급)
**예상 소요 시간**: 50~60분

---

## 학습 목표

이번 과제를 통해 다음 내용을 실습합니다:

1. StandardScaler로 데이터 정규화하기
2. Keras Sequential 모델로 MLP 구축하기
3. 과적합 방지 기법(Dropout, EarlyStopping) 적용하기
4. 학습 곡선을 해석하고 모델 성능 개선하기

---

## 배경 상황

의사결정나무의 성능 한계를 극복하기 위해 **딥러닝 기반 품질 예측 모델**을 구축해 달라는 요청을 받았습니다. MLP(Multi-Layer Perceptron)를 사용하여 더 복잡한 패턴을 학습하고, 과적합을 방지하면서 좋은 일반화 성능을 가진 모델을 만들어 주세요.

---

## 사전 준비

**필요한 라이브러리**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 재현성을 위한 시드 설정
np.random.seed(42)
tf.random.set_seed(42)
```

---

## 과제 내용

### 문제 1: 데이터 준비 및 정규화 (20점)

데이터를 로드하고 **StandardScaler**로 정규화하세요.

**수행 내용**:
```python
# 데이터 로드
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# 특성과 타겟 분리
feature_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[feature_cols]
y = df['Machine failure']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 정규화 (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 정규화 전후 비교
print("정규화 전 (X_train 첫 5행):")
print(X_train.head())
print(f"\n정규화 전 평균: {X_train.mean().values}")
print(f"정규화 전 표준편차: {X_train.std().values}")

print("\n정규화 후 (X_train_scaled 첫 5행):")
print(X_train_scaled[:5])
print(f"\n정규화 후 평균: {X_train_scaled.mean(axis=0).round(2)}")
print(f"정규화 후 표준편차: {X_train_scaled.std(axis=0).round(2)}")
```

**작성할 내용**:
- 정규화 전과 후의 평균, 표준편차 변화를 기록하세요
- StandardScaler가 하는 일은 무엇인가요?
- 왜 신경망에서 정규화가 중요한가요?
- `fit_transform`과 `transform`의 차이점은?

---

### 문제 2: MLP 모델 설계 (25점)

Keras Sequential API를 사용하여 **MLP 모델**을 설계하세요.

**모델 구조**:
- 입력층: 5개 특성
- 은닉층 1: 32개 뉴런, ReLU 활성화
- Dropout: 30%
- 은닉층 2: 16개 뉴런, ReLU 활성화
- 출력층: 1개 뉴런, Sigmoid 활성화 (이진 분류)

**수행 내용**:
```python
# 모델 정의
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model.summary()
```

**작성할 내용**:
- 모델의 총 파라미터 수(Total params)는 얼마인가요?
- 각 층의 파라미터 수가 어떻게 계산되는지 설명하세요
  - Dense(32): (입력차원 + 1) × 32 = ?
  - Dense(16): (32 + 1) × 16 = ?
  - Dense(1): (16 + 1) × 1 = ?
- Dropout(0.3)이 하는 역할은 무엇인가요?
- 왜 출력층에 sigmoid 활성화 함수를 사용하나요?

---

### 문제 3: 모델 학습 (25점)

모델을 학습하고 **EarlyStopping**을 적용하세요.

**수행 내용**:
```python
# EarlyStopping 콜백 정의
early_stop = EarlyStopping(
    monitor='val_loss',      # 검증 손실 모니터링
    patience=5,              # 5 에포크 동안 개선 없으면 중단
    restore_best_weights=True  # 최적 가중치로 복원
)

# 모델 학습
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,    # 학습 데이터의 20%를 검증용으로 사용
    callbacks=[early_stop],
    verbose=1
)

print(f"\n실제 학습 에포크 수: {len(history.history['loss'])}")
```

**작성할 내용**:
- 실제로 몇 에포크에서 학습이 중단되었나요?
- EarlyStopping이 작동한 이유는 무엇인가요?
- `patience=5`의 의미는 무엇인가요?
- `validation_split=0.2`와 test_size=0.2의 차이점은?

---

### 문제 4: 학습 곡선 분석 (15점)

**학습/검증 손실 그래프**를 그리고 과적합 여부를 진단하세요.

**수행 내용**:
```python
# 학습 히스토리에서 데이터 추출
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 손실 그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 손실 곡선
axes[0].plot(train_loss, label='Train Loss')
axes[0].plot(val_loss, label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 오른쪽: 정확도 곡선
axes[1].plot(train_acc, label='Train Accuracy')
axes[1].plot(val_acc, label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**작성할 내용**:
- 학습 손실(Train Loss)과 검증 손실(Val Loss)의 변화 패턴을 설명하세요
- 과적합의 징후가 보이나요? (힌트: 학습 손실↓, 검증 손실↑이면 과적합)
- 손실 곡선에서 최적의 에포크는 몇 번째로 보이나요?
- Dropout과 EarlyStopping이 과적합 방지에 효과가 있었나요?

---

### 문제 5: 모델 평가 (15점)

테스트 데이터로 최종 모델을 평가하세요.

**수행 내용**:
```python
# 테스트 데이터 평가
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")

# 예측
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 분류 리포트
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))

# 혼동행렬
print("\n혼동행렬:")
print(confusion_matrix(y_test, y_pred))
```

**작성할 내용**:
- 테스트 정확도는 얼마인가요?
- Q3(의사결정나무)의 정확도와 비교하면 어떤가요?
- 불량 탐지 재현율(Recall)은 얼마인가요?
- MLP가 의사결정나무보다 좋은 점과 나쁜 점은 무엇인가요?

---

## 심화 문제 (선택)

### 심화 1: 모델 구조 변경 실험
```python
# 더 깊은 모델 실험
model_deep = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_deep.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- 층을 더 추가하면 성능이 좋아지나요?

### 심화 2: 클래스 가중치 적용
```python
from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"클래스 가중치: {class_weight_dict}")

# 가중치 적용 학습
history_weighted = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
)
```
- 클래스 가중치를 적용하면 재현율이 어떻게 변하나요?

---

## 제출 형식

1. **코드**: Jupyter Notebook (.ipynb)
2. **보고서**: 각 문제의 "작성할 내용"에 대한 답변
3. **시각화**: 학습/검증 손실 곡선 이미지

---

## 채점 기준

| 항목 | 배점 | 세부 기준 |
|------|:----:|----------|
| 문제 1 | 20점 | 데이터 정규화 및 이유 설명 |
| 문제 2 | 25점 | MLP 모델 설계 및 파라미터 계산 |
| 문제 3 | 25점 | 모델 학습 및 EarlyStopping 적용 |
| 문제 4 | 15점 | 학습 곡선 분석 및 과적합 진단 |
| 문제 5 | 15점 | 테스트 평가 및 모델 비교 |
| **합계** | **100점** | |

---

## 참고 자료

- 22차시 강의 슬라이드
- [Keras Sequential API 문서](https://keras.io/guides/sequential_model/)
- [EarlyStopping 문서](https://keras.io/api/callbacks/early_stopping/)

---

## 핵심 개념 정리

| 용어 | 설명 |
|------|------|
| MLP | Multi-Layer Perceptron, 여러 은닉층을 가진 완전연결 신경망 |
| StandardScaler | 평균 0, 표준편차 1로 정규화 (Z-score 정규화) |
| Dropout | 학습 시 일부 뉴런을 무작위로 비활성화하여 과적합 방지 |
| EarlyStopping | 검증 성능이 개선되지 않으면 학습 조기 종료 |
| binary_crossentropy | 이진 분류 문제의 손실 함수 |

---

## 힌트

- 신경망은 학습할 때마다 결과가 달라질 수 있습니다. 재현성을 위해 `random_state`, `np.random.seed()`, `tf.random.set_seed()`를 설정하세요
- `validation_split`은 학습 데이터를 다시 나누는 것이고, 테스트 데이터와는 별개입니다
- 예측값이 확률(0~1)로 나오므로, 0.5를 기준으로 클래스를 결정합니다
- 불균형 데이터에서 성능을 높이려면 `class_weight` 파라미터를 활용하세요
