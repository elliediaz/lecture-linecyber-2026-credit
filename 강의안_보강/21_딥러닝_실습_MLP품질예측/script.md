# [21차시] 딥러닝 실습: MLP로 품질 예측 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 21차시 |
| 주제 | 딥러닝 실습: MLP로 품질 예측 |
| 시간 | 30분 (이론 10분 + 실습 18분 + 정리 2분) |
| 학습 목표 | Keras 기초, MLP 구현, 학습 개선 |

---

## 학습 목표

1. Keras 프레임워크의 기본 사용법을 익힌다
2. MLP 모델을 구현하고 학습시킨다
3. 학습 결과를 분석하고 개선한다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | Keras 기초 |
| 대주제 2 | 5분 | MLP 모델 구현 |
| 대주제 3 | 5분 | 학습 및 개선 |
| 실습 | 11분 | 품질 예측 모델 구현 |
| 정리 | 2분 | 요약 및 다음 차시 예고 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 신경망의 기초를 배웠습니다. 인공 뉴런, 활성화 함수, 순전파와 역전파를 NumPy로 직접 구현해봤죠."

> "오늘은 Keras로 같은 작업을 훨씬 쉽게 할 수 있다는 걸 보여드릴게요. 10줄 이내로 신경망을 만들고 학습시킬 수 있습니다."

---

#### 슬라이드 4-5: 학습 목표

> "오늘의 학습 목표는 세 가지입니다. Keras 프레임워크 사용법, MLP 모델 구현, 그리고 학습 결과를 개선하는 방법을 배웁니다."

---

### 대주제 1: Keras 기초 (5분)

#### 슬라이드 6-8: Keras란

> "Keras는 딥러닝 프레임워크입니다. TensorFlow 위에서 동작하고, 산업계에서 가장 많이 사용됩니다."

> "왜 인기가 있냐면, 복잡한 수학 없이도 직관적인 API로 모델을 만들 수 있거든요."

```python
# 설치
pip install tensorflow

# 임포트
from tensorflow import keras
from tensorflow.keras import layers
```

---

#### 슬라이드 9-11: Sequential 모델

> "Keras의 가장 기본적인 모델이 Sequential입니다. 층을 순서대로 쌓는 구조예요."

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

> "이게 끝입니다. 세 줄로 3층 신경망을 만들었어요. 어제 NumPy로 수십 줄 짜던 거랑 비교해보세요."

---

#### 슬라이드 12-14: Dense 층

> "Dense는 완전연결층입니다. 모든 입력 노드가 모든 출력 노드와 연결된 구조예요."

```python
Dense(units=64, activation='relu', input_shape=(10,))
```

> "units는 노드 개수, activation은 활성화 함수입니다. 첫 번째 층에만 input_shape를 지정해요."

---

#### 슬라이드 15-17: 컴파일

> "모델을 만들었으면 컴파일해야 합니다. 컴파일은 학습 설정을 정하는 단계예요."

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

> "optimizer는 경사하강법 종류입니다. adam이 가장 많이 쓰여요. loss는 손실 함수, metrics는 평가 지표입니다."

---

### 대주제 2: MLP 모델 구현 (5분)

#### 슬라이드 18-20: 실습 목표

> "오늘 만들 모델은 제조 품질 예측 모델입니다. 센서 데이터로 불량 여부를 예측해요."

> "입력은 온도, 압력, 속도, 습도 같은 센서 데이터고, 출력은 불량인지 아닌지입니다."

---

#### 슬라이드 21-23: 데이터 준비

> "먼저 데이터를 준비합니다. 스케일링이 중요해요. 신경망은 입력이 비슷한 범위여야 잘 학습합니다."

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> "StandardScaler로 평균 0, 표준편차 1로 정규화합니다. 이건 신경망 학습에 필수예요."

---

#### 슬라이드 24-26: 모델 설계

> "Dropout을 추가해서 과적합을 방지합니다."

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

> "Dropout(0.3)은 학습할 때 30%의 노드를 무작위로 끕니다. 특정 노드에 의존하지 않게 돼서 일반화가 좋아져요."

---

#### 슬라이드 27-29: 학습

> "fit 메서드로 학습합니다."

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

> "epochs는 전체 데이터 반복 횟수, batch_size는 한 번에 학습할 샘플 수입니다. validation_split=0.2면 학습 데이터의 20%를 검증용으로 씁니다."

---

#### 슬라이드 30-32: 학습 곡선

> "history 객체에 학습 기록이 저장됩니다. 이걸 그래프로 그려봐야 해요."

```python
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
```

> "학습 손실은 계속 내려가는데 검증 손실이 올라가면 과적합입니다. 두 선이 같이 내려가야 정상이에요."

---

### 대주제 3: 학습 및 개선 (5분)

#### 슬라이드 33-35: 과적합 문제

> "과적합은 학습 데이터만 외워버려서 새 데이터에 못 맞추는 문제입니다."

> "증상은 학습 정확도는 높은데 검증 정확도가 낮은 거예요. 학습 곡선에서 두 선이 벌어지면 과적합입니다."

---

#### 슬라이드 36-38: EarlyStopping

> "가장 좋은 해결책은 조기 종료입니다. 검증 손실이 더 이상 안 줄어들면 학습을 멈추는 거예요."

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(..., callbacks=[early_stop])
```

> "patience=10이면 10 에포크 동안 개선이 없으면 멈춥니다. restore_best_weights=True면 가장 좋았던 가중치로 복원해요."

---

#### 슬라이드 39-41: 모델 저장

> "학습한 모델은 저장해야 나중에 쓸 수 있어요."

```python
# 저장
model.save('quality_model.keras')

# 로드
from tensorflow.keras.models import load_model
model = load_model('quality_model.keras')
```

> "이렇게 저장해두면 다음에 학습 없이 바로 예측할 수 있습니다."

---

### 실습편 (11분)

#### 슬라이드 42-44: 데이터 생성

> "실습을 시작합니다. 먼저 제조 데이터를 만들게요."

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 1000

# 센서 데이터
temperature = np.random.normal(200, 30, n_samples)
pressure = np.random.normal(50, 15, n_samples)
speed = np.random.normal(100, 20, n_samples)
humidity = np.random.normal(60, 10, n_samples)

X = np.column_stack([temperature, pressure, speed, humidity])

# 불량 (온도, 압력이 높으면 불량 확률 증가)
logit = 0.05 * (temperature - 200) + 0.08 * (pressure - 50)
defect = (np.random.random(n_samples) < 1 / (1 + np.exp(-logit))).astype(int)
y = defect
```

---

#### 슬라이드 45-47: 전처리

```python
# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"학습: {len(X_train)}, 테스트: {len(X_test)}")
print(f"불량률: {y.mean():.2%}")
```

---

#### 슬라이드 48-50: 모델 구현

> "모델을 만듭니다."

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

---

#### 슬라이드 51-53: 학습

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
```

---

#### 슬라이드 54-56: 평가

```python
# 테스트 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.2%}")

# 예측
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 분류 보고서
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

#### 슬라이드 57-59: 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# 정확도
axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('learning_curve.png')
```

---

### 정리 (2분)

#### 슬라이드 60-61: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**Keras 기초**: Sequential 모델에 Dense 층을 쌓습니다. compile로 optimizer, loss, metrics를 설정합니다."

> "**MLP 구현**: fit으로 학습하고, Dropout과 BatchNormalization으로 과적합을 방지합니다."

> "**학습 개선**: EarlyStopping으로 적절한 시점에 멈추고, 학습 곡선으로 과적합을 확인합니다."

---

#### 슬라이드 62-63: 다음 차시 예고

> "다음 시간에는 딥러닝 심화를 배웁니다. CNN, RNN 같은 고급 아키텍처를 소개할 거예요."

> "오늘 수업 마무리합니다. 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: TensorFlow와 Keras 차이가 뭔가요?

> "Keras는 TensorFlow의 고수준 API입니다. TensorFlow가 엔진이고, Keras가 운전대라고 생각하시면 됩니다. 이제는 TensorFlow에 통합되어서 tensorflow.keras로 씁니다."

### Q2: GPU가 없어도 되나요?

> "작은 데이터와 간단한 모델은 CPU로도 충분합니다. 하지만 이미지나 대용량 데이터는 GPU가 있어야 실용적인 시간 내에 학습됩니다."

### Q3: epochs를 얼마로 해야 하나요?

> "정해진 값은 없습니다. EarlyStopping을 쓰면 적절한 시점에 알아서 멈추니까, epochs는 충분히 크게 (100~200) 설정하세요."

### Q4: batch_size는 어떻게 정하나요?

> "보통 32~128 사이를 씁니다. 작으면 업데이트가 잦아서 노이즈가 생기고, 크면 메모리를 많이 씁니다. GPU 메모리에 맞춰서 최대한 크게 하는 게 효율적입니다."

### Q5: validation_split 대신 별도 검증 데이터를 줄 수 있나요?

> "네, validation_data=(X_val, y_val)로 직접 줄 수 있습니다. 시계열 같이 순서가 중요한 데이터는 직접 분할하는 게 좋아요."

### Q6: 모델을 어떻게 개선하나요?

> "1) 은닉층 노드 수 조정, 2) 층 추가/제거, 3) Dropout 비율 조정, 4) 학습률 조정, 5) 배치 크기 조정을 시도해보세요. 검증 정확도를 기준으로 비교합니다."

---

## 참고 자료

### 공식 문서
- [Keras 공식 문서](https://keras.io/)
- [TensorFlow Keras 가이드](https://www.tensorflow.org/guide/keras)

### 추천 자료
- 케라스 창시자에게 배우는 딥러닝 (프랑소와 숄레)
- TensorFlow 2.0 공식 튜토리얼

### 관련 차시
- 19차시: 딥러닝 입문 - 신경망 기초
- 21차시: 딥러닝 심화

---

## 체크리스트

수업 전:
- [ ] TensorFlow/Keras 설치 확인
- [ ] GPU 사용 가능 여부 확인
- [ ] 예제 코드 테스트

수업 중:
- [ ] model.summary() 결과 설명
- [ ] 학습 곡선 해석 강조
- [ ] EarlyStopping 필수 강조
- [ ] 정규화(StandardScaler) 중요성

수업 후:
- [ ] 실습 코드 배포
- [ ] CNN/RNN 예고
