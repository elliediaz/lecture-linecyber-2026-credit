---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 19차시'
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

# 딥러닝 실습: MLP로 품질 예측

## 19차시 | Part III. 문제 중심 모델링 실습

**Keras로 신경망 구축하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **Keras**로 신경망을 구축한다
2. **MLP 모델**을 학습한다
3. 제조 **품질 예측**에 적용한다

---

# Keras란?

## 딥러닝 프레임워크

> 신경망을 쉽게 만들 수 있는 **고수준 API**

```python
# TensorFlow 안에 포함됨
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### 왜 Keras?
- 간단하고 직관적인 코드
- TensorFlow의 공식 고수준 API
- 빠른 프로토타이핑

---

# MLP (다층 퍼셉트론)

## Multi-Layer Perceptron

```
입력층        은닉층1       은닉층2       출력층
 ○ ──────→ ○ ──────→ ○ ──────→ ○
 ○ ──────→ ○ ──────→ ○
 ○ ──────→ ○ ──────→ ○
            ○

3개 입력    8개 뉴런    4개 뉴런    1개 출력
```

> 가장 기본적인 심층 신경망

---

# Sequential 모델

## 층을 순차적으로 쌓기

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

- `Dense`: 완전 연결층
- `activation`: 활성화 함수
- `input_shape`: 첫 층에만 필요

---

# Dense 층

## 완전 연결층 (Fully Connected)

```python
Dense(8, activation='relu', input_shape=(3,))
```

| 파라미터 | 의미 |
|----------|------|
| `8` | 뉴런 개수 |
| `activation='relu'` | 활성화 함수 |
| `input_shape=(3,)` | 입력 크기 |

```
입력 3개 → Dense(8) → 출력 8개
         (가중치: 3×8 = 24개)
```

---

# 모델 컴파일

## compile()

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

| 파라미터 | 의미 |
|----------|------|
| `optimizer` | 최적화 알고리즘 (adam 추천) |
| `loss` | 손실 함수 |
| `metrics` | 평가 지표 |

---

# 손실 함수 선택

## 문제 유형에 따라

| 문제 | 출력층 | 손실 함수 |
|------|--------|-----------|
| 이진 분류 | sigmoid, 1개 | binary_crossentropy |
| 다중 분류 | softmax, N개 | categorical_crossentropy |
| 회귀 | linear, 1개 | mse |

```python
# 이진 분류 (불량/정상)
model.compile(loss='binary_crossentropy')

# 회귀 (생산량 예측)
model.compile(loss='mse')
```

---

# 모델 학습

## fit()

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

| 파라미터 | 의미 |
|----------|------|
| `epochs` | 전체 데이터 학습 횟수 |
| `batch_size` | 한 번에 학습할 샘플 수 |
| `validation_split` | 검증 데이터 비율 |

---

# Epoch와 Batch

## 학습 과정

```
데이터 1000개, batch_size=100, epochs=10

1 Epoch = 1000 / 100 = 10번 업데이트
10 Epochs = 100번 업데이트

Epoch 1: ████████████████████ 100%
Epoch 2: ████████████████████ 100%
...
Epoch 10: ████████████████████ 100%
```

> 작은 배치로 조금씩, 여러 에폭 동안 학습

---

# 학습 곡선

## Loss & Accuracy

```python
# 학습 기록
history.history['loss']      # 학습 손실
history.history['val_loss']  # 검증 손실
history.history['accuracy']  # 학습 정확도
```

```
Loss ↓                Accuracy ↑
  │╲                      │    ╱─────
  │ ╲                     │   ╱
  │  ╲─────               │  ╱
  └──────────             └──────────
    Epochs                  Epochs
```

---

# 과대적합 감지

## Train vs Validation

```
Loss
  │   Train ─────
  │              ╲
  │   Val ───╱───╲───  ← 과대적합!
  │        ╱
  └──────────────────
        Epochs
```

- Train 손실 ↓, Val 손실 ↑ = 과대적합
- 두 곡선이 함께 ↓ = 양호

---

# 예측 및 평가

## predict()

```python
# 예측 (확률)
y_prob = model.predict(X_test)

# 이진 분류: 0.5 기준 변환
y_pred = (y_prob > 0.5).astype(int)

# 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.3f}")
```

---

# 이론 정리

## Keras MLP 핵심

| 개념 | 설명 |
|------|------|
| Sequential | 층을 순차적으로 쌓기 |
| Dense | 완전 연결층 |
| compile | 최적화, 손실 함수 설정 |
| fit | 모델 학습 |
| predict | 예측 |

---

# - 실습편 -

## 19차시

**Keras로 품질 예측 모델 구축**

---

# 실습 개요

## 제조 불량 예측

### 목표
- Keras로 MLP 구축
- 불량 여부 이진 분류
- 학습 곡선 분석

### 실습 환경
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

---

# 실습 1: 데이터 생성

## 제조 불량 데이터

```python
np.random.seed(42)
n = 1000

temperature = np.random.normal(85, 5, n)
humidity = np.random.normal(50, 10, n)
speed = np.random.normal(100, 15, n)

# 불량 확률 (온도, 습도 영향)
prob = 0.05 + 0.03*(temperature-80)/5 + 0.02*(humidity-40)/10
defect = (np.random.random(n) < prob).astype(int)

X = np.column_stack([temperature, humidity, speed])
y = defect
```

---

# 실습 2: 데이터 전처리

## 정규화

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> 딥러닝은 정규화가 중요!

---

# 실습 3: 모델 구축

## Sequential

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
```

---

# 실습 4: 모델 컴파일

## optimizer, loss, metrics

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

- **adam**: 학습률 자동 조절
- **binary_crossentropy**: 이진 분류 손실
- **accuracy**: 정확도 모니터링

---

# 실습 5: 모델 학습

## fit()

```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

```
Epoch 1/50
25/25 [======] - loss: 0.45 - acc: 0.82
Epoch 2/50
25/25 [======] - loss: 0.38 - acc: 0.85
...
```

---

# 실습 6: 학습 곡선

## Loss & Accuracy 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.show()
```

---

# 실습 7: 예측 및 평가

## predict() & classification_report

```python
# 예측
y_prob = model.predict(X_test_scaled)
y_pred = (y_prob > 0.5).astype(int).flatten()

# 평가
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,
                           target_names=['정상', '불량']))
```

---

# 실습 8: RandomForest와 비교

## ML vs DL

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

mlp_acc = accuracy_score(y_test, y_pred)

print(f"RandomForest: {rf_acc:.3f}")
print(f"MLP:          {mlp_acc:.3f}")
```

> 테이블 데이터에서는 비슷하거나 ML이 나을 수 있음

---

# 실습 정리

## 핵심 체크포인트

- [ ] 데이터 정규화 (StandardScaler)
- [ ] Sequential로 모델 구축
- [ ] Dense 층 쌓기 (relu, sigmoid)
- [ ] compile (adam, binary_crossentropy)
- [ ] fit (epochs, batch_size)
- [ ] 학습 곡선 분석

---

# 다음 차시 예고

## 20차시: AI API의 이해와 활용

### 학습 내용
- AI API란?
- 외부 AI 서비스 활용
- API 호출 방법

> 만들어진 AI를 **API로 활용**하는 방법!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **Keras**: 딥러닝 고수준 API
2. **Sequential**: 층을 순차적으로 쌓기
3. **Dense**: 완전 연결층
4. **compile**: optimizer, loss, metrics 설정
5. **fit**: epochs, batch_size로 학습
6. **학습 곡선**: 과대적합 감지

---

# 감사합니다

## 19차시: 딥러닝 실습 - MLP로 품질 예측

**Keras로 첫 딥러닝 모델을 만들었습니다!**
