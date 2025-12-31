---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 20차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 딥러닝 실습: MLP로 품질 예측

## 20차시 | AI 기초체력훈련 (Pre AI-Campus)

**Keras로 신경망 직접 구현하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **Keras**로 MLP 모델을 구축한다
2. 모델을 **학습하고 평가**한다
3. **학습 곡선**을 해석한다

---

# Keras란?

## 딥러닝 프레임워크

```python
# TensorFlow의 고수준 API
from tensorflow import keras
from tensorflow.keras import layers
```

### 특징
- 간단한 문법
- 빠른 프로토타이핑
- TensorFlow 기반

> "쉽게 딥러닝을 시작할 수 있다"

---

# MLP 구조 설계

## 제조 품질 예측 모델

```
  입력층        은닉층1       은닉층2       출력층
  (4개)        (64개)        (32개)        (1개)

   ○  ─┐
   ○  ─┼─→   ○ ○ ○   →   ○ ○ ○   →   ○
   ○  ─┤     ... (64)    ... (32)     (Sigmoid)
   ○  ─┘

특성입력      ReLU         ReLU      확률 출력
온도,습도    특성추출     특성추출    불량확률
속도,압력
```

---

# Sequential API

## 순차적으로 층 쌓기

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # 입력층+은닉층1
    Dense(32, activation='relu'),                     # 은닉층2
    Dense(1, activation='sigmoid')                    # 출력층
])
```

### Dense 층
- **Fully Connected Layer**
- 모든 뉴런이 이전 층과 연결

---

# 모델 컴파일

## 학습 설정

```python
model.compile(
    optimizer='adam',             # 최적화 알고리즘
    loss='binary_crossentropy',   # 손실 함수 (이진 분류)
    metrics=['accuracy']          # 평가 지표
)
```

### 주요 설정
| 문제 | 출력 활성화 | 손실 함수 |
|------|-----------|----------|
| 이진 분류 | sigmoid | binary_crossentropy |
| 다중 분류 | softmax | categorical_crossentropy |
| 회귀 | linear | mse |

---

# 모델 요약

## model.summary()

```python
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                320
dense_1 (Dense)              (None, 32)                2,080
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 2,433
Trainable params: 2,433
```

---

# 모델 학습

## model.fit()

```python
history = model.fit(
    X_train, y_train,
    epochs=50,                  # 전체 데이터 반복 횟수
    batch_size=32,              # 한 번에 처리할 샘플 수
    validation_split=0.2,       # 검증 데이터 비율
    verbose=1                   # 출력 설정
)
```

### 용어
- **Epoch**: 전체 데이터를 한 번 학습
- **Batch**: 한 번에 처리하는 데이터 묶음
- **Validation**: 학습 중 성능 확인용 데이터

---

# 학습 과정 출력

## 진행 상황

```
Epoch 1/50
12/12 [==============================] - 0s 10ms/step
- loss: 0.6923 - accuracy: 0.5200
- val_loss: 0.6915 - val_accuracy: 0.5300

Epoch 25/50
12/12 [==============================] - 0s 2ms/step
- loss: 0.4521 - accuracy: 0.7800
- val_loss: 0.4723 - val_accuracy: 0.7650

Epoch 50/50
...
```

---

# 학습 곡선

## Learning Curve

```python
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

```
 Loss
   │
   │ ─   Train
   │  ─  ─
   │     ─ ─ ─ ─ ─ ─    ← 수렴
   │ ···· Validation
   │ ···
   │    ··· ··· ··· ···
   └────────────────────→ Epoch
```

---

# 과대적합 탐지

## 학습 곡선에서 발견

```
정상                         과대적합

Loss                         Loss
 │ ─ Train                    │ ─ Train
 │  ─                         │  ─ ─ ─ ─ ─ ─ ─
 │   ─ ─ ─                    │
 │ ··· Val                    │ ···    ·····  ← Val 증가!
 │  ···                       │    ···
 │     ··· ···                │
 └──────────→ Epoch          └──────────────→ Epoch
```

> Validation Loss가 증가하면 **과대적합**!

---

# Early Stopping

## 과대적합 방지

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      # 모니터링할 지표
    patience=10,             # 개선 없이 기다릴 에폭
    restore_best_weights=True  # 최적 가중치 복원
)

model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop]   # 콜백 추가
)
```

---

# 모델 평가

## model.evaluate()

```python
# 테스트 데이터로 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.4f}")
```

### 예측

```python
# 확률 예측
probabilities = model.predict(X_test)

# 클래스 예측 (0.5 기준)
predictions = (probabilities > 0.5).astype(int)
```

---

# 데이터 전처리

## 스케일링 중요!

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 왜 스케일링?
- 신경망은 **스케일에 민감**
- 특성 범위가 다르면 학습 불안정
- 표준화로 평균 0, 표준편차 1로 변환

---

# 전체 워크플로우

## 딥러닝 파이프라인

```python
# 1. 데이터 준비 & 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 3. 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. 학습
history = model.fit(X_train_scaled, y_train, epochs=50,
                    validation_split=0.2, callbacks=[early_stop])

# 5. 평가
loss, acc = model.evaluate(X_test_scaled, y_test)
```

---

# 하이퍼파라미터

## 조정할 수 있는 것들

| 파라미터 | 설명 | 권장 |
|---------|------|------|
| 은닉층 수 | 깊이 | 2~4개로 시작 |
| 뉴런 수 | 층의 너비 | 32, 64, 128 |
| 학습률 | 가중치 업데이트 크기 | 0.001 (기본) |
| 배치 크기 | 한 번에 처리할 양 | 32, 64 |
| 에폭 | 반복 횟수 | 50~100, Early Stopping |

---

# ML vs DL 결과 비교

## 같은 문제, 다른 접근

```python
# 랜덤포레스트
rf_accuracy = 0.85

# MLP
mlp_accuracy = 0.83

# 결과
print("테이블 데이터에서는 비슷한 성능!")
print("데이터가 적으면 ML이 유리할 수도 있음")
```

> 항상 딥러닝이 최선은 아님!

---

# Part III 정리

## 머신러닝 & 딥러닝

| 차시 | 내용 |
|------|------|
| 11 | 머신러닝 개요 |
| 12-13 | 분류 모델 (트리, 랜덤포레스트) |
| 14 | 회귀 모델 |
| 15 | 모델 평가, 교차검증 |
| 16 | 하이퍼파라미터 튜닝 |
| 17-18 | 시계열 데이터 |
| 19-20 | 딥러닝 기초 |

---

# 다음 차시 예고

## 21차시부터: Part IV - AI 활용

- AI API 활용
- Streamlit으로 웹앱 만들기
- FastAPI로 모델 서빙
- 특성 중요도 분석
- 모델 저장 및 배포

> 만든 모델을 **실무에 활용**하기!

---

# 감사합니다

## AI 기초체력훈련 20차시

**딥러닝 실습: MLP로 품질 예측**

Part III 완료! 수고하셨습니다!
