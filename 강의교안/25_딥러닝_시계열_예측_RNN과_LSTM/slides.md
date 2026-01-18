---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [25차시] 딥러닝 응용: 시계열 예측 RNN과 LSTM

## 순환 신경망으로 제조 센서 데이터 예측하기

---

# 학습 목표

1. **순환 신경망(RNN)**의 구조와 시퀀스 처리 원리를 이해한다
2. **LSTM과 GRU**의 게이트 구조로 장기 의존성 문제를 해결한다
3. **Predictive Maintenance AI4I 2020** 데이터로 센서값 예측을 수행한다

---

# 지난 시간 복습

- **CNN**: 합성곱 연산으로 이미지 특징 추출
- **공간적 구조**: 2D 필터가 이미지의 지역적 패턴 인식
- **계층적 학습**: 저수준(엣지) -> 고수준(형태) 특징

**오늘**: 시간적 순서가 중요한 **시계열 데이터**를 위한 RNN/LSTM

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 7분 | RNN 기초 - 순환 구조와 시퀀스 처리 |
| Part 2 | 8분 | LSTM과 GRU - 장기 의존성 해결 |
| Part 3 | 8분 | 제조 센서 시계열 예측 실습 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# Part 1
## RNN 기초 - 순환 구조와 시퀀스 처리

---

# 시퀀스 데이터 vs 일반 데이터

| 구분 | 일반 데이터 (MLP) | 시퀀스 데이터 (RNN) |
|------|------------------|-------------------|
| 입력 순서 | 무관 | **중요** |
| 예시 | 정적 센서값 | 시간별 온도 변화 |
| 특징 | 각 입력 독립적 | 이전 상태가 현재에 영향 |
| 적용 | 분류, 회귀 | 시계열 예측, 자연어 처리 |

**핵심**: "이전에 무슨 일이 있었는지"가 중요한 데이터

---

# 제조업의 시계열 데이터 예시

```
시간(분)   온도(°C)   회전수(RPM)   토크(Nm)
   0        25.3        1500         40.2
   1        25.8        1520         41.1
   2        26.5        1535         42.3   <- 이상 징후?
   3        27.9        1548         44.8
   4        30.2        1560         48.5   <- 고장 임박!
```

**질문**: "다음 시점의 센서값은?"
**답**: 이전 패턴을 기억해야 예측 가능!

---

# RNN의 순환 구조

```
일반 신경망 (MLP):
  x₁ → [Layer] → y₁
  x₂ → [Layer] → y₂   (각 입력 독립)
  x₃ → [Layer] → y₃

순환 신경망 (RNN):
  x₁ → [RNN] → y₁
         ↓ h₁
  x₂ → [RNN] → y₂      (이전 상태 h 전달)
         ↓ h₂
  x₃ → [RNN] → y₃
```

**h (hidden state)**: 과거 정보를 담은 "기억"

---

# RNN 셀 내부 구조

```
        ┌─────────────────────┐
  xₜ ─→ │                     │
        │   hₜ = tanh(Wₕhₜ₋₁  │ ─→ hₜ (다음으로 전달)
hₜ₋₁ ─→ │        + Wₓxₜ + b)  │
        │                     │ ─→ yₜ (출력)
        └─────────────────────┘
```

**입력**: 현재 입력 $x_t$ + 이전 은닉 상태 $h_{t-1}$
**출력**: 새로운 은닉 상태 $h_t$

---

# RNN 수식

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

| 기호 | 의미 |
|------|------|
| $h_t$ | 시점 t의 은닉 상태 (기억) |
| $h_{t-1}$ | 이전 시점의 은닉 상태 |
| $x_t$ | 시점 t의 입력 |
| $W_h, W_x$ | 가중치 행렬 |
| $\tanh$ | 활성화 함수 (-1 ~ 1) |

---

# RNN 시퀀스 처리 예시

```
센서 시계열: [25.3, 25.8, 26.5, 27.9]

t=0: h₀ = tanh(W_h·0 + W_x·25.3) = 0.42
t=1: h₁ = tanh(W_h·0.42 + W_x·25.8) = 0.51
t=2: h₂ = tanh(W_h·0.51 + W_x·26.5) = 0.68
t=3: h₃ = tanh(W_h·0.68 + W_x·27.9) = 0.85

최종 h₃에는 전체 시퀀스 정보가 압축됨!
```

---

# 기울기 소실 문제 (Vanishing Gradient)

**문제**: 긴 시퀀스에서 초기 정보가 사라짐

```
센서값:  [25, 26, 27, ..., 35, 36]  (100개 시점)
                  ↓
         역전파 시 기울기가 점점 작아짐
                  ↓
         초기 값(25, 26)의 영향력 소실!
```

**비유**: "100일 전에 온도가 살짝 올랐던 것"을 기억 못함

---

# 기울기 소실의 수학적 원인

역전파 시 기울기:
$$
\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

- $\tanh$의 도함수: 최대 1
- 여러 번 곱하면 → **기울기가 0에 수렴**

**해결책**: LSTM, GRU

---

<!-- _class: lead -->
# Part 2
## LSTM과 GRU - 장기 의존성 해결

---

# LSTM (Long Short-Term Memory)

**핵심 아이디어**: 기울기가 흐르는 "고속도로" 추가

```
RNN:  정보가 좁은 길 하나로만 전달
      ┌─────────┐
      │ 은닉상태 │ → 정보 손실 심함
      └─────────┘

LSTM: 별도의 "셀 상태" 고속도로 추가
      ┌─────────┐
      │ 셀 상태  │ → 장기 기억 보존
      ├─────────┤
      │ 은닉상태 │ → 단기 기억
      └─────────┘
```

---

# LSTM의 3가지 게이트

```
┌──────────────────────────────────────────┐
│                  LSTM 셀                  │
│                                          │
│  ┌──────┐   ┌──────┐   ┌──────┐         │
│  │Forget│   │Input │   │Output│         │
│  │ Gate │   │ Gate │   │ Gate │         │
│  └──────┘   └──────┘   └──────┘         │
│     ↓          ↓          ↓             │
│  "뭘 잊을까?" "뭘 저장?" "뭘 출력?"      │
│                                          │
│  ────────── 셀 상태 (장기 기억) ─────────│
│  ────────── 은닉 상태 (단기 기억) ───────│
└──────────────────────────────────────────┘
```

---

# Forget Gate (잊기 게이트)

**역할**: 이전 기억 중 **잊을 것**을 결정

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

| 출력값 | 의미 |
|--------|------|
| 0 | 완전히 잊음 |
| 1 | 완전히 기억 |
| 0.5 | 절반만 유지 |

**예**: "3일 전 온도 급등은 중요, 5일 전 정상 온도는 잊어도 됨"

---

# Input Gate (입력 게이트)

**역할**: 새 정보 중 **저장할 것**을 결정

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- $i_t$: 얼마나 저장할지 (0~1)
- $\tilde{C}_t$: 저장할 후보 값 (-1~1)

**예**: "현재 온도 급등은 중요한 정보, 강하게 저장!"

---

# 셀 상태 업데이트

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

```
이전 기억(C_{t-1}) × 잊기게이트(f_t) = 남길 기억
        +
새 정보(C̃_t) × 입력게이트(i_t) = 추가할 기억
        =
새로운 셀 상태(C_t)
```

**$\odot$**: 요소별 곱셈 (element-wise)

---

# Output Gate (출력 게이트)

**역할**: 셀 상태에서 **출력할 것**을 결정

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

- 셀 상태(장기 기억)에서 필요한 부분만 꺼내서 출력
- $h_t$: 다음 시점으로 전달 + 예측에 사용

---

# LSTM 전체 구조 다이어그램

```
       ┌─────────────────────────────────────┐
       │              LSTM Cell              │
       │                                     │
C_{t-1}│───[×]────────[+]──────────────────→│C_t
       │    ↑    ↗    ↑                     │
       │   f_t  C̃_t  i_t                    │
       │    ↑    ↑    ↑                     │
       │  ┌────────────────┐                │
       │  │σ│tanh│σ│σ│    ←──── x_t        │
       │  └────────────────┘                │
h_{t-1}│─────────┬────────[×]──────────────→│h_t
       │         │         ↑                │
       │         │        o_t               │
       └─────────┴───────────────────────────┘
```

---

# GRU (Gated Recurrent Unit)

**LSTM의 간소화 버전** - 2개 게이트만 사용

| LSTM | GRU |
|------|-----|
| Forget Gate | Update Gate (z) |
| Input Gate | Update Gate (z) |
| Output Gate | Reset Gate (r) |
| 3개 게이트 | 2개 게이트 |

**장점**: 파라미터 적음, 학습 빠름
**단점**: 매우 긴 의존성에서는 LSTM이 유리할 수 있음

---

# GRU 수식

**Reset Gate**: 이전 은닉 상태 얼마나 무시?
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

**Update Gate**: 새 정보와 이전 정보 비율?
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**은닉 상태 업데이트**:
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

# RNN vs LSTM vs GRU 비교

| 항목 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 게이트 수 | 0 | 3 | 2 |
| 파라미터 | 적음 | 많음 | 중간 |
| 장기 의존성 | 약함 | **강함** | 강함 |
| 학습 속도 | 빠름 | 느림 | **빠름** |
| 권장 용도 | 짧은 시퀀스 | 복잡한 패턴 | 일반적 사용 |

**실무 팁**: GRU로 시작, 성능 부족 시 LSTM으로 변경

---

<!-- _class: lead -->
# Part 3
## 제조 센서 시계열 예측 실습

---

# 실습 데이터: Predictive Maintenance AI4I 2020

| 항목 | 내용 |
|------|------|
| 출처 | UCI ML Repository / Kaggle |
| 샘플 수 | 10,000개 |
| 센서 변수 | 온도, 회전수, 토크, 마모 |
| 목표 | 다음 시점 센서값 예측 |

**제조 시나리오**: 설비 상태 모니터링 및 이상 조기 탐지

---

# 데이터 구성

```
+----------+-----------+------------+--------+-------+
| Air Temp | Proc Temp | Rot Speed  | Torque | Wear  |
|   (K)    |    (K)    |   (rpm)    |  (Nm)  | (min) |
+----------+-----------+------------+--------+-------+
|  298.1   |   308.6   |    1551    |  42.8  |   0   |
|  298.2   |   308.7   |    1408    |  46.3  |   3   |
|  298.1   |   308.5   |    1498    |  49.4  |   5   |
|  298.2   |   308.6   |    1433    |  39.5  |   7   |
+----------+-----------+------------+--------+-------+
```

**예측 목표**: 과거 N개 시점으로 다음 시점 예측

---

# 시퀀스 데이터 준비 (Sliding Window)

```python
def create_sequences(data, seq_length):
    """
    시계열 데이터를 LSTM 입력 형태로 변환

    Parameters:
        data: 원본 시계열 (n_samples, n_features)
        seq_length: 입력 시퀀스 길이

    Returns:
        X: (n_samples - seq_length, seq_length, n_features)
        y: (n_samples - seq_length, n_features)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
```

---

# Sliding Window 시각화

```
원본 데이터: [v₁, v₂, v₃, v₄, v₅, v₆, v₇, v₈]

seq_length = 3:

샘플 1: [v₁, v₂, v₃] → v₄
샘플 2: [v₂, v₃, v₄] → v₅
샘플 3: [v₃, v₄, v₅] → v₆
샘플 4: [v₄, v₅, v₆] → v₇
샘플 5: [v₅, v₆, v₇] → v₈

X shape: (5, 3, n_features)
y shape: (5, n_features)
```

---

# 데이터 전처리 파이프라인

```python
from sklearn.preprocessing import MinMaxScaler

# 1. 센서 컬럼 선택
sensor_cols = ['Air temperature [K]', 'Process temperature [K]',
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# 2. 정규화 (0~1 스케일링)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[sensor_cols])

# 3. 시퀀스 생성
SEQ_LENGTH = 10  # 과거 10개 시점 사용
X, y = create_sequences(scaled_data, SEQ_LENGTH)

print(f"X shape: {X.shape}")  # (9990, 10, 5)
print(f"y shape: {y.shape}")  # (9990, 5)
```

---

# Keras LSTM 모델 구성

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # 첫 번째 LSTM 층 (return_sequences=True)
    LSTM(64, return_sequences=True,
         input_shape=(SEQ_LENGTH, n_features)),
    Dropout(0.2),

    # 두 번째 LSTM 층
    LSTM(32),

    # 출력층
    Dense(16, activation='relu'),
    Dense(n_features)  # 모든 센서값 예측
])
```

---

# LSTM 층 파라미터 설명

```python
LSTM(units, return_sequences, input_shape)
```

| 파라미터 | 설명 | 예시 |
|---------|------|------|
| units | 은닉 상태 차원 (기억 용량) | 64 |
| return_sequences | 모든 시점 출력 여부 | True/False |
| input_shape | 입력 형태 (시퀀스 길이, 특성 수) | (10, 5) |

**return_sequences**:
- True: 모든 시점의 출력 (다음 LSTM 층에 연결)
- False: 마지막 시점만 출력 (Dense 층에 연결)

---

# 모델 구조 시각화

```
입력: (batch, 10, 5) - 10개 시점, 5개 센서
           ↓
┌─────────────────────────────────┐
│ LSTM(64, return_sequences=True) │
│ 출력: (batch, 10, 64)           │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│         Dropout(0.2)            │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│         LSTM(32)                │
│ 출력: (batch, 32)               │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│    Dense(16) → Dense(5)         │
│ 출력: (batch, 5) - 5개 센서 예측│
└─────────────────────────────────┘
```

---

# 모델 컴파일 및 학습

```python
model.compile(
    optimizer='adam',
    loss='mse',  # 회귀 문제: Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

# 조기 종료 콜백
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

---

# 예측 결과 시각화

```python
# 테스트 데이터 예측
y_pred = model.predict(X_test)

# 역정규화
y_pred_orig = scaler.inverse_transform(y_pred)
y_test_orig = scaler.inverse_transform(y_test)

# 온도 예측 결과 비교 (첫 번째 센서)
plt.figure(figsize=(12, 4))
plt.plot(y_test_orig[:100, 0], label='실제', alpha=0.8)
plt.plot(y_pred_orig[:100, 0], label='예측', alpha=0.8)
plt.xlabel('시점')
plt.ylabel('온도 (K)')
plt.title('Air Temperature 예측 vs 실제')
plt.legend()
```

---

# 성능 평가 지표

| 지표 | 공식 | 해석 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | 작을수록 좋음 |
| MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | 평균 오차 크기 |
| RMSE | $\sqrt{MSE}$ | 원래 단위로 해석 |
| MAPE | $\frac{100}{n}\sum|\frac{y - \hat{y}}{y}|$ | 백분율 오차 |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
```

---

# 제조 센서 예측 활용 시나리오

```
실시간 센서 모니터링 파이프라인:

[센서 데이터 수집]
        ↓
[최근 N개 시점 버퍼]
        ↓
[LSTM 모델 예측] → 다음 시점 센서값
        ↓
[임계값 비교]
        ↓
[예측 온도 > 310K ?]
        ↓
  예 → 알람 발생 → 선제적 유지보수
  아니오 → 정상 모니터링 계속
```

---

<!-- _class: lead -->
# 핵심 정리

---

# RNN/LSTM/GRU 수식 요약

| 모델 | 핵심 수식 |
|------|----------|
| RNN | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ |
| LSTM Forget | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ |
| LSTM Input | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| LSTM Output | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ |
| LSTM Cell | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| GRU Update | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$ |

---

# Keras LSTM 구현 요약

```python
# 시퀀스 생성
X, y = create_sequences(data, seq_length=10)

# 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(n_features)
])

# 학습
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50)
```

---

# 오늘 배운 내용

1. **RNN 기초**
   - 순환 구조로 시퀀스 처리
   - 기울기 소실 문제의 원인

2. **LSTM과 GRU**
   - 3가지 게이트로 장기 의존성 해결
   - GRU: LSTM의 간소화 버전

3. **제조 센서 예측**
   - Sliding window로 시퀀스 데이터 준비
   - Keras LSTM 모델 구현 및 학습

---

# 체크리스트

- [ ] RNN 순환 구조와 은닉 상태 역할 이해
- [ ] 기울기 소실 문제와 해결 필요성 인식
- [ ] LSTM 3가지 게이트 (Forget, Input, Output) 역할 설명
- [ ] GRU와 LSTM 차이점 비교
- [ ] create_sequences 함수로 시퀀스 데이터 준비
- [ ] Keras LSTM 모델 구성 및 학습

---

# 다음 차시 예고

## [26차시] 딥러닝 모델 해석과 파인튜닝

- 학습된 모델의 예측 근거 해석
- SHAP/LIME을 활용한 변수 중요도 분석
- 모델 파인튜닝 및 성능 최적화 기법

---

<!-- _class: lead -->
# 수고하셨습니다!

## 질문: RNN/LSTM이 설비 고장 예측에 어떻게 활용될 수 있을까요?
