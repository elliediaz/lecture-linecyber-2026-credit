---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 18차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 시계열 예측 모델

## 18차시 | AI 기초체력훈련 (Pre AI-Campus)

**과거 데이터로 미래 예측하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **시계열 특성**을 엔지니어링한다
2. **ML 모델**로 시계열을 예측한다
3. 시계열 예측 **성능을 평가**한다

---

# 시계열 예측이란?

## 과거 → 미래

```
   과거 데이터         예측 모델        미래 예측
┌───────────────┐    ┌────────┐    ┌──────────┐
│ 1월: 1,000개  │    │        │    │          │
│ 2월: 1,050개  │ →  │ 모델   │ →  │ 7월: ?개 │
│ ...           │    │        │    │ 8월: ?개 │
│ 6월: 1,200개  │    │        │    │          │
└───────────────┘    └────────┘    └──────────┘
```

### 활용
- 생산량 예측 → 원자재 발주
- 매출 예측 → 인력 배치
- 수요 예측 → 재고 관리

---

# 시계열 예측 접근법

## 두 가지 방법

### 1. 전통적 시계열 모델
- ARIMA, Prophet
- 시계열 전용 모델

### 2. ML 기반 접근 (오늘 배울 내용!)
- 시계열 → 특성 변환 → ML 모델
- 이미 배운 RandomForest, LinearRegression 활용

> 특성 엔지니어링이 핵심!

---

# 특성 엔지니어링

## 시계열 → 테이블 데이터

```python
# 원래 시계열
날짜        생산량
2024-01-01  1,000
2024-01-02  1,050
2024-01-03  1,020

# 특성 추가 후
날짜        월  요일  전일_생산량  7일_평균  생산량
2024-01-03   1    2    1,050      ...     1,020
```

> 시간 정보와 과거 값을 **특성으로** 만들기!

---

# 날짜 기반 특성

## datetime에서 추출

```python
# 날짜 특성
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek
df['주차'] = df['날짜'].dt.isocalendar().week

# 특수 날짜
df['월초'] = (df['날짜'].dt.day <= 5).astype(int)
df['월말'] = (df['날짜'].dt.day >= 25).astype(int)
df['주말'] = (df['날짜'].dt.dayofweek >= 5).astype(int)
```

---

# 시차 특성 (Lag Features)

## 과거 값 활용

```python
# 시차 특성
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
df['lag_30'] = df['생산량'].shift(30) # 30일 전

# 결과
날짜        생산량  lag_1  lag_7
2024-01-08  1,030  1,020  1,000
2024-01-09  1,050  1,030  1,050
```

> 과거 n일 전 값이 예측에 도움!

---

# 롤링 특성 (Rolling Features)

## 이동 통계량

```python
# 이동평균
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
df['ma_30'] = df['생산량'].shift(1).rolling(30).mean()

# 이동 표준편차 (변동성)
df['std_7'] = df['생산량'].shift(1).rolling(7).std()

# 이동 최대/최소
df['max_7'] = df['생산량'].shift(1).rolling(7).max()
df['min_7'] = df['생산량'].shift(1).rolling(7).min()
```

> ⚠️ shift(1)을 먼저! (미래 정보 누출 방지)

---

# 데이터 누출 방지

## 중요한 주의사항!

```python
# ❌ 잘못된 예 (미래 정보 포함)
df['ma_7'] = df['생산량'].rolling(7).mean()  # 오늘 값 포함!

# ✅ 올바른 예 (과거 정보만)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()  # 어제까지만!
```

### 규칙
- 특성에는 **예측 시점에 알 수 있는 정보만** 사용
- shift(1)로 하루 늦추기
- 테스트할 때도 미래 정보 사용 금지

---

# 시계열 Train/Test 분할

## 시간 기준 필수!

```python
# ✅ 올바른 분할 (시간 기준)
split_date = '2024-05-01'
train = df[df['날짜'] < split_date]
test = df[df['날짜'] >= split_date]

# ❌ 잘못된 분할 (랜덤)
train_test_split(X, y, random_state=42)  # 시간 뒤섞임!
```

```
|←───── Train ─────→|←── Test ──→|
|      1~4월         |    5~6월    |
|    과거로 학습      |  미래로 평가 |
```

---

# 모델 학습

## RandomForest로 시계열 예측

```python
from sklearn.ensemble import RandomForestRegressor

# 특성과 타겟
features = ['월', '요일', 'lag_1', 'lag_7', 'ma_7']
X = df[features]
y = df['생산량']

# 분할
train = df[df['날짜'] < '2024-05-01']
test = df[df['날짜'] >= '2024-05-01']

# 학습
model = RandomForestRegressor(n_estimators=100)
model.fit(train[features], train['생산량'])

# 예측
predictions = model.predict(test[features])
```

---

# 예측 결과 시각화

## 실제 vs 예측

```python
plt.figure(figsize=(12, 5))
plt.plot(test['날짜'], test['생산량'],
         label='Actual', linewidth=2)
plt.plot(test['날짜'], predictions,
         label='Predicted', linewidth=2, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('Production Forecast')
plt.legend()
plt.show()
```

---

# 평가 지표

## 시계열 예측 평가

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MAE: 평균 절대 오차
mae = mean_absolute_error(test['생산량'], predictions)

# RMSE: 평균 제곱근 오차
rmse = np.sqrt(mean_squared_error(test['생산량'], predictions))

# MAPE: 평균 절대 백분율 오차
mape = np.mean(np.abs((test['생산량'] - predictions) / test['생산량'])) * 100

print(f"MAE: {mae:.1f}개")
print(f"RMSE: {rmse:.1f}개")
print(f"MAPE: {mape:.1f}%")
```

---

# MAPE 해석

## Mean Absolute Percentage Error

$$MAPE = \frac{1}{n}\sum\left|\frac{실제 - 예측}{실제}\right| \times 100\%$$

| MAPE | 해석 |
|------|------|
| < 10% | 매우 좋음 |
| 10~20% | 좋음 |
| 20~50% | 보통 |
| > 50% | 개선 필요 |

> 제조 현장에서는 보통 **10~20%** 이내를 목표로!

---

# 특성 중요도

## 어떤 특성이 중요한가?

```python
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)
```

```
    feature    importance
    lag_1      0.45
    ma_7       0.25
    lag_7      0.15
    요일        0.10
    월          0.05
```

> 전일 생산량(lag_1)이 가장 중요!

---

# 다중 스텝 예측

## 여러 날 앞 예측

```python
# 방법 1: 직접 예측 (각 기간별 모델)
model_1day = ...  # 1일 후 예측 모델
model_7day = ...  # 7일 후 예측 모델

# 방법 2: 재귀적 예측
for i in range(7):
    # 1일 예측 → 다음날 특성으로 사용 → 다시 예측
    next_pred = model.predict(current_features)
    # 특성 업데이트...
```

> 입문 단계에서는 **1일 후 예측**에 집중!

---

# 실습 정리

## 전체 워크플로우

```python
# 1. 특성 엔지니어링
df['월'] = df['날짜'].dt.month
df['lag_1'] = df['생산량'].shift(1)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()

# 2. 결측치 제거 (lag/rolling으로 생긴 NaN)
df = df.dropna()

# 3. 시간 기준 분할
train = df[df['날짜'] < split_date]
test = df[df['날짜'] >= split_date]

# 4. 모델 학습 및 예측
model.fit(train[features], train['생산량'])
predictions = model.predict(test[features])

# 5. 평가
mae = mean_absolute_error(test['생산량'], predictions)
```

---

# 다음 차시 예고

## 19차시: 딥러닝 입문

- 신경망 기초 개념
- 뉴런, 층, 활성화 함수
- 딥러닝 vs 머신러닝

> 드디어 **딥러닝**의 세계로!

---

# 감사합니다

## AI 기초체력훈련 18차시

**시계열 예측 모델**

과거 데이터로 미래를 예측했습니다!
