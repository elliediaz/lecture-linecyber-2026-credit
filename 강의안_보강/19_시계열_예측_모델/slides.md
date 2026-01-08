---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [19차시] 시계열 예측 모델

## 공공데이터 AI 예측 모델 개발

---

# 학습 목표

1. **시계열 특성 엔지니어링**을 수행한다
2. **ML 모델로 시계열**을 예측한다
3. **시계열 예측을 평가**한다

---

# 목차

## 대주제 1: 시계열 특성 엔지니어링
## 대주제 2: ML 모델로 시계열 예측
## 대주제 3: 시계열 예측 평가
## 실습편: 생산량 예측 모델 구축

---

<!-- _class: lead -->
# 대주제 1
## 시계열 특성 엔지니어링을 수행한다

---

# 특성 엔지니어링이란?

## 정의
- 원본 데이터에서 **예측에 유용한 특성**을 생성
- 모델 성능에 가장 큰 영향

## 시계열에서 특히 중요
- 단순히 과거 값만으로는 부족
- **패턴을 담은 특성**을 만들어야 함

---

# 시계열 특성의 종류

| 종류 | 설명 | 예시 |
|------|------|------|
| 날짜 특성 | 시간에서 추출 | hour, dayofweek |
| Lag 특성 | 과거 값 | lag_1, lag_24 |
| Rolling 특성 | 이동 통계 | ma_7, std_7 |
| 변화 특성 | 변화량/변화율 | diff, pct_change |
| 외부 특성 | 외부 요인 | 온도, 휴일 |

---

# 날짜 특성 추출

```python
# 인덱스가 DatetimeIndex일 때
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['quarter'] = df.index.quarter
df['day_of_year'] = df.index.dayofyear
```

## 효과
- 모델이 **시간대별 패턴** 학습
- "월요일은 불량률 높다" 같은 패턴

---

# 주기적 인코딩

## 문제
- hour: 0, 1, 2, ... 23, 0, 1, ...
- 23시와 0시는 가깝지만 숫자로는 23 차이

## 해결: 사인/코사인 인코딩

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

→ 23시와 0시가 연속적으로 표현됨

---

# Lag 특성 생성

```python
# 기본 Lag
df['lag_1'] = df['value'].shift(1)   # 1시간 전
df['lag_24'] = df['value'].shift(24) # 24시간 전
df['lag_168'] = df['value'].shift(168) # 7일 전

# 여러 Lag 한번에
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f'lag_{lag}'] = df['value'].shift(lag)
```

## ⚠️ 항상 shift로 미래 누출 방지!

---

# Rolling 특성 생성

```python
# 이동평균 (shift 먼저!)
df['ma_24'] = df['value'].shift(1).rolling(24).mean()
df['ma_168'] = df['value'].shift(1).rolling(168).mean()

# 이동 표준편차
df['std_24'] = df['value'].shift(1).rolling(24).std()

# 이동 최댓값/최솟값
df['max_24'] = df['value'].shift(1).rolling(24).max()
df['min_24'] = df['value'].shift(1).rolling(24).min()

# 이동 범위 (최대-최소)
df['range_24'] = df['max_24'] - df['min_24']
```

---

# 변화 특성

```python
# 차분 (변화량)
df['diff_1'] = df['value'].shift(1) - df['value'].shift(2)
# 어제 → 그제 대비 변화

# 변화율
df['pct_1'] = df['value'].shift(1).pct_change()

# 24시간 전 대비 변화
df['diff_24'] = df['value'].shift(1) - df['value'].shift(25)
```

---

# 비율 특성

```python
# 이동평균 대비 비율
df['ratio_ma'] = df['value'].shift(1) / df['ma_24']

# 최댓값 대비 비율
df['ratio_max'] = df['value'].shift(1) / df['max_168']

# 범위 내 위치
df['position'] = (df['value'].shift(1) - df['min_168']) / \
                 (df['max_168'] - df['min_168'])
```

---

# 외부 특성 추가

```python
# 휴일 여부
holidays = ['2025-01-01', '2025-01-27', ...]
df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(holidays).astype(int)

# 날씨 데이터 병합
weather_df = pd.read_csv('weather.csv')
df = df.merge(weather_df, on='date', how='left')

# 경제 지표, 이벤트 등
```

---

# 특성 선택 전략

## 1. 도메인 지식 기반
- "어제 생산량이 오늘에 영향" → lag_1
- "주간 패턴 있음" → lag_168

## 2. 실험적 접근
- 다양한 특성 생성 후 성능 비교
- 특성 중요도 분석

## 3. 과적합 주의
- 특성 너무 많으면 과적합 위험
- 교차검증으로 검증

---

<!-- _class: lead -->
# 대주제 2
## ML 모델로 시계열 예측

---

# 시계열 예측 접근법

## 전통적 방법
- ARIMA, Exponential Smoothing
- 시계열 전용 모델

## ML 기반 방법 (오늘 배울 내용)
- 특성 엔지니어링 후 **일반 ML 모델** 적용
- RandomForest, XGBoost 등
- 장점: 다양한 특성 활용 가능

---

# ML 모델 적용 흐름

```
1. 특성 엔지니어링
   → Lag, Rolling, 날짜 특성 생성

2. NaN 제거
   → shift/rolling으로 생긴 NaN 처리

3. 시간 기준 분할
   → 과거=학습, 미래=테스트

4. 모델 학습/예측
   → fit(), predict()

5. 평가
   → MAE, RMSE, MAPE
```

---

# RandomForest로 시계열 예측

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# 모델 생성
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
```

---

# 왜 RandomForest가 좋은가?

## 장점
1. **비선형 관계** 학습 가능
2. **특성 중요도** 자동 산출
3. **과적합에 강함**
4. **하이퍼파라미터 튜닝** 쉬움

## 단점
- 순차 의존성 직접 학습 불가
  → Lag 특성으로 보완!

---

# 특성 중요도 분석

```python
# 특성 중요도
importances = model.feature_importances_
feature_names = X_train.columns

# 정렬
sorted_idx = importances.argsort()[::-1]

print("특성 중요도:")
for i in sorted_idx[:10]:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")

# 보통 lag_1, lag_24 등 Lag 특성이 높게 나옴
```

---

# 다른 모델 비교

| 모델 | 특징 | 시계열 적합도 |
|------|------|--------------|
| LinearRegression | 단순, 해석 쉬움 | 낮음 |
| RandomForest | 비선형, 강건 | 높음 |
| GradientBoosting | 성능 좋음 | 높음 |
| XGBoost | 빠름, 성능 좋음 | 매우 높음 |
| LightGBM | 대용량 빠름 | 매우 높음 |

---

# 다단계 예측 (Multi-step)

## 1단계 예측
- t+1 시점만 예측

## 다단계 예측
- t+1, t+2, ..., t+n 예측

## 방법
1. **Recursive**: 예측값을 다음 입력으로
2. **Direct**: 각 horizon별 별도 모델
3. **Multi-output**: 한 모델이 여러 출력

---

# Recursive 예측

```python
# 24시간 예측
predictions = []
current_features = X_test.iloc[0].copy()

for i in range(24):
    # 예측
    pred = model.predict([current_features])[0]
    predictions.append(pred)

    # 특성 업데이트 (예측값으로)
    current_features['lag_1'] = pred
    # lag_2 = lag_1, lag_3 = lag_2, ...

# 오차가 누적되는 단점
```

---

# TimeSeriesSplit 교차검증

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X):
    X_train_cv = X.iloc[train_idx]
    y_train_cv = y.iloc[train_idx]
    X_val_cv = X.iloc[val_idx]
    y_val_cv = y.iloc[val_idx]

    model.fit(X_train_cv, y_train_cv)
    score = model.score(X_val_cv, y_val_cv)
    scores.append(score)

print(f"평균 R²: {np.mean(scores):.3f}")
```

---

<!-- _class: lead -->
# 대주제 3
## 시계열 예측 평가

---

# 시계열 평가 지표

| 지표 | 공식 | 특징 |
|------|------|------|
| MAE | Σ\|y-ŷ\|/n | 해석 쉬움 |
| RMSE | √(Σ(y-ŷ)²/n) | 큰 오차 패널티 |
| MAPE | Σ\|y-ŷ\|/y/n × 100 | 상대적 오차 (%) |
| R² | 1 - SS_res/SS_tot | 설명력 |

---

# MAE (Mean Absolute Error)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
```

## 해석
- 평균적으로 mae만큼 틀림
- 단위가 원래 데이터와 같아 해석 쉬움
- MAE = 50이면 "평균 50개 오차"

---

# RMSE (Root Mean Squared Error)

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
```

## 특징
- 큰 오차에 더 큰 패널티
- MAE보다 이상치에 민감
- 큰 오차가 중요하면 RMSE 사용

---

# MAPE (Mean Absolute Percentage Error)

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape(y_test, y_pred):.1f}%")
```

## 특징
- 퍼센트로 표현 → 직관적
- MAPE = 5%: "평균 5% 오차"
- ⚠️ y=0이면 계산 불가

---

# 지표 선택 가이드

| 상황 | 권장 지표 |
|------|----------|
| 일반적인 경우 | MAE, RMSE |
| 큰 오차가 치명적 | RMSE |
| 상대적 오차 중요 | MAPE |
| 비즈니스 보고 | MAPE (% 직관적) |
| 0에 가까운 값 있음 | MAE, RMSE (MAPE 사용 불가) |

---

# 시각적 평가

## 실제 vs 예측 플롯

```python
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label='실제', alpha=0.7)
plt.plot(y_test.index, y_pred, label='예측', alpha=0.7)
plt.legend()
plt.title('시계열 예측 결과')
plt.show()
```

---

# 잔차 분석

```python
residuals = y_test - y_pred

# 잔차 분포
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=30)
plt.title('잔차 분포')

# 시간에 따른 잔차
plt.subplot(1, 2, 2)
plt.plot(residuals.index, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('시간에 따른 잔차')
```

## 확인 사항
- 잔차가 0 주변에 분포?
- 시간에 따른 패턴 없음?

---

# 예측 구간

```python
# 단순 예측 구간 (잔차 표준편차 기반)
std = np.std(residuals)
lower = y_pred - 1.96 * std
upper = y_pred + 1.96 * std

plt.fill_between(y_test.index, lower, upper, alpha=0.3, label='95% 구간')
plt.plot(y_test.index, y_test, label='실제')
plt.plot(y_test.index, y_pred, label='예측')
```

---

<!-- _class: lead -->
# 실습편
## 생산량 예측 모델 구축

---

# 실습 개요

## 목표
- 제조 생산량 시계열 예측
- 특성 엔지니어링 → ML 학습 → 평가

## 데이터
- 30일간 시간별 생산량
- 온도, 압력 등 센서 데이터

---

# 실습 1: 데이터 생성

```python
import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range('2025-01-01', periods=24*30, freq='H')
hours = np.array([d.hour for d in dates])

df = pd.DataFrame({
    'timestamp': dates,
    'temperature': 85 + np.random.normal(0, 3, len(dates)),
    'production': 1000 + 30*np.sin(2*np.pi*hours/24) + \
                  np.random.normal(0, 50, len(dates))
})

df = df.set_index('timestamp')
```

---

# 실습 2: 특성 엔지니어링

```python
# 날짜 특성
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Lag 특성 (shift!)
for lag in [1, 24, 168]:
    df[f'lag_{lag}'] = df['production'].shift(lag)

# Rolling 특성 (shift 먼저!)
df['ma_24'] = df['production'].shift(1).rolling(24).mean()
df['std_24'] = df['production'].shift(1).rolling(24).std()

# 타겟
target = df['production'].copy()
features = df.drop('production', axis=1)
```

---

# 실습 3: 시간 분할 및 학습

```python
from sklearn.ensemble import RandomForestRegressor

# NaN 제거
features = features.dropna()
target = target.loc[features.index]

# 시간 분할
split_date = '2025-01-25'
X_train = features[:split_date]
y_train = target[:split_date]
X_test = features[split_date:]
y_test = target[split_date:]

# 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

# 실습 4: 예측 및 평가

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 예측
y_pred = model.predict(X_test)

# 평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.1f}%")
```

---

# 실습 5: 특성 중요도

```python
# 특성 중요도
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

# 시각화
plt.barh(importance['feature'][:10], importance['importance'][:10])
plt.title('특성 중요도 Top 10')
plt.show()
```

---

# 실습 6: 결과 시각화

```python
plt.figure(figsize=(14, 6))

# 실제 vs 예측
plt.subplot(1, 2, 1)
plt.plot(y_test.index, y_test, label='실제', alpha=0.7)
plt.plot(y_test.index, y_pred, label='예측', alpha=0.7)
plt.legend()
plt.title(f'생산량 예측 (MAPE: {mape:.1f}%)')

# 잔차
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=30)
plt.title('잔차 분포')

plt.tight_layout()
```

---

# 핵심 정리

## 1. 특성 엔지니어링
- 날짜, Lag, Rolling, 변화 특성
- shift(1) 필수로 미래 누출 방지

## 2. ML 모델 적용
- RandomForest, XGBoost 등 활용
- TimeSeriesSplit 교차검증

## 3. 평가
- MAE, RMSE, MAPE
- 시각적 분석 (실제 vs 예측, 잔차)

---

# 다음 차시 예고

## 19차시: 딥러닝 입문 - 신경망 기초
- 인공 뉴런과 신경망
- 활성화 함수
- 순전파와 역전파
- 머신러닝 vs 딥러닝

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습 파일: `18_timeseries_forecasting.py`
