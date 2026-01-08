# 19차시: 시계열 예측 모델

## 학습 목표

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | 시계열 특성 엔지니어링을 수행함 |
| 2 | ML 모델로 시계열을 예측함 |
| 3 | 시계열 예측을 평가함 |

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 특성 엔지니어링 | 날짜, Lag, Rolling 특성 |
| 2 | ML 모델 적용 | RandomForest로 시계열 예측 |
| 3 | 평가 | MAE, RMSE, MAPE |

---

## Part 1: 시계열 특성 엔지니어링

### 1.1 특성 엔지니어링이란?

**정의**
- 원본 데이터에서 예측에 유용한 특성을 생성함
- 모델 성능에 가장 큰 영향을 미침

**시계열에서 특히 중요한 이유**
- 단순히 과거 값만으로는 부족함
- 패턴을 담은 특성을 만들어야 함

---

### 1.2 시계열 특성의 종류

| 종류 | 설명 | 예시 |
|------|------|------|
| 날짜 특성 | 시간에서 추출 | hour, dayofweek |
| Lag 특성 | 과거 값 | lag_1, lag_24 |
| Rolling 특성 | 이동 통계 | ma_7, std_7 |
| 변화 특성 | 변화량/변화율 | diff, pct_change |
| 외부 특성 | 외부 요인 | 온도, 휴일 |

---

### 1.3 날짜 특성 추출

```python
# 인덱스가 DatetimeIndex일 때
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['quarter'] = df.index.quarter
df['day_of_year'] = df.index.dayofyear
```

**효과**
- 모델이 시간대별 패턴을 학습함
- "월요일은 불량률 높다" 같은 패턴을 학습할 수 있음

---

### 1.4 주기적 인코딩

**문제**
- hour: 0, 1, 2, ... 23, 0, 1, ...
- 23시와 0시는 가깝지만 숫자로는 23 차이

**해결: 사인/코사인 인코딩**

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

23시와 0시가 연속적으로 표현됨.

---

### 1.5 Lag 특성 생성

```python
# 기본 Lag
df['lag_1'] = df['value'].shift(1)   # 1시간 전
df['lag_24'] = df['value'].shift(24) # 24시간 전
df['lag_168'] = df['value'].shift(168) # 7일 전

# 여러 Lag 한번에
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f'lag_{lag}'] = df['value'].shift(lag)
```

**주의**: 항상 shift로 미래 누출을 방지해야 함!

---

### 1.6 Rolling 특성 생성

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

### 1.7 변화 특성

```python
# 차분 (변화량) - shift 필수
df['diff_1'] = df['value'].shift(1) - df['value'].shift(2)
# 어제 -> 그제 대비 변화

# 변화율
df['pct_1'] = df['value'].shift(1).pct_change()

# 24시간 전 대비 변화
df['diff_24'] = df['value'].shift(1) - df['value'].shift(25)
```

---

### 1.8 비율 특성

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

### 1.9 외부 특성 추가

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

### 1.10 특성 선택 전략

| 전략 | 설명 |
|------|------|
| 도메인 지식 기반 | "어제 생산량이 오늘에 영향" -> lag_1 |
| 실험적 접근 | 다양한 특성 생성 후 성능 비교 |
| 과적합 주의 | 특성 너무 많으면 과적합 위험, 교차검증으로 검증 |

---

### 1.11 실습: 특성 엔지니어링 (날짜 특성)

```python
import numpy as np
import pandas as pd

# AirPassengers 데이터 로드
try:
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
except:
    np.random.seed(42)
    n = 144
    dates = pd.date_range(start='1949-01', periods=n, freq='MS')
    trend = np.linspace(100, 400, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 20, n)
    passengers = trend + seasonal + noise
    df = pd.DataFrame({'Month': dates.strftime('%Y-%m'), 'Passengers': passengers.astype(int)})

df.columns = ['Month', 'Passengers']
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

# 특성 생성
features = pd.DataFrame(index=df.index)

# 기본 날짜 특성
features['year'] = df.index.year
features['month'] = df.index.month
features['quarter'] = df.index.quarter

# 주기적 인코딩
features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

# 분기 인코딩
features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)

# 시간 순서 (추세 반영용)
features['time_idx'] = np.arange(len(df))

print("[날짜 특성]")
print(features.head(12))
```

---

### 1.12 실습: 특성 엔지니어링 (Lag + Rolling)

```python
# Lag 특성 (반드시 shift로 미래 누출 방지!)
lag_months = [1, 2, 3, 6, 12]

for lag in lag_months:
    features[f'lag_{lag}'] = df['Passengers'].shift(lag)

print("\n[Lag 특성]")
print(f"생성된 Lag: {lag_months}")
print(features[[f'lag_{lag}' for lag in lag_months]].tail(10))

# Rolling 특성 (shift 먼저!)
windows = [3, 6, 12]

for window in windows:
    # 이동평균
    features[f'ma_{window}'] = df['Passengers'].shift(1).rolling(window).mean()
    # 이동 표준편차
    features[f'std_{window}'] = df['Passengers'].shift(1).rolling(window).std()

# 추가 Rolling 특성
features['max_12'] = df['Passengers'].shift(1).rolling(12).max()
features['min_12'] = df['Passengers'].shift(1).rolling(12).min()
features['range_12'] = features['max_12'] - features['min_12']

print("\n[Rolling 특성]")
print(features[['ma_3', 'ma_12', 'std_12', 'max_12', 'min_12']].tail(10))
```

---

### 1.13 실습: 특성 엔지니어링 (변화 특성)

```python
# 차분 (변화량) - shift 필수
features['diff_1'] = df['Passengers'].shift(1) - df['Passengers'].shift(2)
features['diff_12'] = df['Passengers'].shift(1) - df['Passengers'].shift(13)

# 변화율
features['pct_1'] = df['Passengers'].shift(1).pct_change()

# 전년 동월 대비 비율
features['yoy_ratio'] = df['Passengers'].shift(1) / df['Passengers'].shift(13)

print("\n[변화 특성]")
print(features[['diff_1', 'diff_12', 'pct_1', 'yoy_ratio']].tail(10))
```

---

## Part 2: ML 모델로 시계열 예측

### 2.1 시계열 예측 접근법

| 방법 | 설명 |
|------|------|
| 전통적 방법 | ARIMA, Exponential Smoothing (시계열 전용 모델) |
| ML 기반 방법 | 특성 엔지니어링 후 일반 ML 모델 적용 |

**ML 기반 방법의 장점**
- 다양한 특성 활용 가능
- RandomForest, XGBoost 등 강력한 모델 사용 가능

---

### 2.2 ML 모델 적용 흐름

```
1. 특성 엔지니어링
   -> Lag, Rolling, 날짜 특성 생성

2. NaN 제거
   -> shift/rolling으로 생긴 NaN 처리

3. 시간 기준 분할
   -> 과거=학습, 미래=테스트

4. 모델 학습/예측
   -> fit(), predict()

5. 평가
   -> MAE, RMSE, MAPE
```

---

### 2.3 RandomForest로 시계열 예측

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

### 2.4 왜 RandomForest가 좋은가?

**장점**
| 장점 | 설명 |
|------|------|
| 비선형 관계 | 비선형 관계 학습 가능 |
| 특성 중요도 | 자동 산출 |
| 과적합에 강함 | 앙상블 효과 |
| 튜닝 쉬움 | 하이퍼파라미터 튜닝이 쉬움 |

**단점**
- 순차 의존성 직접 학습 불가 -> Lag 특성으로 보완!

---

### 2.5 특성 중요도 분석

```python
# 특성 중요도
importances = model.feature_importances_
feature_names = X_train.columns

# 정렬
sorted_idx = importances.argsort()[::-1]

print("특성 중요도:")
for i in sorted_idx[:10]:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")

# 보통 lag_1, lag_12 등 Lag 특성이 높게 나옴
```

---

### 2.6 다른 모델 비교

| 모델 | 특징 | 시계열 적합도 |
|------|------|--------------|
| LinearRegression | 단순, 해석 쉬움 | 낮음 |
| RandomForest | 비선형, 강건 | 높음 |
| GradientBoosting | 성능 좋음 | 높음 |
| XGBoost | 빠름, 성능 좋음 | 매우 높음 |
| LightGBM | 대용량 빠름 | 매우 높음 |

---

### 2.7 다단계 예측 (Multi-step)

| 방법 | 설명 |
|------|------|
| 1단계 예측 | t+1 시점만 예측 |
| 다단계 예측 | t+1, t+2, ..., t+n 예측 |

**다단계 예측 방법**
| 방법 | 설명 |
|------|------|
| Recursive | 예측값을 다음 입력으로 |
| Direct | 각 horizon별 별도 모델 |
| Multi-output | 한 모델이 여러 출력 |

---

### 2.8 TimeSeriesSplit 교차검증

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

print(f"평균 R^2: {np.mean(scores):.3f}")
```

---

### 2.9 실습: 데이터 준비 및 분할

```python
# 타겟
target = df['Passengers']

# NaN 제거
features_clean = features.dropna()
target_clean = target.loc[features_clean.index]

print(f"[NaN 제거]")
print(f"원본: {len(features)}개")
print(f"정제 후: {len(features_clean)}개")
print(f"제거된 행: {len(features) - len(features_clean)}개")

# 시간 기준 분할 (마지막 2년을 테스트로)
split_date = '1958-12-01'

X_train = features_clean[:split_date]
y_train = target_clean[:split_date]
X_test = features_clean[split_date:]
y_test = target_clean[split_date:]

print(f"\n[시간 기준 분할]")
print(f"분할 기준: {split_date}")
print(f"학습: {len(X_train)}개 ({X_train.index.min()} ~ {X_train.index.max()})")
print(f"테스트: {len(X_test)}개 ({X_test.index.min()} ~ {X_test.index.max()})")
```

---

### 2.10 실습: 모델 학습 및 예측

```python
from sklearn.ensemble import RandomForestRegressor

# RandomForest 모델
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 학습
model_rf.fit(X_train, y_train)

# 예측
y_pred_rf = model_rf.predict(X_test)

print("[RandomForest 학습 완료]")
print(f"학습 R^2: {model_rf.score(X_train, y_train):.4f}")
print(f"테스트 R^2: {model_rf.score(X_test, y_test):.4f}")
```

---

## Part 3: 시계열 예측 평가

### 3.1 시계열 평가 지표

| 지표 | 공식 | 특징 |
|------|------|------|
| MAE | mean(\|y-y_hat\|) | 해석 쉬움 |
| RMSE | sqrt(mean((y-y_hat)^2)) | 큰 오차 패널티 |
| MAPE | mean(\|y-y_hat\|/y) x 100 | 상대적 오차 (%) |
| R^2 | 1 - SS_res/SS_tot | 설명력 |

---

### 3.2 MAE (Mean Absolute Error)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
```

**해석**
- 평균적으로 mae만큼 틀림
- 단위가 원래 데이터와 같아 해석 쉬움
- MAE = 50이면 "평균 50개 오차"

---

### 3.3 RMSE (Root Mean Squared Error)

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
```

**특징**
- 큰 오차에 더 큰 패널티
- MAE보다 이상치에 민감
- 큰 오차가 중요하면 RMSE 사용

---

### 3.4 MAPE (Mean Absolute Percentage Error)

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape(y_test, y_pred):.1f}%")
```

**특징**
- 퍼센트로 표현 -> 직관적
- MAPE = 5%: "평균 5% 오차"
- 주의: y=0이면 계산 불가

---

### 3.5 지표 선택 가이드

| 상황 | 권장 지표 |
|------|----------|
| 일반적인 경우 | MAE, RMSE |
| 큰 오차가 치명적 | RMSE |
| 상대적 오차 중요 | MAPE |
| 비즈니스 보고 | MAPE (% 직관적) |
| 0에 가까운 값 있음 | MAE, RMSE (MAPE 사용 불가) |

---

### 3.6 실습: 평가 지표 계산

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """MAPE 계산 (0 제외)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 평가 지표
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
mape = calculate_mape(y_test.values, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("[평가 지표]")
print(f"MAE:  {mae:.2f} (평균 절대 오차)")
print(f"RMSE: {rmse:.2f} (평균 제곱근 오차)")
print(f"MAPE: {mape:.2f}% (평균 절대 비율 오차)")
print(f"R^2:   {r2:.4f} (결정계수)")

# 해석
print("\n[지표 해석]")
print(f"-> 평균적으로 {mae:.0f}명 정도 오차")
print(f"-> 상대적으로 {mape:.1f}% 오차")
if mape < 5:
    print("-> 매우 좋은 예측 성능!")
elif mape < 10:
    print("-> 좋은 예측 성능")
elif mape < 20:
    print("-> 보통 수준의 예측 성능")
else:
    print("-> 개선 필요")
```

---

### 3.7 실습: 특성 중요도 분석

```python
# 특성 중요도
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("[특성 중요도 Top 15]")
print(importance_df.head(15).to_string(index=False))
```

**결과 해설**
- 보통 lag_1, lag_12 등 Lag 특성이 높게 나옴
- 계절성이 있으면 month 관련 특성도 중요함

---

### 3.8 실습: 모델 비교

```python
from sklearn.linear_model import LinearRegression

# 선형회귀 비교
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mape_lr = calculate_mape(y_test.values, y_pred_lr)

print("[모델 비교]")
print(f"{'모델':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-" * 54)
print(f"{'LinearRegression':<20} {mae_lr:>10.2f} {rmse_lr:>10.2f} {mape_lr:>9.2f}%")
print(f"{'RandomForest':<20} {mae:>10.2f} {rmse:>10.2f} {mape:>9.2f}%")

if mape_lr > 0:
    improvement = (mape_lr - mape) / mape_lr * 100
    print(f"\n-> RandomForest가 MAPE 기준 {improvement:.1f}% 개선")
```

---

### 3.9 잔차 분석

```python
residuals = y_test.values - y_pred_rf

print("[잔차 통계]")
print(f"평균: {np.mean(residuals):.2f} (0에 가까워야 함)")
print(f"표준편차: {np.std(residuals):.2f}")
print(f"최소: {np.min(residuals):.2f}")
print(f"최대: {np.max(residuals):.2f}")
```

**확인 사항**
- 잔차가 0 주변에 분포하는지
- 시간에 따른 패턴이 없는지

---

### 3.10 예측 구간

```python
# 잔차 표준편차 기반 예측 구간
std_residual = np.std(residuals)
confidence = 1.96  # 95% 신뢰구간

lower = y_pred_rf - confidence * std_residual
upper = y_pred_rf + confidence * std_residual

# 실제값이 구간 내에 있는 비율
in_interval = np.mean((y_test.values >= lower) & (y_test.values <= upper))

print(f"[95% 예측 구간]")
print(f"잔차 표준편차: {std_residual:.2f}")
print(f"실제값이 구간 내 비율: {in_interval:.1%}")
```

---

### 3.11 실습: 시각화

```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 실제 vs 예측
ax1 = axes[0, 0]
ax1.plot(y_test.index, y_test.values, label='실제', alpha=0.7, linewidth=2)
ax1.plot(y_test.index, y_pred_rf, label='예측', alpha=0.7, linewidth=2)
ax1.set_title(f'실제 vs 예측 (MAPE: {mape:.1f}%)')
ax1.set_xlabel('연도')
ax1.set_ylabel('승객 수')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 특성 중요도 Top 10
ax2 = axes[0, 1]
top10 = importance_df.head(10)
ax2.barh(range(10), top10['importance'].values, color='steelblue')
ax2.set_yticks(range(10))
ax2.set_yticklabels(top10['feature'].values)
ax2.set_xlabel('중요도')
ax2.set_title('특성 중요도 Top 10')
ax2.invert_yaxis()

# 3. 잔차 분포
ax3 = axes[1, 0]
ax3.hist(residuals, bins=15, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('잔차 (실제 - 예측)')
ax3.set_ylabel('빈도')
ax3.set_title('잔차 분포')

# 4. 실제 vs 예측 산점도
ax4 = axes[1, 1]
ax4.scatter(y_test.values, y_pred_rf, alpha=0.6)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2)
ax4.set_xlabel('실제 승객 수')
ax4.set_ylabel('예측 승객 수')
ax4.set_title(f'실제 vs 예측 (R^2 = {r2:.4f})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3.12 실습: TimeSeriesSplit 교차검증

```python
from sklearn.model_selection import TimeSeriesSplit

# 전체 데이터로 교차검증
X_full = features_clean
y_full = target_clean

tscv = TimeSeriesSplit(n_splits=5)

print("[TimeSeriesSplit 교차검증]")
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), 1):
    X_tr = X_full.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_full.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    model_cv = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model_cv.fit(X_tr, y_tr)
    score = model_cv.score(X_val, y_val)
    cv_scores.append(score)

    print(f"Fold {fold}: R^2 = {score:.4f} "
          f"(학습: {len(train_idx)}, 검증: {len(val_idx)})")

print(f"\n평균 R^2: {np.mean(cv_scores):.4f} (+/-{np.std(cv_scores):.4f})")
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 특성 엔지니어링 | 날짜, Lag, Rolling, 변화 특성 생성 |
| shift(1) 필수 | 미래 누출 방지를 위해 반드시 사용 |
| ML 모델 | RandomForest, XGBoost 등 활용 |
| TimeSeriesSplit | 시계열 교차검증 |
| MAE | 평균 절대 오차 (해석 쉬움) |
| RMSE | 평균 제곱근 오차 (큰 오차 패널티) |
| MAPE | 평균 절대 비율 오차 (% 단위) |

---

## sklearn 주요 함수 요약

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 모델 학습
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 특성 중요도
importance = model.feature_importances_

# 평가 지표
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 시계열 교차검증
tscv = TimeSeriesSplit(n_splits=5)
```
