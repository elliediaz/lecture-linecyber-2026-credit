# 19차시: 시계열 데이터 분석과 예측

## 학습 목표

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | 시간에 따라 변하는 데이터의 특성을 이해함 |
| 2 | 시계열 특성 엔지니어링을 수행함 |
| 3 | ML 모델로 제조 설비 상태를 예측함 |

---

## 강의 구성 (25분)

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 시계열 데이터의 이해 | 시간 의존성, 계절성, 추세, datetime 처리 |
| 2 | 시계열 특성 엔지니어링 | Lag 특성, 이동평균 특성, 날짜 특성 |
| 3 | ML 기반 예측 및 평가 | RandomForest 예측, MAE/RMSE/MAPE 평가 |

---

## 수학적 배경

### 이동평균 (Moving Average)

시계열 데이터의 노이즈를 제거하고 추세를 파악하는 기법임.

$$MA_t = \frac{1}{n} \sum_{i=0}^{n-1} X_{t-i}$$

- $MA_t$: 시점 t에서의 이동평균
- $n$: 윈도우 크기
- $X_{t-i}$: 과거 i 시점의 값

---

### 지수평활 (Exponential Smoothing)

최근 관측값에 더 큰 가중치를 부여하는 평활 기법임.

$$S_t = \alpha X_t + (1-\alpha) S_{t-1}$$

- $S_t$: 시점 t에서의 평활값
- $\alpha$: 평활 계수 (0 < $\alpha$ < 1)
- $X_t$: 현재 관측값
- $S_{t-1}$: 이전 평활값

$\alpha$가 클수록 최근 값의 영향이 커지고, 작을수록 과거 값의 영향이 오래 지속됨.

---

### 자기상관함수 (ACF: Autocorrelation Function)

시차 k에서 시계열과 자기 자신 간의 상관관계를 측정함.

$$\rho_k = \frac{\sum_{t=k+1}^{n}(X_t - \bar{X})(X_{t-k} - \bar{X})}{\sum_{t=1}^{n}(X_t - \bar{X})^2}$$

- $\rho_k$: 시차 k에서의 자기상관계수 (-1 ~ 1)
- $\bar{X}$: 시계열의 평균
- $n$: 전체 관측치 수

$\rho_k$가 크면 k 시점 전 값과 현재 값 간에 강한 상관관계가 있음을 의미함.

---

## Part 1: 시계열 데이터의 이해

### 1.1 시계열 데이터란?

**정의**
- 시간 순서로 기록된 데이터임
- 각 데이터 포인트에 타임스탬프가 존재함

**제조 현장 예시**

| 데이터 | 주기 | 활용 |
|-------|:----:|------|
| 센서 온도 | 초/분 | 이상 탐지 |
| 생산량 | 시간/일 | 수요 예측 |
| 불량률 | 일/주 | 품질 관리 |
| 설비 가동 로그 | 이벤트 | 예지 정비 |
| 에너지 사용량 | 분/시간 | 비용 최적화 |

---

### 1.2 시계열의 구성 요소

```
실제 데이터 = 추세 + 계절성 + 잔차

+----------------------------------+
| 원본: ~~~^~~~^~~~^~~~^~~~^~~~    |
|                                  |
| 추세: -----------------/         | (장기 방향)
| 계절성: ^  ^  ^  ^  ^  ^         | (반복 패턴)
| 잔차: ~~~~~~~~~~~~~~~~~~~~~~     | (불규칙 변동)
+----------------------------------+
```

| 구성요소 | 설명 | 예시 |
|---------|------|------|
| 추세 (Trend) | 장기적인 증가/감소 경향 | 설비 노후화로 불량률 증가 |
| 계절성 (Seasonality) | 일정 주기로 반복되는 패턴 | 여름철 에너지 사용 증가 |
| 잔차 (Residual) | 추세와 계절성으로 설명 안 되는 부분 | 돌발 상황, 노이즈 |

---

### 1.3 시계열 데이터의 특성

| 특성 | 설명 | 예시 |
|------|------|------|
| 시간 의존성 | 현재 값이 과거 값에 영향받음 | 오늘 생산량 <- 어제 생산량 |
| 자기상관 | 자기 자신과의 상관관계 | Lag 1 자기상관: 오늘 <-> 어제 |
| 계절성 | 주기적 패턴 | 월요일마다 높음, 12월마다 급증 |

---

### 1.4 일반 데이터 vs 시계열 데이터

| 일반 데이터 | 시계열 데이터 |
|------------|--------------|
| 행(row)이 독립 | 행 간에 순서 존재 |
| 랜덤 셔플 가능 | 셔플하면 의미 손실 |
| 랜덤 분할 가능 | 시간 기준 분할 필수 |
| K-Fold 교차검증 | TimeSeriesSplit 필수 |

---

### 1.5 실습: AirPassengers 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Colab/로컬 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# AirPassengers 데이터셋 로드
try:
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    print("[데이터 로드 성공]")
except Exception as e:
    print(f"[데이터 로드 실패: {e}]")
    # 오프라인 대비 샘플 데이터 생성
    np.random.seed(42)
    n = 144
    dates = pd.date_range(start='1949-01', periods=n, freq='MS')
    trend = np.linspace(100, 400, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 20, n)
    passengers = trend + seasonal + noise
    df = pd.DataFrame({'Month': dates.strftime('%Y-%m'), 'Passengers': passengers.astype(int)})

# 열 이름 표준화
df.columns = ['Month', 'Passengers']

# 날짜 파싱 및 인덱스 설정
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"기간: {df.index.min()} ~ {df.index.max()}")
print(f"\n첫 5행:")
print(df.head())
```

---

### 1.6 실습: 시계열 시각화

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. 원본 시계열
ax1 = axes[0, 0]
df['Passengers'].plot(ax=ax1, linewidth=1.5)
ax1.set_title('AirPassengers - 원본 시계열')
ax1.set_xlabel('연도')
ax1.set_ylabel('승객 수')
ax1.grid(True, alpha=0.3)

# 2. 연도별 승객 수
ax2 = axes[0, 1]
yearly = df.resample('YE').sum()
yearly['Passengers'].plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('연도별 총 승객 수')
ax2.set_xlabel('연도')
ax2.set_ylabel('승객 수')
ax2.set_xticklabels([d.year for d in yearly.index], rotation=45)

# 3. 월별 평균 (계절성 확인)
ax3 = axes[1, 0]
monthly_avg = df.groupby(df.index.month)['Passengers'].mean()
monthly_avg.plot(kind='bar', ax=ax3, color='coral')
ax3.set_title('월별 평균 승객 수 (계절성)')
ax3.set_xlabel('월')
ax3.set_ylabel('평균 승객 수')
ax3.set_xticklabels(range(1, 13), rotation=0)

# 4. 12개월 이동평균 (추세)
ax4 = axes[1, 1]
df['Passengers'].plot(ax=ax4, alpha=0.5, label='원본')
df['Passengers'].rolling(12).mean().plot(ax=ax4, linewidth=2, label='12개월 이동평균')
ax4.set_title('원본 vs 12개월 이동평균 (추세)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n[시계열 특성 분석]")
print("- 추세: 전체적으로 우상향 (승객 수 증가)")
print("- 계절성: 7-8월(여름)에 높고, 11월에 낮음")
print("- 변동성: 시간이 지날수록 변동폭 증가")
```

---

### 1.7 실습: pandas datetime 처리

```python
# 인덱스에서 날짜 정보 추출
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter

print("[날짜 정보 추출]")
print(df[['Passengers', 'year', 'month', 'quarter']].head(12))

# 슬라이싱 예시
print("\n[날짜 슬라이싱]")
print("1950년 데이터:")
print(df['1950'].head())

print("\n1955-01 ~ 1955-06 데이터:")
print(df['1955-01':'1955-06'])
```

---

### 1.8 실습: resample을 이용한 주기 변환

```python
# 월별 -> 분기별 평균
quarterly_mean = df[['Passengers']].resample('QE').mean()
print("[분기별 평균 (월별 -> 분기별)]")
print(f"원본 shape: {df.shape}")
print(f"분기별 shape: {quarterly_mean.shape}")
print(quarterly_mean.head(8))

# 월별 -> 연도별 통계
print("\n[연도별 다양한 집계]")
yearly_agg = df['Passengers'].resample('YE').agg(['mean', 'max', 'min', 'std'])
print(yearly_agg)
```

---

## Part 2: 시계열 특성 엔지니어링

### 2.1 특성 엔지니어링이란?

**정의**
- 원본 데이터에서 예측에 유용한 특성을 생성함
- 모델 성능에 가장 큰 영향을 미침

**시계열에서 특히 중요한 이유**
- 단순히 과거 값만으로는 부족함
- 패턴을 담은 특성을 만들어야 함

---

### 2.2 시계열 특성의 종류

| 종류 | 설명 | 예시 |
|------|------|------|
| 날짜 특성 | 시간에서 추출 | hour, dayofweek, month |
| Lag 특성 | 과거 값 | lag_1, lag_12 |
| Rolling 특성 | 이동 통계 | ma_3, ma_12, std_12 |
| 변화 특성 | 변화량/변화율 | diff, pct_change |

---

### 2.3 실습: 날짜 특성 생성

```python
# 특성 DataFrame 생성
features = pd.DataFrame(index=df.index)

# 기본 날짜 특성
features['year'] = df.index.year
features['month'] = df.index.month
features['quarter'] = df.index.quarter

# 주기적 인코딩 (월의 순환 특성 반영)
# 12월과 1월이 가깝다는 것을 모델이 인식할 수 있음
features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

# 분기 인코딩
features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)

# 시간 순서 인덱스 (추세 반영용)
features['time_idx'] = np.arange(len(df))

print("[날짜 특성]")
print(features.head(12))
```

---

### 2.4 실습: Lag 특성 생성

```python
# Lag 특성 (반드시 shift로 미래 누출 방지!)
lag_months = [1, 2, 3, 6, 12]

for lag in lag_months:
    features[f'lag_{lag}'] = df['Passengers'].shift(lag)

print("[Lag 특성]")
print(f"생성된 Lag: {lag_months}")
print(features[[f'lag_{lag}' for lag in lag_months]].tail(10))

# NaN 개수 확인
print("\n[NaN 개수]")
for lag in lag_months:
    nan_count = features[f'lag_{lag}'].isna().sum()
    print(f"  lag_{lag}: {nan_count}개")
```

---

### 2.5 Lag 특성 시각화

```
원본:     [A, B, C, D, E]
shift(1): [NaN, A, B, C, D]  <- 1시점 전 값
shift(2): [NaN, NaN, A, B, C]  <- 2시점 전 값
shift(-1): [B, C, D, E, NaN] <- 미래 값 (학습 시 사용 금지!)

활용 예시:
df['lag_1'] = df['Passengers'].shift(1)
-> "이번 달 승객 수 예측에 지난 달 승객 수 사용"
```

---

### 2.6 실습: Rolling 특성 생성

```python
# Rolling 특성 (shift 먼저 적용하여 미래 누출 방지!)
windows = [3, 6, 12]

for window in windows:
    # 이동평균 - shift(1) 먼저!
    features[f'ma_{window}'] = df['Passengers'].shift(1).rolling(window).mean()
    # 이동 표준편차
    features[f'std_{window}'] = df['Passengers'].shift(1).rolling(window).std()

# 추가 Rolling 특성
features['max_12'] = df['Passengers'].shift(1).rolling(12).max()
features['min_12'] = df['Passengers'].shift(1).rolling(12).min()
features['range_12'] = features['max_12'] - features['min_12']

print("[Rolling 특성]")
print(features[['ma_3', 'ma_12', 'std_12', 'max_12', 'min_12']].tail(10))
```

---

### 2.7 데이터 누출 방지의 중요성

```python
# 잘못된 방법: 현재 값이 포함됨
wrong_ma = df['Passengers'].rolling(3).mean()

# 올바른 방법: shift 먼저 적용
correct_ma = df['Passengers'].shift(1).rolling(3).mean()

# 비교
comparison = pd.DataFrame({
    'Passengers': df['Passengers'],
    'wrong_ma_3': wrong_ma,
    'correct_ma_3': correct_ma
})

print("[잘못된 vs 올바른 Rolling]")
print(comparison.iloc[3:9])

print("\n[주의사항]")
print("  - wrong_ma_3: 현재 승객수가 평균에 포함됨 (미래 누출!)")
print("  - correct_ma_3: 이전 달까지의 데이터만 사용 (안전)")
```

---

### 2.8 실습: 변화 특성 생성

```python
# 차분 (변화량) - shift 필수
features['diff_1'] = df['Passengers'].shift(1) - df['Passengers'].shift(2)
features['diff_12'] = df['Passengers'].shift(1) - df['Passengers'].shift(13)

# 변화율
features['pct_1'] = df['Passengers'].shift(1).pct_change()

# 전년 동월 대비 비율
features['yoy_ratio'] = df['Passengers'].shift(1) / df['Passengers'].shift(13)

print("[변화 특성]")
print(features[['diff_1', 'diff_12', 'pct_1', 'yoy_ratio']].tail(10))
```

---

### 2.9 특성 요약

```python
# 타겟 추가
features['target'] = df['Passengers']

print("[전체 특성 목록]")
print(f"총 특성 수: {len(features.columns) - 1}개 (타겟 제외)")
print(f"\n특성 목록:")
for i, col in enumerate(features.columns[:-1], 1):
    print(f"  {i:2d}. {col}")
```

---

## Part 3: ML 기반 예측 및 평가

### 3.1 ML 모델 적용 흐름

```
1. 특성 엔지니어링
   -> Lag, Rolling, 날짜 특성 생성

2. NaN 제거
   -> shift/rolling으로 생긴 NaN 처리

3. 시간 기준 분할
   -> 과거=학습, 미래=테스트 (랜덤 분할 금지!)

4. 모델 학습/예측
   -> fit(), predict()

5. 평가
   -> MAE, RMSE, MAPE
```

---

### 3.2 실습: 데이터 준비 및 분할

```python
# 타겟
target = df['Passengers']

# NaN 제거
features_clean = features.dropna()
target_clean = target.loc[features_clean.index]

print("[NaN 제거]")
print(f"원본: {len(features)}개")
print(f"정제 후: {len(features_clean)}개")
print(f"제거된 행: {len(features) - len(features_clean)}개")

# 시간 기준 분할 (마지막 2년을 테스트로)
split_date = '1958-12-01'

X_train = features_clean[:split_date].drop('target', axis=1)
y_train = target_clean[:split_date]
X_test = features_clean[split_date:].drop('target', axis=1)
y_test = target_clean[split_date:]

print(f"\n[시간 기준 분할]")
print(f"분할 기준: {split_date}")
print(f"학습: {len(X_train)}개 ({X_train.index.min().strftime('%Y-%m')} ~ {X_train.index.max().strftime('%Y-%m')})")
print(f"테스트: {len(X_test)}개 ({X_test.index.min().strftime('%Y-%m')} ~ {X_test.index.max().strftime('%Y-%m')})")
```

---

### 3.3 실습: RandomForest 모델 학습

```python
from sklearn.ensemble import RandomForestRegressor

# RandomForest 모델
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

print("[RandomForest 학습 완료]")
print(f"학습 R^2: {model.score(X_train, y_train):.4f}")
print(f"테스트 R^2: {model.score(X_test, y_test):.4f}")
```

---

### 3.4 시계열 평가 지표

| 지표 | 공식 | 특징 |
|------|------|------|
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 해석 쉬움, 단위 동일 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | 큰 오차에 패널티 |
| MAPE | $\frac{100}{n}\sum\frac{\|y - \hat{y}\|}{y}$ | 상대적 오차 (%) |
| R^2 | $1 - \frac{SS_{res}}{SS_{tot}}$ | 설명력 (0~1) |

---

### 3.5 실습: 평가 지표 계산

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """MAPE 계산 (0 제외)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 평가 지표
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = calculate_mape(y_test.values, y_pred)
r2 = r2_score(y_test, y_pred)

print("[평가 지표]")
print(f"MAE:  {mae:.2f} (평균 절대 오차)")
print(f"RMSE: {rmse:.2f} (평균 제곱근 오차)")
print(f"MAPE: {mape:.2f}% (평균 절대 비율 오차)")
print(f"R^2:  {r2:.4f} (결정계수)")

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

### 3.6 실습: 특성 중요도 분석

```python
# 특성 중요도
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("[특성 중요도 Top 10]")
print(importance_df.head(10).to_string(index=False))

print("\n[결과 해석]")
print("- 일반적으로 lag_1, lag_12 등 Lag 특성이 높게 나옴")
print("- 계절성이 있으면 month 관련 특성도 중요함")
```

---

### 3.7 실습: 선형회귀와 비교

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

### 3.8 실습: 잔차 분석

```python
residuals = y_test.values - y_pred

print("[잔차 통계]")
print(f"평균: {np.mean(residuals):.2f} (0에 가까워야 함)")
print(f"표준편차: {np.std(residuals):.2f}")
print(f"최소: {np.min(residuals):.2f}")
print(f"최대: {np.max(residuals):.2f}")

# 잔차 표준편차 기반 95% 예측 구간
std_residual = np.std(residuals)
confidence = 1.96  # 95% 신뢰구간

lower = y_pred - confidence * std_residual
upper = y_pred + confidence * std_residual

# 실제값이 구간 내에 있는 비율
in_interval = np.mean((y_test.values >= lower) & (y_test.values <= upper))

print(f"\n[95% 예측 구간]")
print(f"잔차 표준편차: {std_residual:.2f}")
print(f"실제값이 구간 내 비율: {in_interval:.1%}")
```

---

### 3.9 실습: 결과 시각화

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 실제 vs 예측
ax1 = axes[0, 0]
ax1.plot(y_test.index, y_test.values, label='실제', alpha=0.7, linewidth=2)
ax1.plot(y_test.index, y_pred, label='예측', alpha=0.7, linewidth=2)
ax1.fill_between(y_test.index, lower, upper, alpha=0.2, label='95% 구간')
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
ax4.scatter(y_test.values, y_pred, alpha=0.6)
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

### 3.10 실습: TimeSeriesSplit 교차검증

```python
from sklearn.model_selection import TimeSeriesSplit

# 전체 정제 데이터로 교차검증
X_full = features_clean.drop('target', axis=1)
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

print(f"\n평균 R^2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
```

---

### 3.11 TimeSeriesSplit 시각화

```python
# TimeSeriesSplit 시각화
tscv = TimeSeriesSplit(n_splits=5)

fig, ax = plt.subplots(figsize=(12, 4))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
    # 학습 데이터
    ax.scatter(train_idx, [fold] * len(train_idx), c='blue', s=5, label='학습' if fold == 0 else None)
    # 검증 데이터
    ax.scatter(val_idx, [fold] * len(val_idx), c='red', s=5, label='검증' if fold == 0 else None)

ax.set_xlabel('데이터 인덱스 (시간순)')
ax.set_ylabel('Fold')
ax.set_yticks(range(5))
ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_title('TimeSeriesSplit 교차검증')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("[TimeSeriesSplit 특징]")
print("- 항상 과거 데이터로 미래를 예측함")
print("- Fold가 진행될수록 학습 데이터가 증가함")
print("- 시계열의 시간적 순서를 유지함")
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 시계열 특성 | 시간 의존성, 자기상관, 계절성 |
| 이동평균 | $MA_t = \frac{1}{n}\sum X_{t-i}$, 노이즈 제거 |
| 지수평활 | $S_t = \alpha X_t + (1-\alpha)S_{t-1}$, 최근값 가중 |
| ACF | 시차별 자기상관 계수 |
| 시간 기준 분할 | 랜덤 분할 금지, 순차 분할 필수 |
| shift(1) 필수 | 미래 누출 방지 |
| MAE | 평균 절대 오차 (해석 쉬움) |
| RMSE | 큰 오차에 패널티 |
| MAPE | 상대적 오차 (% 단위) |
| TimeSeriesSplit | 시계열 전용 교차검증 |

---

## Pandas 시계열 핵심 함수

```python
# 날짜 변환
df['date'] = pd.to_datetime(df['date'])

# 날짜 정보 추출 (dt 접근자)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek

# 주기 변환 (resample)
df.resample('W').mean()   # 주별 평균
df.resample('M').sum()    # 월별 합계

# 이동 통계 (rolling)
df['ma_7'] = df['value'].shift(1).rolling(7).mean()  # shift 먼저!

# 시차 변수 (shift)
df['lag_1'] = df['value'].shift(1)   # 1시점 전

# 차분 (diff)
df['diff_1'] = df['value'].diff(1)   # 1시점 차이
```

---

## sklearn 시계열 핵심 함수

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
for train_idx, val_idx in tscv.split(X):
    # 항상 과거 -> 미래 순서 유지
    pass
```

---

## 데이터 누출 방지 체크리스트

```python
# 1. Lag 특성 생성 시 shift 사용
df['lag_1'] = df['value'].shift(1)  # O

# 2. Rolling 특성 생성 시 shift 먼저
df['ma_7'] = df['value'].shift(1).rolling(7).mean()  # O
df['ma_7'] = df['value'].rolling(7).mean()  # X (미래 포함!)

# 3. 변화 특성도 shift 적용
df['diff_1'] = df['value'].shift(1) - df['value'].shift(2)  # O

# 4. 시간 기준 분할 (랜덤 분할 금지!)
X_train = features[:split_date]  # O
X_test = features[split_date:]   # O

# 5. TimeSeriesSplit 사용 (K-Fold 금지!)
from sklearn.model_selection import TimeSeriesSplit  # O
```

이 규칙을 항상 준수해야 함!
