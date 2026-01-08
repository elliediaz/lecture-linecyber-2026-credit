---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [18차시] 시계열 데이터 기초

## 공공데이터 AI 예측 모델 개발

---

# 학습 목표

1. **시계열 데이터의 특성**을 이해한다
2. **Python datetime 처리**를 수행한다
3. **시계열 전처리 기법**을 활용한다

---

# 목차

## 대주제 1: 시계열 데이터의 특성
## 대주제 2: Python datetime 처리
## 대주제 3: 시계열 전처리 기법
## 실습편: 제조 센서 시계열 분석

---

<!-- _class: lead -->
# 대주제 1
## 시계열 데이터의 특성을 이해한다

---

# 시계열 데이터란?

## 정의
- **시간 순서**로 기록된 데이터
- 각 데이터 포인트에 **타임스탬프** 존재

## 예시
- 주식 가격 (매일)
- 센서 측정값 (매초)
- 월별 매출 (매월)
- 설비 가동 로그 (이벤트 발생 시)

---

# 일반 데이터 vs 시계열 데이터

| 일반 데이터 | 시계열 데이터 |
|------------|--------------|
| 행(row)이 독립 | 행 간에 **순서** 존재 |
| 랜덤 셔플 가능 | 셔플하면 **의미 손실** |
| 랜덤 분할 가능 | **시간 기준 분할** 필수 |
| 교차검증 무작위 | **순차 분할** 검증 |

---

# 시계열 데이터의 특성

## 1. 시간 의존성 (Time Dependency)
- 현재 값이 과거 값에 영향받음
- 예: 오늘 생산량 ← 어제 생산량

## 2. 자기상관 (Autocorrelation)
- 자기 자신과의 상관관계
- Lag 1 자기상관: 오늘 ↔ 어제

## 3. 계절성 (Seasonality)
- 주기적 패턴
- 예: 월요일마다 높음, 12월마다 급증

---

# 시계열 구성요소

```
실제 데이터 = 추세 + 계절성 + 잔차

┌──────────────────────────────────┐
│ 원본: ~~~^~~~^~~~^~~~^~~~^~~~    │
│                                  │
│ 추세: ─────────────────/         │ (장기 방향)
│ 계절성: ^  ^  ^  ^  ^  ^         │ (반복 패턴)
│ 잔차: ~~~~~~~~~~~~~~~~~~~~~~     │ (불규칙 변동)
└──────────────────────────────────┘
```

---

# 추세 (Trend)

## 정의
- 장기적인 증가/감소 경향

## 예시
- 설비 노후화로 불량률 점진적 증가
- 공정 개선으로 생산량 점진적 증가

```
시간 →
     /
    /
   /
  /
 /
```

---

# 계절성 (Seasonality)

## 정의
- 일정 주기로 반복되는 패턴

## 예시
- 일별: 오전 생산량 높음
- 주별: 월요일 불량률 높음
- 연별: 여름 에너지 사용량 증가

```
시간 →
^   ^   ^   ^   ^
 \ / \ / \ / \ / \
```

---

# 제조 현장의 시계열 데이터

| 데이터 | 주기 | 활용 |
|-------|------|------|
| 센서 온도 | 초/분 | 이상 탐지 |
| 생산량 | 시간/일 | 수요 예측 |
| 불량률 | 일/주 | 품질 관리 |
| 설비 가동 | 이벤트 | 예지 정비 |
| 에너지 사용 | 분/시간 | 비용 최적화 |

---

# 시계열 예측의 도전

## 일반 ML과 다른 점

1. **데이터 누출 (Data Leakage)**
   - 미래 정보가 학습에 포함되면 안 됨
   - 시간 기준 분할 필수

2. **분할 방식**
   - ❌ 랜덤 분할
   - ✅ 시간 기준 순차 분할

3. **특성 엔지니어링**
   - Lag 특성, Rolling 특성 활용

---

# 시간 기준 분할

```
전체 데이터:
[─────────────|────────|────────]
   학습 (70%)   검증(15%)  테스트(15%)

시간 →

❌ 잘못된 방법: 랜덤 셔플 후 분할
   → 미래 데이터로 과거 예측 (누출!)

✅ 올바른 방법: 시간 순서대로 분할
   → 과거 데이터로 미래 예측
```

---

<!-- _class: lead -->
# 대주제 2
## Python datetime 처리를 수행한다

---

# Python datetime 모듈

```python
from datetime import datetime, timedelta

# 현재 시각
now = datetime.now()
print(now)  # 2025-01-05 14:30:00

# 특정 날짜 생성
dt = datetime(2025, 1, 5, 14, 30)

# 문자열 → datetime
dt = datetime.strptime("2025-01-05", "%Y-%m-%d")

# datetime → 문자열
s = dt.strftime("%Y년 %m월 %d일")
```

---

# datetime 형식 코드

| 코드 | 의미 | 예시 |
|------|------|------|
| %Y | 4자리 연도 | 2025 |
| %m | 2자리 월 | 01 |
| %d | 2자리 일 | 05 |
| %H | 24시간 시 | 14 |
| %M | 분 | 30 |
| %S | 초 | 00 |
| %A | 요일 | Monday |

---

# timedelta로 날짜 계산

```python
from datetime import datetime, timedelta

now = datetime.now()

# 3일 후
future = now + timedelta(days=3)

# 2시간 전
past = now - timedelta(hours=2)

# 날짜 차이 계산
dt1 = datetime(2025, 1, 1)
dt2 = datetime(2025, 1, 10)
diff = dt2 - dt1
print(diff.days)  # 9
```

---

# Pandas 날짜 처리

## pd.to_datetime()

```python
import pandas as pd

# 문자열 → datetime
df['date'] = pd.to_datetime(df['date'])

# 다양한 형식 자동 인식
pd.to_datetime("2025-01-05")
pd.to_datetime("01/05/2025")
pd.to_datetime("Jan 5, 2025")

# 형식 지정
pd.to_datetime("05-01-2025", format="%d-%m-%Y")
```

---

# DatetimeIndex

```python
# 날짜 범위 생성
dates = pd.date_range(
    start='2025-01-01',
    end='2025-01-31',
    freq='D'  # Daily
)

# freq 옵션
# 'D': 일별
# 'H': 시간별
# 'T' 또는 'min': 분별
# 'W': 주별
# 'M': 월말
# 'MS': 월초
```

---

# dt 접근자

```python
# datetime 열에서 정보 추출
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일
df['weekday_name'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 분기, 주차
df['quarter'] = df['date'].dt.quarter
df['week'] = df['date'].dt.isocalendar().week
```

---

# dt 접근자 예시

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=10, freq='D')
})

df['연'] = df['timestamp'].dt.year
df['월'] = df['timestamp'].dt.month
df['일'] = df['timestamp'].dt.day
df['요일'] = df['timestamp'].dt.dayofweek
df['요일명'] = df['timestamp'].dt.day_name()

print(df.head())
```

---

# 날짜를 인덱스로 설정

```python
# 날짜를 인덱스로
df = df.set_index('date')

# 날짜 인덱스의 장점
# 1. 슬라이싱이 편함
df['2025-01']           # 2025년 1월 데이터
df['2025-01-01':'2025-01-15']  # 기간 선택

# 2. resample() 사용 가능
df.resample('W').mean()  # 주별 평균

# 3. 시계열 시각화 자동 지원
df['value'].plot()
```

---

<!-- _class: lead -->
# 대주제 3
## 시계열 전처리 기법을 활용한다

---

# resample() - 주기 변환

## 개념
- 시간 주기를 변환 (다운샘플링/업샘플링)

```python
# 분별 → 시간별 (다운샘플링)
hourly = df.resample('H').mean()

# 일별 → 주별
weekly = df.resample('W').sum()

# 월별 → 분기별
quarterly = df.resample('Q').mean()
```

---

# resample() 집계 방법

```python
# 평균
df.resample('D').mean()

# 합계
df.resample('D').sum()

# 최댓값
df.resample('D').max()

# 여러 통계 한번에
df.resample('D').agg(['mean', 'max', 'min', 'std'])

# 열별 다른 집계
df.resample('D').agg({
    'temperature': 'mean',
    'production': 'sum'
})
```

---

# rolling() - 이동 윈도우

## 개념
- 윈도우를 이동하며 통계 계산

```python
# 7일 이동 평균
df['ma_7'] = df['value'].rolling(window=7).mean()

# 3시간 이동 합계
df['sum_3h'] = df['value'].rolling(window=3).sum()

# 이동 표준편차
df['std_7'] = df['value'].rolling(window=7).std()
```

---

# rolling() 시각화

```
원본 데이터:  [10, 20, 15, 30, 25, 35, 20]

3일 이동평균 계산:
  Day 1-3: (10+20+15)/3 = 15.0
  Day 2-4: (20+15+30)/3 = 21.7
  Day 3-5: (15+30+25)/3 = 23.3
  ...

이동평균:    [NaN, NaN, 15.0, 21.7, 23.3, 30.0, 26.7]

→ 노이즈 제거, 추세 파악에 유용
```

---

# rolling() 옵션

```python
# 기본: 윈도우 내 모든 값 필요
df['value'].rolling(window=7).mean()
# 처음 6개는 NaN

# min_periods: 최소 데이터 수
df['value'].rolling(window=7, min_periods=1).mean()
# 1개만 있어도 계산

# center: 중앙 윈도우
df['value'].rolling(window=7, center=True).mean()
# 현재 값이 윈도우 중앙에 위치
```

---

# shift() - 시차 변수

## 개념
- 데이터를 시간축으로 이동
- Lag 특성 생성에 필수

```python
# 1칸 뒤로 (과거 값)
df['lag_1'] = df['value'].shift(1)

# 3칸 뒤로
df['lag_3'] = df['value'].shift(3)

# 1칸 앞으로 (미래 값) - 주의!
df['future'] = df['value'].shift(-1)
```

---

# shift() 시각화

```
원본:     [A, B, C, D, E]
shift(1): [NaN, A, B, C, D]  ← 과거 값
shift(2): [NaN, NaN, A, B, C]
shift(-1): [B, C, D, E, NaN] ← 미래 값 (학습 시 주의!)

활용:
df['yesterday'] = df['production'].shift(1)
→ "오늘 생산량 예측에 어제 생산량 사용"
```

---

# Lag 특성 생성

```python
# 여러 Lag 특성 한번에
for lag in [1, 2, 3, 7]:
    df[f'lag_{lag}'] = df['production'].shift(lag)

# 결과:
#   date     production  lag_1  lag_2  lag_3  lag_7
# 2025-01-08    100       98     95     102    90
# 2025-01-09    105      100     98      95    92

# → 과거 1일, 2일, 3일, 7일 전 값을 특성으로 사용
```

---

# Rolling 특성 생성

```python
# 이동평균 특성
df['ma_7'] = df['production'].shift(1).rolling(7).mean()
# 주의: shift(1) 먼저! → 미래 정보 누출 방지

# 이동 표준편차 (변동성)
df['std_7'] = df['production'].shift(1).rolling(7).std()

# 이동 최댓값/최솟값
df['max_7'] = df['production'].shift(1).rolling(7).max()
df['min_7'] = df['production'].shift(1).rolling(7).min()
```

---

# 데이터 누출 방지

## ⚠️ 중요!

```python
# ❌ 잘못된 방법 (미래 정보 포함)
df['ma_7'] = df['production'].rolling(7).mean()
# → 오늘 값이 이동평균에 포함됨!

# ✅ 올바른 방법 (shift 먼저)
df['ma_7'] = df['production'].shift(1).rolling(7).mean()
# → 어제까지의 데이터만 사용
```

---

# 날짜 특성 추가

```python
# 날짜에서 유용한 특성 추출
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)

# 주기적 인코딩 (사인/코사인)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---

# diff() - 차분

## 개념
- 연속된 값의 차이
- 추세 제거, 정상성 확보

```python
# 1차 차분
df['diff_1'] = df['value'].diff(1)
# [NaN, v2-v1, v3-v2, v4-v3, ...]

# 7일 차분 (주간 변화)
df['diff_7'] = df['value'].diff(7)
```

---

# pct_change() - 변화율

```python
# 변화율 (퍼센트)
df['pct_change'] = df['value'].pct_change()
# (현재 - 이전) / 이전

# 예: [100, 110, 99]
# pct_change: [NaN, 0.10, -0.10]
# → 10% 증가, 10% 감소
```

---

<!-- _class: lead -->
# 실습편
## 제조 센서 시계열 분석

---

# 실습 개요

## 목표
- 제조 센서 시계열 데이터 분석
- datetime 처리 및 특성 추출
- 시계열 전처리 기법 적용

## 데이터
- 30일간 시간별 센서 데이터
- 온도, 압력, 생산량

---

# 실습 1: 데이터 생성

```python
import pandas as pd
import numpy as np

np.random.seed(42)

# 30일, 시간별 데이터 생성
dates = pd.date_range(
    start='2025-01-01',
    periods=24*30,  # 720시간
    freq='H'
)

df = pd.DataFrame({
    'timestamp': dates,
    'temperature': 85 + np.random.normal(0, 3, len(dates)),
    'pressure': 100 + np.random.normal(0, 5, len(dates)),
    'production': 1000 + np.random.normal(0, 50, len(dates))
})
```

---

# 실습 2: datetime 변환 및 설정

```python
# datetime 확인
print(df['timestamp'].dtype)  # datetime64[ns]

# 날짜를 인덱스로
df = df.set_index('timestamp')

# 인덱스 확인
print(df.index)
# DatetimeIndex(['2025-01-01 00:00:00', ...])

# 기간 슬라이싱
print(df['2025-01-01':'2025-01-03'].head())
```

---

# 실습 3: 날짜 특성 추출

```python
# 인덱스에서 정보 추출
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['month'] = df.index.month
df['day'] = df.index.day

print(df[['hour', 'dayofweek', 'is_weekend']].head(10))
```

---

# 실습 4: resample (주기 변환)

```python
# 시간별 → 일별 평균
daily = df.resample('D').mean()
print("일별 데이터 shape:", daily.shape)
print(daily.head())

# 시간별 → 일별 (여러 통계)
daily_stats = df.resample('D').agg({
    'temperature': ['mean', 'max', 'min'],
    'production': ['sum', 'mean']
})
print(daily_stats.head())
```

---

# 실습 5: rolling (이동 통계)

```python
# 24시간(1일) 이동평균
df['temp_ma_24'] = df['temperature'].rolling(24).mean()

# 168시간(7일) 이동평균
df['prod_ma_168'] = df['production'].rolling(168).mean()

# 이동 표준편차
df['temp_std_24'] = df['temperature'].rolling(24).std()

# 결과 확인
print(df[['temperature', 'temp_ma_24', 'temp_std_24']].tail())
```

---

# 실습 6: shift (Lag 특성)

```python
# Lag 특성 생성 (과거 값)
df['prod_lag_1'] = df['production'].shift(1)   # 1시간 전
df['prod_lag_24'] = df['production'].shift(24) # 24시간(1일) 전
df['prod_lag_168'] = df['production'].shift(168) # 168시간(7일) 전

# 확인
print(df[['production', 'prod_lag_1', 'prod_lag_24']].head(30))

# NaN 확인
print(f"NaN 개수: {df['prod_lag_24'].isna().sum()}")
```

---

# 실습 7: 차분 및 변화율

```python
# 1시간 차분
df['prod_diff_1'] = df['production'].diff(1)

# 24시간 차분 (일간 변화)
df['prod_diff_24'] = df['production'].diff(24)

# 변화율
df['prod_pct'] = df['production'].pct_change()

# 결과
print(df[['production', 'prod_diff_1', 'prod_pct']].tail())
```

---

# 실습 8: 시계열 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 원본 vs 이동평균
ax1 = axes[0]
df['production'].plot(ax=ax1, alpha=0.5, label='원본')
df['prod_ma_24'].plot(ax=ax1, label='24시간 이동평균')
ax1.set_title('생산량: 원본 vs 이동평균')
ax1.legend()

# resample 일별
ax2 = axes[1]
daily['production'].plot(ax=ax2)
ax2.set_title('일별 평균 생산량')

plt.tight_layout()
plt.show()
```

---

# 실습 9: 특성 엔지니어링 종합

```python
# 예측을 위한 특성 세트
features = pd.DataFrame()

# 날짜 특성
features['hour'] = df.index.hour
features['dayofweek'] = df.index.dayofweek
features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Lag 특성 (shift 필수!)
features['lag_1'] = df['production'].shift(1)
features['lag_24'] = df['production'].shift(24)

# Rolling 특성 (shift + rolling)
features['ma_24'] = df['production'].shift(1).rolling(24).mean()
features['std_24'] = df['production'].shift(1).rolling(24).std()
```

---

# 실습 10: NaN 처리 및 시간 분할

```python
# NaN 제거
features = features.dropna()
print(f"유효 데이터: {len(features)}개")

# 시간 기준 분할 (중요!)
split_date = '2025-01-25'
train = features[:split_date]
test = features[split_date:]

print(f"학습 데이터: {len(train)}개")
print(f"테스트 데이터: {len(test)}개")

# ❌ 절대 랜덤 분할 사용 금지!
```

---

# 핵심 정리

## 1. 시계열 특성
- 시간 의존성, 자기상관, 계절성
- 시간 기준 분할 필수

## 2. datetime 처리
- pd.to_datetime(), dt 접근자
- DatetimeIndex 활용

## 3. 전처리 기법
- resample(): 주기 변환
- rolling(): 이동 통계
- shift(): Lag 특성 (미래 누출 방지!)

---

# sklearn과 시계열

```python
# 시계열 분할 (sklearn)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # 항상 과거 → 미래 순서
```

---

# 다음 차시 예고

## 18차시: 시계열 예측 모델
- 특성 엔지니어링 심화
- ML 모델로 시계열 예측
- 평가 지표 (MAE, RMSE, MAPE)
- 다단계 예측

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습 파일: `17_timeseries_basics.py`
