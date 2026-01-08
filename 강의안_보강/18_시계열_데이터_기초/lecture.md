# 18차시: 시계열 데이터 기초

## 학습 목표

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | 시계열 데이터의 특성을 이해함 |
| 2 | Python datetime 처리를 수행함 |
| 3 | 시계열 전처리 기법을 활용함 |

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 시계열 데이터 특성 | 시간 의존성, 계절성, 추세 |
| 2 | Python datetime 처리 | to_datetime, dt 접근자 |
| 3 | 시계열 전처리 | resample, rolling, shift |

---

## Part 1: 시계열 데이터의 특성

### 1.1 시계열 데이터란?

**정의**
- 시간 순서로 기록된 데이터임
- 각 데이터 포인트에 타임스탬프가 존재함

**예시**
| 분야 | 예시 |
|------|------|
| 금융 | 주식 가격 (매일) |
| 제조 | 센서 측정값 (매초) |
| 비즈니스 | 월별 매출 (매월) |
| 설비 | 가동 로그 (이벤트 발생 시) |

---

### 1.2 일반 데이터 vs 시계열 데이터

| 일반 데이터 | 시계열 데이터 |
|------------|--------------|
| 행(row)이 독립 | 행 간에 순서 존재 |
| 랜덤 셔플 가능 | 셔플하면 의미 손실 |
| 랜덤 분할 가능 | 시간 기준 분할 필수 |
| 교차검증 무작위 | 순차 분할 검증 |

---

### 1.3 시계열 데이터의 특성

| 특성 | 설명 | 예시 |
|------|------|------|
| 시간 의존성 | 현재 값이 과거 값에 영향받음 | 오늘 생산량 <- 어제 생산량 |
| 자기상관 | 자기 자신과의 상관관계 | Lag 1 자기상관: 오늘 <-> 어제 |
| 계절성 | 주기적 패턴 | 월요일마다 높음, 12월마다 급증 |

---

### 1.4 시계열 구성요소

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

---

### 1.5 추세 (Trend)

장기적인 증가/감소 경향임.

**예시**
- 설비 노후화로 불량률 점진적 증가
- 공정 개선으로 생산량 점진적 증가

```
시간 ->
     /
    /
   /
  /
 /
```

---

### 1.6 계절성 (Seasonality)

일정 주기로 반복되는 패턴임.

**예시**
| 주기 | 패턴 |
|------|------|
| 일별 | 오전 생산량 높음 |
| 주별 | 월요일 불량률 높음 |
| 연별 | 여름 에너지 사용량 증가 |

```
시간 ->
^   ^   ^   ^   ^
 \ / \ / \ / \ / \
```

---

### 1.7 제조 현장의 시계열 데이터

| 데이터 | 주기 | 활용 |
|-------|------|------|
| 센서 온도 | 초/분 | 이상 탐지 |
| 생산량 | 시간/일 | 수요 예측 |
| 불량률 | 일/주 | 품질 관리 |
| 설비 가동 | 이벤트 | 예지 정비 |
| 에너지 사용 | 분/시간 | 비용 최적화 |

---

### 1.8 시계열 예측의 도전

**일반 ML과 다른 점**

| 항목 | 설명 |
|------|------|
| 데이터 누출 | 미래 정보가 학습에 포함되면 안 됨, 시간 기준 분할 필수 |
| 분할 방식 | 랜덤 분할 금지, 시간 기준 순차 분할 필수 |
| 특성 엔지니어링 | Lag 특성, Rolling 특성 활용 |

---

### 1.9 시간 기준 분할

```
전체 데이터:
[-------------|--------|--------]
   학습 (70%)   검증(15%)  테스트(15%)

시간 ->

X 잘못된 방법: 랜덤 셔플 후 분할
   -> 미래 데이터로 과거 예측 (누출!)

O 올바른 방법: 시간 순서대로 분할
   -> 과거 데이터로 미래 예측
```

---

## Part 2: Python datetime 처리

### 2.1 Python datetime 모듈

```python
from datetime import datetime, timedelta

# 현재 시각
now = datetime.now()
print(now)  # 2025-01-05 14:30:00

# 특정 날짜 생성
dt = datetime(2025, 1, 5, 14, 30)

# 문자열 -> datetime
dt = datetime.strptime("2025-01-05", "%Y-%m-%d")

# datetime -> 문자열
s = dt.strftime("%Y년 %m월 %d일")
```

---

### 2.2 datetime 형식 코드

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

### 2.3 timedelta로 날짜 계산

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

### 2.4 Pandas 날짜 처리: pd.to_datetime()

```python
import pandas as pd

# 문자열 -> datetime
df['date'] = pd.to_datetime(df['date'])

# 다양한 형식 자동 인식
pd.to_datetime("2025-01-05")
pd.to_datetime("01/05/2025")
pd.to_datetime("Jan 5, 2025")

# 형식 지정
pd.to_datetime("05-01-2025", format="%d-%m-%Y")
```

---

### 2.5 DatetimeIndex

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

### 2.6 dt 접근자

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

### 2.7 실습: datetime 기초

```python
from datetime import datetime, timedelta

# 현재 시각
now = datetime.now()
print(f"현재 시각: {now}")

# 특정 날짜 생성
dt = datetime(2025, 1, 5, 14, 30, 0)
print(f"특정 날짜: {dt}")

# 문자열 -> datetime (strptime)
date_string = "2025-01-05"
parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
print(f"파싱된 날짜: {parsed_date}")

# datetime -> 문자열 (strftime)
formatted = dt.strftime("%Y년 %m월 %d일 %H시 %M분")
print(f"포맷된 문자열: {formatted}")

# timedelta로 날짜 계산
future = now + timedelta(days=7)
past = now - timedelta(hours=3)
print(f"\n7일 후: {future}")
print(f"3시간 전: {past}")

# 날짜 차이 계산
dt1 = datetime(2025, 1, 1)
dt2 = datetime(2025, 1, 31)
diff = dt2 - dt1
print(f"\n날짜 차이: {diff.days}일")
```

---

### 2.8 날짜를 인덱스로 설정

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

### 2.9 실습: AirPassengers 데이터 로드

```python
import numpy as np
import pandas as pd

# AirPassengers 데이터셋 로드
try:
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    print("[데이터 로드 성공]")
except Exception as e:
    print(f"[데이터 로드 실패: {e}]")
    # 오프라인 대비 샘플 데이터
    np.random.seed(42)
    dates = pd.date_range(start='1949-01', periods=144, freq='MS')
    passengers = 100 + np.cumsum(np.random.normal(2, 10, 144))
    passengers = np.maximum(passengers, 50)
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

### 2.10 실습: dt 접근자로 정보 추출

```python
# 날짜 특성 추출 (인덱스에서)
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofweek'] = df.index.dayofweek  # 0=월요일
df['month_name'] = df.index.month_name()

print("[추출된 날짜 특성]")
print(df[['Passengers', 'year', 'month', 'quarter', 'month_name']].head(12))

# 월별 평균 (계절성 확인)
print("\n[월별 평균 승객 수 - 계절성 확인]")
monthly_avg = df.groupby('month')['Passengers'].mean()
for month, passengers in monthly_avg.items():
    months_kr = ['1월', '2월', '3월', '4월', '5월', '6월',
                 '7월', '8월', '9월', '10월', '11월', '12월']
    print(f"  {months_kr[month-1]}: {passengers:.1f}명")
```

---

## Part 3: 시계열 전처리 기법

### 3.1 resample() - 주기 변환

시간 주기를 변환함 (다운샘플링/업샘플링).

```python
# 분별 -> 시간별 (다운샘플링)
hourly = df.resample('H').mean()

# 일별 -> 주별
weekly = df.resample('W').sum()

# 월별 -> 분기별
quarterly = df.resample('Q').mean()
```

---

### 3.2 resample() 집계 방법

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

### 3.3 실습: resample

```python
# 월별 -> 분기별 평균
quarterly_mean = df[['Passengers']].resample('Q').mean()
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

### 3.4 rolling() - 이동 윈도우

윈도우를 이동하며 통계를 계산함.

```python
# 7일 이동 평균
df['ma_7'] = df['value'].rolling(window=7).mean()

# 3시간 이동 합계
df['sum_3h'] = df['value'].rolling(window=3).sum()

# 이동 표준편차
df['std_7'] = df['value'].rolling(window=7).std()
```

---

### 3.5 rolling() 시각화

```
원본 데이터:  [10, 20, 15, 30, 25, 35, 20]

3일 이동평균 계산:
  Day 1-3: (10+20+15)/3 = 15.0
  Day 2-4: (20+15+30)/3 = 21.7
  Day 3-5: (15+30+25)/3 = 23.3
  ...

이동평균:    [NaN, NaN, 15.0, 21.7, 23.3, 30.0, 26.7]

-> 노이즈 제거, 추세 파악에 유용
```

---

### 3.6 rolling() 옵션

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

### 3.7 실습: rolling

```python
# 3개월 이동평균
df['ma_3'] = df['Passengers'].rolling(window=3).mean()

# 12개월(1년) 이동평균 - 계절성 제거
df['ma_12'] = df['Passengers'].rolling(window=12).mean()

# 이동 표준편차
df['std_12'] = df['Passengers'].rolling(window=12).std()

print("[이동 통계 결과]")
print(df[['Passengers', 'ma_3', 'ma_12', 'std_12']].tail(12))

# NaN 개수 확인
print(f"\nNaN 개수 (ma_3): {df['ma_3'].isna().sum()}")
print(f"NaN 개수 (ma_12): {df['ma_12'].isna().sum()}")
```

---

### 3.8 shift() - 시차 변수

데이터를 시간축으로 이동함. Lag 특성 생성에 필수임.

```python
# 1칸 뒤로 (과거 값)
df['lag_1'] = df['value'].shift(1)

# 3칸 뒤로
df['lag_3'] = df['value'].shift(3)

# 1칸 앞으로 (미래 값) - 주의!
df['future'] = df['value'].shift(-1)
```

---

### 3.9 shift() 시각화

```
원본:     [A, B, C, D, E]
shift(1): [NaN, A, B, C, D]  <- 과거 값
shift(2): [NaN, NaN, A, B, C]
shift(-1): [B, C, D, E, NaN] <- 미래 값 (학습 시 주의!)

활용:
df['yesterday'] = df['production'].shift(1)
-> "오늘 생산량 예측에 어제 생산량 사용"
```

---

### 3.10 실습: shift (Lag 특성)

```python
# Lag 특성 생성
df['lag_1'] = df['Passengers'].shift(1)      # 1개월 전
df['lag_3'] = df['Passengers'].shift(3)      # 3개월 전
df['lag_12'] = df['Passengers'].shift(12)    # 12개월 전 (작년 같은 달)

print("[Lag 특성 결과]")
print(df[['Passengers', 'lag_1', 'lag_3', 'lag_12']].head(15))

# NaN 개수
print(f"\nNaN 개수 (lag_1): {df['lag_1'].isna().sum()}")
print(f"NaN 개수 (lag_3): {df['lag_3'].isna().sum()}")
print(f"NaN 개수 (lag_12): {df['lag_12'].isna().sum()}")
```

---

### 3.11 데이터 누출 방지

**매우 중요한 개념임!**

```python
# X 잘못된 방법 (미래 정보 포함)
df['ma_7'] = df['production'].rolling(7).mean()
# -> 오늘 값이 이동평균에 포함됨!

# O 올바른 방법 (shift 먼저)
df['ma_7'] = df['production'].shift(1).rolling(7).mean()
# -> 어제까지의 데이터만 사용
```

---

### 3.12 실습: 데이터 누출 방지

```python
# 잘못된 방법: 현재 값 포함
wrong_ma = df['Passengers'].rolling(3).mean()

# 올바른 방법: shift 먼저
correct_ma = df['Passengers'].shift(1).rolling(3).mean()

df['wrong_ma_3'] = wrong_ma
df['correct_ma_3'] = correct_ma

print("[잘못된 vs 올바른 Rolling]")
print(df[['Passengers', 'wrong_ma_3', 'correct_ma_3']].iloc[3:9])

print("\n주의:")
print("  - wrong_ma_3: 현재 승객수가 평균에 포함됨 (미래 누출!)")
print("  - correct_ma_3: 이전 달까지의 데이터만 사용 (안전)")
```

---

### 3.13 diff() - 차분

연속된 값의 차이를 계산함. 추세 제거, 정상성 확보에 사용함.

```python
# 1차 차분
df['diff_1'] = df['value'].diff(1)
# [NaN, v2-v1, v3-v2, v4-v3, ...]

# 7일 차분 (주간 변화)
df['diff_7'] = df['value'].diff(7)
```

---

### 3.14 pct_change() - 변화율

```python
# 변화율 (퍼센트)
df['pct_change'] = df['value'].pct_change()
# (현재 - 이전) / 이전

# 예: [100, 110, 99]
# pct_change: [NaN, 0.10, -0.10]
# -> 10% 증가, 10% 감소
```

---

### 3.15 실습: 차분 및 변화율

```python
# 1개월 차분
df['diff_1'] = df['Passengers'].diff(1)

# 12개월 차분 (전년 동월 대비 변화)
df['diff_12'] = df['Passengers'].diff(12)

# 변화율
df['pct_change'] = df['Passengers'].pct_change()

# 전년 동월 대비 변화율
df['yoy_change'] = df['Passengers'].pct_change(12)

print("[차분 및 변화율]")
print(df[['Passengers', 'diff_1', 'pct_change', 'yoy_change']].tail(12))

# 변화율 통계
print(f"\n월간 변화율 통계:")
print(f"  평균: {df['pct_change'].mean():.4f} ({df['pct_change'].mean()*100:.2f}%)")
print(f"  표준편차: {df['pct_change'].std():.4f}")
```

---

### 3.16 특성 엔지니어링 종합

```python
# 예측을 위한 특성 세트 생성
features = pd.DataFrame(index=df.index)

# 날짜 특성
features['month'] = df.index.month
features['quarter'] = df.index.quarter
features['year'] = df.index.year

# 주기적 인코딩 (월의 순환 특성 반영)
features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

# Lag 특성 (shift 필수!)
features['lag_1'] = df['Passengers'].shift(1)
features['lag_2'] = df['Passengers'].shift(2)
features['lag_3'] = df['Passengers'].shift(3)
features['lag_12'] = df['Passengers'].shift(12)

# Rolling 특성 (shift 먼저!)
features['ma_3'] = df['Passengers'].shift(1).rolling(3).mean()
features['ma_6'] = df['Passengers'].shift(1).rolling(6).mean()
features['ma_12'] = df['Passengers'].shift(1).rolling(12).mean()
features['std_3'] = df['Passengers'].shift(1).rolling(3).std()

# 타겟
features['target'] = df['Passengers']

print("[특성 세트]")
print(features.head(15))
print(f"\n특성 수: {len(features.columns) - 1}")  # target 제외
```

---

### 3.17 시간 기준 분할

```python
# NaN 제거
features_clean = features.dropna()
print(f"\n원본 데이터: {len(features)}개")
print(f"NaN 제거 후: {len(features_clean)}개")

# 시간 기준 분할 (마지막 2년을 테스트로)
split_date = '1958-12-01'

train = features_clean[:split_date]
test = features_clean[split_date:]

print(f"\n[시간 기준 분할]")
print(f"분할 기준: {split_date}")
print(f"학습 데이터: {len(train)}개 ({train.index.min()} ~ {train.index.max()})")
print(f"테스트 데이터: {len(test)}개 ({test.index.min()} ~ {test.index.max()})")

# 학습/테스트 분리
X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop('target', axis=1)
y_test = test['target']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 절대 랜덤 분할 사용 금지!
```

---

### 3.18 TimeSeriesSplit (sklearn)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

print("[TimeSeriesSplit 교차검증]")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    train_start = X_train.index[train_idx[0]]
    train_end = X_train.index[train_idx[-1]]
    val_start = X_train.index[val_idx[0]]
    val_end = X_train.index[val_idx[-1]]

    print(f"Fold {fold}:")
    print(f"  학습: {train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')} ({len(train_idx)}개)")
    print(f"  검증: {val_start.strftime('%Y-%m')} ~ {val_end.strftime('%Y-%m')} ({len(val_idx)}개)")
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 시계열 특성 | 시간 의존성, 자기상관, 계절성 |
| 시간 기준 분할 | 랜덤 분할 금지, 순차 분할 필수 |
| pd.to_datetime() | 문자열을 datetime으로 변환 |
| dt 접근자 | datetime에서 년/월/일/요일 추출 |
| resample() | 주기 변환 (일별->주별 등) |
| rolling() | 이동 통계 (이동평균 등) |
| shift() | Lag 특성 생성 (미래 누출 방지!) |

---

## sklearn과 시계열

```python
# 시계열 분할 (sklearn)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # 항상 과거 -> 미래 순서
```

---

## 데이터 누출 방지 요약

```python
# 잘못됨 (미래 정보 포함)
df['ma_3'] = df['value'].rolling(3).mean()

# 올바름 (shift 먼저!)
df['ma_3'] = df['value'].shift(1).rolling(3).mean()
```

이 규칙을 항상 준수해야 함!
