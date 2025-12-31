---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 17차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 시계열 데이터 기초

## 17차시 | AI 기초체력훈련 (Pre AI-Campus)

**시간에 따라 변하는 데이터 다루기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **시계열 데이터**의 특성을 이해한다
2. **datetime**으로 날짜/시간을 처리한다
3. 시계열 데이터를 **시각화**한다

---

# 시계열 데이터란?

## Time Series Data

> 시간 순서에 따라 관측된 데이터

### 예시
- 일별 생산량
- 월별 매출액
- 시간대별 온도
- 초 단위 센서 데이터

```
날짜        생산량
2024-01-01  1,234
2024-01-02  1,287
2024-01-03  1,156
...
```

---

# 시계열 데이터의 특징

## 일반 데이터와 다른 점

### 1. 순서가 중요
```
일반 데이터: 순서 바꿔도 OK
시계열: 순서 바꾸면 의미 없음!
```

### 2. 시간 의존성
```
오늘 매출은 어제 매출과 관련 있음
→ 자기상관(Autocorrelation)
```

### 3. 계절성
```
12월 매출 ↑ (연말 특수)
8월 아이스크림 ↑
```

---

# Python의 날짜/시간

## datetime 모듈

```python
from datetime import datetime, timedelta

# 현재 시간
now = datetime.now()
print(now)  # 2024-01-15 14:30:25.123456

# 특정 날짜 생성
date = datetime(2024, 1, 15)
print(date)  # 2024-01-15 00:00:00

# 문자열 → datetime
date = datetime.strptime('2024-01-15', '%Y-%m-%d')

# datetime → 문자열
date_str = now.strftime('%Y년 %m월 %d일')
print(date_str)  # 2024년 01월 15일
```

---

# 날짜 연산

## timedelta

```python
from datetime import datetime, timedelta

today = datetime.now()

# 7일 후
next_week = today + timedelta(days=7)

# 30일 전
last_month = today - timedelta(days=30)

# 두 날짜 사이 간격
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 12, 31)
diff = date2 - date1
print(diff.days)  # 365
```

---

# Pandas의 날짜 처리

## pd.to_datetime

```python
import pandas as pd

# 문자열을 datetime으로 변환
df['날짜'] = pd.to_datetime(df['날짜'])

# 다양한 형식 지원
pd.to_datetime('2024-01-15')
pd.to_datetime('01/15/2024')
pd.to_datetime('15-Jan-2024')

# 형식 지정
pd.to_datetime('15/01/2024', format='%d/%m/%Y')
```

---

# 날짜 인덱스

## DatetimeIndex

```python
# 날짜를 인덱스로 설정
df = df.set_index('날짜')

# 날짜로 필터링
df['2024-01']              # 2024년 1월 전체
df['2024-01-01':'2024-06-30']  # 기간 지정
df.loc['2024-01-15']       # 특정 날짜

# 날짜 범위 생성
dates = pd.date_range(
    start='2024-01-01',
    end='2024-12-31',
    freq='D'  # Daily
)
```

---

# 날짜 정보 추출

## dt 접근자

```python
# 날짜에서 정보 추출
df['연도'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0=월요일
df['주차'] = df['날짜'].dt.isocalendar().week

# 요일 이름
df['요일명'] = df['날짜'].dt.day_name()  # Monday, Tuesday...
```

---

# 시계열 시각화

## 기본 선 그래프

```python
import matplotlib.pyplot as plt

# 시계열 플롯
plt.figure(figsize=(12, 5))
plt.plot(df['날짜'], df['생산량'])
plt.xlabel('날짜')
plt.ylabel('생산량')
plt.title('일별 생산량 추이')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

# 시계열 패턴

## 추세, 계절성, 잔차

```
       추세                 계절성               잔차
        ↗                  ╱╲╱╲╱╲            ∿∿∿∿
      ↗                  ╱╲╱╲╱╲            ∿∿∿
    ↗                                      불규칙
  장기적 증가/감소       반복되는 패턴        노이즈
```

### 시계열 = 추세 + 계절성 + 잔차

---

# 리샘플링

## 집계 주기 변경

```python
# 날짜 인덱스 설정
df = df.set_index('날짜')

# 일별 → 월별 집계
monthly = df.resample('M').mean()  # 월 평균
monthly = df.resample('M').sum()   # 월 합계

# 일별 → 주별 집계
weekly = df.resample('W').mean()

# 주요 주기
# 'D': 일, 'W': 주, 'M': 월, 'Q': 분기, 'Y': 연
```

---

# 이동평균

## Rolling Mean

```python
# 7일 이동평균
df['이동평균_7일'] = df['생산량'].rolling(window=7).mean()

# 30일 이동평균
df['이동평균_30일'] = df['생산량'].rolling(window=30).mean()

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(df['날짜'], df['생산량'], alpha=0.5, label='일별')
plt.plot(df['날짜'], df['이동평균_7일'], label='7일 이동평균')
plt.legend()
plt.show()
```

> 이동평균으로 **노이즈 제거**, **추세 파악**

---

# Shift 연산

## 시차 변수 생성

```python
# 1일 전 값
df['전일_생산량'] = df['생산량'].shift(1)

# 7일 전 값
df['일주일전_생산량'] = df['생산량'].shift(7)

# 변화량
df['일별_변화'] = df['생산량'] - df['생산량'].shift(1)

# 변화율
df['일별_변화율'] = df['생산량'].pct_change()
```

---

# 시계열 데이터 주의사항

## 자주 하는 실수

### 1. 미래 데이터 누출
```python
# 잘못된 예: 미래 정보 사용
df['다음날_생산량'] = df['생산량'].shift(-1)  # 테스트에서 사용하면 안됨!
```

### 2. 무작위 분할
```python
# 잘못된 예
train_test_split(X, y, random_state=42)  # 시간 순서 무시

# 올바른 예: 시간 기준 분할
train = df[df['날짜'] < '2024-07-01']
test = df[df['날짜'] >= '2024-07-01']
```

---

# 시계열 train/test 분할

## 시간 기준

```python
# 방법 1: 날짜로 분할
split_date = '2024-07-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

# 방법 2: 비율로 분할
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
```

```
|←───── Train (80%) ─────→|←── Test (20%) ──→|
|  과거 데이터로 학습      |  미래 데이터로 평가  |
```

---

# 제조 현장 시계열 예시

## 일별 생산 데이터

```python
# 데이터 예시
df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=180),
    '생산량': np.random.normal(1000, 50, 180),
    '불량수': np.random.poisson(10, 180)
})

# 요일별 평균 생산량
weekday_avg = df.groupby(df['날짜'].dt.dayofweek)['생산량'].mean()
print(weekday_avg)
# 월요일 낮고, 수요일~목요일 높은 패턴 발견!
```

---

# 정리

## 핵심 개념

| 개념 | 함수/메서드 |
|------|------------|
| 날짜 변환 | pd.to_datetime() |
| 날짜 정보 추출 | .dt.year, .dt.month |
| 리샘플링 | .resample('M') |
| 이동평균 | .rolling(7).mean() |
| 시차 변수 | .shift(1) |

---

# 다음 차시 예고

## 18차시: 시계열 예측 모델

- 시계열 예측 기법
- 특성 엔지니어링
- 시계열 예측 실습

> 과거 데이터로 **미래를 예측**합니다!

---

# 감사합니다

## AI 기초체력훈련 17차시

**시계열 데이터 기초**

시간에 따른 데이터를 다루는 법을 배웠습니다!
