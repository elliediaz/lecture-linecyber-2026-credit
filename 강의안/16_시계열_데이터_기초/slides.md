---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 16차시'
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

# 시계열 데이터 기초

## 16차시 | Part III. 문제 중심 모델링 실습

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

### 제조 현장 예시
- 일별 생산량
- 시간별 설비 온도
- 분 단위 센서 데이터
- 월별 불량률 추이

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
오늘 생산량은 어제 생산량과 관련 있음
→ 자기상관(Autocorrelation)
```

### 3. 계절성
```
12월 매출 ↑ (연말 특수)
월요일 생산량 ↓ (설비 예열)
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

# 문자열 → datetime
date = datetime.strptime('2024-01-15', '%Y-%m-%d')

# datetime → 문자열
date_str = now.strftime('%Y년 %m월 %d일')
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
df['요일명'] = df['날짜'].dt.day_name()
```

---

# 이론 정리

## 시계열 데이터 핵심

| 개념 | 설명 |
|------|------|
| 시계열 | 시간 순서로 관측된 데이터 |
| datetime | Python 날짜/시간 객체 |
| pd.to_datetime | 문자열 → 날짜 변환 |
| dt 접근자 | 연도, 월, 일, 요일 추출 |
| 순서 중요 | 시간 순서 유지 필수 |

---

# - 실습편 -

## 16차시

**시계열 데이터 처리 실습**

---

# 실습 개요

## 시계열 데이터 다루기

### 목표
- 날짜/시간 처리
- 날짜 정보 추출
- 리샘플링과 이동평균
- 시계열 시각화

### 실습 환경
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
```

---

# 실습 1: 데이터 생성

## 일별 생산 데이터

```python
np.random.seed(42)
n_days = 180

dates = pd.date_range('2024-01-01', periods=n_days)
production = 1000 + np.cumsum(np.random.randn(n_days) * 10)

df = pd.DataFrame({
    '날짜': dates,
    '생산량': production.astype(int),
    '온도': np.random.normal(85, 5, n_days)
})
```

---

# 실습 2: 날짜 변환

## pd.to_datetime

```python
# 문자열을 datetime으로
df['날짜'] = pd.to_datetime(df['날짜'])

# 데이터 타입 확인
print(df['날짜'].dtype)  # datetime64[ns]

# 날짜 인덱스로 설정
df = df.set_index('날짜')
```

---

# 실습 3: 날짜 정보 추출

## dt 접근자

```python
df['연도'] = df.index.year
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek  # 0=월요일
df['요일명'] = df.index.day_name()

print(df.head())
```

---

# 실습 4: 날짜 필터링

## 기간 선택

```python
# 특정 월 선택
jan_data = df['2024-01']
print(f"1월 데이터: {len(jan_data)}개")

# 기간 선택
first_quarter = df['2024-01-01':'2024-03-31']
print(f"1분기 데이터: {len(first_quarter)}개")
```

---

# 실습 5: 리샘플링

## 집계 주기 변경

```python
# 일별 → 주별 평균
weekly = df['생산량'].resample('W').mean()

# 일별 → 월별 합계
monthly = df['생산량'].resample('M').sum()

print("주별 생산량 평균:")
print(weekly.head())
```

---

# 실습 6: 이동평균

## rolling

```python
# 7일 이동평균
df['이동평균_7일'] = df['생산량'].rolling(window=7).mean()

# 30일 이동평균
df['이동평균_30일'] = df['생산량'].rolling(window=30).mean()

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['생산량'], alpha=0.5, label='일별')
plt.plot(df.index, df['이동평균_7일'], label='7일 이동평균')
plt.legend()
plt.show()
```

---

# 실습 7: Shift 연산

## 시차 변수 생성

```python
# 1일 전 값
df['전일_생산량'] = df['생산량'].shift(1)

# 일별 변화량
df['변화량'] = df['생산량'] - df['생산량'].shift(1)

# 일별 변화율
df['변화율'] = df['생산량'].pct_change()
```

---

# 실습 8: 시계열 시각화

## 선 그래프

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 생산량 추이
axes[0].plot(df.index, df['생산량'])
axes[0].set_title('일별 생산량')
axes[0].set_ylabel('생산량')

# 요일별 평균
weekday_avg = df.groupby('요일')['생산량'].mean()
axes[1].bar(range(7), weekday_avg)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(['월','화','수','목','금','토','일'])
axes[1].set_title('요일별 평균 생산량')

plt.tight_layout()
plt.show()
```

---

# 시계열 분할 주의사항

## 시간 기준 분할

```python
# ❌ 잘못된 방법 (무작위 분할)
# train_test_split(X, y, random_state=42)

# ✅ 올바른 방법 (시간 기준)
split_date = '2024-05-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]
```

```
|←───── Train ─────→|←── Test ──→|
|  과거 데이터       |  미래 데이터  |
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] pd.to_datetime으로 날짜 변환
- [ ] dt 접근자로 연도, 월, 요일 추출
- [ ] resample로 주기 변경
- [ ] rolling으로 이동평균
- [ ] shift로 시차 변수 생성

---

# 다음 차시 예고

## 17차시: 시계열 예측 모델

### 학습 내용
- 시계열 예측 기법
- 특성 엔지니어링
- 시계열 예측 실습

> 과거 데이터로 **미래를 예측**합니다!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **시계열**: 시간 순서로 관측된 데이터
2. **pd.to_datetime**: 문자열 → 날짜 변환
3. **dt 접근자**: 연도, 월, 요일 추출
4. **resample**: 집계 주기 변경
5. **rolling**: 이동평균 계산

---

# 감사합니다

## 16차시: 시계열 데이터 기초

**시간에 따른 데이터를 다루는 법을 배웠습니다!**
