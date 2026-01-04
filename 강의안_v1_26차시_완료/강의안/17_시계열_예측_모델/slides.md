---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 17차시'
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

# 시계열 예측 모델

## 17차시 | Part III. 문제 중심 모델링 실습

**과거 데이터로 미래를 예측하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **특성 엔지니어링**으로 시계열을 테이블로 변환한다
2. **ML 모델**로 시계열을 예측한다
3. 예측 **성능을 평가**한다 (MAE, RMSE, MAPE)

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

### 제조 현장 활용
- 생산량 예측 → 원자재 발주
- 수요 예측 → 재고 관리
- 설비 상태 예측 → 예방 정비

---

# ML 기반 시계열 예측

## 핵심 아이디어

> 시계열 데이터 → **테이블 형태로 변환** → ML 모델 적용

### 변환 과정
```
원래 시계열            테이블 형태
날짜        생산량      월  요일  전일값  7일평균  → 생산량
2024-01-01  1,000       1    0    NaN    NaN        1,000
2024-01-02  1,050  →    1    1    1,000  NaN        1,050
...                    ...
2024-01-08  1,030       1    0    1,020  1,014      1,030
```

> 이미 배운 RandomForest를 활용!

---

# 특성 엔지니어링

## Feature Engineering

### 1. 날짜 기반 특성
```python
df['월'] = df['날짜'].dt.month
df['요일'] = df['날짜'].dt.dayofweek
df['주차'] = df['날짜'].dt.isocalendar().week
```

### 2. 시차 특성 (Lag)
```python
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
```

### 3. 롤링 특성 (이동통계)
```python
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
```

---

# 시차 특성 (Lag Features)

## 과거 값 활용

```python
# 시차 특성 생성
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
df['lag_30'] = df['생산량'].shift(30) # 30일 전
```

### 결과
```
날짜        생산량  lag_1  lag_7
2024-01-07  1,020  1,010    NaN
2024-01-08  1,030  1,020  1,000
2024-01-09  1,050  1,030  1,050
```

> **어제 생산량**이 오늘 생산량과 가장 관련 있음!

---

# 롤링 특성 (Rolling Features)

## 이동 통계량

```python
# ⚠️ shift(1)을 먼저! (데이터 누출 방지)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
df['ma_30'] = df['생산량'].shift(1).rolling(30).mean()

# 이동 표준편차 (변동성)
df['std_7'] = df['생산량'].shift(1).rolling(7).std()

# 이동 최대/최소
df['max_7'] = df['생산량'].shift(1).rolling(7).max()
df['min_7'] = df['생산량'].shift(1).rolling(7).min()
```

> shift(1) 필수! 없으면 미래 정보 포함됨

---

# 데이터 누출 방지

## Data Leakage

```python
# ❌ 잘못된 예 (오늘 값 포함!)
df['ma_7'] = df['생산량'].rolling(7).mean()

# ✅ 올바른 예 (어제까지만!)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
```

### 규칙
- 예측 시점에 알 수 있는 정보만 사용
- shift(1)로 하루 늦추기
- 테스트할 때도 미래 정보 사용 금지

> 실수하면 테스트 점수는 좋지만 **실제 예측은 실패**

---

# 모델 학습

## RandomForest로 시계열 예측

```python
from sklearn.ensemble import RandomForestRegressor

# 특성 정의
features = ['월', '요일', 'lag_1', 'lag_7', 'ma_7']
X = df[features]
y = df['생산량']

# 시간 기준 분할
train = df[df.index < '2024-05-01']
test = df[df.index >= '2024-05-01']

# 학습 및 예측
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[features], train['생산량'])
predictions = model.predict(test[features])
```

---

# 평가 지표

## 시계열 예측 평가

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MAE: 평균 절대 오차
mae = mean_absolute_error(y_test, predictions)

# RMSE: 평균 제곱근 오차
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# MAPE: 평균 절대 백분율 오차
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
```

| 지표 | 의미 |
|------|------|
| MAE | 평균 오차 (단위: 개) |
| RMSE | 큰 오차에 민감 |
| MAPE | 백분율 오차 (%) |

---

# MAPE 해석

## Mean Absolute Percentage Error

| MAPE | 해석 |
|------|------|
| < 10% | 매우 좋음 |
| 10~20% | 좋음 |
| 20~50% | 보통 |
| > 50% | 개선 필요 |

### 예시
```
실제: 1000개, 예측: 950개
오차: |1000 - 950| / 1000 = 5%
```

> 제조 현장에서는 보통 **10~20% 이내**를 목표!

---

# 이론 정리

## 시계열 예측 핵심

| 개념 | 설명 |
|------|------|
| 특성 엔지니어링 | 시계열 → 테이블 변환 |
| Lag 특성 | shift()로 과거 값 |
| Rolling 특성 | 이동평균, 이동표준편차 |
| 데이터 누출 | shift(1) 필수! |
| MAE/RMSE/MAPE | 예측 평가 지표 |

---

# - 실습편 -

## 17차시

**시계열 예측 모델 실습**

---

# 실습 개요

## 생산량 예측 모델 구축

### 목표
- 특성 엔지니어링
- RandomForest로 예측
- 평가 및 시각화

### 실습 환경
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

# 실습 1: 데이터 준비

## 일별 생산 데이터

```python
np.random.seed(42)
n_days = 180

dates = pd.date_range('2024-01-01', periods=n_days)
trend = np.linspace(1000, 1100, n_days)
seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)
noise = np.random.randn(n_days) * 20
production = trend + seasonal + noise

df = pd.DataFrame({
    '날짜': dates,
    '생산량': production.astype(int)
})
df = df.set_index('날짜')
```

---

# 실습 2: 날짜 특성

## dt 접근자 활용

```python
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek
df['주차'] = df.index.isocalendar().week

# 특수 날짜
df['월초'] = (df.index.day <= 5).astype(int)
df['월말'] = (df.index.day >= 25).astype(int)
df['주말'] = (df.index.dayofweek >= 5).astype(int)
```

---

# 실습 3: 시차 특성

## Lag Features

```python
# 시차 특성 (과거 값)
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
df['lag_14'] = df['생산량'].shift(14) # 14일 전

print("시차 특성 생성:")
print(df[['생산량', 'lag_1', 'lag_7']].tail())
```

---

# 실습 4: 롤링 특성

## Rolling Features

```python
# 이동평균 (shift(1) 필수!)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
df['ma_14'] = df['생산량'].shift(1).rolling(14).mean()

# 이동 표준편차
df['std_7'] = df['생산량'].shift(1).rolling(7).std()

# 이동 최대/최소
df['max_7'] = df['생산량'].shift(1).rolling(7).max()
df['min_7'] = df['생산량'].shift(1).rolling(7).min()
```

---

# 실습 5: 결측치 제거

## dropna

```python
print(f"원본 데이터: {len(df)}개")

# lag/rolling으로 생긴 NaN 제거
df = df.dropna()

print(f"결측치 제거 후: {len(df)}개")
print(f"제거된 행: {180 - len(df)}개")
```

> rolling(7)을 쓰면 처음 7개 행은 NaN

---

# 실습 6: 시간 기준 분할

## Train/Test Split

```python
# 시간 기준 분할 (5월 1일 기준)
split_date = '2024-05-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

print(f"Train: {len(train)}개 ({train.index.min().date()} ~ {train.index.max().date()})")
print(f"Test: {len(test)}개 ({test.index.min().date()} ~ {test.index.max().date()})")
```

```
|←───── Train ─────→|←── Test ──→|
|     1~4월          |    5~6월    |
```

---

# 실습 7: 모델 학습

## RandomForestRegressor

```python
# 특성 정의
features = ['월', '요일', 'lag_1', 'lag_7', 'ma_7', 'std_7']

# 학습/테스트 데이터
X_train = train[features]
y_train = train['생산량']
X_test = test[features]
y_test = test['생산량']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
```

---

# 실습 8: 성능 평가

## MAE, RMSE, MAPE

```python
# MAE
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae:.1f}개")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.1f}개")

# MAPE
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
print(f"MAPE: {mape:.1f}%")
```

---

# 실습 9: 결과 시각화

## 실제 vs 예측

```python
plt.figure(figsize=(12, 5))
plt.plot(test.index, y_test, label='실제', linewidth=2)
plt.plot(test.index, predictions, label='예측',
         linewidth=2, linestyle='--')
plt.xlabel('날짜')
plt.ylabel('생산량')
plt.title('생산량 예측 결과')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# 실습 10: 특성 중요도

## Feature Importance

```python
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("특성 중요도:")
print(importance)
```

```
feature     importance
lag_1       0.45
ma_7        0.25
lag_7       0.15
...
```

> 전일 생산량(lag_1)이 가장 중요!

---

# 실습 정리

## 핵심 체크포인트

- [ ] 날짜 특성 생성 (월, 요일)
- [ ] 시차 특성 생성 (lag_1, lag_7)
- [ ] 롤링 특성 생성 (shift(1) 필수!)
- [ ] 결측치 제거 (dropna)
- [ ] 시간 기준 분할
- [ ] MAE, RMSE, MAPE 평가

---

# 다음 차시 예고

## 18차시: 딥러닝 입문

### 학습 내용
- 신경망 기초 개념
- 뉴런, 층, 활성화 함수
- 딥러닝 vs 머신러닝

> 드디어 **딥러닝**의 세계로!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **특성 엔지니어링**: 시계열 → 테이블 변환
2. **Lag 특성**: shift()로 과거 값 활용
3. **Rolling 특성**: 이동평균 (shift(1) 필수!)
4. **시간 기준 분할**: 랜덤 분할 금지
5. **MAPE**: 10~20% 이내 목표

---

# 감사합니다

## 17차시: 시계열 예측 모델

**과거 데이터로 미래를 예측했습니다!**
