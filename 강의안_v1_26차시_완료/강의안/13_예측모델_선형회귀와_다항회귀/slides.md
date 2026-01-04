---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 13차시'
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

# 예측 모델: 선형회귀와 다항회귀

## 13차시 | Part III. 문제 중심 모델링 실습

**숫자를 예측하는 회귀 모델**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **회귀 문제**의 특징을 이해한다
2. **선형회귀**로 생산량 예측 모델을 구축한다
3. **다항회귀**로 비선형 관계를 모델링한다

---

# 지난 시간 복습

## 분류 모델 (11-12차시)

### 배운 것
- **의사결정나무**: 질문으로 분류
- **랜덤포레스트**: 여러 트리의 투표

### 특징
- 출력: **범주** (정상/불량, A/B/C 등급)

### 오늘
- **회귀**: 연속적인 **숫자** 예측

---

# 분류 vs 회귀

## 무엇을 예측하나요?

| 문제 유형 | 출력 | 제조 현장 예시 |
|----------|------|---------------|
| 분류 | 범주 | 불량/정상, 등급 A/B/C |
| **회귀** | **숫자** | 생산량, 불량률, 설비 수명 |

```
분류: 이 제품은 불량인가? → "예" 또는 "아니오"
회귀: 오늘 생산량은? → "1,247개"
```

---

# 선형회귀란?

## Linear Regression

> 입력(X)과 출력(y) 사이의 **직선 관계**를 찾는 방법

```
        y (생산량)
        │      ●
        │    ●  ────── 회귀선
        │  ●  ●
        │●  ●
        │●
        └────────────── x (온도)
```

$$y = wx + b$$

- **w (기울기)**: 온도가 1도 올라갈 때 생산량 변화
- **b (절편)**: x=0일 때 y 값

---

# 6차시 복습

## 이미 배웠습니다!

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# 기울기와 절편
print(f"기울기: {model.coef_}")
print(f"절편: {model.intercept_}")

# 예측
y_pred = model.predict(X_test)
```

> 6차시에서 간단히 배웠고, 오늘 더 깊이 다룹니다!

---

# 다중 선형회귀

## 여러 특성 사용

```
생산량 = 3.2×온도 + 1.5×습도 + 2.1×속도 + 100
```

```python
# 다중 선형회귀 (동일한 코드!)
X = df[['온도', '습도', '속도']]  # 여러 특성
y = df['생산량']

model = LinearRegression()
model.fit(X, y)

print(model.coef_)        # [3.2, 1.5, 2.1]
print(model.intercept_)   # 100
```

> 특성이 여러 개여도 코드는 동일!

---

# 회귀 모델 평가 지표

## 분류와 다른 평가 방법

### MSE (Mean Squared Error)
- 오차 제곱의 평균, 작을수록 좋음

### RMSE (Root MSE)
- MSE의 제곱근, 원래 단위로 해석 가능

### R² (결정계수)
- 0~1 사이, **1에 가까울수록 좋음**
- "모델이 데이터를 얼마나 잘 설명하는가"

---

# sklearn 평가 함수

## metrics 모듈

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 예측
y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")  # "평균 50개 오차"

# R² (결정계수)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.3f}")  # 0.85 → 85% 설명력
```

---

# 선형회귀의 한계

## 비선형 관계

```
        y                           y
        │   ●                       │       ●
        │  ●  ● ●                   │     ●   ●
        │ ●  ───────                │   ●       ●
        │●  ↑ 직선은 잘 맞음        │ ●           ●
        │                           │●             ●
        └────────── x               └────────────── x

       (선형 관계)                   (비선형 관계)
```

> 데이터가 곡선 형태면 **직선으로는 한계**가 있음

---

# 다항회귀란?

## Polynomial Regression

> **다항식**을 사용하여 곡선 관계를 모델링

```
2차 다항회귀:  y = w₁x + w₂x² + b
3차 다항회귀:  y = w₁x + w₂x² + w₃x³ + b
```

```
        y
        │       ●
        │     ●   ●
        │   ●  ─────●  ← 2차 곡선
        │ ●           ●
        │●             ●
        └────────────── x
```

---

# sklearn으로 다항회귀

## PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 특성 변환 (x → x, x²)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 2. 선형회귀 적용
model = LinearRegression()
model.fit(X_poly, y)

# 3. 예측
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)
```

---

# PolynomialFeatures 동작

## 특성을 늘려주는 역할

```python
# degree=2 일 때
# 원래: [x₁, x₂]
# 변환: [1, x₁, x₂, x₁², x₁x₂, x₂²]

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X = [[2, 3]]  # 원래 특성
X_poly = poly.fit_transform(X)
print(X_poly)
# [[1, 2, 3, 4, 6, 9]]
#   1  x₁ x₂ x₁² x₁x₂ x₂²
```

---

# Pipeline으로 깔끔하게

## 전처리 + 모델 결합

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 파이프라인 구성
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# 한 번에 학습
pipe.fit(X_train, y_train)

# 한 번에 예측
y_pred = pipe.predict(X_test)
```

---

# 차수(degree) 선택

## 과대적합 주의!

```
degree=1  │  degree=2   │  degree=10
직선      │  2차 곡선    │  너무 복잡!
          │             │
   ●      │     ● ●     │   ●~●~●~●
  ───●    │   ●   ●     │  ~  ~  ~
 ●   ●    │  ●     ●    │
          │             │
과소적합  │   적절!     │  과대적합
```

> 보통 **degree=2~3**이면 충분

---

# 트리 기반 회귀

## 분류뿐 아니라 회귀도 가능!

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 의사결정트리 회귀
dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train, y_train)

# 랜덤포레스트 회귀
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)
```

### Classifier → Regressor
- DecisionTreeClassifier → **DecisionTreeRegressor**
- RandomForestClassifier → **RandomForestRegressor**

---

# 이론 정리

## 회귀 모델 핵심

| 항목 | 설명 |
|------|------|
| 회귀란? | 연속적인 숫자 예측 |
| 선형회귀 | y = wx + b, 직선 관계 |
| 다항회귀 | 곡선 관계, PolynomialFeatures |
| 평가 지표 | MSE, RMSE, R² |
| 핵심 | R²가 1에 가까울수록 좋음 |

---

# - 실습편 -

## 13차시

**선형회귀와 다항회귀 실습**

---

# 실습 개요

## 생산량 예측 모델

### 목표
- 온도, 습도, 속도로 **생산량** 예측
- 선형회귀 모델 학습
- 다항회귀로 성능 향상

### 실습 환경
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
```

---

# 실습 1: 데이터 준비

## 제조 데이터 생성

```python
np.random.seed(42)
n = 300

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

# 생산량 = 온도×10 + 습도×3 + 속도×2 + 노이즈
df['생산량'] = (10*df['온도'] + 3*df['습도'] + 2*df['속도']
               + np.random.normal(0, 20, n))
```

---

# 실습 2: 선형회귀 학습

## LinearRegression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['온도', '습도', '속도']]
y = df['생산량']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"기울기: {model.coef_}")
print(f"절편: {model.intercept_:.2f}")
```

---

# 실습 3: 모델 평가

## R², MSE, RMSE

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
```

> R²가 0.9 이상이면 매우 좋은 모델!

---

# 실습 4: 다항회귀

## PolynomialFeatures + Pipeline

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 2차 다항회귀 파이프라인
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_pipe.fit(X_train, y_train)
y_pred_poly = poly_pipe.predict(X_test)

r2_poly = r2_score(y_test, y_pred_poly)
print(f"다항회귀 R²: {r2_poly:.3f}")
```

---

# 실습 5: 선형 vs 다항 비교

## 성능 비교

```python
# 선형회귀
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2 = lr.score(X_test, y_test)

# 다항회귀 (degree=2)
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_pipe.fit(X_train, y_train)
poly_r2 = poly_pipe.score(X_test, y_test)

print(f"선형회귀 R²: {lr_r2:.3f}")
print(f"다항회귀 R²: {poly_r2:.3f}")
```

---

# 실습 6: degree 실험

## 차수에 따른 성능 변화

```python
for degree in [1, 2, 3, 4, 5]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)

    print(f"degree={degree}: 학습 R²={train_r2:.3f}, 테스트 R²={test_r2:.3f}")
```

> degree가 너무 높으면 과대적합!

---

# 실습 7: 새 데이터 예측

## predict 활용

```python
new_data = [[88, 52, 105]]  # 온도, 습도, 속도

pred = model.predict(new_data)
print(f"예측 생산량: {pred[0]:.0f}개")
```

---

# 모델 선택 가이드

## 언제 무엇을 쓸까?

| 모델 | 특징 | 적합한 경우 |
|------|------|------------|
| LinearRegression | 단순, 해석 쉬움 | 선형 관계 |
| PolynomialRegression | 곡선 모델링 | 비선형 관계 |
| RandomForestRegressor | 높은 성능 | 복잡한 관계 |

> 먼저 **LinearRegression**으로 시작!

---

# 실습 정리

## 핵심 체크포인트

- [ ] LinearRegression 학습 및 평가
- [ ] coef_, intercept_ 확인
- [ ] MSE, RMSE, R² 계산
- [ ] PolynomialFeatures + Pipeline
- [ ] degree에 따른 성능 비교

---

# 다음 차시 예고

## 14차시: 모델 평가와 반복 검증

### 학습 내용
- 교차검증 (Cross Validation)
- 과대적합 / 과소적합 진단
- 분류 모델 상세 평가 (정밀도, 재현율)

> 모델을 **제대로 평가**하는 방법!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **회귀**: 숫자를 예측 (분류와 다름!)
2. **선형회귀**: y = wx + b, 직선 관계
3. **다항회귀**: 곡선 관계, degree=2~3
4. **평가 지표**: R²가 1에 가까울수록 좋음

---

# 감사합니다

## 13차시: 예측 모델 - 선형회귀와 다항회귀

**숫자 예측의 기초를 다졌습니다!**
