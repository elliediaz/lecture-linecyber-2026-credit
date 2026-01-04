---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 14차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 회귀 모델: 선형회귀와 다항회귀

## 14차시 | AI 기초체력훈련 (Pre AI-Campus)

**숫자를 예측하는 모델**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **회귀 문제**의 특징을 이해한다
2. **선형회귀**로 예측 모델을 구축한다
3. **다항회귀**로 비선형 관계를 모델링한다

---

# 분류 vs 회귀 복습

## 무엇을 예측하나요?

| 문제 유형 | 출력 | 예시 |
|----------|------|------|
| 분류 | 범주 | 불량/정상, A/B/C등급 |
| **회귀** | **숫자** | 생산량, 판매액, 불량률 |

### 지난 시간까지
- 12차시: 의사결정트리 (분류)
- 13차시: 랜덤포레스트 (분류)

### 오늘
- **회귀**: 연속적인 숫자 예측

---

# 선형회귀란?

## Linear Regression

> 입력(X)과 출력(y) 사이의 **직선 관계**를 찾는 방법

```
        y
        │      ●
        │    ●  ────── 회귀선
        │  ●  ●
        │●  ●
        │●
        └────────────── x
```

$$y = wx + b$$

- **w (기울기)**: x가 1 증가할 때 y 변화량
- **b (절편)**: x=0일 때 y 값

---

# 7차시 복습

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

> 7차시에서 간단히 배웠고, 오늘 더 깊이 다룹니다!

---

# 다중 선형회귀

## 여러 특성 사용

```
y = w₁x₁ + w₂x₂ + w₃x₃ + b

예) 생산량 = 3.2×온도 + 1.5×습도 + 2.1×속도 + 100
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

---

# 회귀 모델 평가 지표

## 분류와 다른 평가 방법

### MSE (Mean Squared Error)
$$MSE = \frac{1}{n}\sum(y_{실제} - y_{예측})^2$$

### RMSE (Root MSE)
$$RMSE = \sqrt{MSE}$$

### R² (결정계수)
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

> R²가 1에 가까울수록 좋은 모델!

---

# sklearn 평가 함수

## metrics 모듈

```python
from sklearn.metrics import mean_squared_error, r2_score

# 예측
y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# R² (결정계수)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.3f}")  # 0~1 사이, 1에 가까울수록 좋음
```

---

# 선형회귀의 한계

## 비선형 관계

```
        y                           y
        │   ●                       │       ●
        │  ●  ● ●                   │     ●   ●
        │ ●  ───────                │   ●       ●
        │●  ↑ 직선은 안 맞음        │ ●           ●
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

# 선형 vs 다항 비교

## 실습 예시

```python
# 선형회귀
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"선형회귀 R²: {lr.score(X_test, y_test):.3f}")

# 다항회귀 (degree=2)
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_pipe.fit(X_train, y_train)
print(f"다항회귀 R²: {poly_pipe.score(X_test, y_test):.3f}")
```

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

# 회귀 모델 비교

## 언제 무엇을?

| 모델 | 특징 | 적합한 경우 |
|------|------|------------|
| LinearRegression | 단순, 해석 쉬움 | 선형 관계 |
| PolynomialRegression | 곡선 모델링 | 비선형 관계 |
| DecisionTreeRegressor | 해석 가능 | 복잡한 비선형 |
| RandomForestRegressor | 높은 성능 | 대부분의 경우 |

> 먼저 **LinearRegression**으로 시작하세요!

---

# 실습 정리

## 전체 흐름

```python
# 1. 데이터 준비
X = df[['온도', '습도', '속도']]
y = df['생산량']

# 2. 분리
X_train, X_test, y_train, y_test = train_test_split(...)

# 3. 모델 학습
model = LinearRegression()  # 또는 다른 회귀 모델
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
```

---

# 다음 차시 예고

## 15차시: 모델 평가와 교차검증

- 교차검증 (Cross Validation)
- 과대적합 / 과소적합 진단
- 혼동행렬, 정밀도, 재현율

> 모델을 **제대로 평가**하는 방법!

---

# 감사합니다

## AI 기초체력훈련 14차시

**회귀 모델: 선형회귀와 다항회귀**

숫자 예측의 기초를 다졌습니다!
