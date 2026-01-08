---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 15차시'
footer: '공공데이터를 활용한 AI 예측 모델 구축'
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
  table { font-size: 0.9em; }
  .highlight { background-color: #fef3c7; padding: 10px; border-radius: 8px; }
  .important { background-color: #fee2e2; padding: 10px; border-radius: 8px; }
  .tip { background-color: #d1fae5; padding: 10px; border-radius: 8px; }
---

# 예측 모델 - 선형/다항 회귀

## 15차시 | Part III. 문제 중심 모델링 실습

**숫자를 예측합니다!**

---

# 지난 시간 복습

## 11-13차시에서 배운 것

- **분류**: 범주 예측 (정상/불량)
- **의사결정나무**: 질문 기반 분류
- **랜덤포레스트**: 앙상블로 성능 향상

<div class="tip">

오늘부터 **회귀**! 연속적인 숫자를 예측합니다.

</div>

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **회귀 문제의 개념**을 이해한다 |
| 2 | **선형회귀 모델**을 사용한다 |
| 3 | **다항회귀로 비선형 관계**를 학습한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  회귀 문제  │ →  │   선형회귀  │ → │   다항회귀  │
│    이해     │    │  Linear Reg │    │  Polynomial │
└─────────────┘    └─────────────┘    └─────────────┘
   분류 vs 회귀     기울기, 절편        차수 선택
   평가 지표        MSE, R²            과대적합 주의
```

---

<!-- _class: lead -->

# Part 1

## 회귀 문제의 이해

---

# 분류 vs 회귀

## 출력의 차이

| 구분 | 분류 (Classification) | 회귀 (Regression) |
|------|---------------------|-------------------|
| **출력** | 범주 (카테고리) | 연속적인 숫자 |
| **예시** | 정상/불량 | 생산량: 1,247개 |
| **질문** | "~인가요?" | "얼마나?" |
| **평가** | 정확도, F1 | MSE, R² |

---

# 회귀 문제 예시

## 제조 현장

| 문제 | 입력 | 출력 (예측값) |
|------|------|--------------|
| 생산량 예측 | 설비, 인력, 원료 | 1,247개 |
| 불량률 예측 | 온도, 습도, 속도 | 3.2% |
| 설비 수명 예측 | 가동 시간, 진동 | 87일 |
| 소요 시간 예측 | 공정, 작업량 | 4.5시간 |

---

# 회귀의 목표

## 연속적인 관계 학습

```
    생산량
      ↑
   1400 │             ●
        │           ●
   1200 │         ●
        │       ●
   1000 │     ●
        │   ●
    800 │ ●
        └────────────────→ 온도
          75   80   85   90
```

데이터의 **추세선**을 찾는 것!

---

# 회귀 모델 평가 지표

## 오차를 측정

### MSE (Mean Squared Error)
```
MSE = (1/n) × Σ(실제값 - 예측값)²
```
- 오차를 제곱해서 평균
- 값이 **작을수록** 좋음

### R² (결정계수)
```
R² = 1 - (예측 오차 / 평균 오차)
```
- 모델이 데이터를 얼마나 설명하는지
- **1에 가까울수록** 좋음

---

# R² 점수 해석

## 결정계수

| R² 값 | 해석 |
|-------|------|
| 1.0 | 완벽한 예측 |
| 0.8+ | 좋음 |
| 0.5~0.8 | 보통 |
| 0.3~0.5 | 약함 |
| 0 이하 | 평균보다 못함 |

<div class="tip">

R² = 0.75 → 모델이 데이터 변동의 **75%를 설명**

</div>

---

<!-- _class: lead -->

# Part 2

## 선형회귀 (Linear Regression)

---

# 선형회귀란?

## Linear Regression

> 입력과 출력 사이의 **직선 관계**를 학습

### 수식
$$y = wx + b$$

- **w**: 기울기 (weight, coefficient)
- **b**: 절편 (bias, intercept)
- **x**: 입력 (특성)
- **y**: 출력 (예측값)

---

# 직선의 의미

## 기울기와 절편

```
    y
    ↑
    │     ╱  기울기(w) = 변화량
    │   ╱    온도 1도 증가 시
    │ ╱      생산량 10개 감소
    │╱───────────────→ x
    │
    b ← 절편 (x=0일 때 y값)
```

### 해석
- `w = -10`: 온도 1도 ↑ → 생산량 10개 ↓
- `b = 1500`: 온도 0도일 때 생산량 1500개

---

# 다중 선형회귀

## 특성이 여러 개일 때

$$y = w_1x_1 + w_2x_2 + w_3x_3 + b$$

### 예시: 생산량 예측
```
생산량 = 5×속도 - 3×온도 + 2×습도 + 1000
```

- 속도 1↑ → 생산량 5↑
- 온도 1↑ → 생산량 3↓
- 습도 1↑ → 생산량 2↑

---

# 선형회귀 학습 원리

## 최소제곱법 (OLS)

> 실제값과 예측값의 **오차 제곱합을 최소화**

```
        ●          │ 오차
      ──●─────●────│
    ●        ●     ↓
  ──────────────── 직선
    ↑  오차
```

모든 점과 직선 사이 거리의 **제곱합이 최소**인 직선!

---

# sklearn LinearRegression

## 사용법

```python
from sklearn.linear_model import LinearRegression

# 모델 생성
model = LinearRegression()

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가 (R² 점수)
r2 = model.score(X_test, y_test)
```

---

# 회귀 계수 확인

## coef_ 와 intercept_

```python
# 학습 후
print(f"절편 (b): {model.intercept_:.2f}")
print(f"계수 (w): {model.coef_}")

# 출력 예시:
# 절편 (b): 1000.50
# 계수 (w): [ 5.2, -3.1,  2.4]
#           속도  온도  습도
```

### 해석
- 속도가 1 증가하면 생산량 5.2 증가
- 온도가 1 증가하면 생산량 3.1 감소

---

# 실습: 데이터 생성

## 생산량 예측 데이터

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
})

# 생산량 (선형 관계 + 노이즈)
df['production'] = (
    1000
    + 5 * df['speed']
    - 3 * df['temperature']
    + np.random.normal(0, 50, n)
)
```

---

# 실습: 모델 학습

## 선형회귀 적용

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['temperature', 'humidity', 'speed']]
y = df['production']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"R² 점수: {model.score(X_test, y_test):.3f}")
print(f"계수: {model.coef_}")
print(f"절편: {model.intercept_:.2f}")
```

---

# 실습: 예측 및 평가

## MSE, RMSE 계산

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# RMSE (해석하기 쉬움)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
# RMSE가 50이면, 평균적으로 50개 정도 오차
```

---

# 예측 결과 시각화

## 실제 vs 예측

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('실제 생산량')
plt.ylabel('예측 생산량')
plt.title(f'선형회귀 (R²={model.score(X_test, y_test):.3f})')
plt.show()
```

대각선에 가까울수록 예측이 정확!

---

<!-- _class: lead -->

# Part 3

## 다항회귀 (Polynomial Regression)

---

# 비선형 관계

## 직선으로 안 맞을 때

```
    y
    ↑
    │    ● ●
    │   ●   ●      실제 관계: 곡선
    │  ●     ●
    │ ●       ●
    │●──────────── 직선 회귀
    └────────────→ x
```

데이터가 **곡선** 형태면 직선으로는 한계!

---

# 다항회귀란?

## Polynomial Regression

> 특성의 **거듭제곱**을 추가하여 곡선 관계 학습

### 2차 다항식
$$y = w_1x + w_2x^2 + b$$

### 3차 다항식
$$y = w_1x + w_2x^2 + w_3x^3 + b$$

---

# 다항 특성 생성

## PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures

# 원본 특성: [x]
X = [[2], [3], [4]]

# 2차 다항 특성 생성
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 결과: [1, x, x²]
# [[1, 2, 4],
#  [1, 3, 9],
#  [1, 4, 16]]
```

---

# 다항회귀 전체 과정

## 파이프라인

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 다항회귀 파이프라인
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# 학습
poly_model.fit(X_train, y_train)

# 예측
y_pred = poly_model.predict(X_test)
```

---

# degree (차수)의 영향

## 차수 선택

```
degree=1    degree=2    degree=5    degree=15
(과소적합)   (적절)     (좋음)      (과대적합)

   │─────     │ ●       │ ~~~●     │~~~●~~~~
   │  ● ●     │● ●      │●   ●     │● ● ● ●
   │ ●        │●        │●         │●
```

<div class="important">

차수가 **너무 높으면 과대적합**!

</div>

---

# 실습: 비선형 데이터

## 곡선 관계 생성

```python
np.random.seed(42)
n = 100

X = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 + 3*X.ravel() - 0.5*X.ravel()**2 + np.random.normal(0, 1, n)

# 시각화
plt.scatter(X, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.title('비선형 관계 데이터')
plt.show()
```

---

# 실습: 선형 vs 다항 비교

## 모델 비교

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 선형회귀
linear = LinearRegression()
linear.fit(X_train, y_train)
print(f"선형 R²: {linear.score(X_test, y_test):.3f}")

# 2차 다항회귀
poly2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly2.fit(X_train, y_train)
print(f"다항(2차) R²: {poly2.score(X_test, y_test):.3f}")
```

---

# 차수별 성능 비교

## degree 실험

```python
degrees = [1, 2, 3, 5, 10, 15]

for deg in degrees:
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('linear', LinearRegression())
    ])
    poly.fit(X_train, y_train)
    train_score = poly.score(X_train, y_train)
    test_score = poly.score(X_test, y_test)
    print(f"degree={deg:2}: 학습 R²={train_score:.3f}, "
          f"테스트 R²={test_score:.3f}")
```

---

# 과대적합 징후

## 학습 vs 테스트 점수

```
degree=1:  학습 R²=0.70, 테스트 R²=0.68  ← 과소적합
degree=2:  학습 R²=0.95, 테스트 R²=0.94  ← 적절
degree=5:  학습 R²=0.98, 테스트 R²=0.92  ← 조금 과대적합
degree=15: 학습 R²=0.99, 테스트 R²=0.60  ← 심한 과대적합
```

<div class="tip">

학습 점수는 높은데 테스트 점수가 낮으면 **과대적합**!

</div>

---

# 시각화로 과대적합 확인

## 차수별 곡선

```python
plt.scatter(X, y, alpha=0.5, label='데이터')

for deg in [1, 2, 5, 15]:
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('linear', LinearRegression())
    ])
    poly.fit(X, y)
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(X_plot, poly.predict(X_plot), label=f'degree={deg}')

plt.legend()
plt.show()
```

---

<!-- _class: lead -->

# 실습편

## 제조 데이터 생산량 예측

---

# 실습 목표

## 완전한 회귀 파이프라인

1. 데이터 생성 및 탐색
2. 선형회귀 적용
3. 다항회귀 비교
4. 최적 차수 선택
5. 최종 예측

---

# 실습 1: 데이터 생성

## 생산량 예측 데이터

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
})

# 비선형 관계 포함
df['production'] = (
    1000
    + 5 * df['speed']
    - 3 * df['temperature']
    - 0.05 * df['temperature']**2  # 비선형
    + np.random.normal(0, 50, n)
)
```

---

# 실습 2: 선형회귀

## 기본 모델

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

X = df[['temperature', 'humidity', 'speed']]
y = df['production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
print(f"선형회귀 R²: {linear_model.score(X_test, y_test):.3f}")
print(f"선형회귀 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
```

---

# 실습 3: 다항회귀

## 2차 다항 적용

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)
print(f"다항회귀 R²: {poly_model.score(X_test, y_test):.3f}")
print(f"다항회귀 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.2f}")
```

---

# 실습 4: 차수 비교

## 최적 차수 찾기

```python
train_scores = []
test_scores = []

for deg in range(1, 6):
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('linear', LinearRegression())
    ])
    poly.fit(X_train, y_train)
    train_scores.append(poly.score(X_train, y_train))
    test_scores.append(poly.score(X_test, y_test))

# 최적 차수
best_deg = np.argmax(test_scores) + 1
print(f"최적 차수: {best_deg}")
```

---

# 실습 5: 새 데이터 예측

## 실무 적용

```python
# 새 제품 조건
new_data = pd.DataFrame({
    'temperature': [85, 90, 80],
    'humidity': [50, 55, 45],
    'speed': [100, 95, 105]
})

# 선형회귀 예측
linear_pred = linear_model.predict(new_data)

# 다항회귀 예측
poly_pred = poly_model.predict(new_data)

print("예측 생산량:")
for i in range(len(new_data)):
    print(f"  조건 {i+1}: 선형 {linear_pred[i]:.0f}개, "
          f"다항 {poly_pred[i]:.0f}개")
```

---

# 핵심 정리

## 15차시 요약

| 개념 | 설명 |
|------|------|
| **회귀** | 연속적인 숫자 예측 |
| **선형회귀** | y = wx + b (직선) |
| **다중선형회귀** | 여러 특성으로 예측 |
| **MSE/RMSE** | 예측 오차 측정 |
| **R²** | 모델 설명력 (0~1) |
| **다항회귀** | 곡선 관계 학습 |

---

# 실무 가이드

## 회귀 모델 선택

<div class="highlight">

### 선형회귀 적합한 경우
- 특성과 타겟이 **직선 관계**
- **해석이 중요**할 때 (계수 의미)
- 빠른 학습/예측 필요

### 다항회귀 적합한 경우
- **곡선 관계**가 있을 때
- degree는 **2~3**부터 시작
- 과대적합 **주의**

</div>

---

# 다음 차시 예고

## 15차시: 모델 평가와 반복 검증

### 학습 내용
- 교차검증 (Cross-Validation)
- 과대적합/과소적합 진단
- 분류 평가 지표 심화

<div class="tip">

모델 평가를 **더 정확하게** 하는 방법을 배웁니다!

</div>

---

# 감사합니다

## 15차시: 예측 모델 - 선형/다항회귀

**숫자 예측의 기초를 배웠습니다!**

Q&A
