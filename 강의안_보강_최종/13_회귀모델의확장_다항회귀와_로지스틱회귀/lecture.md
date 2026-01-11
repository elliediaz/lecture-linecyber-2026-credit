# [13차시] 회귀 모델의 확장: 다항회귀와 로지스틱 회귀

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **다항회귀**로 비선형 관계를 모델링함
2. **분류 문제와 예측 문제의 차이**를 구분함
3. **LogisticRegression**으로 이진 분류 모델을 생성함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Auto MPG** | seaborn-data | 다항회귀 실습 (연비 예측) |
| **Breast Cancer** | sklearn.datasets | 로지스틱 회귀 실습 (양성/악성 분류) |

주요 변수:
- Auto MPG: weight, horsepower, displacement -> mpg 예측
- Breast Cancer: 30개 특성 -> 악성(1)/양성(0) 분류

---

## 강의 구성

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| 1 | 다항회귀로 비선형 관계 모델링 | 8분 |
| 2 | 분류 문제와 시그모이드 함수 | 7분 |
| 3 | 제조 품질 분류 모델 실습 | 10분 |

---

## 파트 1: 다항회귀로 비선형 관계 모델링

### 개념 설명

#### 선형회귀의 한계

데이터가 곡선 형태의 관계를 가질 때 직선으로는 적절히 모델링할 수 없음.

실제 제조 데이터에서의 비선형 관계 예시:
- 온도가 높아질수록 불량률이 가속적으로 증가
- 압력과 품질의 2차 함수적 관계
- 가동 시간과 효율의 감소 곡선

#### 다항회귀 수학적 표현

$$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_n X^n$$

각 항의 의미:
- $\beta_0$: 절편 (X=0일 때 Y 값)
- $\beta_1 X$: 선형 항 (직선적 변화)
- $\beta_2 X^2$: 2차 항 (곡률 결정)
- $\beta_n X^n$: n차 항 (고차 곡선)

```mermaid
graph LR
    A[원본 특성 X] --> B[PolynomialFeatures]
    B --> C["[1, X, X^2, ..., X^n]"]
    C --> D[LinearRegression]
    D --> E[비선형 예측]
```

#### 다항회귀의 핵심 원리

다항회귀는 사실상 **다중선형회귀**와 동일함. X, X^2, X^3을 각각 독립적인 특성으로 취급하여 선형회귀를 적용함.

| 차수 | 수식 | 곡선 형태 |
|------|------|----------|
| 1 | $Y = \beta_0 + \beta_1 X$ | 직선 |
| 2 | $Y = \beta_0 + \beta_1 X + \beta_2 X^2$ | 포물선 |
| 3 | $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3$ | S자 곡선 |

#### 과대적합(Overfitting) 주의

차수가 높아지면 학습 데이터에 과도하게 맞춰지는 문제 발생함.

| 현상 | 설명 |
|------|------|
| 과소적합 | 차수가 낮아 데이터의 패턴을 포착하지 못함 |
| 적절 | 데이터의 실제 관계를 잘 표현함 |
| 과대적합 | 차수가 높아 노이즈까지 학습하여 일반화 실패 |

과대적합 징후: 학습 R^2는 높지만 테스트 R^2가 낮음.

### 실습 코드

#### 라이브러리 및 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# MPG 데이터셋 로드
mpg_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df_mpg = pd.read_csv(mpg_url)

# 결측치 제거 및 필요한 변수 선택
df = df_mpg[['mpg', 'horsepower', 'weight', 'displacement']].dropna()

print(f"데이터 크기: {df.shape}")
print(f"\n처음 5행:")
print(df.head())
print(f"\n기술 통계:")
print(df.describe().round(2))
```

---

#### 비선형 관계 시각화

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 마력과 연비 관계
axes[0].scatter(df['horsepower'], df['mpg'], alpha=0.6)
axes[0].set_xlabel('Horsepower')
axes[0].set_ylabel('MPG')
axes[0].set_title('Horsepower vs MPG')

# 무게와 연비 관계
axes[1].scatter(df['weight'], df['mpg'], alpha=0.6, color='orange')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('MPG')
axes[1].set_title('Weight vs MPG')

# 배기량과 연비 관계
axes[2].scatter(df['displacement'], df['mpg'], alpha=0.6, color='green')
axes[2].set_xlabel('Displacement')
axes[2].set_ylabel('MPG')
axes[2].set_title('Displacement vs MPG')

plt.tight_layout()
plt.show()

print("-> 곡선 형태의 관계가 보임. 선형회귀보다 다항회귀가 적합할 수 있음.")
```

---

#### 선형회귀 vs 다항회귀 비교

```python
# 마력으로 연비 예측
X = df[['horsepower']].values
y = df['mpg'].values

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. 선형회귀
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

# 2. 2차 다항회귀
poly2_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
poly2_model.fit(X_train, y_train)
y_pred_poly2 = poly2_model.predict(X_test)
r2_poly2 = r2_score(y_test, y_pred_poly2)

# 3. 3차 다항회귀
poly3_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linear', LinearRegression())
])
poly3_model.fit(X_train, y_train)
y_pred_poly3 = poly3_model.predict(X_test)
r2_poly3 = r2_score(y_test, y_pred_poly3)

print("=== 모델 비교 ===")
print(f"선형회귀 R^2: {r2_linear:.4f}")
print(f"2차 다항회귀 R^2: {r2_poly2:.4f}")
print(f"3차 다항회귀 R^2: {r2_poly3:.4f}")
print(f"\n-> 2차 다항회귀가 가장 좋은 성능을 보임")
```

---

#### 회귀선 시각화

```python
# 예측을 위한 X 범위 생성
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')

# 선형회귀 라인
plt.plot(X_plot, linear_model.predict(X_plot), 'r-', linewidth=2,
         label=f'Linear (R^2={r2_linear:.3f})')

# 2차 다항회귀 라인
plt.plot(X_plot, poly2_model.predict(X_plot), 'g-', linewidth=2,
         label=f'Polynomial deg=2 (R^2={r2_poly2:.3f})')

# 3차 다항회귀 라인
plt.plot(X_plot, poly3_model.predict(X_plot), 'b--', linewidth=2,
         label=f'Polynomial deg=3 (R^2={r2_poly3:.3f})')

plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

#### 차수별 과대적합 실험

```python
degrees = range(1, 11)
train_scores = []
test_scores = []

for deg in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# 결과 출력
print("=== 차수별 R^2 점수 ===")
print(f"{'차수':>4} {'학습 R^2':>10} {'테스트 R^2':>12}")
print("-" * 28)
for deg, train_r2, test_r2 in zip(degrees, train_scores, test_scores):
    marker = " <-- 최적" if deg == 2 else ""
    print(f"{deg:>4} {train_r2:>10.4f} {test_r2:>12.4f}{marker}")

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(degrees, train_scores, 'b-o', label='Train R^2', linewidth=2)
plt.plot(degrees, test_scores, 'r-o', label='Test R^2', linewidth=2)
plt.axvline(x=2, color='green', linestyle='--', label='Optimal degree=2')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.title('Training vs Test R^2 by Polynomial Degree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n-> 차수가 높아지면 학습 R^2는 증가하지만 테스트 R^2는 감소함 (과대적합)")
```

---

#### 다항회귀 계수 해석

```python
# 2차 다항회귀 모델의 계수 확인
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
feature_names = poly_features.get_feature_names_out(['horsepower'])

linear_reg = LinearRegression()
linear_reg.fit(X_train_poly, y_train)

print("=== 2차 다항회귀 계수 ===")
print(f"절편: {linear_reg.intercept_:.4f}")
for name, coef in zip(feature_names, linear_reg.coef_):
    print(f"{name}: {coef:.6f}")

print(f"\n회귀식: MPG = {linear_reg.intercept_:.2f} + "
      f"({linear_reg.coef_[0]:.4f}) * horsepower + "
      f"({linear_reg.coef_[1]:.6f}) * horsepower^2")
```

### 결과 해설

- 다항회귀는 비선형 관계를 효과적으로 모델링할 수 있음
- 차수가 너무 높으면 과대적합이 발생하여 새로운 데이터에 대한 예측 성능이 저하됨
- 적절한 차수 선택이 중요하며, 일반적으로 2~3차부터 시작하여 검증 점수를 비교함
- 학습 R^2와 테스트 R^2의 차이가 크면 과대적합 징후임

---

## 파트 2: 분류 문제와 시그모이드 함수

### 개념 설명

#### 분류 vs 회귀

| 구분 | 회귀 (Regression) | 분류 (Classification) |
|------|-------------------|----------------------|
| 출력 | 연속적인 숫자 | 범주 (카테고리) |
| 예시 | 온도: 25.7도, 연비: 32.5 mpg | 양품/불량, 정상/고장 |
| 질문 | "얼마나?" | "무엇인가?" |
| 평가 | MSE, RMSE, R^2 | 정확도, 정밀도, 재현율, F1 |

#### 분류 문제에 선형회귀를 사용할 수 없는 이유

```
불량(1) |         ***** <- 1 이상의 값?
        |      ***
        |   ***
        |***
양품(0) |*------------
        +--------------> 온도
```

선형회귀는 연속값을 출력하므로 0~1 범위를 보장하지 않음. 확률로 해석하기 어려움.

#### 시그모이드 함수 (Sigmoid Function)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

시그모이드 함수의 특성:
- 입력 z가 어떤 값이든 출력은 **0~1 사이**
- z = 0일 때 출력은 0.5
- z가 커지면 1에 수렴, z가 작아지면 0에 수렴
- **확률**로 해석 가능

| z 값 | $\sigma(z)$ | 해석 |
|------|-------------|------|
| -5 | 0.0067 | 거의 0 (양품) |
| -2 | 0.1192 | 낮은 확률 |
| 0 | 0.5000 | 반반 |
| +2 | 0.8808 | 높은 확률 |
| +5 | 0.9933 | 거의 1 (불량) |

#### 로지스틱 회귀 수학적 표현

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...)}}$$

단계별 이해:
1. **선형 결합**: $z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...$
2. **시그모이드 변환**: $P(Y=1) = \frac{1}{1 + e^{-z}}$
3. **분류 결정**: $P > 0.5$이면 클래스 1, 아니면 클래스 0

#### 오즈비와 로그 오즈

**오즈 (Odds)**: 사건이 발생할 확률과 발생하지 않을 확률의 비율

$$Odds = \frac{p}{1-p}$$

예시:
- p = 0.8이면 Odds = 0.8/0.2 = 4 (4배 더 발생)
- p = 0.5이면 Odds = 1 (동등)
- p = 0.2이면 Odds = 0.25

**로그 오즈 (Logit)**: 로지스틱 회귀의 핵심

$$\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...$$

로지스틱 회귀는 **로그 오즈를 선형 모델로 학습**함.

**오즈비 (Odds Ratio)**: 계수의 해석

$$e^{\beta_1} = \text{오즈비}$$

오즈비 해석:
- $e^{\beta} > 1$: 해당 특성 증가 시 확률 증가
- $e^{\beta} < 1$: 해당 특성 증가 시 확률 감소
- $e^{\beta} = 1$: 해당 특성이 확률에 영향 없음

### 실습 코드

#### 시그모이드 함수 시각화

```python
def sigmoid(z):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-z))

# z 값 범위
z = np.linspace(-10, 10, 100)

plt.figure(figsize=(10, 5))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5 (threshold)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function: sigma(z) = 1 / (1 + e^(-z))')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 주요 z 값에서의 시그모이드 값
print("=== 시그모이드 함수 값 ===")
for z_val in [-5, -2, -1, 0, 1, 2, 5]:
    print(f"z = {z_val:+2}: sigma(z) = {sigmoid(z_val):.4f}")
```

---

#### 분류 결정 경계 이해

```python
# 간단한 1D 분류 예시
np.random.seed(42)

# 양품 (클래스 0): 온도 낮음
X_good = np.random.normal(75, 5, 50)
y_good = np.zeros(50)

# 불량 (클래스 1): 온도 높음
X_bad = np.random.normal(95, 5, 50)
y_bad = np.ones(50)

X_demo = np.concatenate([X_good, X_bad]).reshape(-1, 1)
y_demo = np.concatenate([y_good, y_bad])

# 로지스틱 회귀 학습
from sklearn.linear_model import LogisticRegression

model_demo = LogisticRegression()
model_demo.fit(X_demo, y_demo)

# 결정 경계 시각화
X_range = np.linspace(60, 110, 200).reshape(-1, 1)
y_proba = model_demo.predict_proba(X_range)[:, 1]

plt.figure(figsize=(12, 5))

# 서브플롯 1: 확률 곡선
plt.subplot(1, 2, 1)
plt.scatter(X_good, y_good, alpha=0.6, label='Good (Class 0)', color='blue')
plt.scatter(X_bad, y_bad, alpha=0.6, label='Bad (Class 1)', color='red')
plt.plot(X_range, y_proba, 'g-', linewidth=2, label='P(Bad|Temp)')
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.xlabel('Temperature')
plt.ylabel('Probability / Class')
plt.title('Logistic Regression: Probability Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 2: 결정 경계
plt.subplot(1, 2, 2)
y_pred_demo = model_demo.predict(X_range)
plt.scatter(X_good, y_good + np.random.normal(0, 0.02, 50), alpha=0.6,
            label='Good (Class 0)', color='blue')
plt.scatter(X_bad, y_bad + np.random.normal(0, 0.02, 50), alpha=0.6,
            label='Bad (Class 1)', color='red')
decision_boundary = -model_demo.intercept_[0] / model_demo.coef_[0][0]
plt.axvline(x=decision_boundary, color='green', linewidth=2, linestyle='--',
            label=f'Decision Boundary ({decision_boundary:.1f})')
plt.xlabel('Temperature')
plt.ylabel('Class')
plt.title('Logistic Regression: Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n결정 경계 (온도): {decision_boundary:.2f}")
print(f"-> 온도가 {decision_boundary:.1f}도 이상이면 불량으로 분류")
```

### 결과 해설

- 시그모이드 함수는 모든 실수 값을 0~1 사이로 변환하여 확률로 해석 가능하게 함
- 로지스틱 회귀는 선형 결합 -> 시그모이드 변환 -> 분류 결정의 단계를 거침
- 결정 경계는 확률이 0.5가 되는 지점이며, 이 경계를 기준으로 클래스를 분류함
- 오즈비를 통해 각 특성이 분류 결과에 미치는 영향력을 해석할 수 있음

---

## 파트 3: 제조 품질 분류 모델 실습

### 개념 설명

#### sklearn LogisticRegression 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `C` | 정규화 강도의 역수 (작을수록 강한 정규화) | 1.0 |
| `max_iter` | 최대 반복 횟수 | 100 |
| `solver` | 최적화 알고리즘 | 'lbfgs' |
| `class_weight` | 클래스 가중치 (불균형 데이터 처리) | None |

#### 분류 평가 지표

**혼동 행렬 (Confusion Matrix)**:
```
                    예측
                양품(0)  불량(1)
         양품(0)   TN      FP
  실제
         불량(1)   FN      TP

TN (True Negative): 양품을 양품으로 예측 (정확)
TP (True Positive): 불량을 불량으로 예측 (정확)
FP (False Positive): 양품을 불량으로 예측 (1종 오류)
FN (False Negative): 불량을 양품으로 예측 (2종 오류, 위험!)
```

**주요 지표**:
| 지표 | 수식 | 의미 |
|------|------|------|
| 정확도 (Accuracy) | $(TP+TN)/(TP+TN+FP+FN)$ | 전체 중 맞은 비율 |
| 정밀도 (Precision) | $TP/(TP+FP)$ | 불량 예측 중 실제 불량 비율 |
| 재현율 (Recall) | $TP/(TP+FN)$ | 실제 불량 중 예측 성공 비율 |
| F1 Score | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | 정밀도와 재현율의 조화평균 |

제조업에서는 **재현율**이 특히 중요함. 불량 제품을 놓치면 고객 불만, 리콜 등 큰 비용 발생.

### 실습 코드

#### 데이터 로드 (Breast Cancer 데이터셋)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)

# 유방암 데이터셋 로드 (악성=1, 양성=0으로 변환하여 제조 품질 분류와 유사하게)
cancer = load_breast_cancer()
X = cancer.data
y = 1 - cancer.target  # 악성을 1(불량), 양성을 0(양품)으로 변환

# DataFrame으로 변환
df_cancer = pd.DataFrame(X, columns=cancer.feature_names)
df_cancer['target'] = y
df_cancer['class'] = y.astype(str).map({'0': 'Good', '1': 'Defect'})

print("=== Breast Cancer 데이터셋 (제조 품질 분류 유사) ===")
print(f"데이터 크기: {X.shape}")
print(f"특성 수: {X.shape[1]}")
print(f"\n클래스 분포:")
print(df_cancer['class'].value_counts())
print(f"\n특성 이름 (처음 10개):")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  {i+1}. {name}")
```

---

#### 데이터 분할

```python
# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== 데이터 분할 결과 ===")
print(f"학습 데이터: {len(X_train)}개 ({len(X_train)/len(X):.0%})")
print(f"테스트 데이터: {len(X_test)}개 ({len(X_test)/len(X):.0%})")
print(f"\n학습 데이터 클래스 비율:")
print(f"  양품(0): {(y_train==0).sum()}개 ({(y_train==0).mean():.1%})")
print(f"  불량(1): {(y_train==1).sum()}개 ({(y_train==1).mean():.1%})")
```

---

#### 로지스틱 회귀 모델 학습

```python
# 모델 생성 및 학습
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("=== 로지스틱 회귀 모델 학습 완료 ===")
print(f"학습 정확도: {model.score(X_train, y_train):.1%}")
print(f"테스트 정확도: {model.score(X_test, y_test):.1%}")
```

---

#### 예측 결과 확인

```python
# 예측 결과 샘플
print("=== 예측 결과 샘플 (처음 10개) ===")
print(f"{'실제':>6} {'예측':>6} {'P(양품)':>10} {'P(불량)':>10} {'결과':>6}")
print("-" * 45)
for i in range(10):
    actual = '양품' if y_test[i] == 0 else '불량'
    predicted = '양품' if y_pred[i] == 0 else '불량'
    result = 'O' if y_test[i] == y_pred[i] else 'X'
    print(f"{actual:>6} {predicted:>6} {y_proba[i][0]:>10.4f} {y_proba[i][1]:>10.4f} {result:>6}")
```

---

#### 혼동 행렬 분석

```python
# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)

print("=== 혼동 행렬 ===")
print(f"              예측")
print(f"            양품(0)  불량(1)")
print(f"실제 양품(0)   {cm[0,0]:4}     {cm[0,1]:4}")
print(f"     불량(1)   {cm[1,0]:4}     {cm[1,1]:4}")

print("\n=== 해석 ===")
print(f"TN (양품->양품): {cm[0,0]}개 - 정확히 양품으로 분류")
print(f"FP (양품->불량): {cm[0,1]}개 - 양품을 불량으로 잘못 분류 (과잉 검출)")
print(f"FN (불량->양품): {cm[1,0]}개 - 불량을 양품으로 잘못 분류 (위험!)")
print(f"TP (불량->불량): {cm[1,1]}개 - 정확히 불량으로 분류")

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Predicted Good', 'Predicted Defect'])
plt.yticks([0, 1], ['Actual Good', 'Actual Defect'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=20, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

---

#### 평가 지표 계산

```python
# 평가 지표
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== 분류 평가 지표 ===")
print(f"정확도 (Accuracy):  {accuracy:.4f} ({accuracy:.1%})")
print(f"정밀도 (Precision): {precision:.4f} ({precision:.1%})")
print(f"재현율 (Recall):    {recall:.4f} ({recall:.1%})")
print(f"F1 Score:           {f1:.4f}")

print("\n=== 지표 해석 ===")
print(f"- 정확도 {accuracy:.1%}: 전체 예측 중 {accuracy:.1%}가 정확함")
print(f"- 정밀도 {precision:.1%}: 불량 예측 중 {precision:.1%}가 실제 불량")
print(f"- 재현율 {recall:.1%}: 실제 불량 중 {recall:.1%}를 불량으로 예측")

if recall < 0.95:
    print("\n[주의] 재현율이 95% 미만. 불량 제품을 놓칠 수 있음!")
```

---

#### 분류 보고서

```python
# 분류 보고서
print("=== 분류 보고서 ===")
print(classification_report(y_test, y_pred, target_names=['Good', 'Defect']))
```

---

#### 계수 분석 (오즈비)

```python
# 상위 10개 중요 특성
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'coefficient': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
}).sort_values('odds_ratio', ascending=False)

print("=== 불량 확률을 높이는 특성 (오즈비 > 1) ===")
top_positive = feature_importance.head(5)
for _, row in top_positive.iterrows():
    print(f"  {row['feature']}: 계수={row['coefficient']:.4f}, 오즈비={row['odds_ratio']:.2f}")
    print(f"    -> 1 증가 시 불량 오즈가 {row['odds_ratio']:.2f}배")

print("\n=== 불량 확률을 낮추는 특성 (오즈비 < 1) ===")
top_negative = feature_importance.tail(5)
for _, row in top_negative.iterrows():
    print(f"  {row['feature']}: 계수={row['coefficient']:.4f}, 오즈비={row['odds_ratio']:.4f}")
    print(f"    -> 1 증가 시 불량 오즈가 {row['odds_ratio']:.2%}로 감소")
```

---

#### 특성 중요도 시각화

```python
# 계수 기준 상위/하위 10개
top_features = feature_importance.head(10)
bottom_features = feature_importance.tail(10)
combined = pd.concat([top_features, bottom_features])

plt.figure(figsize=(12, 8))
colors = ['red' if c > 0 else 'blue' for c in combined['coefficient']]
plt.barh(range(len(combined)), combined['coefficient'], color=colors)
plt.yticks(range(len(combined)), combined['feature'])
plt.xlabel('Coefficient')
plt.title('Logistic Regression Coefficients\n(Red: Increases Defect Probability, Blue: Decreases)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()
```

---

#### 새 데이터 예측

```python
# 새로운 제품 데이터로 예측
# 테스트 데이터의 처음 5개를 새 제품이라고 가정
new_products = X_test[:5]

predictions = model.predict(new_products)
probabilities = model.predict_proba(new_products)

print("=== 새 제품 품질 예측 ===")
print(f"{'제품':>4} {'예측':>6} {'양품 확률':>10} {'불량 확률':>10} {'판정':>8}")
print("-" * 45)
for i in range(5):
    pred = '양품' if predictions[i] == 0 else '불량'
    conf = max(probabilities[i]) * 100
    status = 'HIGH' if conf > 90 else 'MED' if conf > 70 else 'LOW'
    print(f"{i+1:>4} {pred:>6} {probabilities[i][0]:>10.2%} "
          f"{probabilities[i][1]:>10.2%} {status:>8}")
```

---

#### 임계값 조정 (재현율 향상)

```python
# 기본 임계값: 0.5
# 제조업에서는 불량을 놓치지 않기 위해 임계값을 낮출 수 있음

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("=== 임계값별 성능 비교 ===")
print(f"{'임계값':>6} {'정확도':>8} {'정밀도':>8} {'재현율':>8} {'F1':>8}")
print("-" * 45)

for threshold in thresholds:
    y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_th = f1_score(y_test, y_pred_thresh, zero_division=0)

    marker = " <-- 기본" if threshold == 0.5 else ""
    print(f"{threshold:>6.1f} {acc:>8.1%} {prec:>8.1%} {rec:>8.1%} {f1_th:>8.3f}{marker}")

print("\n-> 임계값을 낮추면 재현율이 높아지지만 정밀도가 낮아짐")
print("   제조업에서는 불량을 놓치지 않는 것이 중요하므로 낮은 임계값 고려")
```

### 결과 해설

- 로지스틱 회귀는 이진 분류 문제에 효과적인 선형 분류 모델임
- 혼동 행렬을 통해 모델의 오분류 패턴을 파악할 수 있음
- 정밀도와 재현율은 trade-off 관계이며, 문제의 특성에 따라 적절한 지표를 선택함
- 제조업에서는 불량을 놓치지 않는 것이 중요하므로 재현율(Recall)을 우선시함
- 오즈비를 통해 각 특성이 불량 확률에 미치는 영향력을 해석할 수 있음
- 임계값 조정을 통해 정밀도와 재현율 간의 균형을 조절할 수 있음

---

## 연습 문제

### 연습 1

MPG 데이터에서 weight로 mpg를 예측하는 다항회귀 모델을 학습하고, 최적의 차수를 찾으시오.

```python
# 정답
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X_w = df[['weight']].values
y_w = df['mpg'].values
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.2, random_state=42)

best_score = 0
best_degree = 1

for deg in range(1, 6):
    model_w = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('linear', LinearRegression())
    ])
    model_w.fit(X_train_w, y_train_w)
    score = model_w.score(X_test_w, y_test_w)
    if score > best_score:
        best_score = score
        best_degree = deg
    print(f"degree={deg}: R^2={score:.4f}")

print(f"\n최적 차수: {best_degree}, R^2: {best_score:.4f}")
```

---

### 연습 2

로지스틱 회귀로 품질 분류 모델을 학습하고, 재현율 95% 이상을 달성하는 임계값을 찾으시오.

```python
# 정답
from sklearn.metrics import recall_score

for threshold in np.arange(0.1, 0.6, 0.05):
    y_pred_th = (y_proba[:, 1] >= threshold).astype(int)
    rec = recall_score(y_test, y_pred_th)
    if rec >= 0.95:
        print(f"임계값 {threshold:.2f}: 재현율 {rec:.1%}")
        break
```

---

### 연습 3

로지스틱 회귀 모델에서 불량 확률을 가장 높이는 특성 3개를 찾고 오즈비를 해석하시오.

```python
# 정답
top3 = feature_importance.head(3)
print("불량 확률을 가장 높이는 특성 3개:")
for _, row in top3.iterrows():
    print(f"  {row['feature']}")
    print(f"    계수: {row['coefficient']:.4f}")
    print(f"    오즈비: {row['odds_ratio']:.2f}")
    print(f"    해석: 이 특성이 1 증가하면 불량 오즈가 {row['odds_ratio']:.2f}배 증가")
```

---

## 핵심 정리

### 다항회귀

| 개념 | 설명 |
|------|------|
| **다항회귀** | $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_n X^n$ |
| **특성 변환** | PolynomialFeatures로 $[1, X, X^2, ...]$ 생성 |
| **차수 선택** | 2~3차부터 시작, 과대적합 주의 |
| **과대적합 징후** | 학습 R^2 높고 테스트 R^2 낮음 |

### 로지스틱 회귀

| 개념 | 설명 |
|------|------|
| **시그모이드** | $\sigma(z) = \frac{1}{1 + e^{-z}}$, 0~1 범위 출력 |
| **로지스틱 회귀** | $P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ...)}}$ |
| **로그 오즈** | $\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + ...$ |
| **오즈비** | $e^{\beta}$, 특성의 영향력 해석 |

### 분류 평가 지표

| 지표 | 수식 | 제조업 중요도 |
|------|------|-------------|
| 정확도 | $(TP+TN)/전체$ | 보통 |
| 정밀도 | $TP/(TP+FP)$ | 보통 |
| 재현율 | $TP/(TP+FN)$ | **높음** (불량 검출) |
| F1 | 정밀도와 재현율의 조화평균 | 높음 |

### sklearn 사용법

```python
# 다항회귀
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)
```

---

## 다음 차시 예고

**14차시: 랜덤포레스트**

- 앙상블 학습의 개념
- 랜덤포레스트 알고리즘
- 특성 중요도 분석
- 다양한 분류기 비교
