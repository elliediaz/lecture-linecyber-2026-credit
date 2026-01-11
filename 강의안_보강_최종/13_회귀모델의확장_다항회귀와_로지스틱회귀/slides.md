---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 13차시'
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

# 회귀 모델의 확장

## 13차시 | 다항회귀와 로지스틱 회귀

**비선형 관계와 분류 문제 해결하기**

---

# 지난 시간 복습

## 12차시에서 배운 것

- **머신러닝 문제 유형**: 분류 vs 회귀
- **의사결정나무**: 질문 기반 분류
- **선형회귀**: $Y = \beta_0 + \beta_1 X$ 형태의 직선 관계

<div class="tip">

오늘은 **비선형 관계**와 **분류 문제**를 다루는 회귀 모델을 확장합니다.

</div>

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **다항회귀**로 비선형 관계를 모델링한다 |
| 2 | **분류 문제와 예측 문제의 차이**를 구분한다 |
| 3 | **LogisticRegression**으로 이진 분류 모델을 생성한다 |

---

# 오늘의 학습 흐름

```
+---------------+    +---------------+    +---------------+
|    Part 1     |    |    Part 2     |    |    Part 3     |
|   다항회귀    | -> |  분류 문제의  | -> | 로지스틱 회귀 |
|  Polynomial   |    |     이해      |    |   Logistic    |
+---------------+    +---------------+    +---------------+
  비선형 관계        분류 vs 회귀         확률 기반 분류
  차수 선택          출력의 차이          제조 품질 분류
```

---

<!-- _class: lead -->

# Part 1

## 다항회귀 (Polynomial Regression)

---

# 선형회귀의 한계

## 직선으로 맞추기 어려운 경우

```
    Y
    |    * *
    |   *   *      실제 관계: 곡선
    |  *     *
    | *       *
    |*---------*-- 직선 회귀 (부적합)
    +--------------> X
```

데이터가 **곡선** 형태면 직선으로는 한계!

---

# 다항회귀란?

## 비선형 관계를 학습하는 방법

> 특성의 **거듭제곱**을 추가하여 곡선 관계 학습

### 수학적 표현

$$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_n X^n$$

- $\beta_0$: 절편 (intercept)
- $\beta_1, \beta_2, ..., \beta_n$: 각 항의 계수
- $n$: 다항식의 차수 (degree)

---

# 다항식 차수별 예시

## 차수에 따른 곡선 형태

| 차수 | 수식 | 곡선 형태 |
|------|------|----------|
| 1차 (선형) | $Y = \beta_0 + \beta_1 X$ | 직선 |
| 2차 | $Y = \beta_0 + \beta_1 X + \beta_2 X^2$ | 포물선 |
| 3차 | $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3$ | S자 곡선 |

<div class="highlight">

다항회귀는 사실상 **다중선형회귀**와 동일!
$X, X^2, X^3$을 각각 **다른 특성**으로 취급

</div>

---

# 다항회귀의 원리

## 특성 변환 (Feature Engineering)

```
원본 데이터          변환 후 (degree=2)
    X          ->     [1, X, X^2]

예시:
    2          ->     [1, 2, 4]
    3          ->     [1, 3, 9]
    4          ->     [1, 4, 16]
```

변환 후 **선형회귀** 적용!

---

# sklearn PolynomialFeatures

## 다항 특성 생성

```python
from sklearn.preprocessing import PolynomialFeatures

# 원본 특성: [X]
X = [[2], [3], [4]]

# 2차 다항 특성 생성
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 결과: [1, X, X^2]
# [[1, 2, 4],
#  [1, 3, 9],
#  [1, 4, 16]]
```

---

# 차수(degree) 선택의 중요성

## 과소적합 vs 적절 vs 과대적합

```
degree=1        degree=2        degree=5        degree=15
(과소적합)       (적절)         (살짝 복잡)      (과대적합)

   |-----     |  *            |  ~~~*         |~~~*~~~~
   | * *      | * *           | *   *         |* * * *
   |*         |*              |*              |*
```

<div class="important">

차수가 **너무 높으면 과대적합**!
학습 데이터에만 맞고 새 데이터에 일반화 실패

</div>

---

# 과대적합 징후

## 학습 vs 테스트 점수 차이

```
degree=1:  학습 R^2=0.70, 테스트 R^2=0.68  <- 과소적합
degree=2:  학습 R^2=0.95, 테스트 R^2=0.94  <- 적절
degree=5:  학습 R^2=0.98, 테스트 R^2=0.92  <- 조금 과대적합
degree=15: 학습 R^2=0.99, 테스트 R^2=0.60  <- 심한 과대적합
```

<div class="tip">

학습 점수는 높은데 테스트 점수가 낮으면 **과대적합**!

</div>

---

# Part 1 정리

## 다항회귀 핵심

| 개념 | 설명 |
|------|------|
| **다항회귀** | $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ...$ |
| **특성 변환** | $X$를 $[1, X, X^2, ...]$로 변환 |
| **차수 선택** | 2~3차부터 시작, 과대적합 주의 |
| **sklearn** | `PolynomialFeatures(degree=n)` |

---

<!-- _class: lead -->

# Part 2

## 분류 문제의 이해

---

# 분류 vs 회귀

## 출력의 차이

| 구분 | 회귀 (Regression) | 분류 (Classification) |
|------|-------------------|----------------------|
| **출력** | 연속적인 숫자 | 범주 (카테고리) |
| **예시** | 온도: 25.7도 | 양품/불량 |
| **질문** | "얼마나?" | "무엇인가?" |
| **평가** | MSE, R^2 | 정확도, F1 |

---

# 분류 문제 예시

## 제조 현장

| 문제 | 입력 | 출력 (예측값) |
|------|------|--------------|
| 품질 분류 | 온도, 압력, 속도 | 양품 / 불량 |
| 고장 예측 | 진동, 소음, 온도 | 정상 / 고장 |
| 제품 등급 | 치수, 중량, 외관 | A / B / C 등급 |

<div class="highlight">

**이진 분류**: 2개 클래스 (양품/불량)
**다중 분류**: 3개 이상 클래스 (A/B/C 등급)

</div>

---

# 선형회귀로 분류?

## 문제점

```
    불량(1) |         ******* <- 1 이상?
            |      ***
            |   ***
            |***
    양품(0) |*----------------
            +-------------------> 온도

직선이 0~1 범위를 벗어남!
확률로 해석 불가능
```

회귀는 **연속값**을 예측하므로 **0~1 범위** 보장 안 됨

---

# 해결책: 로지스틱 함수

## 시그모이드 함수 (Sigmoid Function)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| z 값 | $\sigma(z)$ | 해석 |
|------|-------------|------|
| -10 | 0.00005 | 거의 0 |
| 0 | 0.5 | 정확히 0.5 |
| +10 | 0.99995 | 거의 1 |

> 어떤 값이든 **0~1 사이**로 변환!
> **확률**로 해석 가능

---

# 시그모이드 함수 그래프

## S자 곡선

```
    1.0 |              ********
        |          ****
    0.5 |       ***
        |    ***
    0.0 |****
        +-----------------------> z
       -6  -3   0   3   6
```

- z가 크면 1에 가까움 (불량 확률 높음)
- z가 작으면 0에 가까움 (양품 확률 높음)
- z=0이면 정확히 0.5 (반반)

---

# Part 2 정리

## 분류 문제 핵심

| 개념 | 설명 |
|------|------|
| **분류** | 범주(카테고리)를 예측 |
| **이진 분류** | 2개 클래스 (양품/불량) |
| **시그모이드** | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| **출력 범위** | 0~1 (확률로 해석) |

---

<!-- _class: lead -->

# Part 3

## 로지스틱 회귀 (Logistic Regression)

---

# 로지스틱 회귀란?

## 분류를 위한 확률 기반 모델

> 선형회귀 + 시그모이드 함수

### 수학적 표현

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...)}}$$

- $P(Y=1|X)$: X가 주어졌을 때 클래스 1(불량)일 확률
- $\beta_0$: 절편
- $\beta_1, \beta_2, ...$: 각 특성의 계수

---

# 로지스틱 회귀 작동 원리

## 2단계 변환

```
1단계: 선형 결합
z = B0 + B1*X1 + B2*X2 + ...

2단계: 시그모이드 변환
P(불량) = 1 / (1 + e^(-z))

3단계: 분류
P(불량) > 0.5 -> "불량"
P(불량) <= 0.5 -> "양품"
```

---

# 오즈비와 로그 오즈

## 수학적 배경

### 오즈 (Odds)

$$Odds = \frac{p}{1-p}$$

- p: 사건이 일어날 확률
- 예: p=0.8이면 Odds = 0.8/0.2 = 4 (4배 더 발생)

### 로그 오즈 (Logit)

$$\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...$$

> 로지스틱 회귀는 **로그 오즈**를 선형 모델로 학습

---

# 오즈비 해석

## 계수의 의미

$$e^{\beta_1} = \text{오즈비 (Odds Ratio)}$$

### 예시
- $\beta_1 = 0.5$ (온도 계수)
- $e^{0.5} \approx 1.65$

> 온도가 1 증가하면 불량 **오즈**가 1.65배 증가

<div class="tip">

오즈비 > 1: 해당 특성 증가 시 확률 증가
오즈비 < 1: 해당 특성 증가 시 확률 감소

</div>

---

# sklearn LogisticRegression

## 기본 사용법

```python
from sklearn.linear_model import LogisticRegression

# 모델 생성
model = LogisticRegression()

# 학습
model.fit(X_train, y_train)

# 예측 (클래스)
y_pred = model.predict(X_test)

# 예측 (확률)
y_proba = model.predict_proba(X_test)
```

---

# 모델 속성 확인

## 학습된 계수

```python
# 절편
print(f"절편 (B0): {model.intercept_[0]:.4f}")

# 계수
for name, coef in zip(feature_names, model.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{name}: 계수={coef:.4f}, 오즈비={odds_ratio:.2f}")
```

### 출력 예시
```
온도: 계수=0.50, 오즈비=1.65
압력: 계수=-0.30, 오즈비=0.74
```

---

# 분류 평가 지표

## 혼동 행렬 (Confusion Matrix)

```
                    예측
                양품    불량
         양품    TN      FP
  실제
         불량    FN      TP

- TN (True Negative): 양품을 양품으로 예측
- TP (True Positive): 불량을 불량으로 예측
- FP (False Positive): 양품을 불량으로 예측 (1종 오류)
- FN (False Negative): 불량을 양품으로 예측 (2종 오류)
```

---

# 정확도, 정밀도, 재현율

## 평가 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **정확도** | $(TP+TN)/(전체)$ | 전체 중 맞은 비율 |
| **정밀도** | $TP/(TP+FP)$ | 불량 예측 중 실제 불량 |
| **재현율** | $TP/(TP+FN)$ | 실제 불량 중 예측 성공 |
| **F1** | $2 \times \frac{정밀도 \times 재현율}{정밀도 + 재현율}$ | 정밀도와 재현율의 조화평균 |

<div class="important">

제조업에서는 **재현율**이 중요! (불량을 놓치지 않기)

</div>

---

# 제조 품질 분류 예시

## 양품/불량 분류 모델

```python
# 특성: 온도, 압력, 속도
# 타겟: 0=양품, 1=불량

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"정확도: {accuracy_score(y_test, y_pred):.1%}")
print(classification_report(y_test, y_pred))
```

---

# Part 3 정리

## 로지스틱 회귀 핵심

| 개념 | 설명 |
|------|------|
| **로지스틱 회귀** | 분류를 위한 확률 기반 모델 |
| **시그모이드** | 0~1 범위로 변환 |
| **오즈비** | $e^{\beta}$, 특성의 영향력 |
| **sklearn** | `LogisticRegression()` |
| **평가** | 정확도, 정밀도, 재현율, F1 |

---

# 핵심 정리

## 13차시 요약

| 개념 | 설명 |
|------|------|
| **다항회귀** | $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ...$ |
| **시그모이드** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **로지스틱 회귀** | $P(Y=1) = \frac{1}{1 + e^{-z}}$ |
| **로그 오즈** | $\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + ...$ |

---

# 실무 가이드

## 모델 선택 기준

<div class="highlight">

### 다항회귀 적합한 경우
- 데이터가 **곡선 관계**
- **연속적인 숫자** 예측
- degree는 2~3부터 시작

### 로지스틱 회귀 적합한 경우
- **범주(분류)** 예측
- **확률** 해석 필요
- 선형 분리 가능한 데이터

</div>

---

# 다음 차시 예고

## 14차시: 랜덤포레스트

### 학습 내용
- 앙상블 학습의 개념
- 랜덤포레스트 알고리즘
- 다양한 분류기 비교

<div class="tip">

여러 트리를 결합하여 **더 강력한 모델**을 만듭니다!

</div>

---

# 감사합니다

## 13차시: 회귀 모델의 확장

**비선형 관계와 분류 문제를 다루는 방법을 배웠습니다!**

Q&A
