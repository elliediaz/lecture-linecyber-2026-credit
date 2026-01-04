# [14차시] 회귀 모델: 선형회귀와 다항회귀 - 강사 스크립트

## 강의 정보
- **차시**: 14차시 (25분)
- **유형**: 이론 + 실습
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 지난 시간 복습 [1.5분]

> 안녕하세요, 14차시를 시작하겠습니다.
>
> 지난 두 차시에서 **분류 모델**을 배웠습니다. 의사결정트리와 랜덤포레스트로 "정상인가 불량인가"를 예측했죠.
>
> 오늘은 **회귀 모델**을 다룹니다. "생산량이 얼마나 될까?" 같은 **숫자**를 예측해봅니다.

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, 회귀 문제의 특징을 이해합니다.
> 둘째, 선형회귀로 예측 모델을 구축합니다.
> 셋째, 다항회귀로 비선형 관계를 모델링합니다.

---

## 전개 (19분)

### 섹션 1: 선형회귀 복습 (5분)

#### 7차시 내용 복습 [2분]

> **선형회귀(Linear Regression)**는 7차시에서 간단히 배웠었죠. X와 y 사이의 직선 관계를 찾는 방법입니다.
>
> y = wx + b
>
> w는 기울기, b는 절편입니다. 온도가 1도 올라갈 때 생산량이 얼마나 변하는지를 w가 알려주죠.

#### 다중 선형회귀 [3분]

> 실제로는 **여러 특성**을 사용합니다. 이것을 **다중 선형회귀**라고 해요.
>
> ```
> 생산량 = 3.2×온도 + 1.5×습도 + 2.1×속도 + 100
> ```
>
> *(코드 시연)*
>
> ```python
> from sklearn.linear_model import LinearRegression
>
> X = df[['온도', '습도', '속도']]  # 여러 특성
> y = df['생산량']
>
> model = LinearRegression()
> model.fit(X, y)
>
> print(model.coef_)        # 각 특성의 기울기
> print(model.intercept_)   # 절편
> ```
>
> 코드는 단일 특성일 때와 같아요. sklearn이 알아서 처리해줍니다.

---

### 섹션 2: 회귀 모델 평가 (4분)

#### 분류와 다른 평가 지표 [2분]

> 분류에서는 **정확도(Accuracy)**를 사용했죠. 맞은 개수 / 전체 개수.
>
> 회귀에서는 다른 지표를 사용합니다. 예측값과 실제값의 **차이**를 측정해요.

#### MSE, RMSE, R² [2min]

> **MSE (Mean Squared Error)**: 오차 제곱의 평균입니다. 값이 작을수록 좋아요.
>
> **RMSE (Root MSE)**: MSE의 제곱근이에요. 원래 단위와 같아서 해석하기 쉽습니다. 생산량을 예측한다면 "평균적으로 50개 정도 오차"라고 말할 수 있죠.
>
> **R² (결정계수)**: 0~1 사이 값으로, 1에 가까울수록 좋습니다. 모델이 데이터를 얼마나 잘 설명하는지 나타내요.
>
> *(코드 시연)*
>
> ```python
> from sklearn.metrics import mean_squared_error, r2_score
>
> mse = mean_squared_error(y_test, y_pred)
> r2 = r2_score(y_test, y_pred)
> ```

---

### 섹션 3: 다항회귀 (7분)

#### 선형회귀의 한계 [2min]

> 데이터가 곡선 형태라면 직선으로는 잘 안 맞아요.
>
> 예를 들어, 온도가 너무 낮거나 너무 높으면 둘 다 품질이 떨어질 수 있죠. 이런 U자형 관계는 직선으로 표현하기 어렵습니다.

#### 다항회귀 개념 [2min]

> **다항회귀(Polynomial Regression)**는 다항식을 사용해서 곡선 관계를 모델링합니다.
>
> ```
> 2차: y = w₁x + w₂x² + b  (포물선)
> 3차: y = w₁x + w₂x² + w₃x³ + b
> ```
>
> x² 같은 항을 추가해서 곡선을 표현하는 거예요.

#### sklearn 구현 [3min]

> *(코드 시연)*
>
> ```python
> from sklearn.preprocessing import PolynomialFeatures
> from sklearn.linear_model import LinearRegression
>
> # 특성 변환 (x → x, x²)
> poly = PolynomialFeatures(degree=2)
> X_poly = poly.fit_transform(X)
>
> # 선형회귀 적용
> model = LinearRegression()
> model.fit(X_poly, y)
> ```
>
> PolynomialFeatures가 x²을 만들어주고, 그 뒤에 선형회귀를 적용하는 방식이에요.
>
> Pipeline을 사용하면 더 깔끔합니다.
>
> ```python
> from sklearn.pipeline import Pipeline
>
> pipe = Pipeline([
>     ('poly', PolynomialFeatures(degree=2)),
>     ('linear', LinearRegression())
> ])
> pipe.fit(X_train, y_train)
> ```

---

### 섹션 4: 트리 기반 회귀 (3분)

#### 회귀도 가능 [1.5분]

> 의사결정트리와 랜덤포레스트도 회귀에 사용할 수 있어요!
>
> ```python
> from sklearn.tree import DecisionTreeRegressor
> from sklearn.ensemble import RandomForestRegressor
> ```
>
> Classifier가 Regressor로 바뀌었죠? 사용법은 동일합니다.

#### 언제 무엇을 쓸까 [1.5min]

> - **LinearRegression**: 가장 먼저 시도. 해석 쉬움
> - **PolynomialRegression**: 비선형 관계일 때
> - **RandomForestRegressor**: 높은 성능이 필요할 때
>
> 일반적으로 LinearRegression부터 시작해서 성능이 부족하면 다른 모델을 시도하세요.

---

## 정리 (3분)

### 핵심 내용 요약 [1.5분]

> 오늘 배운 핵심 내용을 정리하면:
>
> 1. **회귀**: 숫자를 예측 (분류와 다름!)
> 2. **선형회귀**: y = wx + b, 직선 관계
> 3. **다중 선형회귀**: 여러 특성 사용
> 4. **다항회귀**: 곡선 관계 모델링 (PolynomialFeatures)
> 5. **평가 지표**: MSE, RMSE, R²
>
> R²가 1에 가까울수록 좋은 모델입니다!

### 다음 차시 예고 [1min]

> 다음 15차시에서는 **모델 평가와 교차검증**을 배웁니다.
>
> 지금까지 train_test_split으로 한 번만 나눠서 평가했는데, 이것만으로는 부족해요. 더 신뢰할 수 있는 평가 방법을 배웁니다.

### 마무리 인사 [0.5분]

> 회귀 모델의 기초를 다졌습니다. 수고하셨습니다!

---

## 강의 노트

### 예상 질문
1. "degree를 몇으로 해야 하나요?"
   → 보통 2~3. 너무 높으면 과대적합

2. "R²가 마이너스가 나왔어요"
   → 모델이 평균보다 못함. 데이터나 모델 선택 재검토 필요

3. "선형회귀와 랜덤포레스트 중 뭐가 좋은가요?"
   → 상황에 따라 다름. 선형 관계면 선형회귀, 복잡하면 랜덤포레스트

### 시간 조절 팁
- 시간 부족: 트리 기반 회귀 부분 간략히
- 시간 여유: 선형 vs 다항 시각화 비교
