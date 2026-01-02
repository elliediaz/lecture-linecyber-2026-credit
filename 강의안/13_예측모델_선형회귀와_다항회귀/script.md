# [13차시] 예측 모델: 선형회귀와 다항회귀 - 강사 스크립트

## 강의 정보
- **차시**: 13차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 13차시를 시작하겠습니다.
>
> 지난 두 차시에서 분류 모델을 배웠습니다. 의사결정나무와 랜덤포레스트로 "정상인가 불량인가"를 예측했죠.
>
> 오늘은 **회귀 모델**을 다룹니다. "생산량이 얼마나 될까?" 같은 **숫자**를 예측해봅니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 회귀 문제의 특징을 이해합니다.
> 둘째, 선형회귀로 생산량 예측 모델을 구축합니다.
> 셋째, 다항회귀로 비선형 관계를 모델링합니다.

---

### 핵심 내용 (8분)

#### 분류 vs 회귀 [1분]

> **분류**는 범주를 예측합니다. 불량인지 정상인지, A등급인지 B등급인지.
>
> **회귀**는 숫자를 예측합니다. 생산량이 몇 개인지, 불량률이 몇 퍼센트인지.
>
> 오늘부터 숫자를 예측하는 모델을 배웁니다.

#### 선형회귀 복습 [1.5분]

> **선형회귀(Linear Regression)**는 6차시에서 간단히 배웠었죠. X와 y 사이의 직선 관계를 찾는 방법입니다.
>
> y = wx + b
>
> w는 기울기, b는 절편입니다. 온도가 1도 올라갈 때 생산량이 얼마나 변하는지를 w가 알려주죠.
>
> sklearn에서 LinearRegression으로 쉽게 구현할 수 있습니다.

#### 다중 선형회귀 [1.5분]

> 실제로는 특성이 하나가 아니에요. 온도, 습도, 속도 여러 개를 사용합니다.
>
> 생산량 = 3.2×온도 + 1.5×습도 + 2.1×속도 + 100
>
> 이것을 **다중 선형회귀**라고 합니다. 코드는 단일 특성일 때와 같아요.
>
> ```python
> X = df[['온도', '습도', '속도']]
> y = df['생산량']
> model.fit(X, y)
> ```
>
> sklearn이 알아서 처리해줍니다.

#### 회귀 평가 지표 [1.5분]

> 분류에서는 정확도를 사용했죠. 회귀에서는 다른 지표를 씁니다.
>
> **MSE(Mean Squared Error)**는 오차 제곱의 평균입니다. 작을수록 좋아요.
>
> **RMSE**는 MSE의 제곱근이에요. 원래 단위로 해석할 수 있어서 편리합니다. "평균 50개 정도 오차"라고 말할 수 있죠.
>
> **R²(결정계수)**는 0에서 1 사이 값입니다. 1에 가까울수록 좋은 모델이에요. "모델이 데이터를 얼마나 잘 설명하는가"를 나타냅니다.

#### 다항회귀 [2.5분]

> 데이터가 곡선 형태라면 직선으로는 잘 안 맞아요.
>
> 예를 들어, 온도가 너무 낮거나 너무 높으면 둘 다 품질이 떨어질 수 있죠. 이런 U자형 관계는 직선으로 표현하기 어렵습니다.
>
> **다항회귀(Polynomial Regression)**는 다항식을 사용해서 곡선 관계를 모델링합니다.
>
> y = w₁x + w₂x² + b
>
> x² 같은 항을 추가해서 곡선을 표현하는 거예요.
>
> sklearn에서는 PolynomialFeatures를 사용합니다.
>
> ```python
> from sklearn.preprocessing import PolynomialFeatures
> poly = PolynomialFeatures(degree=2)
> X_poly = poly.fit_transform(X)
> ```
>
> 주의할 점이 있어요. degree를 너무 높이면 과대적합이 됩니다. 보통 2~3이면 충분해요.

---

## 실습편 (15-20분)

### 실습 소개 [2분]

> 이제 실습 시간입니다. 제조 데이터로 생산량 예측 모델을 만들어봅니다.
>
> **실습 목표**입니다.
> 1. 선형회귀 모델을 학습시킵니다.
> 2. MSE, RMSE, R²로 평가합니다.
> 3. 다항회귀로 성능을 비교합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from sklearn.linear_model import LinearRegression
> from sklearn.preprocessing import PolynomialFeatures
> from sklearn.metrics import mean_squared_error, r2_score
> ```

### 실습 1: 데이터 준비 [2분]

> 첫 번째 실습입니다. 제조 데이터를 생성합니다.
>
> 이번에는 **생산량**을 예측할 거예요. 분류가 아니라 회귀 문제입니다.
>
> ```python
> np.random.seed(42)
> n = 300
>
> df = pd.DataFrame({
>     '온도': np.random.normal(85, 5, n),
>     '습도': np.random.normal(50, 10, n),
>     '속도': np.random.normal(100, 15, n),
> })
>
> df['생산량'] = 10*df['온도'] + 3*df['습도'] + 2*df['속도'] + 노이즈
> ```
>
> 생산량은 온도, 습도, 속도에 따라 결정되도록 설정했습니다.

### 실습 2: 선형회귀 학습 [2min]

> 두 번째 실습입니다. 선형회귀 모델을 학습시킵니다.
>
> ```python
> from sklearn.linear_model import LinearRegression
>
> model = LinearRegression()
> model.fit(X_train, y_train)
>
> print(f"기울기: {model.coef_}")
> print(f"절편: {model.intercept_}")
> ```
>
> coef_가 각 특성의 기울기입니다. 온도의 기울기가 10에 가깝게 나오면 잘 학습된 거예요.

### 실습 3: 모델 평가 [2min]

> 세 번째 실습입니다. 모델 성능을 평가합니다.
>
> ```python
> from sklearn.metrics import mean_squared_error, r2_score
>
> y_pred = model.predict(X_test)
>
> mse = mean_squared_error(y_test, y_pred)
> rmse = np.sqrt(mse)
> r2 = r2_score(y_test, y_pred)
>
> print(f"RMSE: {rmse:.2f}")
> print(f"R²: {r2:.3f}")
> ```
>
> R²가 0.9 이상이면 매우 좋은 모델입니다. 90% 이상 설명력이 있다는 뜻이에요.

### 실습 4: 다항회귀 [3min]

> 네 번째 실습입니다. 다항회귀를 적용해봅니다.
>
> Pipeline을 사용하면 깔끔합니다.
>
> ```python
> from sklearn.preprocessing import PolynomialFeatures
> from sklearn.pipeline import Pipeline
>
> poly_pipe = Pipeline([
>     ('poly', PolynomialFeatures(degree=2)),
>     ('linear', LinearRegression())
> ])
>
> poly_pipe.fit(X_train, y_train)
> r2_poly = poly_pipe.score(X_test, y_test)
> ```
>
> 선형회귀보다 R²가 높아지면 비선형 관계가 있다는 뜻이에요.

### 실습 5: degree 실험 [2min]

> 다섯 번째 실습입니다. degree를 바꿔가며 성능 변화를 봅시다.
>
> ```python
> for degree in [1, 2, 3, 4, 5]:
>     pipe = Pipeline([
>         ('poly', PolynomialFeatures(degree=degree)),
>         ('linear', LinearRegression())
>     ])
>     pipe.fit(X_train, y_train)
>     train_r2 = pipe.score(X_train, y_train)
>     test_r2 = pipe.score(X_test, y_test)
>     print(f"degree={degree}: 학습={train_r2:.3f}, 테스트={test_r2:.3f}")
> ```
>
> degree가 높아지면 학습 R²는 올라가는데 테스트 R²는 떨어질 수 있어요. 이게 과대적합입니다.

### 실습 6: 트리 기반 회귀 [2min]

> 여섯 번째 실습입니다. 랜덤포레스트도 회귀에 사용할 수 있어요.
>
> ```python
> from sklearn.ensemble import RandomForestRegressor
>
> rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
> rf_reg.fit(X_train, y_train)
> print(f"RF 회귀 R²: {rf_reg.score(X_test, y_test):.3f}")
> ```
>
> Classifier를 Regressor로 바꾸면 됩니다. 사용법은 동일해요.

### 실습 7: 새 데이터 예측 [2min]

> 마지막 실습입니다. 새 데이터로 예측해봅니다.
>
> ```python
> new_data = [[88, 52, 105]]  # 온도, 습도, 속도
> pred = model.predict(new_data)
> print(f"예측 생산량: {pred[0]:.0f}개")
> ```
>
> 실무에서는 이렇게 새로운 조건에서의 생산량을 예측할 수 있습니다.

---

### 정리 (3분)

#### 핵심 요약 [1.5분]

> 오늘 배운 내용을 정리하겠습니다.
>
> **회귀**는 숫자를 예측합니다. 분류와 다르죠.
>
> **선형회귀**는 y = wx + b로 직선 관계를 모델링합니다.
>
> **다항회귀**는 PolynomialFeatures로 곡선 관계를 표현합니다.
>
> **평가 지표**는 MSE, RMSE, R²입니다. R²가 1에 가까울수록 좋은 모델이에요.

#### 다음 차시 예고 [1분]

> 다음 14차시에서는 **모델 평가와 반복 검증**을 배웁니다.
>
> 지금까지 train_test_split으로 한 번만 나눠서 평가했는데, 이것만으로는 부족해요. 더 신뢰할 수 있는 평가 방법인 교차검증을 배웁니다.

#### 마무리 [0.5분]

> 숫자 예측의 기초를 다졌습니다.
>
> 선형회귀는 가장 기본적인 모델이지만 실무에서도 많이 씁니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)

### 주의사항
- 분류와 회귀의 차이 명확히 설명
- R² 해석 방법 강조 (1에 가까울수록 좋음)
- degree 실험으로 과대적합 직접 확인

### 예상 질문
1. "R²가 마이너스가 나왔어요"
   → 모델이 평균보다 못함. 데이터나 모델 선택 재검토 필요

2. "degree를 몇으로 해야 하나요?"
   → 보통 2~3. 너무 높으면 과대적합

3. "선형회귀와 랜덤포레스트 중 뭐가 좋은가요?"
   → 상황에 따라 다름. 먼저 선형회귀로 시작

4. "MSE랑 RMSE 차이가 뭔가요?"
   → RMSE는 원래 단위로 해석 가능 (예: "50개 오차")
