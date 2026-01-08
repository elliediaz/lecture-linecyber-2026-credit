# [15차시] 예측 모델 - 선형회귀와 다항회귀

## 학습 목표

| 번호 | 목표 |
|:----:|------|
| 1 | 회귀 문제의 개념을 이해함 |
| 2 | 선형회귀 모델(LinearRegression)을 사용함 |
| 3 | 다항회귀로 비선형 관계를 학습함 |

---

## 실습 데이터: California Housing 데이터셋

sklearn에서 제공하는 캘리포니아 주택 가격 데이터셋을 활용함

### 데이터 특성

| 변수 | 설명 | 단위 |
|------|------|------|
| MedInc | 중위 소득 | 만 달러 |
| HouseAge | 주택 연식 | 년 |
| AveRooms | 평균 방 개수 | 개 |
| AveBedrms | 평균 침실 개수 | 개 |
| Population | 인구 | 명 |
| AveOccup | 평균 거주자 수 | 명 |
| Latitude | 위도 | - |
| Longitude | 경도 | - |
| **MedHouseVal** | **중간 주택 가격 (타겟)** | **$100,000** |

---

## Part 1: 회귀 문제의 이해

### 1.1 분류 vs 회귀

| 구분 | 분류 (Classification) | 회귀 (Regression) |
|------|---------------------|-------------------|
| **출력** | 범주 (카테고리) | 연속적인 숫자 |
| **예시** | 정상/불량 | 생산량: 1,247개 |
| **질문** | "~인가요?" | "얼마나?" |
| **평가** | 정확도, F1 | MSE, R² |

### 제조 현장 회귀 문제 예시

| 문제 | 입력 | 출력 (예측값) |
|------|------|--------------|
| 생산량 예측 | 설비, 인력, 원료 | 1,247개 |
| 불량률 예측 | 온도, 습도, 속도 | 3.2% |
| 설비 수명 예측 | 가동 시간, 진동 | 87일 |
| 소요 시간 예측 | 공정, 작업량 | 4.5시간 |

---

### 1.2 회귀 모델 평가 지표

#### MSE (Mean Squared Error)

```
MSE = (1/n) × Σ(실제값 - 예측값)²
```

- 오차를 제곱하여 평균을 구함
- 값이 작을수록 좋음
- 큰 오차에 더 큰 페널티를 부여함

#### RMSE (Root Mean Squared Error)

```
RMSE = √MSE
```

- MSE에 루트를 씌운 값
- 원래 단위로 해석 가능함 (예: 가격 단위)

#### R² (결정계수)

```
R² = 1 - (예측 오차 / 평균 오차)
```

- 모델이 데이터를 얼마나 설명하는지 나타냄
- 1에 가까울수록 좋음

---

### 1.3 R² 점수 해석

| R² 값 | 해석 |
|-------|------|
| 1.0 | 완벽한 예측 |
| 0.8+ | 좋음 |
| 0.5~0.8 | 보통 |
| 0.3~0.5 | 약함 |
| 0 이하 | 평균보다 못함 |

```
R² = 0.75 → 모델이 데이터 변동의 75%를 설명함
```

---

### 1.4 실습 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
```

---

```python
# California Housing 데이터셋 로딩
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 중간 주택 가격 (단위: $100,000)

print(f"데이터 크기: {df.shape}")
print(f"특성 이름: {list(housing.feature_names)}")
```

#### 실행 결과 해설

- 총 20,640개의 주택 데이터로 구성됨
- 8개의 특성과 1개의 타겟(주택 가격)이 있음
- 타겟 단위는 $100,000 (예: 2.5 = $250,000)

---

```python
# 데이터 정보 확인
print(f"\n타겟 (주택 가격) 정보:")
print(f"  평균: ${df['MedHouseVal'].mean()*100000:,.0f}")
print(f"  범위: ${df['MedHouseVal'].min()*100000:,.0f} ~ ${df['MedHouseVal'].max()*100000:,.0f}")

print(f"\n기술통계:")
print(df.describe().round(2))
```

---

### 1.5 데이터 분할

```python
# 주요 특성 선택
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
X = df[feature_columns]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"주택 가격 범위: ${y.min()*100000:,.0f} ~ ${y.max()*100000:,.0f}")
```

#### 특성 선택 이유

| 특성 | 선택 이유 |
|------|----------|
| MedInc | 소득은 주택 가격과 강한 상관관계가 있음 |
| HouseAge | 주택 연식이 가격에 영향을 미침 |
| AveRooms | 방 개수는 주택 크기의 지표임 |
| AveOccup | 거주자 수는 주거 밀도를 나타냄 |

---

## Part 2: 선형회귀 (Linear Regression)

### 2.1 선형회귀란?

입력과 출력 사이의 직선 관계를 학습하는 모델임

#### 단순 선형회귀 수식

```
y = wx + b
```

- **w**: 기울기 (weight, coefficient)
- **b**: 절편 (bias, intercept)
- **x**: 입력 (특성)
- **y**: 출력 (예측값)

#### 다중 선형회귀 수식

```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + b
```

여러 특성을 사용하여 예측함

---

### 2.2 직선의 의미

```
    y
    |
    |     /  기울기(w) = 변화량
    |   /    소득 1만달러 증가 시
    | /      주택 가격 $40,000 증가
    |/───────────────→ x
    |
    b ← 절편 (x=0일 때 y값)
```

#### 계수 해석 예시

```
주택가격 = 0.4 × 소득 + 0.01 × 연식 - 0.05 × 거주자수 + 절편

- 소득 1만달러 증가 → 주택가격 $40,000 증가
- 연식 1년 증가 → 주택가격 $1,000 증가
- 거주자 1명 증가 → 주택가격 $5,000 감소
```

---

### 2.3 선형회귀 모델 학습

```python
# 모델 생성 및 학습
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 예측
y_pred_linear = linear_model.predict(X_test)

# 평가
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

print(f"[선형회귀 결과]")
print(f"R² 점수: {r2_linear:.3f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f} (약 ${rmse_linear*100000:,.0f})")
print(f"MAE: {mae_linear:.4f} (약 ${mae_linear*100000:,.0f})")
```

#### 결과 해설

- R² = 0.5 정도면 모델이 데이터의 50%를 설명함
- RMSE가 0.7이면 평균적으로 $70,000 정도의 오차가 발생함
- MAE는 평균 절대 오차로, 해석이 직관적임

---

### 2.4 회귀 계수 확인

```python
# 계수 확인
print(f"[회귀 계수]")
print(f"절편 (intercept): {linear_model.intercept_:.4f}")
for col, coef in zip(feature_columns, linear_model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f"  {col}: {sign}{coef:.4f}")
```

#### 계수 해석

```python
print(f"[계수 해석]")
print(f"  → 중위 소득(MedInc) 1단위 증가 시 주택 가격 ${linear_model.coef_[0]*100000:,.0f} 증가")
print(f"  → 주택 연식(HouseAge) 1년 증가 시 주택 가격 ${linear_model.coef_[1]*100000:,.0f} 변화")
```

| 계수 | 의미 |
|------|------|
| **coef_** | 각 특성의 기울기 (영향력) |
| **intercept_** | 절편 (모든 특성이 0일 때 기준값) |

---

### 2.5 예측 결과 시각화

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 주택 가격')
plt.ylabel('예측 주택 가격')
plt.title(f'선형회귀 (R²={r2_linear:.3f})')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 시각화 해설

- 빨간 점선: 완벽한 예측선 (실제 = 예측)
- 점이 대각선에 가까울수록 예측이 정확함
- 퍼져 있을수록 오차가 큼

---

## Part 3: 다항회귀 (Polynomial Regression)

### 3.1 비선형 관계의 문제

```
    y
    |
    |    ● ●
    |   ●   ●      실제 관계: 곡선
    |  ●     ●
    | ●       ●
    |●──────────── 직선 회귀 (한계)
    └────────────→ x
```

데이터가 곡선 형태이면 직선으로는 표현에 한계가 있음

---

### 3.2 다항회귀란?

특성의 거듭제곱을 추가하여 곡선 관계를 학습하는 방법임

#### 2차 다항식

```
y = w₁x + w₂x² + b
```

#### 3차 다항식

```
y = w₁x + w₂x² + w₃x³ + b
```

차수가 높아질수록 복잡한 곡선을 표현할 수 있음

---

### 3.3 다항 특성 생성 (PolynomialFeatures)

```python
from sklearn.preprocessing import PolynomialFeatures

# 원본 특성: [x]
X_sample = [[2], [3], [4]]

# 2차 다항 특성 생성
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_sample)

print("원본 -> 다항 변환:")
print(X_poly)
# [[1, 2, 4],    # 1, x, x²
#  [1, 3, 9],
#  [1, 4, 16]]
```

#### 변환 결과 해설

| 원본 x | 변환 후 [1, x, x²] |
|--------|-------------------|
| 2 | [1, 2, 4] |
| 3 | [1, 3, 9] |
| 4 | [1, 4, 16] |

---

### 3.4 다항회귀 Pipeline 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 2차 다항회귀 파이프라인
poly_model_2 = Pipeline([
    ('scaler', StandardScaler()),           # 스케일링 (권장)
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # 다항 특성 생성
    ('linear', LinearRegression())          # 선형회귀
])

# 학습
poly_model_2.fit(X_train, y_train)

# 예측 및 평가
y_pred_poly2 = poly_model_2.predict(X_test)
r2_poly2 = r2_score(y_test, y_pred_poly2)
rmse_poly2 = np.sqrt(mean_squared_error(y_test, y_pred_poly2))

print(f"[2차 다항회귀 결과]")
print(f"R² 점수: {r2_poly2:.3f}")
print(f"RMSE: {rmse_poly2:.4f} (약 ${rmse_poly2*100000:,.0f})")
```

#### Pipeline 구조

```
원본 데이터 [X]
     ↓
StandardScaler (스케일링)
     ↓
PolynomialFeatures (다항 특성 생성)
     ↓
LinearRegression (선형회귀 학습)
     ↓
예측값 [y_pred]
```

---

### 3.5 차수(degree)의 영향

```
degree=1    degree=2    degree=5    degree=15
(과소적합)   (적절)     (좋음)      (과대적합)

   │─────     │ ●       │ ~~~●     │~~~●~~~~
   │  ● ●     │● ●      │●   ●     │● ● ● ●
   │ ●        │●        │●         │●
```

| 차수 | 특징 |
|------|------|
| 낮음 (1) | 과소적합, 단순한 직선 |
| 적절 (2-3) | 데이터 패턴을 잘 포착함 |
| 높음 (10+) | 과대적합, 노이즈까지 학습함 |

---

### 3.6 차수별 성능 비교

```python
degrees = [1, 2, 3]
train_scores = []
test_scores = []

print(f"{'차수':>4} {'학습 R²':>10} {'테스트 R²':>12}")
print("-" * 35)

for deg in degrees:
    if deg == 1:
        poly = LinearRegression()
    else:
        poly = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('linear', LinearRegression())
        ])
    poly.fit(X_train, y_train)

    train_score = poly.score(X_train, y_train)
    test_score = poly.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"{deg:>4} {train_score:>10.3f} {test_score:>12.3f}")

# 최적 차수
best_idx = np.argmax(test_scores)
print(f"\n최적 차수: {degrees[best_idx]} (테스트 R² = {test_scores[best_idx]:.3f})")
```

#### 결과 해설

```
차수=1:  학습 R²=0.52, 테스트 R²=0.51  ← 과소적합 가능성
차수=2:  학습 R²=0.58, 테스트 R²=0.56  ← 적절
차수=3:  학습 R²=0.60, 테스트 R²=0.55  ← 약간의 과대적합
```

학습 점수는 높은데 테스트 점수가 낮으면 과대적합임

---

### 3.7 모델 비교 요약

```python
print(f"{'모델':<20} {'R² 점수':>10} {'RMSE':>12}")
print("-" * 45)
print(f"{'선형회귀':<20} {r2_linear:>10.3f} {rmse_linear:>12.4f}")
print(f"{'다항회귀 (degree=2)':<20} {r2_poly2:>10.3f} {rmse_poly2:>12.4f}")
```

---

### 3.8 시각화: 실제 vs 예측 비교

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 선형회귀
ax1 = axes[0]
ax1.scatter(y_test, y_pred_linear, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('실제 주택 가격')
ax1.set_ylabel('예측 주택 가격')
ax1.set_title(f'선형회귀 (R²={r2_linear:.3f})')
ax1.grid(True, alpha=0.3)

# 다항회귀
ax2 = axes[1]
ax2.scatter(y_test, y_pred_poly2, alpha=0.3, s=10, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('실제 주택 가격')
ax2.set_ylabel('예측 주택 가격')
ax2.set_title(f'다항회귀 degree=2 (R²={r2_poly2:.3f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3.9 새 데이터 예측

```python
# 새 주택 조건
new_data = pd.DataFrame({
    'MedInc': [3.0, 5.0, 8.0, 10.0],      # 중위 소득 (만 달러)
    'HouseAge': [20, 15, 5, 10],           # 주택 연식 (년)
    'AveRooms': [5.0, 6.0, 7.0, 8.0],      # 평균 방 수
    'AveOccup': [3.0, 2.5, 2.0, 2.2]       # 평균 거주자 수
})

print("[새 주택 조건]")
print(new_data)

# 예측
linear_pred = linear_model.predict(new_data)
poly2_pred = poly_model_2.predict(new_data)

print("\n[주택 가격 예측 결과]")
print(f"{'조건':>4} {'선형회귀':>15} {'다항(2차)':>15}")
print("-" * 40)
for i in range(len(new_data)):
    print(f"{i+1:>4} ${linear_pred[i]*100000:>13,.0f} ${poly2_pred[i]*100000:>13,.0f}")
```

---

### 3.10 단변량 회귀 시각화 (소득 vs 주택 가격)

```python
# 중위 소득만 사용한 회귀
X_income = df[['MedInc']]
y_price = df['MedHouseVal']

X_inc_train, X_inc_test, y_inc_train, y_inc_test = train_test_split(
    X_income, y_price, test_size=0.2, random_state=42
)

# 선형회귀
linear_1d = LinearRegression()
linear_1d.fit(X_inc_train, y_inc_train)

# 다항회귀 (2차)
poly2_1d = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly2_1d.fit(X_inc_train, y_inc_train)

# 시각화
X_plot = np.linspace(X_income.min()[0], X_income.max()[0], 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X_income.sample(2000, random_state=42),
            y_price.sample(2000, random_state=42), alpha=0.2, s=10, label='데이터')
plt.plot(X_plot, linear_1d.predict(X_plot), 'g-', lw=2,
         label=f'선형 (R²={linear_1d.score(X_inc_test, y_inc_test):.2f})')
plt.plot(X_plot, poly2_1d.predict(X_plot), 'b-', lw=2,
         label=f'다항 deg=2 (R²={poly2_1d.score(X_inc_test, y_inc_test):.2f})')

plt.xlabel('중위 소득 (MedInc)')
plt.ylabel('주택 가격 (단위: $100,000)')
plt.title('중위 소득 vs 주택 가격: 선형 vs 다항회귀')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 시각화 해설

- 녹색 직선: 선형회귀
- 파란색 곡선: 2차 다항회귀
- 다항회귀가 데이터의 곡선 패턴을 더 잘 포착함

---

### 3.11 특성별 영향력 분석

```python
# 각 특성의 회귀 계수 시각화
coef_df = pd.DataFrame({
    '특성': feature_columns,
    '계수': linear_model.coef_
})
coef_df['절대값'] = abs(coef_df['계수'])
coef_df = coef_df.sort_values('절대값', ascending=True)

plt.figure(figsize=(10, 5))
colors = ['red' if c < 0 else 'blue' for c in coef_df['계수']]
plt.barh(coef_df['특성'], coef_df['계수'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('회귀 계수')
plt.title('선형회귀 특성별 영향력\n(파랑=양의 영향, 빨강=음의 영향)')
plt.grid(True, alpha=0.3, axis='x')
plt.show()
```

#### 영향력 해석

| 색상 | 의미 |
|------|------|
| 파랑 | 양의 영향 (특성 증가 → 가격 증가) |
| 빨강 | 음의 영향 (특성 증가 → 가격 감소) |
| 막대 길이 | 영향력의 크기 |

---

## 15차시 핵심 정리

### 회귀 문제

| 항목 | 내용 |
|------|------|
| 정의 | 연속적인 숫자를 예측하는 문제 |
| 분류와 차이 | 분류: "~인가요?", 회귀: "얼마나?" |
| 평가 지표 | MSE, RMSE, MAE, R² |

### 선형회귀

| 항목 | 내용 |
|------|------|
| 수식 | y = w₁x₁ + w₂x₂ + ... + b |
| 장점 | 해석이 쉬움, 학습이 빠름 |
| 단점 | 비선형 관계 표현 불가 |
| 계수 | coef_ (기울기), intercept_ (절편) |

### 다항회귀

| 항목 | 내용 |
|------|------|
| 수식 | y = w₁x + w₂x² + w₃x³ + ... + b |
| 구성 | PolynomialFeatures + LinearRegression |
| 장점 | 곡선 관계 학습 가능 |
| 주의사항 | 차수가 높으면 과대적합 발생 |

---

## 실무 가이드

### 선형회귀가 적합한 경우

- 특성과 타겟이 직선 관계일 때
- 계수 해석이 중요할 때 (각 특성의 영향력)
- 빠른 학습과 예측이 필요할 때

### 다항회귀가 적합한 경우

- 데이터에 곡선 관계가 있을 때
- 낮은 차수(2~3)부터 시작함
- 학습/테스트 점수 차이로 과대적합 모니터링

### sklearn 사용법 요약

```python
# 선형회귀
model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)  # R²

# 다항회귀
poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)
poly_model.predict(X_test)
```

---

## 다음 차시 예고

### 16차시: 모델 평가와 반복 검증

학습 내용:
- 교차검증 (K-Fold Cross-Validation)
- 과대적합/과소적합 진단
- 혼동행렬과 분류 평가 지표 심화

모델 평가를 더 정확하게 하는 방법을 학습함
