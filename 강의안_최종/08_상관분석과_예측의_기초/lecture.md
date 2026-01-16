# [8차시] 상관분석과 예측의 기초

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **상관계수**의 의미와 해석 방법을 이해함
2. **단순선형회귀**의 개념과 원리를 이해함
3. sklearn으로 **예측 모델**을 구현함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **MPG** | seaborn-data | 자동차 연비 예측 |

주요 변수:
- mpg: 연비 (miles per gallon)
- weight: 자동차 무게 (파운드)
- horsepower: 마력
- displacement: 배기량 (세제곱인치)
- cylinders: 실린더 수

---

## 강의 구성

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| 1 | 상관계수의 의미와 해석 방법 | 10분 |
| 2 | 단순선형회귀의 개념과 원리 | 10분 |
| 3 | sklearn으로 예측 모델 구현 | 10분 |

---

## 파트 1: 상관계수의 의미와 해석 방법

### 개념 설명

#### 상관분석이란?

두 변수가 함께 변하는 정도를 수치화하는 분석 방법임.

**제조 현장의 질문들**:
- "온도가 높아지면 불량률도 높아지는가?"
- "가동 시간과 생산량은 관련이 있는가?"
- "습도가 품질에 영향을 주는가?"

상관분석의 목적: 두 변수 사이의 **관계의 방향**과 **강도**를 파악함.

#### 상관관계의 종류

| 종류 | 의미 | 예시 |
|------|------|------|
| **양의 상관** | X 증가 -> Y 증가 | 온도 상승 -> 불량률 상승 |
| **음의 상관** | X 증가 -> Y 감소 | 교육시간 증가 -> 실수 감소 |
| **무상관** | 관계 없음 | 기계색상과 생산량 |

#### 상관관계 시각화

```
양의 상관 (r=0.9)    음의 상관 (r=-0.8)    무상관 (r=0)

    ***                ***               *  * *  *
   ***                  ***              * ** **
  ***                    ***            **  ** *
 ***                      ***           * *  ***
***                        ***          *  * * *
```

점들이 직선에 가깝게 모일수록 상관관계가 강함.

#### 상관계수 (Correlation Coefficient)

**피어슨 상관계수 (r)**: 두 변수의 선형적 관계의 강도와 방향을 나타냄.

- 범위: **-1 ~ +1**

| r 값 | 의미 |
|------|------|
| **r = +1** | 완벽한 양의 상관 |
| **r = -1** | 완벽한 음의 상관 |
| **r = 0** | 선형 관계 없음 |

#### 상관계수 해석 기준

| \|r\| 범위 | 해석 | 실무적 의미 |
|----------|------|----------|
| 0.0 ~ 0.3 | 약한 상관 | 관계 미미 |
| 0.3 ~ 0.5 | 보통 상관 | 어느 정도 관계 |
| 0.5 ~ 0.7 | 중간 상관 | 명확한 관계 |
| 0.7 ~ 1.0 | 강한 상관 | 밀접한 관계 |

예시:
- 온도 - 불량률: r = 0.85 -> 강한 양의 상관
- 교육시간 - 불량률: r = -0.65 -> 중간 음의 상관

#### 상관계수 계산 공식

```
        sum((Xi - X_mean)(Yi - Y_mean))
r = -----------------------------------------
    sqrt(sum((Xi - X_mean)^2)) x sqrt(sum((Yi - Y_mean)^2))

= 공분산(X, Y) / (표준편차X x 표준편차Y)
```

#### 주의: 상관관계와 인과관계

**상관관계 != 인과관계** (Correlation does not imply causation)

유명한 예시:
```
아이스크림 판매량과 익사 사고 수

- 상관계수: 높음 (r = 0.7)
- 인과관계: 없음!
- 숨겨진 변수: 기온 (더운 날 둘 다 증가)
```

제조업 예시:
```
설비 가동시간과 불량률

- 상관관계: 있을 수 있음
- 인과관계: 추가 분석 필요
- 숨겨진 변수: 작업자 피로, 원자재 상태 등
```

### 실습 코드

#### 데이터 로드 및 확인

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# MPG 데이터셋 로드
mpg_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df_full = pd.read_csv(mpg_url)

# 결측치 제거 및 수치형 변수만 선택
df = df_full[['mpg', 'weight', 'horsepower', 'displacement',
              'cylinders', 'acceleration']].dropna()

print(f"데이터 형태: {df.shape}")
print(f"\n처음 10행:")
print(df.head(10))

print("\n=== 변수 설명 ===")
print("mpg: 연비 (miles per gallon) - 높을수록 연비 좋음")
print("weight: 자동차 무게 (파운드)")
print("horsepower: 마력 - 엔진 출력")
print("displacement: 배기량 (세제곱인치)")
print("cylinders: 실린더 수 (4, 6, 8개)")
print("acceleration: 가속력 (0-60mph 도달 시간, 초)")

print(f"\n기술 통계:")
print(df.describe().round(2))
```

#### 산점도 시각화

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['weight'], df['mpg'], s=100, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=1.5)

plt.xlabel('Weight (lbs)', fontsize=12)
plt.ylabel('MPG (miles per gallon)', fontsize=12)
plt.title('Weight vs MPG Relationship', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print("-> 점들이 오른쪽 아래로 향함: 음의 상관관계 예상 (무게 증가 -> 연비 감소)")
```

#### 상관계수 계산

```python
# 방법 1: numpy
r_numpy = np.corrcoef(df['weight'], df['mpg'])[0, 1]

# 방법 2: pandas
r_pandas = df['weight'].corr(df['mpg'])

print(f"상관계수 (numpy): {r_numpy:.4f}")
print(f"상관계수 (pandas): {r_pandas:.4f}")

# 해석 함수
def interpret_correlation(r):
    """상관계수 해석"""
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "강한"
    elif abs_r >= 0.5:
        strength = "중간"
    elif abs_r >= 0.3:
        strength = "보통"
    else:
        strength = "약한"

    direction = "양의" if r > 0 else "음의" if r < 0 else "없는"
    return f"{strength} {direction} 상관관계"


print(f"\n해석: {interpret_correlation(r_numpy)}")
print(f"       (|r| = {abs(r_numpy):.2f})")
print(f"\n의미: 자동차 무게가 무거울수록 연비(mpg)가 낮아지는 강한 경향")
```

#### 다양한 상관관계 비교

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 강한 음의 상관 (무게 vs 연비)
r1 = df['weight'].corr(df['mpg'])
axes[0].scatter(df['weight'], df['mpg'], alpha=0.7)
axes[0].set_title(f'Weight vs MPG (r = {r1:.2f})')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('MPG')

# 강한 양의 상관 (무게 vs 배기량)
r2 = df['weight'].corr(df['displacement'])
axes[1].scatter(df['weight'], df['displacement'], alpha=0.7, color='coral')
axes[1].set_title(f'Weight vs Displacement (r = {r2:.2f})')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('Displacement')

# 약한 상관 (가속력 vs 연비)
r3 = df['acceleration'].corr(df['mpg'])
axes[2].scatter(df['acceleration'], df['mpg'], alpha=0.7, color='gray')
axes[2].set_title(f'Acceleration vs MPG (r = {r3:.2f})')
axes[2].set_xlabel('Acceleration')
axes[2].set_ylabel('MPG')

plt.tight_layout()
plt.show()

print(f"\n상관계수 요약:")
print(f"  - 무게 vs 연비: {r1:.2f} ({interpret_correlation(r1)})")
print(f"  - 무게 vs 배기량: {r2:.2f} ({interpret_correlation(r2)})")
print(f"  - 가속력 vs 연비: {r3:.2f} ({interpret_correlation(r3)})")
```

#### 상관행렬과 히트맵

```python
# 상관행렬
corr_matrix = df.corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))

# 히트맵
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# 컬러바
cbar = ax.figure.colorbar(im, ax=ax)
cbar.set_label('Correlation Coefficient', fontsize=12)

# 레이블
labels = corr_matrix.columns
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        value = corr_matrix.iloc[i, j]
        color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                fontsize=10, color=color)

ax.set_title('MPG Dataset Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== 주요 상관관계 해석 ===")
print("- mpg와 weight: 강한 음의 상관 (무거울수록 연비 나쁨)")
print("- mpg와 horsepower: 강한 음의 상관 (마력 높을수록 연비 나쁨)")
print("- weight와 displacement: 강한 양의 상관 (배기량 클수록 무거움)")
```

### 결과 해설

- 상관계수(r)는 -1에서 +1 사이의 값으로, 선형 관계의 강도와 방향을 나타냄
- \|r\| > 0.7이면 강한 상관관계로 해석함
- 상관행렬과 히트맵을 통해 여러 변수 간의 관계를 한눈에 파악할 수 있음
- 상관관계가 있다고 해서 인과관계가 있는 것은 아님. 도메인 지식과 함께 해석해야 함

---

## 파트 2: 단순선형회귀의 개념과 원리

### 개념 설명

#### 회귀분석이란?

변수 간 관계를 수학적 함수로 표현하는 분석 방법임.

| 용어 | 의미 | 예시 |
|------|------|------|
| **독립변수 (X)** | 원인, 예측에 사용 | 온도, 습도 |
| **종속변수 (Y)** | 결과, 예측 대상 | 불량률, 생산량 |

목적:
1. 변수 간 **관계 파악** (X가 Y에 미치는 영향)
2. **예측** (새로운 X로 Y 예측)

#### 단순선형회귀 모델

**수식**:

```
Y = B0 + B1*X + e

Y  : 종속변수 (예측 대상)
X  : 독립변수 (입력)
B0 : 절편 (intercept) - X=0일 때 Y값
B1 : 기울기 (slope) - X가 1 증가할 때 Y 변화량
e  : 오차항 (error)
```

예시:
```
불량률 = 1.5 + 0.05 x 온도

-> 온도가 1도 올라가면 불량률이 0.05%p 증가
```

#### 회귀선의 의미

```
Y (불량률)
|           *
|         * /
|       * /
|     * / <- 회귀선 (Y = B0 + B1*X)
|   * /
| * /
|*/
L-----------------> X (온도)
```

- 회귀선: 데이터를 가장 잘 설명하는 직선
- 각 점과 선 사이의 거리(오차)의 합이 최소

#### 최소제곱법 (OLS, Ordinary Least Squares)

목표: 오차의 제곱합을 최소화하는 B0, B1 찾기

```
최소화: sum((yi - y_hat_i)^2) = sum((yi - (B0 + B1*xi))^2)

yi    : 실제값
y_hat_i : 예측값
```

왜 제곱하는가?
- 오차의 부호를 없앰 (+,- 상쇄 방지)
- 큰 오차에 더 큰 페널티 부여

**공식**:

```
        sum((xi - x_mean)(yi - y_mean))
B1 = -----------------------------------
            sum((xi - x_mean)^2)

B0 = y_mean - B1 * x_mean
```

#### R-squared (결정계수)

모델의 설명력을 나타내는 지표임.

**정의**:

```
        SSR         잔차제곱합
R^2 = 1 - --- = 1 - -----------
        SST         총제곱합

SSR = sum((yi - y_hat_i)^2)  (잔차제곱합)
SST = sum((yi - y_mean)^2)   (총제곱합)
```

**해석**:
- 범위: 0 ~ 1
- 의미: 독립변수가 종속변수 변동의 몇 %를 설명하는가
- R^2 = 0.85: 온도가 불량률 변동의 85%를 설명함

| R^2 범위 | 해석 | 조치 |
|---------|------|------|
| 0.9 이상 | 매우 좋음 | 그대로 사용 |
| 0.7 ~ 0.9 | 좋음 | 실무 활용 가능 |
| 0.5 ~ 0.7 | 보통 | 추가 변수 검토 |
| 0.3 ~ 0.5 | 낮음 | 모델 개선 필요 |
| 0.3 미만 | 매우 낮음 | 다른 접근 필요 |

### 실습 코드

#### 최소제곱법 직접 구현

```python
# 데이터 (무게 -> 연비 예측)
X = df['weight'].values
Y = df['mpg'].values

# 최소제곱법 공식
x_mean = X.mean()
y_mean = Y.mean()

# 기울기 (B1)
numerator = np.sum((X - x_mean) * (Y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
beta1 = numerator / denominator

# 절편 (B0)
beta0 = y_mean - beta1 * x_mean

print("=== 최소제곱법 직접 계산 ===")
print(f"절편 (B0): {beta0:.4f}")
print(f"기울기 (B1): {beta1:.6f}")
print(f"\n회귀식: MPG = {beta0:.2f} + ({beta1:.6f}) * Weight")
print(f"해석: 무게가 1파운드 증가하면 연비가 {abs(beta1):.6f} mpg 감소")
print(f"      무게가 1000파운드 증가하면 연비가 약 {abs(beta1*1000):.2f} mpg 감소")
```

#### 잔차와 오차 계산

```python
# 예측값
Y_pred_manual = beta0 + beta1 * X

# 잔차
residuals_manual = Y - Y_pred_manual

# 오차 제곱합 (SSE)
SSE = np.sum(residuals_manual ** 2)

# 총 제곱합 (SST)
SST = np.sum((Y - y_mean) ** 2)

# R^2
R2_manual = 1 - SSE / SST

print("=== 오차 분석 ===")
print(f"SSE (잔차제곱합): {SSE:.4f}")
print(f"SST (총제곱합): {SST:.4f}")
print(f"R^2 (결정계수): {R2_manual:.4f}")
print(f"\n해석: 무게가 연비 변동의 {R2_manual*100:.1f}%를 설명")
```

### 결과 해설

- 회귀계수(B1)의 부호가 음수이면 음의 관계, 양수이면 양의 관계를 나타냄
- R^2가 높을수록 모델의 설명력이 좋음
- 잔차(실제값 - 예측값)의 합은 0에 가까워야 함
- 직접 계산한 결과와 sklearn 결과가 동일함을 확인할 수 있음

---

## 파트 3: sklearn으로 예측 모델 구현

### 개념 설명

#### scikit-learn 소개

Python 머신러닝 라이브러리임.

**특징**:
- 일관된 API (fit, predict, score)
- 다양한 알고리즘 지원
- 전처리, 평가 도구 포함

**설치**:

```bash
pip install scikit-learn
```

**기본 구조**:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # 모델 생성
model.fit(X, y)             # 학습
predictions = model.predict(X_new)  # 예측
```

#### 데이터 형태 주의

sklearn은 2D 배열이 필요함.

```python
# 잘못된 방법
X = temperature  # 1D 배열 (10,)
model.fit(X, y)  # 에러!

# 올바른 방법
# 방법 1: reshape
X = temperature.reshape(-1, 1)  # (10, 1)

# 방법 2: 슬라이싱 (DataFrame)
X = df[['온도']]  # 2D DataFrame

# 방법 3: np.newaxis
X = temperature[:, np.newaxis]
```

### 실습 코드

#### sklearn LinearRegression

```python
from sklearn.linear_model import LinearRegression

# 데이터 준비 (2D 배열로 변환 필수!)
X_2d = df[['weight']].values  # shape: (n, 1)
y = df['mpg'].values           # shape: (n,)

print(f"X shape: {X_2d.shape}")
print(f"y shape: {y.shape}")

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_2d, y)

print("\n=== sklearn 회귀 분석 결과 ===")
print(f"절편 (intercept_): {model.intercept_:.4f}")
print(f"기울기 (coef_): {model.coef_[0]:.6f}")
print(f"\n회귀식: MPG = {model.intercept_:.2f} + ({model.coef_[0]:.6f}) * Weight")

# 직접 계산과 비교
print("\n=== 직접 계산과 비교 ===")
print(f"절편 차이: {abs(model.intercept_ - beta0):.10f}")
print(f"기울기 차이: {abs(model.coef_[0] - beta1):.10f}")
print("-> sklearn과 직접 계산 결과가 동일함")
```

#### 예측

```python
# 기존 데이터 예측
y_pred = model.predict(X_2d)

# 새로운 데이터 예측 (다양한 무게)
new_weights = np.array([[2000], [2500], [3000], [3500], [4000], [4500]])
new_predictions = model.predict(new_weights)

print("=== 새로운 무게 예측 ===")
for weight, pred in zip(new_weights.flatten(), new_predictions):
    print(f"무게 {weight:,}lbs -> 예측 연비: {pred:.1f} mpg")
```

#### 모델 평가

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 평가 지표 계산
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print("=== 모델 평가 지표 ===")
print(f"R^2 (결정계수): {r2:.4f}")
print(f"MSE (평균제곱오차): {mse:.4f}")
print(f"RMSE (평균제곱근오차): {rmse:.4f}")
print(f"MAE (평균절대오차): {mae:.4f}")

# R^2 해석
if r2 >= 0.9:
    quality = "매우 좋음"
elif r2 >= 0.7:
    quality = "좋음"
elif r2 >= 0.5:
    quality = "보통"
else:
    quality = "개선 필요"

print(f"\n모델 품질: {quality}")
print(f"해석: 무게가 연비 변동의 {r2*100:.1f}%를 설명함.")
print(f"      평균적으로 약 {mae:.1f} mpg 오차로 예측함.")
```

#### 회귀선 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 회귀선
X_line = np.linspace(df['weight'].min()-100, df['weight'].max()+100, 100).reshape(-1, 1)
y_line = model.predict(X_line)

axes[0].scatter(df['weight'], df['mpg'], s=100, alpha=0.7,
                label='Actual Data', color='steelblue', edgecolor='black')
axes[0].plot(X_line, y_line, 'r-', linewidth=2, label='Regression Line')

# 회귀식 표시
eq_text = f'MPG = {model.intercept_:.2f} + ({model.coef_[0]:.6f}) * Weight\nR^2 = {r2:.4f}'
axes[0].text(df['weight'].max()-800, df['mpg'].max()-2, eq_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_xlabel('Weight (lbs)', fontsize=12)
axes[0].set_ylabel('MPG', fontsize=12)
axes[0].set_title('Weight vs MPG: Linear Regression', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 실제 vs 예측
axes[1].scatter(y, y_pred, s=100, alpha=0.7, color='green', edgecolor='black')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2,
             label='Perfect Prediction (y=x)')
axes[1].set_xlabel('Actual MPG', fontsize=12)
axes[1].set_ylabel('Predicted MPG', fontsize=12)
axes[1].set_title('Actual vs Predicted', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 잔차 분석

```python
from scipy import stats

residuals = y - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 잔차 플롯
axes[0].scatter(y_pred, residuals, s=80, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted MPG', fontsize=12)
axes[0].set_ylabel('Residual', fontsize=12)
axes[0].set_title('Residual Plot\n(No pattern = Good)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 잔차 히스토그램
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution\n(Should be normal)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# 잔차 Q-Q 플롯
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[2].scatter(osm, osr, s=80, alpha=0.7, color='steelblue', edgecolor='black')
axes[2].plot(osm, slope*osm + intercept, 'r-', linewidth=2)
axes[2].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[2].set_ylabel('Sample Quantiles', fontsize=12)
axes[2].set_title('Q-Q Plot\n(Should follow the line)', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"=== 잔차 통계 ===")
print(f"잔차 평균: {residuals.mean():.6f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.4f}")
print(f"잔차 최소: {residuals.min():.4f}")
print(f"잔차 최대: {residuals.max():.4f}")
```

#### 역 예측 (목표 연비를 위한 무게 계산)

```python
def predict_weight_for_target_mpg(target_mpg, model):
    """목표 연비를 위한 최대 무게 계산"""
    # Y = B0 + B1*X -> X = (Y - B0) / B1
    required_weight = (target_mpg - model.intercept_) / model.coef_[0]
    return required_weight


print("=== 목표 연비를 위한 최대 무게 계산 ===")
for target in [20.0, 25.0, 30.0, 35.0, 40.0]:
    weight = predict_weight_for_target_mpg(target, model)
    print(f"목표 연비 {target} mpg -> 무게 {weight:.0f} lbs 이하 유지")
```

#### 다중선형회귀 미리보기

```python
# 여러 독립변수 사용
X_multi = df[['weight', 'horsepower', 'displacement']].values
y_multi = df['mpg'].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("=== 다중선형회귀 결과 ===")
print(f"절편: {model_multi.intercept_:.4f}")
print("\n각 변수의 기울기 (영향력):")
for name, coef in zip(['weight', 'horsepower', 'displacement'], model_multi.coef_):
    print(f"  {name}: {coef:+.6f}")

# 다중회귀 R^2
y_pred_multi = model_multi.predict(X_multi)
r2_multi = r2_score(y_multi, y_pred_multi)
print(f"\nR^2 (다중회귀): {r2_multi:.4f}")
print(f"R^2 (단순회귀): {r2:.4f}")
print(f"R^2 개선: {r2_multi - r2:.4f}")
print("\n-> 변수 추가로 설명력 향상!")
```

### 결과 해설

- sklearn의 LinearRegression은 일관된 API(fit, predict, score)를 제공함
- X는 반드시 2D 배열로 변환해야 함 (reshape 또는 DataFrame 슬라이싱)
- R^2와 RMSE/MAE를 통해 모델 성능을 평가함
- 잔차 분석을 통해 모델의 가정 충족 여부를 확인함
- 역 예측을 활용하면 목표 Y를 달성하기 위한 X 조건을 계산할 수 있음
- 다중회귀는 여러 변수를 사용하여 설명력을 향상시킬 수 있음

---

## 연습 문제

### 연습 1

MPG 데이터에서 horsepower와 mpg의 상관계수를 계산하고 해석하시오.

```python
# 정답
r_hp_mpg = df['horsepower'].corr(df['mpg'])
print(f"상관계수: {r_hp_mpg:.4f}")
print(f"해석: {interpret_correlation(r_hp_mpg)}")
```

### 연습 2

horsepower로 mpg를 예측하는 선형회귀 모델을 학습하고 회귀식을 출력하시오.

```python
# 정답
X_hp = df[['horsepower']].values
y_mpg = df['mpg'].values
model_hp = LinearRegression()
model_hp.fit(X_hp, y_mpg)
print(f"회귀식: MPG = {model_hp.intercept_:.2f} + ({model_hp.coef_[0]:.4f}) * Horsepower")
print(f"해석: 마력 1 증가 -> 연비 {abs(model_hp.coef_[0]):.4f} mpg 감소")
```

### 연습 3

horsepower가 200일 때 예상 mpg를 예측하시오.

```python
# 정답
pred_200hp = model_hp.predict([[200]])[0]
print(f"Horsepower 200 -> 예상 MPG: {pred_200hp:.2f}")
```

### 연습 4

mpg를 25 이상 달성하려면 horsepower가 얼마 이하여야 하는지 계산하시오.

```python
# 정답
target_mpg = 25
required_hp = (target_mpg - model_hp.intercept_) / model_hp.coef_[0]
print(f"MPG {target_mpg} 이상을 위한 최대 Horsepower: {required_hp:.1f}")
```

---

## 핵심 정리

| 구분 | 내용 |
|------|------|
| **상관계수** | -1 ~ +1, 두 변수의 선형 관계 강도와 방향 |
| **상관계수 해석** | \|r\| > 0.7 강함, 0.5~0.7 중간, 0.3~0.5 보통, < 0.3 약함 |
| **선형회귀** | Y = B0 + B1*X로 관계 표현 |
| **최소제곱법** | 잔차제곱합을 최소화하는 계수 추정 |
| **R^2** | 모델의 설명력 (0~1, 높을수록 좋음) |
| **sklearn** | fit() -> predict() -> score() 패턴 |
| **주의사항** | X는 2D 배열, 상관관계 != 인과관계 |

---

## 다음 차시 예고

**9차시: 제조 데이터 전처리 (1)**

- 결측치 탐지 및 처리
- 이상치 처리 방법
- fillna, dropna 활용
