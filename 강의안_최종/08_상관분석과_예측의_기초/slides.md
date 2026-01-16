---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 8차시'
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
---

# 상관분석과 예측의 기초

## 8차시 | Part II. 기초 수리와 데이터 분석

**두 변수의 관계를 분석하고 예측하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **상관계수**의 의미와 해석 방법을 이해한다
2. **단순선형회귀**의 개념과 원리를 이해한다
3. sklearn으로 **예측 모델**을 구현한다

---

# 강의 구성

| 파트 | 대주제 | 시간 |
|:----:|--------|:----:|
| 1 | 상관계수의 의미와 해석 방법 | 10분 |
| 2 | 단순선형회귀의 개념과 원리 | 10분 |
| 3 | sklearn으로 예측 모델 구현 | 10분 |

---

<!-- _class: lead -->

# Part 1
## 상관계수의 의미와 해석 방법

---

# 상관분석이란?

## 두 변수가 함께 변하는 정도를 수치화

### 제조 현장의 질문들
- "온도가 높아지면 불량률도 높아지는가?"
- "가동 시간과 생산량은 관련이 있는가?"
- "습도가 품질에 영향을 주는가?"

### 상관분석의 목적
> 두 변수 사이의 **관계의 방향**과 **강도**를 파악

---

# 상관관계의 종류

## 양의 상관 vs 음의 상관 vs 무상관

| 종류 | 의미 | 예시 |
|------|------|------|
| **양의 상관** | X 증가 → Y 증가 | 온도↑ → 불량률↑ |
| **음의 상관** | X 증가 → Y 감소 | 교육시간↑ → 실수↓ |
| **무상관** | 관계 없음 | 기계색상 ↔ 생산량 |

---

# 상관관계 시각화

## 산점도로 관계 확인

```
양의 상관 (r≈0.9)    음의 상관 (r≈-0.8)    무상관 (r≈0)

    ●●●                ●●●               ●  ● ●  ●
   ●●●                  ●●●              ● ●● ●●
  ●●●                    ●●●            ●●  ●● ●
 ●●●                      ●●●           ● ●  ●●●
●●●                        ●●●          ●  ● ● ●
```

> 점들이 직선에 가깝게 모일수록 상관관계가 강함

---

# 상관계수 (Correlation Coefficient)

## 피어슨 상관계수 (r)

### 정의
- 두 변수의 **선형적 관계**의 강도와 방향
- 범위: **-1 ~ +1**

### 해석
| r 값 | 의미 |
|------|------|
| **r = +1** | 완벽한 양의 상관 |
| **r = -1** | 완벽한 음의 상관 |
| **r = 0** | 선형 관계 없음 |

---

# 상관계수 해석 기준

## |r| 값에 따른 해석

| |r| 범위 | 해석 | 실무적 의미 |
|----------|------|----------|
| 0.0 ~ 0.3 | 약한 상관 | 관계 미미 |
| 0.3 ~ 0.5 | 보통 상관 | 어느 정도 관계 |
| 0.5 ~ 0.7 | 중간 상관 | 명확한 관계 |
| 0.7 ~ 1.0 | 강한 상관 | 밀접한 관계 |

### 예시
- 온도 - 불량률: r = 0.85 → **강한 양의 상관**
- 교육시간 - 불량률: r = -0.65 → **중간 음의 상관**

---

# 상관계수 계산 공식

## 피어슨 상관계수

```
        Σ(Xᵢ - X̄)(Yᵢ - Ȳ)
r = ─────────────────────────────
    √[Σ(Xᵢ - X̄)²] × √[Σ(Yᵢ - Ȳ)²]

= 공분산(X, Y) / (표준편차X × 표준편차Y)
```

### Python 구현

```python
import numpy as np

# 방법 1: numpy
r = np.corrcoef(X, Y)[0, 1]

# 방법 2: pandas
r = df['X'].corr(df['Y'])
```

---

# 상관계수 계산 예시

## 온도와 불량률 데이터

```python
import numpy as np

# 데이터
temperature = np.array([75, 78, 80, 82, 85, 87, 90, 92, 95, 98])
defect_rate = np.array([5.1, 5.3, 5.5, 5.7, 6.0, 6.1, 6.4, 6.6, 6.9, 7.2])

# 상관계수 계산
r = np.corrcoef(temperature, defect_rate)[0, 1]
print(f"상관계수: {r:.4f}")  # 0.9987
```

> 0.99에 가까움 → **매우 강한 양의 상관**

---

# 상관행렬 (Correlation Matrix)

## 여러 변수 간 상관관계를 한눈에

```python
import pandas as pd

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '불량률': defect_rate,
    '생산량': production
})

# 상관행렬
corr_matrix = df.corr()
print(corr_matrix.round(2))
```

### 결과 예시
```
         온도    습도  불량률  생산량
온도     1.00   0.15   0.85  -0.75
습도     0.15   1.00   0.10  -0.05
불량률   0.85   0.10   1.00  -0.60
생산량  -0.75  -0.05  -0.60   1.00
```

---

# 상관행렬 히트맵

## 시각화로 패턴 파악

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='상관계수')

# 레이블
labels = corr_matrix.columns
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center')

plt.title('변수 간 상관관계')
plt.tight_layout()
plt.show()
```

---

# 주의: 상관관계 ≠ 인과관계

## Correlation does not imply causation

### 유명한 예시
```
아이스크림 판매량 ↔ 익사 사고 수

- 상관계수: 높음 (r ≈ 0.7)
- 인과관계: 없음!
- 숨겨진 변수: 기온 (더운 날 둘 다 증가)
```

### 제조업 예시
```
설비 가동시간 ↔ 불량률

- 상관관계: 있을 수 있음
- 인과관계: 추가 분석 필요
- 숨겨진 변수: 작업자 피로, 원자재 상태 등
```

---

# 허위 상관의 함정

## 우연히 높은 상관

| 변수 A | 변수 B | r | 의미 |
|--------|--------|---|------|
| 니콜라스 케이지 영화 수 | 수영장 익사자 수 | 0.67 | 우연 |
| 치즈 소비량 | 침대 목에 얽힘 사망자 | 0.95 | 우연 |
| 해적 수 감소 | 지구 온도 상승 | -0.99 | 우연 |

### 교훈
> 상관관계만으로 의사결정하지 말 것
> 반드시 **도메인 지식**과 함께 해석

---

# Part 1 정리

## 상관계수의 의미와 해석

### 핵심 개념
- **상관계수(r)**: -1 ~ +1, 선형 관계의 강도와 방향
- **|r| > 0.7**: 강한 상관관계
- **상관행렬**: 여러 변수 관계를 한눈에

### 주의사항
- **상관관계 ≠ 인과관계**
- 숨겨진 변수(confounding) 주의
- 도메인 지식과 함께 해석

---

<!-- _class: lead -->

# Part 2
## 단순선형회귀의 개념과 원리

---

# 회귀분석이란?

## 변수 간 관계를 수학적 함수로 표현

### 용어 정리
| 용어 | 의미 | 예시 |
|------|------|------|
| **독립변수 (X)** | 원인, 예측에 사용 | 온도, 습도 |
| **종속변수 (Y)** | 결과, 예측 대상 | 불량률, 생산량 |
| **회귀**: | 평균으로 "되돌아감" | 통계 용어 |

### 목적
1. 변수 간 **관계 파악** (X가 Y에 미치는 영향)
2. **예측** (새로운 X로 Y 예측)

---

# 단순선형회귀 모델

## 직선으로 관계 표현

### 수식

```
Y = β₀ + β₁X + ε

Y  : 종속변수 (예측 대상)
X  : 독립변수 (입력)
β₀ : 절편 (intercept) - X=0일 때 Y값
β₁ : 기울기 (slope) - X가 1 증가할 때 Y 변화량
ε  : 오차항 (error)
```

### 예시
```
불량률 = 1.5 + 0.05 × 온도

→ 온도가 1도 올라가면 불량률이 0.05%p 증가
```

---

# 회귀선의 의미

## 시각적 이해

```
Y (불량률)
↑
│           ●
│         ● ╱
│       ● ╱
│     ● ╱ ← 회귀선 (Y = β₀ + β₁X)
│   ● ╱
│ ● ╱
│●╱
└────────────────→ X (온도)
```

- 회귀선: 데이터를 가장 잘 설명하는 직선
- 각 점과 선 사이의 거리(오차)의 합이 최소

---

# 최소제곱법 (OLS)

## Ordinary Least Squares

### 목표
오차의 제곱합을 최소화하는 β₀, β₁ 찾기

```
최소화: Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (β₀ + β₁xᵢ))²

yᵢ  : 실제값
ŷᵢ  : 예측값
```

### 왜 제곱?
- 오차의 부호를 없앰 (+,- 상쇄 방지)
- 큰 오차에 더 큰 페널티

---

# 최소제곱법 공식

## β₀, β₁ 계산

```
        Σ(xᵢ - x̄)(yᵢ - ȳ)
β₁ = ─────────────────────
         Σ(xᵢ - x̄)²

β₀ = ȳ - β₁x̄
```

### Python 구현

```python
# 직접 계산
x_mean, y_mean = X.mean(), Y.mean()
beta1 = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean)**2)
beta0 = y_mean - beta1 * x_mean

# sklearn 사용 (자동 계산)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X.reshape(-1,1), Y)
```

---

# 잔차 (Residual)

## 실제값과 예측값의 차이

```
잔차 = 실제값 - 예측값 = yᵢ - ŷᵢ
```

### 잔차의 특성 (좋은 모델)
- 평균이 0에 가까움
- 랜덤하게 분포 (패턴 없음)
- 정규분포를 따름

### 잔차 분석
```python
residuals = y - model.predict(X)
print(f"잔차 평균: {residuals.mean():.4f}")  # 0에 가까워야
print(f"잔차 표준편차: {residuals.std():.4f}")
```

---

# R² (결정계수)

## 모델의 설명력

### 정의

```
        SSR         잔차제곱합
R² = 1 - ─── = 1 - ───────────
        SST         총제곱합

SSR = Σ(yᵢ - ŷᵢ)²  (잔차제곱합)
SST = Σ(yᵢ - ȳ)²   (총제곱합)
```

### 해석
- 범위: 0 ~ 1
- **의미**: 독립변수가 종속변수 변동의 몇 %를 설명하는가
- **R² = 0.85**: 온도가 불량률 변동의 85%를 설명

---

# R² 해석 기준

## 모델 성능 판단

| R² 범위 | 해석 | 조치 |
|---------|------|------|
| 0.9 이상 | 매우 좋음 | 그대로 사용 |
| 0.7 ~ 0.9 | 좋음 | 실무 활용 가능 |
| 0.5 ~ 0.7 | 보통 | 추가 변수 검토 |
| 0.3 ~ 0.5 | 낮음 | 모델 개선 필요 |
| 0.3 미만 | 매우 낮음 | 다른 접근 필요 |

### 주의
> R²가 높다고 항상 좋은 모델은 아님
> 과적합(overfitting) 가능성 확인 필요

---

# 회귀 모델의 가정

## 선형회귀가 유효하려면

| 가정 | 의미 | 확인 방법 |
|------|------|----------|
| **선형성** | X와 Y의 관계가 직선 | 산점도 확인 |
| **독립성** | 오차 간 독립 | Durbin-Watson |
| **등분산성** | 오차의 분산 일정 | 잔차 플롯 |
| **정규성** | 오차가 정규분포 | Q-Q 플롯 |

### 실무 팁
- 데이터가 30개 이상이면 정규성 완화
- 산점도와 잔차 플롯은 필수 확인

---

# Part 2 정리

## 단순선형회귀의 개념과 원리

### 핵심 공식
```
Y = β₀ + β₁X

β₀ : 절편 (X=0일 때 Y)
β₁ : 기울기 (X가 1 증가할 때 Y 변화)
```

### 핵심 지표
- **R²**: 모델의 설명력 (0~1, 높을수록 좋음)
- **잔차**: 실제값 - 예측값 (0에 가까울수록 좋음)

---

<!-- _class: lead -->

# Part 3
## sklearn으로 예측 모델 구현

---

# scikit-learn 소개

## Python 머신러닝 라이브러리

### 특징
- 일관된 API (fit, predict, score)
- 다양한 알고리즘 지원
- 전처리, 평가 도구 포함

### 설치

```bash
pip install scikit-learn
```

### 기본 구조

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # 모델 생성
model.fit(X, y)             # 학습
predictions = model.predict(X_new)  # 예측
```

---

# sklearn 선형회귀 구현

## 전체 흐름

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터 준비
X = temperature.reshape(-1, 1)  # 2D 배열로 변환 필수!
y = defect_rate

# 2. 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 3. 예측
y_pred = model.predict(X)

# 4. 평가
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
```

---

# 데이터 형태 주의

## sklearn은 2D 배열 필요

### 잘못된 방법
```python
X = temperature  # 1D 배열 (10,)
model.fit(X, y)  # 에러!
```

### 올바른 방법
```python
# 방법 1: reshape
X = temperature.reshape(-1, 1)  # (10, 1)

# 방법 2: 슬라이싱 (DataFrame)
X = df[['온도']]  # 2D DataFrame

# 방법 3: np.newaxis
X = temperature[:, np.newaxis]
```

---

# 모델 속성 확인

## 학습 결과 확인

```python
# 절편과 기울기
print(f"절편 (β₀): {model.intercept_:.4f}")
print(f"기울기 (β₁): {model.coef_[0]:.4f}")

# 회귀식 출력
print(f"\n회귀식: Y = {model.intercept_:.2f} + {model.coef_[0]:.4f} × X")
```

### 결과 예시
```
절편 (β₀): 1.5234
기울기 (β₁): 0.0512

회귀식: Y = 1.52 + 0.0512 × X
```

---

# 예측 수행

## 새로운 데이터로 예측

```python
# 새로운 온도 데이터
new_temps = np.array([[85], [90], [95], [100]])

# 예측
predictions = model.predict(new_temps)

# 결과 출력
print("=== 불량률 예측 ===")
for temp, pred in zip(new_temps.flatten(), predictions):
    print(f"온도 {temp}도 → 예측 불량률: {pred:.2f}%")
```

### 결과
```
온도 85도 → 예측 불량률: 5.87%
온도 90도 → 예측 불량률: 6.13%
온도 95도 → 예측 불량률: 6.39%
온도 100도 → 예측 불량률: 6.65%
```

---

# 모델 평가 지표

## 회귀 모델 성능 측정

| 지표 | 수식 | 해석 |
|------|------|------|
| **R²** | 1 - SSR/SST | 설명력 (0~1) |
| **MSE** | Σ(y-ŷ)²/n | 평균제곱오차 |
| **RMSE** | √MSE | Y와 같은 단위 |
| **MAE** | Σ\|y-ŷ\|/n | 평균절대오차 |

### 코드
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
```

---

# 회귀선 시각화

## 데이터와 모델 함께 표시

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# 실제 데이터
plt.scatter(X, y, s=100, alpha=0.7, label='실제 데이터')

# 회귀선
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='회귀선')

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title(f'온도와 불량률: 선형회귀 (R²={r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# 잔차 분석

## 모델 진단

```python
# 잔차 계산
residuals = y - model.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 잔차 플롯
axes[0].scatter(y_pred, residuals, alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('예측값')
axes[0].set_ylabel('잔차')
axes[0].set_title('잔차 플롯')

# 잔차 히스토그램
axes[1].hist(residuals, bins=10, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('잔차')
axes[1].set_ylabel('빈도')
axes[1].set_title('잔차 분포')

plt.tight_layout()
plt.show()
```

---

# 역 예측 활용

## 목표 Y를 위한 X 계산

```python
# 회귀식: Y = β₀ + β₁X
# 역산: X = (Y - β₀) / β₁

# 목표 불량률을 달성하기 위한 온도
target_defect = 5.0  # 목표: 5% 이하

required_temp = (target_defect - model.intercept_) / model.coef_[0]

print(f"불량률 {target_defect}% 이하 유지를 위한 최대 온도: {required_temp:.1f}도")
```

### 실무 활용
- 품질 기준 달성을 위한 공정 조건 도출
- 목표 생산량을 위한 투입량 계산

---

# 다중선형회귀 미리보기

## 여러 독립변수 사용

```python
# 단순선형회귀: Y = β₀ + β₁X
# 다중선형회귀: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ

# 예시: 불량률 = f(온도, 습도, 속도)
X_multi = df[['온도', '습도', '속도']].values
y = df['불량률'].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

print("각 변수의 영향력 (기울기):")
for name, coef in zip(['온도', '습도', '속도'], model_multi.coef_):
    print(f"  {name}: {coef:.4f}")
```

> 다중회귀는 이후 차시에서 상세히 다룸

---

# Part 3 정리

## sklearn으로 예측 모델 구현

### 핵심 코드

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X.reshape(-1, 1), y)  # 학습
y_pred = model.predict(X_new)   # 예측

# 결과 확인
print(f"절편: {model.intercept_}")
print(f"기울기: {model.coef_[0]}")
print(f"R²: {r2_score(y, y_pred)}")
```

### 주의사항
- X는 반드시 2D 배열로 변환
- R²와 잔차 분석 필수

---

<!-- _class: lead -->

# 실습편

## 온도와 불량률 관계 분석 실습

---

# 실습 개요

## 상관분석과 선형회귀

### 실습 목표
1. 상관계수 계산 및 해석
2. 선형회귀 모델 학습
3. 새로운 데이터로 예측

### 실습 환경

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 준비

```python
np.random.seed(42)

# 온도 데이터 (섭씨)
temperature = np.array([75, 78, 80, 82, 85, 87, 90, 92, 95, 98])

# 불량률 (%, 온도와 양의 상관관계 + 노이즈)
defect_rate = 1.5 + 0.05 * temperature + np.random.normal(0, 0.2, 10)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '불량률': defect_rate.round(2)
})

print(df)
print(f"\n온도 범위: {df['온도'].min()} ~ {df['온도'].max()}도")
print(f"불량률 범위: {df['불량률'].min():.2f} ~ {df['불량률'].max():.2f}%")
```

---

# 실습 2: 산점도 시각화

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7,
            color='steelblue', edgecolor='black')

plt.xlabel('온도 (도)', fontsize=12)
plt.ylabel('불량률 (%)', fontsize=12)
plt.title('온도와 불량률의 관계', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

> 점들이 오른쪽 위로 향하면 **양의 상관관계**

---

# 실습 3: 상관계수 계산

```python
# numpy로 계산
r_numpy = np.corrcoef(df['온도'], df['불량률'])[0, 1]

# pandas로 계산
r_pandas = df['온도'].corr(df['불량률'])

print(f"상관계수 (numpy): {r_numpy:.4f}")
print(f"상관계수 (pandas): {r_pandas:.4f}")

# 해석
if abs(r_numpy) > 0.7:
    strength = "강한"
elif abs(r_numpy) > 0.3:
    strength = "중간"
else:
    strength = "약한"
direction = "양의" if r_numpy > 0 else "음의"

print(f"\n해석: {strength} {direction} 상관관계 (|r|={abs(r_numpy):.2f})")
```

---

# 실습 4: 선형회귀 모델 학습

```python
from sklearn.linear_model import LinearRegression

# 데이터 준비 (2D 배열로 변환)
X = df[['온도']].values  # 독립변수 (2D)
y = df['불량률'].values   # 종속변수 (1D)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 결과 확인
print("=== 회귀 분석 결과 ===")
print(f"절편 (β₀): {model.intercept_:.4f}")
print(f"기울기 (β₁): {model.coef_[0]:.4f}")
print(f"\n회귀식: 불량률 = {model.intercept_:.2f} + {model.coef_[0]:.4f} × 온도")
print(f"\n해석: 온도가 1도 상승하면 불량률이 {model.coef_[0]:.4f}%p 증가")
```

---

# 실습 5: 회귀선 시각화

```python
plt.figure(figsize=(10, 6))

# 원본 데이터
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7,
            label='실제 데이터', color='steelblue', edgecolor='black')

# 회귀선
X_line = np.linspace(70, 100, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='회귀선')

# 회귀식 표시
eq_text = f'Y = {model.intercept_:.2f} + {model.coef_[0]:.4f}X'
plt.text(72, df['불량률'].max()-0.1, eq_text, fontsize=12)

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률: 선형회귀')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# 실습 6: 모델 평가

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 예측값
y_pred = model.predict(X)

# 평가 지표
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print("=== 모델 평가 ===")
print(f"R² (결정계수): {r2:.4f}")
print(f"MSE (평균제곱오차): {mse:.4f}")
print(f"RMSE (평균제곱근오차): {rmse:.4f}")
print(f"MAE (평균절대오차): {mae:.4f}")
print()
print(f"해석: 온도가 불량률 변동의 {r2*100:.1f}%를 설명합니다.")
```

---

# 실습 7: 새로운 데이터 예측

```python
# 새로운 온도에서 불량률 예측
new_temps = np.array([[80], [85], [90], [95], [100], [105]])
predictions = model.predict(new_temps)

print("=== 불량률 예측 ===")
for temp, pred in zip(new_temps.flatten(), predictions):
    print(f"온도 {temp:3d}도 → 예측 불량률: {pred:.2f}%")

# 목표 불량률을 위한 온도 역산
target_defect = 5.5
required_temp = (target_defect - model.intercept_) / model.coef_[0]
print(f"\n불량률 {target_defect}% 이하 유지를 위한 최대 온도: {required_temp:.1f}도")
```

---

# 실습 8: 잔차 분석

```python
# 잔차 계산
residuals = y - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 잔차 플롯
axes[0].scatter(y_pred, residuals, alpha=0.7, s=80)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('예측값')
axes[0].set_ylabel('잔차')
axes[0].set_title('잔차 플롯')

# 잔차 히스토그램
axes[1].hist(residuals, bins=7, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('잔차')
axes[1].set_ylabel('빈도')
axes[1].set_title('잔차 분포')

# 실제 vs 예측
axes[2].scatter(y, y_pred, alpha=0.7, s=80)
axes[2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[2].set_xlabel('실제값')
axes[2].set_ylabel('예측값')
axes[2].set_title('실제 vs 예측')

plt.tight_layout()
plt.show()

print(f"잔차 평균: {residuals.mean():.6f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.4f}")
```

---

# 실습 정리

## 핵심 체크포인트

### 상관분석
- [ ] np.corrcoef() 또는 df.corr()로 계산
- [ ] |r| > 0.7이면 강한 상관
- [ ] 상관관계 ≠ 인과관계

### 선형회귀
- [ ] X는 2D 배열로 변환 필수
- [ ] LinearRegression().fit(X, y)로 학습
- [ ] model.predict()로 예측
- [ ] R²로 모델 평가 (1에 가까울수록 좋음)

---

# 다음 차시 예고

## 8차시: 제조 데이터 전처리 (1)

### 학습 내용
- 결측치 탐지 및 처리
- 이상치 처리 방법
- fillna, dropna 활용

### 준비물
- 오늘 배운 코드 복습
- sklearn, pandas 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심

1. **상관계수**: 두 변수의 관계 강도 (-1 ~ +1)
2. **선형회귀**: Y = β₀ + β₁X로 관계 표현
3. **R²**: 모델의 설명력 (높을수록 좋음)

### 자주 하는 실수
- X를 1차원 배열로 넣으면 에러 → 2D 배열로 변환
- 상관관계를 인과관계로 오해
- R²만 보고 모델 평가 (잔차도 확인 필요)

---

# 감사합니다

## 8차시: 상관분석과 예측의 기초

**다음 시간에 데이터 전처리를 배워봅시다!**
