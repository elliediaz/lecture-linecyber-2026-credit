---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 6차시'
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

# 상관분석과 예측의 기초

## 6차시 | Part II. 기초 수리와 데이터 분석

**두 변수의 관계를 분석하고 예측하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **상관계수**의 의미와 해석 방법을 이해한다
2. **단순선형회귀**의 개념과 원리를 이해한다
3. sklearn으로 **예측 모델**을 구현한다

---

# 상관분석이 필요한 이유

## 제조 현장의 질문들

- "온도가 높아지면 불량률도 높아지는가?"
- "가동 시간과 생산량은 관련이 있는가?"
- "습도가 품질에 영향을 주는가?"

> 상관분석 = 두 변수가 **함께 변하는 정도**를 수치화

---

# 상관계수 (Correlation Coefficient)

## 피어슨 상관계수 (r)

- 범위: **-1 ~ +1**
- **+1에 가까움**: 강한 양의 상관 (함께 증가)
- **-1에 가까움**: 강한 음의 상관 (반대로 변화)
- **0에 가까움**: 상관 없음

| |r| 범위 | 해석 |
|----------|------|
| 0.0 ~ 0.3 | 약한 상관 |
| 0.3 ~ 0.7 | 중간 상관 |
| 0.7 ~ 1.0 | 강한 상관 |

---

# 상관계수 시각화

## 산점도와 상관계수의 관계

```
r = +0.9          r = -0.8          r = 0.1
(강한 양의)        (강한 음의)        (거의 없음)

    ●●              ●●               ● ●  ●
   ●●●              ●●●               ●●●●
  ●●●               ●●●              ● ●● ●
 ●●●                 ●●●             ●● ●●
●●                    ●●●            ●  ● ●
```

> 점들이 직선에 가깝게 모일수록 상관계수의 절대값이 큼

---

# 주의: 상관관계 ≠ 인과관계

## 상관이 있다고 원인-결과는 아님

```
예: 아이스크림 판매량 ↔ 익사 사고 수

- 상관계수 높음
- 하지만 인과관계 없음
- 숨겨진 변수: 기온 (더운 날 둘 다 증가)
```

> **상관분석은 관계의 존재만 알려줌**
> 원인은 별도 분석 필요

---

# 선형회귀란?

## 변수 간 관계를 직선으로 표현

### 목적
1. 변수 간 **관계 파악**: X가 Y에 미치는 영향
2. **예측**: X 값으로 Y 값 예측

### 단순선형회귀 공식
```
Y = β₀ + β₁X

Y: 종속변수 (예측 대상, 예: 불량률)
X: 독립변수 (예측에 사용, 예: 온도)
β₀: 절편 (X=0일 때 Y값)
β₁: 기울기 (X가 1 증가할 때 Y 변화량)
```

---

# 최소제곱법 (OLS)

## 최적의 직선 찾기

```
목표: 오차의 제곱합을 최소화하는 β₀, β₁ 찾기

Σ(실제값 - 예측값)² → 최소화

예: 온도 85도에서 실제 불량률 2.5%, 예측 2.3%
    → 오차 = 0.2, 오차² = 0.04
    → 모든 데이터의 오차² 합을 최소화
```

> sklearn이 자동으로 최적의 β₀, β₁을 찾아줌

---

# R² (결정계수)

## 모델의 설명력

```
R² = 1 - (잔차제곱합 / 총제곱합)

- 범위: 0 ~ 1
- 1에 가까울수록 설명력 높음
- 의미: 독립변수가 종속변수 변동의 몇 %를 설명하는가
```

| R² 값 | 해석 |
|-------|------|
| 0.9 이상 | 매우 좋음 |
| 0.7 ~ 0.9 | 좋음 |
| 0.5 ~ 0.7 | 보통 |
| 0.5 미만 | 낮음 |

---

# 이론 정리

## 핵심 포인트

### 상관분석
- **상관계수**: -1 ~ +1 범위
- **|r| > 0.7**: 강한 상관
- 상관관계 ≠ 인과관계

### 선형회귀
- **Y = β₀ + β₁X**: 직선으로 관계 표현
- **R²**: 모델의 설명력 (1에 가까울수록 좋음)
- **예측**: 새로운 X값으로 Y값 예측

---

# - 실습편 -

## 6차시

**온도와 불량률 관계 분석 실습**

---

# 실습 개요

## 상관분석과 선형회귀

### 실습 목표
- 상관계수 계산 및 해석
- 선형회귀 모델 학습
- 새로운 데이터로 예측

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 준비

## 온도-불량률 데이터

```python
np.random.seed(42)

# 온도 데이터 (섭씨)
temperature = np.array([75, 78, 80, 82, 85, 87, 90, 92, 95, 98])

# 불량률 (%, 온도와 양의 상관관계)
defect_rate = 1.5 + 0.05 * temperature + np.random.normal(0, 0.2, 10)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '불량률': defect_rate
})

print(df)
```

---

# 실습 2: 산점도 시각화

## 두 변수의 관계 확인

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7)

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률의 관계')
plt.grid(True, alpha=0.3)
plt.show()
```

> 점들이 오른쪽 위로 향하면 **양의 상관관계**

---

# 실습 3: 상관계수 계산

## numpy와 pandas 활용

```python
# numpy로 계산
r = np.corrcoef(df['온도'], df['불량률'])[0, 1]
print(f"상관계수 (numpy): {r:.4f}")

# pandas로 계산
r_pandas = df['온도'].corr(df['불량률'])
print(f"상관계수 (pandas): {r_pandas:.4f}")

# 해석
if abs(r) > 0.7:
    strength = "강한"
elif abs(r) > 0.3:
    strength = "중간"
else:
    strength = "약한"
direction = "양의" if r > 0 else "음의"
print(f"해석: {strength} {direction} 상관관계")
```

---

# 실습 4: 상관행렬

## 여러 변수 간 상관관계

```python
# 추가 변수 생성
df['습도'] = 60 + np.random.normal(0, 5, 10)
df['생산량'] = 1200 - 5 * df['온도'] + np.random.normal(0, 20, 10)

# 상관행렬
corr_matrix = df.corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))
```

---

# 실습 5: 상관행렬 히트맵

## 시각화로 관계 파악

```python
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto',
           vmin=-1, vmax=1)
plt.colorbar(label='상관계수')

# 레이블 추가
labels = corr_matrix.columns
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=12)

plt.title('변수 간 상관관계 히트맵')
plt.tight_layout()
plt.show()
```

---

# 실습 6: 선형회귀 모델 학습

## sklearn으로 구현

```python
from sklearn.linear_model import LinearRegression

# 데이터 준비 (2D 배열로 변환)
X = df[['온도']].values  # 독립변수
y = df['불량률'].values   # 종속변수

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 결과 확인
print("=== 회귀 분석 결과 ===")
print(f"절편 (β₀): {model.intercept_:.4f}")
print(f"기울기 (β₁): {model.coef_[0]:.4f}")
print(f"\n회귀식: 불량률 = {model.intercept_:.2f} + {model.coef_[0]:.4f} × 온도")
```

---

# 실습 7: 회귀선 시각화

## 데이터와 예측선

```python
plt.figure(figsize=(10, 6))

# 원본 데이터
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7, label='실제 데이터')

# 회귀선
X_line = np.linspace(70, 100, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='회귀선')

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률: 선형회귀')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# 실습 8: 모델 평가

## R² 계산

```python
from sklearn.metrics import r2_score, mean_squared_error

# 예측값
y_pred = model.predict(X)

# R² (결정계수)
r2 = r2_score(y, y_pred)

# RMSE (평균제곱근오차)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("=== 모델 평가 ===")
print(f"R² (결정계수): {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print()
print(f"해석: 온도가 불량률 변동의 {r2*100:.1f}%를 설명합니다.")
```

---

# 실습 9: 새로운 데이터 예측

## 실무 활용

```python
# 새로운 온도에서 불량률 예측
new_temps = np.array([[85], [90], [95], [100]])
predictions = model.predict(new_temps)

print("=== 불량률 예측 ===")
for temp, pred in zip(new_temps.flatten(), predictions):
    print(f"온도 {temp}도 → 예측 불량률: {pred:.2f}%")

# 목표 불량률을 위한 온도 역산
target_defect = 5.0
required_temp = (target_defect - model.intercept_) / model.coef_[0]
print(f"\n불량률 {target_defect}% 이하 유지를 위한 최대 온도: {required_temp:.1f}도")
```

---

# 실습 10: 잔차 분석

## 모델 가정 확인

```python
# 잔차 계산
residuals = y - y_pred

plt.figure(figsize=(10, 4))

# 잔차 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')

# 잔차 히스토그램
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=5, edgecolor='black', alpha=0.7)
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 분포')

plt.tight_layout()
plt.show()

print(f"잔차 평균: {residuals.mean():.4f} (0에 가까워야 함)")
```

---

# 실습 정리

## 핵심 체크포인트

### 상관분석
- [ ] np.corrcoef() 또는 df.corr()로 계산
- [ ] |r| > 0.7이면 강한 상관
- [ ] 상관관계 ≠ 인과관계

### 선형회귀
- [ ] LinearRegression().fit(X, y)로 학습
- [ ] model.predict()로 예측
- [ ] R²로 모델 평가 (1에 가까울수록 좋음)

---

# 다음 차시 예고

## 7차시: 제조 데이터 전처리 (1)

### 학습 내용
- 결측치 탐지 및 처리
- 이상치 처리 방법
- 데이터 정규화와 표준화

### 준비물
- 오늘 배운 코드 복습
- sklearn 설치 확인

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

## 6차시: 상관분석과 예측의 기초

**다음 시간에 데이터 전처리를 배워봅시다!**
