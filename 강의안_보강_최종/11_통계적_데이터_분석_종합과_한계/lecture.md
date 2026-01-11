# [11차시] 통계적 데이터 분석 종합과 한계

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. 복잡한 제조 데이터에 **탐색적 분석(EDA) 기법을 종합 적용**함
2. 단순회귀와 중다회귀의 차이를 이해하고 **다중공선성을 진단**함
3. 선형 모델의 한계를 인식하고 **머신러닝 도입 필요성을 설명**함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **MPG** | seaborn-data | 자동차 연비 예측 및 중다회귀 실습 |
| **California Housing** | sklearn | 복잡한 데이터의 선형 모델 한계 확인 |

MPG 데이터셋 변수:
- mpg: 연비 (miles per gallon) - 예측 대상
- weight: 자동차 무게
- horsepower: 마력
- displacement: 배기량
- cylinders: 실린더 수
- acceleration: 가속력

---

## 강의 구성

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| 1 | 6-10차시 기법 종합 적용 | 8분 |
| 2 | 단순회귀에서 중다회귀로: 설명력 향상과 다중공선성 | 10분 |
| 3 | 선형 모델의 한계와 머신러닝으로의 전환 | 7분 |

---

## 파트 1: 6-10차시 기법 종합 적용

### 개념 설명

#### 지금까지 배운 EDA 기법 요약

6-10차시에서 학습한 핵심 기법:

| 차시 | 주제 | 핵심 기법 |
|:----:|------|----------|
| 6 | 확률분포와 품질검정 | 정규분포, Z-score, 정규성 검정 |
| 7 | 통계검정 실습 | t-검정, 가설검정, p-value |
| 8 | 상관분석과 예측 기초 | 상관계수(r), 단순선형회귀, R^2 |
| 9 | 제조 데이터 전처리 (1) | 결측치 처리, 이상치 탐지(IQR) |
| 10 | 제조 데이터 전처리 (2) | 스케일링, 원-핫 인코딩 |

#### EDA 종합 워크플로우

```
1단계: 데이터 로드 및 개요 파악
       shape, dtypes, info(), head()
              |
              v
2단계: 기술통계량 확인
       describe(), value_counts()
              |
              v
3단계: 결측치/이상치 처리
       isnull(), IQR 방법
              |
              v
4단계: 분포 시각화
       히스토그램, 박스플롯, KDE
              |
              v
5단계: 상관관계 분석
       상관행렬, 히트맵, 산점도
              |
              v
6단계: 그룹별 비교 및 검정
       t-검정, ANOVA
              |
              v
7단계: 인사이트 도출
       발견 정리, 권고사항
```

#### 기술통계량 복습

**중심 경향 측도**:
- 평균(mean): 데이터의 산술 평균
- 중앙값(median): 정렬 시 중앙에 위치한 값
- 최빈값(mode): 가장 빈번한 값

**산포도 측도**:
- 표준편차(std): 평균으로부터의 평균적 거리
- 분산(variance): 표준편차의 제곱
- IQR: Q3 - Q1, 이상치 탐지에 활용

#### 상관계수 공식 복습

피어슨 상관계수:

```
        sum((Xi - X_mean)(Yi - Y_mean))
r = -------------------------------------------
    sqrt(sum((Xi - X_mean)^2)) * sqrt(sum((Yi - Y_mean)^2))
```

해석 기준:
- |r| > 0.7: 강한 상관
- 0.5 <= |r| <= 0.7: 중간 상관
- 0.3 <= |r| < 0.5: 보통 상관
- |r| < 0.3: 약한 상관

### 실습 코드

#### 라이브러리 및 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# MPG 데이터셋 로드
mpg_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df = pd.read_csv(mpg_url)

print("=== MPG 데이터셋 로드 완료 ===")
print(f"데이터 크기: {df.shape} (행, 열)")
print(f"\n컬럼 목록:\n{df.columns.tolist()}")
```

---

#### 1단계: 데이터 개요 파악

```python
# 데이터 타입 확인
print("=== 데이터 타입 ===")
print(df.dtypes)

# 처음 10행 확인
print("\n=== 처음 10행 ===")
print(df.head(10))

# 기술 통계량
print("\n=== 수치형 변수 기술 통계 ===")
print(df.describe().round(3))

# 결측치 확인
print("\n=== 결측치 현황 ===")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(pd.DataFrame({'결측치 수': missing, '비율(%)': missing_pct}))
```

---

#### 2단계: 결측치 및 이상치 처리

```python
# 수치형 변수만 선택하고 결측치 제거
numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
                'weight', 'acceleration', 'model_year']
df_clean = df[numeric_cols].dropna().copy()

print(f"결측치 제거 후 데이터 크기: {df_clean.shape}")

# 이상치 확인 (IQR 방법)
print("\n=== 이상치 현황 (IQR 방법) ===")
outlier_summary = []
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_pct = outliers / len(df_clean) * 100
    outlier_summary.append({
        'Variable': col,
        'Outliers': outliers,
        'Percentage': f'{outlier_pct:.1f}%',
        'Lower': round(lower, 2),
        'Upper': round(upper, 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)
```

---

#### 3단계: 분포 시각화

```python
# 주요 변수 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 히스토그램 + 박스플롯 조합
plot_vars = ['mpg', 'weight', 'horsepower', 'displacement', 'acceleration', 'cylinders']

for idx, col in enumerate(plot_vars):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]

    # 히스토그램
    ax.hist(df_clean[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(df_clean[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(df_clean[col].median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{col} Distribution')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('distribution_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("분포 시각화 저장: distribution_overview.png")
```

---

#### 4단계: 상관관계 분석

```python
# 상관행렬 계산
corr_matrix = df_clean.corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))

# mpg와의 상관계수 정렬
print("\n=== mpg와의 상관계수 (절대값 기준 내림차순) ===")
mpg_corr = corr_matrix['mpg'].drop('mpg').abs().sort_values(ascending=False)
for var in mpg_corr.index:
    r = corr_matrix.loc['mpg', var]
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.5 else "weak"
    direction = "positive" if r > 0 else "negative"
    print(f"  {var:15}: r = {r:+.4f} ({strength} {direction})")
```

---

#### 상관행렬 히트맵

```python
# 히트맵 시각화
fig, ax = plt.subplots(figsize=(10, 8))
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
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9, color=color)

ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("상관행렬 히트맵 저장: correlation_heatmap.png")
```

---

#### 5단계: 그룹별 비교 (cylinders 기준)

```python
# 실린더 수 기준 그룹별 mpg 비교
print("=== 실린더 수별 mpg 통계 ===")
cyl_stats = df_clean.groupby('cylinders')['mpg'].agg(['mean', 'std', 'count'])
print(cyl_stats.round(3))

# 4기통 vs 8기통 t-검정
cyl_4 = df_clean[df_clean['cylinders'] == 4]['mpg']
cyl_8 = df_clean[df_clean['cylinders'] == 8]['mpg']

t_stat, p_value = stats.ttest_ind(cyl_4, cyl_8)
print(f"\n=== 4기통 vs 8기통 독립표본 t-검정 ===")
print(f"4기통 평균 mpg: {cyl_4.mean():.2f}")
print(f"8기통 평균 mpg: {cyl_8.mean():.2f}")
print(f"차이: {cyl_4.mean() - cyl_8.mean():.2f}")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("결론: p < 0.05이므로 두 그룹 간 mpg 차이가 통계적으로 유의함")
else:
    print("결론: p >= 0.05이므로 두 그룹 간 유의미한 차이 없음")
```

---

### 결과 해설

- MPG 데이터셋은 398개 샘플과 9개 변수로 구성됨
- horsepower에 결측치가 존재하여 제거 후 분석 진행함
- weight, horsepower, displacement는 mpg와 강한 음의 상관관계를 보임
- 4기통 차량이 8기통 차량보다 연비가 유의미하게 높음 (p < 0.001)

---

## 파트 2: 단순회귀에서 중다회귀로

### 개념 설명

#### 단순선형회귀 vs 중다선형회귀

**단순선형회귀 (Simple Linear Regression)**:
```
Y = B0 + B1*X + e

- 독립변수 1개
- 예: weight만으로 mpg 예측
```

**중다선형회귀 (Multiple Linear Regression)**:
```
Y = B0 + B1*X1 + B2*X2 + ... + Bp*Xp + e

- 독립변수 p개
- 예: weight, horsepower, displacement로 mpg 예측
```

#### 결정계수 R^2

모델이 종속변수의 변동을 얼마나 설명하는지 나타내는 지표:

```
R^2 = 1 - (SS_res / SS_tot)
    = 1 - (sum((yi - yi_hat)^2) / sum((yi - y_mean)^2))

범위: 0 ~ 1
해석: R^2 = 0.75 -> 모델이 Y 변동의 75%를 설명
```

#### 조정 결정계수 (Adjusted R^2)

변수 개수를 고려한 설명력 지표:

```
Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)

n: 표본 크기
p: 독립변수 개수

특징:
- 불필요한 변수 추가 시 감소할 수 있음
- 모델 비교 시 R^2보다 Adjusted R^2 권장
```

#### 다중공선성 (Multicollinearity)

**정의**: 독립변수들 간에 높은 상관관계가 존재하는 현상

**문제점**:
1. 회귀계수 추정이 불안정해짐
2. 표준오차가 증가함
3. 계수의 부호가 뒤바뀔 수 있음
4. 개별 변수의 영향력 해석이 어려움

#### VIF (Variance Inflation Factor)

다중공선성 진단 지표:

```
VIF_j = 1 / (1 - R_j^2)

R_j^2: X_j를 나머지 독립변수들로 회귀했을 때의 결정계수

해석 기준:
- VIF = 1: 상관 없음 (이상적)
- VIF 1~5: 약한 상관 (대체로 허용)
- VIF 5~10: 중간 상관 (주의 필요)
- VIF > 10: 강한 상관 (변수 제거 검토)
```

### 실습 코드

#### 단순회귀: weight -> mpg

```python
# 단순선형회귀: weight만 사용
X_simple = df_clean[['weight']].values
y = df_clean['mpg'].values

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

y_pred_simple = model_simple.predict(X_simple)
r2_simple = r2_score(y, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y, y_pred_simple))

print("=== 단순선형회귀 결과 (weight -> mpg) ===")
print(f"절편 (B0): {model_simple.intercept_:.4f}")
print(f"기울기 (B1): {model_simple.coef_[0]:.6f}")
print(f"회귀식: mpg = {model_simple.intercept_:.2f} + ({model_simple.coef_[0]:.6f}) * weight")
print(f"R^2: {r2_simple:.4f}")
print(f"RMSE: {rmse_simple:.4f}")
print(f"\n해석: weight가 mpg 변동의 {r2_simple*100:.1f}%를 설명함")
```

---

#### 중다회귀: 여러 변수 사용

```python
# 중다선형회귀: weight, horsepower, displacement 사용
feature_cols = ['weight', 'horsepower', 'displacement']
X_multi = df_clean[feature_cols].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

y_pred_multi = model_multi.predict(X_multi)
r2_multi = r2_score(y, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y, y_pred_multi))

# 조정 R^2 계산
n = len(y)
p = len(feature_cols)
adj_r2_multi = 1 - (1 - r2_multi) * (n - 1) / (n - p - 1)

print("=== 중다선형회귀 결과 (weight, horsepower, displacement -> mpg) ===")
print(f"절편: {model_multi.intercept_:.4f}")
print("\n각 변수의 회귀계수:")
for name, coef in zip(feature_cols, model_multi.coef_):
    print(f"  {name:15}: {coef:+.6f}")

print(f"\nR^2: {r2_multi:.4f}")
print(f"Adjusted R^2: {adj_r2_multi:.4f}")
print(f"RMSE: {rmse_multi:.4f}")
```

---

#### 단순회귀 vs 중다회귀 비교

```python
# 성능 비교
print("=== 단순회귀 vs 중다회귀 비교 ===")
print(f"{'지표':<20} {'단순회귀':>12} {'중다회귀':>12} {'개선':>12}")
print("-" * 60)
print(f"{'R^2':<20} {r2_simple:>12.4f} {r2_multi:>12.4f} {(r2_multi - r2_simple):>+12.4f}")
print(f"{'RMSE':<20} {rmse_simple:>12.4f} {rmse_multi:>12.4f} {(rmse_multi - rmse_simple):>+12.4f}")

# 조정 R^2 비교를 위해 단순회귀의 조정 R^2도 계산
adj_r2_simple = 1 - (1 - r2_simple) * (n - 1) / (n - 1 - 1)
print(f"{'Adjusted R^2':<20} {adj_r2_simple:>12.4f} {adj_r2_multi:>12.4f} {(adj_r2_multi - adj_r2_simple):>+12.4f}")

print(f"\n결론: 변수 추가로 설명력이 {(r2_multi - r2_simple)*100:.1f}%p 향상됨")
```

---

#### VIF 계산 및 다중공선성 진단

```python
# VIF 계산
print("=== VIF (다중공선성 진단) ===")

X_vif = df_clean[feature_cols]
vif_data = []

for i, col in enumerate(feature_cols):
    vif_value = variance_inflation_factor(X_vif.values, i)
    vif_data.append({
        'Variable': col,
        'VIF': round(vif_value, 2)
    })

vif_df = pd.DataFrame(vif_data)
print(vif_df)

# VIF 해석
print("\n=== VIF 해석 ===")
for _, row in vif_df.iterrows():
    var = row['Variable']
    vif = row['VIF']
    if vif > 10:
        status = "CRITICAL - removal recommended"
    elif vif > 5:
        status = "WARNING - review needed"
    elif vif > 1:
        status = "OK - acceptable"
    else:
        status = "IDEAL - no correlation"
    print(f"  {var:15}: VIF = {vif:6.2f} -> {status}")
```

---

#### 다중공선성 원인 분석

```python
# 독립변수들 간의 상관관계 확인
print("=== 독립변수 간 상관관계 ===")
X_corr = df_clean[feature_cols].corr()
print(X_corr.round(3))

print("\n해석:")
print("- weight와 displacement의 상관계수가 매우 높음 (r > 0.9)")
print("- 이것이 높은 VIF의 원인임")
print("- 두 변수 중 하나를 제거하거나 주성분 분석(PCA) 활용 검토 필요")
```

---

#### 변수 선택 후 재분석

```python
# displacement 제거 후 분석
feature_cols_reduced = ['weight', 'horsepower']
X_reduced = df_clean[feature_cols_reduced].values

model_reduced = LinearRegression()
model_reduced.fit(X_reduced, y)

y_pred_reduced = model_reduced.predict(X_reduced)
r2_reduced = r2_score(y, y_pred_reduced)
adj_r2_reduced = 1 - (1 - r2_reduced) * (n - 1) / (n - len(feature_cols_reduced) - 1)
rmse_reduced = np.sqrt(mean_squared_error(y, y_pred_reduced))

print("=== 변수 축소 후 결과 (weight, horsepower -> mpg) ===")
print(f"R^2: {r2_reduced:.4f}")
print(f"Adjusted R^2: {adj_r2_reduced:.4f}")
print(f"RMSE: {rmse_reduced:.4f}")

# VIF 재계산
print("\n=== 축소 모델 VIF ===")
X_vif_reduced = df_clean[feature_cols_reduced]
for i, col in enumerate(feature_cols_reduced):
    vif_value = variance_inflation_factor(X_vif_reduced.values, i)
    print(f"  {col:15}: VIF = {vif_value:.2f}")

print(f"\n결론: displacement 제거 후에도 설명력(Adjusted R^2)은 유사하게 유지됨")
print(f"      다중공선성 문제가 크게 완화되어 해석이 더 명확해짐")
```

---

#### 모델 비교 시각화

```python
# 3개 모델 비교 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    ('Simple (weight only)', y_pred_simple, r2_simple),
    ('Multiple (3 vars)', y_pred_multi, r2_multi),
    ('Reduced (2 vars)', y_pred_reduced, r2_reduced)
]

for ax, (name, y_pred, r2) in zip(axes, models):
    ax.scatter(y, y_pred, alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title(f'{name}\nR^2 = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("모델 비교 시각화 저장: model_comparison.png")
```

---

### 결과 해설

- 단순회귀(weight만 사용)의 R^2는 약 0.69로, weight가 mpg 변동의 69%를 설명함
- 중다회귀(3개 변수)로 R^2가 약 0.71로 향상되었으나, VIF가 매우 높음 (displacement > 20)
- weight, horsepower, displacement 간에 강한 상관관계가 존재하여 다중공선성 문제 발생
- displacement 제거 후 Adjusted R^2는 유사하게 유지되며 VIF가 크게 감소함
- 변수 선택 시 단순히 R^2 향상만 볼 것이 아니라 다중공선성과 해석 가능성도 고려해야 함

---

## 파트 3: 선형 모델의 한계와 머신러닝으로의 전환

### 개념 설명

#### 선형 모델의 가정

선형회귀가 유효하려면 다음 가정이 충족되어야 함:

| 가정 | 내용 | 위반 시 문제 |
|------|------|-------------|
| **선형성** | X와 Y의 관계가 직선 | 예측 왜곡 |
| **독립성** | 오차 간 독립 | 추정 비효율 |
| **등분산성** | 오차 분산 일정 | 신뢰구간 왜곡 |
| **정규성** | 오차가 정규분포 | 검정 왜곡 |

#### 선형 모델의 한계

**한계 1: 비선형 관계 포착 불가**
```
실제 관계: Y = X^2 + noise
선형 모델: Y = b0 + b1*X

결과: 모델이 곡선 관계를 직선으로 근사
      잔차에 체계적 패턴(곡선) 발생
```

**한계 2: 복잡한 상호작용 효과**
```
예시:
고온(90C) + 저습(30%) -> 불량률 5%
고온(90C) + 고습(80%) -> 불량률 15%

온도와 습도의 조합 효과가 존재
단순 선형 모델로는 포착 어려움
```

**한계 3: 고차원 데이터**
- 변수가 많아지면 다중공선성 증가
- 범주형 변수가 많으면 더미 변수 폭발
- 수동으로 변수 변환/선택 필요

#### 머신러닝의 장점

| 관점 | 통계 모델 | 머신러닝 |
|------|----------|----------|
| 목적 | 해석, 추론 | 예측 성능 |
| 가정 | 명시적 가정 필요 | 가정 최소화 |
| 비선형 | 수동 변환 필요 | 자동 학습 |
| 상호작용 | 수동 지정 | 자동 탐지 |
| 고차원 | 다중공선성 문제 | 자동 특성 선택 |

### 실습 코드

#### 잔차 분석: 가정 검증

```python
# 축소 모델의 잔차 분석
residuals = y - y_pred_reduced

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 잔차 vs 예측값 (등분산성, 선형성 확인)
axes[0, 0].scatter(y_pred_reduced, residuals, alpha=0.5, edgecolor='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].grid(True, alpha=0.3)

# 2. 잔차 히스토그램 (정규성 확인)
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
                'r-', linewidth=2, label='Normal')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].legend()

# 3. Q-Q 플롯 (정규성 확인)
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(True, alpha=0.3)

# 4. 잔차 vs 각 독립변수
axes[1, 1].scatter(df_clean['weight'], residuals, alpha=0.5, edgecolor='black', linewidth=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Weight')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Weight')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("잔차 분석 시각화 저장: residual_analysis.png")

# 정규성 검정
stat, p_value = stats.shapiro(residuals[:50])  # Shapiro-Wilk (표본 크기 제한)
print(f"\n=== 잔차 정규성 검정 (Shapiro-Wilk) ===")
print(f"통계량: {stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("결론: 잔차가 정규분포를 따르지 않음 (가정 위반)")
else:
    print("결론: 잔차가 정규분포를 따름 (가정 충족)")
```

---

#### 비선형 관계 확인

```python
# 실제 관계가 비선형인지 확인
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# weight vs mpg 산점도 + 회귀선
X_plot = np.linspace(df_clean['weight'].min(), df_clean['weight'].max(), 100)
y_pred_line = model_simple.intercept_ + model_simple.coef_[0] * X_plot

axes[0].scatter(df_clean['weight'], df_clean['mpg'], alpha=0.5, label='Data')
axes[0].plot(X_plot, y_pred_line, 'r-', linewidth=2, label='Linear fit')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('MPG')
axes[0].set_title('Weight vs MPG: Linear Fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2차 다항식 비교
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df_clean[['weight']])
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
r2_poly = r2_score(y, model_poly.predict(X_poly))

X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
y_pred_poly = model_poly.predict(X_plot_poly)

axes[1].scatter(df_clean['weight'], df_clean['mpg'], alpha=0.5, label='Data')
axes[1].plot(X_plot, y_pred_line, 'r-', linewidth=2, label=f'Linear (R^2={r2_simple:.3f})')
axes[1].plot(X_plot, y_pred_poly, 'g-', linewidth=2, label=f'Polynomial (R^2={r2_poly:.3f})')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('MPG')
axes[1].set_title('Linear vs Polynomial Fit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nonlinear_relationship.png', dpi=150, bbox_inches='tight')
plt.close()
print("비선형 관계 시각화 저장: nonlinear_relationship.png")

print(f"\n=== 선형 vs 다항회귀 비교 ===")
print(f"선형 R^2: {r2_simple:.4f}")
print(f"2차 다항 R^2: {r2_poly:.4f}")
print(f"개선: +{(r2_poly - r2_simple)*100:.1f}%p")
print("\n결론: 비선형 관계가 존재하며, 다항식 또는 머신러닝 모델이 더 적합할 수 있음")
```

---

#### 선형 모델의 한계 데모: California Housing

```python
from sklearn.datasets import fetch_california_housing

# California Housing 데이터 로드
housing = fetch_california_housing()
X_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
y_housing = housing.target

print("=== California Housing 데이터셋 ===")
print(f"데이터 크기: {X_housing.shape}")
print(f"특성 목록: {list(X_housing.columns)}")
print(f"타겟: 집값 (단위: $100,000)")

# 선형회귀 적용
model_housing = LinearRegression()
model_housing.fit(X_housing, y_housing)
y_pred_housing = model_housing.predict(X_housing)
r2_housing = r2_score(y_housing, y_pred_housing)

print(f"\n=== 선형회귀 성능 ===")
print(f"R^2: {r2_housing:.4f}")
print(f"해석: 선형 모델이 집값 변동의 {r2_housing*100:.1f}%만 설명")
print(f"      나머지 {(1-r2_housing)*100:.1f}%는 비선형 관계나 상호작용으로 추정됨")
```

---

#### 머신러닝 미리보기: 랜덤포레스트

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# 선형회귀
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr_test = r2_score(y_test, y_pred_lr)

# 랜덤포레스트 (머신러닝)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf_test = r2_score(y_test, y_pred_rf)

print("=== 선형회귀 vs 랜덤포레스트 (테스트 세트) ===")
print(f"{'모델':<20} {'R^2':>10}")
print("-" * 35)
print(f"{'Linear Regression':<20} {r2_lr_test:>10.4f}")
print(f"{'Random Forest':<20} {r2_rf_test:>10.4f}")
print(f"{'개선':>20} {(r2_rf_test - r2_lr_test)*100:>+9.1f}%p")

print("\n결론:")
print("- 랜덤포레스트가 선형회귀보다 훨씬 높은 성능을 보임")
print("- 이는 데이터에 비선형 관계와 상호작용이 존재함을 시사함")
print("- 다음 차시부터 본격적인 머신러닝 모델을 학습할 예정임")
```

---

#### 종합 인사이트 정리

```python
print("""
=================================================================
                    11차시 분석 결과 요약
=================================================================

[1] EDA 종합 적용
    - MPG 데이터: 398개 샘플, 9개 변수
    - weight, horsepower, displacement가 mpg와 강한 음의 상관
    - 4기통 vs 8기통 mpg 차이 통계적 유의 (p < 0.001)

[2] 중다회귀와 다중공선성
    - 단순회귀(weight) R^2: 0.69
    - 중다회귀(3변수) R^2: 0.71 (개선됨)
    - 그러나 displacement VIF > 20 (다중공선성 심각)
    - displacement 제거 후에도 설명력 유지, VIF 개선

[3] 선형 모델의 한계
    - 잔차 분석: 패턴 존재 (가정 일부 위반)
    - 다항회귀로 성능 개선 가능 (비선형 관계 존재)
    - California Housing: 선형 R^2 = 0.61 vs RF R^2 = 0.81
    - 복잡한 데이터에서 머신러닝이 우수함

[4] 권고사항
    - 단순한 관계: 선형회귀로 충분 (해석 용이)
    - 비선형 관계: 다항회귀 또는 머신러닝 검토
    - 다중공선성: VIF > 10이면 변수 제거 또는 PCA
    - 복잡한 예측: 머신러닝 모델 권장

=================================================================
""")
```

---

### 결과 해설

- 선형 모델은 해석이 용이하고 단순한 관계에 효과적임
- 그러나 비선형 관계, 복잡한 상호작용, 고차원 데이터에서는 한계가 있음
- 잔차 분석을 통해 가정 충족 여부를 반드시 확인해야 함
- California Housing 예시에서 랜덤포레스트가 선형회귀보다 20%p 높은 R^2를 보임
- 이는 머신러닝이 복잡한 패턴을 더 잘 포착할 수 있음을 의미함
- 다음 차시부터 본격적인 머신러닝 알고리즘을 학습함

---

## 연습 문제

### 연습 1

MPG 데이터에서 acceleration 변수를 추가하여 중다회귀 모델을 학습하고, VIF를 계산하시오.

```python
# 정답
feature_ex1 = ['weight', 'horsepower', 'acceleration']
X_ex1 = df_clean[feature_ex1]
model_ex1 = LinearRegression()
model_ex1.fit(X_ex1, y)
r2_ex1 = r2_score(y, model_ex1.predict(X_ex1))

print(f"R^2: {r2_ex1:.4f}")
print("\nVIF:")
for i, col in enumerate(feature_ex1):
    vif = variance_inflation_factor(X_ex1.values, i)
    print(f"  {col}: {vif:.2f}")
```

### 연습 2

model_year를 추가하면 성능이 향상되는지 확인하시오. Adjusted R^2를 비교하시오.

```python
# 정답
feature_ex2 = ['weight', 'horsepower', 'model_year']
X_ex2 = df_clean[feature_ex2]
model_ex2 = LinearRegression()
model_ex2.fit(X_ex2, y)
r2_ex2 = r2_score(y, model_ex2.predict(X_ex2))
adj_r2_ex2 = 1 - (1 - r2_ex2) * (n - 1) / (n - len(feature_ex2) - 1)

print(f"3변수 모델 (weight, hp, model_year)")
print(f"R^2: {r2_ex2:.4f}")
print(f"Adjusted R^2: {adj_r2_ex2:.4f}")
print(f"\nvs 2변수 모델 (weight, hp)")
print(f"Adjusted R^2 개선: {adj_r2_ex2 - adj_r2_reduced:+.4f}")
```

### 연습 3

잔차의 정규성 검정(Shapiro-Wilk)을 수행하고 결과를 해석하시오.

```python
# 정답
residuals_ex3 = y - model_ex2.predict(X_ex2)
stat, p = stats.shapiro(residuals_ex3[:50])
print(f"Shapiro-Wilk 검정")
print(f"통계량: {stat:.4f}")
print(f"p-value: {p:.4f}")
print(f"결론: {'정규성 가정 충족' if p >= 0.05 else '정규성 가정 위반'}")
```

---

## 핵심 정리

### EDA 종합 체크리스트

| 단계 | 핵심 활동 | 도구 |
|:----:|----------|------|
| 1 | 데이터 개요 파악 | shape, dtypes, head, describe |
| 2 | 결측치/이상치 처리 | isnull(), IQR |
| 3 | 분포 시각화 | 히스토그램, 박스플롯 |
| 4 | 상관관계 분석 | 상관행렬, 히트맵 |
| 5 | 그룹별 비교 | t-검정, ANOVA |
| 6 | 인사이트 도출 | 발견 정리, 권고사항 |

### 주요 수학 공식

| 공식 | 수식 | 해석 |
|------|------|------|
| 중다회귀 | Y = B0 + B1X1 + ... + BpXp | 여러 변수로 Y 예측 |
| 조정 R^2 | 1 - (1-R^2)(n-1)/(n-p-1) | 변수 개수 고려한 설명력 |
| VIF | 1 / (1 - Rj^2) | 다중공선성 지표, >10이면 문제 |

### 선형 모델 한계와 대안

| 한계 | 대안 |
|------|------|
| 비선형 관계 | 다항회귀, 의사결정나무 |
| 상호작용 | 교호작용 항 추가, 랜덤포레스트 |
| 다중공선성 | 변수 제거, PCA, Ridge/Lasso |
| 복잡한 패턴 | 머신러닝 모델 |

---

## 다음 차시 예고

**12차시: 머신러닝 소개와 문제 유형**

- 머신러닝이란?
- 지도학습 vs 비지도학습
- 분류 vs 회귀 문제
- sklearn 기본 사용법

지금까지 배운 통계적 기법이 머신러닝의 기초가 됨!
