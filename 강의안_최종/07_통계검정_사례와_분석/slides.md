---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 7차시'
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

# 통계 검정 실습

## 7차시 | Part II. 기초 수리와 데이터 분석

**실제 데이터로 배우는 가설검정**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **가설검정**의 기본 개념과 p-value를 이해한다
2. **t-검정, ANOVA**로 집단 간 평균을 비교한다
3. **카이제곱 검정**으로 범주형 변수의 관계를 분석한다
4. **비모수 검정**의 활용 상황을 판단한다

---

# 강의 구성

| 파트 | 대주제 | 시간 |
|:----:|--------|:----:|
| 1 | 가설검정 기초 | 10분 |
| 2 | 모수 검정 (t-검정, ANOVA) | 10분 |
| 3 | 범주형 데이터 검정 (카이제곱) | 5분 |
| 4 | 비모수 검정 | 5분 |

---

# 실습에서 사용할 데이터셋

## 공개 데이터셋 3종

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Iris** | sklearn | 품종별 꽃잎 길이 비교 |
| **Titanic** | seaborn | 성별-생존 관계 분석 |
| **Wine Quality** | UCI | 품질 등급별 알코올 함량 비교 |

### 환경 설정

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.datasets import load_iris
```

---

<!-- _class: lead -->

# Part 1
## 가설검정 기초

---

# 왜 통계 검정이 필요한가?

## 제조 현장의 질문들

| 상황 | 질문 |
|------|------|
| 품질 비교 | "A라인과 B라인 불량률이 정말 다른가?" |
| 공정 개선 | "새 공정으로 품질이 실제로 좋아졌는가?" |
| 원인 분석 | "교대 시간과 불량률은 관련이 있는가?" |
| 효과 검증 | "교육 후 작업 효율이 정말 올랐는가?" |

### 핵심 문제

> "눈으로 보이는 차이"가 "통계적으로 의미 있는 차이"인가?

---

# 가설검정이란?

## 데이터를 바탕으로 주장을 검증하는 절차

### 일상적 예시

```
주장: "이 공정 개선으로 불량률이 줄었다"

질문: 이 차이가 우연인가? 실제 효과인가?
```

### 통계적 접근

1. 가설 설정 (주장을 수학적으로 표현)
2. 데이터 수집
3. 검정 통계량 계산
4. **p-value로 판단**

---

# 귀무가설 (H0)

## Null Hypothesis - "차이가 없다"

### 정의

> 귀무가설은 **"아무 일도 일어나지 않았다"**는 가정
>
> 기존 상태, 차이 없음, 효과 없음을 주장

### 예시

| 상황 | 귀무가설 (H0) |
|------|--------------|
| 두 공정 비교 | 두 공정의 불량률은 **같다** |
| 교육 효과 | 교육 전후 생산성은 **차이 없다** |
| 라인 비교 | A라인과 B라인 품질은 **동일하다** |

---

# 대립가설 (H1 또는 Ha)

## Alternative Hypothesis - "차이가 있다"

### 정의

> 대립가설은 **"실제로 차이/효과가 있다"**는 주장
>
> 연구자가 입증하고 싶은 가설

### 예시

| 상황 | 대립가설 (H1) |
|------|--------------|
| 두 공정 비교 | 두 공정의 불량률은 **다르다** |
| 교육 효과 | 교육 후 생산성이 **높아졌다** |
| 라인 비교 | A라인과 B라인 품질은 **다르다** |

---

# 가설검정의 논리

## 귀무가설을 기각하는 방식

```
           가설검정의 흐름

┌──────────────────────────────────────┐
│  1. 귀무가설(H0)이 참이라고 가정     │
│                 ↓                    │
│  2. 관측된 결과가 나올 확률 계산     │
│            (= p-value)               │
│                 ↓                    │
│  3. p-value가 매우 작으면            │
│     → H0가 참이기 어려움             │
│     → H0 기각, H1 채택               │
└──────────────────────────────────────┘
```

---

# p-value란?

## 귀무가설이 참일 때, 관측 결과가 나올 확률

### 정의

```
p-value = P(현재 데이터 또는 더 극단적인 데이터 | H0가 참)
```

### 직관적 이해

- p-value가 **작을수록** → 관측 결과가 우연이 아닐 가능성 높음
- p-value가 **클수록** → 우연히 발생했을 가능성 높음

### 예시

> p = 0.02 → "H0가 참이라면, 이런 결과가 나올 확률이 2%"
> → 너무 희귀함 → H0를 기각

---

# p-value 해석 주의사항

## 자주 하는 오해

| 잘못된 해석 | 올바른 해석 |
|------------|------------|
| "H0가 참일 확률" | H0가 참일 때 이 결과가 나올 확률 |
| "H1이 참일 확률" | 결과가 H0와 얼마나 상반되는지 척도 |
| "효과의 크기" | 통계적 유의성만 측정, 효과 크기 아님 |

### 핵심

> p-value는 **증거의 강도**를 나타냄
>
> 실제 **효과의 크기**는 별도로 확인해야 함

---

# 유의수준 (Significance Level)

## Alpha (α) - 기각 기준

### 정의

```
유의수준(α) = H0를 기각하기 위한 p-value 기준
```

### 표준 기준

| α 값 | 의미 | 사용 상황 |
|------|------|----------|
| **0.05** | 5% | 일반적 연구 (가장 보편적) |
| 0.01 | 1% | 엄격한 기준 필요 시 |
| 0.10 | 10% | 탐색적 분석 |

### 판단 규칙

> **p < α** → H0 기각 → "통계적으로 유의함"
>
> **p ≥ α** → H0 기각 못함 → "유의하지 않음"

---

# 유의수준 0.05의 의미

## 왜 0.05인가?

### 역사적 배경

- 통계학자 Ronald Fisher가 제안
- "20번 중 1번 정도는 우연으로 발생 가능"

### 실무적 해석

```
α = 0.05 선택 시

- 100번 검정하면 약 5번은 잘못된 결론 가능
- 실제로 차이가 없는데 "차이가 있다"고 할 위험 5%
- 이 정도 위험은 감수할 수 있다는 합의
```

### 주의

> α = 0.05는 **절대적 기준이 아님**
> 상황에 따라 α = 0.01 또는 0.10 사용 가능

---

# 1종 오류와 2종 오류

## 두 가지 잘못된 결정

| | H0가 실제로 참 | H0가 실제로 거짓 |
|---|---|---|
| **H0 기각** | **1종 오류 (α)** | 올바른 결정 |
| **H0 기각 못함** | 올바른 결정 | **2종 오류 (β)** |

### 제조업 예시

| 오류 유형 | 설명 | 결과 |
|----------|------|------|
| 1종 오류 | 차이 없는데 "차이 있다" | 불필요한 공정 변경 |
| 2종 오류 | 차이 있는데 "차이 없다" | 문제 방치 |

---

# 가설검정 단계 정리

## 5단계 절차

```
1. 가설 설정
   H0: 두 그룹의 평균은 같다 (μ1 = μ2)
   H1: 두 그룹의 평균은 다르다 (μ1 ≠ μ2)

2. 유의수준 결정
   α = 0.05

3. 검정 통계량 계산
   t-value, F-value, χ² 등

4. p-value 계산
   scipy.stats 활용

5. 결론 도출
   p < 0.05 → H0 기각 → "통계적으로 유의한 차이"
   p ≥ 0.05 → H0 기각 못함 → "유의한 차이 없음"
```

---

# Part 1 정리

## 가설검정 기초 개념

### 핵심 용어

| 용어 | 의미 |
|------|------|
| 귀무가설 (H0) | "차이 없다" - 기본 가정 |
| 대립가설 (H1) | "차이 있다" - 입증하고 싶은 주장 |
| p-value | H0가 참일 때 관측 결과가 나올 확률 |
| 유의수준 (α) | 기각 기준, 보통 0.05 |

### 판단 규칙

> **p < 0.05** → 통계적으로 유의함 (H0 기각)
>
> **p ≥ 0.05** → 통계적으로 유의하지 않음

---

<!-- _class: lead -->

# Part 2
## 모수 검정 (t-검정, ANOVA)

---

# 모수 검정이란?

## 정규분포를 가정하는 검정 방법

### 전제 조건

- 데이터가 **정규분포**를 따름
- 모집단의 **모수**(평균, 분산)를 추정

### 대표적 검정 방법

| 검정 | 목적 | 비교 그룹 수 |
|------|------|-------------|
| **독립표본 t-검정** | 두 그룹 평균 비교 | 2개 (독립) |
| **대응표본 t-검정** | 전후 평균 비교 | 2개 (쌍) |
| **일원분산분석 (ANOVA)** | 여러 그룹 평균 비교 | 3개 이상 |

---

# 독립표본 t-검정

## Independent Samples t-test

### 언제 사용?

- **서로 다른 두 그룹**의 평균 비교
- 각 그룹이 독립적 (다른 대상)

### 제조업 예시

| 비교 대상 | 예시 |
|----------|------|
| 라인 비교 | A라인 vs B라인 품질 |
| 설비 비교 | 신규 설비 vs 기존 설비 생산량 |
| 원료 비교 | 공급사A vs 공급사B 원료 품질 |

---

# 독립표본 t-검정: Iris 데이터셋

## 품종별 꽃잎 길이 비교

```python
from sklearn.datasets import load_iris
import pandas as pd
from scipy import stats

# Iris 데이터 로드
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 품종 이름 매핑 (0: setosa, 1: versicolor, 2: virginica)
df['species_name'] = df['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

print(df.head())
print(f"\n전체 샘플 수: {len(df)}")
print(f"품종별 샘플 수:\n{df['species_name'].value_counts()}")
```

---

# Iris 데이터 탐색

## 품종별 꽃잎 길이 분포

```python
# 품종별 꽃잎 길이 요약
petal_length = 'petal length (cm)'
summary = df.groupby('species_name')[petal_length].agg(['mean', 'std', 'count'])
print(summary.round(2))
```

### 결과

```
              mean   std  count
species_name
setosa        1.46  0.17     50
versicolor    4.26  0.47     50
virginica     5.55  0.55     50
```

> 눈으로 보면 차이가 크다. 하지만 **통계적으로 유의한가?**

---

# t-검정 실행: setosa vs versicolor

## scipy.stats.ttest_ind 활용

```python
# 두 품종 데이터 분리
setosa = df[df['species_name'] == 'setosa']['petal length (cm)']
versicolor = df[df['species_name'] == 'versicolor']['petal length (cm)']

# 독립표본 t-검정
t_stat, p_value = stats.ttest_ind(setosa, versicolor)

print("=== 독립표본 t-검정 결과 ===")
print(f"그룹 1 (setosa): 평균 = {setosa.mean():.2f}, n = {len(setosa)}")
print(f"그룹 2 (versicolor): 평균 = {versicolor.mean():.2f}, n = {len(versicolor)}")
print(f"\nt-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.2e}")
```

---

# t-검정 결과 해석

## 통계적 유의성 판단

```python
# 결과 출력
print("=== 결과 해석 ===")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.2e}")

alpha = 0.05
if p_value < alpha:
    print(f"\np < {alpha} → 귀무가설 기각")
    print("결론: 두 품종의 꽃잎 길이는 통계적으로 유의한 차이가 있다")
else:
    print(f"\np >= {alpha} → 귀무가설 기각 못함")
    print("결론: 두 품종의 꽃잎 길이 차이가 통계적으로 유의하지 않다")
```

### 실제 결과

> p-value ≈ 1.08e-31 (거의 0)
> → **매우 강력한 증거**로 H0 기각

---

# t-검정 시각화

## 두 그룹 분포 비교

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# 히스토그램
ax.hist(setosa, bins=15, alpha=0.7, label='Setosa', color='blue')
ax.hist(versicolor, bins=15, alpha=0.7, label='Versicolor', color='orange')

# 평균선
ax.axvline(setosa.mean(), color='blue', linestyle='--', linewidth=2)
ax.axvline(versicolor.mean(), color='orange', linestyle='--', linewidth=2)

ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Frequency')
ax.set_title(f't-test: p-value = {p_value:.2e}')
ax.legend()
plt.show()
```

---

# 대응표본 t-검정

## Paired Samples t-test

### 언제 사용?

- **동일 대상**의 전후 비교
- 각 쌍이 연결되어 있음

### 제조업 예시

| 비교 대상 | 예시 |
|----------|------|
| 교육 효과 | 같은 작업자의 교육 전후 생산성 |
| 공정 개선 | 같은 설비의 개선 전후 불량률 |
| 계절 효과 | 같은 라인의 여름/겨울 품질 |

---

# 대응표본 t-검정 예시

## 교육 전후 작업 효율 비교

```python
# 가상 데이터: 10명 작업자의 교육 전후 생산량 (개/시간)
np.random.seed(42)
before = np.array([45, 52, 48, 55, 50, 47, 53, 49, 51, 46])
after = before + np.random.normal(5, 3, 10)  # 평균 5개 향상

# 대응표본 t-검정
t_stat, p_value = stats.ttest_rel(before, after)

print("=== 대응표본 t-검정 결과 ===")
print(f"교육 전 평균: {before.mean():.1f}")
print(f"교육 후 평균: {after.mean():.1f}")
print(f"평균 변화: {(after - before).mean():.1f}")
print(f"\nt-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

# 대응표본 t-검정 해석

## 교육 효과 검증

```python
alpha = 0.05

print("\n=== 결과 해석 ===")
if p_value < alpha:
    print(f"p = {p_value:.4f} < {alpha}")
    print("결론: 교육 후 생산성이 통계적으로 유의하게 향상되었다")
else:
    print(f"p = {p_value:.4f} >= {alpha}")
    print("결론: 교육 효과가 통계적으로 유의하지 않다")

# 효과 크기 (Cohen's d)
diff = after - before
cohens_d = diff.mean() / diff.std()
print(f"\n효과 크기 (Cohen's d): {cohens_d:.2f}")
```

### Cohen's d 해석

| d 값 | 효과 크기 |
|------|----------|
| 0.2 | 작음 |
| 0.5 | 중간 |
| 0.8 | 큼 |

---

# 독립 vs 대응 t-검정 비교

## 언제 어떤 검정을?

| 구분 | 독립표본 t-검정 | 대응표본 t-검정 |
|------|---------------|---------------|
| **데이터** | 서로 다른 대상 | 동일 대상의 반복 측정 |
| **함수** | ttest_ind() | ttest_rel() |
| **예시** | A라인 vs B라인 | 개선 전 vs 개선 후 |
| **장점** | 독립 그룹 비교 | 개인차 통제 |

### 선택 기준

```
Q: 같은 대상을 두 번 측정했는가?
   YES → 대응표본 t-검정 (ttest_rel)
   NO  → 독립표본 t-검정 (ttest_ind)
```

---

# 일원분산분석 (One-way ANOVA)

## 3개 이상 그룹의 평균 비교

### 언제 사용?

- 비교할 그룹이 **3개 이상**
- t-검정을 여러 번 하면 1종 오류 증가

### 가설

```
H0: μ1 = μ2 = μ3 = ... = μk (모든 그룹 평균 동일)
H1: 적어도 하나의 그룹 평균이 다름
```

### 제조업 예시

- 3개 라인의 품질 비교
- 4개 교대조의 생산성 비교
- 5개 공급사 원료의 품질 비교

---

# ANOVA: Wine Quality 데이터셋

## 품질 등급별 알코올 함량 비교

```python
# Wine Quality 데이터 로드 (UCI Repository)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url, sep=';')

print(f"데이터 크기: {wine.shape}")
print(f"\n품질 등급 분포:\n{wine['quality'].value_counts().sort_index()}")

# 품질 등급별 알코올 함량 요약
print("\n품질 등급별 알코올 함량:")
print(wine.groupby('quality')['alcohol'].agg(['mean', 'std', 'count']).round(2))
```

---

# Wine Quality 데이터 구조

## 품질 등급과 알코올 함량

```
품질 등급 분포:
3      10
4      53
5     681
6     638
7     199
8      18

품질 등급별 알코올 함량:
         mean   std  count
quality
3        9.96  0.83     10
4        10.27 1.00     53
5        9.90  0.74    681
6        10.63 1.05    638
7        11.47 0.98    199
8        12.09 0.77     18
```

> 품질이 높을수록 알코올 함량이 높아 보인다. 통계적으로 유의한가?

---

# ANOVA 실행: 품질별 알코올 함량

## scipy.stats.f_oneway 활용

```python
# 품질 등급별 알코올 데이터 분리
groups = [wine[wine['quality'] == q]['alcohol'] for q in wine['quality'].unique()]

# 또는 더 명확하게
q3 = wine[wine['quality'] == 3]['alcohol']
q4 = wine[wine['quality'] == 4]['alcohol']
q5 = wine[wine['quality'] == 5]['alcohol']
q6 = wine[wine['quality'] == 6]['alcohol']
q7 = wine[wine['quality'] == 7]['alcohol']
q8 = wine[wine['quality'] == 8]['alcohol']

# 일원분산분석
f_stat, p_value = stats.f_oneway(q3, q4, q5, q6, q7, q8)

print("=== 일원분산분석 (ANOVA) 결과 ===")
print(f"F-통계량: {f_stat:.4f}")
print(f"p-value: {p_value:.2e}")
```

---

# ANOVA 결과 해석

## 통계적 유의성 판단

```python
alpha = 0.05

print("\n=== 결과 해석 ===")
if p_value < alpha:
    print(f"p = {p_value:.2e} < {alpha}")
    print("결론: 품질 등급에 따라 알코올 함량에 유의한 차이가 있다")
    print("\n주의: 어느 그룹 간에 차이가 있는지는 사후분석 필요")
else:
    print(f"p = {p_value:.2e} >= {alpha}")
    print("결론: 품질 등급별 알코올 함량 차이가 유의하지 않다")
```

### ANOVA의 한계

> ANOVA는 "차이가 있다/없다"만 알려줌
>
> **어느 그룹 간** 차이인지는 **사후분석** 필요

---

# ANOVA 시각화

## 품질 등급별 알코올 함량 분포

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot
wine.boxplot(column='alcohol', by='quality', ax=ax)

ax.set_xlabel('Quality Score')
ax.set_ylabel('Alcohol (%)')
ax.set_title(f'Alcohol by Quality (ANOVA p = {p_value:.2e})')
plt.suptitle('')  # 기본 제목 제거
plt.show()
```

---

# ANOVA 사후분석 (Post-hoc)

## Tukey HSD 검정

```python
from scipy.stats import tukey_hsd

# Tukey HSD 검정 (어느 그룹 간 차이인지)
result = tukey_hsd(q5, q6, q7)

print("=== Tukey HSD 사후분석 ===")
print("\n그룹 비교 (5, 6, 7등급):")
print(result)

# 신뢰구간 확인
print("\n신뢰구간:")
print(result.confidence_interval())
```

### 해석

- 신뢰구간이 0을 포함하지 않으면 → 유의한 차이
- 여러 쌍 비교 시 다중비교 보정 적용됨

---

# Part 2 정리

## 모수 검정 요약

| 검정 | 함수 | 사용 상황 |
|------|------|----------|
| 독립표본 t-검정 | `ttest_ind()` | 두 독립 그룹 평균 비교 |
| 대응표본 t-검정 | `ttest_rel()` | 동일 대상 전후 비교 |
| 일원분산분석 | `f_oneway()` | 3개 이상 그룹 비교 |

### 핵심 코드

```python
from scipy import stats

# 독립표본 t-검정
t, p = stats.ttest_ind(group1, group2)

# 대응표본 t-검정
t, p = stats.ttest_rel(before, after)

# ANOVA
f, p = stats.f_oneway(g1, g2, g3)
```

---

<!-- _class: lead -->

# Part 3
## 범주형 데이터 검정

---

# 카이제곱 검정이란?

## Chi-square (χ²) Test

### 언제 사용?

- **범주형 변수** 간의 관계 분석
- 관측 빈도와 기대 빈도 비교

### 유형

| 유형 | 목적 | 예시 |
|------|------|------|
| 독립성 검정 | 두 범주형 변수의 관계 | 성별 vs 생존 |
| 적합도 검정 | 관측이 이론과 일치하는지 | 불량 유형 분포 |

---

# 카이제곱 독립성 검정

## 두 범주형 변수의 관련성

### 가설

```
H0: 두 변수는 독립이다 (관련 없음)
H1: 두 변수는 독립이 아니다 (관련 있음)
```

### 제조업 예시

| 변수 1 | 변수 2 | 질문 |
|--------|--------|------|
| 교대조 | 불량 발생 | 교대조에 따라 불량률이 다른가? |
| 라인 | 불량 유형 | 라인에 따라 불량 유형이 다른가? |
| 경력 | 사고 발생 | 경력과 사고 발생이 관련있는가? |

---

# 카이제곱 검정: Titanic 데이터셋

## 성별과 생존의 관계

```python
import seaborn as sns

# Titanic 데이터 로드
titanic = sns.load_dataset('titanic')

print(f"데이터 크기: {titanic.shape}")
print(f"\n변수 목록: {titanic.columns.tolist()}")

# 성별 분포
print(f"\n성별 분포:\n{titanic['sex'].value_counts()}")

# 생존 분포
print(f"\n생존 분포:\n{titanic['survived'].value_counts()}")
```

---

# 교차표 (Crosstab) 만들기

## 성별 vs 생존 빈도표

```python
# 교차표 생성
crosstab = pd.crosstab(titanic['sex'], titanic['survived'],
                       margins=True)
crosstab.columns = ['사망', '생존', '합계']
crosstab.index = ['여성', '남성', '합계']

print("=== 성별-생존 교차표 ===")
print(crosstab)
```

### 결과

```
       사망  생존  합계
여성     81  233  314
남성    468  109  577
합계    549  342  891
```

> 여성: 233/314 = 74% 생존
> 남성: 109/577 = 19% 생존

---

# 교차표 비율 확인

## 행별 비율로 해석

```python
# 행 비율 (성별 내 생존율)
crosstab_pct = pd.crosstab(titanic['sex'], titanic['survived'],
                           normalize='index') * 100

print("=== 성별별 생존율 (%) ===")
print(crosstab_pct.round(1))
```

### 결과

```
        사망   생존
female  25.8  74.2
male    81.1  18.9
```

> 여성 생존율 74% vs 남성 생존율 19%
>
> 이 차이가 통계적으로 유의한가?

---

# 카이제곱 검정 실행

## scipy.stats.chi2_contingency 활용

```python
from scipy.stats import chi2_contingency

# 교차표 (margins 없이)
crosstab = pd.crosstab(titanic['sex'], titanic['survived'])

# 카이제곱 검정
chi2, p_value, dof, expected = chi2_contingency(crosstab)

print("=== 카이제곱 검정 결과 ===")
print(f"카이제곱 통계량: {chi2:.4f}")
print(f"자유도: {dof}")
print(f"p-value: {p_value:.2e}")

print("\n기대 빈도:")
print(pd.DataFrame(expected,
                   index=['female', 'male'],
                   columns=['died', 'survived']).round(1))
```

---

# 카이제곱 검정 결과 해석

## 통계적 유의성 판단

```python
alpha = 0.05

print("=== 결과 해석 ===")
print(f"p-value: {p_value:.2e}")

if p_value < alpha:
    print(f"\np < {alpha} → 귀무가설 기각")
    print("결론: 성별과 생존 여부는 통계적으로 유의한 관계가 있다")
else:
    print(f"\np >= {alpha} → 귀무가설 기각 못함")
    print("결론: 성별과 생존 여부의 관계가 유의하지 않다")
```

### 실제 결과

> χ² = 260.72, p ≈ 1.2e-58
> → **매우 강력한 증거**로 H0 기각
> → 성별과 생존은 관련이 있다

---

# 교차표 히트맵 시각화

## 관측 빈도와 기대 빈도 비교

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 관측 빈도
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Observed Frequencies')

# 기대 빈도
expected_df = pd.DataFrame(expected,
                           index=crosstab.index,
                           columns=crosstab.columns)
sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Oranges', ax=axes[1])
axes[1].set_title('Expected Frequencies (if independent)')

plt.tight_layout()
plt.show()
```

---

# Part 3 정리

## 카이제곱 검정 요약

### 핵심 개념

- **교차표**: 두 범주형 변수의 빈도 집계
- **기대 빈도**: 변수가 독립일 때 예상되는 빈도
- **카이제곱 통계량**: 관측과 기대의 차이 측정

### 핵심 코드

```python
from scipy.stats import chi2_contingency

# 교차표 생성
crosstab = pd.crosstab(df['var1'], df['var2'])

# 카이제곱 검정
chi2, p_value, dof, expected = chi2_contingency(crosstab)

# 해석: p < 0.05 → 두 변수는 관련이 있다
```

---

<!-- _class: lead -->

# Part 4
## 비모수 검정

---

# 비모수 검정이란?

## 정규분포를 가정하지 않는 검정

### 언제 사용?

- 데이터가 **정규분포를 따르지 않을 때**
- **표본 크기가 작을 때** (n < 30)
- **순서형 데이터**일 때
- **이상치가 많을 때**

### 장단점

| 장점 | 단점 |
|------|------|
| 분포 가정 불필요 | 모수 검정보다 검정력 낮음 |
| 이상치에 강건함 | 순위만 사용하여 정보 손실 |
| 표본 크기 작아도 됨 | 해석이 덜 직관적 |

---

# 비모수 검정 종류

## 모수 검정과 대응 관계

| 모수 검정 | 비모수 검정 | 사용 상황 |
|----------|------------|----------|
| 독립표본 t-검정 | **Mann-Whitney U** | 두 독립 그룹 비교 |
| 대응표본 t-검정 | **Wilcoxon 부호순위** | 전후 비교 (쌍) |
| 일원분산분석 | **Kruskal-Wallis** | 3개 이상 그룹 |

### 선택 기준

```
정규성 검정 → 정규분포 따름?
  YES → 모수 검정 (t-test, ANOVA)
  NO  → 비모수 검정 (Mann-Whitney, Kruskal-Wallis)
```

---

# 정규성 검정

## Shapiro-Wilk 검정

```python
from scipy.stats import shapiro

# 예시 데이터
data = np.random.exponential(scale=2, size=50)  # 비정규 분포

# Shapiro-Wilk 검정
stat, p_value = shapiro(data)

print("=== 정규성 검정 (Shapiro-Wilk) ===")
print(f"통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n결론: 정규분포를 따르지 않음 → 비모수 검정 권장")
else:
    print("\n결론: 정규분포를 따름 → 모수 검정 사용 가능")
```

---

# Mann-Whitney U 검정

## 두 독립 그룹의 중앙값 비교

### 특징

- 독립표본 t-검정의 비모수 대안
- **순위**를 기반으로 비교
- 정규분포 가정 불필요

### 코드

```python
from scipy.stats import mannwhitneyu

# 두 그룹 데이터
group1 = np.array([12, 15, 18, 14, 16, 13, 17, 11, 19, 15])
group2 = np.array([22, 25, 21, 28, 24, 26, 23, 27, 29, 25])

# Mann-Whitney U 검정
stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"U-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

# Mann-Whitney U 예시

## Iris 데이터로 비교

```python
# setosa vs versicolor 꽃잎 길이 비교 (비모수)
from scipy.stats import mannwhitneyu

setosa = df[df['species_name'] == 'setosa']['petal length (cm)']
versicolor = df[df['species_name'] == 'versicolor']['petal length (cm)']

# Mann-Whitney U 검정
u_stat, p_value = mannwhitneyu(setosa, versicolor, alternative='two-sided')

print("=== Mann-Whitney U 검정 ===")
print(f"U-통계량: {u_stat:.4f}")
print(f"p-value: {p_value:.2e}")

# t-검정 결과와 비교
t_stat, t_p = stats.ttest_ind(setosa, versicolor)
print(f"\n[비교] t-검정 p-value: {t_p:.2e}")
```

---

# Wilcoxon 부호순위 검정

## 대응 표본의 비모수 검정

### 특징

- 대응표본 t-검정의 비모수 대안
- 차이의 **부호와 순위** 기반
- 전후 비교에 적합

### 코드

```python
from scipy.stats import wilcoxon

# 전후 데이터
before = np.array([45, 52, 48, 55, 50, 47, 53, 49, 51, 46])
after = np.array([50, 58, 52, 60, 55, 52, 58, 54, 56, 51])

# Wilcoxon 부호순위 검정
stat, p_value = wilcoxon(before, after)

print("=== Wilcoxon 부호순위 검정 ===")
print(f"통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

# Kruskal-Wallis 검정

## 3개 이상 그룹의 비모수 비교

### 특징

- 일원분산분석(ANOVA)의 비모수 대안
- 여러 그룹의 **순위 분포** 비교
- 정규분포 가정 불필요

### 코드

```python
from scipy.stats import kruskal

# 세 그룹 데이터 (Wine Quality)
q5 = wine[wine['quality'] == 5]['alcohol']
q6 = wine[wine['quality'] == 6]['alcohol']
q7 = wine[wine['quality'] == 7]['alcohol']

# Kruskal-Wallis 검정
h_stat, p_value = kruskal(q5, q6, q7)

print("=== Kruskal-Wallis 검정 ===")
print(f"H-통계량: {h_stat:.4f}")
print(f"p-value: {p_value:.2e}")
```

---

# 모수 vs 비모수 비교

## Wine 데이터로 결과 비교

```python
# ANOVA vs Kruskal-Wallis

# 모수 검정 (ANOVA)
f_stat, anova_p = stats.f_oneway(q5, q6, q7)

# 비모수 검정 (Kruskal-Wallis)
h_stat, kw_p = kruskal(q5, q6, q7)

print("=== 모수 vs 비모수 검정 비교 ===")
print(f"\nANOVA:")
print(f"  F-통계량: {f_stat:.4f}")
print(f"  p-value: {anova_p:.2e}")

print(f"\nKruskal-Wallis:")
print(f"  H-통계량: {h_stat:.4f}")
print(f"  p-value: {kw_p:.2e}")

print("\n두 방법 모두 유의한 결과 → 결론 일관됨")
```

---

# Part 4 정리

## 비모수 검정 요약

### 언제 사용?

- 정규분포 가정을 만족하지 않을 때
- 표본 크기가 작을 때 (n < 30)
- 순서형(서열) 데이터일 때

### 주요 함수

```python
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

# Mann-Whitney U (두 독립 그룹)
u, p = mannwhitneyu(group1, group2)

# Wilcoxon (전후 비교)
w, p = wilcoxon(before, after)

# Kruskal-Wallis (3개 이상 그룹)
h, p = kruskal(g1, g2, g3)
```

---

<!-- _class: lead -->

# 실습편
## 종합 실습

---

# 종합 실습 개요

## 실제 데이터로 통계 검정 수행

### 실습 목표

1. 적절한 검정 방법 선택
2. scipy.stats로 검정 수행
3. 결과 해석 및 의사결정

### 사용 데이터

- Iris dataset (sklearn)
- Titanic dataset (seaborn)
- Wine Quality dataset (UCI)

---

# 실습 1: 환경 설정

## 라이브러리 및 데이터 로드

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (ttest_ind, ttest_rel, f_oneway,
                         chi2_contingency, mannwhitneyu,
                         wilcoxon, kruskal, shapiro)
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("라이브러리 로드 완료!")
```

---

# 실습 2: Iris 데이터 t-검정

## 품종별 꽃잎 너비 비교

```python
# 데이터 로드
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target_names[iris.target]

# versicolor vs virginica 꽃잎 너비
versicolor = df_iris[df_iris['species'] == 'versicolor']['petal width (cm)']
virginica = df_iris[df_iris['species'] == 'virginica']['petal width (cm)']

# 기술통계
print("=== 기술통계 ===")
print(f"Versicolor: 평균={versicolor.mean():.2f}, 표준편차={versicolor.std():.2f}")
print(f"Virginica: 평균={virginica.mean():.2f}, 표준편차={virginica.std():.2f}")
```

---

# 실습 2 계속: t-검정 수행

```python
# 정규성 검정
print("\n=== 정규성 검정 (Shapiro-Wilk) ===")
_, p1 = shapiro(versicolor)
_, p2 = shapiro(virginica)
print(f"Versicolor p-value: {p1:.4f}")
print(f"Virginica p-value: {p2:.4f}")

# 독립표본 t-검정
t_stat, p_value = ttest_ind(versicolor, virginica)

print("\n=== 독립표본 t-검정 결과 ===")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.2e}")

if p_value < 0.05:
    print("\n결론: 두 품종의 꽃잎 너비는 통계적으로 유의하게 다르다")
```

---

# 실습 3: Wine Quality ANOVA

## 품질 그룹별 산도 비교

```python
# Wine Quality 데이터 로드
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url, sep=';')

# 품질 그룹 생성 (Low: 3-4, Medium: 5-6, High: 7-8)
wine['quality_group'] = pd.cut(wine['quality'],
                                bins=[2, 4, 6, 8],
                                labels=['Low', 'Medium', 'High'])

# 그룹별 산도 요약
print("=== 품질 그룹별 산도 (volatile acidity) ===")
print(wine.groupby('quality_group')['volatile acidity'].agg(['mean', 'std', 'count']))
```

---

# 실습 3 계속: ANOVA 수행

```python
# 그룹별 데이터 분리
low = wine[wine['quality_group'] == 'Low']['volatile acidity']
medium = wine[wine['quality_group'] == 'Medium']['volatile acidity']
high = wine[wine['quality_group'] == 'High']['volatile acidity']

# 일원분산분석
f_stat, p_value = f_oneway(low, medium, high)

print("\n=== 일원분산분석 (ANOVA) 결과 ===")
print(f"F-통계량: {f_stat:.4f}")
print(f"p-value: {p_value:.2e}")

if p_value < 0.05:
    print("\n결론: 품질 그룹에 따라 산도에 유의한 차이가 있다")
    print("해석: 저품질 와인일수록 휘발성 산도가 높은 경향")
```

---

# 실습 4: Titanic 카이제곱 검정

## 객실 등급과 생존의 관계

```python
# Titanic 데이터 로드
titanic = sns.load_dataset('titanic')

# 교차표: 객실 등급 vs 생존
crosstab = pd.crosstab(titanic['class'], titanic['survived'])
print("=== 객실 등급별 생존 교차표 ===")
print(crosstab)

# 비율 확인
crosstab_pct = pd.crosstab(titanic['class'], titanic['survived'],
                           normalize='index') * 100
print("\n=== 객실 등급별 생존율 (%) ===")
print(crosstab_pct.round(1))
```

---

# 실습 4 계속: 카이제곱 검정 수행

```python
# 카이제곱 독립성 검정
chi2, p_value, dof, expected = chi2_contingency(crosstab)

print("\n=== 카이제곱 검정 결과 ===")
print(f"카이제곱 통계량: {chi2:.4f}")
print(f"자유도: {dof}")
print(f"p-value: {p_value:.2e}")

print("\n기대 빈도:")
print(pd.DataFrame(expected.round(1),
                   index=crosstab.index,
                   columns=['사망', '생존']))

if p_value < 0.05:
    print("\n결론: 객실 등급과 생존 여부는 통계적으로 유의한 관계가 있다")
    print("해석: 1등석 승객의 생존율이 가장 높다")
```

---

# 실습 5: 비모수 검정 비교

## 정규성 위반 시 비모수 검정

```python
# 비정규 분포 데이터 생성
np.random.seed(42)
group_a = np.random.exponential(scale=10, size=30)  # 지수분포
group_b = np.random.exponential(scale=15, size=30)

# 정규성 검정
_, p_a = shapiro(group_a)
_, p_b = shapiro(group_b)
print("=== 정규성 검정 ===")
print(f"Group A: p = {p_a:.4f} {'(정규성 위반)' if p_a < 0.05 else ''}")
print(f"Group B: p = {p_b:.4f} {'(정규성 위반)' if p_b < 0.05 else ''}")
```

---

# 실습 5 계속: 모수 vs 비모수 비교

```python
# 모수 검정 (t-test)
t_stat, t_p = ttest_ind(group_a, group_b)

# 비모수 검정 (Mann-Whitney U)
u_stat, u_p = mannwhitneyu(group_a, group_b)

print("\n=== 검정 결과 비교 ===")
print(f"t-검정: t = {t_stat:.4f}, p = {t_p:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.4f}, p = {u_p:.4f}")

print("\n해석:")
print("- 정규성 위반 시 비모수 검정 결과가 더 신뢰성 있음")
print("- 두 방법의 결론이 다르면 비모수 검정 결과 채택")
```

---

# 검정 선택 가이드

## 상황별 적절한 검정 방법

```
데이터 유형 확인
     │
     ├── 연속형 변수 ─────────────────────────────┐
     │        │                                   │
     │   그룹 수?                             정규분포?
     │        │                                   │
     │    ┌───┴───┐                        ┌──────┴──────┐
     │    2개    3개+                      Yes          No
     │     │      │                         │            │
     │  독립? ANOVA                     모수검정     비모수검정
     │   │                                  │            │
     │ ┌─┴─┐                           t-test     Mann-Whitney
     │ Yes No                          ANOVA      Kruskal-Wallis
     │  │   │
     │ t-ind t-rel
     │
     └── 범주형 변수 → 카이제곱 검정
```

---

# 검정 선택 요약표

## 한눈에 보는 검정 방법

| 상황 | 정규분포 가정 | 비정규분포 |
|------|-------------|-----------|
| 2개 독립 그룹 | `ttest_ind()` | `mannwhitneyu()` |
| 2개 대응 그룹 | `ttest_rel()` | `wilcoxon()` |
| 3개+ 그룹 | `f_oneway()` | `kruskal()` |
| 범주형 vs 범주형 | `chi2_contingency()` | - |

### 판단 순서

1. 변수 유형 확인 (연속형 vs 범주형)
2. 그룹 수 확인 (2개 vs 3개 이상)
3. 정규성 검정 (`shapiro()`)
4. 적절한 검정 선택

---

# 핵심 정리

## 오늘 배운 내용 요약

### Part 1: 가설검정 기초

- H0 (귀무가설): 차이 없다
- H1 (대립가설): 차이 있다
- p-value < 0.05 → 통계적으로 유의

### Part 2: 모수 검정

- t-검정: 두 그룹 평균 비교
- ANOVA: 3개 이상 그룹 비교

---

# 핵심 정리 (계속)

## 오늘 배운 내용 요약

### Part 3: 범주형 검정

- 카이제곱 검정: 범주형 변수 간 관계
- 교차표 작성 → chi2_contingency()

### Part 4: 비모수 검정

- 정규성 위반 시 사용
- Mann-Whitney U, Wilcoxon, Kruskal-Wallis

---

# 자주 하는 실수

## 주의사항 정리

| 실수 | 올바른 접근 |
|------|------------|
| p=0.06을 "거의 유의" | 유의하지 않음 (기준 준수) |
| p값만 보고 판단 | 효과 크기도 함께 확인 |
| 여러 번 검정 후 유의한 것만 보고 | 다중비교 보정 필요 |
| 표본 작은데 모수 검정 | 비모수 검정 고려 |
| ANOVA 유의하면 끝 | 사후분석으로 어느 그룹인지 확인 |

---

# 제조업 적용 예시

## 실무 활용 시나리오

| 상황 | 적용 검정 |
|------|----------|
| A/B라인 불량률 비교 | 독립표본 t-검정 |
| 공정 개선 전후 비교 | 대응표본 t-검정 |
| 3개 교대조 생산성 비교 | ANOVA |
| 라인별 불량 유형 분포 | 카이제곱 검정 |
| 소량 샘플 품질 비교 | Mann-Whitney U |

---

# 다음 차시 예고

## 8차시: 상관분석과 예측의 기초

### 학습 내용

- **상관계수**의 의미와 해석
- **단순선형회귀**의 개념
- sklearn으로 **예측 모델** 구현

### 준비물

- 오늘 배운 통계 검정 복습
- scipy, sklearn 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심 코드

```python
from scipy import stats

# t-검정
t, p = stats.ttest_ind(group1, group2)

# ANOVA
f, p = stats.f_oneway(g1, g2, g3)

# 카이제곱
chi2, p, dof, exp = stats.chi2_contingency(crosstab)

# 비모수 검정
u, p = stats.mannwhitneyu(g1, g2)
```

**p < 0.05 → 통계적으로 유의함**

---

# 감사합니다

## 7차시: 통계 검정 실습

**다음 시간에 상관분석과 예측을 배워봅시다!**

