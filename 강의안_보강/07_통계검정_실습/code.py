"""
7차시: 통계 검정 실습 (Statistical Hypothesis Testing)
=======================================================

학습 목표:
1. 가설검정의 기본 개념과 p-value 이해
2. 모수 검정 (t-검정, ANOVA) 수행
3. 범주형 검정 (카이제곱 검정) 수행
4. 비모수 검정 이해 및 적용
5. 상황에 맞는 검정 방법 선택

실습 데이터:
- Iris 데이터셋 (꽃 품종별 특성)
- Titanic 데이터셋 (생존 여부 분석)
- Wine Quality 데이터셋 (와인 품질 분석)
"""

# =============================================================================
# 1. 환경 설정 및 데이터 로드
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("7차시: 통계 검정 실습")
print("=" * 70)

# -----------------------------------------------------------------------------
# 데이터셋 로드
# -----------------------------------------------------------------------------

# 1) Iris 데이터셋 (붓꽃 데이터)
from sklearn.datasets import load_iris
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=['sepal_length', 'sepal_width',
                                              'petal_length', 'petal_width'])
iris['species'] = pd.Categorical.from_codes(iris_data.target,
                                             ['setosa', 'versicolor', 'virginica'])

print("\n[1] Iris 데이터셋 로드 완료")
print(f"    - 샘플 수: {len(iris)}")
print(f"    - 품종: {iris['species'].unique().tolist()}")

# 2) Titanic 데이터셋 (타이타닉 생존 데이터)
titanic = sns.load_dataset('titanic')
print("\n[2] Titanic 데이터셋 로드 완료")
print(f"    - 샘플 수: {len(titanic)}")
print(f"    - 주요 변수: sex, survived, pclass, age, fare")

# 3) Wine Quality 데이터셋 (와인 품질 데이터)
wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(wine_url, sep=';')
print("\n[3] Wine Quality 데이터셋 로드 완료")
print(f"    - 샘플 수: {len(wine)}")
print(f"    - 품질 등급: {sorted(wine['quality'].unique())}")

# -----------------------------------------------------------------------------
# 데이터 기본 탐색
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("데이터셋 기본 정보")
print("=" * 70)

print("\n[Iris 데이터 - 처음 5행]")
print(iris.head())

print("\n[Titanic 데이터 - 처음 5행]")
print(titanic[['survived', 'pclass', 'sex', 'age', 'fare']].head())

print("\n[Wine Quality 데이터 - 처음 5행]")
print(wine.head())

"""
예상 출력:
======================================================================
7차시: 통계 검정 실습
======================================================================

[1] Iris 데이터셋 로드 완료
    - 샘플 수: 150
    - 품종: ['setosa', 'versicolor', 'virginica']

[2] Titanic 데이터셋 로드 완료
    - 샘플 수: 891
    - 주요 변수: sex, survived, pclass, age, fare

[3] Wine Quality 데이터셋 로드 완료
    - 샘플 수: 1599
    - 품질 등급: [3, 4, 5, 6, 7, 8]
"""


# =============================================================================
# Part 1: 가설검정 기초
# =============================================================================

print("\n" + "=" * 70)
print("Part 1: 가설검정 기초")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 가설검정의 개념
# -----------------------------------------------------------------------------

"""
가설검정(Hypothesis Testing)이란?
- 표본 데이터를 사용하여 모집단에 대한 주장(가설)을 검증하는 통계적 방법

핵심 개념:
1. 귀무가설(H0): "차이가 없다", "효과가 없다" (기각하고 싶은 가설)
2. 대립가설(H1): "차이가 있다", "효과가 있다" (증명하고 싶은 가설)
3. p-value: 귀무가설이 참일 때, 관측된 데이터가 나올 확률
4. 유의수준(α): 보통 0.05 (5%), 귀무가설 기각의 기준

판단 기준:
- p-value < α (0.05) → 귀무가설 기각 → "통계적으로 유의한 차이가 있다"
- p-value ≥ α (0.05) → 귀무가설 채택 → "통계적으로 유의한 차이가 없다"
"""

print("\n[가설검정의 핵심 개념]")
print("-" * 50)
print("귀무가설(H0): '차이가 없다' - 기각하고 싶은 가설")
print("대립가설(H1): '차이가 있다' - 증명하고 싶은 가설")
print("p-value: 귀무가설이 참일 때 관측 데이터가 나올 확률")
print("유의수준(α): 0.05 (5%) - 귀무가설 기각의 기준")
print("-" * 50)
print("판단: p < 0.05 → 유의한 차이 있음")
print("      p ≥ 0.05 → 유의한 차이 없음")

# -----------------------------------------------------------------------------
# 1.2 p-value 개념 이해 (실제 데이터로 시연)
# -----------------------------------------------------------------------------

print("\n\n[p-value 개념 시연 - Iris 데이터]")
print("-" * 50)

# Iris에서 setosa와 versicolor의 꽃잎 길이(petal_length) 비교
setosa_petal = iris[iris['species'] == 'setosa']['petal_length']
versicolor_petal = iris[iris['species'] == 'versicolor']['petal_length']

print(f"\nSetosa 꽃잎 길이:")
print(f"  - 평균: {setosa_petal.mean():.2f} cm")
print(f"  - 표준편차: {setosa_petal.std():.2f} cm")
print(f"  - 샘플 수: {len(setosa_petal)}")

print(f"\nVersicolor 꽃잎 길이:")
print(f"  - 평균: {versicolor_petal.mean():.2f} cm")
print(f"  - 표준편차: {versicolor_petal.std():.2f} cm")
print(f"  - 샘플 수: {len(versicolor_petal)}")

# 평균 차이
mean_diff = versicolor_petal.mean() - setosa_petal.mean()
print(f"\n평균 차이: {mean_diff:.2f} cm")

# t-검정 수행
t_stat, p_value = stats.ttest_ind(setosa_petal, versicolor_petal)

print(f"\nt-검정 결과:")
print(f"  - t 통계량: {t_stat:.4f}")
print(f"  - p-value: {p_value:.2e} (매우 작은 값)")

# 해석
alpha = 0.05
if p_value < alpha:
    print(f"\n[결론] p-value({p_value:.2e}) < α({alpha})")
    print("       → 귀무가설 기각: 두 품종의 꽃잎 길이에 통계적으로 유의한 차이가 있습니다.")
else:
    print(f"\n[결론] p-value({p_value:.4f}) ≥ α({alpha})")
    print("       → 귀무가설 채택: 두 품종의 꽃잎 길이에 유의한 차이가 없습니다.")

"""
예상 출력:
[p-value 개념 시연 - Iris 데이터]
--------------------------------------------------

Setosa 꽃잎 길이:
  - 평균: 1.46 cm
  - 표준편차: 0.17 cm
  - 샘플 수: 50

Versicolor 꽃잎 길이:
  - 평균: 4.26 cm
  - 표준편차: 0.47 cm
  - 샘플 수: 50

평균 차이: 2.80 cm

t-검정 결과:
  - t 통계량: -39.4929
  - p-value: 2.86e-62 (매우 작은 값)

[결론] p-value(2.86e-62) < α(0.05)
       → 귀무가설 기각: 두 품종의 꽃잎 길이에 통계적으로 유의한 차이가 있습니다.
"""


# =============================================================================
# Part 2: 모수 검정 (Parametric Tests)
# =============================================================================

print("\n\n" + "=" * 70)
print("Part 2: 모수 검정 (Parametric Tests)")
print("=" * 70)

"""
모수 검정의 가정:
1. 정규성: 데이터가 정규분포를 따름
2. 등분산성: 비교 그룹들의 분산이 동일
3. 독립성: 관측치들이 서로 독립

주요 모수 검정:
- 독립표본 t-검정: 두 독립 그룹의 평균 비교
- 대응표본 t-검정: 같은 대상의 전/후 비교
- ANOVA (분산분석): 세 개 이상 그룹의 평균 비교
"""

# -----------------------------------------------------------------------------
# 2.1 독립표본 t-검정 (Independent Samples t-test)
# -----------------------------------------------------------------------------

print("\n[2.1] 독립표본 t-검정")
print("-" * 50)
print("목적: 두 독립 그룹의 평균 차이 검정")
print("예제: Iris - Setosa vs Versicolor의 꽃잎 길이 비교")

# 이미 위에서 데이터 준비됨
# setosa_petal, versicolor_petal

# 정규성 검정 (Shapiro-Wilk test)
print("\n[Step 1] 정규성 검정 (Shapiro-Wilk test)")
_, p_setosa = stats.shapiro(setosa_petal)
_, p_versi = stats.shapiro(versicolor_petal)

print(f"  Setosa p-value: {p_setosa:.4f}", end="")
print(" → 정규분포 따름" if p_setosa >= 0.05 else " → 정규분포 아님")
print(f"  Versicolor p-value: {p_versi:.4f}", end="")
print(" → 정규분포 따름" if p_versi >= 0.05 else " → 정규분포 아님")

# 등분산성 검정 (Levene's test)
print("\n[Step 2] 등분산성 검정 (Levene's test)")
_, p_levene = stats.levene(setosa_petal, versicolor_petal)
print(f"  Levene p-value: {p_levene:.4f}", end="")
print(" → 등분산" if p_levene >= 0.05 else " → 이분산")

# 독립표본 t-검정 수행
print("\n[Step 3] 독립표본 t-검정 (scipy.stats.ttest_ind)")

# equal_var 파라미터: True=등분산 가정, False=이분산 가정(Welch's t-test)
t_stat, p_value = stats.ttest_ind(setosa_petal, versicolor_petal,
                                   equal_var=(p_levene >= 0.05))

print(f"\n  가설 설정:")
print(f"    H0: μ_setosa = μ_versicolor (평균 차이 없음)")
print(f"    H1: μ_setosa ≠ μ_versicolor (평균 차이 있음)")

print(f"\n  결과:")
print(f"    t 통계량: {t_stat:.4f}")
print(f"    p-value: {p_value:.2e}")

print(f"\n  결론:", end=" ")
if p_value < 0.05:
    print("통계적으로 유의한 차이가 있습니다 (p < 0.05)")
    print(f"         Setosa(평균 {setosa_petal.mean():.2f}cm)와")
    print(f"         Versicolor(평균 {versicolor_petal.mean():.2f}cm)의")
    print(f"         꽃잎 길이는 유의하게 다릅니다.")
else:
    print("통계적으로 유의한 차이가 없습니다 (p ≥ 0.05)")

# -----------------------------------------------------------------------------
# 2.2 Titanic 데이터로 t-검정 추가 예제
# -----------------------------------------------------------------------------

print("\n\n[추가 예제] Titanic - 생존자 vs 사망자의 요금(fare) 비교")
print("-" * 50)

survived_fare = titanic[titanic['survived'] == 1]['fare'].dropna()
died_fare = titanic[titanic['survived'] == 0]['fare'].dropna()

print(f"\n생존자 요금: 평균 {survived_fare.mean():.2f}, 중앙값 {survived_fare.median():.2f}")
print(f"사망자 요금: 평균 {died_fare.mean():.2f}, 중앙값 {died_fare.median():.2f}")

t_stat, p_value = stats.ttest_ind(survived_fare, died_fare, equal_var=False)
print(f"\nWelch's t-검정 결과:")
print(f"  t 통계량: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n  결론: 생존자와 사망자의 요금에 유의한 차이가 있습니다.")
    print("        (비싼 티켓 구매자의 생존율이 더 높았을 수 있음)")

# -----------------------------------------------------------------------------
# 2.3 대응표본 t-검정 (Paired Samples t-test)
# -----------------------------------------------------------------------------

print("\n\n[2.2] 대응표본 t-검정")
print("-" * 50)
print("목적: 같은 대상의 전/후 측정값 비교")
print("예제: Iris - 같은 꽃의 sepal_length vs petal_length 비교")

"""
대응표본 t-검정 설명:
- 동일한 대상을 두 번 측정한 경우 사용
- 예: 약 복용 전/후, 교육 전/후 등
- Iris 예: 같은 꽃의 서로 다른 부위 길이 비교 (개념 이해용)
"""

# Iris 전체 데이터에서 sepal_length와 petal_length 비교
# (실제로는 전/후 데이터가 이상적이지만, 개념 이해용으로 활용)
sepal = iris['sepal_length'].values
petal = iris['petal_length'].values

print(f"\n꽃받침(sepal) 길이: 평균 {sepal.mean():.2f} cm")
print(f"꽃잎(petal) 길이: 평균 {petal.mean():.2f} cm")
print(f"차이: 평균 {(sepal - petal).mean():.2f} cm")

# 대응표본 t-검정
t_stat, p_value = stats.ttest_rel(sepal, petal)

print(f"\n대응표본 t-검정 (scipy.stats.ttest_rel) 결과:")
print(f"  t 통계량: {t_stat:.4f}")
print(f"  p-value: {p_value:.2e}")

if p_value < 0.05:
    print("\n  결론: 꽃받침과 꽃잎의 길이에 유의한 차이가 있습니다.")

# -----------------------------------------------------------------------------
# 2.4 Wine Quality - 대응표본 t-검정 예제
# -----------------------------------------------------------------------------

print("\n\n[추가 예제] Wine - fixed acidity vs volatile acidity")
print("-" * 50)

fixed_acid = wine['fixed acidity'].values
volatile_acid = wine['volatile acidity'].values

# 스케일이 다르므로 표준화 후 비교
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fixed_scaled = scaler.fit_transform(fixed_acid.reshape(-1, 1)).flatten()
volatile_scaled = scaler.fit_transform(volatile_acid.reshape(-1, 1)).flatten()

print(f"Fixed acidity (표준화): 평균 {fixed_scaled.mean():.4f}")
print(f"Volatile acidity (표준화): 평균 {volatile_scaled.mean():.4f}")

t_stat, p_value = stats.ttest_rel(fixed_scaled, volatile_scaled)
print(f"\n대응표본 t-검정 결과:")
print(f"  t 통계량: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

# -----------------------------------------------------------------------------
# 2.5 일원분산분석 (One-way ANOVA)
# -----------------------------------------------------------------------------

print("\n\n[2.3] 일원분산분석 (One-way ANOVA)")
print("-" * 50)
print("목적: 세 개 이상 그룹의 평균 차이 검정")
print("예제: Wine Quality - 품질 등급별 알코올 함량 비교")

"""
ANOVA (Analysis of Variance):
- 세 개 이상의 그룹 평균을 동시에 비교
- t-검정을 여러 번 하면 오류율이 증가하므로 ANOVA 사용
- F 통계량 사용

가설:
- H0: μ1 = μ2 = μ3 = ... (모든 그룹 평균 동일)
- H1: 적어도 하나의 그룹 평균이 다름
"""

# Wine Quality 데이터에서 품질 등급별 그룹 생성
# 품질 등급: 3, 4, 5, 6, 7, 8
quality_groups = []
quality_labels = sorted(wine['quality'].unique())

print(f"\n품질 등급별 알코올 함량:")
for q in quality_labels:
    group = wine[wine['quality'] == q]['alcohol']
    quality_groups.append(group)
    print(f"  등급 {q}: 평균 {group.mean():.2f}%, 샘플 수 {len(group)}")

# One-way ANOVA
f_stat, p_value = stats.f_oneway(*quality_groups)

print(f"\n가설 설정:")
print(f"  H0: 모든 품질 등급의 알코올 함량 평균이 동일")
print(f"  H1: 적어도 하나의 품질 등급의 알코올 함량이 다름")

print(f"\nOne-way ANOVA (scipy.stats.f_oneway) 결과:")
print(f"  F 통계량: {f_stat:.4f}")
print(f"  p-value: {p_value:.2e}")

if p_value < 0.05:
    print("\n  결론: 품질 등급에 따라 알코올 함량에 유의한 차이가 있습니다.")
    print("        → 사후검정(Post-hoc test)으로 어떤 그룹 간 차이인지 확인 필요")

# -----------------------------------------------------------------------------
# 2.6 Tukey HSD 사후검정
# -----------------------------------------------------------------------------

print("\n\n[2.4] Tukey HSD 사후검정")
print("-" * 50)
print("목적: ANOVA에서 유의한 결과 후, 어떤 그룹 간 차이가 있는지 확인")

from scipy.stats import tukey_hsd

# Tukey HSD 수행
result = tukey_hsd(*quality_groups)

print("\nTukey HSD 결과 (일부 주요 비교):")
print("-" * 40)

# 결과 출력 (품질 등급 간 비교)
for i in range(len(quality_labels)):
    for j in range(i+1, len(quality_labels)):
        # pvalue 추출
        p_val = result.pvalue[i, j]
        if p_val < 0.05:
            print(f"등급 {quality_labels[i]} vs {quality_labels[j]}: p = {p_val:.4f} *")

print("\n* 표시: 통계적으로 유의한 차이 (p < 0.05)")

# 또는 statsmodels 사용 (더 상세한 출력)
print("\n[statsmodels를 사용한 Tukey HSD - 더 상세한 결과]")
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(wine['alcohol'], wine['quality'], alpha=0.05)
print(tukey.summary())

"""
예상 출력 (일부):
Multiple Comparison of Means - Tukey HSD, FWER=0.05
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
     3      4   0.2958 0.9887 -1.0847  1.6763  False
     3      5   0.0788 1.0000 -1.1318  1.2893  False
     5      6   0.5785 0.0000  0.3809  0.7762   True
...
"""


# =============================================================================
# Part 3: 범주형 검정 (Categorical Tests)
# =============================================================================

print("\n\n" + "=" * 70)
print("Part 3: 범주형 검정 (Categorical Tests)")
print("=" * 70)

"""
범주형 검정:
- 범주형 변수들 간의 관계(독립성/연관성) 검정
- 카이제곱 검정 (Chi-square test)이 대표적
"""

# -----------------------------------------------------------------------------
# 3.1 카이제곱 검정 (Chi-square Test)
# -----------------------------------------------------------------------------

print("\n[3.1] 카이제곱 검정 (Chi-square Test)")
print("-" * 50)
print("목적: 두 범주형 변수 간의 독립성 검정")
print("예제: Titanic - 성별(sex)과 생존여부(survived)의 관계")

"""
카이제곱 검정 설명:
- 관측 빈도와 기대 빈도의 차이를 검정
- 기대 빈도: 두 변수가 독립일 때 예상되는 빈도

가설:
- H0: 두 변수는 독립 (관련 없음)
- H1: 두 변수는 독립이 아님 (관련 있음)
"""

# 교차표(Cross-tabulation) 생성
print("\n[Step 1] 교차표(Contingency Table) 생성")
contingency_table = pd.crosstab(titanic['sex'], titanic['survived'],
                                 margins=True, margins_name='Total')
# 열 이름 변경
contingency_table.columns = ['사망(0)', '생존(1)', 'Total']
print(contingency_table)

# 비율 확인
print("\n[Step 2] 성별별 생존율")
survival_rate = titanic.groupby('sex')['survived'].mean() * 100
print(f"  여성 생존율: {survival_rate['female']:.1f}%")
print(f"  남성 생존율: {survival_rate['male']:.1f}%")

# 카이제곱 검정 수행
print("\n[Step 3] 카이제곱 검정 (scipy.stats.chi2_contingency)")

# margins 제외한 교차표
ct = pd.crosstab(titanic['sex'], titanic['survived'])
chi2, p_value, dof, expected = stats.chi2_contingency(ct)

print(f"\n  가설 설정:")
print(f"    H0: 성별과 생존여부는 독립 (관련 없음)")
print(f"    H1: 성별과 생존여부는 독립이 아님 (관련 있음)")

print(f"\n  결과:")
print(f"    카이제곱 통계량: {chi2:.4f}")
print(f"    자유도(df): {dof}")
print(f"    p-value: {p_value:.2e}")

print(f"\n  기대 빈도표:")
expected_df = pd.DataFrame(expected,
                           index=['female', 'male'],
                           columns=['사망(0)', '생존(1)'])
print(expected_df.round(2))

if p_value < 0.05:
    print(f"\n  결론: 성별과 생존여부는 통계적으로 유의한 관련이 있습니다.")
    print("        → 여성의 생존율이 남성보다 유의하게 높았습니다.")

# -----------------------------------------------------------------------------
# 3.2 Titanic - 객실 등급과 생존 관계
# -----------------------------------------------------------------------------

print("\n\n[추가 예제] Titanic - 객실 등급(pclass)과 생존여부의 관계")
print("-" * 50)

ct_class = pd.crosstab(titanic['pclass'], titanic['survived'])
print("\n교차표:")
print(ct_class)

chi2, p_value, dof, expected = stats.chi2_contingency(ct_class)
print(f"\n카이제곱 검정 결과:")
print(f"  카이제곱 통계량: {chi2:.4f}")
print(f"  p-value: {p_value:.2e}")

print(f"\n등급별 생존율:")
for pclass in [1, 2, 3]:
    rate = titanic[titanic['pclass'] == pclass]['survived'].mean() * 100
    print(f"  {pclass}등급: {rate:.1f}%")

if p_value < 0.05:
    print(f"\n결론: 객실 등급과 생존여부는 유의한 관련이 있습니다.")
    print("      → 상위 등급 승객의 생존율이 더 높았습니다.")

"""
예상 출력:
교차표:
survived    0    1
pclass
1          80  136
2          97   87
3         372  119

카이제곱 검정 결과:
  카이제곱 통계량: 102.8890
  p-value: 4.55e-23

등급별 생존율:
  1등급: 62.96%
  2등급: 47.28%
  3등급: 24.24%
"""


# =============================================================================
# Part 4: 비모수 검정 (Non-parametric Tests)
# =============================================================================

print("\n\n" + "=" * 70)
print("Part 4: 비모수 검정 (Non-parametric Tests)")
print("=" * 70)

"""
비모수 검정을 사용하는 경우:
1. 데이터가 정규분포를 따르지 않음
2. 표본 크기가 작음 (n < 30)
3. 순위 데이터를 다룸
4. 이상치의 영향을 줄이고 싶음

주요 비모수 검정:
- Mann-Whitney U 검정: 독립표본 t-검정의 비모수 버전
- Wilcoxon 부호순위 검정: 대응표본 t-검정의 비모수 버전
- Kruskal-Wallis 검정: ANOVA의 비모수 버전
"""

# -----------------------------------------------------------------------------
# 4.1 정규성 검정 (Shapiro-Wilk Test)
# -----------------------------------------------------------------------------

print("\n[4.1] 정규성 검정 (Shapiro-Wilk Test)")
print("-" * 50)
print("목적: 데이터가 정규분포를 따르는지 검정")

"""
Shapiro-Wilk 검정:
- H0: 데이터가 정규분포를 따름
- H1: 데이터가 정규분포를 따르지 않음
- p < 0.05: 정규분포가 아님 → 비모수 검정 사용 권장
"""

# Titanic fare 데이터의 정규성 검정
fare_data = titanic['fare'].dropna()

# 샘플 크기가 5000 초과시 오류 발생하므로 일부 샘플링
sample_fare = fare_data.sample(min(len(fare_data), 500), random_state=42)

stat, p_value = stats.shapiro(sample_fare)

print(f"\nTitanic 요금(fare) 정규성 검정:")
print(f"  표본 크기: {len(sample_fare)}")
print(f"  Shapiro-Wilk 통계량: {stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"\n  결론: 요금 데이터는 정규분포를 따르지 않습니다 (p < 0.05)")
    print("        → 비모수 검정 사용 권장")
else:
    print(f"\n  결론: 요금 데이터는 정규분포를 따릅니다 (p ≥ 0.05)")

# Wine Quality 데이터 정규성 검정
print(f"\n\nWine Quality 주요 변수 정규성 검정:")
print("-" * 40)

wine_vars = ['alcohol', 'pH', 'residual sugar', 'quality']
for var in wine_vars:
    sample = wine[var].sample(min(len(wine), 500), random_state=42)
    stat, p_val = stats.shapiro(sample)
    normal = "정규분포" if p_val >= 0.05 else "비정규분포"
    print(f"  {var}: p = {p_val:.4f} → {normal}")

# -----------------------------------------------------------------------------
# 4.2 Mann-Whitney U 검정
# -----------------------------------------------------------------------------

print("\n\n[4.2] Mann-Whitney U 검정")
print("-" * 50)
print("목적: 두 독립 그룹의 분포 비교 (비모수)")
print("예제: Titanic - 생존자 vs 사망자의 요금 비교")

"""
Mann-Whitney U 검정:
- 독립표본 t-검정의 비모수 대안
- 순위를 기반으로 검정
- 정규성 가정 불필요
"""

survived_fare = titanic[titanic['survived'] == 1]['fare'].dropna()
died_fare = titanic[titanic['survived'] == 0]['fare'].dropna()

# Mann-Whitney U 검정
stat, p_value = stats.mannwhitneyu(survived_fare, died_fare, alternative='two-sided')

print(f"\n생존자 요금: 중앙값 {survived_fare.median():.2f}")
print(f"사망자 요금: 중앙값 {died_fare.median():.2f}")

print(f"\nMann-Whitney U 검정 (scipy.stats.mannwhitneyu) 결과:")
print(f"  U 통계량: {stat:.2f}")
print(f"  p-value: {p_value:.4e}")

if p_value < 0.05:
    print(f"\n  결론: 생존자와 사망자의 요금 분포에 유의한 차이가 있습니다.")

# Iris 데이터로 Mann-Whitney U 검정
print("\n\n[추가 예제] Iris - Setosa vs Virginica 꽃잎 너비")
print("-" * 40)

setosa_width = iris[iris['species'] == 'setosa']['petal_width']
virginica_width = iris[iris['species'] == 'virginica']['petal_width']

stat, p_value = stats.mannwhitneyu(setosa_width, virginica_width)
print(f"Setosa 중앙값: {setosa_width.median():.2f} cm")
print(f"Virginica 중앙값: {virginica_width.median():.2f} cm")
print(f"Mann-Whitney U p-value: {p_value:.2e}")

# -----------------------------------------------------------------------------
# 4.3 Wilcoxon 부호순위 검정
# -----------------------------------------------------------------------------

print("\n\n[4.3] Wilcoxon 부호순위 검정")
print("-" * 50)
print("목적: 짝지어진 두 측정값의 차이 검정 (비모수)")
print("예제: Wine - fixed acidity vs citric acid (표준화 후)")

"""
Wilcoxon 부호순위 검정:
- 대응표본 t-검정의 비모수 대안
- 차이의 순위를 기반으로 검정
"""

# Wine 데이터에서 두 산도 변수 비교
fixed = wine['fixed acidity'].values
citric = wine['citric acid'].values

# 스케일 맞추기 (min-max 정규화)
fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min())
citric_norm = (citric - citric.min()) / (citric.max() - citric.min())

# Wilcoxon 부호순위 검정
stat, p_value = stats.wilcoxon(fixed_norm, citric_norm)

print(f"\nFixed acidity (정규화): 평균 {fixed_norm.mean():.4f}")
print(f"Citric acid (정규화): 평균 {citric_norm.mean():.4f}")

print(f"\nWilcoxon 부호순위 검정 (scipy.stats.wilcoxon) 결과:")
print(f"  통계량: {stat:.2f}")
print(f"  p-value: {p_value:.4e}")

if p_value < 0.05:
    print(f"\n  결론: 두 산도 지표 간에 유의한 차이가 있습니다.")

# -----------------------------------------------------------------------------
# 4.4 Kruskal-Wallis 검정
# -----------------------------------------------------------------------------

print("\n\n[4.4] Kruskal-Wallis 검정")
print("-" * 50)
print("목적: 세 개 이상 그룹의 분포 비교 (비모수)")
print("예제: Wine Quality - 품질 등급별 알코올 함량")

"""
Kruskal-Wallis 검정:
- ANOVA의 비모수 대안
- 세 개 이상 그룹의 순위 기반 비교
"""

# 품질 등급별 알코올 데이터
quality_groups = [wine[wine['quality'] == q]['alcohol']
                  for q in sorted(wine['quality'].unique())]

# Kruskal-Wallis 검정
stat, p_value = stats.kruskal(*quality_groups)

print(f"\nKruskal-Wallis 검정 (scipy.stats.kruskal) 결과:")
print(f"  H 통계량: {stat:.4f}")
print(f"  p-value: {p_value:.2e}")

if p_value < 0.05:
    print(f"\n  결론: 품질 등급에 따라 알코올 함량 분포에 유의한 차이가 있습니다.")

# Iris 데이터로 Kruskal-Wallis 검정
print("\n\n[추가 예제] Iris - 3품종의 꽃잎 길이 비교")
print("-" * 40)

setosa_pl = iris[iris['species'] == 'setosa']['petal_length']
versicolor_pl = iris[iris['species'] == 'versicolor']['petal_length']
virginica_pl = iris[iris['species'] == 'virginica']['petal_length']

stat, p_value = stats.kruskal(setosa_pl, versicolor_pl, virginica_pl)
print(f"Kruskal-Wallis H 통계량: {stat:.4f}")
print(f"p-value: {p_value:.2e}")

print("\n품종별 꽃잎 길이 중앙값:")
for species in ['setosa', 'versicolor', 'virginica']:
    median = iris[iris['species'] == species]['petal_length'].median()
    print(f"  {species}: {median:.2f} cm")


# =============================================================================
# 5. 종합 분석 함수
# =============================================================================

print("\n\n" + "=" * 70)
print("5. 종합 분석 함수")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 적절한 검정 방법 자동 선택 함수
# -----------------------------------------------------------------------------

def choose_statistical_test(data1, data2=None, paired=False, groups=None,
                            categorical=False, alpha=0.05):
    """
    데이터 특성에 따라 적절한 통계 검정 방법을 자동으로 선택하고 수행합니다.

    Parameters:
    -----------
    data1 : array-like
        첫 번째 데이터 또는 독립변수
    data2 : array-like, optional
        두 번째 데이터 (두 그룹 비교시)
    paired : bool
        대응표본 여부 (기본값: False)
    groups : list of array-like, optional
        세 개 이상의 그룹 비교시 그룹 리스트
    categorical : bool
        범주형 데이터 여부 (기본값: False)
    alpha : float
        유의수준 (기본값: 0.05)

    Returns:
    --------
    dict : 검정 결과 딕셔너리
    """

    result = {
        'test_name': None,
        'statistic': None,
        'p_value': None,
        'significant': None,
        'interpretation': None,
        'recommendation': None
    }

    # 1. 범주형 데이터인 경우 → 카이제곱 검정
    if categorical:
        if isinstance(data1, pd.DataFrame):
            chi2, p, dof, expected = stats.chi2_contingency(data1)
        else:
            ct = pd.crosstab(data1, data2)
            chi2, p, dof, expected = stats.chi2_contingency(ct)

        result['test_name'] = '카이제곱 검정 (Chi-square test)'
        result['statistic'] = chi2
        result['p_value'] = p
        result['significant'] = p < alpha
        result['interpretation'] = '유의한 연관성 있음' if p < alpha else '유의한 연관성 없음'
        return result

    # 2. 세 개 이상 그룹 비교
    if groups is not None:
        # 정규성 검정 (첫 번째 그룹으로 대표)
        _, p_normal = stats.shapiro(groups[0][:min(len(groups[0]), 500)])

        if p_normal >= alpha:
            # 정규분포 → ANOVA
            f_stat, p = stats.f_oneway(*groups)
            result['test_name'] = '일원분산분석 (One-way ANOVA)'
            result['statistic'] = f_stat
        else:
            # 비정규분포 → Kruskal-Wallis
            h_stat, p = stats.kruskal(*groups)
            result['test_name'] = 'Kruskal-Wallis 검정'
            result['statistic'] = h_stat

        result['p_value'] = p
        result['significant'] = p < alpha
        result['interpretation'] = '그룹 간 유의한 차이 있음' if p < alpha else '그룹 간 유의한 차이 없음'
        return result

    # 3. 두 그룹 비교
    if data2 is not None:
        # 정규성 검정
        _, p1 = stats.shapiro(data1[:min(len(data1), 500)])
        _, p2 = stats.shapiro(data2[:min(len(data2), 500)])
        normal = (p1 >= alpha) and (p2 >= alpha)

        if paired:
            # 대응표본
            if normal:
                t_stat, p = stats.ttest_rel(data1, data2)
                result['test_name'] = '대응표본 t-검정 (Paired t-test)'
                result['statistic'] = t_stat
            else:
                stat, p = stats.wilcoxon(data1, data2)
                result['test_name'] = 'Wilcoxon 부호순위 검정'
                result['statistic'] = stat
        else:
            # 독립표본
            if normal:
                # 등분산성 검정
                _, p_levene = stats.levene(data1, data2)
                t_stat, p = stats.ttest_ind(data1, data2, equal_var=(p_levene >= alpha))
                test_type = "등분산" if p_levene >= alpha else "이분산(Welch's)"
                result['test_name'] = f'독립표본 t-검정 ({test_type})'
                result['statistic'] = t_stat
            else:
                stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                result['test_name'] = 'Mann-Whitney U 검정'
                result['statistic'] = stat

        result['p_value'] = p
        result['significant'] = p < alpha
        result['interpretation'] = '유의한 차이 있음' if p < alpha else '유의한 차이 없음'
        return result

    return result


# 함수 테스트
print("\n[5.1] 자동 검정 선택 함수 테스트")
print("-" * 50)

# 테스트 1: Iris - 두 품종 비교
print("\n테스트 1: Iris Setosa vs Versicolor 꽃잎 길이")
result = choose_statistical_test(
    iris[iris['species'] == 'setosa']['petal_length'].values,
    iris[iris['species'] == 'versicolor']['petal_length'].values
)
print(f"  선택된 검정: {result['test_name']}")
print(f"  p-value: {result['p_value']:.4e}")
print(f"  결론: {result['interpretation']}")

# 테스트 2: Wine - 세 그룹 비교
print("\n테스트 2: Wine Quality (Low/Medium/High) 알코올 비교")
wine_low = wine[wine['quality'] <= 4]['alcohol'].values
wine_mid = wine[(wine['quality'] >= 5) & (wine['quality'] <= 6)]['alcohol'].values
wine_high = wine[wine['quality'] >= 7]['alcohol'].values

result = choose_statistical_test(groups=[wine_low, wine_mid, wine_high])
print(f"  선택된 검정: {result['test_name']}")
print(f"  p-value: {result['p_value']:.4e}")
print(f"  결론: {result['interpretation']}")

# 테스트 3: Titanic - 범주형
print("\n테스트 3: Titanic 성별과 생존 관계 (범주형)")
result = choose_statistical_test(titanic['sex'], titanic['survived'], categorical=True)
print(f"  선택된 검정: {result['test_name']}")
print(f"  p-value: {result['p_value']:.4e}")
print(f"  결론: {result['interpretation']}")

# -----------------------------------------------------------------------------
# 5.2 종합 분석 리포트 생성 함수
# -----------------------------------------------------------------------------

def generate_analysis_report(data, group_var, value_var, title="분석 리포트"):
    """
    그룹별 비교에 대한 종합 분석 리포트를 생성합니다.

    Parameters:
    -----------
    data : DataFrame
        분석할 데이터프레임
    group_var : str
        그룹 변수명
    value_var : str
        분석할 수치 변수명
    title : str
        리포트 제목
    """

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    # 1. 기술통계
    print(f"\n[1] 기술통계: {value_var} by {group_var}")
    print("-" * 50)

    groups = data[group_var].unique()
    group_data = []

    for g in sorted(groups):
        subset = data[data[group_var] == g][value_var].dropna()
        group_data.append(subset)
        print(f"  {g}:")
        print(f"    - N: {len(subset)}")
        print(f"    - 평균: {subset.mean():.4f}")
        print(f"    - 중앙값: {subset.median():.4f}")
        print(f"    - 표준편차: {subset.std():.4f}")
        print(f"    - 범위: {subset.min():.4f} ~ {subset.max():.4f}")

    # 2. 정규성 검정
    print(f"\n[2] 정규성 검정 (Shapiro-Wilk)")
    print("-" * 50)

    normality_results = []
    for i, g in enumerate(sorted(groups)):
        sample = group_data[i].values[:500]  # 최대 500개 샘플
        _, p_val = stats.shapiro(sample)
        is_normal = p_val >= 0.05
        normality_results.append(is_normal)
        status = "정규분포" if is_normal else "비정규분포"
        print(f"  {g}: p = {p_val:.4f} → {status}")

    all_normal = all(normality_results)

    # 3. 통계 검정
    print(f"\n[3] 통계 검정")
    print("-" * 50)

    if len(groups) == 2:
        # 두 그룹 비교
        if all_normal:
            _, p_levene = stats.levene(*group_data)
            t_stat, p_val = stats.ttest_ind(*group_data, equal_var=(p_levene >= 0.05))
            test_name = "독립표본 t-검정"
            stat_name = "t"
        else:
            stat, p_val = stats.mannwhitneyu(*group_data, alternative='two-sided')
            t_stat = stat
            test_name = "Mann-Whitney U 검정"
            stat_name = "U"
    else:
        # 세 그룹 이상 비교
        if all_normal:
            t_stat, p_val = stats.f_oneway(*group_data)
            test_name = "일원분산분석 (ANOVA)"
            stat_name = "F"
        else:
            t_stat, p_val = stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis 검정"
            stat_name = "H"

    print(f"  사용 검정: {test_name}")
    print(f"  {stat_name} 통계량: {t_stat:.4f}")
    print(f"  p-value: {p_val:.4e}")

    # 4. 결론
    print(f"\n[4] 결론")
    print("-" * 50)

    if p_val < 0.05:
        print(f"  p-value ({p_val:.4e}) < 0.05")
        print(f"  → {group_var}에 따른 {value_var}의 차이가 통계적으로 유의합니다.")

        if len(groups) > 2:
            print(f"\n  [사후검정 권장] Tukey HSD 또는 Dunn 검정으로")
            print(f"  어떤 그룹 간 차이가 있는지 추가 분석 필요")
    else:
        print(f"  p-value ({p_val:.4f}) ≥ 0.05")
        print(f"  → {group_var}에 따른 {value_var}의 유의한 차이가 없습니다.")

    print(f"\n{'=' * 60}\n")


# 함수 테스트
print("\n[5.2] 종합 분석 리포트 함수 테스트")

# 리포트 1: Iris 품종별 꽃잎 길이
generate_analysis_report(iris, 'species', 'petal_length',
                        "Iris 품종별 꽃잎 길이 분석")

# 리포트 2: Titanic 객실 등급별 요금
generate_analysis_report(titanic, 'pclass', 'fare',
                        "Titanic 객실 등급별 요금 분석")


# =============================================================================
# 6. 시각화
# =============================================================================

print("\n" + "=" * 70)
print("6. 시각화")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 박스플롯 (Box Plot) - 그룹 간 비교
# -----------------------------------------------------------------------------

print("\n[6.1] 박스플롯 - 그룹 간 비교 시각화")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1) Iris - 품종별 꽃잎 길이
ax1 = axes[0, 0]
iris.boxplot(column='petal_length', by='species', ax=ax1)
ax1.set_title('Iris: Petal Length by Species\n(ANOVA p < 0.001)', fontsize=11)
ax1.set_xlabel('Species')
ax1.set_ylabel('Petal Length (cm)')
plt.suptitle('')  # 기본 제목 제거

# 2) Titanic - 생존 여부별 요금
ax2 = axes[0, 1]
titanic.boxplot(column='fare', by='survived', ax=ax2)
ax2.set_title('Titanic: Fare by Survival\n(t-test p < 0.001)', fontsize=11)
ax2.set_xlabel('Survived (0=No, 1=Yes)')
ax2.set_ylabel('Fare')
plt.suptitle('')

# 3) Wine - 품질 등급별 알코올
ax3 = axes[1, 0]
wine.boxplot(column='alcohol', by='quality', ax=ax3)
ax3.set_title('Wine: Alcohol by Quality\n(ANOVA p < 0.001)', fontsize=11)
ax3.set_xlabel('Quality Rating')
ax3.set_ylabel('Alcohol (%)')
plt.suptitle('')

# 4) Titanic - 객실 등급별 나이
ax4 = axes[1, 1]
titanic.boxplot(column='age', by='pclass', ax=ax4)
ax4.set_title('Titanic: Age by Passenger Class\n(Kruskal-Wallis)', fontsize=11)
ax4.set_xlabel('Passenger Class')
ax4.set_ylabel('Age')
plt.suptitle('')

plt.tight_layout()
plt.savefig('/home/devfish/Project/lecture-linecyber-2026-credit/강의안_보강/07_통계검정_실습/boxplot_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  → boxplot_comparison.png 저장 완료")

# -----------------------------------------------------------------------------
# 6.2 카이제곱 검정 시각화 - 교차표 히트맵
# -----------------------------------------------------------------------------

print("\n[6.2] 교차표 히트맵 - 범주형 변수 관계 시각화")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1) Titanic - 성별 vs 생존
ax1 = axes[0]
ct1 = pd.crosstab(titanic['sex'], titanic['survived'], normalize='index') * 100
sns.heatmap(ct1, annot=True, fmt='.1f', cmap='Blues', ax=ax1,
            xticklabels=['Died', 'Survived'],
            yticklabels=['Female', 'Male'])
ax1.set_title('Titanic: Sex vs Survival (%)\nChi-square p < 0.001', fontsize=11)
ax1.set_xlabel('Survived')
ax1.set_ylabel('Sex')

# 2) Titanic - 객실 등급 vs 생존
ax2 = axes[1]
ct2 = pd.crosstab(titanic['pclass'], titanic['survived'], normalize='index') * 100
sns.heatmap(ct2, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
            xticklabels=['Died', 'Survived'],
            yticklabels=['1st', '2nd', '3rd'])
ax2.set_title('Titanic: Class vs Survival (%)\nChi-square p < 0.001', fontsize=11)
ax2.set_xlabel('Survived')
ax2.set_ylabel('Passenger Class')

plt.tight_layout()
plt.savefig('/home/devfish/Project/lecture-linecyber-2026-credit/강의안_보강/07_통계검정_실습/contingency_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  → contingency_heatmap.png 저장 완료")

# -----------------------------------------------------------------------------
# 6.3 정규성 검정 시각화 - Q-Q Plot
# -----------------------------------------------------------------------------

print("\n[6.3] Q-Q Plot - 정규성 검정 시각화")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1) Iris sepal_length (정규분포에 가까움)
ax1 = axes[0, 0]
stats.probplot(iris['sepal_length'], dist="norm", plot=ax1)
_, p_val = stats.shapiro(iris['sepal_length'])
ax1.set_title(f'Iris Sepal Length\nShapiro-Wilk p = {p_val:.4f}', fontsize=10)

# 2) Titanic fare (비정규분포)
ax2 = axes[0, 1]
fare_clean = titanic['fare'].dropna()
stats.probplot(fare_clean, dist="norm", plot=ax2)
_, p_val = stats.shapiro(fare_clean.sample(500, random_state=42))
ax2.set_title(f'Titanic Fare\nShapiro-Wilk p = {p_val:.4f}', fontsize=10)

# 3) Wine pH (정규분포에 가까움)
ax3 = axes[1, 0]
stats.probplot(wine['pH'], dist="norm", plot=ax3)
_, p_val = stats.shapiro(wine['pH'].sample(500, random_state=42))
ax3.set_title(f'Wine pH\nShapiro-Wilk p = {p_val:.4f}', fontsize=10)

# 4) Wine residual sugar (비정규분포)
ax4 = axes[1, 1]
stats.probplot(wine['residual sugar'], dist="norm", plot=ax4)
_, p_val = stats.shapiro(wine['residual sugar'].sample(500, random_state=42))
ax4.set_title(f'Wine Residual Sugar\nShapiro-Wilk p = {p_val:.4f}', fontsize=10)

plt.tight_layout()
plt.savefig('/home/devfish/Project/lecture-linecyber-2026-credit/강의안_보강/07_통계검정_실습/qqplot_normality.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  → qqplot_normality.png 저장 완료")

# -----------------------------------------------------------------------------
# 6.4 검정 결과 요약 바 차트
# -----------------------------------------------------------------------------

print("\n[6.4] 검정 결과 요약 시각화")

# 여러 검정 결과를 시각화
tests = [
    ('Iris: Setosa vs Versicolor\n(Petal Length)',
     stats.ttest_ind(setosa_petal, versicolor_petal)[1]),
    ('Titanic: Survived vs Died\n(Fare)',
     stats.ttest_ind(survived_fare, died_fare)[1]),
    ('Wine: Quality Groups\n(Alcohol)',
     stats.f_oneway(*quality_groups)[1]),
    ('Titanic: Sex vs Survival\n(Chi-square)',
     stats.chi2_contingency(pd.crosstab(titanic['sex'], titanic['survived']))[1]),
    ('Titanic: Class vs Survival\n(Chi-square)',
     stats.chi2_contingency(pd.crosstab(titanic['pclass'], titanic['survived']))[1])
]

fig, ax = plt.subplots(figsize=(12, 6))

test_names = [t[0] for t in tests]
p_values = [t[1] for t in tests]
log_p = [-np.log10(p) for p in p_values]  # -log10(p) 변환

colors = ['green' if p < 0.05 else 'gray' for p in p_values]
bars = ax.barh(test_names, log_p, color=colors, edgecolor='black')

# 유의수준 선 추가
ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2,
           label='alpha = 0.05')

ax.set_xlabel('-log10(p-value)', fontsize=11)
ax.set_title('Statistical Test Results Summary\n(Green: Significant, Gray: Not Significant)',
             fontsize=12)
ax.legend(loc='lower right')

# p-value 값 표시
for i, (bar, p) in enumerate(zip(bars, p_values)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'p = {p:.2e}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/devfish/Project/lecture-linecyber-2026-credit/강의안_보강/07_통계검정_실습/test_results_summary.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  → test_results_summary.png 저장 완료")


# =============================================================================
# 7. 연습 문제
# =============================================================================

print("\n\n" + "=" * 70)
print("7. 연습 문제")
print("=" * 70)

# -----------------------------------------------------------------------------
# 연습문제 1
# -----------------------------------------------------------------------------

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  연습문제 1: Iris 데이터 분석                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Versicolor와 Virginica의 sepal_width를 비교하세요.                  │
│                                                                      │
│  수행할 작업:                                                        │
│  1. 두 그룹의 기술통계량 계산                                        │
│  2. 정규성 검정 수행                                                 │
│  3. 적절한 통계 검정 선택 및 수행                                    │
│  4. 결론 도출                                                        │
└─────────────────────────────────────────────────────────────────────┘
""")

print("[정답]")
print("-" * 50)

# 데이터 준비
versicolor_sw = iris[iris['species'] == 'versicolor']['sepal_width']
virginica_sw = iris[iris['species'] == 'virginica']['sepal_width']

# 1. 기술통계
print("\n1. 기술통계량:")
print(f"   Versicolor - 평균: {versicolor_sw.mean():.3f}, 표준편차: {versicolor_sw.std():.3f}")
print(f"   Virginica  - 평균: {virginica_sw.mean():.3f}, 표준편차: {virginica_sw.std():.3f}")

# 2. 정규성 검정
_, p1 = stats.shapiro(versicolor_sw)
_, p2 = stats.shapiro(virginica_sw)
print(f"\n2. 정규성 검정 (Shapiro-Wilk):")
print(f"   Versicolor: p = {p1:.4f} ({'정규분포' if p1 >= 0.05 else '비정규분포'})")
print(f"   Virginica:  p = {p2:.4f} ({'정규분포' if p2 >= 0.05 else '비정규분포'})")

# 3. 통계 검정
if p1 >= 0.05 and p2 >= 0.05:
    # 정규분포 → t-검정
    _, p_levene = stats.levene(versicolor_sw, virginica_sw)
    t_stat, p_val = stats.ttest_ind(versicolor_sw, virginica_sw,
                                     equal_var=(p_levene >= 0.05))
    test_name = "독립표본 t-검정"
else:
    # 비정규분포 → Mann-Whitney U
    t_stat, p_val = stats.mannwhitneyu(versicolor_sw, virginica_sw)
    test_name = "Mann-Whitney U 검정"

print(f"\n3. {test_name}:")
print(f"   통계량: {t_stat:.4f}")
print(f"   p-value: {p_val:.4f}")

# 4. 결론
print(f"\n4. 결론:")
if p_val < 0.05:
    print(f"   p-value ({p_val:.4f}) < 0.05")
    print("   → Versicolor와 Virginica의 sepal_width에 유의한 차이가 있습니다.")
else:
    print(f"   p-value ({p_val:.4f}) ≥ 0.05")
    print("   → Versicolor와 Virginica의 sepal_width에 유의한 차이가 없습니다.")

# -----------------------------------------------------------------------------
# 연습문제 2
# -----------------------------------------------------------------------------

print("""
\n
┌─────────────────────────────────────────────────────────────────────┐
│  연습문제 2: Wine 데이터 분석                                        │
├─────────────────────────────────────────────────────────────────────┤
│  품질 등급을 '낮음(3-4)', '중간(5-6)', '높음(7-8)'으로 그룹화하고,   │
│  pH 값에 차이가 있는지 검정하세요.                                   │
│                                                                      │
│  수행할 작업:                                                        │
│  1. 품질 그룹 생성                                                   │
│  2. 각 그룹의 pH 기술통계 계산                                       │
│  3. ANOVA 또는 Kruskal-Wallis 수행                                   │
│  4. 결론 도출                                                        │
└─────────────────────────────────────────────────────────────────────┘
""")

print("[정답]")
print("-" * 50)

# 1. 품질 그룹 생성
def quality_group(q):
    if q <= 4:
        return 'Low (3-4)'
    elif q <= 6:
        return 'Medium (5-6)'
    else:
        return 'High (7-8)'

wine['quality_group'] = wine['quality'].apply(quality_group)

print("\n1. 품질 그룹 생성 완료:")
print(wine['quality_group'].value_counts().sort_index())

# 2. 각 그룹의 pH 기술통계
print("\n2. 그룹별 pH 기술통계:")
for group in ['Low (3-4)', 'Medium (5-6)', 'High (7-8)']:
    ph_data = wine[wine['quality_group'] == group]['pH']
    print(f"   {group}: 평균 {ph_data.mean():.3f}, 표준편차 {ph_data.std():.3f}, N = {len(ph_data)}")

# 3. 정규성 검정 및 적절한 검정 선택
low_ph = wine[wine['quality_group'] == 'Low (3-4)']['pH'].values
mid_ph = wine[wine['quality_group'] == 'Medium (5-6)']['pH'].values
high_ph = wine[wine['quality_group'] == 'High (7-8)']['pH'].values

# 정규성 검정
_, p_low = stats.shapiro(low_ph[:500])
_, p_mid = stats.shapiro(mid_ph[:500])
_, p_high = stats.shapiro(high_ph[:500])

print("\n3. 정규성 검정:")
print(f"   Low:    p = {p_low:.4f}")
print(f"   Medium: p = {p_mid:.4f}")
print(f"   High:   p = {p_high:.4f}")

# ANOVA 또는 Kruskal-Wallis
if p_low >= 0.05 and p_mid >= 0.05 and p_high >= 0.05:
    f_stat, p_val = stats.f_oneway(low_ph, mid_ph, high_ph)
    test_name = "One-way ANOVA"
    stat_name = "F"
else:
    f_stat, p_val = stats.kruskal(low_ph, mid_ph, high_ph)
    test_name = "Kruskal-Wallis"
    stat_name = "H"

print(f"\n   {test_name} 결과:")
print(f"   {stat_name} 통계량: {f_stat:.4f}")
print(f"   p-value: {p_val:.4f}")

# 4. 결론
print(f"\n4. 결론:")
if p_val < 0.05:
    print(f"   p-value ({p_val:.4f}) < 0.05")
    print("   → 품질 그룹에 따라 pH에 유의한 차이가 있습니다.")
else:
    print(f"   p-value ({p_val:.4f}) ≥ 0.05")
    print("   → 품질 그룹에 따라 pH에 유의한 차이가 없습니다.")

# -----------------------------------------------------------------------------
# 연습문제 3
# -----------------------------------------------------------------------------

print("""
\n
┌─────────────────────────────────────────────────────────────────────┐
│  연습문제 3: Titanic 카이제곱 검정                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Titanic 데이터에서 승선 항구(embark_town)와 생존여부(survived)의    │
│  관계를 카이제곱 검정으로 분석하세요.                                │
│                                                                      │
│  수행할 작업:                                                        │
│  1. 교차표 생성                                                      │
│  2. 항구별 생존율 계산                                               │
│  3. 카이제곱 검정 수행                                               │
│  4. 결론 도출                                                        │
└─────────────────────────────────────────────────────────────────────┘
""")

print("[정답]")
print("-" * 50)

# 결측치 제거
titanic_clean = titanic.dropna(subset=['embark_town', 'survived'])

# 1. 교차표 생성
print("\n1. 교차표:")
ct = pd.crosstab(titanic_clean['embark_town'], titanic_clean['survived'],
                 margins=True, margins_name='Total')
ct.columns = ['사망(0)', '생존(1)', 'Total']
print(ct)

# 2. 항구별 생존율
print("\n2. 항구별 생존율:")
for town in titanic_clean['embark_town'].unique():
    rate = titanic_clean[titanic_clean['embark_town'] == town]['survived'].mean() * 100
    print(f"   {town}: {rate:.1f}%")

# 3. 카이제곱 검정
ct_for_test = pd.crosstab(titanic_clean['embark_town'], titanic_clean['survived'])
chi2, p_val, dof, expected = stats.chi2_contingency(ct_for_test)

print(f"\n3. 카이제곱 검정 결과:")
print(f"   카이제곱 통계량: {chi2:.4f}")
print(f"   자유도: {dof}")
print(f"   p-value: {p_val:.4f}")

# 4. 결론
print(f"\n4. 결론:")
if p_val < 0.05:
    print(f"   p-value ({p_val:.4f}) < 0.05")
    print("   → 승선 항구와 생존여부 간에 유의한 관련이 있습니다.")
    print("   (Cherbourg 항구 승객의 생존율이 가장 높았습니다)")
else:
    print(f"   p-value ({p_val:.4f}) ≥ 0.05")
    print("   → 승선 항구와 생존여부 간에 유의한 관련이 없습니다.")

# 정리
wine = wine.drop(columns=['quality_group'], errors='ignore')


# =============================================================================
# 8. 학습 정리
# =============================================================================

print("\n\n" + "=" * 70)
print("8. 학습 정리")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                       통계 검정 선택 가이드                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [1] 두 그룹 평균 비교                                               │
│      ├─ 정규분포 + 독립표본 → 독립표본 t-검정 (ttest_ind)           │
│      ├─ 정규분포 + 대응표본 → 대응표본 t-검정 (ttest_rel)           │
│      ├─ 비정규분포 + 독립표본 → Mann-Whitney U (mannwhitneyu)       │
│      └─ 비정규분포 + 대응표본 → Wilcoxon 부호순위 (wilcoxon)        │
│                                                                      │
│  [2] 세 그룹 이상 평균 비교                                          │
│      ├─ 정규분포 → One-way ANOVA (f_oneway)                         │
│      │   └─ 유의시 → Tukey HSD 사후검정                             │
│      └─ 비정규분포 → Kruskal-Wallis (kruskal)                       │
│                                                                      │
│  [3] 범주형 변수 관계                                                │
│      └─ 카이제곱 검정 (chi2_contingency)                            │
│                                                                      │
│  [4] 정규성 검정                                                     │
│      └─ Shapiro-Wilk (shapiro)                                      │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         핵심 기억사항                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  • p-value < 0.05 → 귀무가설 기각 → 유의한 차이/관계 있음           │
│  • p-value ≥ 0.05 → 귀무가설 채택 → 유의한 차이/관계 없음           │
│  • 정규성 검정 먼저! → 모수/비모수 검정 결정                         │
│  • ANOVA 유의시 → 사후검정으로 어떤 그룹 간 차이인지 확인           │
│  • 통계적 유의성 ≠ 실질적 중요성 (효과 크기도 고려)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n오늘 수업에서 사용한 주요 함수:")
print("-" * 50)
print("from scipy import stats")
print()
print("# 정규성 검정")
print("stats.shapiro(data)  # Shapiro-Wilk test")
print()
print("# 모수 검정")
print("stats.ttest_ind(group1, group2)  # 독립표본 t-검정")
print("stats.ttest_rel(before, after)   # 대응표본 t-검정")
print("stats.f_oneway(g1, g2, g3, ...)  # One-way ANOVA")
print()
print("# 비모수 검정")
print("stats.mannwhitneyu(group1, group2)  # Mann-Whitney U")
print("stats.wilcoxon(before, after)       # Wilcoxon 부호순위")
print("stats.kruskal(g1, g2, g3, ...)      # Kruskal-Wallis")
print()
print("# 범주형 검정")
print("stats.chi2_contingency(contingency_table)  # 카이제곱")
print()
print("# 사후검정")
print("from scipy.stats import tukey_hsd")
print("from statsmodels.stats.multicomp import pairwise_tukeyhsd")

print("\n" + "=" * 70)
print("7차시 실습 완료!")
print("=" * 70)
