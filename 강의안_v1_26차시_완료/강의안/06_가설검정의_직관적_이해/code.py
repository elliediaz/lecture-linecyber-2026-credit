"""
[6차시] 가설검정의 직관적 이해 - 실습 코드

가설검정과 A/B 테스트의 기본 개념을 실습합니다.

학습목표:
- 가설검정의 기본 개념 이해
- p-value의 의미 이해
- A/B 테스트의 원리 이해
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("6차시: 가설검정의 직관적 이해")
print("=" * 60)
print()


# =============================================================
# 1. p-value 직관적 이해 - 동전 던지기
# =============================================================

print("=" * 60)
print("1. p-value 직관적 이해 - 동전 던지기")
print("=" * 60)

# 동전 10번 던져 앞면 9번
n_tosses = 10
n_heads = 9

# 공정한 동전(p=0.5)에서 9번 이상 앞면이 나올 확률
# P(X >= 9) = P(X=9) + P(X=10)
p_value = sum(stats.binom.pmf(k, n_tosses, 0.5) for k in range(n_heads, n_tosses+1))

# 양측검정이면 양쪽 극단 모두 고려
p_value_two_sided = 2 * min(p_value, 1 - p_value)

print(f"실험: 동전 {n_tosses}번 던져 앞면 {n_heads}번")
print()
print("H₀: 동전은 공정하다 (p = 0.5)")
print("H₁: 동전은 공정하지 않다")
print()
print(f"p-value (단측): {p_value:.4f}")
print(f"p-value (양측): {p_value_two_sided:.4f}")
print()

if p_value_two_sided < 0.05:
    print("결론: p < 0.05이므로 귀무가설 기각")
    print("→ 이 동전은 공정하지 않다고 볼 수 있음")
else:
    print("결론: p >= 0.05이므로 귀무가설 유지")
    print("→ 공정한 동전이 아니라고 단정할 수 없음")
print()


# =============================================================
# 2. p-value 시각화
# =============================================================

print("=" * 60)
print("2. p-value 시각화")
print("=" * 60)

# 이항분포 확률질량함수 시각화
x = np.arange(0, 11)
pmf = stats.binom.pmf(x, n_tosses, 0.5)

plt.figure(figsize=(10, 6))
colors = ['red' if k >= 9 or k <= 1 else 'steelblue' for k in x]
plt.bar(x, pmf, color=colors, edgecolor='black', alpha=0.7)

# p-value 영역 표시
plt.axvline(8.5, color='red', linestyle='--', alpha=0.5)
plt.text(9.2, 0.15, f'p-value 영역\n(9~10)\n= {p_value:.4f}', fontsize=10, color='red')

plt.xlabel('앞면 횟수')
plt.ylabel('확률')
plt.title('공정한 동전 10회 던지기: 이항분포 B(10, 0.5)')
plt.xticks(x)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('01_pvalue_visualization.png', dpi=150)
plt.show()

print("'01_pvalue_visualization.png' 저장 완료")
print()


# =============================================================
# 3. 독립표본 t-검정 (두 그룹 평균 비교)
# =============================================================

print("=" * 60)
print("3. 독립표본 t-검정")
print("=" * 60)

# 두 생산 라인의 생산량 데이터
np.random.seed(42)
line_a = np.random.normal(1200, 30, 30)  # 라인A: 평균 1200
line_b = np.random.normal(1180, 35, 30)  # 라인B: 평균 1180

print("라인 A 생산량:")
print(f"  평균: {np.mean(line_a):.2f}, 표준편차: {np.std(line_a):.2f}")
print(f"  데이터 수: {len(line_a)}")

print("\n라인 B 생산량:")
print(f"  평균: {np.mean(line_b):.2f}, 표준편차: {np.std(line_b):.2f}")
print(f"  데이터 수: {len(line_b)}")

# 독립표본 t-검정
t_statistic, p_value = stats.ttest_ind(line_a, line_b)

print(f"\n[독립표본 t-검정 결과]")
print(f"H₀: 두 라인의 평균 생산량은 같다")
print(f"H₁: 두 라인의 평균 생산량은 다르다")
print()
print(f"t-통계량: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print()

alpha = 0.05
if p_value < alpha:
    print(f"결론: p-value ({p_value:.4f}) < α ({alpha})")
    print("→ 귀무가설 기각. 두 라인의 생산량에 유의미한 차이가 있음")
else:
    print(f"결론: p-value ({p_value:.4f}) >= α ({alpha})")
    print("→ 귀무가설 유지. 두 라인의 생산량 차이가 유의미하지 않음")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 상자그림
axes[0].boxplot([line_a, line_b], labels=['라인 A', '라인 B'])
axes[0].set_ylabel('생산량')
axes[0].set_title('라인별 생산량 비교 (상자그림)')
axes[0].grid(True, alpha=0.3, axis='y')

# 히스토그램
axes[1].hist(line_a, bins=15, alpha=0.7, label=f'라인 A (μ={np.mean(line_a):.1f})', color='steelblue')
axes[1].hist(line_b, bins=15, alpha=0.7, label=f'라인 B (μ={np.mean(line_b):.1f})', color='coral')
axes[1].set_xlabel('생산량')
axes[1].set_ylabel('빈도')
axes[1].set_title(f't-검정 결과: p={p_value:.4f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('02_ttest_independent.png', dpi=150)
plt.show()

print("\n'02_ttest_independent.png' 저장 완료")
print()


# =============================================================
# 4. 대응표본 t-검정 (전/후 비교)
# =============================================================

print("=" * 60)
print("4. 대응표본 t-검정 (전/후 비교)")
print("=" * 60)

# 같은 라인에서 개선 전/후 불량률
np.random.seed(42)
before = np.random.normal(0.035, 0.008, 20)  # 개선 전: 평균 3.5%
after = before - np.random.normal(0.005, 0.003, 20)  # 개선 후: 약 0.5% 감소
after = np.clip(after, 0, 1)

print("개선 전 불량률:")
print(f"  평균: {np.mean(before)*100:.2f}%, 표준편차: {np.std(before)*100:.2f}%")

print("\n개선 후 불량률:")
print(f"  평균: {np.mean(after)*100:.2f}%, 표준편차: {np.std(after)*100:.2f}%")

print(f"\n차이: {(np.mean(before) - np.mean(after))*100:.2f}%p 감소")

# 대응표본 t-검정
t_statistic, p_value = stats.ttest_rel(before, after)

print(f"\n[대응표본 t-검정 결과]")
print(f"H₀: 개선 전후 불량률 차이가 없다")
print(f"H₁: 개선 전후 불량률 차이가 있다")
print()
print(f"t-통계량: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print()

if p_value < 0.05:
    print("결론: 개선 효과가 통계적으로 유의미함")
else:
    print("결론: 개선 효과가 통계적으로 유의미하지 않음")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 전후 비교
axes[0].boxplot([before*100, after*100], labels=['개선 전', '개선 후'])
axes[0].set_ylabel('불량률 (%)')
axes[0].set_title(f'개선 전후 불량률 비교 (p={p_value:.4f})')
axes[0].grid(True, alpha=0.3, axis='y')

# 개별 변화
for i in range(len(before)):
    axes[1].plot([0, 1], [before[i]*100, after[i]*100], 'b-o', alpha=0.3)
axes[1].plot([0, 1], [np.mean(before)*100, np.mean(after)*100], 'r-o',
             linewidth=3, markersize=10, label='평균')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['개선 전', '개선 후'])
axes[1].set_ylabel('불량률 (%)')
axes[1].set_title('각 측정점의 전후 변화')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_ttest_paired.png', dpi=150)
plt.show()

print("'03_ttest_paired.png' 저장 완료")
print()


# =============================================================
# 5. A/B 테스트 (비율 비교)
# =============================================================

print("=" * 60)
print("5. A/B 테스트 (비율 비교)")
print("=" * 60)

# 기존 공정 vs 새 공정 불량률 비교
n_a = 1000  # A그룹 (기존 공정) 샘플 수
n_b = 1000  # B그룹 (새 공정) 샘플 수

defects_a = 35  # A그룹 불량품 수 (3.5%)
defects_b = 25  # B그룹 불량품 수 (2.5%)

rate_a = defects_a / n_a
rate_b = defects_b / n_b

print("A/B 테스트 설정:")
print(f"  A그룹 (기존 공정): {n_a}개 중 {defects_a}개 불량 ({rate_a*100:.1f}%)")
print(f"  B그룹 (새 공정): {n_b}개 중 {defects_b}개 불량 ({rate_b*100:.1f}%)")
print(f"  차이: {(rate_a - rate_b)*100:.1f}%p")
print()

# 비율 검정 (Z-test for proportions)
from statsmodels.stats.proportion import proportions_ztest

count = np.array([defects_a, defects_b])
nobs = np.array([n_a, n_b])

z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

print("[비율 검정 (Z-test) 결과]")
print(f"H₀: 두 공정의 불량률은 같다")
print(f"H₁: 두 공정의 불량률은 다르다")
print()
print(f"Z-통계량: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print()

if p_value < 0.05:
    print("결론: 새 공정의 불량률 감소가 통계적으로 유의미함")
else:
    print("결론: 불량률 차이가 통계적으로 유의미하지 않음")

# 신뢰구간 계산
from statsmodels.stats.proportion import confint_proportions_2indep

ci_low, ci_high = confint_proportions_2indep(defects_a, n_a, defects_b, n_b)
print(f"\n불량률 차이의 95% 신뢰구간: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
print()


# =============================================================
# 6. 표본 크기와 검정력
# =============================================================

print("=" * 60)
print("6. 표본 크기와 검정력")
print("=" * 60)

# 같은 효과 크기에서 표본 크기에 따른 p-value 변화
np.random.seed(42)

sample_sizes = [10, 30, 100, 300, 1000]
p_values = []

# 모집단 설정
pop_mean_a = 100
pop_mean_b = 98  # 2% 차이
pop_std = 10

print("모집단 설정:")
print(f"  집단 A 평균: {pop_mean_a}, 집단 B 평균: {pop_mean_b}")
print(f"  표준편차: {pop_std}")
print(f"  실제 차이: {pop_mean_a - pop_mean_b}")
print()

print("[표본 크기별 검정 결과]")
for n in sample_sizes:
    # 표본 추출 및 t-검정 (100회 반복 평균)
    p_vals = []
    for _ in range(100):
        sample_a = np.random.normal(pop_mean_a, pop_std, n)
        sample_b = np.random.normal(pop_mean_b, pop_std, n)
        _, p = stats.ttest_ind(sample_a, sample_b)
        p_vals.append(p)

    mean_p = np.mean(p_vals)
    power = np.mean([p < 0.05 for p in p_vals])  # 검정력 추정
    p_values.append(mean_p)

    print(f"  n={n:4d}: 평균 p-value={mean_p:.4f}, 검정력={power:.1%}")

print()
print("→ 표본 크기가 커질수록 같은 효과 크기에서 p-value가 작아지고 검정력이 높아짐")
print()


# =============================================================
# 7. 통계적 유의성 vs 실용적 유의성
# =============================================================

print("=" * 60)
print("7. 통계적 유의성 vs 실용적 유의성")
print("=" * 60)

# 매우 큰 표본에서 작은 차이
np.random.seed(42)

n_large = 100000
group_a = np.random.normal(100.0, 10, n_large)
group_b = np.random.normal(100.1, 10, n_large)  # 0.1% 차이

mean_diff = np.mean(group_a) - np.mean(group_b)
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# 효과 크기 (Cohen's d)
pooled_std = np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
cohens_d = abs(mean_diff) / pooled_std

print(f"표본 크기: {n_large:,}개 (각 그룹)")
print(f"그룹 A 평균: {np.mean(group_a):.4f}")
print(f"그룹 B 평균: {np.mean(group_b):.4f}")
print(f"평균 차이: {abs(mean_diff):.4f}")
print()
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Cohen's d (효과 크기): {cohens_d:.4f}")
print()

print("[해석]")
if p_value < 0.05:
    print(f"✓ 통계적으로 유의미함 (p = {p_value:.6f} < 0.05)")
else:
    print(f"✗ 통계적으로 유의미하지 않음 (p = {p_value:.6f} >= 0.05)")

if cohens_d < 0.2:
    print(f"✗ 효과 크기가 매우 작음 (d = {cohens_d:.4f} < 0.2)")
elif cohens_d < 0.5:
    print(f"△ 효과 크기가 작음 (0.2 <= d = {cohens_d:.4f} < 0.5)")
elif cohens_d < 0.8:
    print(f"○ 효과 크기가 중간 (0.5 <= d = {cohens_d:.4f} < 0.8)")
else:
    print(f"✓ 효과 크기가 큼 (d = {cohens_d:.4f} >= 0.8)")

print()
print("→ 통계적으로 유의미하지만, 실용적으로는 의미없는 차이일 수 있음!")
print("→ 항상 효과 크기(Cohen's d)를 함께 확인할 것")
print()


# =============================================================
# 8. 다양한 검정 방법 요약
# =============================================================

print("=" * 60)
print("8. 다양한 검정 방법 요약")
print("=" * 60)

# ANOVA (세 그룹 이상 비교)
np.random.seed(42)
group1 = np.random.normal(100, 10, 30)
group2 = np.random.normal(105, 10, 30)
group3 = np.random.normal(103, 10, 30)

f_stat, p_value_anova = stats.f_oneway(group1, group2, group3)

print("[일원분산분석 (One-way ANOVA)]")
print("세 그룹의 평균 비교")
print(f"  그룹1 평균: {np.mean(group1):.2f}")
print(f"  그룹2 평균: {np.mean(group2):.2f}")
print(f"  그룹3 평균: {np.mean(group3):.2f}")
print(f"  F-통계량: {f_stat:.4f}")
print(f"  p-value: {p_value_anova:.4f}")
print()

# 카이제곱 검정 (범주형 데이터)
observed = np.array([[50, 30], [40, 40]])  # 관측 빈도
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(observed)

print("[카이제곱 검정 (Chi-square test)]")
print("범주형 변수 간 독립성 검정")
print(f"  관측 빈도: {observed.tolist()}")
print(f"  카이제곱 통계량: {chi2:.4f}")
print(f"  p-value: {p_value_chi2:.4f}")
print()

print("[검정 방법 선택 가이드]")
print("┌────────────────────┬──────────────────────┐")
print("│ 상황               │ 검정 방법            │")
print("├────────────────────┼──────────────────────┤")
print("│ 두 그룹 평균 비교  │ 독립표본 t-검정      │")
print("│ 전/후 비교         │ 대응표본 t-검정      │")
print("│ 세 그룹+ 평균 비교 │ ANOVA                │")
print("│ 비율 비교          │ Z-검정, 카이제곱     │")
print("│ 상관관계 검정      │ 상관계수 검정        │")
print("└────────────────────┴──────────────────────┘")
print()

print("=" * 60)
print("6차시 실습 완료!")
print("=" * 60)
