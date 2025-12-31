"""
[5차시] 확률분포와 AI 예측의 연결 - 실습 코드

확률분포의 기본 개념과 AI 예측에서의 활용을 실습합니다.

학습목표:
- 정규분포의 개념과 특성 이해
- 확률분포가 AI 예측에서 사용되는 방식 이해
- 불확실성을 수치로 표현하는 방법 이해
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("5차시: 확률분포와 AI 예측의 연결")
print("=" * 60)
print()


# =============================================================
# 1. 정규분포의 특성
# =============================================================

print("=" * 60)
print("1. 정규분포의 특성")
print("=" * 60)

# 정규분포 생성
mu = 1200  # 평균
sigma = 50  # 표준편차

# scipy.stats로 정규분포 객체 생성
norm_dist = stats.norm(loc=mu, scale=sigma)

# 68-95-99.7 규칙 확인
print(f"정규분포 N({mu}, {sigma}²)")
print()
print("[68-95-99.7 규칙]")
print(f"P({mu-sigma} < X < {mu+sigma}) = {norm_dist.cdf(mu+sigma) - norm_dist.cdf(mu-sigma):.1%} (이론값: 68%)")
print(f"P({mu-2*sigma} < X < {mu+2*sigma}) = {norm_dist.cdf(mu+2*sigma) - norm_dist.cdf(mu-2*sigma):.1%} (이론값: 95%)")
print(f"P({mu-3*sigma} < X < {mu+3*sigma}) = {norm_dist.cdf(mu+3*sigma) - norm_dist.cdf(mu-3*sigma):.2%} (이론값: 99.7%)")
print()


# =============================================================
# 2. 정규분포 시각화
# =============================================================

print("=" * 60)
print("2. 정규분포 시각화")
print("=" * 60)

# 정규분포 곡선 그리기
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm_dist.pdf(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'b-', linewidth=2, label='정규분포 PDF')

# 영역 채우기
x_fill_1sigma = np.linspace(mu-sigma, mu+sigma, 100)
x_fill_2sigma = np.linspace(mu-2*sigma, mu+2*sigma, 100)

plt.fill_between(x_fill_2sigma, norm_dist.pdf(x_fill_2sigma), alpha=0.2, color='blue', label='±2σ (95%)')
plt.fill_between(x_fill_1sigma, norm_dist.pdf(x_fill_1sigma), alpha=0.4, color='blue', label='±1σ (68%)')

# 평균선
plt.axvline(mu, color='red', linestyle='--', label=f'평균 (μ={mu})')

# 표준편차 표시
for i in [-2, -1, 1, 2]:
    plt.axvline(mu + i*sigma, color='gray', linestyle=':', alpha=0.5)
    plt.text(mu + i*sigma, max(y)*0.1, f'{i}σ', ha='center')

plt.xlabel('생산량')
plt.ylabel('확률밀도')
plt.title('정규분포와 68-95-99.7 규칙')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_normal_distribution.png', dpi=150)
plt.show()

print("'01_normal_distribution.png' 저장 완료")
print()


# =============================================================
# 3. Z-score를 활용한 이상치 탐지
# =============================================================

print("=" * 60)
print("3. Z-score를 활용한 이상치 탐지")
print("=" * 60)

# 샘플 데이터 생성 (일부 이상치 포함)
np.random.seed(42)
production = np.concatenate([
    np.random.normal(1200, 50, 95),  # 정상 데이터
    np.array([900, 950, 1400, 1450, 1500])  # 이상치
])

# Z-score 계산
mean_val = np.mean(production)
std_val = np.std(production)
z_scores = (production - mean_val) / std_val

print(f"평균: {mean_val:.2f}")
print(f"표준편차: {std_val:.2f}")
print()

# 이상치 식별 (|Z| > 2)
outlier_mask = np.abs(z_scores) > 2
print(f"[이상치 탐지 (|Z| > 2)]")
print(f"이상치 개수: {np.sum(outlier_mask)}")
print(f"이상치 값: {production[outlier_mask]}")
print(f"이상치 Z-score: {z_scores[outlier_mask]}")
print()

# 시각화
plt.figure(figsize=(12, 5))

# Z-score 분포
plt.subplot(1, 2, 1)
colors = ['red' if z else 'steelblue' for z in outlier_mask]
plt.scatter(range(len(production)), z_scores, c=colors, alpha=0.6)
plt.axhline(2, color='red', linestyle='--', label='Z=±2')
plt.axhline(-2, color='red', linestyle='--')
plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
plt.xlabel('데이터 인덱스')
plt.ylabel('Z-score')
plt.title('Z-score로 이상치 탐지')
plt.legend()

# 원본 데이터 분포
plt.subplot(1, 2, 2)
plt.hist(production[~outlier_mask], bins=20, alpha=0.7, label='정상', color='steelblue')
plt.hist(production[outlier_mask], bins=5, alpha=0.7, label='이상치', color='red')
plt.axvline(mean_val, color='black', linestyle='--', label=f'평균: {mean_val:.0f}')
plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 분포')
plt.legend()

plt.tight_layout()
plt.savefig('02_zscore_outlier.png', dpi=150)
plt.show()

print("'02_zscore_outlier.png' 저장 완료")
print()


# =============================================================
# 4. 이항분포 (불량품 개수)
# =============================================================

print("=" * 60)
print("4. 이항분포 (불량품 개수)")
print("=" * 60)

# 불량률 3%인 제품 100개 검사
n = 100  # 시행 횟수
p = 0.03  # 불량 확률

binom_dist = stats.binom(n, p)

print(f"이항분포 B({n}, {p})")
print(f"기대값 (평균 불량품 수): {binom_dist.mean():.1f}개")
print(f"표준편차: {binom_dist.std():.2f}")
print()

# 확률 계산
print("[확률 계산]")
for k in range(8):
    print(f"P(불량품 = {k}개): {binom_dist.pmf(k):.4f} ({binom_dist.pmf(k)*100:.2f}%)")

print()
print(f"P(불량품 ≤ 5개): {binom_dist.cdf(5):.4f} ({binom_dist.cdf(5)*100:.2f}%)")
print(f"P(불량품 > 5개): {1 - binom_dist.cdf(5):.4f} ({(1-binom_dist.cdf(5))*100:.2f}%)")
print()

# 시각화
plt.figure(figsize=(10, 6))
x = np.arange(0, 15)
pmf = binom_dist.pmf(x)

plt.bar(x, pmf, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(binom_dist.mean(), color='red', linestyle='--', label=f'기대값: {binom_dist.mean():.1f}')
plt.xlabel('불량품 개수')
plt.ylabel('확률')
plt.title(f'이항분포 B({n}, {p}) - 100개 제품 중 불량품 개수')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('03_binomial.png', dpi=150)
plt.show()

print("'03_binomial.png' 저장 완료")
print()


# =============================================================
# 5. 신뢰구간 계산
# =============================================================

print("=" * 60)
print("5. 신뢰구간 계산")
print("=" * 60)

# 샘플 데이터
np.random.seed(42)
sample_data = np.random.normal(1200, 50, 50)  # 50개 샘플

# 기본 통계
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)  # 표본 표준편차
sample_sem = stats.sem(sample_data)  # 표준오차
n_samples = len(sample_data)

print(f"표본 크기: {n_samples}")
print(f"표본 평균: {sample_mean:.2f}")
print(f"표본 표준편차: {sample_std:.2f}")
print(f"표준오차 (SEM): {sample_sem:.2f}")
print()

# 다양한 신뢰수준의 신뢰구간
confidence_levels = [0.90, 0.95, 0.99]

print("[신뢰구간]")
for conf in confidence_levels:
    ci = stats.t.interval(conf, df=n_samples-1, loc=sample_mean, scale=sample_sem)
    print(f"{conf*100:.0f}% 신뢰구간: [{ci[0]:.2f}, {ci[1]:.2f}]")
print()

# 시각화
plt.figure(figsize=(10, 6))

# 히스토그램
plt.hist(sample_data, bins=15, density=True, alpha=0.7, color='steelblue', edgecolor='black')

# 신뢰구간 표시
colors = ['green', 'orange', 'red']
for conf, color in zip(confidence_levels, colors):
    ci = stats.t.interval(conf, df=n_samples-1, loc=sample_mean, scale=sample_sem)
    plt.axvspan(ci[0], ci[1], alpha=0.1, color=color, label=f'{conf*100:.0f}% CI: [{ci[0]:.1f}, {ci[1]:.1f}]')

plt.axvline(sample_mean, color='red', linestyle='--', linewidth=2, label=f'표본 평균: {sample_mean:.1f}')
plt.axvline(1200, color='black', linestyle=':', linewidth=2, label='모평균 (1200)')

plt.xlabel('생산량')
plt.ylabel('밀도')
plt.title('표본 분포와 신뢰구간')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_confidence_interval.png', dpi=150)
plt.show()

print("'04_confidence_interval.png' 저장 완료")
print()


# =============================================================
# 6. 중심극한정리 시뮬레이션
# =============================================================

print("=" * 60)
print("6. 중심극한정리 시뮬레이션")
print("=" * 60)

# 원래 분포: 균등분포 (정규분포가 아님)
original_data = np.random.uniform(0, 100, 10000)

# 다양한 표본 크기에서 표본 평균의 분포
sample_sizes = [5, 10, 30, 100]
n_simulations = 1000

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, sample_size in enumerate(sample_sizes):
    sample_means = []
    for _ in range(n_simulations):
        sample = np.random.choice(original_data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))

    # 정규분포와 비교
    axes[idx].hist(sample_means, bins=30, density=True, alpha=0.7,
                   color='steelblue', edgecolor='black', label='표본 평균 분포')

    # 이론적 정규분포
    mu_theory = np.mean(original_data)
    sigma_theory = np.std(original_data) / np.sqrt(sample_size)
    x_norm = np.linspace(min(sample_means), max(sample_means), 100)
    y_norm = stats.norm.pdf(x_norm, mu_theory, sigma_theory)
    axes[idx].plot(x_norm, y_norm, 'r-', linewidth=2, label='이론적 정규분포')

    axes[idx].set_title(f'표본 크기 n={sample_size}')
    axes[idx].set_xlabel('표본 평균')
    axes[idx].set_ylabel('밀도')
    axes[idx].legend()

plt.suptitle('중심극한정리: 균등분포에서 추출한 표본 평균의 분포', fontsize=14)
plt.tight_layout()
plt.savefig('05_clt_simulation.png', dpi=150)
plt.show()

print("중심극한정리 핵심:")
print("- 원래 분포가 정규분포가 아니어도")
print("- 표본 크기가 충분히 크면 (n≥30)")
print("- 표본 평균의 분포는 정규분포에 가까워짐")
print()
print("'05_clt_simulation.png' 저장 완료")
print()


# =============================================================
# 7. 확률적 예측 vs 점 예측
# =============================================================

print("=" * 60)
print("7. 확률적 예측 vs 점 예측")
print("=" * 60)

# 가상의 AI 예측 결과
np.random.seed(42)
days = np.arange(1, 31)
actual_production = 1200 + np.cumsum(np.random.normal(0, 20, 30))

# 점 예측 (단순 이동평균)
window = 5
point_forecast = np.convolve(actual_production, np.ones(window)/window, mode='valid')
point_forecast = np.concatenate([[actual_production[:window].mean()]*(window-1), point_forecast])

# 불확실성 추정 (과거 오차 기반)
errors = actual_production - point_forecast
uncertainty = np.std(errors) * 1.96  # 95% 신뢰구간

print("[예측 결과 비교]")
print("점 예측: '30일 차 생산량은 {:.0f}개'".format(point_forecast[-1]))
print("확률적 예측: '30일 차 생산량은 {:.0f}개 (95% CI: [{:.0f}, {:.0f}])'".format(
    point_forecast[-1], point_forecast[-1]-uncertainty, point_forecast[-1]+uncertainty))
print()

# 시각화
plt.figure(figsize=(14, 6))

# 실제값
plt.plot(days, actual_production, 'ko-', markersize=4, label='실제 생산량', linewidth=1)

# 점 예측
plt.plot(days, point_forecast, 'b-', linewidth=2, label='점 예측')

# 95% 신뢰구간
plt.fill_between(days, point_forecast - uncertainty, point_forecast + uncertainty,
                 alpha=0.3, color='blue', label='95% 신뢰구간')

plt.xlabel('일자')
plt.ylabel('생산량')
plt.title('점 예측 vs 확률적 예측 (신뢰구간 포함)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_probabilistic_forecast.png', dpi=150)
plt.show()

print("'06_probabilistic_forecast.png' 저장 완료")
print()


# =============================================================
# 8. 확률분포 비교
# =============================================================

print("=" * 60)
print("8. 다양한 확률분포 비교")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 정규분포 (연속)
x_norm = np.linspace(-4, 4, 100)
axes[0, 0].plot(x_norm, stats.norm.pdf(x_norm), 'b-', linewidth=2)
axes[0, 0].fill_between(x_norm, stats.norm.pdf(x_norm), alpha=0.3)
axes[0, 0].set_title('정규분포 N(0, 1)')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('확률밀도')

# 2. 이항분포 (이산)
n, p = 20, 0.3
x_binom = np.arange(0, 21)
axes[0, 1].bar(x_binom, stats.binom.pmf(x_binom, n, p), color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_title(f'이항분포 B({n}, {p})')
axes[0, 1].set_xlabel('성공 횟수')
axes[0, 1].set_ylabel('확률')

# 3. 포아송분포 (이산) - 희귀 사건
lam = 3  # 평균 발생 횟수
x_pois = np.arange(0, 15)
axes[1, 0].bar(x_pois, stats.poisson.pmf(x_pois, lam), color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_title(f'포아송분포 Pois({lam})')
axes[1, 0].set_xlabel('발생 횟수')
axes[1, 0].set_ylabel('확률')

# 4. 지수분포 (연속) - 대기 시간
scale = 10  # 평균 대기 시간
x_exp = np.linspace(0, 50, 100)
axes[1, 1].plot(x_exp, stats.expon.pdf(x_exp, scale=scale), 'purple', linewidth=2)
axes[1, 1].fill_between(x_exp, stats.expon.pdf(x_exp, scale=scale), alpha=0.3, color='purple')
axes[1, 1].set_title(f'지수분포 Exp(λ=1/{scale})')
axes[1, 1].set_xlabel('대기 시간')
axes[1, 1].set_ylabel('확률밀도')

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_distributions_comparison.png', dpi=150)
plt.show()

print("각 분포의 활용:")
print("- 정규분포: 연속적인 측정값 (생산량, 온도 등)")
print("- 이항분포: 성공/실패 횟수 (불량품 개수)")
print("- 포아송분포: 단위 시간당 발생 횟수 (설비 고장)")
print("- 지수분포: 대기 시간 (고장 간 시간 간격)")
print()
print("'07_distributions_comparison.png' 저장 완료")
print()

print("=" * 60)
print("5차시 실습 완료!")
print("=" * 60)
