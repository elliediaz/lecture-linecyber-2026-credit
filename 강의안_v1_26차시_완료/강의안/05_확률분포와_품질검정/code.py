# [5차시] 확률분포와 품질 검정 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 실습 1: 정규분포 시각화
# ============================================================
print("=" * 50)
print("실습 1: 정규분포 시각화")
print("=" * 50)

np.random.seed(42)

# 정규분포 데이터 생성 (평균=1200, 표준편차=50)
production = np.random.normal(1200, 50, 1000)

plt.figure(figsize=(10, 6))
plt.hist(production, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(1200, color='red', linestyle='--', label='평균')
plt.axvline(1200-50, color='orange', linestyle=':', label='-1σ')
plt.axvline(1200+50, color='orange', linestyle=':', label='+1σ')
plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 정규분포 (μ=1200, σ=50)')
plt.legend()
plt.show()

# ============================================================
# 실습 2: 68-95-99.7 규칙 확인
# ============================================================
print("\n" + "=" * 50)
print("실습 2: 68-95-99.7 규칙 확인")
print("=" * 50)

mean = np.mean(production)
std = np.std(production)

# 각 범위의 데이터 비율 계산
within_1std = np.sum((production >= mean - std) &
                      (production <= mean + std)) / len(production)
within_2std = np.sum((production >= mean - 2*std) &
                      (production <= mean + 2*std)) / len(production)
within_3std = np.sum((production >= mean - 3*std) &
                      (production <= mean + 3*std)) / len(production)

print(f"평균: {mean:.1f}")
print(f"표준편차: {std:.1f}")
print()
print(f"±1σ 범위 ({mean-std:.0f} ~ {mean+std:.0f}): {within_1std:.1%}")
print(f"±2σ 범위 ({mean-2*std:.0f} ~ {mean+2*std:.0f}): {within_2std:.1%}")
print(f"±3σ 범위 ({mean-3*std:.0f} ~ {mean+3*std:.0f}): {within_3std:.1%}")

# ============================================================
# 실습 3: Z-score 이상치 탐지
# ============================================================
print("\n" + "=" * 50)
print("실습 3: Z-score 이상치 탐지")
print("=" * 50)

# 실제 생산 데이터 (하나의 이상치 포함)
daily_production = np.array([1185, 1210, 1195, 1180, 1420, 1200, 1190])
days = ['월', '화', '수', '목', '금', '토', '일']

# Z-score 계산
mean = daily_production.mean()
std = daily_production.std()
z_scores = (daily_production - mean) / std

print(f"평균: {mean:.1f}")
print(f"표준편차: {std:.1f}")
print()
print("일별 생산량과 Z-score:")
print("-" * 40)
for day, prod, z in zip(days, daily_production, z_scores):
    status = "이상치!" if abs(z) > 2 else "정상"
    print(f"{day}: {prod}개 (Z={z:+.2f}) - {status}")

# ============================================================
# 실습 4: 이상치 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 4: 이상치 시각화")
print("=" * 50)

plt.figure(figsize=(10, 6))

colors = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
bars = plt.bar(days, daily_production, color=colors, edgecolor='black')

plt.axhline(mean, color='green', linestyle='--', label=f'평균: {mean:.0f}')
plt.axhline(mean + 2*std, color='orange', linestyle=':', label=f'+2σ: {mean+2*std:.0f}')
plt.axhline(mean - 2*std, color='orange', linestyle=':', label=f'-2σ: {mean-2*std:.0f}')

plt.xlabel('요일')
plt.ylabel('생산량')
plt.title('일별 생산량과 이상치 탐지 (빨간색 = 이상치)')
plt.legend()
plt.show()

print("빨간색 막대가 이상치입니다.")

# ============================================================
# 실습 5: 라인별 데이터 준비
# ============================================================
print("\n" + "=" * 50)
print("실습 5: 라인별 데이터 준비")
print("=" * 50)

np.random.seed(123)

# 라인 A: 평균 불량률 2.2%
line_a_defect = np.random.normal(2.2, 0.3, 30)

# 라인 B: 평균 불량률 2.8%
line_b_defect = np.random.normal(2.8, 0.35, 30)

print("=== 라인별 불량률 통계 ===")
print(f"라인 A: 평균 {line_a_defect.mean():.2f}%, 표준편차 {line_a_defect.std():.2f}")
print(f"라인 B: 평균 {line_b_defect.mean():.2f}%, 표준편차 {line_b_defect.std():.2f}")

# ============================================================
# 실습 6: t-검정 수행
# ============================================================
print("\n" + "=" * 50)
print("실습 6: t-검정 수행")
print("=" * 50)

# 독립표본 t-검정
t_stat, p_value = stats.ttest_ind(line_a_defect, line_b_defect)

print("=== t-검정 결과 ===")
print(f"t-통계량: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")
print()

if p_value < 0.05:
    print("결론: 두 라인의 불량률 차이가 통계적으로 유의미합니다.")
    print("      → 라인 B의 품질 개선이 필요합니다.")
else:
    print("결론: 두 라인의 불량률 차이가 유의미하지 않습니다.")

# ============================================================
# 실습 7: 검정 결과 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 7: 검정 결과 시각화")
print("=" * 50)

plt.figure(figsize=(10, 6))

box_data = plt.boxplot([line_a_defect, line_b_defect],
                        labels=['라인 A', '라인 B'],
                        patch_artist=True)

# 색상 설정
colors = ['lightblue', 'lightcoral']
for patch, color in zip(box_data['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('불량률 (%)')
plt.title(f'라인별 불량률 비교 (p-value: {p_value:.4f})')

# 유의성 표시
if p_value < 0.05:
    plt.annotate('* 유의미한 차이', xy=(1.5, max(line_b_defect)),
                 fontsize=12, color='red', ha='center')

plt.show()

# ============================================================
# 실습 8: 종합 분석 리포트
# ============================================================
print("\n" + "=" * 50)
print("실습 8: 종합 분석 리포트")
print("=" * 50)

print("=" * 50)
print("           품질 관리 분석 리포트")
print("=" * 50)

# 이상치 현황
outlier_count = np.sum(np.abs(z_scores) > 2)
outlier_days = [day for day, z in zip(days, z_scores) if abs(z) > 2]

print(f"\n[이상치 탐지 결과]")
print(f"분석 기간: 7일")
print(f"이상치 발생: {outlier_count}건")
if outlier_days:
    print(f"이상치 발생일: {', '.join(outlier_days)}")

# 라인 비교
print(f"\n[라인별 품질 비교]")
print(f"라인 A 평균 불량률: {line_a_defect.mean():.2f}%")
print(f"라인 B 평균 불량률: {line_b_defect.mean():.2f}%")
print(f"차이: {abs(line_a_defect.mean() - line_b_defect.mean()):.2f}%p")
print(f"통계적 유의성: {'있음' if p_value < 0.05 else '없음'} (p={p_value:.4f})")

# 권고사항
print(f"\n[권고사항]")
if outlier_count > 0:
    print(f"- 이상치 발생일({', '.join(outlier_days)})의 원인 조사 필요")
if p_value < 0.05 and line_b_defect.mean() > line_a_defect.mean():
    print(f"- 라인 B의 불량률이 높음, 설비 점검 및 공정 개선 권고")

print("=" * 50)

# ============================================================
# 추가 실습: scipy.stats 확률 계산
# ============================================================
print("\n" + "=" * 50)
print("추가: scipy.stats 확률 계산")
print("=" * 50)

# 정규분포 객체 생성
dist = stats.norm(loc=1200, scale=50)

# 확률 계산
prob_below_1150 = dist.cdf(1150)
prob_above_1300 = 1 - dist.cdf(1300)
prob_between = dist.cdf(1250) - dist.cdf(1150)

print(f"생산량 정규분포 N(1200, 50²)")
print(f"- 1150개 이하 확률: {prob_below_1150:.1%}")
print(f"- 1300개 이상 확률: {prob_above_1300:.1%}")
print(f"- 1150~1250 사이 확률: {prob_between:.1%}")
