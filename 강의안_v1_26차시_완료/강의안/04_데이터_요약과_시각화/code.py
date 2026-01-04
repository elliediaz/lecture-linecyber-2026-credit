"""
[4차시] 기술통계의 시각적 이해 - 실습 코드

Matplotlib을 사용한 데이터 시각화 실습입니다.

학습목표:
- 기술통계량(평균, 중앙값, 표준편차)의 의미 이해
- Matplotlib으로 기본 그래프 그리기
- 히스토그램, 상자그림, 산점도 해석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# 0. 한글 폰트 설정 (Windows/Mac/Linux)
# =============================================================

# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'
# Mac: plt.rcParams['font.family'] = 'AppleGothic'
# Linux: plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Matplotlib 시각화 실습")
print("=" * 60)
print()


# =============================================================
# 1. 기술통계량 이해
# =============================================================

print("=" * 60)
print("1. 기술통계량 이해")
print("=" * 60)

# 샘플 데이터 생성
np.random.seed(42)
production_normal = np.random.normal(1200, 50, 100)  # 정규분포

# 이상치가 있는 데이터
production_with_outliers = np.concatenate([
    np.random.normal(1200, 30, 95),  # 대부분의 데이터
    np.array([800, 850, 1500, 1550, 1600])  # 이상치
])

print("[정상 데이터 통계]")
print(f"평균: {np.mean(production_normal):.2f}")
print(f"중앙값: {np.median(production_normal):.2f}")
print(f"표준편차: {np.std(production_normal):.2f}")
print(f"최소값: {np.min(production_normal):.2f}")
print(f"최대값: {np.max(production_normal):.2f}")

print("\n[이상치 포함 데이터 통계]")
print(f"평균: {np.mean(production_with_outliers):.2f}")
print(f"중앙값: {np.median(production_with_outliers):.2f}")
print(f"표준편차: {np.std(production_with_outliers):.2f}")

print("\n→ 이상치가 있으면 평균이 왜곡됨. 중앙값이 더 안정적!")
print()


# =============================================================
# 2. 표준편차의 의미
# =============================================================

print("=" * 60)
print("2. 표준편차의 의미")
print("=" * 60)

# 두 라인 비교: 평균은 같지만 분산이 다름
np.random.seed(42)
line_a = np.random.normal(1200, 10, 100)   # 일관된 품질
line_b = np.random.normal(1200, 80, 100)   # 불안정한 품질

print("[라인 A - 일관된 품질]")
print(f"평균: {np.mean(line_a):.2f}, 표준편차: {np.std(line_a):.2f}")

print("\n[라인 B - 불안정한 품질]")
print(f"평균: {np.mean(line_b):.2f}, 표준편차: {np.std(line_b):.2f}")

print("\n→ 표준편차가 작을수록 품질이 일관됨!")
print()


# =============================================================
# 3. 기본 선 그래프
# =============================================================

print("=" * 60)
print("3. 기본 선 그래프")
print("=" * 60)

# 일별 생산량 데이터
days = np.arange(1, 31)
daily_production = 1200 + np.random.normal(0, 30, 30)

plt.figure(figsize=(12, 5))
plt.plot(days, daily_production, marker='o', linestyle='-', color='steelblue',
         markersize=4, linewidth=1.5)
plt.axhline(y=np.mean(daily_production), color='red', linestyle='--',
            label=f'평균: {np.mean(daily_production):.0f}')
plt.xlabel('일자')
plt.ylabel('생산량')
plt.title('일별 생산량 추이')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_line_graph.png', dpi=150)
plt.show()

print("'01_line_graph.png' 저장 완료")
print()


# =============================================================
# 4. 히스토그램 (Histogram)
# =============================================================

print("=" * 60)
print("4. 히스토그램")
print("=" * 60)

# 생산량 데이터 (정규분포)
np.random.seed(42)
production_data = np.random.normal(1200, 50, 1000)

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(production_data, bins=30, edgecolor='black',
                            alpha=0.7, color='steelblue')

# 평균과 중앙값 표시
mean_val = np.mean(production_data)
median_val = np.median(production_data)

plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
            label=f'평균: {mean_val:.1f}')
plt.axvline(median_val, color='green', linestyle=':', linewidth=2,
            label=f'중앙값: {median_val:.1f}')

plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 분포 (히스토그램)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_histogram.png', dpi=150)
plt.show()

print("'02_histogram.png' 저장 완료")
print()


# =============================================================
# 5. 분포 형태 비교
# =============================================================

print("=" * 60)
print("5. 분포 형태 비교")
print("=" * 60)

np.random.seed(42)

# 세 가지 분포
normal_dist = np.random.normal(100, 15, 1000)           # 정규분포
right_skewed = np.random.exponential(20, 1000) + 50     # 오른쪽 꼬리
left_skewed = 150 - np.random.exponential(20, 1000)     # 왼쪽 꼬리

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 정규분포
axes[0].hist(normal_dist, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(np.mean(normal_dist), color='red', linestyle='--', label='평균')
axes[0].axvline(np.median(normal_dist), color='green', linestyle=':', label='중앙값')
axes[0].set_title('정규분포 (대칭)')
axes[0].set_xlabel('값')
axes[0].legend()

# 오른쪽 꼬리
axes[1].hist(right_skewed, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(np.mean(right_skewed), color='red', linestyle='--', label='평균')
axes[1].axvline(np.median(right_skewed), color='green', linestyle=':', label='중앙값')
axes[1].set_title('오른쪽 꼬리 (평균 > 중앙값)')
axes[1].set_xlabel('값')
axes[1].legend()

# 왼쪽 꼬리
axes[2].hist(left_skewed, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[2].axvline(np.mean(left_skewed), color='red', linestyle='--', label='평균')
axes[2].axvline(np.median(left_skewed), color='green', linestyle=':', label='중앙값')
axes[2].set_title('왼쪽 꼬리 (평균 < 중앙값)')
axes[2].set_xlabel('값')
axes[2].legend()

plt.tight_layout()
plt.savefig('03_distribution_types.png', dpi=150)
plt.show()

print("'03_distribution_types.png' 저장 완료")
print()


# =============================================================
# 6. 상자그림 (Box Plot)
# =============================================================

print("=" * 60)
print("6. 상자그림")
print("=" * 60)

np.random.seed(42)

# 라인별 생산량 데이터
line1 = np.random.normal(1200, 30, 100)
line2 = np.random.normal(1180, 60, 100)
line3 = np.concatenate([np.random.normal(1220, 40, 95),
                        np.array([900, 950, 1400, 1450, 1500])])  # 이상치 포함

plt.figure(figsize=(10, 6))
box_data = [line1, line2, line3]
bp = plt.boxplot(box_data, labels=['라인1\n(일관)', '라인2\n(변동)', '라인3\n(이상치)'],
                 patch_artist=True)

# 색상 지정
colors = ['steelblue', 'coral', 'green']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('생산량')
plt.title('라인별 생산량 분포 (상자그림)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('04_boxplot.png', dpi=150)
plt.show()

print("'04_boxplot.png' 저장 완료")

# 상자그림 해석
print("\n[상자그림 해석]")
for i, (name, data) in enumerate([('라인1', line1), ('라인2', line2), ('라인3', line3)]):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    print(f"{name}: Q1={q1:.0f}, 중앙값={np.median(data):.0f}, Q3={q3:.0f}, 이상치 {len(outliers)}개")
print()


# =============================================================
# 7. 산점도 (Scatter Plot)
# =============================================================

print("=" * 60)
print("7. 산점도")
print("=" * 60)

np.random.seed(42)

# 온도와 불량률 관계 (양의 상관)
temperature = np.random.normal(85, 5, 100)
defect_rate = 0.02 + 0.003 * (temperature - 80) + np.random.normal(0, 0.005, 100)
defect_rate = np.clip(defect_rate, 0, 1)  # 0~1 범위로 제한

plt.figure(figsize=(10, 6))
plt.scatter(temperature, defect_rate * 100, alpha=0.6, c='steelblue', s=50)
plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률의 관계')

# 추세선 추가
z = np.polyfit(temperature, defect_rate * 100, 1)
p = np.poly1d(z)
temp_sorted = np.sort(temperature)
plt.plot(temp_sorted, p(temp_sorted), color='red', linestyle='--',
         label=f'추세선 (기울기: {z[0]:.3f})')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('05_scatter.png', dpi=150)
plt.show()

# 상관계수 계산
corr = np.corrcoef(temperature, defect_rate)[0, 1]
print(f"상관계수: {corr:.3f}")
print("→ 양의 상관관계: 온도가 높을수록 불량률도 증가")
print("'05_scatter.png' 저장 완료")
print()


# =============================================================
# 8. 막대 그래프 (Bar Plot)
# =============================================================

print("=" * 60)
print("8. 막대 그래프")
print("=" * 60)

# 라인별 평균 생산량
lines = ['라인1', '라인2', '라인3', '라인4']
avg_production = [1200, 1150, 1280, 1100]
std_production = [30, 50, 40, 60]  # 표준편차 (에러바용)

plt.figure(figsize=(10, 6))
bars = plt.bar(lines, avg_production, yerr=std_production,
               color=['steelblue', 'coral', 'green', 'purple'],
               capsize=5, alpha=0.7, edgecolor='black')

# 목표선 추가
plt.axhline(y=1200, color='red', linestyle='--', label='목표: 1200')

# 값 표시
for bar, val in zip(bars, avg_production):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val}', ha='center', va='bottom', fontsize=12)

plt.xlabel('생산 라인')
plt.ylabel('평균 생산량')
plt.title('라인별 평균 생산량 비교')
plt.legend()
plt.ylim(0, 1400)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('06_barplot.png', dpi=150)
plt.show()

print("'06_barplot.png' 저장 완료")
print()


# =============================================================
# 9. 여러 그래프 한 번에 (Subplot)
# =============================================================

print("=" * 60)
print("9. 종합 대시보드 (Subplot)")
print("=" * 60)

# 제조 데이터 생성
np.random.seed(42)
n = 200

df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=n),
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(25, n),
    '온도': np.random.normal(85, 5, n),
    '라인': np.random.choice(['라인1', '라인2', '라인3'], n)
})
df['불량률'] = df['불량수'] / df['생산량']

# 4분할 대시보드
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) 생산량 히스토그램
axes[0, 0].hist(df['생산량'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['생산량'].mean(), color='red', linestyle='--',
                   label=f"평균: {df['생산량'].mean():.0f}")
axes[0, 0].set_xlabel('생산량')
axes[0, 0].set_ylabel('빈도')
axes[0, 0].set_title('생산량 분포')
axes[0, 0].legend()

# (0,1) 라인별 상자그림
line_data = [df[df['라인'] == line]['생산량'].values
             for line in ['라인1', '라인2', '라인3']]
bp = axes[0, 1].boxplot(line_data, labels=['라인1', '라인2', '라인3'], patch_artist=True)
colors = ['steelblue', 'coral', 'green']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel('생산량')
axes[0, 1].set_title('라인별 생산량 비교')

# (1,0) 일별 생산량 추이 (처음 30일)
df_30 = df.head(30)
axes[1, 0].plot(df_30['날짜'], df_30['생산량'], marker='o', markersize=4,
                color='steelblue', linewidth=1)
axes[1, 0].axhline(y=df['생산량'].mean(), color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('날짜')
axes[1, 0].set_ylabel('생산량')
axes[1, 0].set_title('일별 생산량 추이 (30일)')
axes[1, 0].tick_params(axis='x', rotation=45)

# (1,1) 온도 vs 불량률 산점도
axes[1, 1].scatter(df['온도'], df['불량률'] * 100, alpha=0.5, c='steelblue', s=30)
axes[1, 1].set_xlabel('온도 (°C)')
axes[1, 1].set_ylabel('불량률 (%)')
axes[1, 1].set_title('온도와 불량률 관계')

plt.tight_layout()
plt.savefig('07_dashboard.png', dpi=150)
plt.show()

print("'07_dashboard.png' 저장 완료")
print()


# =============================================================
# 10. 그래프 스타일 커스터마이징
# =============================================================

print("=" * 60)
print("10. 그래프 스타일 커스터마이징")
print("=" * 60)

# 여러 스타일 비교
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)

# 기본 스타일
axes[0, 0].plot(x, y, 'b-')
axes[0, 0].set_title('기본 스타일')

# 마커 + 색상
axes[0, 1].plot(x, y, 'ro-', markersize=4, linewidth=1)
axes[0, 1].set_title("마커 추가 (color='r', marker='o')")

# 라인 스타일
axes[1, 0].plot(x, y, 'g--', linewidth=2)
axes[1, 1].plot(x, y, 'k:', linewidth=2)
axes[1, 0].set_title("점선 (linestyle='--')")
axes[1, 1].set_title("점점선 (linestyle=':')")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()
plt.savefig('08_styles.png', dpi=150)
plt.show()

print("'08_styles.png' 저장 완료")
print()


# =============================================================
# 11. 그래프 저장 옵션
# =============================================================

print("=" * 60)
print("11. 그래프 저장 옵션")
print("=" * 60)

# 샘플 그래프
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4, 5], [2, 4, 1, 5, 3], 'b-o')
plt.title('저장 테스트')
plt.xlabel('X')
plt.ylabel('Y')

# 다양한 포맷으로 저장
plt.savefig('save_low.png', dpi=72)      # 저해상도 (웹용)
plt.savefig('save_high.png', dpi=300)    # 고해상도 (인쇄용)
plt.savefig('save_tight.png', dpi=150, bbox_inches='tight')  # 여백 최소화
plt.close()

print("저장 옵션:")
print("- dpi=72: 웹용 (파일 크기 작음)")
print("- dpi=300: 인쇄/보고서용 (고품질)")
print("- bbox_inches='tight': 여백 자동 조절")
print()


# =============================================================
# 12. 종합 실습: 제조 품질 분석 보고서
# =============================================================

print("=" * 60)
print("12. 종합 실습: 제조 품질 분석 보고서")
print("=" * 60)

# 분석 보고서 생성
fig = plt.figure(figsize=(16, 12))

# 레이아웃 설정
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 생산량 히스토그램 (왼쪽 상단)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['생산량'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(df['생산량'].mean(), color='red', linestyle='--')
ax1.set_title('생산량 분포')
ax1.set_xlabel('생산량')

# 2. 불량률 히스토그램 (중앙 상단)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['불량률'] * 100, bins=25, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(df['불량률'].mean() * 100, color='red', linestyle='--')
ax2.set_title('불량률 분포')
ax2.set_xlabel('불량률 (%)')

# 3. 라인별 생산량 비교 (오른쪽 상단)
ax3 = fig.add_subplot(gs[0, 2])
line_means = df.groupby('라인')['생산량'].mean()
line_stds = df.groupby('라인')['생산량'].std()
ax3.bar(line_means.index, line_means.values, yerr=line_stds.values,
        capsize=5, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax3.axhline(y=1200, color='red', linestyle='--', label='목표')
ax3.set_title('라인별 평균 생산량')
ax3.set_ylabel('생산량')
ax3.legend()

# 4. 생산량 추이 (전체 중앙)
ax4 = fig.add_subplot(gs[1, :])
for line, color in zip(['라인1', '라인2', '라인3'], ['steelblue', 'coral', 'green']):
    line_df = df[df['라인'] == line].head(50)
    ax4.plot(line_df['날짜'], line_df['생산량'], label=line, alpha=0.7, marker='o', markersize=3)
ax4.set_title('라인별 생산량 추이 (50일)')
ax4.set_xlabel('날짜')
ax4.set_ylabel('생산량')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

# 5. 온도 vs 불량률 (왼쪽 하단)
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(df['온도'], df['불량률'] * 100, alpha=0.4, c='steelblue', s=20)
ax5.set_title('온도 vs 불량률')
ax5.set_xlabel('온도 (°C)')
ax5.set_ylabel('불량률 (%)')

# 6. 라인별 상자그림 (중앙 하단)
ax6 = fig.add_subplot(gs[2, 1])
line_data = [df[df['라인'] == line]['불량률'].values * 100
             for line in ['라인1', '라인2', '라인3']]
bp = ax6.boxplot(line_data, labels=['라인1', '라인2', '라인3'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'coral', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_title('라인별 불량률 분포')
ax6.set_ylabel('불량률 (%)')

# 7. 요약 통계 텍스트 (오른쪽 하단)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = f"""
[데이터 요약]

총 데이터 수: {len(df):,}건
분석 기간: {df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}

[생산량 통계]
평균: {df['생산량'].mean():,.1f}
표준편차: {df['생산량'].std():,.1f}
최소: {df['생산량'].min():,.1f}
최대: {df['생산량'].max():,.1f}

[불량률 통계]
평균: {df['불량률'].mean()*100:.2f}%
표준편차: {df['불량률'].std()*100:.2f}%

[상관분석]
온도-불량률: {df['온도'].corr(df['불량률']):.3f}
"""
ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.suptitle('제조 품질 분석 보고서', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('09_quality_report.png', dpi=200, bbox_inches='tight')
plt.show()

print("'09_quality_report.png' 저장 완료")
print()
print("=" * 60)
print("4차시 실습 완료!")
print("=" * 60)
