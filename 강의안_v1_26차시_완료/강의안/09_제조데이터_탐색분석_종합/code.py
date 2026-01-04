# [9차시] 제조 데이터 탐색 분석 종합 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("9차시: 제조 데이터 탐색 분석(EDA) 종합")
print("Part II 마무리: 데이터 이해부터 인사이트까지")
print("=" * 60)
print()


# ============================================================
# 0. 데이터 생성 (6개월 제조 데이터)
# ============================================================

print("=" * 60)
print("0. 제조 데이터 생성")
print("=" * 60)

np.random.seed(42)
n = 1000  # 6개월, 하루 약 5~6건

# 날짜 생성
dates = pd.date_range('2024-01-01', periods=n, freq='4H')

# 데이터 생성
df = pd.DataFrame({
    '날짜': dates,
    '라인': np.random.choice(['A', 'B', 'C'], n, p=[0.4, 0.35, 0.25]),
    '시간대': np.where(dates.hour < 12, '오전', '오후'),
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '생산량': np.random.normal(1200, 100, n),
})

# 불량수 생성 (온도, 라인에 따라 다르게)
base_defect = 20
df['불량수'] = (
    base_defect +
    (df['온도'] - 85) * 1.5 +  # 온도 영향
    np.where(df['라인'] == 'B', 8, 0) +  # 라인 B가 높음
    np.where(df['시간대'] == '오후', 3, 0) +  # 오후가 높음
    np.random.normal(0, 5, n)
).astype(int).clip(0)

df['불량률'] = df['불량수'] / df['생산량']

# 결측치 추가 (약 5%)
missing_idx = np.random.choice(n, int(n * 0.05), replace=False)
df.loc[missing_idx[:len(missing_idx)//2], '온도'] = np.nan
df.loc[missing_idx[len(missing_idx)//2:], '습도'] = np.nan

# 이상치 추가
outlier_idx = np.random.choice(n, 20, replace=False)
df.loc[outlier_idx[:10], '온도'] = np.random.choice([60, 65, 105, 110], 10)
df.loc[outlier_idx[10:], '불량수'] = np.random.choice([80, 90, 100], 10)

print(f"데이터 생성 완료: {len(df)}행")
print()


# ============================================================
# 1단계: 데이터 개요 파악
# ============================================================

print("=" * 60)
print("1단계: 데이터 개요 파악")
print("=" * 60)

print("[데이터 크기]")
print(f"행: {df.shape[0]}, 열: {df.shape[1]}")
print()

print("[데이터 타입]")
print(df.dtypes)
print()

print("[처음 5행]")
print(df.head())
print()

print("[기술통계]")
print(df.describe())
print()

print("[결측치 현황]")
print(df.isnull().sum())
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"총 결측치: {total_missing}개 ({total_missing/total_cells*100:.2f}%)")
print()


# ============================================================
# 2단계: 단변량 분석
# ============================================================

print("=" * 60)
print("2단계: 단변량 분석")
print("=" * 60)

# 수치형 변수: 불량률 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 불량률 히스토그램
axes[0].hist(df['불량률']*100, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df['불량률'].mean()*100, color='red', linestyle='--',
                label=f'평균: {df["불량률"].mean()*100:.2f}%')
axes[0].axvline(df['불량률'].median()*100, color='green', linestyle=':',
                label=f'중앙값: {df["불량률"].median()*100:.2f}%')
axes[0].set_xlabel('불량률 (%)')
axes[0].set_ylabel('빈도')
axes[0].set_title('불량률 분포')
axes[0].legend()

# 온도 히스토그램
temp_data = df['온도'].dropna()
axes[1].hist(temp_data, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(temp_data.mean(), color='red', linestyle='--',
                label=f'평균: {temp_data.mean():.1f}°C')
axes[1].set_xlabel('온도 (°C)')
axes[1].set_ylabel('빈도')
axes[1].set_title('온도 분포')
axes[1].legend()

# 생산량 상자그림
bp = axes[2].boxplot(df['생산량'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
axes[2].set_ylabel('생산량')
axes[2].set_title('생산량 분포')

plt.tight_layout()
plt.show()

print("단변량 분석 시각화 완료")
print()

# 범주형 변수: 라인, 시간대 빈도
print("[범주형 변수 빈도]")
print("\n라인별 빈도:")
print(df['라인'].value_counts())
print("\n시간대별 빈도:")
print(df['시간대'].value_counts())
print()


# ============================================================
# 3단계: 이변량 분석
# ============================================================

print("=" * 60)
print("3단계: 이변량 분석")
print("=" * 60)

# 3-1. 범주 vs 수치: 라인별 불량률
print("[라인별 평균 불량률]")
line_defect = df.groupby('라인')['불량률'].agg(['mean', 'std', 'count'])
line_defect['mean'] = line_defect['mean'] * 100
line_defect['std'] = line_defect['std'] * 100
print(line_defect.round(2))
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 라인별 불량률
line_means = df.groupby('라인')['불량률'].mean() * 100
axes[0].bar(line_means.index, line_means.values,
            color=['steelblue', 'coral', 'green'], edgecolor='black')
axes[0].axhline(df['불량률'].mean()*100, color='red', linestyle='--', label='전체 평균')
axes[0].set_ylabel('불량률 (%)')
axes[0].set_xlabel('라인')
axes[0].set_title('라인별 평균 불량률')
axes[0].legend()

# 시간대별 불량률
time_means = df.groupby('시간대')['불량률'].mean() * 100
axes[1].bar(time_means.index, time_means.values,
            color=['coral', 'steelblue'], edgecolor='black')
axes[1].axhline(df['불량률'].mean()*100, color='red', linestyle='--', label='전체 평균')
axes[1].set_ylabel('불량률 (%)')
axes[1].set_xlabel('시간대')
axes[1].set_title('시간대별 평균 불량률')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3-2. 수치 vs 수치: 온도와 불량률
print("[온도와 불량률의 관계]")
temp_clean = df['온도'].dropna()
defect_clean = df.loc[df['온도'].notna(), '불량률']

r = np.corrcoef(temp_clean, defect_clean)[0, 1]
print(f"상관계수: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(temp_clean, defect_clean*100, alpha=0.3, c='steelblue', s=30)

# 추세선
z = np.polyfit(temp_clean, defect_clean*100, 1)
p = np.poly1d(z)
temp_sorted = np.sort(temp_clean)
plt.plot(temp_sorted, p(temp_sorted), 'r--', linewidth=2, label=f'추세선 (r={r:.3f})')

plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도 vs 불량률')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print()


# ============================================================
# 4단계: 다변량 분석
# ============================================================

print("=" * 60)
print("4단계: 다변량 분석")
print("=" * 60)

# 4-1. 상관행렬
numeric_cols = ['온도', '습도', '생산량', '불량수', '불량률']
corr_matrix = df[numeric_cols].corr()

print("[상관계수 행렬]")
print(corr_matrix.round(3))
print()

# 상관계수 히트맵
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, label='상관계수')
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
plt.yticks(range(len(numeric_cols)), numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', ha='center', va='center')
plt.title('상관계수 히트맵')
plt.tight_layout()
plt.show()

# 4-2. 피벗 테이블: 라인 x 시간대
print("[라인 x 시간대별 평균 불량률]")
pivot = df.pivot_table(values='불량률', index='라인', columns='시간대', aggfunc='mean')
print((pivot * 100).round(2))
print()

# 히트맵
plt.figure(figsize=(8, 6))
im = plt.imshow(pivot * 100, cmap='YlOrRd')
plt.colorbar(im, label='불량률 (%)')
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        plt.text(j, i, f'{pivot.iloc[i,j]*100:.2f}%', ha='center', va='center')
plt.title('라인 x 시간대별 평균 불량률')
plt.xlabel('시간대')
plt.ylabel('라인')
plt.tight_layout()
plt.show()

# 4-3. 월별 추이
df['월'] = df['날짜'].dt.to_period('M')
monthly = df.groupby('월')['불량률'].mean() * 100

plt.figure(figsize=(12, 5))
monthly.plot(kind='line', marker='o', color='steelblue', linewidth=2)
plt.axhline(monthly.mean(), color='red', linestyle='--', label=f'평균: {monthly.mean():.2f}%')
plt.title('월별 평균 불량률 추이')
plt.xlabel('월')
plt.ylabel('불량률 (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print()


# ============================================================
# 5단계: 통계적 검정
# ============================================================

print("=" * 60)
print("5단계: 통계적 검정")
print("=" * 60)

# 라인 B vs 다른 라인 비교
line_b = df[df['라인'] == 'B']['불량률']
line_other = df[df['라인'] != 'B']['불량률']

t_stat, p_value = stats.ttest_ind(line_b, line_other)

print("[라인 B vs 다른 라인 불량률 t-검정]")
print(f"라인 B 평균: {line_b.mean()*100:.2f}%")
print(f"다른 라인 평균: {line_other.mean()*100:.2f}%")
print(f"차이: {(line_b.mean() - line_other.mean())*100:.2f}%p")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.05:
    print("→ 라인 B의 불량률이 통계적으로 유의미하게 높음 (p < 0.05)")
print()

# 오전 vs 오후 비교
am = df[df['시간대'] == '오전']['불량률']
pm = df[df['시간대'] == '오후']['불량률']

t_stat2, p_value2 = stats.ttest_ind(am, pm)

print("[오전 vs 오후 불량률 t-검정]")
print(f"오전 평균: {am.mean()*100:.2f}%")
print(f"오후 평균: {pm.mean()*100:.2f}%")
print(f"차이: {(pm.mean() - am.mean())*100:.2f}%p")
print(f"p-value: {p_value2:.6f}")
if p_value2 < 0.05:
    print("→ 오후 불량률이 통계적으로 유의미하게 높음 (p < 0.05)")
print()


# ============================================================
# 6단계: 이상치 분석
# ============================================================

print("=" * 60)
print("6단계: 이상치 분석")
print("=" * 60)

# 온도 이상치
Q1_temp = df['온도'].quantile(0.25)
Q3_temp = df['온도'].quantile(0.75)
IQR_temp = Q3_temp - Q1_temp
outliers_temp = (df['온도'] < Q1_temp - 1.5*IQR_temp) | (df['온도'] > Q3_temp + 1.5*IQR_temp)

print(f"[온도 이상치]")
print(f"정상 범위: [{Q1_temp - 1.5*IQR_temp:.1f}, {Q3_temp + 1.5*IQR_temp:.1f}]°C")
print(f"이상치 수: {outliers_temp.sum()}개")

if outliers_temp.sum() > 0:
    normal_defect = df.loc[~outliers_temp, '불량률'].mean()
    outlier_defect = df.loc[outliers_temp, '불량률'].mean()
    print(f"정상 온도 데이터의 평균 불량률: {normal_defect*100:.2f}%")
    print(f"온도 이상치의 평균 불량률: {outlier_defect*100:.2f}%")
print()


# ============================================================
# 7단계: 인사이트 정리
# ============================================================

print("=" * 60)
print("7단계: 인사이트 정리")
print("=" * 60)

line_b_diff = (df[df['라인']=='B']['불량률'].mean() - df['불량률'].mean()) * 100
time_diff = (pm.mean() - am.mean()) * 100

print("""
[EDA 주요 발견사항]

1. 데이터 품질
   - 결측치: 약 5% (온도, 습도)
   - 이상치: 온도에서 일부 극단값 발견

2. 불량률 주요 요인
   ① 라인별 차이: 라인 B가 평균 대비 약 {:.1f}%p 높음 (p < 0.05)
   ② 시간대별 차이: 오후가 오전보다 약 {:.1f}%p 높음
   ③ 온도 영향: 온도와 불량률의 상관계수 {:.3f}

3. 복합 요인
   - 라인 B + 오후 조합에서 가장 높은 불량률

[권고사항]

1. 라인 B 집중 점검
   - 설비 상태 확인
   - 작업 환경 비교 분석

2. 온도 관리 강화
   - 85°C 이상에서 불량률 증가 경향
   - 온도 모니터링 시스템 강화

3. 오후 시간대 품질 관리
   - 작업자 피로도 관리
   - 환경 조건 재검토
""".format(line_b_diff, time_diff, r))


# ============================================================
# 8단계: 종합 대시보드
# ============================================================

print("=" * 60)
print("8단계: 종합 대시보드 생성")
print("=" * 60)

fig = plt.figure(figsize=(16, 12))

# 1. 불량률 분포
ax1 = fig.add_subplot(2, 3, 1)
ax1.hist(df['불량률']*100, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(df['불량률'].mean()*100, color='red', linestyle='--', label=f"평균: {df['불량률'].mean()*100:.2f}%")
ax1.set_title('불량률 분포')
ax1.set_xlabel('불량률 (%)')
ax1.legend()

# 2. 라인별 불량률
ax2 = fig.add_subplot(2, 3, 2)
line_means = df.groupby('라인')['불량률'].mean() * 100
ax2.bar(line_means.index, line_means.values, color=['steelblue', 'coral', 'green'], edgecolor='black')
ax2.axhline(df['불량률'].mean()*100, color='red', linestyle='--')
ax2.set_title('라인별 평균 불량률')
ax2.set_ylabel('불량률 (%)')

# 3. 시간대별 불량률
ax3 = fig.add_subplot(2, 3, 3)
time_means = df.groupby('시간대')['불량률'].mean() * 100
ax3.bar(time_means.index, time_means.values, color=['coral', 'steelblue'], edgecolor='black')
ax3.axhline(df['불량률'].mean()*100, color='red', linestyle='--')
ax3.set_title('시간대별 평균 불량률')
ax3.set_ylabel('불량률 (%)')

# 4. 온도 vs 불량률
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(df['온도'], df['불량률']*100, alpha=0.3, c='steelblue', s=20)
ax4.set_xlabel('온도 (°C)')
ax4.set_ylabel('불량률 (%)')
ax4.set_title(f'온도 vs 불량률 (r={r:.3f})')

# 5. 월별 추이
ax5 = fig.add_subplot(2, 3, 5)
monthly.plot(kind='line', marker='o', ax=ax5, color='steelblue')
ax5.axhline(monthly.mean(), color='red', linestyle='--')
ax5.set_title('월별 불량률 추이')
ax5.set_ylabel('불량률 (%)')

# 6. 라인 x 시간대 히트맵
ax6 = fig.add_subplot(2, 3, 6)
im = ax6.imshow(pivot * 100, cmap='YlOrRd')
ax6.set_xticks(range(len(pivot.columns)))
ax6.set_xticklabels(pivot.columns)
ax6.set_yticks(range(len(pivot.index)))
ax6.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax6.text(j, i, f'{pivot.iloc[i,j]*100:.1f}%', ha='center', va='center')
ax6.set_title('라인 x 시간대 불량률')
plt.colorbar(im, ax=ax6)

plt.suptitle('제조 품질 분석 대시보드', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("종합 대시보드 생성 완료!")
print()


# ============================================================
# 최종 요약
# ============================================================

print("=" * 60)
print("9차시 실습 완료!")
print("=" * 60)
print("""
[EDA 5단계 체크리스트]
✓ 1단계: 데이터 개요 (shape, info, describe)
✓ 2단계: 단변량 분석 (히스토그램, 상자그림)
✓ 3단계: 이변량 분석 (상관계수, 그룹 비교)
✓ 4단계: 다변량 분석 (피벗 테이블, 히트맵)
✓ 5단계: 인사이트 도출 (통계 검정, 권고사항)

Part II 완료! 수고하셨습니다!
다음 Part III에서 머신러닝을 시작합니다!
""")
print("=" * 60)
