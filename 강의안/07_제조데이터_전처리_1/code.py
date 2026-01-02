# [7차시] 제조 데이터 전처리 (1) - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 실습 1: 결측치 있는 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 결측치 있는 데이터 생성")
print("=" * 50)

np.random.seed(42)
n = 100

df = pd.DataFrame({
    '일자': pd.date_range('2024-01-01', periods=n),
    '온도': np.random.normal(85, 5, n),
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(30, n),
    '라인': np.random.choice(['A', 'B', 'C'], n)
})

# 결측치 삽입 (온도 10%, 생산량 5%)
missing_idx_temp = np.random.choice(n, 10, replace=False)
df.loc[missing_idx_temp, '온도'] = np.nan

missing_idx_prod = np.random.choice(n, 5, replace=False)
df.loc[missing_idx_prod, '생산량'] = np.nan

print("데이터 샘플 (결측치 포함):")
print(df.head(15))
print(f"\n데이터 크기: {df.shape}")

# ============================================================
# 실습 2: 결측치 탐지
# ============================================================
print("\n" + "=" * 50)
print("실습 2: 결측치 탐지")
print("=" * 50)

# 열별 결측치 수
print("=== 열별 결측치 수 ===")
print(df.isnull().sum())

# 결측 비율
print("\n=== 결측 비율 (%) ===")
missing_ratio = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_ratio)

# 전체 정보
print("\n=== 데이터 정보 ===")
print(df.info())

# 결측치 시각화
plt.figure(figsize=(10, 4))
df.isnull().sum().plot(kind='bar', color='coral', edgecolor='black')
plt.ylabel('결측치 수')
plt.title('열별 결측치 현황')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# 실습 3: 결측치 처리
# ============================================================
print("\n" + "=" * 50)
print("실습 3: 결측치 처리")
print("=" * 50)

# 원본 보존
df_clean = df.copy()

# 온도: 중앙값으로 대체
temp_median = df_clean['온도'].median()
df_clean['온도'].fillna(temp_median, inplace=True)
print(f"온도 결측치를 중앙값 {temp_median:.1f}도로 대체")

# 생산량: 평균으로 대체
prod_mean = df_clean['생산량'].mean()
df_clean['생산량'].fillna(prod_mean, inplace=True)
print(f"생산량 결측치를 평균 {prod_mean:.0f}개로 대체")

# 결과 확인
print(f"\n처리 후 결측치: {df_clean.isnull().sum().sum()}개")

# ============================================================
# 실습 4: 이상치 삽입
# ============================================================
print("\n" + "=" * 50)
print("실습 4: 이상치 삽입")
print("=" * 50)

# 이상치 삽입
df_clean.loc[5, '생산량'] = 2000   # 매우 높음
df_clean.loc[15, '생산량'] = 500   # 매우 낮음
df_clean.loc[25, '온도'] = 120     # 비정상 온도

print("이상치가 삽입된 행:")
print(df_clean.loc[[5, 15, 25], ['온도', '생산량']])

# ============================================================
# 실습 5: IQR로 이상치 탐지
# ============================================================
print("\n" + "=" * 50)
print("실습 5: IQR로 이상치 탐지")
print("=" * 50)

# 생산량 이상치 탐지
Q1 = df_clean['생산량'].quantile(0.25)
Q3 = df_clean['생산량'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = (df_clean['생산량'] < lower) | (df_clean['생산량'] > upper)

print("=== IQR 방법 (생산량) ===")
print(f"Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
print(f"하한: {lower:.0f}, 상한: {upper:.0f}")
print(f"이상치 개수: {outliers_iqr.sum()}개")
print(f"이상치 인덱스: {df_clean[outliers_iqr].index.tolist()}")

# 이상치 값 확인
if outliers_iqr.sum() > 0:
    print("\n이상치 상세:")
    print(df_clean[outliers_iqr][['일자', '생산량']])

# ============================================================
# 실습 6: Z-score로 이상치 탐지
# ============================================================
print("\n" + "=" * 50)
print("실습 6: Z-score로 이상치 탐지")
print("=" * 50)

# 온도 이상치 탐지
mean_temp = df_clean['온도'].mean()
std_temp = df_clean['온도'].std()
z_scores_temp = (df_clean['온도'] - mean_temp) / std_temp

outliers_z = np.abs(z_scores_temp) > 2

print("=== Z-score 방법 (온도) ===")
print(f"평균: {mean_temp:.1f}, 표준편차: {std_temp:.1f}")
print(f"이상치 개수 (|Z|>2): {outliers_z.sum()}개")

# 이상치 상세
if outliers_z.sum() > 0:
    print("\n이상치 상세:")
    outlier_df = df_clean[outliers_z][['일자', '온도']].copy()
    outlier_df['Z-score'] = z_scores_temp[outliers_z].round(2)
    print(outlier_df)

# ============================================================
# 실습 7: 이상치 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 7: 이상치 시각화")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 생산량 상자그림
bp1 = axes[0].boxplot(df_clean['생산량'], patch_artist=True)
bp1['boxes'][0].set_facecolor('lightblue')
axes[0].set_ylabel('생산량')
axes[0].set_title('생산량 분포 (이상치 포함)')
axes[0].axhline(y=lower, color='r', linestyle='--', alpha=0.5, label=f'하한: {lower:.0f}')
axes[0].axhline(y=upper, color='r', linestyle='--', alpha=0.5, label=f'상한: {upper:.0f}')
axes[0].legend()

# 온도 상자그림
bp2 = axes[1].boxplot(df_clean['온도'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightcoral')
axes[1].set_ylabel('온도')
axes[1].set_title('온도 분포 (이상치 포함)')

plt.tight_layout()
plt.show()

# ============================================================
# 실습 8: 이상치 처리
# ============================================================
print("\n" + "=" * 50)
print("실습 8: 이상치 처리")
print("=" * 50)

# 처리 전 통계 저장
before_stats = {
    '생산량_min': df_clean['생산량'].min(),
    '생산량_max': df_clean['생산량'].max(),
    '온도_min': df_clean['온도'].min(),
    '온도_max': df_clean['온도'].max()
}

# 처리
df_final = df_clean.copy()

# 생산량: 클리핑
df_final['생산량'] = df_final['생산량'].clip(lower, upper)

# 온도: Z-score 기준 이상치를 중앙값으로 대체
temp_median_final = df_final['온도'].median()
df_final.loc[outliers_z, '온도'] = temp_median_final

print("=== 처리 결과 ===")
print(f"생산량: {before_stats['생산량_min']:.0f}~{before_stats['생산량_max']:.0f} → "
      f"{df_final['생산량'].min():.0f}~{df_final['생산량'].max():.0f}")
print(f"온도: {before_stats['온도_min']:.1f}~{before_stats['온도_max']:.1f} → "
      f"{df_final['온도'].min():.1f}~{df_final['온도'].max():.1f}")

# ============================================================
# 실습 9: 전후 비교
# ============================================================
print("\n" + "=" * 50)
print("실습 9: 전후 비교")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 처리 전 생산량
axes[0, 0].hist(df_clean['생산량'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('처리 전 생산량')
axes[0, 0].set_xlabel('생산량')
axes[0, 0].set_ylabel('빈도')

# 처리 후 생산량
axes[0, 1].hist(df_final['생산량'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].set_title('처리 후 생산량')
axes[0, 1].set_xlabel('생산량')
axes[0, 1].set_ylabel('빈도')

# 처리 전 온도
axes[1, 0].hist(df_clean['온도'], bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].set_title('처리 전 온도')
axes[1, 0].set_xlabel('온도')
axes[1, 0].set_ylabel('빈도')

# 처리 후 온도
axes[1, 1].hist(df_final['온도'], bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].set_title('처리 후 온도')
axes[1, 1].set_xlabel('온도')
axes[1, 1].set_ylabel('빈도')

plt.tight_layout()
plt.show()

# ============================================================
# 실습 10: 전처리 요약
# ============================================================
print("\n" + "=" * 50)
print("실습 10: 전처리 요약")
print("=" * 50)

print("=" * 50)
print("         데이터 전처리 요약 리포트")
print("=" * 50)

print(f"\n[원본 데이터]")
print(f"행 수: {len(df)}")
print(f"결측치: 온도 {df['온도'].isnull().sum()}개, 생산량 {df['생산량'].isnull().sum()}개")

print(f"\n[처리 내용]")
print(f"- 온도 결측치: 중앙값({temp_median:.1f})으로 대체")
print(f"- 생산량 결측치: 평균({prod_mean:.0f})으로 대체")
print(f"- 생산량 이상치: 클리핑({lower:.0f}~{upper:.0f})")
print(f"- 온도 이상치: 중앙값({temp_median_final:.1f})으로 대체")

print(f"\n[최종 데이터]")
print(f"결측치: {df_final.isnull().sum().sum()}개")
print(f"생산량 범위: {df_final['생산량'].min():.0f} ~ {df_final['생산량'].max():.0f}")
print(f"온도 범위: {df_final['온도'].min():.1f} ~ {df_final['온도'].max():.1f}")
print("=" * 50)

# ============================================================
# 추가: 기술통계 비교
# ============================================================
print("\n" + "=" * 50)
print("추가: 기술통계 비교")
print("=" * 50)

print("=== 처리 전 ===")
print(df_clean[['온도', '생산량', '불량수']].describe().round(2))

print("\n=== 처리 후 ===")
print(df_final[['온도', '생산량', '불량수']].describe().round(2))
