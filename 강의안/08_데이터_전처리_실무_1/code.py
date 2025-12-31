"""
[8차시] 데이터 전처리 실무 (1) - 실습 코드

결측치와 이상치 처리를 실습합니다.

학습목표:
- 결측치를 탐지하고 적절히 처리
- 이상치를 탐지하는 다양한 방법 적용
- 상황에 맞는 전처리 전략 선택
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("8차시: 데이터 전처리 실무 (1)")
print("=" * 60)
print()


# =============================================================
# 1. 샘플 데이터 생성 (결측치, 이상치 포함)
# =============================================================

print("=" * 60)
print("1. 샘플 데이터 생성")
print("=" * 60)

np.random.seed(42)
n = 200

# 기본 데이터 생성
df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=n),
    '라인': np.random.choice(['A', 'B', 'C'], n),
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(25, n)
})

# 결측치 추가 (약 10%)
missing_mask_temp = np.random.random(n) < 0.08
missing_mask_humidity = np.random.random(n) < 0.12
missing_mask_prod = np.random.random(n) < 0.05

df.loc[missing_mask_temp, '온도'] = np.nan
df.loc[missing_mask_humidity, '습도'] = np.nan
df.loc[missing_mask_prod, '생산량'] = np.nan

# 이상치 추가
outlier_indices = np.random.choice(n, 10, replace=False)
df.loc[outlier_indices[:5], '온도'] = np.random.choice([60, 65, 105, 110], 5)
df.loc[outlier_indices[5:], '생산량'] = np.random.choice([800, 850, 1500, 1550], 5)

print("데이터 샘플:")
print(df.head(10))
print(f"\n데이터 크기: {df.shape}")
print()


# =============================================================
# 2. 결측치 탐지
# =============================================================

print("=" * 60)
print("2. 결측치 탐지")
print("=" * 60)

# 결측치 수
print("[열별 결측치 수]")
print(df.isnull().sum())
print()

# 결측치 비율
print("[열별 결측치 비율 (%)]")
print((df.isnull().mean() * 100).round(2))
print()

# 전체 결측치
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"전체 결측치: {total_missing}개 / {total_cells}개 ({total_missing/total_cells*100:.2f}%)")
print()

# 결측치가 있는 행 확인
missing_rows = df[df.isnull().any(axis=1)]
print(f"결측치가 있는 행 수: {len(missing_rows)}")
print()


# =============================================================
# 3. 결측치 시각화
# =============================================================

print("=" * 60)
print("3. 결측치 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 열별 결측치 수 막대그래프
missing_counts = df.isnull().sum()
axes[0].bar(missing_counts.index, missing_counts.values, color='coral', edgecolor='black')
axes[0].set_xlabel('열')
axes[0].set_ylabel('결측치 수')
axes[0].set_title('열별 결측치 수')
axes[0].tick_params(axis='x', rotation=45)

# 결측치 패턴 히트맵 (샘플)
sample_df = df.head(50).isnull()
axes[1].imshow(sample_df, aspect='auto', cmap='Reds', interpolation='nearest')
axes[1].set_xlabel('열')
axes[1].set_ylabel('행 (처음 50개)')
axes[1].set_title('결측치 패턴 (빨간색 = 결측)')
axes[1].set_xticks(range(len(df.columns)))
axes[1].set_xticklabels(df.columns, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('01_missing_visualization.png', dpi=150)
plt.show()

print("'01_missing_visualization.png' 저장 완료")
print()


# =============================================================
# 4. 결측치 처리 - 다양한 방법
# =============================================================

print("=" * 60)
print("4. 결측치 처리 방법")
print("=" * 60)

# 원본 복사
df_original = df.copy()

# 방법 1: 삭제
df_dropped = df.dropna()
print(f"[방법 1: 삭제]")
print(f"원본: {len(df)}행 → 삭제 후: {len(df_dropped)}행 ({len(df)-len(df_dropped)}행 삭제)")
print()

# 방법 2: 평균으로 대체
df_mean = df.copy()
df_mean['온도'].fillna(df_mean['온도'].mean(), inplace=True)
df_mean['습도'].fillna(df_mean['습도'].mean(), inplace=True)
df_mean['생산량'].fillna(df_mean['생산량'].mean(), inplace=True)
print(f"[방법 2: 평균 대체]")
print(f"결측치: {df_mean.isnull().sum().sum()}개 (처리 완료)")
print()

# 방법 3: 중앙값으로 대체
df_median = df.copy()
df_median['온도'].fillna(df_median['온도'].median(), inplace=True)
df_median['습도'].fillna(df_median['습도'].median(), inplace=True)
df_median['생산량'].fillna(df_median['생산량'].median(), inplace=True)
print(f"[방법 3: 중앙값 대체]")
print(f"결측치: {df_median.isnull().sum().sum()}개 (처리 완료)")
print()

# 방법 4: 그룹별 중앙값으로 대체
df_group = df.copy()
for col in ['온도', '습도', '생산량']:
    df_group[col] = df_group.groupby('라인')[col].transform(
        lambda x: x.fillna(x.median())
    )
print(f"[방법 4: 그룹(라인)별 중앙값 대체]")
print(f"결측치: {df_group.isnull().sum().sum()}개")
print()


# =============================================================
# 5. 결측치 처리 전후 분포 비교
# =============================================================

print("=" * 60)
print("5. 결측치 처리 전후 비교")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, col in enumerate(['온도', '습도', '생산량']):
    # 원본 (결측치 제외)
    original = df_original[col].dropna()
    # 평균 대체
    mean_filled = df_mean[col]
    # 중앙값 대체
    median_filled = df_median[col]

    axes[idx].hist(original, bins=20, alpha=0.5, label='원본', color='blue')
    axes[idx].hist(mean_filled, bins=20, alpha=0.5, label='평균 대체', color='red')
    axes[idx].axvline(original.mean(), color='blue', linestyle='--', linewidth=2)
    axes[idx].axvline(mean_filled.mean(), color='red', linestyle='--', linewidth=2)
    axes[idx].set_title(f'{col} 분포 비교')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('02_missing_comparison.png', dpi=150)
plt.show()

print("'02_missing_comparison.png' 저장 완료")
print()


# =============================================================
# 6. 이상치 탐지 - IQR 방법
# =============================================================

print("=" * 60)
print("6. 이상치 탐지 - IQR 방법")
print("=" * 60)

# 중앙값으로 결측치 처리된 데이터 사용
df_clean = df_median.copy()

def detect_outliers_iqr(data, column):
    """IQR 방법으로 이상치 탐지"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (data[column] < lower) | (data[column] > upper)
    return outliers, lower, upper

print("[IQR 기반 이상치 탐지]")
for col in ['온도', '습도', '생산량']:
    outliers, lower, upper = detect_outliers_iqr(df_clean, col)
    n_outliers = outliers.sum()
    print(f"\n{col}:")
    print(f"  정상 범위: [{lower:.2f}, {upper:.2f}]")
    print(f"  이상치 수: {n_outliers}개 ({n_outliers/len(df_clean)*100:.1f}%)")
    if n_outliers > 0:
        print(f"  이상치 값: {df_clean.loc[outliers, col].values[:5]}")
print()


# =============================================================
# 7. 이상치 탐지 - Z-score 방법
# =============================================================

print("=" * 60)
print("7. 이상치 탐지 - Z-score 방법")
print("=" * 60)

def detect_outliers_zscore(data, column, threshold=3):
    """Z-score 방법으로 이상치 탐지"""
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = z_scores > threshold
    return outliers, z_scores

print("[Z-score 기반 이상치 탐지 (|Z| > 3)]")
for col in ['온도', '습도', '생산량']:
    outliers, z_scores = detect_outliers_zscore(df_clean, col)
    n_outliers = outliers.sum()
    print(f"\n{col}:")
    print(f"  이상치 수: {n_outliers}개")
    if n_outliers > 0:
        print(f"  이상치 Z-score: {z_scores[outliers][:5]}")
print()


# =============================================================
# 8. 이상치 시각화
# =============================================================

print("=" * 60)
print("8. 이상치 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, col in enumerate(['온도', '습도', '생산량']):
    # 상자그림
    bp = axes[0, idx].boxplot(df_clean[col], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    axes[0, idx].set_title(f'{col} 상자그림')
    axes[0, idx].set_ylabel(col)

    # 산점도 + 이상치 강조
    outliers_iqr, lower, upper = detect_outliers_iqr(df_clean, col)
    colors = ['red' if o else 'steelblue' for o in outliers_iqr]
    axes[1, idx].scatter(range(len(df_clean)), df_clean[col], c=colors, alpha=0.5, s=20)
    axes[1, idx].axhline(lower, color='red', linestyle='--', label=f'하한: {lower:.1f}')
    axes[1, idx].axhline(upper, color='red', linestyle='--', label=f'상한: {upper:.1f}')
    axes[1, idx].set_xlabel('인덱스')
    axes[1, idx].set_ylabel(col)
    axes[1, idx].set_title(f'{col} 이상치 탐지 (IQR)')
    axes[1, idx].legend()

plt.tight_layout()
plt.savefig('03_outlier_detection.png', dpi=150)
plt.show()

print("'03_outlier_detection.png' 저장 완료")
print()


# =============================================================
# 9. 이상치 처리 방법
# =============================================================

print("=" * 60)
print("9. 이상치 처리 방법")
print("=" * 60)

# 온도 열의 이상치 처리 예시
outliers, lower, upper = detect_outliers_iqr(df_clean, '온도')
print(f"원본 온도 이상치: {outliers.sum()}개")
print()

# 방법 1: 삭제
df_outlier_removed = df_clean[~outliers].copy()
print(f"[방법 1: 삭제]")
print(f"원본: {len(df_clean)}행 → 처리 후: {len(df_outlier_removed)}행")
print()

# 방법 2: 클리핑 (상/하한 설정)
df_clipped = df_clean.copy()
df_clipped['온도'] = df_clipped['온도'].clip(lower, upper)
print(f"[방법 2: 클리핑]")
print(f"온도 범위: [{df_clipped['온도'].min():.2f}, {df_clipped['온도'].max():.2f}]")
print()

# 방법 3: 중앙값 대체
df_median_replaced = df_clean.copy()
median_temp = df_clean['온도'].median()
df_median_replaced.loc[outliers, '온도'] = median_temp
print(f"[방법 3: 중앙값 대체]")
print(f"중앙값으로 대체: {median_temp:.2f}")
print()

# 방법 4: 플래그 추가
df_flagged = df_clean.copy()
df_flagged['온도_이상치'] = outliers
print(f"[방법 4: 플래그 추가]")
print(f"이상치 플래그 추가됨")
print()


# =============================================================
# 10. 이상치 처리 전후 비교
# =============================================================

print("=" * 60)
print("10. 이상치 처리 전후 비교")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본
axes[0, 0].hist(df_clean['온도'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df_clean['온도'].mean(), color='red', linestyle='--', label=f"평균: {df_clean['온도'].mean():.1f}")
axes[0, 0].set_title('원본 (이상치 포함)')
axes[0, 0].set_xlabel('온도')
axes[0, 0].legend()

# 삭제 후
axes[0, 1].hist(df_outlier_removed['온도'], bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].axvline(df_outlier_removed['온도'].mean(), color='red', linestyle='--',
                   label=f"평균: {df_outlier_removed['온도'].mean():.1f}")
axes[0, 1].set_title('이상치 삭제 후')
axes[0, 1].set_xlabel('온도')
axes[0, 1].legend()

# 클리핑 후
axes[1, 0].hist(df_clipped['온도'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].axvline(df_clipped['온도'].mean(), color='red', linestyle='--',
                   label=f"평균: {df_clipped['온도'].mean():.1f}")
axes[1, 0].set_title('클리핑 후')
axes[1, 0].set_xlabel('온도')
axes[1, 0].legend()

# 통계 비교
stats_comparison = pd.DataFrame({
    '원본': [df_clean['온도'].mean(), df_clean['온도'].std(), df_clean['온도'].median()],
    '삭제': [df_outlier_removed['온도'].mean(), df_outlier_removed['온도'].std(), df_outlier_removed['온도'].median()],
    '클리핑': [df_clipped['온도'].mean(), df_clipped['온도'].std(), df_clipped['온도'].median()],
}, index=['평균', '표준편차', '중앙값'])

axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_comparison.round(2).values,
                         rowLabels=stats_comparison.index,
                         colLabels=stats_comparison.columns,
                         cellLoc='center',
                         loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
axes[1, 1].set_title('통계량 비교', pad=20)

plt.tight_layout()
plt.savefig('04_outlier_comparison.png', dpi=150)
plt.show()

print("'04_outlier_comparison.png' 저장 완료")
print()


# =============================================================
# 11. 종합 전처리 파이프라인
# =============================================================

print("=" * 60)
print("11. 종합 전처리 파이프라인")
print("=" * 60)

def preprocess_manufacturing_data(df, missing_strategy='median', outlier_strategy='clip'):
    """
    제조 데이터 전처리 파이프라인

    Parameters:
    - missing_strategy: 'drop', 'mean', 'median'
    - outlier_strategy: 'remove', 'clip', 'flag'
    """
    df_processed = df.copy()

    # 1. 결측치 처리
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

    if missing_strategy == 'drop':
        df_processed = df_processed.dropna()
    elif missing_strategy == 'mean':
        for col in numeric_cols:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # 2. 이상치 처리
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df_processed[col] < lower) | (df_processed[col] > upper)

        if outlier_strategy == 'remove':
            df_processed = df_processed[~outliers]
        elif outlier_strategy == 'clip':
            df_processed[col] = df_processed[col].clip(lower, upper)
        elif outlier_strategy == 'flag':
            df_processed[f'{col}_이상치'] = outliers

    return df_processed


# 파이프라인 적용
print("전처리 전:")
print(f"  데이터 크기: {df_original.shape}")
print(f"  결측치: {df_original.isnull().sum().sum()}개")

df_final = preprocess_manufacturing_data(df_original,
                                         missing_strategy='median',
                                         outlier_strategy='clip')

print("\n전처리 후:")
print(f"  데이터 크기: {df_final.shape}")
print(f"  결측치: {df_final.isnull().sum().sum()}개")
print()

# 최종 데이터 확인
print("[전처리된 데이터 샘플]")
print(df_final.head())
print()

print("[전처리된 데이터 통계]")
print(df_final.describe())
print()

print("=" * 60)
print("8차시 실습 완료!")
print("=" * 60)
