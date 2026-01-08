"""
[9차시] 제조 데이터 전처리 (1): 결측치와 이상치
========================================

학습 목표:
1. 결측치 탐지 및 처리 (isnull, fillna, dropna)
2. 이상치 탐지 방법 적용 (IQR, Z-score)
3. 상황에 맞는 전처리 전략 선택 (clipping, 플래그)

실습 환경:
- Python 3.8+
- pandas, numpy, scipy, matplotlib, seaborn

데이터:
- Titanic 데이터셋 (타이타닉 승객 데이터)
- 출처: seaborn built-in dataset
- 결측치가 실제로 존재하여 전처리 학습에 적합
- 변수: survived(생존), pclass(좌석등급), age(나이), fare(요금) 등
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[9차시] 제조 데이터 전처리 (1): 결측치와 이상치")
print("=" * 60)


# ============================================================
# Part 1: 결측치 탐지 및 처리
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 결측치 탐지 및 처리")
print("=" * 60)


# --------------------------------------
# 1.1 실습 데이터 로드 - Titanic 데이터셋
# --------------------------------------
print("\n[1.1] 실습 데이터 로드 - Titanic 데이터셋")
print("-" * 40)

# Titanic 데이터셋 로드 (실제 결측치 포함!)
try:
    df = sns.load_dataset('titanic')
    print("Titanic 데이터셋 로드 성공!")
except Exception as e:
    print(f"seaborn 로드 실패: {e}")
    print("온라인에서 데이터를 다운로드합니다...")
    try:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
        df = pd.read_csv(url)
        print("온라인 다운로드 성공!")
    except Exception as e2:
        print(f"온라인 로드도 실패: {e2}")
        print("대체 데이터를 생성합니다...")
        # 대체 데이터 생성
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'survived': np.random.choice([0, 1], n),
            'pclass': np.random.choice([1, 2, 3], n),
            'sex': np.random.choice(['male', 'female'], n),
            'age': np.random.normal(30, 10, n),
            'fare': np.random.exponential(30, n),
            'embarked': np.random.choice(['S', 'C', 'Q', np.nan], n),
            'deck': np.random.choice(['A', 'B', 'C', None], n)
        })
        # 결측치 추가
        df.loc[np.random.choice(n, 15, replace=False), 'age'] = np.nan

print(f"\n데이터 형태: {df.shape}")
print(f"\n처음 10행:\n{df.head(10)}")

# 데이터 설명
print("\n=== 변수 설명 ===")
print("survived: 생존 여부 (0=사망, 1=생존)")
print("pclass: 좌석 등급 (1=1등급, 2=2등급, 3=3등급)")
print("sex: 성별")
print("age: 나이 (결측치 존재!)")
print("fare: 요금")
print("embarked: 탑승 항구 (S=Southampton, C=Cherbourg, Q=Queenstown)")
print("deck: 갑판 (A~G, 결측치 다수!)")


# --------------------------------------
# 1.2 결측치 탐지
# --------------------------------------
print("\n[1.2] 결측치 탐지")
print("-" * 40)

# 방법 1: isnull() + sum()
print("=== 방법 1: isnull().sum() ===")
missing_count = df.isnull().sum()
print(missing_count)

# 방법 2: 결측치 비율
print("\n=== 결측치 비율 (%) ===")
missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio.round(2))

# 결측치가 있는 컬럼만 필터링
print("\n=== 결측치가 있는 컬럼 ===")
cols_with_missing = missing_ratio[missing_ratio > 0]
print(cols_with_missing.round(2))

# 방법 3: info()
print("\n=== df.info() 결과 ===")
print(df.info())


# --------------------------------------
# 1.3 결측치 시각화
# --------------------------------------
print("\n[1.3] 결측치 시각화")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 히트맵으로 결측치 패턴 확인
ax1 = axes[0]
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax1, cmap='YlOrRd')
ax1.set_title('Missing Value Heatmap - Titanic Dataset')
ax1.set_xlabel('Columns')

# 결측치 개수 막대그래프
ax2 = axes[1]
cols_with_missing = missing_count[missing_count > 0]
cols_with_missing.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Missing Value Count by Column')
ax2.set_xlabel('Columns')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

# 각 막대 위에 개수 표시
for i, v in enumerate(cols_with_missing):
    ax2.text(i, v + 2, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('missing_visualization.png', dpi=100, bbox_inches='tight')
plt.close()
print("결측치 시각화 저장: missing_visualization.png")

print("\n=== 결측치 해석 ===")
print("- age: 약 20%의 결측치 -> 중앙값 대체 추천")
print("- deck: 약 70%+ 결측치 -> 삭제 또는 'Unknown' 대체 고려")
print("- embarked: 소수의 결측치 -> 최빈값 대체 추천")


# --------------------------------------
# 1.4 결측치 처리 - 삭제
# --------------------------------------
print("\n[1.4] 결측치 처리 - 삭제 (dropna)")
print("-" * 40)

# 원본 복사
df_dropna = df.copy()

# 방법 1: 모든 결측치가 있는 행 삭제
df_drop_all = df_dropna.dropna()
print(f"dropna() 후 행 수: {len(df)} -> {len(df_drop_all)} ({len(df) - len(df_drop_all)}개 삭제)")

# 방법 2: 특정 컬럼 기준 삭제 (age만 고려)
df_drop_subset = df_dropna.dropna(subset=['age'])
print(f"dropna(subset=['age']) 후: {len(df_drop_subset)}행")

# 방법 3: 결측치가 너무 많은 열 삭제 (예: 50% 이상)
threshold = len(df) * 0.5
df_drop_cols = df_dropna.dropna(axis=1, thresh=threshold)
print(f"dropna(axis=1, thresh={int(threshold)}) 후 열 수: {df.shape[1]} -> {df_drop_cols.shape[1]}")
print(f"  삭제된 열: {set(df.columns) - set(df_drop_cols.columns)}")


# --------------------------------------
# 1.5 결측치 처리 - 대체
# --------------------------------------
print("\n[1.5] 결측치 처리 - 대체 (fillna)")
print("-" * 40)

# 원본 복사
df_filled = df.copy()

# 수치형 컬럼: 중앙값으로 대체
numeric_cols = ['age', 'fare']
for col in numeric_cols:
    if df_filled[col].isnull().sum() > 0:
        median_val = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_val)
        print(f"{col}: 결측치를 중앙값 {median_val:.2f}로 대체")

# 범주형 컬럼: 최빈값으로 대체
categorical_cols = ['embarked', 'embark_town']
for col in categorical_cols:
    if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
        mode_val = df_filled[col].mode()[0]
        df_filled[col] = df_filled[col].fillna(mode_val)
        print(f"{col}: 결측치를 최빈값 '{mode_val}'로 대체")

# deck 컬럼: 'Unknown'으로 대체 (결측치가 너무 많아서)
if 'deck' in df_filled.columns:
    df_filled['deck'] = df_filled['deck'].fillna('Unknown')
    print("deck: 결측치를 'Unknown'으로 대체")

# 결과 확인
print(f"\n처리 후 결측치:")
print(df_filled.isnull().sum())


# --------------------------------------
# 1.6 대체 전략 비교
# --------------------------------------
print("\n[1.6] 대체 전략 비교")
print("-" * 40)

# 원본에서 age 컬럼만 추출
age_with_nan = df['age'].copy()

# 다양한 대체 전략
strategies = {
    'mean': age_with_nan.fillna(age_with_nan.mean()),
    'median': age_with_nan.fillna(age_with_nan.median()),
    'zero': age_with_nan.fillna(0),
    'ffill': age_with_nan.ffill(),
    'interpolate': age_with_nan.interpolate()
}

print("age 컬럼 대체 전략별 통계:")
print("-" * 50)
print(f"{'전략':<12} {'평균':>10} {'표준편차':>10} {'최솟값':>10} {'최댓값':>10}")
print("-" * 50)
for name, data in strategies.items():
    print(f"{name:<12} {data.mean():>10.2f} {data.std():>10.2f} {data.min():>10.2f} {data.max():>10.2f}")

print("\n=== 추천 전략 ===")
print("- 정규분포: mean 또는 median")
print("- 치우친 분포: median 권장 (이상치 영향 적음)")
print("- 시계열: ffill, bfill, interpolate")
print("- 절대 사용 금지: zero (의미 왜곡)")


# ============================================================
# Part 2: 이상치 탐지 방법 적용
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 이상치 탐지 방법 적용")
print("=" * 60)

# 결측치가 처리된 데이터 사용
df_clean = df_filled.copy()
numeric_cols_analysis = ['age', 'fare']


# --------------------------------------
# 2.1 기술 통계로 이상치 확인
# --------------------------------------
print("\n[2.1] 기술 통계로 이상치 확인")
print("-" * 40)

print(df_clean[numeric_cols_analysis].describe())

print("\n=== 최솟값/최댓값 확인 ===")
for col in numeric_cols_analysis:
    print(f"{col}: min={df_clean[col].min():.2f}, max={df_clean[col].max():.2f}")

print("\n=== 해석 ===")
print("- fare: 최댓값 512.33은 매우 높음 (1등급 특실 가격)")
print("- age: 범위 0.42~80 (영아부터 노인까지)")


# --------------------------------------
# 2.2 IQR 방법으로 이상치 탐지
# --------------------------------------
print("\n[2.2] IQR 방법으로 이상치 탐지")
print("-" * 40)

def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    IQR 방법으로 이상치 탐지

    Parameters:
    -----------
    data : DataFrame
    column : str - 분석할 컬럼명
    multiplier : float - IQR 배수 (기본 1.5)

    Returns:
    --------
    outlier_mask : Series - 이상치 위치 (True/False)
    lower : float - 하한
    upper : float - 상한
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    outlier_mask = (data[column] < lower) | (data[column] > upper)

    return outlier_mask, lower, upper

# age 이상치 탐지
age_outliers, age_lower, age_upper = detect_outliers_iqr(df_clean, 'age')
print(f"=== age IQR 분석 ===")
print(f"  Q1: {df_clean['age'].quantile(0.25):.2f}")
print(f"  Q3: {df_clean['age'].quantile(0.75):.2f}")
print(f"  IQR: {df_clean['age'].quantile(0.75) - df_clean['age'].quantile(0.25):.2f}")
print(f"  정상 범위: {age_lower:.2f} ~ {age_upper:.2f}")
print(f"  이상치 개수: {age_outliers.sum()}")
if age_outliers.sum() > 0:
    print(f"  이상치 값: {df_clean.loc[age_outliers, 'age'].head(10).values}")

# fare 이상치 탐지
fare_outliers, fare_lower, fare_upper = detect_outliers_iqr(df_clean, 'fare')
print(f"\n=== fare IQR 분석 ===")
print(f"  Q1: {df_clean['fare'].quantile(0.25):.2f}")
print(f"  Q3: {df_clean['fare'].quantile(0.75):.2f}")
print(f"  IQR: {df_clean['fare'].quantile(0.75) - df_clean['fare'].quantile(0.25):.2f}")
print(f"  정상 범위: {fare_lower:.2f} ~ {fare_upper:.2f}")
print(f"  이상치 개수: {fare_outliers.sum()}")
if fare_outliers.sum() > 0:
    print(f"  이상치 값 (상위 10개): {sorted(df_clean.loc[fare_outliers, 'fare'].values, reverse=True)[:10]}")


# --------------------------------------
# 2.3 Z-score 방법으로 이상치 탐지
# --------------------------------------
print("\n[2.3] Z-score 방법으로 이상치 탐지")
print("-" * 40)

def detect_outliers_zscore(data, column, threshold=3):
    """
    Z-score 방법으로 이상치 탐지

    Parameters:
    -----------
    data : DataFrame
    column : str - 분석할 컬럼명
    threshold : float - Z-score 임계값 (기본 3)

    Returns:
    --------
    outlier_mask : Series - 이상치 위치 (True/False)
    z_scores : ndarray - Z-score 값들
    """
    z_scores = stats.zscore(data[column])
    outlier_mask = np.abs(z_scores) > threshold

    return outlier_mask, z_scores

# age 이상치 탐지
age_outliers_z, age_zscores = detect_outliers_zscore(df_clean, 'age')
print(f"=== age Z-score 분석 (threshold=3) ===")
print(f"  이상치 개수: {age_outliers_z.sum()}")
if age_outliers_z.any():
    print(f"  이상치 값: {df_clean.loc[age_outliers_z, 'age'].values}")
    print(f"  Z-scores: {age_zscores[age_outliers_z]}")

# fare 이상치 탐지
fare_outliers_z, fare_zscores = detect_outliers_zscore(df_clean, 'fare')
print(f"\n=== fare Z-score 분석 (threshold=3) ===")
print(f"  이상치 개수: {fare_outliers_z.sum()}")
if fare_outliers_z.any():
    print(f"  이상치 값 (상위 5개): {sorted(df_clean.loc[fare_outliers_z, 'fare'].values, reverse=True)[:5]}")


# --------------------------------------
# 2.4 IQR vs Z-score 비교
# --------------------------------------
print("\n[2.4] IQR vs Z-score 비교")
print("-" * 40)

print("=== age 이상치 비교 ===")
print(f"  IQR 방법: {age_outliers.sum()}개")
print(f"  Z-score 방법: {age_outliers_z.sum()}개")

print("\n=== fare 이상치 비교 ===")
print(f"  IQR 방법: {fare_outliers.sum()}개")
print(f"  Z-score 방법: {fare_outliers_z.sum()}개")

print("\n=== 방법 선택 가이드 ===")
print("- IQR: 분포 가정 없음, 중앙값 기반, 치우친 분포에 강건")
print("- Z-score: 정규분포 가정, 평균 기반, 대칭 분포에 적합")
print("- fare처럼 치우친 분포 -> IQR 권장")


# --------------------------------------
# 2.5 이상치 시각화
# --------------------------------------
print("\n[2.5] 이상치 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# age 박스플롯
ax1 = axes[0, 0]
ax1.boxplot(df_clean['age'].dropna(), vert=True)
ax1.set_title('Age - Box Plot')
ax1.set_ylabel('Age')
ax1.axhline(y=age_lower, color='r', linestyle='--', label=f'Lower: {age_lower:.1f}')
ax1.axhline(y=age_upper, color='r', linestyle='--', label=f'Upper: {age_upper:.1f}')
ax1.legend()

# age 히스토그램 + 이상치 표시
ax2 = axes[0, 1]
ax2.hist(df_clean['age'], bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(x=age_lower, color='r', linestyle='--', label=f'Lower: {age_lower:.1f}')
ax2.axvline(x=age_upper, color='r', linestyle='--', label=f'Upper: {age_upper:.1f}')
ax2.set_title('Age - Histogram with IQR Bounds')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.legend()

# fare 박스플롯
ax3 = axes[1, 0]
ax3.boxplot(df_clean['fare'].dropna(), vert=True)
ax3.set_title('Fare - Box Plot (Note: Highly Skewed!)')
ax3.set_ylabel('Fare')
ax3.axhline(y=fare_lower, color='r', linestyle='--', label=f'Lower: {fare_lower:.1f}')
ax3.axhline(y=fare_upper, color='r', linestyle='--', label=f'Upper: {fare_upper:.1f}')
ax3.legend()

# fare 히스토그램 (로그 스케일 고려)
ax4 = axes[1, 1]
ax4.hist(df_clean['fare'], bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=fare_upper, color='r', linestyle='--', label=f'Upper: {fare_upper:.1f}')
ax4.set_title('Fare - Histogram (Skewed Distribution)')
ax4.set_xlabel('Fare')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
plt.savefig('outlier_visualization.png', dpi=100, bbox_inches='tight')
plt.close()
print("이상치 시각화 저장: outlier_visualization.png")


# ============================================================
# Part 3: 상황에 맞는 전처리 전략 선택
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 상황에 맞는 전처리 전략 선택")
print("=" * 60)


# --------------------------------------
# 3.1 이상치 처리 - 제거
# --------------------------------------
print("\n[3.1] 이상치 처리 - 제거")
print("-" * 40)

# 원본 복사
df_remove = df_clean.copy()

# fare 이상치만 제거 (age는 유지 - 실제 고령자 데이터)
outlier_mask_fare = fare_outliers

print(f"fare 이상치 행: {outlier_mask_fare.sum()}개")

df_removed = df_remove[~outlier_mask_fare]
print(f"제거 후 행 수: {len(df_clean)} -> {len(df_removed)}")
print(f"제거된 행: {len(df_clean) - len(df_removed)}개")


# --------------------------------------
# 3.2 이상치 처리 - Clipping
# --------------------------------------
print("\n[3.2] 이상치 처리 - Clipping")
print("-" * 40)

# 원본 복사
df_clipped = df_clean.copy()

# fare clipping
print("=== fare clipping ===")
print(f"  처리 전 범위: {df_clipped['fare'].min():.2f} ~ {df_clipped['fare'].max():.2f}")
df_clipped['fare'] = df_clipped['fare'].clip(lower=fare_lower, upper=fare_upper)
print(f"  처리 후 범위: {df_clipped['fare'].min():.2f} ~ {df_clipped['fare'].max():.2f}")

# age clipping (참고용)
print("\n=== age clipping ===")
print(f"  처리 전 범위: {df_clean['age'].min():.2f} ~ {df_clean['age'].max():.2f}")
# age는 실제 값이므로 clipping 안 함
print("  -> age는 실제 나이이므로 clipping 하지 않음")


# --------------------------------------
# 3.3 이상치 처리 - 플래그 추가
# --------------------------------------
print("\n[3.3] 이상치 처리 - 플래그 추가")
print("-" * 40)

# 원본 복사
df_flagged = df_clean.copy()

# 각 컬럼별 이상치 플래그
for col in numeric_cols_analysis:
    mask, lower, upper = detect_outliers_iqr(df_flagged, col)
    df_flagged[f'{col}_outlier'] = mask
    print(f"{col}_outlier 추가: {mask.sum()}개 이상치")

print(f"\n플래그 컬럼 추가 후 형태: {df_flagged.shape}")

# 이상치 승객 확인
print("\n=== fare 이상치 승객 (상위 5명) ===")
fare_outlier_passengers = df_flagged[df_flagged['fare_outlier'] == True].sort_values('fare', ascending=False)
if len(fare_outlier_passengers) > 0:
    print(fare_outlier_passengers[['pclass', 'sex', 'age', 'fare', 'survived']].head())


# --------------------------------------
# 3.4 전처리 전후 비교
# --------------------------------------
print("\n[3.4] 전처리 전후 비교")
print("-" * 40)

print("=== fare 통계 비교 ===")
print(f"  원본      : mean={df_clean['fare'].mean():.2f}, std={df_clean['fare'].std():.2f}")
print(f"  제거 후   : mean={df_removed['fare'].mean():.2f}, std={df_removed['fare'].std():.2f}")
print(f"  Clipping  : mean={df_clipped['fare'].mean():.2f}, std={df_clipped['fare'].std():.2f}")

# 생존율 비교 (이상치 제거가 분석에 미치는 영향)
print("\n=== 생존율 비교 (fare 이상치 영향) ===")
print(f"  전체 생존율: {df_clean['survived'].mean():.3f}")
print(f"  fare 이상치 승객 생존율: {df_flagged[df_flagged['fare_outlier']]['survived'].mean():.3f}")
print(f"  fare 정상 승객 생존율: {df_flagged[~df_flagged['fare_outlier']]['survived'].mean():.3f}")
print("  -> 고가 요금(1등급) 승객의 생존율이 높음!")


# --------------------------------------
# 3.5 전처리 전후 시각화
# --------------------------------------
print("\n[3.5] 전처리 전후 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# fare 비교
ax1 = axes[0, 0]
ax1.boxplot(df_clean['fare'], vert=True)
ax1.set_title('Fare - Original')

ax2 = axes[0, 1]
ax2.boxplot(df_removed['fare'], vert=True)
ax2.set_title('Fare - After Removal')

ax3 = axes[0, 2]
ax3.boxplot(df_clipped['fare'], vert=True)
ax3.set_title('Fare - After Clipping')

# 분포 비교
ax4 = axes[1, 0]
ax4.hist(df_clean['fare'], bins=50, edgecolor='black', alpha=0.7)
ax4.set_title('Fare Distribution - Original')
ax4.set_xlabel('Fare')

ax5 = axes[1, 1]
ax5.hist(df_removed['fare'], bins=50, edgecolor='black', alpha=0.7)
ax5.set_title('Fare Distribution - After Removal')
ax5.set_xlabel('Fare')

ax6 = axes[1, 2]
ax6.hist(df_clipped['fare'], bins=50, edgecolor='black', alpha=0.7)
ax6.set_title('Fare Distribution - After Clipping')
ax6.set_xlabel('Fare')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("전처리 비교 시각화 저장: preprocessing_comparison.png")


# ============================================================
# 종합 실습: 전처리 파이프라인
# ============================================================
print("\n" + "=" * 60)
print("종합 실습: 전처리 파이프라인")
print("=" * 60)


def preprocess_titanic_data(df, numeric_cols, categorical_cols=None,
                            missing_strategy='median',
                            outlier_method='iqr',
                            outlier_treatment='clip',
                            iqr_multiplier=1.5,
                            zscore_threshold=3):
    """
    Titanic 데이터 전처리 파이프라인

    Parameters:
    -----------
    df : DataFrame - 원본 데이터
    numeric_cols : list - 수치형 컬럼 목록
    categorical_cols : list - 범주형 컬럼 목록
    missing_strategy : str - 결측치 처리 ('mean', 'median', 'drop')
    outlier_method : str - 이상치 탐지 방법 ('iqr', 'zscore')
    outlier_treatment : str - 이상치 처리 ('clip', 'remove', 'flag')
    iqr_multiplier : float - IQR 배수
    zscore_threshold : float - Z-score 임계값

    Returns:
    --------
    df_processed : DataFrame - 전처리된 데이터
    report : dict - 전처리 리포트
    """
    df_processed = df.copy()
    report = {
        'original_shape': df.shape,
        'missing_before': df.isnull().sum().to_dict(),
        'outliers_detected': {},
        'final_shape': None
    }

    # 1단계: 결측치 처리
    print("[Step 1] 결측치 처리")

    # 수치형 컬럼
    for col in numeric_cols:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            if missing_strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_processed[col].median()
            elif missing_strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            else:
                fill_value = df_processed[col].median()

            df_processed[col] = df_processed[col].fillna(fill_value)
            print(f"  {col}: 결측치 -> {fill_value:.2f}")

    # 범주형 컬럼
    if categorical_cols:
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                fill_value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
                df_processed[col] = df_processed[col].fillna(fill_value)
                print(f"  {col}: 결측치 -> {fill_value}")

    # 2단계: 이상치 처리
    print("\n[Step 2] 이상치 처리")

    for col in numeric_cols:
        if col not in df_processed.columns:
            continue

        # 이상치 탐지
        if outlier_method == 'iqr':
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            outlier_mask = (df_processed[col] < lower) | (df_processed[col] > upper)
        else:  # zscore
            z_scores = stats.zscore(df_processed[col])
            outlier_mask = np.abs(z_scores) > zscore_threshold
            lower = df_processed[col].mean() - zscore_threshold * df_processed[col].std()
            upper = df_processed[col].mean() + zscore_threshold * df_processed[col].std()

        n_outliers = outlier_mask.sum()
        report['outliers_detected'][col] = n_outliers

        # 이상치 처리
        if outlier_treatment == 'clip':
            df_processed[col] = df_processed[col].clip(lower=max(0, lower), upper=upper)
            print(f"  {col}: {n_outliers}개 이상치 clipping")
        elif outlier_treatment == 'remove':
            df_processed = df_processed[~outlier_mask]
            print(f"  {col}: {n_outliers}개 이상치 제거")
        elif outlier_treatment == 'flag':
            df_processed[f'{col}_outlier'] = outlier_mask
            print(f"  {col}: {n_outliers}개 이상치 플래그 추가")

    report['final_shape'] = df_processed.shape

    return df_processed, report


# 파이프라인 실행
print("\n파이프라인 실행:")
print("-" * 40)

df_final, preprocess_report = preprocess_titanic_data(
    df=df,
    numeric_cols=['age', 'fare'],
    categorical_cols=['embarked', 'deck'],
    missing_strategy='median',
    outlier_method='iqr',
    outlier_treatment='clip',
    iqr_multiplier=1.5
)

print("\n" + "=" * 40)
print("전처리 리포트:")
print("=" * 40)
print(f"원본 형태: {preprocess_report['original_shape']}")
print(f"최종 형태: {preprocess_report['final_shape']}")
print(f"\n결측치 (처리 전):")
for col, count in preprocess_report['missing_before'].items():
    if count > 0:
        print(f"  {col}: {count}개")
print(f"\n이상치 탐지:")
for col, count in preprocess_report['outliers_detected'].items():
    print(f"  {col}: {count}개")


# ============================================================
# 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("9차시 핵심 정리")
print("=" * 60)

print("""
[결측치 처리]
------------------------------------------------------
탐지: df.isnull().sum()
삭제: df.dropna()
대체: df.fillna(값)
     - 수치형: mean(), median()
     - 범주형: mode()[0]
     - 시계열: ffill, interpolate

[이상치 탐지]
------------------------------------------------------
IQR 방법:
  Q1 = quantile(0.25)
  Q3 = quantile(0.75)
  IQR = Q3 - Q1
  범위: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR

Z-score 방법:
  z = (x - mean) / std
  |z| > 3 -> 이상치

[이상치 처리]
------------------------------------------------------
제거: df = df[~outlier_mask]
Clipping: df['col'].clip(lower, upper)
플래그: df['outlier'] = mask

[전처리 순서]
------------------------------------------------------
1. 결측치 처리 (먼저!)
2. 이상치 처리
3. 스케일링/인코딩 (10차시)

[Titanic 데이터 인사이트]
------------------------------------------------------
- age: 결측치 20% -> 중앙값 대체 적합
- fare: 우측 치우침, 고가 요금 = 1등급 = 높은 생존율
- deck: 결측치 70%+ -> 삭제 또는 'Unknown'
""")

print("\n다음 차시 예고: 제조 데이터 전처리 (2) - 스케일링, 인코딩, 특성 엔지니어링")
print("=" * 60)
