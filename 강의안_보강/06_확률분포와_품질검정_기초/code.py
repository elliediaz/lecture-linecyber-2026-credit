"""
[6차시] 확률분포와 품질검정 기초 - 실습 코드

학습 목표:
1. 정규분포의 개념과 68-95-99.7 규칙을 이해한다
2. Z-score를 활용하여 이상치를 탐지한다

사용 데이터셋:
- UCI Wine Quality 데이터셋 (레드 와인 품질)
- seaborn의 diamonds 데이터셋 (다이아몬드 가격)

실습 환경:
- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정 (운영체제에 따라 조정)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("        6차시: 확률분포와 품질검정 기초 실습")
print("=" * 60)


# ============================================================
# 실제 공개 데이터셋 로드
# ============================================================
print("\n" + "=" * 60)
print("실제 공개 데이터셋 로드")
print("=" * 60)

# UCI Wine Quality 데이터셋 (레드 와인)
WINE_QUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    wine_quality = pd.read_csv(WINE_QUALITY_URL, sep=';')
    WINE_AVAILABLE = True
    print(f"\n[Wine Quality 데이터셋 로드 성공]")
    print(f"크기: {wine_quality.shape}")
    print(f"변수: {list(wine_quality.columns)}")
    print(f"\n처음 5행:")
    print(wine_quality.head())
except Exception as e:
    print(f"UCI 데이터 로드 실패 (인터넷 연결 확인): {e}")
    # 오프라인용 샘플 데이터
    np.random.seed(42)
    n = 500
    wine_quality = pd.DataFrame({
        'fixed acidity': np.random.normal(8.3, 1.7, n).round(1),
        'volatile acidity': np.random.normal(0.53, 0.18, n).round(2),
        'citric acid': np.random.normal(0.27, 0.19, n).clip(0).round(2),
        'residual sugar': np.random.normal(2.5, 1.4, n).clip(0.9).round(1),
        'chlorides': np.random.normal(0.087, 0.047, n).clip(0.01).round(3),
        'free sulfur dioxide': np.random.normal(16, 10, n).clip(1).round(0),
        'total sulfur dioxide': np.random.normal(46, 33, n).clip(6).round(0),
        'density': np.random.normal(0.997, 0.002, n).round(4),
        'pH': np.random.normal(3.31, 0.15, n).round(2),
        'sulphates': np.random.normal(0.66, 0.17, n).clip(0.3).round(2),
        'alcohol': np.random.normal(10.4, 1.1, n).clip(8.4).round(1),
        'quality': np.random.choice([3, 4, 5, 6, 7, 8], n, p=[0.01, 0.05, 0.42, 0.40, 0.10, 0.02])
    })
    WINE_AVAILABLE = True
    print("\n[오프라인 모드] 시뮬레이션 데이터 생성")
    print(wine_quality.head())

# seaborn의 diamonds 데이터셋
try:
    import seaborn as sns
    diamonds = sns.load_dataset('diamonds')
    DIAMONDS_AVAILABLE = True
    print(f"\n[Diamonds 데이터셋 로드 성공]")
    print(f"크기: {diamonds.shape}")
    print(f"변수: {list(diamonds.columns)}")
except ImportError:
    DIAMONDS_AVAILABLE = False
    print("seaborn이 설치되어 있지 않습니다.")


# ============================================================
# Part 1: 정규분포의 개념과 68-95-99.7 규칙
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 정규분포의 개념과 68-95-99.7 규칙")
print("=" * 60)


# 실습 1-1: Wine Quality 데이터의 정규분포 확인
print("\n[실습 1-1] Wine Quality 데이터 분포 확인")
print("-" * 40)

if WINE_AVAILABLE:
    # 알코올 도수 분포 분석
    alcohol = wine_quality['alcohol'].values

    print(f"데이터 개수: {len(alcohol)}")
    print(f"평균: {alcohol.mean():.2f}%")
    print(f"표준편차: {alcohol.std():.2f}%")
    print(f"최소값: {alcohol.min():.1f}%")
    print(f"최대값: {alcohol.max():.1f}%")

    # 정규성 검정 (Shapiro-Wilk test)
    # 샘플 크기가 크면 일부만 사용
    sample_size = min(len(alcohol), 500)
    stat, p_value = stats.shapiro(alcohol[:sample_size])
    print(f"\nShapiro-Wilk 정규성 검정:")
    print(f"  통계량: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  -> 정규분포를 따른다고 볼 수 있음 (p > 0.05)")
    else:
        print("  -> 정규분포가 아닐 수 있음 (p <= 0.05)")


# 실습 1-2: 정규분포 시각화
print("\n[실습 1-2] 정규분포 시각화")
print("-" * 40)

if WINE_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 알코올 도수 히스토그램
    mean, std = alcohol.mean(), alcohol.std()

    axes[0].hist(alcohol, bins=30, edgecolor='black', alpha=0.7,
                 color='steelblue', density=True)
    axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'평균: {mean:.1f}%')
    axes[0].axvline(mean - std, color='orange', linestyle=':', linewidth=2, label=f'-1s: {mean-std:.1f}%')
    axes[0].axvline(mean + std, color='orange', linestyle=':', linewidth=2, label=f'+1s: {mean+std:.1f}%')
    axes[0].axvline(mean - 2*std, color='yellow', linestyle=':', linewidth=1.5)
    axes[0].axvline(mean + 2*std, color='yellow', linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('알코올 도수 (%)')
    axes[0].set_ylabel('밀도')
    axes[0].set_title('와인 알코올 도수 분포')
    axes[0].legend(loc='upper right')

    # 이론적 정규분포 곡선과 비교
    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    y = stats.norm.pdf(x, mean, std)

    axes[1].hist(alcohol, bins=30, density=True, alpha=0.5,
                 color='steelblue', edgecolor='black', label='실제 데이터')
    axes[1].plot(x, y, 'r-', linewidth=2, label='이론적 정규분포')
    axes[1].set_xlabel('알코올 도수 (%)')
    axes[1].set_ylabel('밀도')
    axes[1].set_title('실제 데이터 vs 이론적 정규분포')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('01_wine_alcohol_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("그래프가 01_wine_alcohol_distribution.png로 저장되었습니다.")


# 실습 1-3: 68-95-99.7 규칙 검증
print("\n[실습 1-3] 68-95-99.7 규칙 검증 (Wine Quality)")
print("-" * 40)

if WINE_AVAILABLE:
    # 각 범위에 속하는 데이터 비율 계산
    within_1std = np.sum((alcohol >= mean - std) &
                         (alcohol <= mean + std)) / len(alcohol)
    within_2std = np.sum((alcohol >= mean - 2*std) &
                         (alcohol <= mean + 2*std)) / len(alcohol)
    within_3std = np.sum((alcohol >= mean - 3*std) &
                         (alcohol <= mean + 3*std)) / len(alcohol)

    print("=== 68-95-99.7 규칙 검증 (알코올 도수) ===")
    print(f"평균: {mean:.2f}%, 표준편차: {std:.2f}%")
    print(f"\n+-1s 범위 ({mean-std:.1f}% ~ {mean+std:.1f}%): {within_1std:.1%} (이론: 68.0%)")
    print(f"+-2s 범위 ({mean-2*std:.1f}% ~ {mean+2*std:.1f}%): {within_2std:.1%} (이론: 95.0%)")
    print(f"+-3s 범위 ({mean-3*std:.1f}% ~ {mean+3*std:.1f}%): {within_3std:.1%} (이론: 99.7%)")


# 실습 1-4: 다른 변수들의 정규분포 확인
print("\n[실습 1-4] 다른 변수들의 정규분포 확인")
print("-" * 40)

if WINE_AVAILABLE:
    variables_to_check = ['pH', 'density', 'fixed acidity', 'volatile acidity']

    print("=== 와인 품질 데이터 변수별 정규성 검정 ===\n")

    for var in variables_to_check:
        data = wine_quality[var].values
        sample = data[:500] if len(data) > 500 else data
        stat, p_value = stats.shapiro(sample)

        result = "정규분포" if p_value > 0.05 else "비정규분포"
        print(f"{var:20s}: p-value = {p_value:.4f} -> {result}")


# ============================================================
# Part 2: Z-score를 활용한 이상치 탐지
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Z-score를 활용한 이상치 탐지")
print("=" * 60)


# 실습 2-1: Wine Quality 데이터의 Z-score 계산
print("\n[실습 2-1] Z-score 계산 (Wine Quality)")
print("-" * 40)

if WINE_AVAILABLE:
    # 알코올 도수의 Z-score 계산
    mean_alcohol = wine_quality['alcohol'].mean()
    std_alcohol = wine_quality['alcohol'].std()

    wine_quality['alcohol_zscore'] = (wine_quality['alcohol'] - mean_alcohol) / std_alcohol

    print(f"알코올 도수 평균: {mean_alcohol:.2f}%")
    print(f"알코올 도수 표준편차: {std_alcohol:.2f}%")
    print(f"\n=== Z-score 분포 ===")
    print(wine_quality['alcohol_zscore'].describe().round(3))

    # 극단값 확인
    print(f"\n가장 높은 알코올 도수 (상위 5개):")
    top5 = wine_quality.nlargest(5, 'alcohol')[['alcohol', 'alcohol_zscore', 'quality']]
    print(top5)

    print(f"\n가장 낮은 알코올 도수 (하위 5개):")
    bottom5 = wine_quality.nsmallest(5, 'alcohol')[['alcohol', 'alcohol_zscore', 'quality']]
    print(bottom5)


# 실습 2-2: scipy를 활용한 Z-score 계산
print("\n[실습 2-2] scipy를 활용한 Z-score 계산")
print("-" * 40)

if WINE_AVAILABLE:
    # scipy 방식
    z_scores_scipy = stats.zscore(wine_quality['alcohol'])

    print("scipy.stats.zscore() 결과 확인:")
    print(f"직접 계산 Z-score (처음 5개): {wine_quality['alcohol_zscore'].head().values.round(4)}")
    print(f"scipy Z-score (처음 5개): {z_scores_scipy[:5].round(4)}")


# 실습 2-3: 이상치 탐지 함수
print("\n[실습 2-3] 이상치 탐지 함수")
print("-" * 40)

def detect_outliers_zscore(data, threshold=2):
    """
    Z-score 기반 이상치 탐지 함수

    Parameters:
        data: 데이터 배열 또는 Series
        threshold: Z-score 임계값 (기본값 2)

    Returns:
        outlier_mask: 이상치 여부 (True/False)
        z_scores: Z-score 배열
    """
    data_clean = data.dropna() if hasattr(data, 'dropna') else data
    mean = np.mean(data_clean)
    std = np.std(data_clean)
    z_scores = (data_clean - mean) / std
    outlier_mask = np.abs(z_scores) > threshold

    return outlier_mask, z_scores


# Wine Quality 데이터에 적용
if WINE_AVAILABLE:
    # 알코올 도수 이상치 탐지
    outlier_mask, z_scores = detect_outliers_zscore(wine_quality['alcohol'], threshold=2)

    print(f"이상치 기준: |Z| > 2")
    print(f"전체 데이터 수: {len(wine_quality)}")
    print(f"이상치 개수: {outlier_mask.sum()}")
    print(f"이상치 비율: {outlier_mask.sum()/len(wine_quality)*100:.2f}%")

    # 이상치 상세 정보
    outliers = wine_quality[outlier_mask][['alcohol', 'quality']]
    print(f"\n이상치 데이터 (처음 10개):")
    print(outliers.head(10))


# 실습 2-4: 이상치 시각화
print("\n[실습 2-4] 이상치 시각화")
print("-" * 40)

if WINE_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 알코올 도수 분포와 이상치
    outlier_mask, z_scores = detect_outliers_zscore(wine_quality['alcohol'], threshold=2)

    colors = ['red' if is_outlier else 'steelblue' for is_outlier in outlier_mask]
    axes[0].scatter(range(len(wine_quality)), wine_quality['alcohol'],
                    c=colors, alpha=0.5, s=10)
    axes[0].axhline(mean_alcohol, color='green', linestyle='--', linewidth=2,
                    label=f'평균: {mean_alcohol:.1f}%')
    axes[0].axhline(mean_alcohol + 2*std_alcohol, color='orange', linestyle=':',
                    linewidth=2, label=f'+2s: {mean_alcohol + 2*std_alcohol:.1f}%')
    axes[0].axhline(mean_alcohol - 2*std_alcohol, color='orange', linestyle=':',
                    linewidth=2, label=f'-2s: {mean_alcohol - 2*std_alcohol:.1f}%')
    axes[0].set_xlabel('샘플 인덱스')
    axes[0].set_ylabel('알코올 도수 (%)')
    axes[0].set_title('알코올 도수 이상치 탐지\n(빨간점 = 이상치)')
    axes[0].legend(loc='upper right')

    # Z-score 분포
    colors_z = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
    axes[1].scatter(range(len(z_scores)), z_scores, c=colors_z, alpha=0.5, s=10)
    axes[1].axhline(2, color='orange', linestyle='--', linewidth=2, label='+2 기준')
    axes[1].axhline(-2, color='orange', linestyle='--', linewidth=2, label='-2 기준')
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('샘플 인덱스')
    axes[1].set_ylabel('Z-score')
    axes[1].set_title('Z-score 분포\n(|Z| > 2 = 이상치 의심)')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('02_zscore_outliers_wine.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("그래프가 02_zscore_outliers_wine.png로 저장되었습니다.")


# 실습 2-5: Z-score vs IQR 비교
print("\n[실습 2-5] Z-score vs IQR 비교")
print("-" * 40)

def detect_outliers_iqr(data):
    """IQR 기반 이상치 탐지"""
    data_clean = data.dropna() if hasattr(data, 'dropna') else data
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
    return outlier_mask, lower_bound, upper_bound


if WINE_AVAILABLE:
    # 두 방법 비교
    alcohol_data = wine_quality['alcohol']

    outliers_z, _ = detect_outliers_zscore(alcohol_data, threshold=2)
    outliers_iqr, lower, upper = detect_outliers_iqr(alcohol_data)

    print("=== 이상치 탐지 방법 비교 (알코올 도수) ===")
    print(f"\nZ-score 방법 (|Z| > 2):")
    print(f"  이상치 개수: {outliers_z.sum()}개 ({outliers_z.sum()/len(alcohol_data)*100:.2f}%)")

    print(f"\nIQR 방법 (1.5 x IQR):")
    print(f"  이상치 범위: {lower:.1f}% 미만 또는 {upper:.1f}% 초과")
    print(f"  이상치 개수: {outliers_iqr.sum()}개 ({outliers_iqr.sum()/len(alcohol_data)*100:.2f}%)")

    # 두 방법의 교집합
    both_methods = outliers_z & outliers_iqr
    print(f"\n두 방법 모두에서 탐지된 이상치: {both_methods.sum()}개")


# ============================================================
# Part 3: Diamonds 데이터셋을 활용한 추가 분석
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Diamonds 데이터셋 분석")
print("=" * 60)

if DIAMONDS_AVAILABLE:
    # 3-1. 다이아몬드 가격 분포
    print("\n[3-1] 다이아몬드 가격 분포 분석")
    print("-" * 40)

    price = diamonds['price'].values
    carat = diamonds['carat'].values

    print(f"다이아몬드 개수: {len(diamonds):,}")
    print(f"\n가격 통계:")
    print(f"  평균: ${price.mean():,.2f}")
    print(f"  중앙값: ${np.median(price):,.2f}")
    print(f"  표준편차: ${price.std():,.2f}")
    print(f"  최소: ${price.min():,}")
    print(f"  최대: ${price.max():,}")

    # 3-2. 가격 이상치 탐지
    print("\n[3-2] 가격 이상치 탐지")
    print("-" * 40)

    outliers_z_price, z_price = detect_outliers_zscore(diamonds['price'], threshold=3)
    outliers_iqr_price, lower_p, upper_p = detect_outliers_iqr(diamonds['price'])

    print(f"Z-score 방법 (|Z| > 3): {outliers_z_price.sum():,}개 이상치")
    print(f"IQR 방법: ${lower_p:,.0f} ~ ${upper_p:,.0f} 범위 벗어나는 {outliers_iqr_price.sum():,}개 이상치")

    # 3-3. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 가격 히스토그램
    axes[0, 0].hist(price, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(price.mean(), color='red', linestyle='--', label=f'평균: ${price.mean():,.0f}')
    axes[0, 0].axvline(np.median(price), color='green', linestyle='--', label=f'중앙값: ${np.median(price):,.0f}')
    axes[0, 0].set_xlabel('가격 ($)')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].set_title('다이아몬드 가격 분포')
    axes[0, 0].legend()

    # 로그 변환된 가격
    log_price = np.log10(price)
    axes[0, 1].hist(log_price, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('log10(가격)')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].set_title('다이아몬드 가격 분포 (로그 변환)')

    # 캐럿 vs 가격 산점도
    axes[1, 0].scatter(carat, price, alpha=0.1, s=5)
    axes[1, 0].set_xlabel('캐럿')
    axes[1, 0].set_ylabel('가격 ($)')
    axes[1, 0].set_title('캐럿 vs 가격')

    # 품질(cut)별 가격 상자그림
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut_prices = [diamonds[diamonds['cut'] == c]['price'].values for c in cut_order]
    axes[1, 1].boxplot(cut_prices, labels=cut_order)
    axes[1, 1].set_xlabel('컷 품질')
    axes[1, 1].set_ylabel('가격 ($)')
    axes[1, 1].set_title('컷 품질별 가격 분포')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('03_diamonds_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n그래프가 03_diamonds_analysis.png로 저장되었습니다.")


# ============================================================
# 종합 실습: 품질 관리 리포트
# ============================================================
print("\n" + "=" * 60)
print("종합 실습: 품질 관리 리포트")
print("=" * 60)

def generate_quality_report(df, variable, name="데이터"):
    """
    품질 관리 종합 리포트 생성

    Parameters:
        df: DataFrame
        variable: 분석할 변수명
        name: 리포트 이름
    """
    print("=" * 60)
    print(f"              {name} 품질 분석 리포트")
    print("=" * 60)

    data = df[variable].dropna()

    # 1. 기술통계
    print(f"\n[1. 기술통계]")
    print(f"    샘플 수: {len(data):,}")
    print(f"    평균: {data.mean():.2f}")
    print(f"    표준편차: {data.std():.2f}")
    print(f"    최소값: {data.min():.2f}")
    print(f"    최대값: {data.max():.2f}")

    # 2. 정규성 검정
    print(f"\n[2. 정규성 검정]")
    sample = data[:500] if len(data) > 500 else data
    stat, p_value = stats.shapiro(sample)
    print(f"    Shapiro-Wilk: p-value = {p_value:.4f}")
    if p_value > 0.05:
        print(f"    -> 정규분포를 따름")
    else:
        print(f"    -> 정규분포가 아닐 수 있음")

    # 3. 68-95-99.7 규칙
    print(f"\n[3. 68-95-99.7 규칙 적용]")
    mean, std = data.mean(), data.std()
    within_1 = ((data >= mean - std) & (data <= mean + std)).sum() / len(data)
    within_2 = ((data >= mean - 2*std) & (data <= mean + 2*std)).sum() / len(data)
    within_3 = ((data >= mean - 3*std) & (data <= mean + 3*std)).sum() / len(data)
    print(f"    +-1s 범위: {within_1:.1%} (이론: 68%)")
    print(f"    +-2s 범위: {within_2:.1%} (이론: 95%)")
    print(f"    +-3s 범위: {within_3:.1%} (이론: 99.7%)")

    # 4. 이상치 탐지
    print(f"\n[4. 이상치 탐지]")
    outliers_z, _ = detect_outliers_zscore(data, threshold=2)
    outliers_iqr, lower, upper = detect_outliers_iqr(data)
    print(f"    Z-score (|Z|>2): {outliers_z.sum()}개")
    print(f"    IQR 방법: {outliers_iqr.sum()}개")

    # 5. 권고사항
    print(f"\n[5. 권고사항]")
    if outliers_z.sum() > len(data) * 0.05:
        print(f"    - 이상치 비율이 5%를 초과합니다. 데이터 점검 필요")
    else:
        print(f"    - 이상치 비율이 정상 범위입니다")

    if p_value < 0.05:
        print(f"    - 정규분포를 따르지 않으므로 비모수 방법 고려")

    print("\n" + "=" * 60)

    return {
        'mean': data.mean(),
        'std': data.std(),
        'outlier_count': outliers_z.sum(),
        'normality_pvalue': p_value
    }


# 리포트 생성
if WINE_AVAILABLE:
    report = generate_quality_report(wine_quality, 'alcohol', '와인 알코올 도수')


# ============================================================
# 연습 문제
# ============================================================
print("\n" + "=" * 60)
print("연습 문제")
print("=" * 60)

print("""
[연습 1] Wine Quality 데이터에서 pH의 Z-score를 계산하고 이상치를 찾으세요.
         기준: |Z| > 2

[연습 2] diamonds 데이터에서 carat의 분포를 분석하고,
         68-95-99.7 규칙이 적용되는지 확인하세요.

[연습 3] Wine Quality 데이터에서 quality(품질 등급)별로
         알코올 도수의 평균과 표준편차를 비교하세요.
""")


# 연습 1 정답
print("\n[연습 1 정답]")
if WINE_AVAILABLE:
    outliers, z = detect_outliers_zscore(wine_quality['pH'], threshold=2)
    print(f"pH 이상치 개수: {outliers.sum()}")
    print(f"pH 이상치 값 (처음 5개): {wine_quality[outliers]['pH'].head().values}")

# 연습 2 정답
print("\n[연습 2 정답]")
if DIAMONDS_AVAILABLE:
    carat = diamonds['carat']
    mean_c, std_c = carat.mean(), carat.std()
    within_1 = ((carat >= mean_c - std_c) & (carat <= mean_c + std_c)).sum() / len(carat)
    within_2 = ((carat >= mean_c - 2*std_c) & (carat <= mean_c + 2*std_c)).sum() / len(carat)
    print(f"캐럿 +-1s: {within_1:.1%} (이론: 68%)")
    print(f"캐럿 +-2s: {within_2:.1%} (이론: 95%)")
    print(f"-> 정규분포와 차이가 있음 (오른쪽 꼬리가 긴 분포)")

# 연습 3 정답
print("\n[연습 3 정답]")
if WINE_AVAILABLE:
    quality_stats = wine_quality.groupby('quality')['alcohol'].agg(['mean', 'std', 'count'])
    print(quality_stats.round(2))
    print("\n-> 품질이 높을수록 알코올 도수가 높은 경향")


print("\n" + "=" * 60)
print("6차시 실습을 완료했습니다!")
print("=" * 60)

print("""
오늘 배운 핵심 내용:

1. 정규분포와 68-95-99.7 규칙
   - 평균 +-1s: 약 68%
   - 평균 +-2s: 약 95%
   - 평균 +-3s: 약 99.7%

2. Z-score 계산
   z = (x - mean) / std
   scipy: stats.zscore(data)

3. 이상치 탐지
   - Z-score 방법: |Z| > 2 또는 3
   - IQR 방법: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR

4. 사용한 실제 데이터셋
   - UCI Wine Quality: 와인 품질 및 화학 성분
   - seaborn diamonds: 다이아몬드 가격 및 특성

다음 시간: 7차시 - 통계 검정 실습 (t-검정, 카이제곱 검정, ANOVA 등)
""")
