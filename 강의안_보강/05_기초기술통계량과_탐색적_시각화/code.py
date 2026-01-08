"""
========================================
5차시: 기초 기술통계량과 탐색적 시각화
========================================
실습 코드

학습내용:
1. 대표값의 의미
2. 데이터의 퍼짐 정도
3. 제조 품질 측정값의 탐색적 시각화

사용 데이터셋:
- sklearn의 Iris 데이터셋 (붓꽃 측정 데이터)
- seaborn의 tips 데이터셋 (레스토랑 팁 데이터)
- seaborn의 penguins 데이터셋 (펭귄 측정 데이터)
"""

# =====================================================
# 실습 환경 준비
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 또는 Mac의 경우
# plt.rcParams['font.family'] = 'AppleGothic'

print("=" * 60)
print("5차시: 기초 기술통계량과 탐색적 시각화")
print("=" * 60)

# =====================================================
# 실제 데이터셋 로드
# =====================================================
print("\n" + "=" * 60)
print("실제 공개 데이터셋 로드")
print("=" * 60)

# sklearn Iris 데이터셋 로드
try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    IRIS_AVAILABLE = True
    print(f"\n[Iris 데이터셋 로드 성공]")
    print(f"크기: {iris_df.shape}")
    print(f"변수: {list(iris_df.columns)}")
except ImportError:
    IRIS_AVAILABLE = False
    print("sklearn이 설치되어 있지 않습니다.")

# seaborn 데이터셋 로드
try:
    import seaborn as sns
    tips = sns.load_dataset('tips')
    penguins = sns.load_dataset('penguins')
    SEABORN_AVAILABLE = True
    print(f"\n[Tips 데이터셋 로드 성공]")
    print(f"크기: {tips.shape}")
    print(f"\n[Penguins 데이터셋 로드 성공]")
    print(f"크기: {penguins.shape}")
except ImportError:
    SEABORN_AVAILABLE = False
    print("seaborn이 설치되어 있지 않습니다.")


# =====================================================
# 실습 1: 대표값 계산 (Iris 데이터셋 활용)
# =====================================================
print("\n" + "=" * 60)
print("실습 1: 대표값 (평균, 중앙값, 최빈값)")
print("=" * 60)

if IRIS_AVAILABLE:
    # Iris 데이터셋의 꽃잎 길이(petal length) 사용
    petal_length = iris_df['petal length (cm)'].values

    print("\n[1-1. Iris 꽃잎 길이 대표값]")
    print(f"데이터 개수: {len(petal_length)}")
    print(f"평균 (Mean): {np.mean(petal_length):.2f} cm")
    print(f"중앙값 (Median): {np.median(petal_length):.2f} cm")
    print(f"최빈값 (Mode): {pd.Series(petal_length.round(1)).mode().values} cm")

    # 1-2. 종별 대표값 비교
    print("\n[1-2. 종별 꽃잎 길이 대표값]")
    for species in iris_df['species'].unique():
        species_data = iris_df[iris_df['species'] == species]['petal length (cm)']
        print(f"\n{species}:")
        print(f"  평균: {species_data.mean():.2f} cm")
        print(f"  중앙값: {species_data.median():.2f} cm")
        print(f"  표준편차: {species_data.std():.2f} cm")

# 1-3. 대표값 비교 함수
def compare_central_tendency(data, name="데이터"):
    """대표값 비교 함수"""
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = pd.Series(data).mode().values

    print(f"\n[{name} 대표값 분석]")
    print(f"  평균: {mean_val:.2f}")
    print(f"  중앙값: {median_val:.2f}")
    print(f"  최빈값: {mode_val}")
    print(f"  평균-중앙값 차이: {mean_val - median_val:.2f}")

    if abs(mean_val - median_val) > 0.5:
        print("  * 평균과 중앙값 차이가 큼 -> 비대칭 분포 가능성")
    else:
        print("  * 평균과 중앙값이 유사 -> 대칭 분포")

if IRIS_AVAILABLE:
    compare_central_tendency(iris_df['sepal length (cm)'], "꽃받침 길이")
    compare_central_tendency(iris_df['petal length (cm)'], "꽃잎 길이")


# =====================================================
# 실습 2: 산포도 계산 (Tips 데이터셋 활용)
# =====================================================
print("\n" + "=" * 60)
print("실습 2: 산포도 (범위, 분산, 표준편차, IQR)")
print("=" * 60)

if SEABORN_AVAILABLE:
    # Tips 데이터셋의 total_bill 사용
    total_bill = tips['total_bill'].values

    print("\n[2-1. 식사 금액(total_bill) 산포도]")
    print(f"최소값: ${np.min(total_bill):.2f}")
    print(f"최대값: ${np.max(total_bill):.2f}")
    print(f"범위 (Range): ${np.ptp(total_bill):.2f}")
    print(f"분산 (Variance): {np.var(total_bill, ddof=1):.2f}")
    print(f"표준편차 (Std): ${np.std(total_bill, ddof=1):.2f}")

    # 2-2. 사분위수와 IQR
    Q1 = np.percentile(total_bill, 25)
    Q2 = np.percentile(total_bill, 50)  # 중앙값
    Q3 = np.percentile(total_bill, 75)
    IQR = Q3 - Q1

    print("\n[2-2. 사분위수]")
    print(f"Q1 (25%): ${Q1:.2f}")
    print(f"Q2 (50%, 중앙값): ${Q2:.2f}")
    print(f"Q3 (75%): ${Q3:.2f}")
    print(f"IQR (Q3-Q1): ${IQR:.2f}")

    # 2-3. 이상치 기준
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("\n[2-3. 이상치 기준]")
    print(f"이상치 하한: ${lower_bound:.2f}")
    print(f"이상치 상한: ${upper_bound:.2f}")

    # 이상치 탐지
    outliers = total_bill[(total_bill < lower_bound) | (total_bill > upper_bound)]
    print(f"이상치 개수: {len(outliers)}")
    if len(outliers) > 0:
        print(f"이상치 값: {outliers}")

# 2-4. 산포도 분석 함수
def analyze_dispersion(data, name="데이터"):
    """산포도 분석 함수"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    print(f"\n[{name} 산포도 분석]")
    print(f"  범위: {np.ptp(data):.2f}")
    print(f"  표준편차: {np.std(data, ddof=1):.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  변동계수(CV): {np.std(data, ddof=1)/np.mean(data)*100:.1f}%")

    # 이상치 탐지
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = np.sum((data < lower) | (data > upper))
    print(f"  이상치 개수: {outlier_count}")

if SEABORN_AVAILABLE:
    analyze_dispersion(tips['total_bill'], "식사 금액")
    analyze_dispersion(tips['tip'], "팁 금액")


# =====================================================
# 실습 3: describe() 활용 (Penguins 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 3: describe() 한 줄 요약")
print("=" * 60)

if SEABORN_AVAILABLE:
    print("\n[Penguins 데이터셋 기술통계]")
    # 결측치 제거 후 분석
    penguins_clean = penguins.dropna()
    print(f"데이터 크기: {penguins_clean.shape}")
    print("\n[describe() 결과]")
    print(penguins_clean.describe().round(2))

    # 종별 기술통계
    print("\n[종별 체중 통계]")
    print(penguins_clean.groupby('species')['body_mass_g'].describe().round(1))


# =====================================================
# 실습 4: 히스토그램 (Iris 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 4: 히스토그램")
print("=" * 60)

if IRIS_AVAILABLE:
    # 4-1. Iris 꽃잎 길이 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 전체 분포
    petal_length = iris_df['petal length (cm)']
    axes[0].hist(petal_length, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(petal_length.mean(), color='red', linestyle='-',
                    linewidth=2, label=f'평균: {petal_length.mean():.1f}')
    axes[0].axvline(petal_length.median(), color='green', linestyle='--',
                    linewidth=2, label=f'중앙값: {petal_length.median():.1f}')
    axes[0].set_xlabel('꽃잎 길이 (cm)')
    axes[0].set_ylabel('빈도')
    axes[0].set_title('Iris 꽃잎 길이 분포 (전체)')
    axes[0].legend()

    # 종별 분포
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, species in enumerate(iris_df['species'].unique()):
        species_data = iris_df[iris_df['species'] == species]['petal length (cm)']
        axes[1].hist(species_data, bins=15, alpha=0.6, label=species,
                     color=colors[i], edgecolor='black')
    axes[1].set_xlabel('꽃잎 길이 (cm)')
    axes[1].set_ylabel('빈도')
    axes[1].set_title('Iris 꽃잎 길이 분포 (종별)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('histogram_iris.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n히스토그램이 'histogram_iris.png'로 저장되었습니다.")


# =====================================================
# 실습 5: 상자그림 (Penguins 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 5: 상자그림 (Box Plot)")
print("=" * 60)

if SEABORN_AVAILABLE:
    penguins_clean = penguins.dropna()

    # 5-1. 종별 체중 상자그림
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 종별 체중
    species_data = [penguins_clean[penguins_clean['species'] == s]['body_mass_g'].values
                    for s in penguins_clean['species'].unique()]
    species_labels = penguins_clean['species'].unique()

    box_colors = ['lightgreen', 'lightyellow', 'lightcoral']
    bp = axes[0].boxplot(species_data, labels=species_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    axes[0].set_ylabel('체중 (g)')
    axes[0].set_title('펭귄 종별 체중 분포')
    axes[0].grid(True, alpha=0.3)

    # 섬별 부리 길이
    island_data = [penguins_clean[penguins_clean['island'] == i]['bill_length_mm'].values
                   for i in penguins_clean['island'].unique()]
    island_labels = penguins_clean['island'].unique()

    bp2 = axes[1].boxplot(island_data, labels=island_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightyellow', 'lightpink']):
        patch.set_facecolor(color)
    axes[1].set_ylabel('부리 길이 (mm)')
    axes[1].set_title('섬별 펭귄 부리 길이 분포')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boxplot_penguins.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 5-2. 종별 통계 비교
    print("\n[종별 체중 통계 비교]")
    for species in penguins_clean['species'].unique():
        species_weights = penguins_clean[penguins_clean['species'] == species]['body_mass_g']
        Q1 = species_weights.quantile(0.25)
        Q3 = species_weights.quantile(0.75)
        IQR = Q3 - Q1
        outliers = species_weights[(species_weights < Q1 - 1.5*IQR) |
                                    (species_weights > Q3 + 1.5*IQR)]
        print(f"\n{species}:")
        print(f"  평균: {species_weights.mean():.1f}g, 표준편차: {species_weights.std():.1f}g")
        print(f"  IQR: {IQR:.1f}g, 이상치: {len(outliers)}개")


# =====================================================
# 실습 6: 산점도 (Tips 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 6: 산점도 (Scatter Plot)")
print("=" * 60)

if SEABORN_AVAILABLE:
    # 6-1. total_bill vs tip 산점도
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 전체 데이터
    axes[0].scatter(tips['total_bill'], tips['tip'], alpha=0.6, c='blue')
    z = np.polyfit(tips['total_bill'], tips['tip'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(tips['total_bill'].min(), tips['total_bill'].max(), 100)
    axes[0].plot(x_line, p(x_line), 'r--', alpha=0.8, label='추세선')
    corr = tips['total_bill'].corr(tips['tip'])
    axes[0].set_xlabel('식사 금액 ($)')
    axes[0].set_ylabel('팁 ($)')
    axes[0].set_title(f'식사 금액 vs 팁 (r={corr:.2f})')
    axes[0].legend()

    # 시간대별 (점심 vs 저녁)
    for time_val, color in [('Lunch', 'orange'), ('Dinner', 'purple')]:
        subset = tips[tips['time'] == time_val]
        axes[1].scatter(subset['total_bill'], subset['tip'],
                        alpha=0.6, label=time_val, c=color)
    axes[1].set_xlabel('식사 금액 ($)')
    axes[1].set_ylabel('팁 ($)')
    axes[1].set_title('시간대별 식사 금액 vs 팁')
    axes[1].legend()

    # 흡연 여부별
    for smoker, color in [('Yes', 'red'), ('No', 'green')]:
        subset = tips[tips['smoker'] == smoker]
        axes[2].scatter(subset['total_bill'], subset['tip'],
                        alpha=0.6, label=f'흡연: {smoker}', c=color)
    axes[2].set_xlabel('식사 금액 ($)')
    axes[2].set_ylabel('팁 ($)')
    axes[2].set_title('흡연 여부별 식사 금액 vs 팁')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('scatter_tips.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n[상관계수 분석]")
    print(f"식사 금액-팁: r = {tips['total_bill'].corr(tips['tip']):.3f}")
    print(f"인원-팁: r = {tips['size'].corr(tips['tip']):.3f}")
    print(f"식사 금액-인원: r = {tips['total_bill'].corr(tips['size']):.3f}")


# =====================================================
# 실습 7: 종합 대시보드 (Iris 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 7: 종합 대시보드")
print("=" * 60)

if IRIS_AVAILABLE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Iris 데이터셋 탐색적 분석 대시보드', fontsize=14, fontweight='bold')

    # 1. 꽃잎 길이 히스토그램
    axes[0, 0].hist(iris_df['petal length (cm)'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(iris_df['petal length (cm)'].mean(), color='red', linestyle='--',
                       label=f"평균: {iris_df['petal length (cm)'].mean():.1f}")
    axes[0, 0].set_title('꽃잎 길이 분포')
    axes[0, 0].set_xlabel('길이 (cm)')
    axes[0, 0].legend()

    # 2. 종별 꽃잎 길이 상자그림
    species_petal = [iris_df[iris_df['species'] == s]['petal length (cm)'].values
                     for s in iris_df['species'].unique()]
    axes[0, 1].boxplot(species_petal, labels=iris_df['species'].unique())
    axes[0, 1].set_title('종별 꽃잎 길이')
    axes[0, 1].set_ylabel('길이 (cm)')

    # 3. 꽃잎 길이 vs 꽃잎 너비 산점도
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species in iris_df['species'].unique():
        subset = iris_df[iris_df['species'] == species]
        axes[0, 2].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                           alpha=0.7, label=species, c=colors[species])
    axes[0, 2].set_title('꽃잎 길이 vs 너비')
    axes[0, 2].set_xlabel('길이 (cm)')
    axes[0, 2].set_ylabel('너비 (cm)')
    axes[0, 2].legend()

    # 4. 꽃받침 길이 분포
    axes[1, 0].hist(iris_df['sepal length (cm)'], bins=20, edgecolor='black',
                    alpha=0.7, color='orange')
    axes[1, 0].set_title('꽃받침 길이 분포')
    axes[1, 0].set_xlabel('길이 (cm)')

    # 5. 종별 꽃받침 너비 막대 그래프
    sepal_width_mean = iris_df.groupby('species')['sepal width (cm)'].mean()
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[1, 1].bar(sepal_width_mean.index, sepal_width_mean.values, color=colors_bar)
    axes[1, 1].set_title('종별 평균 꽃받침 너비')
    axes[1, 1].set_ylabel('너비 (cm)')

    # 6. 기술통계 테이블
    axes[1, 2].axis('off')
    stats_text = iris_df[['sepal length (cm)', 'sepal width (cm)',
                          'petal length (cm)', 'petal width (cm)']].describe().round(2).to_string()
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='center', family='monospace')
    axes[1, 2].set_title('기술통계 요약')

    plt.tight_layout()
    plt.savefig('iris_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n대시보드가 'iris_dashboard.png'로 저장되었습니다.")


# =====================================================
# 실습 8: 이상치 탐지 (Penguins 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 8: 이상치 탐지")
print("=" * 60)

def detect_outliers(data, name="데이터"):
    """IQR 방법으로 이상치 탐지"""
    data_clean = data.dropna()  # 결측치 제거
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_mask = (data_clean < lower) | (data_clean > upper)
    outliers = data_clean[outliers_mask]

    print(f"\n[{name} 이상치 탐지]")
    print(f"  정상 범위: {lower:.2f} ~ {upper:.2f}")
    print(f"  이상치 개수: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  이상치 값: {outliers.values}")

    return outliers_mask

if SEABORN_AVAILABLE:
    penguins_clean = penguins.dropna()

    # 체중 이상치 탐지
    outlier_mask = detect_outliers(penguins_clean['body_mass_g'], "펭귄 체중")

    # 부리 길이 이상치 탐지
    detect_outliers(penguins_clean['bill_length_mm'], "부리 길이")

    # 이상치 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    body_mass = penguins_clean['body_mass_g']
    Q1 = body_mass.quantile(0.25)
    Q3 = body_mass.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_mask = (body_mass < lower) | (body_mass > upper)
    ax.scatter(range(len(body_mass)), body_mass,
               c=outlier_mask.map({True: 'red', False: 'blue'}), alpha=0.6)
    ax.axhline(lower, color='orange', linestyle='--', label=f'하한: {lower:.0f}g')
    ax.axhline(upper, color='orange', linestyle='--', label=f'상한: {upper:.0f}g')
    ax.set_xlabel('인덱스')
    ax.set_ylabel('체중 (g)')
    ax.set_title('펭귄 체중 이상치 탐지 (빨간점: 이상치)')
    ax.legend()
    plt.savefig('outlier_detection_penguins.png', dpi=150, bbox_inches='tight')
    plt.show()


# =====================================================
# 실습 9: 상관분석 (Iris 데이터셋)
# =====================================================
print("\n" + "=" * 60)
print("실습 9: 상관분석")
print("=" * 60)

if IRIS_AVAILABLE:
    # 상관행렬 계산
    numeric_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    corr_matrix = iris_df[numeric_cols].corr()

    print("\n[Iris 특성 간 상관행렬]")
    print(corr_matrix.round(3))

    # 상관행렬 히트맵
    fig, ax = plt.subplots(figsize=(8, 6))

    # 짧은 레이블 생성
    short_labels = ['꽃받침길이', '꽃받침너비', '꽃잎길이', '꽃잎너비']

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)

    # 값 표시
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=12)

    plt.colorbar(im)
    ax.set_title('Iris 특성 간 상관행렬')
    plt.tight_layout()
    plt.savefig('correlation_heatmap_iris.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n[높은 상관관계]")
    print("- 꽃잎 길이와 꽃잎 너비: 매우 강한 양의 상관 (r=0.96)")
    print("- 꽃받침 길이와 꽃잎 길이: 강한 양의 상관 (r=0.87)")
    print("- 꽃받침 너비와 꽃잎 길이: 음의 상관 (r=-0.43)")


# =====================================================
# 실습 10: Tips 데이터셋 종합 분석
# =====================================================
print("\n" + "=" * 60)
print("실습 10: Tips 데이터셋 종합 분석")
print("=" * 60)

if SEABORN_AVAILABLE:
    # 팁 비율 계산
    tips['tip_rate'] = tips['tip'] / tips['total_bill']

    print("\n[Tips 데이터셋 기술통계]")
    print(tips[['total_bill', 'tip', 'tip_rate', 'size']].describe().round(3))

    print("\n[요일별 분석]")
    day_stats = tips.groupby('day').agg({
        'total_bill': ['count', 'mean'],
        'tip': 'mean',
        'tip_rate': 'mean'
    }).round(3)
    print(day_stats)

    print("\n[시간대별 분석]")
    time_stats = tips.groupby('time').agg({
        'total_bill': ['count', 'mean'],
        'tip': 'mean',
        'tip_rate': 'mean'
    }).round(3)
    print(time_stats)

    # 종합 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. total_bill 분포
    axes[0, 0].hist(tips['total_bill'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('식사 금액 분포')
    axes[0, 0].set_xlabel('금액 ($)')

    # 2. 요일별 팁 상자그림
    day_order = ['Thur', 'Fri', 'Sat', 'Sun']
    day_data = [tips[tips['day'] == d]['tip'].values for d in day_order]
    axes[0, 1].boxplot(day_data, labels=day_order)
    axes[0, 1].set_title('요일별 팁 분포')
    axes[0, 1].set_ylabel('팁 ($)')

    # 3. 시간대별 평균 비교
    time_means = tips.groupby('time')[['total_bill', 'tip']].mean()
    x = np.arange(len(time_means))
    width = 0.35
    axes[1, 0].bar(x - width/2, time_means['total_bill'], width, label='식사금액')
    axes[1, 0].bar(x + width/2, time_means['tip'] * 5, width, label='팁 x 5')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(time_means.index)
    axes[1, 0].set_title('시간대별 평균 금액')
    axes[1, 0].legend()

    # 4. 팁 비율 분포
    axes[1, 1].hist(tips['tip_rate'] * 100, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(tips['tip_rate'].mean() * 100, color='red', linestyle='--',
                       label=f"평균: {tips['tip_rate'].mean()*100:.1f}%")
    axes[1, 1].set_title('팁 비율 분포')
    axes[1, 1].set_xlabel('팁 비율 (%)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('tips_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


# =====================================================
# 마무리: 핵심 정리
# =====================================================
print("\n" + "=" * 60)
print("5차시 실습 완료!")
print("=" * 60)

print("""
오늘 배운 핵심 내용:

1. 대표값의 의미
   - 평균: np.mean() - 모든 값의 중심
   - 중앙값: np.median() - 이상치에 강건
   - 최빈값: pd.Series().mode() - 가장 빈번한 값

2. 데이터의 퍼짐 정도
   - 범위: np.ptp() - 최대-최소
   - 표준편차: np.std(ddof=1) - 평균에서의 거리
   - IQR: Q3-Q1 - 이상치 탐지에 활용

3. 탐색적 시각화
   - 히스토그램: plt.hist() - 분포 확인
   - 상자그림: plt.boxplot() - 이상치 확인
   - 산점도: plt.scatter() - 변수 간 관계

4. 사용한 실제 데이터셋
   - sklearn: load_iris() - 붓꽃 분류 데이터
   - seaborn: tips - 레스토랑 팁 데이터
   - seaborn: penguins - 펭귄 측정 데이터

핵심 코드 요약:
   df.describe()          # 한 줄로 모든 통계량
   plt.hist(data)         # 분포 확인
   plt.boxplot([a, b, c]) # 그룹 비교
   plt.scatter(x, y)      # 관계 확인

다음 시간: 6차시 - 확률분포와 품질 검정
""")
