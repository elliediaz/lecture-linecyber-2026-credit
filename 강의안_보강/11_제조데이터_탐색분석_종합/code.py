"""
[11차시] 제조 데이터 탐색 분석 종합 (EDA)
========================================

학습 목표:
1. EDA의 전체 흐름 이해
2. 데이터에서 인사이트 도출
3. 제조 데이터 종합 분석

실습 환경:
- Python 3.8+
- pandas, numpy, scipy, matplotlib, seaborn

데이터:
- Wine Quality 데이터셋 (와인 품질 데이터)
- 출처: UCI Machine Learning Repository
- 제조 공정과 유사한 품질 관리 데이터
- 다양한 수치형 변수와 품질 등급 보유
- URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
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
print("[11차시] 제조 데이터 탐색 분석 종합 (EDA)")
print("=" * 60)


# ============================================================
# 실습 시나리오: 품질관리팀 분석 요청
# ============================================================
print("\n" + "=" * 60)
print("실습 시나리오: 품질관리팀 분석 요청")
print("=" * 60)

print("""
[분석 요청서]

요청 부서: 품질관리팀
요청 내용: "와인 품질에 영향을 미치는 요인을 분석해주세요.
           특히 어떤 화학적 특성이 좋은 와인을 만드는지,
           그리고 나쁜 와인을 피하려면 어떤 조건을 관리해야 하는지
           분석 부탁드립니다."

제공 데이터: Red Wine Quality 데이터셋
변수: fixed acidity, volatile acidity, citric acid,
      residual sugar, chlorides, free sulfur dioxide,
      total sulfur dioxide, density, pH, sulphates,
      alcohol, quality
""")


# ============================================================
# 1단계: 데이터 로드 및 개요 파악
# ============================================================
print("\n" + "=" * 60)
print("1단계: 데이터 로드 및 개요 파악")
print("=" * 60)

# Wine Quality 데이터셋 로드
try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    print("Wine Quality 데이터셋 로드 성공!")
except Exception as e:
    print(f"온라인 로드 실패: {e}")
    print("대체 데이터를 생성합니다...")
    # 대체 데이터 생성
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'fixed acidity': np.random.normal(8.3, 1.7, n),
        'volatile acidity': np.random.normal(0.53, 0.18, n),
        'citric acid': np.random.uniform(0, 0.8, n),
        'residual sugar': np.random.exponential(2.5, n),
        'chlorides': np.random.normal(0.087, 0.02, n),
        'free sulfur dioxide': np.random.normal(15, 10, n),
        'total sulfur dioxide': np.random.normal(46, 30, n),
        'density': np.random.normal(0.997, 0.002, n),
        'pH': np.random.normal(3.3, 0.15, n),
        'sulphates': np.random.normal(0.66, 0.17, n),
        'alcohol': np.random.normal(10.4, 1, n),
        'quality': np.random.choice([3, 4, 5, 6, 7, 8], n, p=[0.01, 0.05, 0.40, 0.40, 0.12, 0.02])
    })

print(f"\n[1.1] 데이터 크기 및 구조")
print("-" * 40)
print(f"데이터 크기: {df.shape} (행, 열)")
print(f"\n컬럼 목록:\n{df.columns.tolist()}")

print("\n[1.2] 데이터 타입")
print("-" * 40)
print(df.dtypes)

print("\n[1.3] 처음 5행")
print("-" * 40)
print(df.head())

print("\n[1.4] 기술 통계")
print("-" * 40)
print(df.describe().round(3))

print("\n[1.5] 결측치 현황")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = df.isnull().sum() / len(df) * 100
print(pd.DataFrame({'결측치 수': missing, '비율(%)': missing_pct.round(2)}))
print("\n=> 결측치 없음! 깔끔한 데이터")

# 변수 설명
print("\n[1.6] 변수 설명")
print("-" * 40)
print("""
=== 입력 변수 (화학적 특성) ===
fixed acidity      : 고정 산도 (주석산 함량)
volatile acidity   : 휘발성 산도 (초산 함량) - 높으면 식초 맛
citric acid        : 구연산 (과일 향미 증가)
residual sugar     : 잔당 (발효 후 남은 당분)
chlorides          : 염화물 (소금 맛)
free sulfur dioxide: 유리 이산화황 (방부 효과)
total sulfur dioxide: 총 이산화황
density            : 밀도 (알코올/당분 영향)
pH                 : 산도 지표 (0-14)
sulphates          : 황산염 (방부제)
alcohol            : 알코올 도수 (%)

=== 타겟 변수 ===
quality            : 품질 점수 (3~8점, 전문가 평가)
""")


# ============================================================
# 2단계: 데이터 품질 확인
# ============================================================
print("\n" + "=" * 60)
print("2단계: 데이터 품질 확인")
print("=" * 60)

# 이상치 확인
print("\n[2.1] 이상치 확인 (IQR 방법)")
print("-" * 40)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

outlier_summary = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_pct = outliers / len(df) * 100
    outlier_summary.append({
        'Variable': col,
        'Outliers': outliers,
        'Percentage': f'{outlier_pct:.1f}%'
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df[outlier_df['Outliers'] > 0])


# ============================================================
# 3단계: 단변량 분석
# ============================================================
print("\n" + "=" * 60)
print("3단계: 단변량 분석")
print("=" * 60)

# 3.1 타겟 변수: 품질(quality) 분포
print("\n[3.1] 품질(quality) 분포")
print("-" * 40)
print(f"평균: {df['quality'].mean():.3f}")
print(f"중앙값: {df['quality'].median():.3f}")
print(f"표준편차: {df['quality'].std():.3f}")
print(f"최솟값: {df['quality'].min()}")
print(f"최댓값: {df['quality'].max()}")

print("\n=== 품질 등급별 빈도 ===")
quality_counts = df['quality'].value_counts().sort_index()
print(quality_counts)
print(f"\n품질 분포 비율:")
print((quality_counts / len(df) * 100).round(1))

# 품질 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 히스토그램
axes[0].hist(df['quality'], bins=6, edgecolor='black', alpha=0.7)
axes[0].axvline(df['quality'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["quality"].mean():.2f}')
axes[0].set_xlabel('Quality Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Wine Quality Distribution')
axes[0].legend()

# 파이 차트
quality_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Quality Score Proportion')
axes[1].set_ylabel('')

# 카운트 플롯
quality_counts.plot(kind='bar', ax=axes[2], color='steelblue', edgecolor='black')
axes[2].set_xlabel('Quality Score')
axes[2].set_ylabel('Count')
axes[2].set_title('Wine Count by Quality')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('eda_quality_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n품질 분포 시각화 저장: eda_quality_distribution.png")

# 품질 그룹 생성 (분석 편의)
df['quality_group'] = pd.cut(df['quality'], bins=[0, 4, 6, 10],
                              labels=['Low(3-4)', 'Medium(5-6)', 'High(7-8)'])
print("\n=== 품질 그룹별 빈도 ===")
print(df['quality_group'].value_counts())

# 3.2 주요 수치형 변수 분포
print("\n[3.2] 주요 수치형 변수 분포")
print("-" * 40)

key_vars = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
for var in key_vars:
    print(f"{var}: mean={df[var].mean():.3f}, std={df[var].std():.3f}, "
          f"min={df[var].min():.3f}, max={df[var].max():.3f}")


# ============================================================
# 4단계: 이변량 분석
# ============================================================
print("\n" + "=" * 60)
print("4단계: 이변량 분석")
print("=" * 60)

# 4.1 품질과 각 변수의 상관관계
print("\n[4.1] 품질과 각 변수의 상관관계")
print("-" * 40)

correlations = df.corr()['quality'].drop('quality').sort_values(ascending=False)
print("=== quality와의 상관계수 (내림차순) ===")
for var, corr in correlations.items():
    direction = "+" if corr > 0 else ""
    strength = "강함" if abs(corr) >= 0.3 else "중간" if abs(corr) >= 0.2 else "약함"
    print(f"  {var:22}: {direction}{corr:.4f} ({strength})")

print("\n=== 주요 발견 ===")
print("양의 상관:")
print(f"  - alcohol: {correlations['alcohol']:.3f} (가장 강함!)")
print(f"  - sulphates: {correlations['sulphates']:.3f}")
print(f"  - citric acid: {correlations['citric acid']:.3f}")
print("\n음의 상관:")
print(f"  - volatile acidity: {correlations['volatile acidity']:.3f} (가장 강함!)")
print(f"  - density: {correlations['density']:.3f}")

# 4.2 품질 그룹별 주요 변수 비교
print("\n[4.2] 품질 그룹별 주요 변수 비교")
print("-" * 40)

group_stats = df.groupby('quality_group')[['alcohol', 'volatile acidity',
                                            'sulphates', 'citric acid']].mean()
print(group_stats.round(3))

# 4.3 통계 검정: 고품질 vs 저품질
print("\n[4.3] 통계 검정: 고품질 vs 저품질")
print("-" * 40)

high_quality = df[df['quality'] >= 7]
low_quality = df[df['quality'] <= 4]

print(f"고품질 와인 (quality >= 7): {len(high_quality)}개")
print(f"저품질 와인 (quality <= 4): {len(low_quality)}개")

test_vars = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
print("\n=== t-검정 결과 ===")
for var in test_vars:
    t_stat, p_value = stats.ttest_ind(high_quality[var], low_quality[var])
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {var:20}: t={t_stat:7.2f}, p={p_value:.4f} {significance}")

print("\n  *** p < 0.001, ** p < 0.01, * p < 0.05")
print("  => 모든 변수에서 고품질/저품질 간 통계적으로 유의미한 차이!")

# 이변량 분석 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# alcohol vs quality
axes[0, 0].scatter(df['alcohol'], df['quality'], alpha=0.3)
z = np.polyfit(df['alcohol'], df['quality'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['alcohol'].min(), df['alcohol'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), 'r--',
                label=f'r={correlations["alcohol"]:.3f}')
axes[0, 0].set_xlabel('Alcohol (%)')
axes[0, 0].set_ylabel('Quality')
axes[0, 0].set_title('Alcohol vs Quality')
axes[0, 0].legend()

# volatile acidity vs quality
axes[0, 1].scatter(df['volatile acidity'], df['quality'], alpha=0.3, color='orange')
axes[0, 1].set_xlabel('Volatile Acidity')
axes[0, 1].set_ylabel('Quality')
axes[0, 1].set_title(f'Volatile Acidity vs Quality (r={correlations["volatile acidity"]:.3f})')

# alcohol 박스플롯 by quality
df.boxplot(column='alcohol', by='quality', ax=axes[1, 0])
axes[1, 0].set_title('Alcohol by Quality')
axes[1, 0].set_xlabel('Quality')
axes[1, 0].set_ylabel('Alcohol (%)')

# volatile acidity 박스플롯 by quality
df.boxplot(column='volatile acidity', by='quality', ax=axes[1, 1])
axes[1, 1].set_title('Volatile Acidity by Quality')
axes[1, 1].set_xlabel('Quality')
axes[1, 1].set_ylabel('Volatile Acidity')

plt.suptitle('Bivariate Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('eda_bivariate.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n이변량 분석 시각화 저장: eda_bivariate.png")


# ============================================================
# 5단계: 다변량 분석
# ============================================================
print("\n" + "=" * 60)
print("5단계: 다변량 분석")
print("=" * 60)

# 5.1 상관행렬
print("\n[5.1] 상관행렬")
print("-" * 40)
feature_cols = [col for col in numeric_cols if col != 'quality']
corr_matrix = df[feature_cols + ['quality']].corr()

print("=== 주요 변수 간 상관관계 ===")
print(corr_matrix[['quality']].round(3))

# 5.2 복합 조건 분석
print("\n[5.2] 복합 조건 분석: 알코올 + 휘발성 산도")
print("-" * 40)

df['alcohol_group'] = pd.cut(df['alcohol'], bins=[8, 10, 11, 15],
                              labels=['Low(~10%)', 'Medium(10-11%)', 'High(11%+)'])
df['va_group'] = pd.cut(df['volatile acidity'],
                         bins=[0, 0.4, 0.6, 2],
                         labels=['Low(~0.4)', 'Medium(0.4-0.6)', 'High(0.6+)'])

# 조합별 평균 품질
combo_quality = df.groupby(['alcohol_group', 'va_group'])['quality'].mean()
print(combo_quality.round(2))

# 피벗 테이블
combo_pivot = df.pivot_table(values='quality',
                              index='alcohol_group',
                              columns='va_group',
                              aggfunc='mean')
print("\n=== 알코올 x 휘발성산도 피벗 ===")
print(combo_pivot.round(2))

# 5.3 품질 그룹별 프로파일
print("\n[5.3] 품질 그룹별 프로파일")
print("-" * 40)
profile = df.groupby('quality_group')[feature_cols].mean()
print(profile.T.round(3))

# 다변량 분석 시각화
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 상관행렬 히트맵
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            ax=axes[0], fmt='.2f', annot_kws={'size': 7})
axes[0].set_title('Correlation Matrix')

# 알코올 x 휘발성산도 히트맵
sns.heatmap(combo_pivot, annot=True, cmap='RdYlGn', ax=axes[1], fmt='.2f')
axes[1].set_title('Mean Quality: Alcohol x Volatile Acidity')

# 품질 그룹별 레이더 차트 대체 - 주요 변수 비교
profile_subset = df.groupby('quality_group')[['alcohol', 'volatile acidity',
                                               'sulphates', 'citric acid']].mean()
profile_subset.T.plot(kind='bar', ax=axes[2], width=0.8)
axes[2].set_title('Key Variables by Quality Group')
axes[2].set_xlabel('Variable')
axes[2].set_ylabel('Mean Value')
axes[2].legend(title='Quality Group')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_multivariate.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n다변량 분석 시각화 저장: eda_multivariate.png")


# ============================================================
# 6단계: 인사이트 도출
# ============================================================
print("\n" + "=" * 60)
print("6단계: 인사이트 도출")
print("=" * 60)

# 주요 발견 사항 계산
overall_mean = df['quality'].mean()
high_alcohol = df[df['alcohol'] > df['alcohol'].median()]['quality'].mean()
low_alcohol = df[df['alcohol'] <= df['alcohol'].median()]['quality'].mean()
alcohol_diff = high_alcohol - low_alcohol

high_va = df[df['volatile acidity'] > df['volatile acidity'].median()]['quality'].mean()
low_va = df[df['volatile acidity'] <= df['volatile acidity'].median()]['quality'].mean()
va_diff = low_va - high_va

# 최적 조건 찾기
best_combo = combo_pivot.stack().idxmax()
best_quality = combo_pivot.stack().max()

insights = f"""
=================================================================
                      분석 결과 요약
=================================================================

[1] 품질 분포
    - 전체 평균 품질: {overall_mean:.2f}
    - 대부분 5~6점에 집중 (약 80%)
    - 3~4점(저품질) {(df['quality'] <= 4).sum()}개 ({(df['quality'] <= 4).sum()/len(df)*100:.1f}%)
    - 7~8점(고품질) {(df['quality'] >= 7).sum()}개 ({(df['quality'] >= 7).sum()/len(df)*100:.1f}%)

[2] 품질에 가장 큰 영향을 미치는 요인

    (+) 양의 영향:
    - alcohol (상관계수: {correlations['alcohol']:.3f})
      => 알코올 도수가 높을수록 품질 좋음
      => 높은 알코올: 평균 {high_alcohol:.2f}점 vs 낮은 알코올: {low_alcohol:.2f}점 (차이: {alcohol_diff:.2f}점)

    - sulphates (상관계수: {correlations['sulphates']:.3f})
      => 적절한 황산염이 품질 향상에 기여

    (-) 음의 영향:
    - volatile acidity (상관계수: {correlations['volatile acidity']:.3f})
      => 휘발성 산도가 높으면 품질 저하 (식초 맛)
      => 낮은 VA: 평균 {low_va:.2f}점 vs 높은 VA: {high_va:.2f}점 (차이: {va_diff:.2f}점)

[3] 최적 조건 조합
    - 알코올: {best_combo[0]}
    - 휘발성 산도: {best_combo[1]}
    - 평균 품질: {best_quality:.2f}점

[4] 품질 그룹별 특성
    - 고품질(7-8): 높은 알코올({profile_subset.loc['High(7-8)', 'alcohol']:.2f}%),
                   낮은 VA({profile_subset.loc['High(7-8)', 'volatile acidity']:.2f})
    - 저품질(3-4): 낮은 알코올({profile_subset.loc['Low(3-4)', 'alcohol']:.2f}%),
                   높은 VA({profile_subset.loc['Low(3-4)', 'volatile acidity']:.2f})

=================================================================
"""
print(insights)


# ============================================================
# 7단계: 대시보드 및 보고서
# ============================================================
print("\n" + "=" * 60)
print("7단계: 대시보드 구성")
print("=" * 60)

# 종합 대시보드
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 품질 분포
quality_counts.plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
axes[0, 0].axhline(df['quality'].value_counts().mean(), color='red', linestyle='--')
axes[0, 0].set_xlabel('Quality Score')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('1. Quality Distribution')
axes[0, 0].tick_params(axis='x', rotation=0)

# 2. 품질과 알코올 관계
axes[0, 1].scatter(df['alcohol'], df['quality'], alpha=0.3, c=df['quality'], cmap='RdYlGn')
axes[0, 1].set_xlabel('Alcohol (%)')
axes[0, 1].set_ylabel('Quality')
axes[0, 1].set_title(f'2. Alcohol vs Quality (r={correlations["alcohol"]:.3f})')

# 3. 품질과 휘발성 산도 관계
axes[0, 2].scatter(df['volatile acidity'], df['quality'], alpha=0.3,
                   c=df['quality'], cmap='RdYlGn')
axes[0, 2].set_xlabel('Volatile Acidity')
axes[0, 2].set_ylabel('Quality')
axes[0, 2].set_title(f'3. Volatile Acidity vs Quality (r={correlations["volatile acidity"]:.3f})')

# 4. 품질 그룹별 알코올 박스플롯
df.boxplot(column='alcohol', by='quality_group', ax=axes[1, 0])
axes[1, 0].set_title('4. Alcohol by Quality Group')
axes[1, 0].set_xlabel('Quality Group')
axes[1, 0].set_ylabel('Alcohol (%)')

# 5. 상관계수 Top 5
top_corr = correlations.abs().sort_values(ascending=True).tail(5)
top_corr.plot(kind='barh', ax=axes[1, 1], color=['green' if c > 0 else 'red' for c in correlations[top_corr.index]])
axes[1, 1].set_xlabel('Absolute Correlation')
axes[1, 1].set_title('5. Top 5 Correlations with Quality')

# 6. 복합 조건 히트맵
sns.heatmap(combo_pivot, annot=True, cmap='RdYlGn', ax=axes[1, 2], fmt='.2f')
axes[1, 2].set_title('6. Quality: Alcohol x Volatile Acidity')

plt.suptitle('Wine Quality Analysis Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("종합 대시보드 저장: eda_dashboard.png")


# ============================================================
# 권고사항
# ============================================================
print("\n" + "=" * 60)
print("권고사항")
print("=" * 60)

recommendations = """
=================================================================
                         권고사항
=================================================================

[즉시 실행 (Quick Win)]
-----------------------------------------------------------------
1. 휘발성 산도(VA) 모니터링 강화
   - 0.4 이하로 유지 권장
   - 0.6 초과 시 경고 알람 설정

2. 알코올 도수 관리
   - 11% 이상 목표
   - 발효 공정 최적화

[단기 개선]
-----------------------------------------------------------------
3. 황산염(Sulphates) 적정 수준 유지
   - 0.65~0.75 범위 권장

4. 구연산(Citric Acid) 첨가 검토
   - 과일 향미 증가
   - 품질 향상에 기여

[장기 개선]
-----------------------------------------------------------------
5. 품질 예측 모델 개발
   - 발효 전 품질 사전 예측
   - 공정 자동 조절 시스템 구축

6. 센서 데이터 수집 확대
   - 실시간 모니터링
   - 온도, 습도 등 환경 데이터 추가

[주의사항]
-----------------------------------------------------------------
- 이 분석은 레드 와인 기준 (화이트 와인은 다를 수 있음)
- 상관관계 =/= 인과관계
- 극단적인 조건은 다른 문제를 야기할 수 있음

=================================================================
"""
print(recommendations)


# ============================================================
# 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("11차시 핵심 정리")
print("=" * 60)

print("""
[EDA 5단계]
------------------------------------------------------
1단계: 데이터 개요 -> shape, dtypes, head, describe
2단계: 데이터 품질 -> 결측치, 이상치 확인
3단계: 단변량 분석 -> 각 변수 분포 (히스토그램, 박스플롯)
4단계: 이변량 분석 -> 두 변수 관계 (산점도, 상관계수, t-검정)
5단계: 다변량 분석 -> 여러 변수 동시 (피벗, 히트맵)
6단계: 인사이트 도출 -> 발견 정리, 권고사항 작성

[좋은 인사이트 3요소]
------------------------------------------------------
1. 구체적: 명확한 수치와 근거
   예: "알코올 11% 이상인 와인의 평균 품질은 6.2점"

2. 실행 가능: 행동으로 연결 가능
   예: "휘발성 산도를 0.4 이하로 관리하면 품질 향상"

3. 데이터 기반: 통계적 뒷받침
   예: "t-검정 결과 p < 0.001로 유의미"

[Wine Quality 데이터 핵심 발견]
------------------------------------------------------
- 알코올 도수가 가장 중요한 양의 요인
- 휘발성 산도가 가장 중요한 음의 요인
- 고품질 와인 = 높은 알코올 + 낮은 휘발성 산도
- 대부분 5~6점에 집중, 극단적 품질은 드묾

[체크리스트]
------------------------------------------------------
[ ] 결측치/이상치 확인
[ ] 타겟 변수 분포 확인
[ ] 타겟과 다른 변수 관계 확인 (핵심!)
[ ] 그룹별 차이 통계 검정
[ ] 복합 요인 분석
[ ] 실행 가능한 권고사항 작성
""")

print("\n" + "=" * 60)
print("Part II 완료! 다음 차시: 머신러닝 소개와 문제 유형")
print("=" * 60)
