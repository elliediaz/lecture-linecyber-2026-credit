# [6차시] 확률분포와 품질검정 기초

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **정규분포**의 개념과 68-95-99.7 규칙을 이해함
2. **Z-score**를 활용하여 이상치를 탐지함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Wine Quality** | UCI ML Repository | 와인 품질 및 화학 성분 분석 |
| **Diamonds** | seaborn | 다이아몬드 가격 분포 분석 |

---

## 강의 구성

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| 1 | 정규분포의 개념과 68-95-99.7 규칙 | 15분 |
| 2 | Z-score를 활용한 이상치 탐지 | 15분 |

---

## 파트 1: 정규분포의 개념과 68-95-99.7 규칙

### 개념 설명

#### 왜 확률분포를 알아야 하는가?

제조 현장에서 발생하는 주요 질문:

| 상황 | 질문 |
|------|------|
| 품질 측정 | "이 측정값이 정상 범위인가?" |
| 라인 비교 | "두 라인의 품질 차이가 실제로 있는가?" |
| 생산 관리 | "오늘 생산량이 이상하게 높은데, 정말 이상한 건가?" |
| 공정 변경 | "새 공정이 불량률을 정말 줄였을까?" |

#### 확률분포란?

데이터가 어떻게 흩어져 있는지 설명하는 함수임.

```
확률분포 = 어떤 값이 나올 확률의 전체 패턴
```

확률분포가 중요한 이유:
- **예측**: 앞으로 어떤 값이 나올지 예상 가능함
- **판단**: 관측값이 정상인지 이상인지 판단함
- **의사결정**: 감이 아닌 데이터 기반 결정이 가능함

#### 대표적인 확률분포

**연속형 확률분포**

| 분포 | 특징 | 예시 |
|------|------|------|
| **정규분포** | 종 모양, 평균 중심 대칭 | 품질 측정값, 생산량 |
| 균등분포 | 모든 값이 동일한 확률 | 무작위 추출 |
| 지수분포 | 사건 간 시간 간격 | 고장 발생 간격 |

**이산형 확률분포**

| 분포 | 특징 | 예시 |
|------|------|------|
| 이항분포 | 성공/실패 반복 | 불량품 개수 |
| 포아송분포 | 희귀 사건 빈도 | 시간당 결함 수 |

#### 정규분포 (Normal Distribution)

가장 중요한 확률분포임.

**특성**:
- 평균(mu)을 중심으로 좌우 대칭인 종 모양
- 평균에서 멀어질수록 확률 감소
- 평균과 표준편차 두 값으로 완전히 정의됨

**표기법**:

```
X ~ N(mu, sigma^2)

mu (뮤) = 평균 = 데이터의 중심
sigma (시그마) = 표준편차 = 데이터의 퍼짐 정도
```

**정규분포의 모양**:

```
                    정규분포 곡선

                        |
                       ***
                      *   *
                     *     *
                    *       *
                   *         *
                  *           *
                ***************
             ----------------------
             mu-3s  mu-2s  mu-s  mu  mu+s  mu+2s  mu+3s
```

#### 68-95-99.7 규칙 (경험적 규칙)

```
       <------- 99.7% (+-3s) ------->
         <----- 95% (+-2s) ----->
            <-- 68% (+-1s) -->

                    ***
                  *******
                ***********
             *******************
          -------------------------
          mu-3s mu-2s mu-s  mu  mu+s mu+2s mu+3s
```

| 범위 | 데이터 비율 | 의미 | 제조 관점 |
|------|------------|------|----------|
| mu +- 1s | **68%** | 대부분 | 일반적인 변동 |
| mu +- 2s | **95%** | 거의 전부 | 허용 범위 |
| mu +- 3s | **99.7%** | 사실상 전부 | 품질 관리 한계 |

**핵심 포인트**: +-3s 밖에 있는 데이터는 0.3%도 안 됨. 이 범위를 벗어나면 이상치로 의심함.

### 실습 코드

#### 데이터 로드 및 기본 설정

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# UCI Wine Quality 데이터셋 로드
WINE_QUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_quality = pd.read_csv(WINE_QUALITY_URL, sep=';')

print(f"데이터 형태: {wine_quality.shape}")
print(f"변수: {list(wine_quality.columns)}")
print(wine_quality.head())
```

#### 정규분포 확인

```python
# 알코올 도수 분포 분석
alcohol = wine_quality['alcohol'].values

print(f"데이터 개수: {len(alcohol)}")
print(f"평균: {alcohol.mean():.2f}%")
print(f"표준편차: {alcohol.std():.2f}%")
print(f"최소값: {alcohol.min():.1f}%")
print(f"최대값: {alcohol.max():.1f}%")

# 정규성 검정 (Shapiro-Wilk test)
sample_size = min(len(alcohol), 500)
stat, p_value = stats.shapiro(alcohol[:sample_size])
print(f"\nShapiro-Wilk 정규성 검정:")
print(f"  통계량: {stat:.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value > 0.05:
    print("  -> 정규분포를 따른다고 볼 수 있음 (p > 0.05)")
else:
    print("  -> 정규분포가 아닐 수 있음 (p <= 0.05)")
```

#### 정규분포 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mean, std = alcohol.mean(), alcohol.std()

# 히스토그램
axes[0].hist(alcohol, bins=30, edgecolor='black', alpha=0.7,
             color='steelblue', density=True)
axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'평균: {mean:.1f}%')
axes[0].axvline(mean - std, color='orange', linestyle=':', linewidth=2, label=f'-1s: {mean-std:.1f}%')
axes[0].axvline(mean + std, color='orange', linestyle=':', linewidth=2, label=f'+1s: {mean+std:.1f}%')
axes[0].set_xlabel('알코올 도수 (%)')
axes[0].set_ylabel('밀도')
axes[0].set_title('와인 알코올 도수 분포')
axes[0].legend()

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
plt.show()
```

#### 68-95-99.7 규칙 검증

```python
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
```

### 결과 해설

- 정규분포는 평균을 중심으로 좌우 대칭인 종 모양 분포임
- 68-95-99.7 규칙은 표준편차를 기준으로 데이터 분포를 파악하는 경험적 규칙임
- Wine Quality 데이터의 알코올 도수는 대체로 정규분포에 가까운 형태를 보임
- 실제 데이터가 이론적 비율과 유사하면 정규분포 가정이 타당함

---

## 파트 2: Z-score를 활용한 이상치 탐지

### 개념 설명

#### 이상치(Outlier)란?

다른 데이터와 동떨어진 값임.

**제조업에서의 의미**:

| 상황 | 이상치 예시 | 원인 가능성 |
|------|------------|------------|
| 생산량 | 평소 1200, 오늘 1500 | 특근, 측정 오류, 기록 실수 |
| 품질 | 평균 500g, 측정 480g | 원료 문제, 설비 이상 |
| 불량률 | 평소 2%, 오늘 8% | 공정 이상, 원자재 불량 |

이상치 탐지가 중요한 이유: 이상치는 문제의 신호일 수 있음. 빠른 발견이 빠른 대응과 손실 최소화로 이어짐.

#### Z-score (표준점수)

이상치 탐지의 핵심 도구임.

**정의**:

```
Z-score = (관측값 - 평균) / 표준편차

Z = (X - mu) / sigma
```

**의미**:
- **Z = 0**: 평균과 같음
- **Z = 1**: 평균에서 1s 떨어짐 (상위 16%)
- **Z = 2**: 평균에서 2s 떨어짐 (상위 2.5%)
- **Z = 3**: 평균에서 3s 떨어짐 (상위 0.15%)

#### Z-score 해석 기준

| Z-score 범위 | 비율 | 해석 | 조치 |
|-------------|------|------|------|
| \|Z\| <= 1 | 68% | 정상 | 문제 없음 |
| 1 < \|Z\| <= 2 | 27% | 주의 | 모니터링 |
| 2 < \|Z\| <= 3 | 4.3% | 경고 | 점검 필요 |
| **\|Z\| > 3** | **0.3%** | **이상치** | **즉시 조사** |

**실무 권장 기준**:
- **\|Z\| > 2**: 이상치 의심 (보수적)
- **\|Z\| > 3**: 이상치 확정 (일반적)

#### IQR vs Z-score 비교

| 비교 항목 | IQR 방식 | Z-score 방식 |
|----------|---------|-------------|
| **계산** | Q3-Q1 기반 | 평균/표준편차 기반 |
| **장점** | 이상치에 강건함 | 해석이 직관적 |
| **단점** | 정규분포 가정 불필요 | 정규분포 가정 필요 |
| **기준** | 1.5 x IQR 밖 | \|Z\| > 2 또는 3 |
| **적합** | 비대칭 분포 | 대칭 분포 |

**권장**: 정규분포에 가까우면 Z-score, 분포 형태가 불확실하면 IQR을 사용함.

### 실습 코드

#### Z-score 계산

```python
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
```

#### scipy를 활용한 Z-score 계산

```python
from scipy import stats

# scipy 방식
z_scores_scipy = stats.zscore(wine_quality['alcohol'])

print("scipy.stats.zscore() 결과 확인:")
print(f"직접 계산 Z-score (처음 5개): {wine_quality['alcohol_zscore'].head().values.round(4)}")
print(f"scipy Z-score (처음 5개): {z_scores_scipy[:5].round(4)}")
```

#### 이상치 탐지 함수

```python
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
outlier_mask, z_scores = detect_outliers_zscore(wine_quality['alcohol'], threshold=2)

print(f"이상치 기준: |Z| > 2")
print(f"전체 데이터 수: {len(wine_quality)}")
print(f"이상치 개수: {outlier_mask.sum()}")
print(f"이상치 비율: {outlier_mask.sum()/len(wine_quality)*100:.2f}%")

# 이상치 상세 정보
outliers = wine_quality[outlier_mask][['alcohol', 'quality']]
print(f"\n이상치 데이터 (처음 10개):")
print(outliers.head(10))
```

#### 이상치 시각화

```python
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
axes[0].legend()

# Z-score 분포
colors_z = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
axes[1].scatter(range(len(z_scores)), z_scores, c=colors_z, alpha=0.5, s=10)
axes[1].axhline(2, color='orange', linestyle='--', linewidth=2, label='+2 기준')
axes[1].axhline(-2, color='orange', linestyle='--', linewidth=2, label='-2 기준')
axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
axes[1].set_xlabel('샘플 인덱스')
axes[1].set_ylabel('Z-score')
axes[1].set_title('Z-score 분포\n(|Z| > 2 = 이상치 의심)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

#### Z-score vs IQR 비교

```python
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
```

#### 품질 관리 종합 리포트 함수

```python
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
    outliers_iqr, _, _ = detect_outliers_iqr(data)
    print(f"    Z-score (|Z|>2): {outliers_z.sum()}개")
    print(f"    IQR 방법: {outliers_iqr.sum()}개")

    # 5. 권고사항
    print(f"\n[5. 권고사항]")
    if outliers_z.sum() > len(data) * 0.05:
        print(f"    - 이상치 비율이 5%를 초과함. 데이터 점검 필요")
    else:
        print(f"    - 이상치 비율이 정상 범위임")

    if p_value < 0.05:
        print(f"    - 정규분포를 따르지 않으므로 비모수 방법 고려")

    print("\n" + "=" * 60)


# 리포트 생성
generate_quality_report(wine_quality, 'alcohol', '와인 알코올 도수')
```

### 결과 해설

- Z-score는 평균에서 몇 표준편차 떨어져 있는지를 나타내는 표준화된 점수임
- \|Z\| > 2는 이상치 의심, \|Z\| > 3은 이상치 확정으로 판단함
- Z-score 방법은 정규분포 가정이 필요하며, IQR 방법은 분포 가정이 없음
- 실무에서는 두 방법을 병행하여 이상치를 교차 검증하는 것이 권장됨
- 이상치 발견 시 무조건 제거하지 말고, 원인을 파악한 후 처리 방법을 결정해야 함

---

## 연습 문제

### 연습 1

Wine Quality 데이터에서 pH의 Z-score를 계산하고 이상치를 찾으시오. (기준: \|Z\| > 2)

```python
# 정답
outliers, z = detect_outliers_zscore(wine_quality['pH'], threshold=2)
print(f"pH 이상치 개수: {outliers.sum()}")
print(f"pH 이상치 값 (처음 5개): {wine_quality[outliers]['pH'].head().values}")
```

### 연습 2

diamonds 데이터에서 carat의 분포를 분석하고, 68-95-99.7 규칙이 적용되는지 확인하시오.

```python
# 정답
import seaborn as sns
diamonds = sns.load_dataset('diamonds')

carat = diamonds['carat']
mean_c, std_c = carat.mean(), carat.std()
within_1 = ((carat >= mean_c - std_c) & (carat <= mean_c + std_c)).sum() / len(carat)
within_2 = ((carat >= mean_c - 2*std_c) & (carat <= mean_c + 2*std_c)).sum() / len(carat)
print(f"캐럿 +-1s: {within_1:.1%} (이론: 68%)")
print(f"캐럿 +-2s: {within_2:.1%} (이론: 95%)")
print("-> 정규분포와 차이가 있음 (오른쪽 꼬리가 긴 분포)")
```

### 연습 3

Wine Quality 데이터에서 quality(품질 등급)별로 알코올 도수의 평균과 표준편차를 비교하시오.

```python
# 정답
quality_stats = wine_quality.groupby('quality')['alcohol'].agg(['mean', 'std', 'count'])
print(quality_stats.round(2))
print("\n-> 품질이 높을수록 알코올 도수가 높은 경향")
```

---

## 핵심 정리

| 구분 | 내용 |
|------|------|
| **정규분포** | 평균 중심의 종 모양 분포, mu와 sigma로 정의됨 |
| **68-95-99.7 규칙** | 1s(68%), 2s(95%), 3s(99.7%) 범위 내 데이터 비율 |
| **Z-score** | Z = (X - mu) / sigma, 평균에서 몇 표준편차 떨어졌는지 |
| **이상치 기준** | \|Z\| > 2 (의심), \|Z\| > 3 (확정) |
| **IQR 방법** | Q1 - 1.5xIQR ~ Q3 + 1.5xIQR 범위 밖 |

---

## 다음 차시 예고

**7차시: 통계 검정 실습**

- t-검정: 두 그룹의 평균 비교
- 카이제곱 검정: 범주형 데이터 분석
- ANOVA: 3개 이상 그룹 비교
