# [9차시] 제조 데이터 전처리 (1)

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **결측치**를 탐지하고 적절히 처리함
2. **이상치**를 탐지하는 방법을 적용함
3. 상황에 맞는 **전처리 전략**을 선택함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Titanic** | seaborn | 결측치 처리 학습 (실제 결측치 포함) |

주요 변수:
- survived: 생존 여부 (0=사망, 1=생존)
- pclass: 좌석 등급 (1, 2, 3등급)
- age: 나이 (결측치 존재)
- fare: 요금
- embarked: 탑승 항구 (S, C, Q)
- deck: 갑판 (결측치 다수)

---

## 강의 구성

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| 1 | 결측치 탐지 및 처리 | 10분 |
| 2 | 이상치 탐지 방법 적용 | 10분 |
| 3 | 상황에 맞는 전처리 전략 선택 | 10분 |

---

## 파트 1: 결측치 탐지 및 처리

### 개념 설명

#### 왜 전처리가 중요한가?

데이터 분석의 현실:

```
전체 프로젝트 시간 배분:

+------------------------------------------+
| 데이터 수집/전처리    ############## 60~80% |
| 모델링              #### 10~20%          |
| 평가/배포           #### 10~20%          |
+------------------------------------------+
```

**"Garbage In, Garbage Out"**: 나쁜 데이터는 나쁜 모델을 만듦.

#### 전처리의 목표

분석에 적합한 데이터로 변환하는 것임.

| 문제 | 해결 | 방법 |
|------|------|------|
| **결측치** | 비어있는 값 처리 | 삭제, 대체 |
| **이상치** | 극단값 처리 | 탐지, 클리핑, 대체 |
| **불균형** | 클래스 비율 조정 | 오버/언더 샘플링 |
| **스케일** | 단위 통일 | 표준화, 정규화 |
| **형식** | 자료형 변환 | 인코딩, 변환 |

#### 결측치 (Missing Values)

비어있는 값임.

**제조 현장에서 발생 원인**:

| 원인 | 설명 | 예시 |
|------|------|------|
| **센서 오류** | 통신 두절, 배터리 방전 | 온도 센서 끊김 |
| **입력 누락** | 수기 입력 시 실수 | 불량수 미기록 |
| **시스템 장애** | PLC, MES 연결 끊김 | 로그 누락 |
| **정상 결측** | 특정 조건에서만 측정 | 비가동 시간 |

#### pandas에서 결측치 표현

```python
import numpy as np
import pandas as pd

# 결측치 표현 방식
np.nan       # NumPy의 Not a Number
None         # Python의 None
pd.NA        # pandas의 NA (권장)

# DataFrame에서의 표현
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [None, 2, 3, 4],
    'C': [1, pd.NA, 3, 4]
})
```

#### 결측치 처리 가이드

| 결측 비율 | 권장 방법 | 비고 |
|----------|----------|------|
| < 5% | 삭제 또는 평균/중앙값 대체 | 영향 미미 |
| 5~15% | 중앙값, 그룹별 평균 | 대체 방법 비교 |
| 15~30% | 보간, 예측 모델 대체 | 신중한 검토 |
| > 30% | 열 삭제 고려 | 정보 손실 감수 |

핵심 원칙:
1. 결측 패턴 먼저 파악 (무작위인지 체계적인지)
2. 도메인 지식 활용
3. 대체 전후 분포 비교

### 실습 코드

#### 데이터 로드

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Titanic 데이터셋 로드
df = sns.load_dataset('titanic')

print(f"데이터 형태: {df.shape}")
print(f"\n처음 10행:\n{df.head(10)}")

print("\n=== 변수 설명 ===")
print("survived: 생존 여부 (0=사망, 1=생존)")
print("pclass: 좌석 등급 (1=1등급, 2=2등급, 3=3등급)")
print("sex: 성별")
print("age: 나이 (결측치 존재!)")
print("fare: 요금")
print("embarked: 탑승 항구 (S=Southampton, C=Cherbourg, Q=Queenstown)")
print("deck: 갑판 (A~G, 결측치 다수!)")
```

#### 결측치 탐지

```python
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
```

#### 결측치 시각화

```python
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
plt.show()

print("\n=== 결측치 해석 ===")
print("- age: 약 20%의 결측치 -> 중앙값 대체 추천")
print("- deck: 약 70%+ 결측치 -> 삭제 또는 'Unknown' 대체 고려")
print("- embarked: 소수의 결측치 -> 최빈값 대체 추천")
```

#### 결측치 처리 - 삭제 (dropna)

```python
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
```

#### 결측치 처리 - 대체 (fillna)

```python
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
```

#### 대체 전략 비교

```python
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
```

### 결과 해설

- 결측치 탐지는 isnull().sum()으로 간단히 확인 가능함
- 결측 비율에 따라 삭제/대체 전략을 선택함
- 수치형은 중앙값, 범주형은 최빈값 대체가 일반적임
- 결측치가 30% 이상인 열은 삭제를 고려함
- 대체 전후 분포 비교로 왜곡 여부를 확인해야 함

---

## 파트 2: 이상치 탐지 방법 적용

### 개념 설명

#### 이상치 (Outliers)

비정상적으로 극단적인 값임.

**제조 현장에서 발생 원인**:

| 원인 | 설명 | 예시 |
|------|------|------|
| **측정 오류** | 센서 고장, 캘리브레이션 | 온도 -999 |
| **입력 실수** | 단위 오류, 오타 | 생산량 12000 (1200) |
| **실제 극단값** | 공정 이탈, 설비 고장 | 불량률 급등 |
| **시스템 이상** | 리셋 값, 오버플로우 | 0 또는 999999 |

#### 이상치의 두 얼굴

분석 목적에 따라 처리 방법이 다름:

```
불량품 예측 모델:
  -> 이상치가 핵심 정보! (제거하면 안 됨)

평균 생산량 추정:
  -> 이상치 제거 고려 (왜곡 방지)

이상 탐지 시스템:
  -> 이상치를 찾는 게 목적
```

**핵심 질문**: 이상치가 **오류**인가, **신호**인가?

#### IQR 방법

사분위수 범위(Interquartile Range)를 사용함.

```
Q1 = 25% 백분위수
Q3 = 75% 백분위수
IQR = Q3 - Q1

하한 = Q1 - 1.5 x IQR
상한 = Q3 + 1.5 x IQR

이상치: 하한 미만 또는 상한 초과
```

**시각화 (상자그림)**:

```
                    이상치 (Q3 + 1.5xIQR 초과)
                          *
                          |
         +----------------+------------------+ <- 상한
         |                                    |
    +----+------------------------------------+----+
    |    |               Q3                   |    |
    |    +------------------------------------+    |
    |    |            중앙값                   |    |
    |    +------------------------------------+    |
    |    |               Q1                   |    |
    +----+------------------------------------+----+
         |                                    |
         +----------------+------------------+ <- 하한
                          |
                          *
                    이상치 (Q1 - 1.5xIQR 미만)
```

#### Z-score 방법

표준점수를 사용함.

```
Z = (X - mean) / std

|Z| > 2: 이상치 의심 (4.6%)
|Z| > 3: 이상치 확정 (0.3%)
```

#### IQR vs Z-score 비교

| 비교 항목 | IQR | Z-score |
|----------|-----|---------|
| **분포 가정** | 없음 | 정규분포 가정 |
| **이상치 영향** | 강건함 | 영향 받음 |
| **기준** | 1.5 x IQR | 보통 2~3 sigma |
| **적합 상황** | 비대칭 분포 | 대칭 분포 |
| **시각화** | 상자그림 | 히스토그램 |

**권장**: 정규분포에 가까우면 Z-score, 분포 형태가 불확실하면 IQR을 사용함.

### 실습 코드

#### 이상치 탐지 함수 정의

```python
# 결측치가 처리된 데이터 사용
df_clean = df_filled.copy()
numeric_cols_analysis = ['age', 'fare']


def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    IQR 방법으로 이상치 탐지

    Parameters:
        data: DataFrame
        column: 분석할 컬럼명
        multiplier: IQR 배수 (기본 1.5)

    Returns:
        outlier_mask: 이상치 위치 (True/False)
        lower: 하한
        upper: 상한
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    outlier_mask = (data[column] < lower) | (data[column] > upper)

    return outlier_mask, lower, upper


def detect_outliers_zscore(data, column, threshold=3):
    """
    Z-score 방법으로 이상치 탐지

    Parameters:
        data: DataFrame
        column: 분석할 컬럼명
        threshold: Z-score 임계값 (기본 3)

    Returns:
        outlier_mask: 이상치 위치 (True/False)
        z_scores: Z-score 값들
    """
    z_scores = stats.zscore(data[column])
    outlier_mask = np.abs(z_scores) > threshold

    return outlier_mask, z_scores
```

#### 기술 통계로 이상치 확인

```python
print(df_clean[numeric_cols_analysis].describe())

print("\n=== 최솟값/최댓값 확인 ===")
for col in numeric_cols_analysis:
    print(f"{col}: min={df_clean[col].min():.2f}, max={df_clean[col].max():.2f}")

print("\n=== 해석 ===")
print("- fare: 최댓값 512.33은 매우 높음 (1등급 특실 가격)")
print("- age: 범위 0.42~80 (영아부터 노인까지)")
```

#### IQR 방법으로 이상치 탐지

```python
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
```

#### Z-score 방법으로 이상치 탐지

```python
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
```

#### IQR vs Z-score 비교

```python
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
```

#### 이상치 시각화

```python
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

# fare 히스토그램
ax4 = axes[1, 1]
ax4.hist(df_clean['fare'], bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=fare_upper, color='r', linestyle='--', label=f'Upper: {fare_upper:.1f}')
ax4.set_title('Fare - Histogram (Skewed Distribution)')
ax4.set_xlabel('Fare')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
plt.show()
```

### 결과 해설

- IQR 방법은 분포 가정이 없어 비대칭 분포에 강건함
- Z-score 방법은 정규분포 가정이 필요하며, 대칭 분포에 적합함
- fare처럼 오른쪽으로 치우친 분포는 IQR 방법이 더 적합함
- 두 방법을 병행하여 교차 검증하는 것이 권장됨
- 시각화를 통해 이상치 분포를 직관적으로 파악할 수 있음

---

## 파트 3: 상황에 맞는 전처리 전략 선택

### 개념 설명

#### 이상치 처리 전략

**전략 1: 삭제**

```python
# 이상치가 아닌 데이터만 선택
df_clean = df[~outliers]

# 또는 이상치 행 삭제
df_clean = df.drop(df[outliers].index)
```

사용 시점:
- 이상치가 명백한 오류일 때
- 이상치 비율이 낮을 때 (< 5%)
- 이상치가 분석에 중요하지 않을 때

**전략 2: 클리핑**

```python
# IQR 기준 클리핑
df['생산량'] = df['생산량'].clip(lower, upper)

# 특정 값으로 클리핑
df['온도'] = df['온도'].clip(lower=60, upper=100)

# 백분위수 기준
lower_5 = df['값'].quantile(0.05)
upper_95 = df['값'].quantile(0.95)
df['값'] = df['값'].clip(lower_5, upper_95)
```

사용 시점:
- 극단값은 줄이되 데이터는 보존하고 싶을 때
- 예측 모델 학습 시 극단값 영향 완화

**전략 3: 대체**

```python
# 중앙값으로 대체
median_val = df['생산량'].median()
df.loc[outliers, '생산량'] = median_val

# 그룹별 중앙값으로 대체
df['생산량'] = df.groupby('라인')['생산량'].transform(
    lambda x: x.where(~outliers_iqr, x.median())
)
```

사용 시점:
- 이상치가 측정 오류로 확인될 때
- 데이터 크기를 유지해야 할 때

**전략 4: 플래그**

```python
# 이상치 플래그 열 추가
df['이상치_여부'] = outliers_iqr

# 이상치 유형 기록
df['이상치_유형'] = 'normal'
df.loc[df['생산량'] > upper, '이상치_유형'] = 'high'
df.loc[df['생산량'] < lower, '이상치_유형'] = 'low'
```

사용 시점:
- 이상치 분석이 필요할 때
- 원본 정보를 보존하면서 처리할 때

#### 이상치 처리 결정 프로세스

```
                  이상치 발견
                      |
                      v
            +------------------+
            |    원인 조사     |
            +--------+---------+
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
   측정 오류      실제 극단값     불확실
       |             |             |
       v             v             v
    삭제/대체      플래그       전문가 상의
```

#### 상황별 전처리 전략

| 분석 목적 | 결측치 처리 | 이상치 처리 |
|----------|------------|------------|
| **평균 추정** | 중앙값 대체 | 클리핑 또는 삭제 |
| **예측 모델** | 평균/보간 대체 | 클리핑 + 플래그 |
| **불량 예측** | 삭제 또는 대체 | 유지 (중요 신호) |
| **이상 탐지** | 플래그 표시 | 유지 (찾는 대상) |
| **탐색 분석** | 플래그 표시 | 분리 분석 |

#### 데이터 누출 (Data Leakage) 주의

```python
# 잘못된 방법 (데이터 누출)
df['온도'].fillna(df['온도'].mean(), inplace=True)  # 전체 평균
train, test = train_test_split(df)

# 올바른 방법
train, test = train_test_split(df)
train_mean = train['온도'].mean()  # 학습 데이터 평균만 사용
train['온도'].fillna(train_mean, inplace=True)
test['온도'].fillna(train_mean, inplace=True)  # 같은 값 사용
```

테스트 데이터 정보가 학습에 사용되면 안 됨!

### 실습 코드

#### 이상치 처리 - 제거

```python
df_remove = df_clean.copy()

# fare 이상치만 제거 (age는 유지 - 실제 고령자 데이터)
outlier_mask_fare = fare_outliers

print(f"fare 이상치 행: {outlier_mask_fare.sum()}개")

df_removed = df_remove[~outlier_mask_fare]
print(f"제거 후 행 수: {len(df_clean)} -> {len(df_removed)}")
print(f"제거된 행: {len(df_clean) - len(df_removed)}개")
```

#### 이상치 처리 - 클리핑

```python
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
```

#### 이상치 처리 - 플래그 추가

```python
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
```

#### 전처리 전후 비교

```python
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
```

#### 전처리 전후 시각화

```python
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
plt.show()
```

#### 전처리 파이프라인 함수

```python
def preprocess_titanic_data(df, numeric_cols, categorical_cols=None,
                            missing_strategy='median',
                            outlier_method='iqr',
                            outlier_treatment='clip',
                            iqr_multiplier=1.5,
                            zscore_threshold=3):
    """
    Titanic 데이터 전처리 파이프라인

    Parameters:
        df: DataFrame - 원본 데이터
        numeric_cols: list - 수치형 컬럼 목록
        categorical_cols: list - 범주형 컬럼 목록
        missing_strategy: str - 결측치 처리 ('mean', 'median', 'drop')
        outlier_method: str - 이상치 탐지 방법 ('iqr', 'zscore')
        outlier_treatment: str - 이상치 처리 ('clip', 'remove', 'flag')
        iqr_multiplier: float - IQR 배수
        zscore_threshold: float - Z-score 임계값

    Returns:
        df_processed: DataFrame - 전처리된 데이터
        report: dict - 전처리 리포트
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
print("파이프라인 실행:")
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
```

### 결과 해설

- 이상치 처리 전략은 분석 목적에 따라 선택해야 함
- 삭제는 정보 손실이 발생하지만 깨끗한 데이터를 얻을 수 있음
- 클리핑은 극단값을 줄이면서 데이터 크기를 유지함
- 플래그는 원본 정보를 보존하면서 이상치를 표시함
- 전처리 전후 통계 비교로 처리의 영향을 확인해야 함
- Titanic 데이터에서 고가 요금 승객이 생존율이 높음을 확인할 수 있음

---

## 연습 문제

### 연습 1

Wine Quality 데이터에서 pH의 결측치를 확인하고 중앙값으로 대체하시오.

```python
# 정답
import pandas as pd

WINE_QUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(WINE_QUALITY_URL, sep=';')

print(f"pH 결측치 개수: {wine['pH'].isnull().sum()}")
median_ph = wine['pH'].median()
wine['pH'] = wine['pH'].fillna(median_ph)
print(f"대체 후 결측치: {wine['pH'].isnull().sum()}")
```

### 연습 2

Titanic 데이터에서 age의 이상치를 IQR 방법으로 탐지하고 개수를 출력하시오.

```python
# 정답
Q1 = df_clean['age'].quantile(0.25)
Q3 = df_clean['age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = (df_clean['age'] < lower) | (df_clean['age'] > upper)
print(f"age 이상치 개수: {outliers.sum()}")
print(f"정상 범위: {lower:.1f} ~ {upper:.1f}")
```

### 연습 3

전처리 파이프라인을 사용하여 Titanic 데이터를 처리하고, 결측치와 이상치 처리 결과를 확인하시오.

```python
# 정답
df_processed, report = preprocess_titanic_data(
    df=df,
    numeric_cols=['age', 'fare'],
    categorical_cols=['embarked'],
    missing_strategy='median',
    outlier_method='iqr',
    outlier_treatment='clip'
)
print(f"처리 후 형태: {df_processed.shape}")
print(f"처리 후 결측치: {df_processed.isnull().sum().sum()}")
```

---

## 핵심 정리

| 구분 | 내용 |
|------|------|
| **결측치 탐지** | df.isnull().sum(), df.info() |
| **결측치 처리** | dropna() (삭제), fillna() (대체), interpolate() (보간) |
| **이상치 탐지** | IQR (Q1-1.5xIQR ~ Q3+1.5xIQR), Z-score (\|Z\| > 2 또는 3) |
| **이상치 처리** | 삭제, clip() (클리핑), 대체, 플래그 |
| **전처리 순서** | 결측치 탐지 -> 결측치 처리 -> 이상치 탐지 -> 이상치 처리 -> 검증 |
| **주의사항** | 데이터 누출 방지, 전후 비교 필수 |

---

## 다음 차시 예고

**10차시: 제조 데이터 전처리 (2)**

- 데이터 정규화와 표준화
- 범주형 데이터 인코딩
- 피처 엔지니어링 기초
