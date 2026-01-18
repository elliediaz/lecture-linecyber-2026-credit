---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 9차시'
footer: '공공데이터를 활용한 AI 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
  table { font-size: 0.9em; }
---

# 제조 데이터 전처리 (1)

## 9차시 | Part II. 기초 수리와 데이터 분석

**결측치와 이상치 처리**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **결측치**를 탐지하고 적절히 처리한다
2. **이상치**를 탐지하는 방법을 적용한다
3. 상황에 맞는 **전처리 전략**을 선택한다

---

# 강의 구성

| 파트 | 대주제 | 시간 |
|:----:|--------|:----:|
| 1 | 결측치 탐지 및 처리 | 10분 |
| 2 | 이상치 탐지 방법 적용 | 10분 |
| 3 | 상황에 맞는 전처리 전략 선택 | 10분 |

---

<!-- _class: lead -->

# Part 1
## 결측치 탐지 및 처리

---

# 왜 전처리가 중요한가?

## 데이터 분석의 현실

```
전체 프로젝트 시간 배분:

┌─────────────────────────────────────────┐
│ 데이터 수집·전처리    ██████████████ 60~80% │
│ 모델링              ████ 10~20%          │
│ 평가·배포           ████ 10~20%          │
└─────────────────────────────────────────┘
```

> **"Garbage In, Garbage Out"**
> 나쁜 데이터 → 나쁜 모델

---

# 전처리의 목표

## 분석에 적합한 데이터로 변환

| 문제 | 해결 | 방법 |
|------|------|------|
| **결측치** | 비어있는 값 처리 | 삭제, 대체 |
| **이상치** | 극단값 처리 | 탐지, 클리핑, 대체 |
| **불균형** | 클래스 비율 조정 | 오버/언더 샘플링 |
| **스케일** | 단위 통일 | 표준화, 정규화 |
| **형식** | 자료형 변환 | 인코딩, 변환 |

---

# 결측치 (Missing Values)

## 비어있는 값

### 제조 현장에서 발생 원인

| 원인 | 설명 | 예시 |
|------|------|------|
| **센서 오류** | 통신 두절, 배터리 방전 | 온도 센서 끊김 |
| **입력 누락** | 수기 입력 시 실수 | 불량수 미기록 |
| **시스템 장애** | PLC, MES 연결 끊김 | 로그 누락 |
| **정상 결측** | 특정 조건에서만 측정 | 비가동 시간 |

---

# pandas에서 결측치 표현

## NaN, None, NA

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

---

# 결측치 탐지 메서드

## 기본 확인

```python
# 전체 정보 확인
df.info()

# 열별 결측치 수
df.isnull().sum()

# 결측치 비율 (%)
(df.isnull().sum() / len(df) * 100).round(2)

# 행별 결측치 수
df.isnull().sum(axis=1)

# 결측치 있는 행 확인
df[df.isnull().any(axis=1)]
```

---

# 결측치 탐지 실습

```python
import pandas as pd
import numpy as np

# 샘플 데이터
df = pd.DataFrame({
    '온도': [85, np.nan, 87, 86, np.nan, 88],
    '생산량': [1200, 1150, np.nan, 1180, 1190, 1200],
    '불량수': [30, 28, 25, np.nan, 27, 29]
})

print("=== 결측치 현황 ===")
print(df.isnull().sum())
print()
print("=== 결측 비율 ===")
print((df.isnull().sum() / len(df) * 100).round(1))
```

---

# 결측치 시각화

```python
import matplotlib.pyplot as plt

# 결측치 막대 그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 결측치 수
df.isnull().sum().plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('열별 결측치 수')
axes[0].set_ylabel('결측치 수')

# 결측치 히트맵
axes[1].imshow(df.isnull(), aspect='auto', cmap='Reds')
axes[1].set_title('결측치 위치 (빨강=결측)')
axes[1].set_xlabel('열')
axes[1].set_ylabel('행')

plt.tight_layout()
plt.show()
```

---

# 결측치 처리 전략 1: 삭제

## 결측치가 적을 때 사용

```python
# 결측치 있는 행 전체 삭제
df_clean = df.dropna()

# 특정 열에 결측치 있는 행만 삭제
df_clean = df.dropna(subset=['온도', '생산량'])

# 모든 값이 결측인 행만 삭제
df_clean = df.dropna(how='all')

# 특정 개수 이상 결측인 행 삭제
df_clean = df.dropna(thresh=2)  # 유효값이 2개 이상
```

### 사용 시점
- 결측치 비율 < 5%
- 무작위(MCAR) 결측일 때

---

# 결측치 처리 전략 2: 대체 (기본)

## Imputation

```python
# 상수로 대체
df['온도'].fillna(0, inplace=True)

# 평균으로 대체
df['온도'].fillna(df['온도'].mean(), inplace=True)

# 중앙값으로 대체 (이상치에 강건)
df['온도'].fillna(df['온도'].median(), inplace=True)

# 최빈값으로 대체 (범주형)
df['등급'].fillna(df['등급'].mode()[0], inplace=True)
```

---

# 결측치 처리 전략 3: 대체 (시계열)

## 앞/뒤 값으로 채우기

```python
# 앞 값으로 채우기 (Forward Fill)
df['센서값'] = df['센서값'].ffill()

# 뒤 값으로 채우기 (Backward Fill)
df['센서값'] = df['센서값'].bfill()

# 선형 보간
df['온도'] = df['온도'].interpolate(method='linear')

# 시계열 보간
df['측정값'] = df['측정값'].interpolate(method='time')
```

### 사용 시점
- 연속적인 시계열 데이터
- 값의 변화가 부드러운 경우

---

# 결측치 처리 전략 4: 그룹별 대체

## 더 정교한 대체

```python
# 라인별 평균으로 대체
df['온도'] = df.groupby('라인')['온도'].transform(
    lambda x: x.fillna(x.mean())
)

# 날짜별 중앙값으로 대체
df['생산량'] = df.groupby('날짜')['생산량'].transform(
    lambda x: x.fillna(x.median())
)
```

### 사용 시점
- 그룹 간 차이가 큰 경우
- 더 정확한 대체가 필요할 때

---

# 결측치 처리 가이드

## 비율별 권장 전략

| 결측 비율 | 권장 방법 | 비고 |
|----------|----------|------|
| < 5% | 삭제 또는 평균/중앙값 대체 | 영향 미미 |
| 5~15% | 중앙값, 그룹별 평균 | 대체 방법 비교 |
| 15~30% | 보간, 예측 모델 대체 | 신중한 검토 |
| > 30% | 열 삭제 고려 | 정보 손실 감수 |

### 핵심 원칙
1. 결측 패턴 먼저 파악 (무작위? 체계적?)
2. 도메인 지식 활용
3. 대체 전후 분포 비교

---

# Part 1 정리

## 결측치 처리

### 탐지
- `df.isnull().sum()`: 열별 결측치 수
- `df.info()`: 전체 정보

### 처리
- `df.dropna()`: 삭제
- `df.fillna()`: 대체
- `df.interpolate()`: 보간

### 원칙
- 5% 미만: 삭제 OK
- 5% 이상: 대체 검토
- 30% 이상: 열 삭제 고려

---

<!-- _class: lead -->

# Part 2
## 이상치 탐지 방법 적용

---

# 이상치 (Outliers)

## 비정상적으로 극단적인 값

### 제조 현장에서 발생 원인

| 원인 | 설명 | 예시 |
|------|------|------|
| **측정 오류** | 센서 고장, 캘리브레이션 | 온도 -999 |
| **입력 실수** | 단위 오류, 오타 | 생산량 12000 (1200) |
| **실제 극단값** | 공정 이탈, 설비 고장 | 불량률 급등 |
| **시스템 이상** | 리셋 값, 오버플로우 | 0 또는 999999 |

---

# 이상치의 두 얼굴

## 오류? 중요한 신호?

```
분석 목적에 따라 다름:

불량품 예측 모델:
  → 이상치가 핵심 정보! (제거하면 안 됨)

평균 생산량 추정:
  → 이상치 제거 고려 (왜곡 방지)

이상 탐지 시스템:
  → 이상치를 찾는 게 목적
```

> **핵심 질문**: 이상치가 **오류**인가, **신호**인가?

---

# 이상치 탐지 방법 1: IQR

## 사분위수 범위 (Interquartile Range)

```python
Q1 = df['생산량'].quantile(0.25)  # 25%
Q3 = df['생산량'].quantile(0.75)  # 75%
IQR = Q3 - Q1

# 경계값 계산
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 탐지
outliers = (df['생산량'] < lower) | (df['생산량'] > upper)
print(f"이상치 개수: {outliers.sum()}")
```

> 상자그림(Boxplot)과 동일한 기준

---

# IQR 방법 시각화

## 상자그림으로 이해

```
                    이상치 (Q3 + 1.5×IQR 초과)
                          ●
                          │
         ┌────────────────┴──────────────────┐ ← 상한
         │                                    │
    ┌────┼────────────────────────────────────┼────┐
    │    │               Q3                   │    │
    │    ├────────────────────────────────────┤    │
    │    │            중앙값                   │    │
    │    ├────────────────────────────────────┤    │
    │    │               Q1                   │    │
    └────┼────────────────────────────────────┼────┘
         │                                    │
         └────────────────┬──────────────────┘ ← 하한
                          │
                          ●
                    이상치 (Q1 - 1.5×IQR 미만)
```

---

# IQR 탐지 함수

```python
def detect_outliers_iqr(data, factor=1.5):
    """
    IQR 방법으로 이상치 탐지

    Parameters:
        data: 데이터 Series 또는 배열
        factor: IQR 배수 (기본 1.5)

    Returns:
        이상치 마스크, 하한, 상한
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    outliers = (data < lower) | (data > upper)
    return outliers, lower, upper
```

---

# 이상치 탐지 방법 2: Z-score

## 표준점수

```python
mean = df['생산량'].mean()
std = df['생산량'].std()

# Z-score 계산
z_scores = (df['생산량'] - mean) / std

# 이상치 탐지 (|Z| > 3)
outliers = np.abs(z_scores) > 3
print(f"이상치 개수: {outliers.sum()}")
```

| Z-score | 비율 | 해석 |
|---------|------|------|
| \|Z\| > 2 | 4.6% | 주의 필요 |
| \|Z\| > 3 | 0.3% | 이상치 |

---

# Z-score 탐지 함수

```python
def detect_outliers_zscore(data, threshold=3):
    """
    Z-score 방법으로 이상치 탐지

    Parameters:
        data: 데이터 Series 또는 배열
        threshold: Z-score 임계값 (기본 3)

    Returns:
        이상치 마스크, z_scores
    """
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std

    outliers = np.abs(z_scores) > threshold
    return outliers, z_scores
```

---

# IQR vs Z-score 비교

## 어떤 방법을 선택할까?

| 비교 항목 | IQR | Z-score |
|----------|-----|---------|
| **분포 가정** | 없음 | 정규분포 가정 |
| **이상치 영향** | 강건함 | 영향 받음 |
| **기준** | 1.5×IQR | 보통 2~3σ |
| **적합 상황** | 비대칭 분포 | 대칭 분포 |
| **시각화** | 상자그림 | 히스토그램 |

### 권장
- **정규분포에 가까우면**: Z-score
- **분포 형태가 불확실하면**: IQR

---

# 양쪽 방법 비교 코드

```python
# IQR 방법
outliers_iqr, lower, upper = detect_outliers_iqr(df['생산량'])

# Z-score 방법
outliers_z, z_scores = detect_outliers_zscore(df['생산량'])

print(f"IQR 이상치: {outliers_iqr.sum()}개")
print(f"Z-score 이상치: {outliers_z.sum()}개")

# 두 방법 모두에서 탐지된 이상치
both = outliers_iqr & outliers_z
print(f"공통 이상치: {both.sum()}개")
```

---

# 이상치 시각화

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 상자그림
axes[0].boxplot(df['생산량'])
axes[0].set_title('상자그림')

# 히스토그램 + 경계
axes[1].hist(df['생산량'], bins=20, edgecolor='black')
axes[1].axvline(lower, color='r', linestyle='--', label='IQR 하한')
axes[1].axvline(upper, color='r', linestyle='--', label='IQR 상한')
axes[1].legend()
axes[1].set_title('히스토그램')

# 산점도 + 이상치 표시
colors = ['red' if o else 'blue' for o in outliers_iqr]
axes[2].scatter(range(len(df)), df['생산량'], c=colors, alpha=0.5)
axes[2].set_title('산점도 (빨강=이상치)')

plt.tight_layout()
plt.show()
```

---

# Part 2 정리

## 이상치 탐지

### 방법
- **IQR**: Q1 - 1.5×IQR ~ Q3 + 1.5×IQR
- **Z-score**: |Z| > 2 또는 3

### 선택 기준
- 정규분포 → Z-score
- 비대칭/불확실 → IQR

### 주의사항
- 무조건 제거하지 말 것!
- 도메인 지식 활용
- 분석 목적 고려

---

<!-- _class: lead -->

# Part 3
## 상황에 맞는 전처리 전략 선택

---

# 이상치 처리 전략 1: 삭제

## 가장 단순한 방법

```python
# 이상치가 아닌 데이터만 선택
df_clean = df[~outliers]

# 또는 이상치 행 삭제
df_clean = df.drop(df[outliers].index)
```

### 사용 시점
- 이상치가 **명백한 오류**일 때
- 이상치 비율이 낮을 때 (< 5%)
- 이상치가 분석에 중요하지 않을 때

### 주의점
- 정보 손실 발생
- 데이터 감소

---

# 이상치 처리 전략 2: 클리핑

## 상/하한으로 제한

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

### 사용 시점
- 극단값은 줄이되 데이터는 보존하고 싶을 때
- 예측 모델 학습 시 극단값 영향 완화

---

# 이상치 처리 전략 3: 대체

## 대표값으로 교체

```python
# 중앙값으로 대체
median_val = df['생산량'].median()
df.loc[outliers, '생산량'] = median_val

# 그룹별 중앙값으로 대체
df['생산량'] = df.groupby('라인')['생산량'].transform(
    lambda x: x.where(~outliers_iqr, x.median())
)
```

### 사용 시점
- 이상치가 측정 오류로 확인될 때
- 데이터 크기를 유지해야 할 때

---

# 이상치 처리 전략 4: 플래그

## 이상치 표시만 하기

```python
# 이상치 플래그 열 추가
df['이상치_여부'] = outliers_iqr

# 이상치 유형 기록
df['이상치_유형'] = 'normal'
df.loc[df['생산량'] > upper, '이상치_유형'] = 'high'
df.loc[df['생산량'] < lower, '이상치_유형'] = 'low'

# 이상치 정보 보존
df['원본_생산량'] = df['생산량']
df['생산량'] = df['생산량'].clip(lower, upper)
```

### 사용 시점
- 이상치 분석이 필요할 때
- 원본 정보를 보존하면서 처리할 때

---

# 이상치 처리 결정 프로세스

```
                  이상치 발견
                      │
                      ▼
            ┌─────────────────┐
            │ 원인 조사       │
            └────────┬────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   측정 오류      실제 극단값     불확실
       │             │             │
       ▼             ▼             ▼
    삭제/대체      플래그       전문가 상의
```

---

# 상황별 전처리 전략

## 분석 목적에 따른 선택

| 분석 목적 | 결측치 처리 | 이상치 처리 |
|----------|------------|------------|
| **평균 추정** | 중앙값 대체 | 클리핑 또는 삭제 |
| **예측 모델** | 평균/보간 대체 | 클리핑 + 플래그 |
| **불량 예측** | 삭제 또는 대체 | 유지 (중요 신호) |
| **이상 탐지** | 플래그 표시 | 유지 (찾는 대상) |
| **탐색 분석** | 플래그 표시 | 분리 분석 |

---

# 전처리 파이프라인

## 표준 순서

```python
def preprocess_data(df):
    """데이터 전처리 파이프라인"""
    df = df.copy()

    # 1. 결측치 탐지 및 기록
    missing_info = df.isnull().sum()

    # 2. 결측치 처리
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    # 3. 이상치 탐지 (IQR)
    for col in ['온도', '생산량', '불량수']:
        outliers, lower, upper = detect_outliers_iqr(df[col])
        df[f'{col}_이상치'] = outliers
        df[col] = df[col].clip(lower, upper)

    return df
```

---

# 전처리 전후 비교

## 검증 필수!

```python
def compare_preprocessing(before, after, column):
    """전처리 전후 비교"""
    print(f"=== {column} 전후 비교 ===")
    print(f"평균: {before[column].mean():.2f} → {after[column].mean():.2f}")
    print(f"중앙값: {before[column].median():.2f} → {after[column].median():.2f}")
    print(f"표준편차: {before[column].std():.2f} → {after[column].std():.2f}")
    print(f"최소: {before[column].min():.2f} → {after[column].min():.2f}")
    print(f"최대: {before[column].max():.2f} → {after[column].max():.2f}")
```

---

# 전처리 주의사항

## 피해야 할 실수

| 실수 | 문제점 | 올바른 방법 |
|------|--------|------------|
| 결측치를 0으로 채움 | 의미 왜곡 | 평균/중앙값 대체 |
| 이상치 무조건 제거 | 정보 손실 | 원인 파악 후 결정 |
| 전체 데이터로 대체값 계산 | 데이터 누출 | 학습 데이터만 사용 |
| 전후 비교 안 함 | 왜곡 미발견 | 분포 비교 필수 |

---

# 데이터 누출 (Data Leakage)

## 주의해야 할 상황

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

> 테스트 데이터 정보가 학습에 사용되면 안 됨!

---

# Part 3 정리

## 전처리 전략 선택

### 핵심 원칙
1. **목적에 맞게**: 분석 목표에 따라 전략 선택
2. **도메인 지식**: 데이터의 의미 이해
3. **전후 비교**: 분포 변화 확인
4. **문서화**: 처리 내용 기록

### 처리 순서
결측치 탐지 → 결측치 처리 → 이상치 탐지 → 이상치 처리 → 검증

---

<!-- _class: lead -->

# 실습편

## 결측치와 이상치 처리 실습

---

# 실습 개요

## 제조 데이터 전처리

### 실습 목표
1. 결측치 탐지 및 처리
2. 이상치 탐지 및 처리
3. 전처리 전후 비교

### 실습 환경

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 생성

```python
np.random.seed(42)
n = 100

df = pd.DataFrame({
    '일자': pd.date_range('2024-01-01', periods=n),
    '온도': np.random.normal(85, 5, n),
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(30, n),
    '라인': np.random.choice(['A', 'B', 'C'], n)
})

# 결측치 삽입
df.loc[np.random.choice(n, 10, replace=False), '온도'] = np.nan
df.loc[np.random.choice(n, 5, replace=False), '생산량'] = np.nan

# 이상치 삽입
df.loc[5, '생산량'] = 2000
df.loc[15, '생산량'] = 500
df.loc[25, '온도'] = 120

print(df.head(10))
```

---

# 실습 2: 결측치 탐지

```python
print("=== 결측치 현황 ===")
print(df.isnull().sum())
print()

# 결측 비율
missing_ratio = (df.isnull().sum() / len(df) * 100).round(2)
print("=== 결측 비율 (%) ===")
print(missing_ratio)

# 시각화
fig, ax = plt.subplots(figsize=(8, 4))
df.isnull().sum().plot(kind='bar', color='coral', edgecolor='black', ax=ax)
ax.set_ylabel('결측치 수')
ax.set_title('열별 결측치 현황')
plt.tight_layout()
plt.show()
```

---

# 실습 3: 결측치 처리

```python
df_clean = df.copy()

# 온도: 중앙값으로 대체
temp_median = df_clean['온도'].median()
df_clean['온도'].fillna(temp_median, inplace=True)
print(f"온도 결측치를 {temp_median:.1f}도로 대체")

# 생산량: 라인별 평균으로 대체
df_clean['생산량'] = df_clean.groupby('라인')['생산량'].transform(
    lambda x: x.fillna(x.mean())
)
print("생산량 결측치를 라인별 평균으로 대체")

# 결과 확인
print(f"\n처리 후 결측치: {df_clean.isnull().sum().sum()}개")
```

---

# 실습 4: 이상치 탐지 (IQR)

```python
Q1 = df_clean['생산량'].quantile(0.25)
Q3 = df_clean['생산량'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_prod = (df_clean['생산량'] < lower) | (df_clean['생산량'] > upper)

print("=== 생산량 이상치 (IQR) ===")
print(f"Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
print(f"하한: {lower:.0f}, 상한: {upper:.0f}")
print(f"이상치 개수: {outliers_prod.sum()}개")
print(f"\n이상치 값:\n{df_clean[outliers_prod][['일자', '생산량', '라인']]}")
```

---

# 실습 5: 이상치 탐지 (Z-score)

```python
mean = df_clean['온도'].mean()
std = df_clean['온도'].std()
z_scores = (df_clean['온도'] - mean) / std

outliers_temp = np.abs(z_scores) > 2

print("=== 온도 이상치 (Z-score) ===")
print(f"평균: {mean:.1f}, 표준편차: {std:.1f}")
print(f"이상치 개수 (|Z|>2): {outliers_temp.sum()}개")

if outliers_temp.sum() > 0:
    print(f"\n이상치 상세:")
    outlier_df = df_clean[outliers_temp][['일자', '온도']].copy()
    outlier_df['Z-score'] = z_scores[outliers_temp]
    print(outlier_df)
```

---

# 실습 6: 이상치 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 생산량 상자그림
bp = axes[0].boxplot(df_clean['생산량'])
axes[0].set_ylabel('생산량')
axes[0].set_title('생산량 분포 (이상치 포함)')

# 온도 히스토그램 + 이상치 표시
axes[1].hist(df_clean['온도'], bins=20, edgecolor='black', alpha=0.7)
for idx in df_clean[outliers_temp].index:
    axes[1].axvline(df_clean.loc[idx, '온도'], color='red', linestyle='--')
axes[1].set_xlabel('온도')
axes[1].set_ylabel('빈도')
axes[1].set_title('온도 분포 (빨간선=이상치)')

plt.tight_layout()
plt.show()
```

---

# 실습 7: 이상치 처리

```python
df_final = df_clean.copy()

# 생산량: 클리핑
df_final['생산량'] = df_final['생산량'].clip(lower, upper)

# 온도: 이상치를 중앙값으로 대체
temp_median = df_final['온도'].median()
df_final.loc[outliers_temp, '온도'] = temp_median

print("=== 처리 결과 ===")
print(f"생산량 범위: {df_final['생산량'].min():.0f} ~ {df_final['생산량'].max():.0f}")
print(f"온도 범위: {df_final['온도'].min():.1f} ~ {df_final['온도'].max():.1f}")
```

---

# 실습 8: 전후 비교

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 생산량 전후
axes[0, 0].hist(df_clean['생산량'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('처리 전 생산량')
axes[0, 1].hist(df_final['생산량'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('처리 후 생산량')

# 온도 전후
axes[1, 0].hist(df_clean['온도'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('처리 전 온도')
axes[1, 1].hist(df_final['온도'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[1, 1].set_title('처리 후 온도')

plt.tight_layout()
plt.show()
```

---

# 실습 정리

## 핵심 체크포인트

### 결측치
- [ ] `isnull().sum()`으로 탐지
- [ ] `fillna()`로 대체
- [ ] 비율에 따라 전략 선택

### 이상치
- [ ] IQR 또는 Z-score로 탐지
- [ ] `clip()`으로 클리핑
- [ ] 도메인에 따라 처리 결정

### 검증
- [ ] 전후 분포 비교

---

# 다음 차시 예고

## 9차시: 제조 데이터 전처리 (2)

### 학습 내용
- 데이터 정규화와 표준화
- 범주형 데이터 인코딩
- 피처 엔지니어링 기초

### 준비물
- 오늘 배운 코드 복습
- sklearn 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심

1. **결측치**: 탐지(`isnull`) → 처리(`dropna`/`fillna`)
2. **이상치**: 탐지(IQR/Z-score) → 처리(`clip`/대체)
3. **원칙**: 데이터 이해 → 도메인 지식 → 전후 비교

### 자주 하는 실수
- 결측치를 0으로 채우는 것 (의미 왜곡)
- 이상치를 무조건 제거하는 것

---

# 감사합니다

## 9차시: 제조 데이터 전처리 (1)

**다음 시간에 스케일링과 인코딩을 배워봅시다!**
