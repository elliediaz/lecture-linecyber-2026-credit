# [11차시] 제조 데이터 탐색 분석 종합 (EDA)

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **EDA의 전체 흐름**을 체계적으로 이해함
2. **데이터에서 인사이트를 도출**하는 방법을 습득함
3. **제조 데이터를 종합적으로 분석**하여 권고사항을 제시함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Wine Quality** | UCI ML Repository | EDA 종합 실습 |

Wine Quality 데이터셋은 와인의 화학적 특성과 품질 점수로 구성되어 있으며, 제조 공정의 품질 관리와 유사한 분석이 가능함.

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | EDA의 전체 흐름 이해 | 5단계 프로세스, 체크리스트 |
| 2 | 데이터에서 인사이트 도출 | 패턴 발견, 가설 수립, 문서화 |
| 3 | 제조 데이터 종합 분석 프로젝트 | 실전 EDA, 대시보드, 보고서 |

---

## 파트 1: EDA의 전체 흐름 이해

### 개념 설명

#### EDA(Exploratory Data Analysis)란?

데이터를 탐색하여 패턴, 이상치, 관계를 발견하는 과정임.

| 구분 | EDA (탐색적) | CDA (확증적) |
|------|-------------|--------------|
| 목적 | 가설 **생성** | 가설 **검증** |
| 방법 | 시각화, 요약 | 통계적 검정 |
| 질문 | "무엇이 있나?" | "가설이 맞나?" |
| 순서 | **먼저** | 나중에 |

#### EDA 5단계 프로세스

```
1단계: 데이터 개요 파악
       shape, dtypes, info, head
              |
              v
2단계: 단변량 분석
       각 변수 개별 분포 확인
              |
              v
3단계: 이변량 분석
       두 변수 간 관계 확인
              |
              v
4단계: 다변량 분석
       여러 변수 동시 분석
              |
              v
5단계: 인사이트 도출
       발견 정리, 가설 수립
```

#### 1단계: 데이터 개요 파악

| 질문 | 확인 방법 |
|------|----------|
| 데이터 크기는? | `df.shape` |
| 어떤 변수들이 있나? | `df.columns`, `df.dtypes` |
| 결측치가 있나? | `df.isnull().sum()` |
| 중복 행이 있나? | `df.duplicated().sum()` |
| 데이터 범위는 적절한가? | `df.describe()` |

#### 2단계: 단변량 분석

| 변수 유형 | 그래프 | 확인 내용 |
|----------|--------|----------|
| 수치형 | 히스토그램 | 분포 형태 (정규? 왜도?) |
| 수치형 | 박스플롯 | 이상치, 사분위수 |
| 범주형 | 막대그래프 | 범주별 빈도 |
| 범주형 | 파이차트 | 비율 (3~5개 범주 이하) |

#### 3단계: 이변량 분석

| 조합 | 시각화 | 통계 |
|------|--------|------|
| 수치 vs 수치 | 산점도 | 상관계수 |
| 범주 vs 수치 | 그룹별 박스플롯 | t-검정, ANOVA |
| 범주 vs 범주 | 스택드 바차트, 히트맵 | 교차표, 카이제곱 |

### 실습 코드

#### 데이터 로드 및 개요 파악

```python
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

# Wine Quality 데이터셋 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')
print("Wine Quality 데이터셋 로드 완료")

# 데이터 크기 및 구조
print(f"데이터 크기: {df.shape} (행, 열)")
print(f"\n컬럼 목록:\n{df.columns.tolist()}")
```

#### 기본 정보 확인

```python
# 데이터 타입
print(df.dtypes)

# 처음 5행
print(df.head())

# 기술 통계
print(df.describe().round(3))

# 결측치 현황
missing = df.isnull().sum()
missing_pct = df.isnull().sum() / len(df) * 100
print(pd.DataFrame({'결측치 수': missing, '비율(%)': missing_pct.round(2)}))
```

#### 이상치 확인 (IQR 방법)

```python
# 이상치 확인
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
```

### 결과 해설

- Wine Quality 데이터셋은 1,599개 샘플과 12개 변수로 구성됨
- 결측치가 없어 별도의 결측 처리가 불필요함
- 일부 변수에서 IQR 기준 이상치가 존재하나, 제거 여부는 도메인 지식 기반으로 판단해야 함

---

## 파트 2: 데이터에서 인사이트 도출

### 개념 설명

#### 인사이트란?

데이터에서 발견한 의미 있는 패턴으로, 다음 3요소를 갖추어야 함:

1. **구체적**: 명확한 수치와 근거
2. **실행 가능**: 행동으로 연결 가능
3. **데이터 기반**: 통계적 뒷받침

#### 좋은 인사이트 vs 나쁜 인사이트

| 특징 | 나쁜 인사이트 | 좋은 인사이트 |
|------|-------------|--------------|
| 구체성 | "불량률이 높다" | "라인 B 불량률이 평균(3.2%) 대비 **5.1%로 1.9%p 높음**" |
| 실행 가능 | "품질을 개선해야 한다" | "**온도 90도C 이상**에서 경고 알람 설정 권장" |
| 데이터 기반 | "아마 그럴 것 같다" | "상관계수 **r=0.42**, **p<0.05**로 통계적 유의" |

#### 인사이트 도출 프레임워크

```
1. 패턴 발견
   "라인 B의 불량률이 높게 나타남"
        |
        v
2. 수치화
   "평균 5.1% vs 전체 평균 3.2% (+1.9%p)"
        |
        v
3. 통계적 검증
   "t-검정 결과 p-value = 0.003 (유의)"
        |
        v
4. 원인 추론
   "라인 B는 노후 장비, 오후 가동 多"
        |
        v
5. 행동 제안
   "라인 B 장비 점검, 온도 모니터링 강화"
```

### 실습 코드

#### 단변량 분석: 타겟 변수 분포

```python
# 품질(quality) 분포 확인
print(f"평균: {df['quality'].mean():.3f}")
print(f"중앙값: {df['quality'].median():.3f}")
print(f"표준편차: {df['quality'].std():.3f}")
print(f"최솟값: {df['quality'].min()}")
print(f"최댓값: {df['quality'].max()}")

# 품질 등급별 빈도
quality_counts = df['quality'].value_counts().sort_index()
print(quality_counts)
print(f"\n품질 분포 비율:")
print((quality_counts / len(df) * 100).round(1))
```

#### 품질 그룹 생성

```python
# 품질 그룹 생성 (분석 편의)
df['quality_group'] = pd.cut(df['quality'], bins=[0, 4, 6, 10],
                              labels=['Low(3-4)', 'Medium(5-6)', 'High(7-8)'])
print("품질 그룹별 빈도:")
print(df['quality_group'].value_counts())
```

#### 이변량 분석: 상관관계

```python
# 품질과 각 변수의 상관관계
correlations = df.corr()['quality'].drop('quality').sort_values(ascending=False)
print("=== quality와의 상관계수 (내림차순) ===")
for var, corr in correlations.items():
    direction = "+" if corr > 0 else ""
    strength = "강함" if abs(corr) >= 0.3 else "중간" if abs(corr) >= 0.2 else "약함"
    print(f"  {var:22}: {direction}{corr:.4f} ({strength})")
```

#### 통계 검정: 고품질 vs 저품질

```python
# 고품질 vs 저품질 비교
high_quality = df[df['quality'] >= 7]
low_quality = df[df['quality'] <= 4]

print(f"고품질 와인 (quality >= 7): {len(high_quality)}개")
print(f"저품질 와인 (quality <= 4): {len(low_quality)}개")

# t-검정
test_vars = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
print("\n=== t-검정 결과 ===")
for var in test_vars:
    t_stat, p_value = stats.ttest_ind(high_quality[var], low_quality[var])
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {var:20}: t={t_stat:7.2f}, p={p_value:.4f} {significance}")

print("\n  *** p < 0.001, ** p < 0.01, * p < 0.05")
```

### 결과 해설

- **alcohol(알코올 도수)**이 품질과 가장 강한 양의 상관관계(r=0.48)를 보임
- **volatile acidity(휘발성 산도)**가 품질과 가장 강한 음의 상관관계(r=-0.39)를 보임
- 고품질 와인과 저품질 와인 간 주요 변수에서 통계적으로 유의미한 차이가 확인됨 (p < 0.001)

---

## 파트 3: 제조 데이터 종합 분석 프로젝트

### 개념 설명

#### EDA 워크플로우

```
1단계: 데이터 로드 및 개요
    |
    v
2단계: 데이터 품질 확인 (결측치, 이상치)
    |
    v
3단계: 단변량 분석 (각 변수 분포)
    |
    v
4단계: 이변량 분석 (타겟과의 관계)
    |
    v
5단계: 다변량 분석 (복합 요인)
    |
    v
6단계: 인사이트 정리 및 시각화
    |
    v
7단계: 보고서 작성
```

#### EDA 결과 보고서 구조

```
+-------------------------------------+
| 1. 분석 개요                        |
|    - 분석 목적, 데이터 설명         |
+-------------------------------------+
| 2. 데이터 품질                      |
|    - 결측치/이상치 현황 및 처리     |
+-------------------------------------+
| 3. 주요 발견 사항                   |
|    - 핵심 인사이트 3~5개            |
|    - 시각화 자료                    |
+-------------------------------------+
| 4. 결론 및 권고사항                 |
|    - 실행 가능한 제안               |
|    - 추가 분석 필요 사항            |
+-------------------------------------+
```

### 실습 코드

#### 다변량 분석: 복합 조건

```python
# 복합 조건 분석: 알코올 + 휘발성 산도
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
```

#### 품질 그룹별 프로파일

```python
# 품질 그룹별 프로파일
feature_cols = [col for col in numeric_cols if col != 'quality']
profile = df.groupby('quality_group')[feature_cols].mean()
print(profile.T.round(3))
```

#### 종합 대시보드 생성

```python
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
top_corr.plot(kind='barh', ax=axes[1, 1],
              color=['green' if c > 0 else 'red' for c in correlations[top_corr.index]])
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
```

#### 인사이트 정리

```python
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

print(f"""
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
    - sulphates (상관계수: {correlations['sulphates']:.3f})

    (-) 음의 영향:
    - volatile acidity (상관계수: {correlations['volatile acidity']:.3f})
    - density (상관계수: {correlations['density']:.3f})

[3] 최적 조건 조합
    - 알코올: {best_combo[0]}
    - 휘발성 산도: {best_combo[1]}
    - 평균 품질: {best_quality:.2f}점

=================================================================
""")
```

### 결과 해설

분석 결과 도출된 주요 인사이트:

1. **품질 분포**: 대부분의 와인이 5~6점에 집중되어 있으며, 극단적인 품질은 드묾
2. **핵심 영향 요인**: 알코올 도수(+)와 휘발성 산도(-)가 품질에 가장 큰 영향을 미침
3. **최적 조건**: 높은 알코올(11%+) + 낮은 휘발성 산도(~0.4) 조합에서 최고 품질 달성

---

## 권고사항

### 즉시 실행 (Quick Win)

1. **휘발성 산도(VA) 모니터링 강화**: 0.4 이하로 유지 권장, 0.6 초과 시 경고 알람 설정
2. **알코올 도수 관리**: 11% 이상 목표, 발효 공정 최적화

### 단기 개선

3. **황산염(Sulphates) 적정 수준 유지**: 0.65~0.75 범위 권장
4. **구연산(Citric Acid) 첨가 검토**: 과일 향미 증가, 품질 향상에 기여

### 장기 개선

5. **품질 예측 모델 개발**: 발효 전 품질 사전 예측
6. **센서 데이터 수집 확대**: 실시간 모니터링

---

## 핵심 정리

### EDA 5단계 체크리스트

| 단계 | 핵심 활동 | 도구 |
|:----:|----------|------|
| 1 | 데이터 개요 파악 | shape, dtypes, head, describe |
| 2 | 데이터 품질 확인 | 결측치, 이상치 확인 |
| 3 | 단변량 분석 | 히스토그램, 박스플롯 |
| 4 | 이변량 분석 | 산점도, 상관계수, t-검정 |
| 5 | 다변량 분석 | 피벗, 히트맵 |
| 6 | 인사이트 도출 | 발견 정리, 권고사항 작성 |

### 좋은 인사이트 3요소

| 요소 | 설명 | 예시 |
|------|------|------|
| 구체적 | 명확한 수치와 근거 | "알코올 11% 이상인 와인의 평균 품질은 6.2점" |
| 실행 가능 | 행동으로 연결 가능 | "휘발성 산도를 0.4 이하로 관리하면 품질 향상" |
| 데이터 기반 | 통계적 뒷받침 | "t-검정 결과 p < 0.001로 유의미" |

### Wine Quality 데이터 핵심 발견

- 알코올 도수가 가장 중요한 양의 요인
- 휘발성 산도가 가장 중요한 음의 요인
- 고품질 와인 = 높은 알코올 + 낮은 휘발성 산도
- 대부분 5~6점에 집중, 극단적 품질은 드묾
