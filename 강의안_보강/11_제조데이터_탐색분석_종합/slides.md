---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 11차시'
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
  .highlight { background-color: #fef3c7; padding: 10px; border-radius: 8px; }
  .important { background-color: #fee2e2; padding: 10px; border-radius: 8px; }
  .tip { background-color: #d1fae5; padding: 10px; border-radius: 8px; }
---

# 제조 데이터 탐색 분석 종합

## 11차시 | Part II. 기초 수리와 데이터 분석

**Part II 마무리: EDA 전체 워크플로우와 종합 프로젝트**

---

# Part II 학습 여정

## 5~9차시 복습

| 차시 | 주제 | 핵심 개념 |
|:----:|------|----------|
| 5 | 기초 기술통계량과 시각화 | 평균, 중앙값, 히스토그램, 박스플롯 |
| 6 | 확률분포와 품질 검정 | 정규분포, Z-score, t-검정 |
| 7 | 상관분석과 예측의 기초 | 상관계수, 선형회귀, R² |
| 8 | 제조 데이터 전처리 (1) | 결측치, 이상치, IQR |
| 9 | 제조 데이터 전처리 (2) | 스케일링, 인코딩, Pipeline |

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **EDA의 전체 흐름**을 이해한다 |
| 2 | **데이터에서 인사이트를 도출**한다 |
| 3 | **제조 데이터를 종합적으로 분석**한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  EDA 전체   │ →  │  인사이트   │ → │  종합 분석  │
│    흐름     │    │    도출     │    │  프로젝트   │
└─────────────┘    └─────────────┘    └─────────────┘
   5단계 흐름        패턴 발견         실전 EDA
   체크리스트        가설 수립         대시보드
```

---

<!-- _class: lead -->

# Part 1

## EDA의 전체 흐름 이해

---

# EDA란?

## Exploratory Data Analysis (탐색적 데이터 분석)

> 데이터를 **탐색**하여 패턴, 이상치, 관계를 **발견**하는 과정

### EDA의 목적
1. **데이터 이해**: 어떤 데이터인가?
2. **품질 확인**: 결측치, 이상치, 오류는?
3. **패턴 발견**: 숨겨진 관계가 있는가?
4. **가설 수립**: 모델링 방향은?

---

# EDA vs 확증적 분석

## 두 가지 분석 접근법

| 구분 | EDA (탐색적) | CDA (확증적) |
|------|-------------|--------------|
| 목적 | 가설 **생성** | 가설 **검증** |
| 방법 | 시각화, 요약 | 통계적 검정 |
| 질문 | "무엇이 있나?" | "가설이 맞나?" |
| 순서 | **먼저** | 나중에 |

<div class="tip">

**분석 순서**: EDA → 가설 수립 → 확증적 분석 → 결론

</div>

---

# EDA 5단계

## 체계적 접근법

```
┌─────────────────────────────────┐
│ 1. 데이터 개요 파악             │
│    shape, dtypes, info, head    │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 2. 단변량 분석                  │
│    각 변수 개별 분포 확인       │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 3. 이변량 분석                  │
│    두 변수 간 관계 확인         │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 4. 다변량 분석                  │
│    여러 변수 동시 분석          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 5. 인사이트 도출                │
│    발견 정리, 가설 수립         │
└─────────────────────────────────┘
```

---

# 1단계: 데이터 개요 파악

## 데이터와 첫 만남

```python
# 기본 정보 확인
df.shape              # (행 수, 열 수)
df.dtypes             # 각 컬럼 데이터 타입
df.columns            # 컬럼 이름 목록
df.head(10)           # 처음 10행 확인
df.tail(5)            # 마지막 5행 확인
df.info()             # 요약 정보 (결측치 포함)
df.describe()         # 수치형 기술통계
df.describe(include='object')  # 범주형 요약
```

---

# 1단계: 핵심 질문

## 이 단계에서 확인할 것

| 질문 | 확인 방법 |
|------|----------|
| 데이터 크기는? | `df.shape` |
| 어떤 변수들이 있나? | `df.columns`, `df.dtypes` |
| 결측치가 있나? | `df.isnull().sum()` |
| 중복 행이 있나? | `df.duplicated().sum()` |
| 데이터 범위는 적절한가? | `df.describe()` |

---

# 2단계: 단변량 분석

## 각 변수 개별 분석

### 수치형 변수
```python
# 분포 확인
df['column'].hist(bins=30)
df['column'].plot(kind='box')

# 기술통계
df['column'].describe()
df['column'].value_counts(bins=10)
```

### 범주형 변수
```python
# 빈도 확인
df['column'].value_counts()
df['column'].value_counts(normalize=True)  # 비율
```

---

# 단변량 분석: 시각화

## 변수 유형별 그래프

| 변수 유형 | 그래프 | 확인 내용 |
|----------|--------|----------|
| 수치형 | 히스토그램 | 분포 형태 (정규? 왜도?) |
| 수치형 | 박스플롯 | 이상치, 사분위수 |
| 수치형 | KDE | 밀도 분포 |
| 범주형 | 막대그래프 | 범주별 빈도 |
| 범주형 | 파이차트 | 비율 (3~5개 범주 이하) |

---

# 단변량 분석: 체크포인트

## 각 변수에서 확인할 것

<div class="highlight">

### 수치형 변수
- [ ] 분포 형태 (정규분포? 왜도?)
- [ ] 중심 경향 (평균 vs 중앙값 차이)
- [ ] 이상치 존재 여부
- [ ] 결측치 비율

### 범주형 변수
- [ ] 고유값 개수
- [ ] 불균형 여부 (특정 범주 편중)
- [ ] 결측치 비율

</div>

---

# 3단계: 이변량 분석

## 두 변수 간 관계 분석

### 변수 조합별 분석 방법

| 조합 | 시각화 | 통계 |
|------|--------|------|
| 수치 vs 수치 | 산점도 | 상관계수 |
| 범주 vs 수치 | 그룹별 박스플롯 | t-검정, ANOVA |
| 범주 vs 범주 | 스택드 바차트, 히트맵 | 교차표, 카이제곱 |

---

# 이변량 분석: 수치 vs 수치

## 상관관계 분석

```python
# 산점도
plt.scatter(df['온도'], df['불량률'])
plt.xlabel('온도')
plt.ylabel('불량률')

# 상관계수
r = df['온도'].corr(df['불량률'])
print(f"상관계수: {r:.4f}")

# 상관행렬 히트맵
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

---

# 이변량 분석: 범주 vs 수치

## 그룹별 비교

```python
# 그룹별 박스플롯
df.boxplot(column='불량률', by='라인')

# 그룹별 평균 비교
df.groupby('라인')['불량률'].agg(['mean', 'std', 'count'])

# t-검정 (두 그룹 비교)
from scipy import stats
group_a = df[df['라인'] == 'A']['불량률']
group_b = df[df['라인'] == 'B']['불량률']
t_stat, p_value = stats.ttest_ind(group_a, group_b)
```

---

# 이변량 분석: 범주 vs 범주

## 교차표 분석

```python
# 교차표 생성
cross_tab = pd.crosstab(df['라인'], df['불량유형'])

# 비율로 변환
cross_tab_pct = pd.crosstab(df['라인'], df['불량유형'],
                            normalize='index') * 100

# 히트맵
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
```

---

# 4단계: 다변량 분석

## 여러 변수 동시 분석

```python
# 상관행렬 (전체)
corr_matrix = df.corr()

# 피벗 테이블
pivot = df.pivot_table(
    values='불량률',
    index='라인',
    columns='시간대',
    aggfunc='mean'
)

# 그룹별 다중 집계
df.groupby(['라인', '시간대']).agg({
    '불량률': ['mean', 'std'],
    '생산량': 'sum'
})
```

---

# 다변량 분석: 시각화

## 복합 시각화 기법

| 기법 | 설명 | 사용 시점 |
|------|------|----------|
| **Pairplot** | 모든 변수 쌍 산점도 | 변수 관계 전체 탐색 |
| **Heatmap** | 상관행렬 시각화 | 상관관계 패턴 |
| **Facet Grid** | 범주별 분리 그래프 | 그룹별 패턴 비교 |
| **3D 산점도** | 3개 변수 동시 | 복잡한 관계 탐색 |

---

# EDA 체크리스트

## 빠뜨리지 말아야 할 것

<div class="important">

### 필수 체크 항목
- [ ] 데이터 크기와 구조 확인
- [ ] 결측치 현황 및 처리 방안
- [ ] 이상치 현황 및 처리 방안
- [ ] 각 변수의 분포 형태
- [ ] **타겟 변수**와 다른 변수들의 관계
- [ ] 변수 간 상관관계 (다중공선성)
- [ ] 시간적 패턴 (시계열인 경우)
- [ ] 그룹별 차이 (범주형 변수 존재 시)

</div>

---

<!-- _class: lead -->

# Part 2

## 데이터에서 인사이트 도출

---

# 인사이트란?

## 데이터에서 발견한 의미 있는 패턴

> **인사이트** = 데이터 + 해석 + 실행 가능성

### 좋은 인사이트의 3요소
1. **구체적**: 명확한 수치와 근거
2. **실행 가능**: 행동으로 연결 가능
3. **데이터 기반**: 통계적 뒷받침

---

# 좋은 인사이트 vs 나쁜 인사이트

## 예시 비교

| 특징 | 나쁜 인사이트 | 좋은 인사이트 |
|------|-------------|--------------|
| 구체성 | "불량률이 높다" | "라인 B 불량률이 평균(3.2%) 대비 **5.1%로 1.9%p 높음**" |
| 실행 가능 | "품질을 개선해야 한다" | "**온도 90°C 이상**에서 경고 알람 설정 권장" |
| 데이터 기반 | "아마 그럴 것 같다" | "상관계수 **r=0.42**, **p<0.05**로 통계적 유의" |

---

# 패턴 발견 방법

## 데이터에서 패턴 찾기

### 1. 시각적 패턴
- 그래프에서 눈에 띄는 형태
- 군집, 트렌드, 이상점

### 2. 통계적 패턴
- 유의미한 상관관계
- 그룹 간 유의미한 차이
- 분포의 특이성

### 3. 도메인 패턴
- 제조 현장 지식과 연결
- 물리적/화학적 원리 반영

---

# 인사이트 도출 프레임워크

## 체계적 접근

```
1. 패턴 발견
   "라인 B의 불량률이 높게 나타남"
        ↓
2. 수치화
   "평균 5.1% vs 전체 평균 3.2% (+1.9%p)"
        ↓
3. 통계적 검증
   "t-검정 결과 p-value = 0.003 (유의)"
        ↓
4. 원인 추론
   "라인 B는 노후 장비, 오후 가동 多"
        ↓
5. 행동 제안
   "라인 B 장비 점검, 온도 모니터링 강화"
```

---

# 인사이트 유형

## 제조 분야 주요 인사이트

| 유형 | 예시 | 활용 |
|------|------|------|
| **영향 요인** | "온도가 불량률의 42% 설명" | 관리 우선순위 |
| **임계점** | "85°C 초과 시 불량률 급증" | 경계값 설정 |
| **그룹 차이** | "A라인 vs B라인 유의미 차이" | 개선 대상 선정 |
| **시간 패턴** | "오후 불량률 15% 높음" | 점검 시간 조정 |
| **조합 효과** | "고온+고습 시 불량률 3배" | 복합 관리 |

---

# 가설 수립

## 인사이트에서 가설로

### 가설의 형태
```
"[원인]이/가 [결과]에 영향을 미친다"

예시:
- "온도 상승이 불량률 증가에 영향을 미친다"
- "오후 시간대에 불량률이 더 높다"
- "라인 B의 불량률이 다른 라인보다 높다"
```

### 가설의 조건
- **측정 가능**: 데이터로 검증 가능
- **반증 가능**: 틀릴 수 있어야 함
- **구체적**: 모호하지 않음

---

# 분석 결과 정리

## 인사이트 문서화

### 표준 형식
```
[인사이트 제목]
예: "라인 B 불량률 이상 현상"

[발견 내용]
라인 B의 불량률이 다른 라인 대비 높음

[수치적 근거]
- 라인 B: 5.1%
- 다른 라인 평균: 3.2%
- 차이: +1.9%p

[통계적 검증]
- t-검정 p-value: 0.003 (유의수준 0.05 미만)

[권고 사항]
- 라인 B 장비 점검 실시
- 온도 센서 추가 설치 검토
```

---

# 시각화로 인사이트 전달

## 효과적인 시각화 원칙

<div class="highlight">

### 시각화 3원칙
1. **단순화**: 핵심 메시지 하나에 집중
2. **강조**: 중요한 부분 하이라이트
3. **맥락**: 비교 기준 제공 (평균선 등)

### 피해야 할 것
- 3D 그래프 (왜곡 발생)
- 너무 많은 정보 (복잡함)
- 범례 없는 그래프

</div>

---

# 인사이트 우선순위

## 중요도 평가

| 기준 | 높음 | 낮음 |
|------|------|------|
| 영향력 | 타겟에 큰 영향 | 영향 미미 |
| 실행 가능성 | 바로 적용 가능 | 적용 어려움 |
| 신뢰도 | 통계적 유의 | 불확실 |
| 비용 | 개선 비용 낮음 | 비용 높음 |

<div class="tip">

**우선순위 공식**: 영향력 × 실행가능성 × 신뢰도 / 비용

</div>

---

<!-- _class: lead -->

# Part 3

## 제조 데이터 종합 분석 프로젝트

---

# 실습 시나리오

## 품질관리팀 분석 요청

```
📋 분석 요청서

요청 부서: 품질관리팀
요청 일자: 2025-01-05

요청 내용:
"최근 불량률이 증가하고 있습니다.
 어떤 요인이 불량에 영향을 미치는지 분석해주세요.
 특히 라인별, 시간대별 차이가 있는지 확인 부탁드립니다."

제공 데이터:
- 기간: 최근 6개월 생산 데이터
- 변수: 날짜, 라인, 시간대, 온도, 습도, 생산량, 불량수
```

---

# 실습 데이터

## 제조 품질 데이터 구조

| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| date | 생산 일자 | datetime |
| line | 생산 라인 (A, B, C) | category |
| shift | 근무 시간대 (주간, 야간) | category |
| temperature | 공정 온도 (°C) | float |
| humidity | 습도 (%) | float |
| production | 생산량 | int |
| defect_count | 불량 개수 | int |
| defect_rate | 불량률 (%) | float |

---

# EDA 워크플로우

## 실습 진행 순서

```
1단계: 데이터 로드 및 개요
    ↓
2단계: 데이터 품질 확인 (결측치, 이상치)
    ↓
3단계: 단변량 분석 (각 변수 분포)
    ↓
4단계: 이변량 분석 (타겟과의 관계)
    ↓
5단계: 다변량 분석 (복합 요인)
    ↓
6단계: 인사이트 정리 및 시각화
    ↓
7단계: 보고서 작성
```

---

# 실습 1: 데이터 로드 및 개요

## 첫인상 파악

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('manufacturing_data.csv')

# 기본 정보
print(f"데이터 크기: {df.shape}")
print(f"\n컬럼 정보:\n{df.dtypes}")
print(f"\n처음 5행:\n{df.head()}")
print(f"\n기술통계:\n{df.describe()}")
```

---

# 실습 2: 데이터 품질 확인

## 결측치 및 이상치

```python
# 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())
print(f"\n결측치 비율:\n{df.isnull().sum() / len(df) * 100}")

# 이상치 확인 (IQR 방법)
for col in ['temperature', 'humidity', 'defect_rate']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) |
                (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"{col} 이상치: {outliers}개")
```

---

# 실습 3: 단변량 분석 - 불량률

## 타겟 변수 분포

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 히스토그램
axes[0].hist(df['defect_rate'], bins=30, edgecolor='black')
axes[0].axvline(df['defect_rate'].mean(), color='red',
                linestyle='--', label='평균')
axes[0].set_title('불량률 분포')
axes[0].legend()

# 박스플롯
axes[1].boxplot(df['defect_rate'])
axes[1].set_title('불량률 박스플롯')

# 시간 추이
df.groupby('date')['defect_rate'].mean().plot(ax=axes[2])
axes[2].set_title('일별 불량률 추이')
```

---

# 실습 4: 이변량 분석 - 라인별

## 라인별 불량률 비교

```python
# 그룹별 통계
line_stats = df.groupby('line')['defect_rate'].agg(
    ['mean', 'std', 'count']
)
print(line_stats)

# 그룹별 박스플롯
df.boxplot(column='defect_rate', by='line')
plt.title('라인별 불량률')
plt.suptitle('')
plt.show()

# t-검정: B라인 vs 나머지
from scipy import stats
b_line = df[df['line'] == 'B']['defect_rate']
other = df[df['line'] != 'B']['defect_rate']
t, p = stats.ttest_ind(b_line, other)
print(f"t-통계량: {t:.3f}, p-value: {p:.4f}")
```

---

# 실습 5: 이변량 분석 - 온도

## 온도와 불량률 관계

```python
# 산점도
plt.scatter(df['temperature'], df['defect_rate'], alpha=0.3)
plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')

# 추세선 추가
z = np.polyfit(df['temperature'], df['defect_rate'], 1)
p = np.poly1d(z)
plt.plot(df['temperature'].sort_values(),
         p(df['temperature'].sort_values()),
         "r--", label='추세선')

# 상관계수
r = df['temperature'].corr(df['defect_rate'])
plt.title(f'온도 vs 불량률 (r={r:.3f})')
plt.legend()
```

---

# 실습 6: 다변량 분석

## 상관행렬 및 피벗

```python
# 상관행렬
numeric_cols = ['temperature', 'humidity', 'production', 'defect_rate']
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('상관행렬')

# 피벗 테이블
pivot = df.pivot_table(
    values='defect_rate',
    index='line',
    columns='shift',
    aggfunc='mean'
)
print("=== 라인 x 시간대 평균 불량률 ===")
print(pivot.round(3))
```

---

# 실습 7: 복합 요인 분석

## 조건 조합별 분석

```python
# 고온 + 고습 조합
df['temp_high'] = df['temperature'] > df['temperature'].median()
df['humidity_high'] = df['humidity'] > df['humidity'].median()

# 조합별 불량률
combo_analysis = df.groupby(['temp_high', 'humidity_high'])\
                   ['defect_rate'].mean()
print("=== 온도/습도 조합별 불량률 ===")
print(combo_analysis)

# 시각화
combo_pivot = df.pivot_table(
    values='defect_rate',
    index='temp_high',
    columns='humidity_high',
    aggfunc='mean'
)
sns.heatmap(combo_pivot, annot=True, cmap='Reds', fmt='.3f')
```

---

# 실습 8: 인사이트 정리

## 주요 발견 사항

```
[분석 결과 요약]

1. 라인별 차이
   - 라인 B 불량률: 5.1% (평균 3.2% 대비 +1.9%p)
   - t-검정 p-value: 0.003 (통계적 유의)

2. 시간대별 차이
   - 야간: 4.2%, 주간: 2.8% (차이 1.4%p)

3. 온도 영향
   - 온도-불량률 상관계수: r = 0.42
   - 85°C 초과 시 불량률 급증

4. 복합 요인
   - 고온+고습 조합: 불량률 6.8% (최고)
   - 저온+저습 조합: 불량률 2.1% (최저)
```

---

# 실습 9: 대시보드 구성

## 한눈에 보는 분석 결과

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 불량률 분포
axes[0,0].hist(df['defect_rate'], bins=30, edgecolor='black')
axes[0,0].set_title('불량률 분포')

# 2. 라인별 불량률
df.boxplot(column='defect_rate', by='line', ax=axes[0,1])
axes[0,1].set_title('라인별 불량률')

# 3. 온도 vs 불량률
axes[1,0].scatter(df['temperature'], df['defect_rate'], alpha=0.3)
axes[1,0].set_title('온도 vs 불량률')

# 4. 상관행렬 히트맵
sns.heatmap(corr, annot=True, ax=axes[1,1])
axes[1,1].set_title('상관행렬')

plt.suptitle('제조 품질 분석 대시보드', fontsize=16)
plt.tight_layout()
```

---

# 실습 10: 보고서 작성

## EDA 결과 보고서 구조

```
┌─────────────────────────────────────┐
│ 1. 분석 개요                        │
│    - 분석 목적, 데이터 설명          │
├─────────────────────────────────────┤
│ 2. 데이터 품질                      │
│    - 결측치/이상치 현황 및 처리      │
├─────────────────────────────────────┤
│ 3. 주요 발견 사항                   │
│    - 핵심 인사이트 3~5개            │
│    - 시각화 자료                    │
├─────────────────────────────────────┤
│ 4. 결론 및 권고사항                 │
│    - 실행 가능한 제안               │
│    - 추가 분석 필요 사항            │
└─────────────────────────────────────┘
```

---

# 권고사항 예시

## 분석 결과 기반 제안

<div class="tip">

### 즉시 실행 (Quick Win)
1. **라인 B 긴급 점검**: 장비 상태 확인
2. **온도 경계값 설정**: 85°C 초과 시 알람

### 단기 개선
3. **야간 품질 모니터링 강화**
4. **고온+고습 환경 관리 개선**

### 장기 개선
5. **예측 모델 개발**: 불량 사전 예측
6. **센서 추가 설치**: 실시간 모니터링

</div>

---

# 실습 정리

## EDA 핵심 체크포인트

### 완료 확인
- [ ] 데이터 개요 파악 완료
- [ ] 결측치/이상치 확인 및 처리
- [ ] 타겟 변수 분포 분석
- [ ] 주요 변수와 타겟 관계 분석
- [ ] 그룹별 차이 통계 검정
- [ ] 복합 요인 분석
- [ ] 인사이트 3개 이상 도출
- [ ] 대시보드 시각화 완료

---

# Part II 종합 정리

## 5~11차시 핵심 개념

| 역량 | 학습 내용 | 활용 |
|------|----------|------|
| **기술통계** | 평균, 중앙값, 표준편차 | 데이터 요약 |
| **시각화** | 히스토그램, 박스플롯 | 분포 확인 |
| **통계검정** | t-검정, Z-score | 차이 검증 |
| **상관분석** | 상관계수, 회귀 | 관계 파악 |
| **전처리** | 결측치, 이상치, 스케일링 | 데이터 정제 |
| **EDA** | 5단계 프로세스 | 인사이트 도출 |

---

# 다음 차시 예고

## Part III 시작: 머신러닝

### 11차시: 머신러닝 소개와 문제 유형

- 머신러닝이란?
- 지도학습 vs 비지도학습
- 분류 vs 회귀
- sklearn 기본 사용법

<div class="highlight">

**Part II 역량** + **머신러닝** = **예측 모델 구축**

</div>

---

# 감사합니다

## 11차시: 제조 데이터 탐색 분석 종합

**Part II 완료! 수고하셨습니다!**

다음 파트에서 본격적인 AI 모델링을 시작합니다!
