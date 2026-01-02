---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 9차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
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
---

# 제조 데이터 탐색 분석 종합

## 9차시 | Part II. 기초 수리와 데이터 분석

**Part II 마무리: EDA 전체 워크플로우**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **EDA의 전체 흐름**을 이해한다
2. 데이터에서 **인사이트를 도출**한다
3. **제조 데이터**를 종합적으로 분석한다

---

# EDA란?

## Exploratory Data Analysis (탐색적 데이터 분석)

> 데이터를 **탐색**하여 패턴, 이상치, 관계를 **발견**하는 과정

### 목적
1. 데이터 이해
2. 가설 수립
3. 모델링 방향 설정
4. 데이터 품질 확인

---

# EDA 5단계

## 체계적 접근

```
1. 데이터 로드 및 개요 파악
   ↓
2. 단변량 분석 (각 변수 개별 분석)
   ↓
3. 이변량 분석 (변수 간 관계)
   ↓
4. 다변량 분석 (여러 변수 동시)
   ↓
5. 인사이트 정리 및 가설 수립
```

---

# 1단계: 데이터 개요

## 첫인상 파악

```python
# 기본 정보 확인
df.shape           # 크기 (행, 열)
df.dtypes          # 데이터 타입
df.head()          # 샘플 확인
df.info()          # 요약 정보
df.describe()      # 기술통계
df.isnull().sum()  # 결측치 현황
```

> 데이터와 처음 만나면 "인사"부터!

---

# 2단계: 단변량 분석

## 각 변수 개별 분석

### 수치형 변수
- **히스토그램**: 분포 형태
- **상자그림**: 이상치, 범위
- **기술통계**: 평균, 중앙값, 표준편차

### 범주형 변수
- **빈도표**: value_counts()
- **막대그래프**: 범주별 빈도

---

# 3단계: 이변량 분석

## 두 변수 관계

### 수치 vs 수치
- 산점도
- 상관계수

### 범주 vs 수치
- 그룹별 상자그림
- 그룹별 평균 비교

### 범주 vs 범주
- 교차표 (crosstab)
- 히트맵

---

# 4단계: 다변량 분석

## 여러 변수 동시 분석

```python
# 상관행렬
df.corr()

# 피벗 테이블
df.pivot_table(values='불량률',
               index='라인',
               columns='시간대',
               aggfunc='mean')

# 그룹별 집계
df.groupby(['라인', '시간대'])['불량률'].mean()
```

---

# 5단계: 인사이트 도출

## 발견한 패턴 정리

### 좋은 인사이트의 특징

| 특징 | 나쁜 예 | 좋은 예 |
|------|---------|---------|
| 구체적 | "불량률이 높다" | "라인 B 불량률이 평균 대비 2%p 높음" |
| 실행 가능 | "품질을 개선해야 한다" | "온도 85도 이상 시 경고 알람 설정" |
| 데이터 기반 | "아마 그럴 것 같다" | "상관계수 0.42, p<0.05로 유의" |

---

# EDA 체크리스트

## 빠뜨리지 않기

- [ ] 결측치 확인 및 처리 방안
- [ ] 이상치 확인 및 처리 방안
- [ ] 각 변수의 분포 확인
- [ ] 타겟 변수와의 관계 확인
- [ ] 변수 간 상관관계 확인
- [ ] 시간적 패턴 확인 (시계열인 경우)
- [ ] 그룹별 차이 확인

---

# 이론 정리

## Part II 핵심 복습

| 차시 | 주제 | 핵심 도구 |
|------|------|----------|
| 4 | 데이터 요약과 시각화 | Matplotlib |
| 5 | 확률분포와 품질 검정 | scipy.stats, t-test |
| 6 | 상관분석과 예측의 기초 | corrcoef, LinearRegression |
| 7-8 | 제조 데이터 전처리 | 결측치, 이상치, 스케일링 |
| 9 | EDA 종합 | 전체 워크플로우 |

---

# - 실습편 -

## 9차시

**제조 품질 데이터 종합 분석**

---

# 실습 개요

## 시나리오

```
품질관리팀 요청:
"최근 불량률이 증가하고 있습니다.
 어떤 요인이 불량에 영향을 미치는지 분석해주세요."

데이터:
- 6개월 간 생산 데이터 (1,000건)
- 변수: 날짜, 라인, 시간대, 온도, 습도, 생산량, 불량수
```

---

# 실습 환경

## 라이브러리 준비

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 로드 및 개요

## 첫인상 파악

```python
# 데이터 로드 (실습에서는 생성)
print(f"데이터 크기: {df.shape}")
print(df.info())
print(df.describe())

# 결측치 확인
print("결측치 현황:")
print(df.isnull().sum())
```

---

# 실습 2: 단변량 분석

## 불량률 분포

```python
# 히스토그램
plt.hist(df['불량률']*100, bins=30, edgecolor='black')
plt.axvline(df['불량률'].mean()*100, color='red',
            linestyle='--', label='평균')
plt.xlabel('불량률 (%)')
plt.ylabel('빈도')
plt.title('불량률 분포')
plt.legend()
plt.show()
```

---

# 실습 3: 범주별 분석

## 라인별, 시간대별 불량률

```python
# 라인별 평균 불량률
line_defect = df.groupby('라인')['불량률'].mean() * 100
print(line_defect)

# 막대그래프
line_defect.plot(kind='bar', color='steelblue')
plt.ylabel('불량률 (%)')
plt.title('라인별 평균 불량률')
plt.show()
```

---

# 실습 4: 상관관계 분석

## 온도와 불량률

```python
# 산점도
plt.scatter(df['온도'], df['불량률']*100, alpha=0.3)
plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도 vs 불량률')

# 상관계수
r = df['온도'].corr(df['불량률'])
print(f"상관계수: {r:.4f}")
```

---

# 실습 5: 다변량 분석

## 라인 x 시간대 피벗

```python
# 피벗 테이블
pivot = df.pivot_table(
    values='불량률',
    index='라인',
    columns='시간대',
    aggfunc='mean'
) * 100

print(pivot.round(2))
```

> 라인 B + 오후 조합이 가장 높은가?

---

# 실습 6: 통계적 검정

## 라인 B vs 다른 라인

```python
from scipy import stats

line_b = df[df['라인'] == 'B']['불량률']
line_other = df[df['라인'] != 'B']['불량률']

t_stat, p_value = stats.ttest_ind(line_b, line_other)
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("→ 라인 B의 불량률이 통계적으로 유의미하게 다름")
```

---

# 실습 7: 인사이트 정리

## 분석 결과 요약

```
[주요 발견사항]

1. 라인별 차이
   - 라인 B 불량률이 평균 대비 약 X%p 높음 (p < 0.05)

2. 시간대별 차이
   - 오후 불량률이 오전보다 약 X%p 높음

3. 온도 영향
   - 온도와 불량률 상관계수: 0.XX
   - 85°C 이상에서 불량률 증가 경향
```

---

# 실습 8: 종합 대시보드

## 한눈에 보는 분석 결과

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 불량률 분포
# 2. 라인별 불량률
# 3. 온도 vs 불량률
# 4. 라인 x 시간대 히트맵

plt.suptitle('제조 품질 분석 대시보드')
plt.tight_layout()
plt.show()
```

---

# EDA 결과 보고서 구조

## 표준 형식

```
1. 분석 개요
   - 데이터 설명, 분석 목적

2. 데이터 품질 확인
   - 결측치, 이상치 현황

3. 주요 발견 사항
   - 핵심 인사이트 (3~5개)
   - 시각화 자료

4. 결론 및 권고사항
   - 실행 가능한 제안
```

---

# 실습 정리

## 핵심 체크포인트

### EDA 5단계
- [ ] 1단계: 데이터 개요 (shape, info, describe)
- [ ] 2단계: 단변량 분석 (분포, 이상치)
- [ ] 3단계: 이변량 분석 (상관관계, 그룹 비교)
- [ ] 4단계: 다변량 분석 (피벗, 히트맵)
- [ ] 5단계: 인사이트 도출

---

# 다음 차시 예고

## Part III 시작: 머신러닝

### 10차시: 머신러닝 소개와 문제 유형

- 머신러닝이란?
- 지도학습 vs 비지도학습
- 분류 vs 회귀
- sklearn 모델링 흐름

> Part II에서 익힌 데이터 분석 능력이 ML의 기초!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **EDA 5단계**: 개요 → 단변량 → 이변량 → 다변량 → 인사이트
2. **좋은 인사이트**: 구체적, 실행 가능, 데이터 기반
3. **체계적 접근**: 체크리스트 활용

### Part II 완료!
- 4~9차시: 통계, 시각화, 전처리, EDA
- 다음 파트에서 본격적인 AI 모델링 시작

---

# 감사합니다

## 9차시: 제조 데이터 탐색 분석 종합

**Part II 완료! 수고하셨습니다!**
