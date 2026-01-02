---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 5차시'
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

# 확률분포와 품질 검정

## 5차시 | Part II. 기초 수리와 데이터 분석

**정규분포로 제조 품질 관리하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **정규분포**의 개념과 68-95-99.7 규칙을 이해한다
2. **Z-score**를 활용하여 이상치를 탐지한다
3. **t-검정**으로 두 그룹의 차이를 검정한다

---

# 왜 확률분포를 알아야 하는가?

## 제조 현장의 질문들

- "이 측정값이 정상 범위인가?"
- "두 라인의 품질 차이가 실제로 있는가?"
- "이 생산량이 이상치인가?"

> 확률분포를 이해하면 **데이터 기반 의사결정**이 가능

---

# 정규분포 (Normal Distribution)

## 가장 중요한 확률분포

### 특성
- 평균을 중심으로 **대칭인 종 모양**
- 평균에서 멀어질수록 확률 감소
- 대부분의 자연현상과 제조 데이터가 정규분포

```
평균(μ) = 데이터의 중심
표준편차(σ) = 데이터의 퍼짐 정도
```

---

# 68-95-99.7 규칙

## 표준편차와 데이터 분포

```
       ← 99.7% (±3σ) →
     ←── 95% (±2σ) ──→
      ←─ 68% (±1σ) ─→

          ▄▄▄
        ▄█████▄
      ▄█████████▄
   ▄▄███████████████▄▄
━━━━━━━━━━━━━━━━━━━━━━━━
  μ-3σ μ-2σ μ-σ  μ  μ+σ μ+2σ μ+3σ
```

> 평균 ±3σ 밖의 데이터는 **0.3%** 미만 (이상치 의심)

---

# 정규분포 활용 예시

## 품질 관리 기준 설정

```python
# 생산량이 정규분포 N(1200, 50²)를 따를 때

평균(μ) = 1200
표준편차(σ) = 50

# 68%: 1150 ~ 1250 범위
# 95%: 1100 ~ 1300 범위
# 99.7%: 1050 ~ 1350 범위

# 관리 기준 예시
# 1050 미만 또는 1350 초과 → 이상치!
```

---

# Z-score (표준점수)

## 이상치 탐지의 핵심 도구

```python
Z-score = (관측값 - 평균) / 표준편차
```

### Z-score 해석
| Z-score 범위 | 의미 | 조치 |
|-------------|------|------|
| \|z\| ≤ 1 | 정상 (68%) | - |
| 1 < \|z\| ≤ 2 | 주의 (27%) | 모니터링 |
| 2 < \|z\| ≤ 3 | 경고 (4.3%) | 점검 필요 |
| \|z\| > 3 | 이상치 (0.3%) | 즉시 조사 |

---

# Z-score 계산 예시

## 생산량 이상치 탐지

```python
import numpy as np

# 일주일 생산량 데이터
production = np.array([1200, 1180, 1210, 1195, 1400, 1205, 1190])

mean = np.mean(production)  # 1225.7
std = np.std(production)    # 70.8

# Z-score 계산
z_scores = (production - mean) / std
# [−0.36, −0.65, −0.22, −0.43, 2.46, −0.29, −0.50]

# 이상치 탐지 (|z| > 2)
outliers = production[np.abs(z_scores) > 2]
# [1400] → 5일차 생산량이 이상치!
```

---

# 가설검정 기초

## 데이터로 의사결정하기

### 제조 현장의 질문
> "새 공정이 불량률을 줄였을까?"

### 가설 설정
```
귀무가설(H₀): 차이가 없다
  → "새 공정은 불량률에 영향 없음"

대립가설(H₁): 차이가 있다
  → "새 공정은 불량률을 줄임"
```

---

# p-value란?

## 관측 결과가 우연일 확률

```
p-value = 귀무가설이 참일 때,
          관측된 차이가 우연히 나타날 확률
```

### 판단 기준 (유의수준 α = 0.05)
- **p < 0.05**: 통계적으로 유의미 → 차이가 있다!
- **p ≥ 0.05**: 유의미하지 않음 → 차이 없다고 볼 수 없음

> p-value가 작을수록 차이가 "진짜"일 가능성 높음

---

# t-검정 (t-test)

## 두 그룹의 평균 비교

```python
from scipy import stats

# 라인 A와 라인 B의 불량률 (%)
line_a = [2.1, 2.3, 2.0, 2.2, 2.4]
line_b = [2.8, 3.0, 2.7, 2.9, 3.1]

# 독립표본 t-검정
t_stat, p_value = stats.ttest_ind(line_a, line_b)

print(f"t-통계량: {t_stat:.3f}")  # -6.708
print(f"p-value: {p_value:.4f}")  # 0.0001
```

> p < 0.05이므로 두 라인의 불량률 차이는 **유의미**

---

# 이론 정리

## 핵심 포인트

### 정규분포
- **68-95-99.7 규칙**: 표준편차로 데이터 분포 파악
- **품질 기준**: 평균 ±3σ 벗어나면 이상치

### 품질 검정
- **Z-score**: 개별 값의 이상 여부 판단
- **t-검정**: 두 그룹 평균의 차이 검정
- **p-value < 0.05**: 통계적으로 유의미

---

# - 실습편 -

## 5차시

**제조 데이터 품질 검정 실습**

---

# 실습 개요

## 정규분포와 품질 검정

### 실습 목표
- 정규분포 시각화
- Z-score로 이상치 탐지
- t-검정으로 라인 비교

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 정규분포 시각화

## 생산량 분포 확인

```python
np.random.seed(42)

# 정규분포 데이터 생성 (평균=1200, 표준편차=50)
production = np.random.normal(1200, 50, 1000)

plt.figure(figsize=(10, 6))
plt.hist(production, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(1200, color='red', linestyle='--', label='평균')
plt.axvline(1200-50, color='orange', linestyle=':', label='-1σ')
plt.axvline(1200+50, color='orange', linestyle=':',label='+1σ')
plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 정규분포 (μ=1200, σ=50)')
plt.legend()
plt.show()
```

---

# 실습 2: 68-95-99.7 규칙 확인

## 범위별 데이터 비율

```python
mean = np.mean(production)
std = np.std(production)

# 각 범위의 데이터 비율 계산
within_1std = np.sum((production >= mean - std) &
                      (production <= mean + std)) / len(production)
within_2std = np.sum((production >= mean - 2*std) &
                      (production <= mean + 2*std)) / len(production)
within_3std = np.sum((production >= mean - 3*std) &
                      (production <= mean + 3*std)) / len(production)

print(f"±1σ 범위: {within_1std:.1%}")  # 약 68%
print(f"±2σ 범위: {within_2std:.1%}")  # 약 95%
print(f"±3σ 범위: {within_3std:.1%}")  # 약 99.7%
```

---

# 실습 3: Z-score 이상치 탐지

## 일주일 생산 데이터 분석

```python
# 실제 생산 데이터 (하나의 이상치 포함)
daily_production = np.array([1185, 1210, 1195, 1180, 1420, 1200, 1190])
days = ['월', '화', '수', '목', '금', '토', '일']

# Z-score 계산
mean = daily_production.mean()
std = daily_production.std()
z_scores = (daily_production - mean) / std

# 결과 출력
for day, prod, z in zip(days, daily_production, z_scores):
    status = "이상치!" if abs(z) > 2 else "정상"
    print(f"{day}: {prod}개 (Z={z:.2f}) - {status}")
```

---

# 실습 4: 이상치 시각화

## Z-score 기준 색상 표시

```python
plt.figure(figsize=(10, 6))

colors = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
bars = plt.bar(days, daily_production, color=colors, edgecolor='black')

plt.axhline(mean, color='green', linestyle='--', label=f'평균: {mean:.0f}')
plt.axhline(mean + 2*std, color='orange', linestyle=':', label='+2σ')
plt.axhline(mean - 2*std, color='orange', linestyle=':', label='-2σ')

plt.xlabel('요일')
plt.ylabel('생산량')
plt.title('일별 생산량과 이상치 탐지')
plt.legend()
plt.show()
```

> 빨간 막대가 이상치 (Z-score > 2)

---

# 실습 5: 라인별 데이터 준비

## t-검정을 위한 데이터

```python
np.random.seed(123)

# 라인 A: 평균 불량률 2.2%
line_a_defect = np.random.normal(2.2, 0.3, 30)

# 라인 B: 평균 불량률 2.8%
line_b_defect = np.random.normal(2.8, 0.35, 30)

print("=== 라인별 불량률 통계 ===")
print(f"라인 A: 평균 {line_a_defect.mean():.2f}%, 표준편차 {line_a_defect.std():.2f}")
print(f"라인 B: 평균 {line_b_defect.mean():.2f}%, 표준편차 {line_b_defect.std():.2f}")
```

---

# 실습 6: t-검정 수행

## 두 라인의 불량률 차이 검정

```python
from scipy import stats

# 독립표본 t-검정
t_stat, p_value = stats.ttest_ind(line_a_defect, line_b_defect)

print("=== t-검정 결과 ===")
print(f"t-통계량: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")
print()

if p_value < 0.05:
    print("결론: 두 라인의 불량률 차이가 통계적으로 유의미합니다.")
    print("      → 라인 B의 품질 개선이 필요합니다.")
else:
    print("결론: 두 라인의 불량률 차이가 유의미하지 않습니다.")
```

---

# 실습 7: 검정 결과 시각화

## 라인별 불량률 분포

```python
plt.figure(figsize=(10, 6))

plt.boxplot([line_a_defect, line_b_defect],
            labels=['라인 A', '라인 B'])

plt.ylabel('불량률 (%)')
plt.title(f'라인별 불량률 비교 (p-value: {p_value:.4f})')

# 유의성 표시
if p_value < 0.05:
    plt.annotate('* 유의미한 차이', xy=(1.5, max(line_b_defect)),
                 fontsize=12, color='red', ha='center')

plt.show()
```

---

# 실습 8: 종합 분석

## 품질 관리 리포트

```python
print("=" * 50)
print("           품질 관리 분석 리포트")
print("=" * 50)

# 이상치 현황
outlier_count = np.sum(np.abs(z_scores) > 2)
print(f"\n[이상치 탐지 결과]")
print(f"분석 기간: 7일")
print(f"이상치 발생: {outlier_count}건")

# 라인 비교
print(f"\n[라인별 품질 비교]")
print(f"라인 A 평균 불량률: {line_a_defect.mean():.2f}%")
print(f"라인 B 평균 불량률: {line_b_defect.mean():.2f}%")
print(f"차이 유의성: {'있음' if p_value < 0.05 else '없음'} (p={p_value:.4f})")

print("=" * 50)
```

---

# 실습 정리

## 핵심 체크포인트

### 정규분포
- [ ] 68-95-99.7 규칙으로 범위 파악
- [ ] 히스토그램으로 분포 시각화

### Z-score
- [ ] (값 - 평균) / 표준편차로 계산
- [ ] |Z| > 2면 이상치 의심

### t-검정
- [ ] stats.ttest_ind()로 두 그룹 비교
- [ ] p < 0.05면 차이가 유의미

---

# 다음 차시 예고

## 6차시: 상관분석과 예측의 기초

### 학습 내용
- 상관계수의 의미와 해석
- 두 변수 간의 관계 분석
- 단순선형회귀 소개

### 준비물
- 오늘 배운 코드 복습
- scipy, matplotlib 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심

1. **정규분포**: 68-95-99.7 규칙으로 품질 기준 설정
2. **Z-score**: 개별 데이터의 이상 여부 판단
3. **t-검정**: 두 그룹 간 차이의 통계적 유의성 검정

### 자주 하는 실수
- 표본 크기가 너무 작으면 t-검정 결과 신뢰도 낮음
- p-value는 "효과의 크기"가 아닌 "우연의 확률"

---

# 감사합니다

## 5차시: 확률분포와 품질 검정

**다음 시간에 상관분석을 배워봅시다!**
