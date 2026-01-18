---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 6차시'
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

# 확률분포와 품질검정 기초

## 6차시 | Part II. 기초 수리와 데이터 분석

**정규분포로 제조 품질 관리하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **정규분포**의 개념과 68-95-99.7 규칙을 이해한다
2. **Z-score**를 활용하여 이상치를 탐지한다

---

# 강의 구성

| 파트 | 대주제 | 시간 |
|:----:|--------|:----:|
| 1 | 정규분포의 개념과 68-95-99.7 규칙 | 15분 |
| 2 | Z-score를 활용한 이상치 탐지 | 15분 |

---

<!-- _class: lead -->

# Part 1
## 정규분포의 개념과 68-95-99.7 규칙

---

# 왜 확률분포를 알아야 하는가?

## 제조 현장의 질문들

| 상황 | 질문 |
|------|------|
| 품질 측정 | "이 측정값이 정상 범위인가?" |
| 라인 비교 | "두 라인의 품질 차이가 실제로 있는가?" |
| 생산 관리 | "오늘 생산량이 이상하게 높은데, 정말 이상한 건가?" |
| 공정 변경 | "새 공정이 불량률을 정말 줄였을까?" |

---

# 확률분포란?

## 데이터가 어떻게 흩어져 있는지 설명하는 함수

### 핵심 개념

```
확률분포 = 어떤 값이 나올 확률의 전체 패턴
```

### 왜 중요한가?
- **예측**: 앞으로 어떤 값이 나올지 예상 가능
- **판단**: 관측값이 정상인지 이상인지 판단
- **의사결정**: 감이 아닌 **데이터 기반 결정**

---

# 대표적인 확률분포

## 연속형 확률분포

| 분포 | 특징 | 예시 |
|------|------|------|
| **정규분포** | 종 모양, 평균 중심 대칭 | 품질 측정값, 생산량 |
| 균등분포 | 모든 값이 동일한 확률 | 무작위 추출 |
| 지수분포 | 사건 간 시간 간격 | 고장 발생 간격 |

## 이산형 확률분포

| 분포 | 특징 | 예시 |
|------|------|------|
| 이항분포 | 성공/실패 반복 | 불량품 개수 |
| 포아송분포 | 희귀 사건 빈도 | 시간당 결함 수 |

---

# 정규분포 (Normal Distribution)

## 가장 중요한 확률분포

### 특성
- 평균(μ)을 중심으로 **좌우 대칭인 종 모양**
- 평균에서 멀어질수록 확률 감소
- 평균과 표준편차 두 값으로 완전히 정의됨

### 표기법

```
X ~ N(μ, σ²)

μ (뮤) = 평균 = 데이터의 중심
σ (시그마) = 표준편차 = 데이터의 퍼짐 정도
```

---

# 정규분포의 모양

```
                    정규분포 곡선

                        ▲
                       █ █
                      █   █
                     █     █
                    █       █
                   █         █
                  █           █
                ▄█▄▄▄▄▄▄▄▄▄▄▄▄▄█▄
             ━━━━━━━━━━━━━━━━━━━━━━━━━
             μ-3σ  μ-2σ  μ-σ  μ  μ+σ  μ+2σ  μ+3σ
```

- 종 모양 (bell curve)
- 평균(μ)에서 가장 높고, 멀어질수록 낮아짐
- 좌우 대칭

---

# 정규분포가 중요한 이유

## 자연과 산업의 많은 현상이 정규분포를 따름

### 제조업 예시
| 측정항목 | 평균(μ) | 표준편차(σ) |
|---------|---------|-------------|
| 제품 무게 | 500g | 2g |
| 두께 | 1.0mm | 0.05mm |
| 조립 시간 | 45초 | 3초 |
| 일일 생산량 | 1,200개 | 50개 |

### 중심극한정리
> 표본의 크기가 충분히 크면, 표본 평균의 분포는
> 원래 분포와 관계없이 정규분포에 가까워진다

---

# 68-95-99.7 규칙

## 경험적 규칙 (Empirical Rule)

```
       ← ─────────── 99.7% (±3σ) ─────────── →
         ← ─────── 95% (±2σ) ─────── →
            ← ── 68% (±1σ) ── →

                    ▄▄▄
                  ▄█████▄
                ▄█████████▄
             ▄▄███████████████▄▄
          ━━━━━━━━━━━━━━━━━━━━━━━━━
          μ-3σ μ-2σ μ-σ  μ  μ+σ μ+2σ μ+3σ
```

---

# 68-95-99.7 규칙 상세

## 각 범위의 의미

| 범위 | 데이터 비율 | 의미 | 제조 관점 |
|------|------------|------|----------|
| μ ± 1σ | **68%** | 대부분 | 일반적인 변동 |
| μ ± 2σ | **95%** | 거의 전부 | 허용 범위 |
| μ ± 3σ | **99.7%** | 사실상 전부 | 품질 관리 한계 |

### 핵심 포인트
> **±3σ 밖에 있는 데이터는 0.3%도 안 됨**
> → 이 범위를 벗어나면 **이상치로 의심**

---

# 정규분포 활용 예시 (1)

## 생산량 관리

```python
# 일일 생산량이 정규분포 N(1200, 50²)를 따를 때

평균(μ) = 1200개
표준편차(σ) = 50개

# 68%: 1150 ~ 1250 범위 (정상 변동)
# 95%: 1100 ~ 1300 범위 (주의 범위)
# 99.7%: 1050 ~ 1350 범위 (관리 한계)
```

### 관리 기준 예시
- **1050 미만** → 생산량 부족 조사
- **1350 초과** → 과잉 생산 확인
- 양쪽 다 **이상치**로 점검 필요

---

# 정규분포 활용 예시 (2)

## 품질 측정값 관리

```python
# 제품 무게가 N(500, 2²)를 따를 때

평균(μ) = 500g
표준편차(σ) = 2g

# 규격: 500g ± 6g (3σ 기준)
# 하한: 494g
# 상한: 506g
```

### 품질 관리 활용

| 측정값 | 판정 | 조치 |
|--------|------|------|
| 498g | 정상 (μ-1σ 이내) | - |
| 493g | 이상 (3σ 초과) | 점검 필요 |
| 507g | 이상 (3σ 초과) | 즉시 조사 |

---

# 정규분포 시각화 (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 정규분포 데이터 생성 (평균=1200, 표준편차=50)
production = np.random.normal(1200, 50, 1000)

plt.figure(figsize=(10, 6))
plt.hist(production, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(1200, color='red', linestyle='--', label='평균')
plt.axvline(1150, color='orange', linestyle=':', label='-1σ')
plt.axvline(1250, color='orange', linestyle=':', label='+1σ')
plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 정규분포 (μ=1200, σ=50)')
plt.legend()
plt.show()
```

---

# 68-95-99.7 검증 코드

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

# Part 1 정리

## 정규분포와 68-95-99.7 규칙

### 핵심 개념
- **정규분포**: 평균 중심의 종 모양 분포
- **68-95-99.7 규칙**: 표준편차로 데이터 분포 파악

### 품질 관리 적용
- μ ± 1σ: 일상적인 변동
- μ ± 2σ: 주의 범위
- μ ± 3σ: 품질 관리 한계 (이상치 기준)

---

<!-- _class: lead -->

# Part 2
## Z-score를 활용한 이상치 탐지

---

# 이상치(Outlier)란?

## 다른 데이터와 동떨어진 값

### 제조업에서의 의미
| 상황 | 이상치 예시 | 원인 가능성 |
|------|------------|------------|
| 생산량 | 평소 1200, 오늘 1500 | 특근, 측정 오류, 기록 실수 |
| 품질 | 평균 500g, 측정 480g | 원료 문제, 설비 이상 |
| 불량률 | 평소 2%, 오늘 8% | 공정 이상, 원자재 불량 |

### 이상치 탐지가 중요한 이유
> 이상치는 **문제의 신호**일 수 있음
> 빠른 발견 → 빠른 대응 → 손실 최소화

---

# Z-score (표준점수)

## 이상치 탐지의 핵심 도구

### 정의

```python
Z-score = (관측값 - 평균) / 표준편차

Z = (X - μ) / σ
```

### 의미
- **Z = 0**: 평균과 같음
- **Z = 1**: 평균에서 1σ 떨어짐 (상위 16%)
- **Z = 2**: 평균에서 2σ 떨어짐 (상위 2.5%)
- **Z = 3**: 평균에서 3σ 떨어짐 (상위 0.15%)

---

# Z-score 해석 기준

## 절대값 기준 판단

| Z-score 범위 | 비율 | 해석 | 조치 |
|-------------|------|------|------|
| \|Z\| ≤ 1 | 68% | 정상 | 문제 없음 |
| 1 < \|Z\| ≤ 2 | 27% | 주의 | 모니터링 |
| 2 < \|Z\| ≤ 3 | 4.3% | 경고 | 점검 필요 |
| **\|Z\| > 3** | **0.3%** | **이상치** | **즉시 조사** |

### 실무 권장 기준
- **\|Z\| > 2**: 이상치 의심 (보수적)
- **\|Z\| > 3**: 이상치 확정 (일반적)

---

# Z-score 계산 예시

## 일주일 생산량 데이터

```python
import numpy as np

# 일주일 생산량 데이터
production = np.array([1200, 1180, 1210, 1195, 1400, 1205, 1190])
days = ['월', '화', '수', '목', '금', '토', '일']

# 통계량 계산
mean = np.mean(production)  # 1225.7
std = np.std(production)    # 70.8

# Z-score 계산
z_scores = (production - mean) / std

print(f"평균: {mean:.1f}, 표준편차: {std:.1f}")
for day, prod, z in zip(days, production, z_scores):
    print(f"{day}: {prod}개, Z={z:+.2f}")
```

---

# Z-score 계산 결과

## 이상치 탐지

| 요일 | 생산량 | Z-score | 판정 |
|------|--------|---------|------|
| 월 | 1,200 | -0.36 | 정상 |
| 화 | 1,180 | -0.65 | 정상 |
| 수 | 1,210 | -0.22 | 정상 |
| 목 | 1,195 | -0.43 | 정상 |
| **금** | **1,400** | **+2.46** | **이상치!** |
| 토 | 1,205 | -0.29 | 정상 |
| 일 | 1,190 | -0.50 | 정상 |

> 금요일 생산량이 Z > 2로 **이상치**

---

# NumPy로 Z-score 계산

## 벡터 연산의 장점

```python
import numpy as np

production = np.array([1200, 1180, 1210, 1195, 1400, 1205, 1190])

# 방법 1: 직접 계산
mean = np.mean(production)
std = np.std(production)
z_scores = (production - mean) / std

# 방법 2: scipy 활용
from scipy import stats
z_scores = stats.zscore(production)

# 이상치 마스크 (|Z| > 2)
outlier_mask = np.abs(z_scores) > 2
outliers = production[outlier_mask]
print(f"이상치: {outliers}")  # [1400]
```

---

# 이상치 탐지 함수

## 재사용 가능한 함수 만들기

```python
def detect_outliers_zscore(data, threshold=2):
    """
    Z-score 기반 이상치 탐지

    Parameters:
        data: 데이터 배열
        threshold: Z-score 임계값 (기본값 2)

    Returns:
        이상치 마스크, Z-scores
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    outlier_mask = np.abs(z_scores) > threshold
    return outlier_mask, z_scores
```

---

# 이상치 시각화

## 색상으로 이상치 표시

```python
import matplotlib.pyplot as plt

days = ['월', '화', '수', '목', '금', '토', '일']
production = np.array([1200, 1180, 1210, 1195, 1400, 1205, 1190])
outlier_mask, z_scores = detect_outliers_zscore(production)

# 색상 지정 (이상치: 빨강, 정상: 파랑)
colors = ['red' if is_outlier else 'steelblue'
          for is_outlier in outlier_mask]

plt.figure(figsize=(10, 6))
plt.bar(days, production, color=colors, edgecolor='black')
plt.axhline(np.mean(production), color='green', linestyle='--')
plt.xlabel('요일')
plt.ylabel('생산량')
plt.title('일별 생산량과 이상치 탐지')
plt.show()
```

---

# IQR vs Z-score 비교

## 두 가지 이상치 탐지 방법

| 비교 항목 | IQR 방식 | Z-score 방식 |
|----------|---------|-------------|
| **계산** | Q3-Q1 기반 | 평균/표준편차 기반 |
| **장점** | 이상치에 강건함 | 해석이 직관적 |
| **단점** | 정규분포 가정 불필요 | 정규분포 가정 필요 |
| **기준** | 1.5×IQR 밖 | \|Z\| > 2 또는 3 |
| **적합** | 비대칭 분포 | 대칭 분포 |

### 권장
- **정규분포에 가까우면**: Z-score
- **분포 형태가 불확실하면**: IQR

---

# 양쪽 방법 비교 코드

```python
def compare_outlier_methods(data):
    """두 방법으로 이상치 비교"""

    # Z-score 방법
    z_scores = stats.zscore(data)
    outliers_z = np.abs(z_scores) > 2

    # IQR 방법
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers_iqr = (data < lower) | (data > upper)

    print(f"Z-score 이상치: {np.sum(outliers_z)}개")
    print(f"IQR 이상치: {np.sum(outliers_iqr)}개")

    return outliers_z, outliers_iqr
```

---

# Part 2 정리

## Z-score 기반 이상치 탐지

### 핵심 공식
```python
Z = (X - μ) / σ
```

### 판단 기준
- **\|Z\| > 2**: 이상치 의심 (보수적)
- **\|Z\| > 3**: 이상치 확정 (일반적)

### 실무 적용
1. 데이터 수집
2. Z-score 계산
3. 기준 초과 확인
4. 이상치 원인 조사

---

# 핵심 정리

## Part 1-2 요약

### 정규분포와 68-95-99.7 규칙
- 정규분포는 평균 중심의 **종 모양** 분포
- **68%**가 평균 ± 1σ 범위 안에 존재
- **95%**가 평균 ± 2σ 범위 안에 존재
- **99.7%**가 평균 ± 3σ 범위 안에 존재

### Z-score 이상치 탐지
- Z = (X - μ) / σ
- **|Z| > 2**: 이상치 의심
- **|Z| > 3**: 이상치 확정

---

# 다음 차시 예고

## 7차시: 통계 검정 실습

### 학습 내용
- **t-검정**: 두 그룹의 평균 비교
- **카이제곱 검정**: 범주형 데이터 분석
- **ANOVA**: 3개 이상 그룹 비교

### 준비물
- 오늘 배운 정규분포, Z-score 개념 복습
- scipy 라이브러리 설치 확인

---

# 감사합니다

## 6차시: 확률분포와 품질검정 기초

**다음 시간에 통계 검정 방법을 배워봅시다!**
