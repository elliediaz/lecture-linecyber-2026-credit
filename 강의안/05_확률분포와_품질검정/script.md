# [5차시] 확률분포와 품질 검정 - 강사 스크립트

## 강의 정보
- **차시**: 5차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 5차시를 시작하겠습니다.
>
> 지난 시간에 Matplotlib으로 시각화를 배웠죠? 히스토그램, 상자그림, 산점도로 데이터를 눈으로 확인하는 방법을 익혔습니다.
>
> 오늘은 한 단계 더 나아가서, 데이터가 **정상인지 이상인지** 판단하는 방법을 배웁니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 정규분포의 68-95-99.7 규칙을 이해합니다.
> 둘째, Z-score를 활용해서 이상치를 탐지합니다.
> 셋째, t-검정으로 두 그룹의 차이를 검정합니다.
>
> 제조 현장에서 품질 관리할 때 꼭 필요한 내용입니다. 잘 따라와 주세요.

---

### 핵심 내용 (8분)

#### 왜 확률분포를 알아야 하는가? [1분]

> 제조 현장에서 이런 질문들 해보셨죠?
>
> "이 측정값이 정상 범위인가?"
> "두 라인의 품질 차이가 실제로 있는가?"
> "오늘 생산량이 이상하게 높은데, 정말 이상한 건가?"
>
> 이런 질문에 **감이 아니라 데이터로** 답하려면 확률분포를 알아야 합니다.

#### 정규분포 [2분]

> **정규분포**는 가장 중요한 확률분포입니다.
>
> 모양이 **종 모양**이에요. 평균을 중심으로 좌우 대칭이고, 평균에서 멀어질수록 데이터가 적어집니다.
>
> 왜 중요하냐면, 대부분의 자연현상과 제조 데이터가 정규분포를 따르기 때문입니다.
>
> 생산량, 측정 오차, 품질 수치 대부분이 정규분포에 가깝습니다.

#### 68-95-99.7 규칙 [2분]

> 정규분포의 핵심은 **68-95-99.7 규칙**입니다.
>
> 평균에서 표준편차 1개 범위 안에 68%의 데이터가 있습니다.
> 표준편차 2개 범위 안에 95%가 있고요.
> 표준편차 3개 범위 안에 99.7%가 있습니다.
>
> 이 말은 뭐냐면, **평균 ±3σ 밖에 있는 데이터는 0.3%도 안 된다**는 거예요.
> 그래서 이 범위를 벗어나면 **이상치**라고 의심할 수 있습니다.
>
> 예를 들어 평균 1200, 표준편차 50인 생산량 데이터에서 1350을 넘거나 1050 미만이면 이상치입니다.

#### Z-score [1.5분]

> 이상치를 판단하는 도구가 **Z-score**입니다.
>
> 계산 공식은 간단해요.
>
> Z = (관측값 - 평균) / 표준편차
>
> Z-score가 2보다 크거나 -2보다 작으면 주의가 필요하고, 3을 넘으면 이상치로 판단합니다.
>
> 예를 들어 생산량이 1300이고, 평균이 1200, 표준편차가 50이면
> Z = (1300 - 1200) / 50 = 2
>
> Z가 2니까 상위 2.5% 정도에 해당하는, 조금 높은 값입니다.

#### t-검정 기초 [1.5분]

> 이제 **t-검정**을 배워봅시다.
>
> 두 그룹의 평균이 다른지 비교할 때 사용합니다.
>
> 예를 들어 "라인 A와 라인 B의 불량률이 다른가?"라는 질문에 답할 때 쓰죠.
>
> t-검정을 하면 **p-value**가 나옵니다. 이 값이 0.05보다 작으면 "두 그룹은 통계적으로 다르다"고 결론냅니다.
>
> p-value는 "이 차이가 우연히 나타났을 확률"입니다. 5% 미만이면 우연이 아니라 진짜 차이라는 뜻이에요.

---

## 실습편 (15-20분)

### 실습 소개 [2분]

> 이제 실습 시간입니다. 오늘은 정규분포를 시각화하고, Z-score로 이상치를 찾고, t-검정으로 라인을 비교해보겠습니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from scipy import stats
>
> plt.rcParams['font.family'] = 'Malgun Gothic'
> plt.rcParams['axes.unicode_minus'] = False
> ```
>
> scipy가 새로 추가됐습니다. 통계 검정에 사용하는 라이브러리예요.

### 실습 1: 정규분포 시각화 [3분]

> 첫 번째 실습입니다. 정규분포 데이터를 만들고 히스토그램으로 확인해봅시다.
>
> ```python
> np.random.seed(42)
>
> # 정규분포 데이터 생성 (평균=1200, 표준편차=50)
> production = np.random.normal(1200, 50, 1000)
>
> plt.figure(figsize=(10, 6))
> plt.hist(production, bins=30, edgecolor='black', alpha=0.7)
> plt.axvline(1200, color='red', linestyle='--', label='평균')
> plt.axvline(1200-50, color='orange', linestyle=':', label='-1σ')
> plt.axvline(1200+50, color='orange', linestyle=':', label='+1σ')
> plt.xlabel('생산량')
> plt.ylabel('빈도')
> plt.title('생산량 정규분포 (μ=1200, σ=50)')
> plt.legend()
> plt.show()
> ```
>
> 종 모양이 보이시죠? 빨간 선이 평균이고, 주황 점선이 ±1σ 범위입니다.

### 실습 2: 68-95-99.7 확인 [2분]

> 두 번째 실습입니다. 실제로 68-95-99.7 규칙이 맞는지 확인해봅시다.
>
> ```python
> mean = np.mean(production)
> std = np.std(production)
>
> within_1std = np.sum((production >= mean - std) &
>                       (production <= mean + std)) / len(production)
> within_2std = np.sum((production >= mean - 2*std) &
>                       (production <= mean + 2*std)) / len(production)
> within_3std = np.sum((production >= mean - 3*std) &
>                       (production <= mean + 3*std)) / len(production)
>
> print(f"±1σ 범위: {within_1std:.1%}")
> print(f"±2σ 범위: {within_2std:.1%}")
> print(f"±3σ 범위: {within_3std:.1%}")
> ```
>
> 68%, 95%, 99.7%에 가까운 값이 나올 겁니다. 정규분포의 특성이에요.

### 실습 3: Z-score 이상치 탐지 [3분]

> 세 번째 실습입니다. 일주일 생산 데이터에서 이상치를 찾아봅시다.
>
> ```python
> daily_production = np.array([1185, 1210, 1195, 1180, 1420, 1200, 1190])
> days = ['월', '화', '수', '목', '금', '토', '일']
>
> mean = daily_production.mean()
> std = daily_production.std()
> z_scores = (daily_production - mean) / std
>
> for day, prod, z in zip(days, daily_production, z_scores):
>     status = "이상치!" if abs(z) > 2 else "정상"
>     print(f"{day}: {prod}개 (Z={z:.2f}) - {status}")
> ```
>
> 금요일에 1420개로 Z-score가 2를 넘습니다. 이상치로 탐지되었죠.
>
> 실제 현장이라면 "금요일에 무슨 일이 있었나?" 조사해봐야 합니다.

### 실습 4: 이상치 시각화 [2분]

> 네 번째 실습입니다. 이상치를 빨간색으로 표시해봅시다.
>
> ```python
> plt.figure(figsize=(10, 6))
>
> colors = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
> bars = plt.bar(days, daily_production, color=colors, edgecolor='black')
>
> plt.axhline(mean, color='green', linestyle='--', label=f'평균: {mean:.0f}')
> plt.axhline(mean + 2*std, color='orange', linestyle=':', label='+2σ')
> plt.axhline(mean - 2*std, color='orange', linestyle=':', label='-2σ')
>
> plt.xlabel('요일')
> plt.ylabel('생산량')
> plt.title('일별 생산량과 이상치 탐지')
> plt.legend()
> plt.show()
> ```
>
> 빨간 막대가 이상치입니다. 한눈에 어느 날이 문제인지 알 수 있죠.

### 실습 5: 라인별 데이터 준비 [2분]

> 다섯 번째 실습입니다. t-검정을 위한 데이터를 만들어봅시다.
>
> ```python
> np.random.seed(123)
>
> # 라인 A: 평균 불량률 2.2%
> line_a_defect = np.random.normal(2.2, 0.3, 30)
>
> # 라인 B: 평균 불량률 2.8%
> line_b_defect = np.random.normal(2.8, 0.35, 30)
>
> print("=== 라인별 불량률 통계 ===")
> print(f"라인 A: 평균 {line_a_defect.mean():.2f}%, 표준편차 {line_a_defect.std():.2f}")
> print(f"라인 B: 평균 {line_b_defect.mean():.2f}%, 표준편차 {line_b_defect.std():.2f}")
> ```
>
> 라인 A가 약 2.2%, 라인 B가 약 2.8% 정도로 나올 겁니다.

### 실습 6: t-검정 수행 [3분]

> 여섯 번째 실습입니다. 두 라인의 불량률 차이가 통계적으로 유의미한지 검정해봅시다.
>
> ```python
> from scipy import stats
>
> t_stat, p_value = stats.ttest_ind(line_a_defect, line_b_defect)
>
> print("=== t-검정 결과 ===")
> print(f"t-통계량: {t_stat:.3f}")
> print(f"p-value: {p_value:.6f}")
>
> if p_value < 0.05:
>     print("결론: 두 라인의 불량률 차이가 통계적으로 유의미합니다.")
> else:
>     print("결론: 두 라인의 불량률 차이가 유의미하지 않습니다.")
> ```
>
> `stats.ttest_ind()`가 독립표본 t-검정 함수입니다.
>
> p-value가 0.05보다 훨씬 작게 나오면, 두 라인의 차이가 "우연이 아니다"라는 뜻입니다.

### 실습 7: 검정 결과 시각화 [2분]

> 마지막 실습입니다. 상자그림으로 두 라인을 비교해봅시다.
>
> ```python
> plt.figure(figsize=(10, 6))
>
> plt.boxplot([line_a_defect, line_b_defect],
>             labels=['라인 A', '라인 B'])
>
> plt.ylabel('불량률 (%)')
> plt.title(f'라인별 불량률 비교 (p-value: {p_value:.4f})')
> plt.show()
> ```
>
> 상자 위치가 확실히 다르죠? 이게 통계적으로도 유의미한 차이라는 걸 t-검정으로 확인한 겁니다.

---

### 정리 (3분)

#### 핵심 요약 [1.5분]

> 오늘 배운 내용을 정리하겠습니다.
>
> **정규분포**: 평균 ±3σ 안에 99.7%의 데이터가 있습니다. 이 범위를 벗어나면 이상치입니다.
>
> **Z-score**: (값 - 평균) / 표준편차로 계산합니다. |Z| > 2면 주의, |Z| > 3이면 이상치입니다.
>
> **t-검정**: 두 그룹의 평균을 비교합니다. p-value < 0.05면 차이가 유의미합니다.

#### 주의사항 [0.5분]

> 주의할 점이 있습니다.
>
> 표본 크기가 너무 작으면 t-검정 결과를 신뢰하기 어렵습니다. 최소 각 그룹당 10개 이상은 있어야 해요.
>
> 그리고 p-value가 작다고 해서 "효과가 크다"는 게 아닙니다. "이 차이가 우연이 아니다"라는 뜻일 뿐이에요.

#### 다음 차시 예고 [0.5분]

> 다음 6차시에서는 **상관분석과 예측의 기초**를 배웁니다.
>
> 온도와 불량률 같은 두 변수가 어떤 관계가 있는지, 그리고 그 관계를 이용해서 예측하는 방법을 알아봅니다.

#### 마무리 [0.5분]

> 오늘 확률분포와 통계 검정의 기초를 배웠습니다.
>
> 이제 "이 데이터가 이상한가?", "두 그룹이 정말 다른가?"라는 질문에 데이터로 답할 수 있습니다.
>
> 수고하셨습니다. 다음 시간에 뵙겠습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- scipy 라이브러리 설치 확인

### 주의사항
- scipy가 설치 안 됐으면: `pip install scipy`
- t-검정 결과 해석 시 표본 크기 언급 필수
- 정규분포 가정이 필요함을 설명

### 예상 질문
1. "표본이 작으면 어떻게 하나요?"
   → 비모수 검정(Mann-Whitney U test) 사용. 지금은 t-검정만 알면 됨

2. "p-value가 0.051이면 어떻게 해석하나요?"
   → 0.05는 관례적 기준. 맥락에 따라 판단. 0.051도 무시 못 함

3. "정규분포가 아닌 데이터는?"
   → 변환하거나 비모수 검정 사용. 대부분의 제조 데이터는 정규분포에 가까움

4. "Z-score와 표준화가 같은 건가요?"
   → 맞음. 표준화(standardization) = Z-score 변환
