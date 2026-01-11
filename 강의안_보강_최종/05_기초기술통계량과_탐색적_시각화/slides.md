---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 5차시'
footer: '기초 기술통계량과 탐색적 시각화'
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

# 기초 기술통계량과 탐색적 시각화

## 5차시 | Part II. 기초 수리와 데이터 분석

**데이터의 특성을 숫자와 그래프로 파악하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **평균, 중앙값, 표준편차** 등 데이터 요약 수치의 의미를 이해한다
2. **Matplotlib**으로 기본 그래프를 그린다
3. **히스토그램, 상자그림, 산점도**를 해석한다

---

# 오늘의 진행 순서

## 이론 (15분) + 실습 (15분)

| 순서 | 학습내용 | 시간 |
|------|----------|------|
| 1 | **대표값의 의미** | 5분 |
| 2 | **데이터의 퍼짐 정도** | 5분 |
| 3 | **제조 품질 측정값의 탐색적 시각화** | 5분 |
| 4 | **실습: 시각화 실습** | 15분 |

---

<!-- _class: lead -->

# 1. 대표값의 의미

## 데이터를 하나의 숫자로 요약하기

---

# 왜 대표값이 필요한가?

## 데이터 요약의 필요성

```
생산라인 A의 온도 데이터 (100개):
85.2, 86.1, 84.8, 87.3, 85.5, 86.0, 84.9, 85.8, ...

→ 100개의 숫자를 다 볼 수 없다!
→ "평균 85.5°C" 한 마디로 요약
```

## 대표값(Central Tendency)
- 데이터의 **중심 위치**를 나타내는 값
- 데이터 전체를 **하나의 숫자**로 요약
- 대표적인 3가지: **평균, 중앙값, 최빈값**

---

# 평균 (Mean)

## 정의

> 모든 값을 더해서 개수로 나눈 값

$$\bar{x} = \frac{x_1 + x_2 + ... + x_n}{n} = \frac{\sum_{i=1}^{n} x_i}{n}$$

## 예시
```python
temps = [85, 86, 84, 87, 85, 86, 84, 85]
mean = sum(temps) / len(temps)
# mean = 682 / 8 = 85.25
```

## 특징
- 가장 많이 사용되는 대표값
- **이상치에 민감** (하나의 큰 값이 전체 평균을 왜곡)

---

# 평균의 이상치 민감성

## 이상치가 있을 때

```
정상 데이터: [85, 86, 84, 87, 85, 86, 84, 85]
→ 평균 = 85.25

이상치 포함: [85, 86, 84, 87, 85, 86, 84, 200]
→ 평균 = 99.63  (14도 이상 차이!)
```

## 제조 현장에서의 의미
- 센서 오류, 설비 이상 시 비정상 값 발생
- 평균만 보면 잘못된 판단 가능
- **중앙값과 함께 비교** 필요

---

# 중앙값 (Median)

## 정의

> 데이터를 **크기 순으로 정렬**했을 때 **가운데** 위치한 값

## 계산 방법
- 홀수 개: 정중앙 값
- 짝수 개: 가운데 두 값의 평균

```python
# 홀수 개 (7개)
data = [84, 85, 85, 86, 86, 87, 88]
#                  ↑ 중앙값 = 86

# 짝수 개 (8개)
data = [84, 84, 85, 85, 86, 86, 87, 88]
#                  ↑   ↑  중앙값 = (85+86)/2 = 85.5
```

---

# 중앙값의 장점

## 이상치에 강건함 (Robust)

```
정상 데이터: [84, 84, 85, 85, 86, 86, 87, 88]
→ 중앙값 = 85.5

이상치 포함: [84, 84, 85, 85, 86, 86, 87, 200]
→ 중앙값 = 85.5  (변화 없음!)
```

## 언제 중앙값을 쓰나?
- 이상치가 많을 때
- 분포가 **비대칭**일 때
- **소득, 집값** 등 극단값이 있는 데이터

---

# 최빈값 (Mode)

## 정의

> 가장 **자주 나타나는** 값

## 예시
```python
data = [85, 86, 85, 87, 85, 86, 84, 85]
# 85가 4번 등장 → 최빈값 = 85
```

## 특징
- **범주형 데이터**에 유용
- 연속형 데이터에서는 잘 안 씀 (모든 값이 다를 수 있음)
- 여러 개 존재 가능 (다중 최빈값)

---

# 대표값 비교

## 평균 vs 중앙값 vs 최빈값

| 대표값 | 특징 | 언제 사용 |
|--------|------|----------|
| **평균** | 모든 값 반영 | 정규분포, 이상치 없을 때 |
| **중앙값** | 이상치에 강함 | 이상치 있을 때, 비대칭 |
| **최빈값** | 가장 흔한 값 | 범주형 데이터 |

## 평균 = 중앙값 = 최빈값?
- **정규분포**: 세 값이 거의 같음
- **비대칭 분포**: 세 값이 다름 → 분포 형태 파악 가능

---

# 분포 형태와 대표값

## 정규분포 vs 비대칭 분포

```
정규분포:
     평균 = 중앙값 = 최빈값
            ↓
      ████████████
     ██████████████
    ████████████████
   ██████████████████

왼쪽 꼬리 분포 (Left-skewed):
   최빈값 > 중앙값 > 평균
     ████████████
    █████████████████
   ███████████████████████

오른쪽 꼬리 분포 (Right-skewed):
   평균 > 중앙값 > 최빈값
           ████████████
     █████████████████
   ███████████████████████
```

---

# Python에서 대표값 계산

## NumPy와 Pandas 활용

```python
import numpy as np
import pandas as pd

# 샘플 데이터
temps = [85, 86, 84, 87, 85, 86, 84, 85, 200]

# NumPy
print(f"평균: {np.mean(temps):.2f}")      # 95.78
print(f"중앙값: {np.median(temps):.2f}")  # 85.00

# Pandas Series
s = pd.Series(temps)
print(f"평균: {s.mean():.2f}")
print(f"중앙값: {s.median():.2f}")
print(f"최빈값: {s.mode().values}")       # [85]
```

---

<!-- _class: lead -->

# 2. 데이터의 퍼짐 정도

## 데이터가 얼마나 흩어져 있는가?

---

# 퍼짐(산포도)이란?

## 같은 평균, 다른 분포

```
라인 A: [84, 85, 85, 86, 86, 87]  → 평균 85.5, 촘촘
라인 B: [70, 75, 85, 86, 95, 102] → 평균 85.5, 널찍

두 라인의 평균은 같지만, 품질 관리 상태는 다르다!
```

## 산포도(Dispersion) 지표
- **범위 (Range)**: 최대 - 최소
- **분산 (Variance)**: 평균과의 차이 제곱 평균
- **표준편차 (Standard Deviation)**: 분산의 제곱근
- **사분위 범위 (IQR)**: Q3 - Q1

---

# 범위 (Range)

## 정의

> 최대값 - 최소값

$$Range = X_{max} - X_{min}$$

## 예시
```python
temps = [84, 85, 85, 86, 86, 87]
range_val = max(temps) - min(temps)  # 87 - 84 = 3
```

## 특징
- 계산이 가장 간단
- **이상치에 매우 민감**
- 극단값 2개만 사용 (나머지 무시)

---

# 분산 (Variance)

## 정의

> 각 값이 평균에서 얼마나 떨어져 있는지의 **제곱 평균**

$$s^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}$$

## 계산 과정
```
데이터: [84, 85, 86, 87]
평균: 85.5

편차: 84-85.5=-1.5, 85-85.5=-0.5, 86-85.5=0.5, 87-85.5=1.5
편차제곱: 2.25, 0.25, 0.25, 2.25
합계: 5.0
분산: 5.0 / 3 = 1.67  (n-1로 나눔)
```

---

# 표준편차 (Standard Deviation)

## 정의

> 분산의 **제곱근**

$$s = \sqrt{s^2} = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}}$$

## 분산 vs 표준편차
- **분산**: 단위가 제곱 (예: °C²)
- **표준편차**: 원래 단위와 같음 (예: °C)

> 표준편차 = 평균에서 대략 이 정도 떨어져 있다

---

# 표준편차의 해석

## 68-95-99.7 규칙 (정규분포)

```
       ┌───────────────────────────────────────────┐
       │                   99.7%                   │
       │         ┌───────────────────┐             │
       │         │       95%        │             │
       │         │  ┌───────────┐   │             │
       │         │  │    68%    │   │             │
       │         │  │           │   │             │
       │         │  │     μ     │   │             │
       └─────────┼──┼───────────┼───┼─────────────┘
              -3σ -2σ   -σ   μ   σ   2σ   3σ
```

- **68%**: 평균 ± 1σ 안에
- **95%**: 평균 ± 2σ 안에
- **99.7%**: 평균 ± 3σ 안에

---

# 표준편차 활용: 품질 관리

## 예시: 제품 온도 관리

```
목표 온도: 85°C
표준편차: 2°C

관리 범위:
- 68% 확률: 83~87°C
- 95% 확률: 81~89°C
- 99.7% 확률: 79~91°C

→ 91°C 초과 시 이상 신호!
```

---

# 사분위수와 IQR

## 사분위수 (Quartile)

```
데이터를 4등분하는 위치의 값

   최소    Q1      Q2(중앙값)   Q3      최대
    │      │         │         │        │
    ├──────┼─────────┼─────────┼────────┤
    │ 25%  │   25%   │   25%   │  25%   │
```

- **Q1 (25%)**: 하위 25% 위치
- **Q2 (50%)**: 중앙값
- **Q3 (75%)**: 상위 25% 위치

## IQR (Interquartile Range)

$$IQR = Q3 - Q1$$

---

# IQR을 이용한 이상치 탐지

## 이상치 기준

```
이상치 하한: Q1 - 1.5 × IQR
이상치 상한: Q3 + 1.5 × IQR
```

## 예시
```python
Q1, Q3 = 83, 87
IQR = Q3 - Q1  # 4

lower = Q1 - 1.5 * IQR  # 83 - 6 = 77
upper = Q3 + 1.5 * IQR  # 87 + 6 = 93

# 77 미만 또는 93 초과 → 이상치
```

---

# Python에서 산포도 계산

## NumPy와 Pandas 활용

```python
import numpy as np
import pandas as pd

temps = [84, 85, 85, 86, 86, 87, 200]

# 기본 통계
print(f"범위: {np.max(temps) - np.min(temps)}")  # 116
print(f"분산: {np.var(temps, ddof=1):.2f}")       # 1653.14
print(f"표준편차: {np.std(temps, ddof=1):.2f}")  # 40.66

# 사분위수
print(f"Q1: {np.percentile(temps, 25)}")
print(f"Q3: {np.percentile(temps, 75)}")

# describe()로 한 번에
print(pd.Series(temps).describe())
```

---

# describe() 결과 해석

## Pandas describe() 출력

```python
>>> pd.Series([84, 85, 85, 86, 86, 87]).describe()

count     6.000000    # 개수
mean     85.500000    # 평균
std       1.048809    # 표준편차
min      84.000000    # 최소값
25%      85.000000    # Q1 (1사분위수)
50%      85.500000    # Q2 (중앙값)
75%      86.000000    # Q3 (3사분위수)
max      87.000000    # 최대값
```

> **데이터를 받으면 describe() 먼저!**

---

<!-- _class: lead -->

# 3. 제조 품질 측정값의 탐색적 시각화

## 데이터를 눈으로 이해하기

---

# 탐색적 데이터 분석 (EDA)

## Exploratory Data Analysis

> 시각화와 통계를 통해 데이터의 **패턴, 이상, 관계**를 파악하는 과정

## EDA의 목적
1. 데이터 분포 파악
2. 이상치 탐지
3. 변수 간 관계 확인
4. 모델링 전 인사이트 획득

## 핵심 시각화 도구
- **히스토그램**: 분포 확인
- **상자그림**: 이상치, 사분위수
- **산점도**: 변수 간 관계

---

# Matplotlib 기초

## Python 시각화 라이브러리

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 그래프 그리기
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)           # 선 그래프
plt.xlabel('X축')
plt.ylabel('Y축')
plt.title('기본 그래프')
plt.show()
```

---

# 히스토그램 (Histogram)

## 정의

> 데이터를 **구간(bin)**으로 나누고, 각 구간의 **빈도**를 막대로 표현

## 코드
```python
import matplotlib.pyplot as plt
import numpy as np

temps = np.random.normal(85, 2, 1000)  # 평균 85, 표준편차 2

plt.hist(temps, bins=20, edgecolor='black')
plt.xlabel('온도 (°C)')
plt.ylabel('빈도')
plt.title('온도 분포')
plt.axvline(np.mean(temps), color='red', linestyle='--', label='평균')
plt.legend()
plt.show()
```

---

# 히스토그램 해석

## 분포 형태 파악

```
정규분포:             왼쪽 꼬리:           오른쪽 꼬리:
    ████                    ████████          ████
   ██████                  ██████████        ██████
  ████████                ████████████      ████████████
 ██████████              ██████████████    █████████████████
████████████            ████████████████████████████████████████
```

## 확인 포인트
- **중심 위치**: 어디에 몰려 있나?
- **퍼짐 정도**: 넓게/좁게 퍼져 있나?
- **대칭성**: 좌우 대칭인가?
- **이상치**: 동떨어진 막대가 있나?

---

# 상자그림 (Box Plot)

## 정의

> 사분위수와 이상치를 **한눈에** 보여주는 그래프

```
                    이상치 ○
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │  ← 최대값 (수염)
    │  ┌───────────────┴───────────────┐  │
    │  │                               │  │  ← Q3
    │  │           ─────────           │  │  ← 중앙값
    │  │                               │  │  ← Q1
    │  └───────────────────────────────┘  │
    │                                     │  ← 최소값 (수염)
    └─────────────────────────────────────┘
```

---

# 상자그림 코드

## Matplotlib으로 Box Plot 그리기

```python
import matplotlib.pyplot as plt
import numpy as np

# 3개 라인 데이터
line_a = np.random.normal(85, 2, 100)
line_b = np.random.normal(86, 3, 100)
line_c = np.random.normal(84, 5, 100)

plt.boxplot([line_a, line_b, line_c],
            labels=['라인 A', '라인 B', '라인 C'])
plt.ylabel('온도 (°C)')
plt.title('라인별 온도 분포')
plt.show()
```

---

# 상자그림 해석

## 비교 분석

```
라인 A        라인 B        라인 C
   │            │       ○      │
┌──┴──┐     ┌───┴───┐  ┌──┴──┐
│  ─  │     │   ─   │  │     │
└──┬──┘     └───┬───┘  │  ─  │
   │            │      │     │
                       └──┬──┘
                          │
```

| 라인 | 해석 |
|------|------|
| A | 안정적, 변동 작음 |
| B | 약간 높음, 변동 중간 |
| C | 변동 크고 이상치 존재 → 품질 문제! |

---

# 산점도 (Scatter Plot)

## 정의

> 두 변수의 **관계**를 점으로 표현

## 코드
```python
import matplotlib.pyplot as plt
import numpy as np

temp = np.random.normal(85, 3, 100)
defect_rate = 0.5 * temp + np.random.normal(0, 2, 100)

plt.scatter(temp, defect_rate, alpha=0.6)
plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도 vs 불량률')
plt.show()
```

---

# 산점도 해석

## 상관관계 파악

```
양의 상관:           음의 상관:          상관 없음:
   ●                  ●                 ●   ●
     ●                  ●             ●   ●   ●
       ●                  ●         ●   ●   ●   ●
         ●                  ●         ●   ●
           ●                  ●         ●   ●
```

- **양의 상관**: X 증가 → Y 증가
- **음의 상관**: X 증가 → Y 감소
- **상관 없음**: 패턴 없음

---

# 여러 그래프 한 번에 (subplots)

## 그래프 배치

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

data = np.random.normal(85, 3, 200)

# 히스토그램
axes[0].hist(data, bins=20)
axes[0].set_title('히스토그램')

# 상자그림
axes[1].boxplot(data)
axes[1].set_title('상자그림')

# 시계열
axes[2].plot(data[:50])
axes[2].set_title('시계열')

plt.tight_layout()
plt.show()
```

---

# 제조 데이터 시각화 사례

## 품질 대시보드

```python
# 라인별 불량률 비교 (막대 그래프)
lines = ['A', 'B', 'C', 'D']
defect_rates = [2.1, 3.5, 1.8, 4.2]

plt.bar(lines, defect_rates, color=['green', 'orange', 'green', 'red'])
plt.axhline(3.0, color='red', linestyle='--', label='기준선')
plt.xlabel('생산 라인')
plt.ylabel('불량률 (%)')
plt.title('라인별 불량률 현황')
plt.legend()
plt.show()
```

---

<!-- _class: lead -->

# - 실습편 -

## 5차시

**기술통계와 시각화 실습**

---

# 실습 개요

## 오늘의 실습 (15분)

1. **대표값 계산**: 평균, 중앙값, 최빈값 (3분)
2. **산포도 계산**: 범위, 표준편차, IQR (3분)
3. **시각화**: 히스토그램, 상자그림, 산점도 (9분)

## 준비물
- Jupyter Notebook
- pandas, numpy, matplotlib

---

# 실습 1: 대표값 계산

```python
import numpy as np
import pandas as pd

# 제조 데이터 생성
np.random.seed(42)
temps = np.random.normal(85, 3, 100)
temps[95:100] = [110, 115, 120, 50, 45]  # 이상치 추가

# 대표값 계산
print(f"평균: {np.mean(temps):.2f}")
print(f"중앙값: {np.median(temps):.2f}")
print(f"최빈값: {pd.Series(temps.round()).mode().values}")

# 비교: 평균 vs 중앙값
print(f"\n평균-중앙값 차이: {np.mean(temps) - np.median(temps):.2f}")
# 차이가 크면 이상치 존재 가능
```

---

# 실습 2: 산포도 계산

```python
# 산포도 계산
print(f"최소값: {np.min(temps):.2f}")
print(f"최대값: {np.max(temps):.2f}")
print(f"범위: {np.max(temps) - np.min(temps):.2f}")
print(f"분산: {np.var(temps, ddof=1):.2f}")
print(f"표준편차: {np.std(temps, ddof=1):.2f}")

# 사분위수
Q1 = np.percentile(temps, 25)
Q3 = np.percentile(temps, 75)
IQR = Q3 - Q1

print(f"\nQ1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")

# 이상치 범위
print(f"\n이상치 하한: {Q1 - 1.5*IQR:.2f}")
print(f"이상치 상한: {Q3 + 1.5*IQR:.2f}")
```

---

# 실습 3: 히스토그램

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.hist(temps, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(temps), color='red', linestyle='-',
            linewidth=2, label=f'평균: {np.mean(temps):.1f}')
plt.axvline(np.median(temps), color='green', linestyle='--',
            linewidth=2, label=f'중앙값: {np.median(temps):.1f}')

plt.xlabel('온도 (°C)')
plt.ylabel('빈도')
plt.title('온도 분포 히스토그램')
plt.legend()
plt.show()
```

---

# 실습 4: 상자그림

```python
# 라인별 데이터 생성
np.random.seed(42)
line_a = np.random.normal(85, 2, 50)
line_b = np.random.normal(86, 4, 50)
line_c = np.random.normal(84, 6, 50)

plt.figure(figsize=(8, 6))
plt.boxplot([line_a, line_b, line_c],
            labels=['라인 A', '라인 B', '라인 C'],
            patch_artist=True)
plt.ylabel('온도 (°C)')
plt.title('라인별 온도 분포 비교')
plt.axhline(85, color='red', linestyle='--', alpha=0.5, label='목표')
plt.legend()
plt.show()
```

---

# 실습 5: 산점도

```python
# 온도와 불량률 관계
np.random.seed(42)
temperature = np.random.normal(85, 5, 100)
defect_rate = 0.3 * temperature + np.random.normal(0, 3, 100)

plt.figure(figsize=(8, 6))
plt.scatter(temperature, defect_rate, alpha=0.6, c='blue')
plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도 vs 불량률 관계')

# 추세선 추가
z = np.polyfit(temperature, defect_rate, 1)
p = np.poly1d(z)
plt.plot(temperature, p(temperature), 'r--', alpha=0.8, label='추세선')
plt.legend()
plt.show()
```

---

# 실습 6: 종합 대시보드

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 히스토그램
axes[0, 0].hist(temps, bins=30, edgecolor='black')
axes[0, 0].set_title('온도 분포')

# 2. 상자그림
axes[0, 1].boxplot([line_a, line_b, line_c],
                    labels=['A', 'B', 'C'])
axes[0, 1].set_title('라인별 비교')

# 3. 산점도
axes[1, 0].scatter(temperature, defect_rate)
axes[1, 0].set_title('온도 vs 불량률')

# 4. 시계열
axes[1, 1].plot(temps[:50])
axes[1, 1].set_title('시간별 온도 변화')

plt.tight_layout()
plt.show()
```

---

# 핵심 요약

## 5차시 학습내용 정리

### 1. 대표값의 의미
- **평균**: 모든 값의 중심, 이상치에 민감
- **중앙값**: 정렬 후 중간값, 이상치에 강건
- **최빈값**: 가장 자주 나오는 값

### 2. 데이터의 퍼짐 정도
- **범위**: 최대 - 최소
- **표준편차**: 평균에서의 평균적 거리
- **IQR**: Q3 - Q1, 이상치 탐지에 활용

---

# 핵심 요약 (계속)

### 3. 제조 품질 측정값의 탐색적 시각화
- **히스토그램**: 분포 형태 파악
- **상자그림**: 사분위수, 이상치 확인
- **산점도**: 두 변수 간 관계

## 핵심 코드
```python
# 한 줄 요약
df.describe()

# 시각화 기본
plt.hist(data)      # 분포
plt.boxplot(data)   # 이상치
plt.scatter(x, y)   # 관계
```

---

# 다음 차시 예고

## 6차시: 확률분포와 품질 검정

- 정규분포와 Z-score
- 가설검정의 기초
- t-검정으로 라인별 품질 비교

> 오늘 배운 통계량을 바탕으로 데이터 분석의 깊이를 더합니다!

---

# 감사합니다

## 5차시: 기초 기술통계량과 탐색적 시각화

**제조데이터를 활용한 AI 이해와 예측 모델 구축**

수고하셨습니다!
