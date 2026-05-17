# 5차시: 기초 기술통계량과 탐색적 시각화

## 학습 목표

본 차시를 마치면 다음을 수행할 수 있음:

1. **대표값**(평균, 중앙값, 최빈값)의 의미와 차이를 이해함
2. **산포도**(범위, 분산, 표준편차, IQR)로 데이터의 퍼짐 정도를 측정함
3. **탐색적 시각화**(히스토그램, 상자그림, 산점도)로 데이터 패턴을 파악함
4. **제조 품질 데이터**에 기술통계와 시각화를 적용함

---

## 강의 구성

| 순서 | 내용 | 시간 |
|------|------|------|
| Part 1 | 대표값의 의미 | 5분 |
| Part 2 | 데이터의 퍼짐 정도 | 5분 |
| Part 3 | 제조 품질 측정값의 탐색적 시각화 | 5분 |
| Part 4 | 실습: 기술통계와 시각화 | 15분 |

---

## Part 1: 대표값의 의미

### 개념 설명

수백, 수천 개의 데이터를 일일이 확인할 수 없으므로 **하나의 숫자로 요약**하는 대표값(Central Tendency)이 필요함.

**평균 (Mean)**

모든 값을 더해 개수로 나눈 값임. 모든 데이터를 반영하지만 **이상치에 민감**함.

```python
# data = [84, 85, 86, 87, 88, 84, 85, 83]
# mean = 682 / 8 = 85.25
```

이상치가 있을 때 평균은 크게 흔들림. 예를 들어 정상값에 120°C 같은 극단값이 섞이면 평균이 위로 끌려감.

**중앙값 (Median)**

데이터를 정렬했을 때 가운데 위치한 값임. **이상치에 강건(Robust)**하여 소득, 부동산 가격처럼 한쪽으로 치우친 데이터에 적합함.

```python
# 홀수 개(7개): 정렬 후 정중앙 값
# 짝수 개(8개): 가운데 두 값의 평균
```

**최빈값 (Mode)**

가장 자주 등장하는 값임. 범주형 데이터(불량 유형, 라인 등)에서 특히 유용함.

**대표값 비교와 분포 형태**

- 정규분포(대칭): 평균 ≈ 중앙값 ≈ 최빈값
- 오른쪽 꼬리(양의 왜도): 평균 > 중앙값 > 최빈값
- 왼쪽 꼬리(음의 왜도): 평균 < 중앙값 < 최빈값

> **평균과 중앙값의 차이가 크면 이상치 또는 비대칭 분포를 의심해야 함**

```python
import numpy as np
import pandas as pd

data = pd.Series([84, 85, 86, 87, 88, 84, 85, 83])
print(f"평균: {data.mean():.2f}")
print(f"중앙값: {data.median():.2f}")
print(f"최빈값: {data.mode().values}")
```

---

## Part 2: 데이터의 퍼짐 정도

### 개념 설명

평균이 같아도 데이터가 흩어진 정도(산포도, Dispersion)는 다를 수 있음. 산포도는 데이터의 안정성과 품질 일관성을 보여줌.

**범위 (Range)**

```
범위 = 최대값 - 최소값
```

계산이 간단하지만 극단값 2개에만 의존하므로 민감함.

**분산 (Variance)**

각 값이 평균에서 떨어진 거리의 제곱 평균임.

```
s² = Σ(xᵢ - x̄)² / (n - 1)

데이터: [84, 85, 86, 87], 평균 85.5
편차제곱 합계: 5.0
분산 = 5.0 / 3 = 1.67   (표본분산은 n-1로 나눔)
```

**표준편차 (Standard Deviation)**

분산의 제곱근임. 원래 데이터와 단위가 같아 해석이 직관적임(분산은 단위가 제곱).

```
s = √s²
```

정규분포에서는 **68-95-99.7 규칙**이 성립함.

- 평균 ± 1σ 안에 약 68%
- 평균 ± 2σ 안에 약 95%
- 평균 ± 3σ 안에 약 99.7%

이를 품질 관리에 활용하면, 목표 온도 85°C·표준편차 2°C일 때 99.7% 범위는 79~91°C이며 91°C 초과 시 이상 신호로 판단함.

**사분위수와 IQR**

데이터를 4등분한 위치값이 사분위수임(Q1=25%, Q2=중앙값, Q3=75%).

```
IQR = Q3 - Q1

이상치 하한: Q1 - 1.5 × IQR
이상치 상한: Q3 + 1.5 × IQR
```

예) Q1=83, Q3=87 → IQR=4 → 하한 77, 상한 93. 77 미만 또는 93 초과는 이상치로 분류함. IQR 방식은 이상치에 강건하여 박스플롯의 기준으로 사용됨.

```python
temps = np.array([84, 85, 86, 87, 88, 84, 85, 83, 110])  # 110은 이상치
print(f"범위: {temps.max() - temps.min()}")
print(f"분산: {np.var(temps, ddof=1):.2f}")
print(f"표준편차: {np.std(temps, ddof=1):.2f}")

Q1, Q3 = np.percentile(temps, [25, 75])
IQR = Q3 - Q1
print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"이상치 상한: {Q3 + 1.5 * IQR}")

# describe()로 한 번에 확인
print(pd.Series(temps).describe())
```

---

## Part 3: 제조 품질 측정값의 탐색적 시각화

### 개념 설명

**탐색적 데이터 분석(EDA, Exploratory Data Analysis)**은 통계 수치와 그래프를 통해 데이터의 분포, 관계, 이상치를 파악하는 과정임. 모델링 전 반드시 거쳐야 하는 단계임.

핵심 시각화 도구는 다음과 같음.

| 그래프 | 용도 |
|--------|------|
| 히스토그램 | 한 변수의 분포 형태 파악 |
| 상자그림(박스플롯) | 사분위수·이상치 확인, 그룹 비교 |
| 산점도 | 두 변수 간 관계 파악 |

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

**히스토그램**은 값의 구간(bin)별 빈도를 막대로 표현하여 분포가 정규형인지 치우쳤는지 보여줌. **상자그림**은 Q1, Q2, Q3와 수염(whisker), 이상치 점을 한 번에 보여주어 여러 라인의 품질을 비교하기에 좋음. **산점도**는 두 수치 변수의 관계(예: 온도와 불량률)를 점으로 표시하며 추세선을 추가하면 경향을 명확히 볼 수 있음.

---

## Part 4: 실습 - 기술통계와 시각화

### 실습 1: 대표값과 산포도 계산

```python
import numpy as np
import pandas as pd

np.random.seed(42)
temps = np.random.normal(85, 3, 100)
temps[95:100] = [110, 115, 120, 50, 45]  # 이상치 추가

print(f"평균: {np.mean(temps):.2f}")
print(f"중앙값: {np.median(temps):.2f}")
print(f"최빈값: {pd.Series(temps.round()).mode().values}")
print(f"평균-중앙값 차이: {np.mean(temps) - np.median(temps):.2f}")
# 차이가 크면 이상치 존재 가능

print(f"\n표준편차: {np.std(temps, ddof=1):.2f}")
Q1, Q3 = np.percentile(temps, [25, 75])
IQR = Q3 - Q1
print(f"IQR: {IQR:.2f}")
print(f"이상치 하한: {Q1 - 1.5 * IQR:.2f}, 상한: {Q3 + 1.5 * IQR:.2f}")
```

### 실습 2: 히스토그램

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

### 실습 3: 상자그림으로 라인별 비교

```python
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

### 실습 4: 산점도와 추세선

```python
np.random.seed(42)
temperature = np.random.normal(85, 5, 100)
defect_rate = 0.3 * temperature + np.random.normal(0, 3, 100)

plt.figure(figsize=(8, 6))
plt.scatter(temperature, defect_rate, alpha=0.6, c='blue')

z = np.polyfit(temperature, defect_rate, 1)
p = np.poly1d(z)
plt.plot(temperature, p(temperature), 'r--', alpha=0.8, label='추세선')

plt.xlabel('온도 (°C)')
plt.ylabel('불량률 (%)')
plt.title('온도 vs 불량률 관계')
plt.legend()
plt.show()
```

---

## 핵심 정리

```
┌─────────────────────────────────────────────────────────────┐
│                    5차시 핵심 요약                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 대표값                                                   │
│     - 평균: 모든 값 반영, 이상치에 민감                       │
│     - 중앙값: 정렬 후 중간값, 이상치에 강건                   │
│     - 최빈값: 가장 자주 나오는 값                             │
│     - 평균≠중앙값이면 이상치/비대칭 의심                      │
│                                                              │
│  2. 산포도                                                   │
│     - 범위 = 최대 - 최소                                      │
│     - 분산/표준편차: 평균으로부터의 평균 거리                 │
│     - 68-95-99.7 규칙으로 품질 관리                           │
│     - IQR = Q3 - Q1, 1.5×IQR로 이상치 탐지                    │
│                                                              │
│  3. 탐색적 시각화                                            │
│     - 히스토그램: 분포 형태                                   │
│     - 상자그림: 사분위수·이상치·그룹 비교                     │
│     - 산점도: 두 변수 간 관계                                 │
│     - df.describe()로 통계량 한 번에                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 실습 과제

| 항목 | 내용 |
|------|------|
| 과제 파일 | `실습과제/Q1_기초통계량_시각화.md` |
| 참조 노트북 | `실습과제/Q1_기초통계량_시각화.ipynb` |
| 난이도 | ★☆☆☆☆ (입문) |

1. AI4I 2020 데이터셋 로드 및 탐색
2. 온도 컬럼의 기술통계량 계산
3. 히스토그램과 상자그림 시각화
4. 불량 여부별 온도 평균 비교

---

## 다음 차시 예고

### 6차시: 확률분포와 품질 검정 기초

- 정규분포와 Z-score를 활용한 이상치 탐지
- 정규성 검정(Shapiro-Wilk)
- t-검정으로 라인별 품질 비교

> 오늘 배운 통계량을 바탕으로 데이터 분석의 깊이를 더함
