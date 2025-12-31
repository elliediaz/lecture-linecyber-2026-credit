---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 4차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section {
    font-family: 'Malgun Gothic', sans-serif;
  }
  h1 {
    color: #2563eb;
  }
  h2 {
    color: #1e40af;
  }
  code {
    background-color: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
  }
  pre {
    background-color: #1e293b;
    color: #e2e8f0;
  }
---

# 기술통계의 시각적 이해

## 4차시 | AI 기초체력훈련 (Pre AI-Campus)

**Matplotlib으로 데이터 시각화하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **기술통계량**(평균, 중앙값, 표준편차)의 의미를 이해한다
2. **Matplotlib**으로 기본 그래프를 그린다
3. **히스토그램, 상자그림, 산점도**를 해석한다

---

# 왜 시각화가 중요한가?

## 앤스콤의 4분면 (Anscombe's Quartet)

4개의 데이터셋이 있습니다:
- 평균, 분산, 상관계수 **모두 동일**
- 하지만 그래프로 보면 **완전히 다름**

> **교훈**: 숫자만 보지 말고, 반드시 그래프로 확인하자!

---

# 기술통계란?

## 데이터를 요약하는 숫자들

### 중심 경향
- **평균 (Mean)**: 모든 값의 합 / 개수
- **중앙값 (Median)**: 정렬 후 가운데 값
- **최빈값 (Mode)**: 가장 많이 나타나는 값

### 퍼짐 정도
- **분산 (Variance)**: 평균으로부터의 편차 제곱의 평균
- **표준편차 (Std)**: 분산의 제곱근
- **범위 (Range)**: 최대값 - 최소값

---

# 평균 vs 중앙값

## 언제 뭘 써야 할까?

```
데이터: [100, 100, 100, 100, 100, 100, 100, 100, 100, 1000]

평균: 190 (이상치에 영향 받음)
중앙값: 100 (이상치에 강건함)
```

### 가이드라인
- **평균**: 데이터가 정규분포에 가까울 때
- **중앙값**: 극단값(이상치)이 있을 때

> 제조 데이터는 이상치가 많으므로 **중앙값**도 함께 확인!

---

# 표준편차의 의미

## 데이터가 얼마나 퍼져있는가?

```
라인 A 생산량: [1190, 1200, 1195, 1205, 1210]
라인 B 생산량: [1000, 1100, 1200, 1300, 1400]

두 라인 모두 평균 = 1200

라인 A 표준편차: 7.1 (일관됨)
라인 B 표준편차: 141.4 (불안정)
```

> 표준편차가 작을수록 **품질이 일관**됩니다

---

# Matplotlib 소개

## Python의 표준 시각화 라이브러리

```python
import matplotlib.pyplot as plt

# 기본 그래프
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel("X축")
plt.ylabel("Y축")
plt.title("제목")
plt.show()
```

> `plt`는 matplotlib.pyplot의 관례적 별명

---

# 한글 폰트 설정

## 한글이 깨지지 않게 하려면

```python
import matplotlib.pyplot as plt

# 방법 1: 맑은 고딕 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 방법 2: 나눔고딕 (설치 필요)
plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
```

---

# 그래프 종류 선택 가이드

## 목적에 맞는 그래프

| 목적 | 그래프 종류 |
|------|------------|
| 분포 확인 | 히스토그램, 상자그림 |
| 두 변수 관계 | 산점도 |
| 시간에 따른 변화 | 선 그래프 |
| 범주별 비교 | 막대 그래프 |
| 구성 비율 | 원 그래프 |

---

# 히스토그램 (Histogram)

## 데이터 분포를 보는 그래프

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(1200, 50, 1000)  # 평균 1200, 표준편차 50

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 분포')
plt.axvline(np.mean(data), color='red', linestyle='--', label='평균')
plt.legend()
plt.show()
```

---

# 히스토그램 해석

## 분포의 형태 읽기

```
정규분포 (종 모양)
  ▲
  ██
 ████
███████
──────────

오른쪽 꼬리 (Right-skewed)
  ▲
██
████
███████──
──────────
→ 평균 > 중앙값 (이상치 방향)
```

---

# 상자그림 (Box Plot)

## 5가지 요약 통계를 한눈에

```python
plt.figure(figsize=(8, 6))
plt.boxplot([line1, line2, line3], labels=['라인1', '라인2', '라인3'])
plt.ylabel('생산량')
plt.title('라인별 생산량 분포')
plt.show()
```

---

# 상자그림 구성요소

## 해석 방법

```
    ○  ← 이상치 (outlier)
    │
    ┬  ← 최대값 (Q3 + 1.5*IQR 이내)
    │
 ┌──┴──┐
 │     │ ← Q3 (75%)
 │──┼──│ ← 중앙값 (50%)
 │     │ ← Q1 (25%)
 └──┬──┘
    │
    ┴  ← 최소값 (Q1 - 1.5*IQR 이내)
    │
    ○  ← 이상치
```

> IQR = Q3 - Q1 (사분위 범위)

---

# 산점도 (Scatter Plot)

## 두 변수의 관계 확인

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['온도'], df['불량률'], alpha=0.5)
plt.xlabel('온도 (°C)')
plt.ylabel('불량률')
plt.title('온도와 불량률의 관계')
plt.show()
```

---

# 산점도 해석

## 상관관계 패턴

```
양의 상관        음의 상관        상관 없음
    ●              ●●           ● ●  ●
   ●●             ●●             ●●●●
  ●●              ●●            ● ●● ●
 ●●                ●●           ●● ●●
●●                  ●●          ●  ● ●
───────        ───────         ───────

온도↑ 불량률↑   온도↑ 불량률↓   관련 없음
```

---

# 선 그래프 (Line Plot)

## 시간에 따른 변화 추적

```python
dates = pd.date_range('2024-01-01', periods=30)
production = np.random.randint(1100, 1300, 30)

plt.figure(figsize=(12, 6))
plt.plot(dates, production, marker='o', linestyle='-')
plt.xlabel('날짜')
plt.ylabel('생산량')
plt.title('일별 생산량 추이')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

# 막대 그래프 (Bar Plot)

## 범주별 비교

```python
lines = ['라인1', '라인2', '라인3']
avg_production = [1200, 1150, 1280]

plt.figure(figsize=(8, 6))
plt.bar(lines, avg_production, color=['steelblue', 'coral', 'green'])
plt.xlabel('생산 라인')
plt.ylabel('평균 생산량')
plt.title('라인별 평균 생산량')
plt.show()
```

---

# 여러 그래프 한 번에 그리기

## subplot 활용

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 좌상: 히스토그램
axes[0, 0].hist(data, bins=20)
axes[0, 0].set_title('생산량 분포')

# 우상: 상자그림
axes[0, 1].boxplot(data)
axes[0, 1].set_title('생산량 상자그림')

# 좌하: 선 그래프
axes[1, 0].plot(dates, production)
axes[1, 0].set_title('일별 생산량')

# 우하: 산점도
axes[1, 1].scatter(temp, defect_rate)
axes[1, 1].set_title('온도 vs 불량률')

plt.tight_layout()
plt.show()
```

---

# 그래프 저장하기

## 파일로 내보내기

```python
# 그래프 그린 후
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("제목")

# 파일 저장
plt.savefig("graph.png", dpi=300, bbox_inches='tight')
plt.savefig("graph.pdf")  # PDF 형식

# 화면에도 보이려면
plt.show()
```

> `dpi=300`: 고해상도 (보고서/논문용)

---

# 실습: 제조 데이터 시각화

## 종합 분석 대시보드

```python
# 데이터 준비
np.random.seed(42)
n = 100
df = pd.DataFrame({
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.normal(30, 10, n),
    '온도': np.random.normal(85, 5, n)
})

# 4개 그래프 그리기
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... (히스토그램, 상자그림, 산점도, 막대그래프)
```

---

# 시각화 Best Practices

## 좋은 그래프를 위한 팁

1. **제목과 축 레이블** 반드시 포함
2. **적절한 그래프 종류** 선택
3. **색상 과다 사용** 피하기 (3~4개 이내)
4. **이상치와 패턴** 강조하기
5. **범례** 명확하게 표시
6. **해상도** 충분히 높게 (보고서용 300dpi)

---

# 통계량과 시각화의 조합

## 숫자 + 그래프 = 완전한 이해

| 통계량 | 시각화 | 알 수 있는 것 |
|--------|--------|-------------|
| 평균, 표준편차 | 히스토그램 | 분포 형태, 중심, 퍼짐 |
| 사분위수 | 상자그림 | 이상치, 범위 |
| 상관계수 | 산점도 | 두 변수 관계 |
| 시계열 통계 | 선 그래프 | 추세, 계절성 |

---

# 학습 정리

## 오늘 배운 내용

### 1. 기술통계량
- 평균, 중앙값 (중심)
- 표준편차, 분산 (퍼짐)

### 2. Matplotlib 시각화
- 히스토그램: 분포 확인
- 상자그림: 이상치와 범위
- 산점도: 두 변수 관계
- 선/막대 그래프: 시계열, 범주 비교

---

# 다음 차시 예고

## 5차시: 확률분포와 AI 예측의 연결

- 정규분포의 개념
- 확률분포와 예측의 관계
- 불확실성 이해하기

### 과제 (선택)
- 자신의 데이터로 히스토그램, 상자그림 그려보기
- 이상치가 있는지 확인해보기

---

# Q&A

## 질문이 있으신가요?

### 자주 하는 실수
- `plt.show()` 호출 잊음 → 그래프 안 보임
- 한글 깨짐 → 폰트 설정 필요
- 그래프 겹침 → `plt.figure()` 새로 호출

### 추가 학습 자료
- Matplotlib 갤러리: https://matplotlib.org/gallery/
- Seaborn (고급 시각화): https://seaborn.pydata.org/

---

# 감사합니다

## AI 기초체력훈련 4차시

**기술통계의 시각적 이해**

다음 시간에 만나요!
