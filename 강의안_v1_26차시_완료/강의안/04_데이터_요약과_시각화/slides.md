---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 4차시'
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

# 데이터 요약과 시각화

## 4차시 | Part II. 기초 수리와 데이터 분석

**Matplotlib으로 제조 데이터 시각화하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **기술통계량**(평균, 중앙값, 표준편차)의 의미를 이해한다
2. **Matplotlib**으로 기본 그래프를 그린다
3. **히스토그램, 상자그림, 산점도**를 해석한다

---

# 왜 시각화가 중요한가?

## 숫자만으로는 부족합니다

```
데이터셋 A, B, C, D:
- 평균: 모두 동일
- 분산: 모두 동일
- 상관계수: 모두 동일

하지만 그래프로 보면? → 완전히 다른 패턴!
```

> **교훈**: 숫자만 보지 말고, **반드시 그래프로 확인**하자!

---

# 기술통계량

## 데이터를 요약하는 숫자들

| 구분 | 통계량 | 설명 |
|------|--------|------|
| 중심 | 평균 (Mean) | 모든 값의 합 / 개수 |
| 중심 | 중앙값 (Median) | 정렬 후 가운데 값 |
| 퍼짐 | 표준편차 (Std) | 평균으로부터 얼마나 퍼져있나 |
| 범위 | 최대/최소 | 데이터의 경계 |

```python
import numpy as np
data = np.array([1200, 1150, 1300, 1180, 1250])
print(f"평균: {np.mean(data)}, 표준편차: {np.std(data):.1f}")
```

---

# 평균 vs 중앙값

## 언제 뭘 써야 할까?

```
데이터: [100, 100, 100, 100, 100, 100, 100, 100, 100, 1000]

평균: 190 (이상치에 영향 받음!)
중앙값: 100 (이상치에 강건함)
```

### 가이드라인
- **평균**: 데이터가 정규분포에 가까울 때
- **중앙값**: 이상치(극단값)가 있을 때

> 제조 데이터는 이상치가 많으므로 **중앙값도 함께 확인**!

---

# 표준편차의 의미

## 품질 일관성의 지표

```
라인 A: [1190, 1200, 1195, 1205, 1210] → 표준편차: 7.1
라인 B: [1000, 1100, 1200, 1300, 1400] → 표준편차: 141.4

두 라인 모두 평균 = 1200
```

| 라인 | 표준편차 | 해석 |
|------|---------|------|
| A | 7.1 | 생산량 일관됨 (안정적) |
| B | 141.4 | 생산량 불안정 (관리 필요) |

> 표준편차가 작을수록 **품질이 일관**됩니다

---

# Matplotlib 소개

## Python의 표준 시각화 라이브러리

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 그래프
x = [1, 2, 3, 4, 5]
y = [1200, 1150, 1300, 1180, 1250]

plt.plot(x, y)
plt.xlabel("일차")
plt.ylabel("생산량")
plt.title("일별 생산량")
plt.show()
```

---

# 그래프 종류 선택 가이드

## 목적에 맞는 그래프

| 목적 | 그래프 종류 | 예시 |
|------|------------|------|
| 분포 확인 | 히스토그램 | 생산량 분포 |
| 이상치 탐지 | 상자그림 | 라인별 품질 비교 |
| 두 변수 관계 | 산점도 | 온도 vs 불량률 |
| 시간 변화 | 선 그래프 | 일별 생산량 추이 |
| 범주 비교 | 막대 그래프 | 라인별 평균 비교 |

---

# 이론 정리

## 핵심 포인트

### 기술통계량
- **평균/중앙값**: 데이터의 중심
- **표준편차**: 데이터의 퍼짐 (품질 일관성)

### Matplotlib
- **히스토그램**: 분포 형태 확인
- **상자그림**: 이상치와 범위 확인
- **산점도**: 두 변수의 관계 확인

---

# - 실습편 -

## 4차시

**제조 데이터 시각화 실습**

---

# 실습 개요

## 생산 데이터 시각화

### 실습 목표
- 히스토그램으로 분포 확인
- 상자그림으로 라인별 비교
- 산점도로 온도-불량률 관계 분석

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 준비

## 샘플 데이터 생성

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 100

df = pd.DataFrame({
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(30, n),
    '온도': np.random.normal(85, 5, n),
    '라인': np.random.choice([1, 2, 3], n)
})

df['불량률'] = df['불량수'] / df['생산량']
print(df.head())
print(df.describe())
```

---

# 실습 2: 히스토그램

## 생산량 분포 확인

```python
plt.figure(figsize=(10, 6))
plt.hist(df['생산량'], bins=20, edgecolor='black', alpha=0.7)

# 평균선 추가
mean_val = df['생산량'].mean()
plt.axvline(mean_val, color='red', linestyle='--',
            label=f'평균: {mean_val:.0f}')

plt.xlabel('생산량')
plt.ylabel('빈도')
plt.title('생산량 분포')
plt.legend()
plt.show()
```

> 종 모양(정규분포)인지, 한쪽으로 치우쳤는지 확인

---

# 실습 3: 상자그림

## 라인별 생산량 비교

```python
# 라인별 데이터 분리
line1 = df[df['라인'] == 1]['생산량']
line2 = df[df['라인'] == 2]['생산량']
line3 = df[df['라인'] == 3]['생산량']

plt.figure(figsize=(8, 6))
plt.boxplot([line1, line2, line3],
            labels=['라인1', '라인2', '라인3'])
plt.ylabel('생산량')
plt.title('라인별 생산량 분포')
plt.show()
```

> 상자 바깥의 점(o)은 이상치입니다

---

# 상자그림 해석

## 5가지 요약 통계

```
    o  ← 이상치 (outlier)
    │
    ┬  ← 최대값 (Q3 + 1.5×IQR 이내)
    │
 ┌──┴──┐
 │     │ ← Q3 (75%)
 │──┼──│ ← 중앙값 (50%)
 │     │ ← Q1 (25%)
 └──┬──┘
    │
    ┴  ← 최소값

IQR = Q3 - Q1 (사분위 범위)
```

---

# 실습 4: 산점도

## 온도와 불량률의 관계

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['온도'], df['불량률'], alpha=0.5, c='steelblue')

plt.xlabel('온도 (도)')
plt.ylabel('불량률')
plt.title('온도와 불량률의 관계')

# 추세선 추가 (선택)
z = np.polyfit(df['온도'], df['불량률'], 1)
p = np.poly1d(z)
plt.plot(df['온도'].sort_values(), p(df['온도'].sort_values()),
         "r--", alpha=0.8, label='추세선')
plt.legend()
plt.show()
```

---

# 실습 5: 선 그래프

## 일별 생산량 추이

```python
# 일별 데이터 생성
dates = pd.date_range('2024-01-01', periods=30)
daily_production = np.random.normal(1200, 50, 30)

plt.figure(figsize=(12, 6))
plt.plot(dates, daily_production, marker='o', linestyle='-')

plt.xlabel('날짜')
plt.ylabel('생산량')
plt.title('일별 생산량 추이')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

# 실습 6: 여러 그래프 한 번에

## 4분할 대시보드

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 좌상: 히스토그램
axes[0, 0].hist(df['생산량'], bins=20, edgecolor='black')
axes[0, 0].set_title('생산량 분포')

# 우상: 상자그림
axes[0, 1].boxplot([line1, line2, line3])
axes[0, 1].set_title('라인별 생산량')

# 좌하: 산점도
axes[1, 0].scatter(df['온도'], df['불량률'], alpha=0.5)
axes[1, 0].set_title('온도 vs 불량률')

# 우하: 막대 그래프
line_avg = df.groupby('라인')['생산량'].mean()
axes[1, 1].bar(line_avg.index, line_avg.values)
axes[1, 1].set_title('라인별 평균 생산량')

plt.tight_layout()
plt.show()
```

---

# 실습 7: 그래프 저장

## 파일로 내보내기

```python
# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.hist(df['생산량'], bins=20, edgecolor='black')
plt.title('생산량 분포')

# 파일 저장
plt.savefig('production_hist.png', dpi=300, bbox_inches='tight')
print("그래프가 저장되었습니다.")

plt.show()
```

> `dpi=300`: 고해상도 (보고서/문서용)
> `bbox_inches='tight'`: 여백 자동 조절

---

# 시각화 Best Practices

## 좋은 그래프를 위한 팁

| 항목 | 권장사항 |
|------|----------|
| 제목/레이블 | 반드시 포함 |
| 색상 | 3~4개 이내 |
| 폰트 크기 | 읽기 쉽게 |
| 범례 | 명확하게 표시 |
| 해상도 | 보고서용 300dpi |

> 그래프는 **한눈에 메시지 전달**이 목표

---

# 실습 정리

## 핵심 체크포인트

### 기술통계
- [ ] describe()로 기본 통계 확인
- [ ] 평균과 중앙값 비교

### 시각화
- [ ] plt.hist()로 분포 확인
- [ ] plt.boxplot()으로 이상치 탐지
- [ ] plt.scatter()로 상관관계 확인
- [ ] plt.savefig()로 저장

---

# 다음 차시 예고

## 5차시: 확률분포와 품질 검정

### 학습 내용
- 정규분포의 개념
- 확률분포와 품질 관리
- 가설검정 기초

### 준비물
- 오늘 배운 코드 복습
- 자신의 데이터로 시각화 연습

---

# 정리 및 Q&A

## 오늘의 핵심

1. **기술통계**: 평균, 중앙값, 표준편차
2. **시각화**: 히스토그램, 상자그림, 산점도
3. **해석**: 분포, 이상치, 상관관계 파악

### 자주 하는 실수
- `plt.show()` 호출 잊음
- 한글 깨짐 → 폰트 설정 필요
- 그래프 겹침 → `plt.figure()` 새로 호출

---

# 감사합니다

## 4차시: 데이터 요약과 시각화

**다음 시간에 확률분포를 배워봅시다!**
