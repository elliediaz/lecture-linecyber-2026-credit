---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 3차시'
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

# 제조 데이터 다루기 기초

## 3차시 | Part I. AI 윤리와 환경 구축

**NumPy와 Pandas로 제조 데이터 분석 시작하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **NumPy 배열**의 개념과 기본 연산을 이해한다
2. **Pandas DataFrame**으로 표 형태 데이터를 다룬다
3. **CSV 파일**을 불러오고 제조 데이터 탐색을 수행한다

---

# 왜 NumPy와 Pandas인가?

## Python 기본 vs 데이터 분석 라이브러리

```python
# Python 리스트로 평균 계산
production = [1200, 1150, 1300, 1180, 1250]
total = 0
for x in production:
    total += x
average = total / len(production)

# NumPy로 평균 계산 - 한 줄로 끝!
import numpy as np
production = np.array([1200, 1150, 1300, 1180, 1250])
average = production.mean()
```

> **속도**: NumPy는 Python 리스트보다 최대 **100배 빠름**

---

# NumPy 소개

## NumPy = Numerical Python

| 특징 | 설명 |
|------|------|
| 다차원 배열 | ndarray 객체로 행렬 연산 |
| 벡터화 연산 | 반복문 없이 빠른 계산 |
| 통계 함수 | mean, std, max, min 등 내장 |
| 과학 계산 | 선형대수, 푸리에 변환 등 |

```python
import numpy as np
print(np.__version__)  # 버전 확인
```

---

# NumPy 배열과 연산

## 제조 데이터를 배열로 다루기

```python
import numpy as np

# 온도 센서 데이터
temperatures = np.array([82, 85, 88, 95, 84])

# 벡터화 연산 - 모든 요소에 자동 적용
temps_adjusted = temperatures - 32  # 보정값 적용

# 조건 필터링
high_temps = temperatures[temperatures > 90]
print(high_temps)  # [95]

# 기본 통계
print(f"평균: {temperatures.mean():.1f}")  # 86.8
print(f"최대: {temperatures.max()}")        # 95
```

---

# NumPy 통계 함수

## 제조 데이터 기본 통계

```python
production = np.array([1200, 1150, 1300, 1180, 1250])

print(f"합계: {np.sum(production)}")        # 6080
print(f"평균: {np.mean(production)}")       # 1216.0
print(f"표준편차: {np.std(production):.2f}") # 53.45
print(f"최대값: {np.max(production)}")      # 1300
print(f"최소값: {np.min(production)}")      # 1150
```

> 한 줄 코드로 통계 계산 가능

---

# Pandas 소개

## 표 형태 데이터의 표준 도구

| 특징 | 설명 |
|------|------|
| DataFrame | Excel처럼 행/열로 구성 |
| 파일 지원 | CSV, Excel, SQL 등 다양한 형식 |
| 데이터 조작 | 필터링, 정렬, 그룹화 |
| 통계 분석 | 기술통계, 집계 함수 |

```python
import pandas as pd
print(pd.__version__)  # 버전 확인
```

---

# DataFrame 생성

## 딕셔너리에서 DataFrame 만들기

```python
import pandas as pd

data = {
    "제품코드": ["SM-001", "SM-002", "SM-003"],
    "생산량": [1200, 1150, 1300],
    "불량수": [24, 35, 26],
    "라인": [1, 2, 1]
}

df = pd.DataFrame(data)
print(df)
```

```
  제품코드  생산량  불량수  라인
0  SM-001   1200     24    1
1  SM-002   1150     35    2
2  SM-003   1300     26    1
```

---

# CSV 파일과 데이터 탐색

## 가장 많이 사용하는 기능

```python
# CSV 파일 읽기
df = pd.read_csv("production_data.csv", encoding="utf-8")

# 기본 탐색
df.head()      # 처음 5행
df.shape       # (행수, 열수)
df.info()      # 데이터 타입과 결측치
df.describe()  # 기본 통계

# CSV 파일 저장
df.to_csv("output.csv", index=False)
```

---

# 이론 정리

## 핵심 포인트

### NumPy
- **배열(ndarray)**: 빠른 수치 계산
- **벡터화 연산**: 반복문 없이 계산
- **통계 함수**: mean, std, max, min

### Pandas
- **DataFrame**: 표 형태 데이터
- **CSV 읽기/쓰기**: read_csv, to_csv
- **데이터 탐색**: head, info, describe

---

# - 실습편 -

## 3차시

**제조 데이터 분석 실습**

---

# 실습 개요

## 센서 데이터와 품질 분석

### 실습 목표
- NumPy로 센서 데이터 계산
- Pandas로 CSV 파일 다루기
- 불량률 계산과 필터링

### 실습 환경
```python
import numpy as np
import pandas as pd
```

---

# 실습 1: NumPy 배열 다루기

## 온도 센서 데이터 분석

```python
import numpy as np

# 24시간 온도 측정 데이터
temperatures = np.array([
    82, 84, 85, 87, 88, 90, 92, 95,
    93, 91, 89, 86, 85, 84, 83, 82,
    81, 80, 79, 80, 81, 82, 83, 84
])

# 기본 통계
print(f"평균 온도: {temperatures.mean():.1f}도")
print(f"최고 온도: {temperatures.max()}도")
print(f"최저 온도: {temperatures.min()}도")

# 경고 기준(90도) 초과 횟수
over_90 = temperatures[temperatures > 90]
print(f"90도 초과: {len(over_90)}회")
```

---

# 실습 2: DataFrame 생성

## 생산 데이터 만들기

```python
import pandas as pd

production_data = {
    "날짜": ["2024-01-01", "2024-01-02", "2024-01-03",
             "2024-01-04", "2024-01-05"],
    "라인": [1, 2, 1, 2, 1],
    "생산량": [1200, 1150, 1300, 1180, 1250],
    "불량수": [24, 35, 26, 42, 25]
}

df = pd.DataFrame(production_data)
print(df)
print(f"\n데이터 크기: {df.shape}")
```

---

# 실습 3: 불량률 계산

## 새 열 추가하기

```python
# 불량률 계산
df["불량률"] = df["불량수"] / df["생산량"]

# 퍼센트로 표시
df["불량률_%"] = (df["불량률"] * 100).round(2)

print(df[["날짜", "생산량", "불량수", "불량률_%"]])
```

출력:
```
         날짜  생산량  불량수  불량률_%
0  2024-01-01   1200     24     2.00
1  2024-01-02   1150     35     3.04
2  2024-01-03   1300     26     2.00
3  2024-01-04   1180     42     3.56
4  2024-01-05   1250     25     2.00
```

---

# 실습 4: 조건 필터링

## 원하는 데이터만 추출

```python
# 불량률 3% 초과 데이터
high_defect = df[df["불량률"] > 0.03]
print("=== 불량률 3% 초과 ===")
print(high_defect)

# 1번 라인 데이터만
line1 = df[df["라인"] == 1]
print("\n=== 1번 라인 ===")
print(line1)

# 복합 조건: 1번 라인 AND 생산량 1200 이상
filtered = df[(df["라인"] == 1) & (df["생산량"] >= 1200)]
print("\n=== 1번 라인, 생산량 1200+ ===")
print(filtered)
```

---

# 실습 5: 그룹별 집계

## 라인별 통계 계산

```python
# 라인별 평균
line_avg = df.groupby("라인")[["생산량", "불량률"]].mean()
print("=== 라인별 평균 ===")
print(line_avg)

# 라인별 합계
line_sum = df.groupby("라인")["생산량"].sum()
print("\n=== 라인별 총 생산량 ===")
print(line_sum)
```

출력:
```
=== 라인별 평균 ===
      생산량    불량률
라인
1   1250.0  0.020000
2   1165.0  0.033000
```

---

# 실습 6: 종합 예제

## 일일 품질 리포트

```python
# 전체 요약
print("=" * 40)
print("       일일 품질 리포트")
print("=" * 40)
print(f"총 생산량: {df['생산량'].sum():,}개")
print(f"총 불량수: {df['불량수'].sum():,}개")
print(f"평균 불량률: {df['불량률'].mean()*100:.2f}%")

# 최고/최저 라인
best_line = df.groupby("라인")["불량률"].mean().idxmin()
worst_line = df.groupby("라인")["불량률"].mean().idxmax()
print(f"\n최우수 라인: {best_line}번")
print(f"개선 필요 라인: {worst_line}번")
print("=" * 40)
```

---

# 자주 하는 실수

## 주의할 점

```python
# 1. 단일 열 vs 다중 열 선택
df["생산량"]      # Series 반환
df[["생산량"]]    # DataFrame 반환

# 2. 조건 연결에서 and/or 대신 &/| 사용
df[(df["A"] > 1) & (df["B"] < 2)]  # 정답
df[(df["A"] > 1) and (df["B"] < 2)]  # 에러!

# 3. 조건마다 괄호로 감싸기 필수
df[(조건1) & (조건2)]  # 정답
df[조건1 & 조건2]      # 에러 가능
```

---

# 실습 정리

## 핵심 체크포인트

### NumPy
- [ ] np.array()로 배열 생성
- [ ] 통계 함수: mean, max, min, std
- [ ] 조건 필터링: arr[arr > 값]

### Pandas
- [ ] pd.DataFrame()으로 테이블 생성
- [ ] pd.read_csv()로 파일 읽기
- [ ] df.head(), df.info(), df.describe()
- [ ] df.groupby()로 그룹별 집계

---

# 다음 차시 예고

## 4차시: 데이터 요약과 시각화

### 학습 내용
- Matplotlib으로 기본 그래프 그리기
- 히스토그램, 상자그림, 산점도
- 제조 데이터 시각화 실습

### 준비물
- 오늘 배운 코드 복습
- NumPy, Pandas 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심

1. **NumPy**: 빠른 수치 계산, 통계 함수
2. **Pandas**: 표 데이터, CSV 파일, 그룹 집계
3. **실무 적용**: 불량률 계산, 라인별 분석

### 추가 학습 자료
- Pandas 공식 문서: https://pandas.pydata.org/docs/
- NumPy 공식 문서: https://numpy.org/doc/

---

# 감사합니다

## 3차시: 제조 데이터 다루기 기초

**다음 시간에 데이터를 시각화해봅시다!**
