---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 3차시'
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

# 데이터 다루기 기초

## 3차시 | AI 기초체력훈련 (Pre AI-Campus)

**NumPy와 Pandas로 데이터 분석 시작하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **NumPy 배열**의 개념과 기본 연산을 이해한다
2. **Pandas DataFrame**으로 표 형태 데이터를 다룬다
3. **CSV 파일**을 불러오고 기본 탐색을 수행한다

---

# 왜 NumPy와 Pandas인가?

## Python 기본 vs 데이터 분석 라이브러리

```python
# Python 리스트로 평균 계산
data = [1, 2, 3, 4, 5]
total = 0
for x in data:
    total += x
average = total / len(data)

# NumPy로 평균 계산
import numpy as np
data = np.array([1, 2, 3, 4, 5])
average = data.mean()  # 한 줄로 끝!
```

> **속도**: NumPy는 Python 리스트보다 최대 100배 빠름

---

# NumPy 소개

## NumPy = Numerical Python

- 수치 계산을 위한 핵심 라이브러리
- 다차원 배열(ndarray) 객체 제공
- 벡터화 연산으로 빠른 계산
- 선형대수, 통계 함수 내장

### 설치 확인
```python
import numpy as np
print(np.__version__)  # 1.24.x
```

---

# NumPy 배열 생성

## 리스트에서 배열 만들기

```python
import numpy as np

# 1차원 배열
temperatures = np.array([82, 85, 88, 95, 84])
print(temperatures)
# [82 85 88 95 84]

# 2차원 배열 (행렬)
sensor_data = np.array([
    [82, 45, 1.2],   # 온도, 습도, 압력
    [85, 48, 1.1],
    [88, 52, 1.3]
])
print(sensor_data.shape)  # (3, 3)
```

---

# NumPy 유용한 배열 생성

## 특수 배열 만들기

```python
# 0으로 채운 배열
zeros = np.zeros((3, 4))

# 1로 채운 배열
ones = np.ones((2, 3))

# 연속된 숫자
sequence = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 균등 간격
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# 난수 배열
random = np.random.rand(3, 3)  # 0~1 균등분포
```

---

# NumPy 인덱싱과 슬라이싱

## 배열 요소 접근

```python
arr = np.array([10, 20, 30, 40, 50])

# 인덱싱
print(arr[0])    # 10
print(arr[-1])   # 50

# 슬라이싱
print(arr[1:4])  # [20 30 40]

# 조건 인덱싱 (불리언 마스크)
print(arr[arr > 25])  # [30 40 50]
```

---

# NumPy 2차원 배열 접근

## 행/열 선택

```python
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 특정 요소
print(data[0, 2])    # 3 (0행 2열)

# 행 선택
print(data[1, :])    # [4 5 6] (1행 전체)

# 열 선택
print(data[:, 0])    # [1 4 7] (0열 전체)
```

---

# NumPy 기본 연산

## 벡터화 연산 (Broadcasting)

```python
temps = np.array([82, 85, 88, 95, 84])

# 스칼라 연산 (모든 요소에 적용)
temps_celsius = (temps - 32) * 5/9
print(temps_celsius)

# 배열 간 연산
weights = np.array([1.0, 1.2, 1.1, 0.9, 1.0])
weighted = temps * weights

# 비교 연산
high_temp = temps > 90
print(high_temp)  # [False False False True False]
```

---

# NumPy 통계 함수

## 기본 통계량 계산

```python
data = np.array([1200, 1150, 1300, 1180, 1250])

print(f"합계: {np.sum(data)}")        # 6080
print(f"평균: {np.mean(data)}")       # 1216.0
print(f"표준편차: {np.std(data):.2f}")  # 53.45
print(f"최대값: {np.max(data)}")      # 1300
print(f"최소값: {np.min(data)}")      # 1150
print(f"중앙값: {np.median(data)}")   # 1200.0
```

---

# NumPy 축(axis) 개념

## 2차원 배열의 행/열 방향 연산

```python
data = np.array([
    [100, 200, 150],  # 라인1
    [120, 180, 160],  # 라인2
    [110, 190, 140]   # 라인3
])

# axis=0: 열 방향 (세로로 계산)
col_mean = np.mean(data, axis=0)
print(f"제품별 평균: {col_mean}")  # [110. 190. 150.]

# axis=1: 행 방향 (가로로 계산)
row_mean = np.mean(data, axis=1)
print(f"라인별 평균: {row_mean}")  # [150. 153.33 146.67]
```

---

# Pandas 소개

## Pandas = Panel Data

- 표 형태 데이터 처리의 표준
- Excel처럼 행과 열로 구성된 DataFrame
- CSV, Excel, SQL 등 다양한 형식 지원
- 데이터 분석에 필수적인 도구

### 설치 확인
```python
import pandas as pd
print(pd.__version__)  # 2.x.x
```

---

# DataFrame 생성

## 딕셔너리에서 DataFrame 만들기

```python
import pandas as pd

data = {
    "제품코드": ["A001", "A002", "A003"],
    "생산량": [1200, 1150, 1300],
    "불량수": [24, 35, 26],
    "라인": [1, 2, 1]
}

df = pd.DataFrame(data)
print(df)
```

```
  제품코드  생산량  불량수  라인
0  A001   1200     24    1
1  A002   1150     35    2
2  A003   1300     26    1
```

---

# CSV 파일 읽기/쓰기

## 가장 많이 사용하는 기능

```python
# CSV 파일 읽기
df = pd.read_csv("production_data.csv")

# 인코딩 지정 (한글 파일)
df = pd.read_csv("data.csv", encoding="utf-8")
# 또는
df = pd.read_csv("data.csv", encoding="cp949")

# CSV 파일 저장
df.to_csv("output.csv", index=False)
```

---

# DataFrame 기본 탐색

## 데이터 확인하기

```python
# 처음 5행 보기
df.head()

# 마지막 5행 보기
df.tail()

# 데이터 크기
df.shape  # (행수, 열수)

# 열 이름
df.columns

# 데이터 타입
df.dtypes

# 기본 통계
df.describe()
```

---

# DataFrame 정보 확인

## df.info()로 전체 파악

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 5 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   제품코드   1000 non-null   object
 1   생산량    1000 non-null   int64
 2   불량수    985 non-null    float64
 3   라인     1000 non-null   int64
 4   날짜     1000 non-null   object
dtypes: float64(1), int64(2), object(2)
memory usage: 39.2 KB
```

---

# 열(Column) 선택

## 특정 열 접근하기

```python
# 단일 열 선택 (Series 반환)
production = df["생산량"]
print(production)

# 여러 열 선택 (DataFrame 반환)
subset = df[["제품코드", "생산량", "불량수"]]
print(subset)

# 점 표기법 (열 이름에 공백 없을 때)
production = df.생산량
```

---

# 행(Row) 선택

## loc와 iloc

```python
# loc: 라벨(이름) 기반 선택
df.loc[0]           # 0번 행
df.loc[0:2]         # 0~2번 행 (포함)
df.loc[0, "생산량"]  # 0번 행의 생산량 열

# iloc: 정수 인덱스 기반 선택
df.iloc[0]          # 첫 번째 행
df.iloc[0:2]        # 0~1번 행 (미포함)
df.iloc[0, 1]       # 0행 1열

# 핵심 차이: loc는 끝 포함, iloc는 끝 미포함
```

---

# 조건 필터링

## 원하는 데이터만 추출

```python
# 단일 조건
high_production = df[df["생산량"] > 1200]

# 복합 조건 (and: &, or: |)
filtered = df[(df["생산량"] > 1100) & (df["라인"] == 1)]

# 특정 값 포함
line1_2 = df[df["라인"].isin([1, 2])]

# 문자열 조건
a_products = df[df["제품코드"].str.startswith("A")]
```

---

# 새 열 추가

## 계산 결과로 열 생성

```python
# 불량률 계산
df["불량률"] = df["불량수"] / df["생산량"]

# 조건에 따른 분류
df["등급"] = df["불량률"].apply(
    lambda x: "A" if x < 0.02 else ("B" if x < 0.05 else "C")
)

# 여러 열 조합
df["총생산"] = df["라인1"] + df["라인2"] + df["라인3"]
```

---

# 기본 통계 계산

## 열별 통계

```python
# 특정 열 통계
print(df["생산량"].mean())   # 평균
print(df["생산량"].sum())    # 합계
print(df["생산량"].std())    # 표준편차
print(df["생산량"].max())    # 최대값
print(df["생산량"].min())    # 최소값

# 전체 수치형 열 통계
df.describe()
```

---

# 그룹별 집계

## groupby 활용

```python
# 라인별 평균 생산량
line_avg = df.groupby("라인")["생산량"].mean()
print(line_avg)

# 라인별 여러 통계
line_stats = df.groupby("라인").agg({
    "생산량": ["mean", "sum"],
    "불량수": ["mean", "sum"]
})
print(line_stats)
```

---

# 결측치 처리

## 누락된 데이터 다루기

```python
# 결측치 확인
print(df.isnull().sum())

# 결측치 있는 행 제거
df_clean = df.dropna()

# 결측치 채우기
df["불량수"] = df["불량수"].fillna(0)  # 0으로 채우기
df["불량수"] = df["불량수"].fillna(df["불량수"].mean())  # 평균으로

# 결측치 있는 행 확인
df[df["불량수"].isnull()]
```

---

# 정렬

## 데이터 정렬하기

```python
# 단일 열 기준 정렬 (오름차순)
df_sorted = df.sort_values("생산량")

# 내림차순 정렬
df_sorted = df.sort_values("생산량", ascending=False)

# 여러 열 기준 정렬
df_sorted = df.sort_values(
    ["라인", "생산량"],
    ascending=[True, False]
)

# 인덱스 정렬
df_sorted = df.sort_index()
```

---

# 실습: 제조 데이터 분석

## 예제 데이터

```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    "날짜": pd.date_range("2024-01-01", periods=30),
    "라인": np.random.choice([1, 2, 3], 30),
    "생산량": np.random.randint(1100, 1400, 30),
    "불량수": np.random.randint(10, 50, 30)
})

df["불량률"] = df["불량수"] / df["생산량"]
print(df.head())
```

---

# 실습: 기본 분석

## 분석 코드

```python
# 1. 기본 통계
print("=== 기본 통계 ===")
print(df.describe())

# 2. 라인별 분석
print("\n=== 라인별 평균 ===")
print(df.groupby("라인")[["생산량", "불량률"]].mean())

# 3. 이상치 탐지 (불량률 5% 초과)
print("\n=== 이상 데이터 ===")
abnormal = df[df["불량률"] > 0.05]
print(abnormal)
```

---

# 학습 정리

## 오늘 배운 내용

### 1. NumPy
- 배열 생성과 인덱싱
- 벡터화 연산 (빠른 계산)
- 통계 함수 (mean, std, max, min)

### 2. Pandas
- DataFrame 생성과 CSV 읽기/쓰기
- 열/행 선택과 조건 필터링
- 기본 통계와 그룹별 집계

---

# 다음 차시 예고

## 4차시: 기술통계의 시각적 이해

- Matplotlib으로 기본 그래프 그리기
- 히스토그램, 상자그림, 산점도
- 제조 데이터 시각화 실습

### 과제 (선택)
- 오늘 배운 코드 직접 실행해보기
- 자신만의 CSV 파일 만들어서 불러오기

---

# Q&A

## 질문이 있으신가요?

### 자주 하는 실수
- `df["열이름"]` vs `df[["열이름"]]` 차이
  - 전자는 Series, 후자는 DataFrame 반환
- 조건 필터링에서 `and` 대신 `&` 사용
- 열 이름에 한글/공백 있으면 점 표기법 불가

### 추가 학습 자료
- Pandas 공식 문서: https://pandas.pydata.org/docs/
- NumPy 공식 문서: https://numpy.org/doc/

---

# 감사합니다

## AI 기초체력훈련 3차시

**데이터 다루기 기초 (NumPy, Pandas)**

다음 시간에 만나요!
