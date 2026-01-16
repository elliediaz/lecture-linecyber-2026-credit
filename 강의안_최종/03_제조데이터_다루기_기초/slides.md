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
4. **실무 분석**: 불량률 계산, 라인별 분석을 수행한다

---

# 오늘의 진행 순서

## 이론 + 실습 (25-30분)

| 순서 | 내용 | 시간 |
|------|------|------|
| 1 | NumPy 소개 및 배열 연산 | 5분 |
| 2 | Pandas DataFrame 기초 | 5분 |
| 3 | 실습: 센서 데이터 분석 | 8분 |
| 4 | 실습: 생산 데이터 분석 | 7분 |
| 5 | 정리 및 Q&A | 5분 |

---

# 왜 NumPy와 Pandas인가?

## Python 기본 vs 데이터 분석 라이브러리

```python
# Python 리스트로 평균 계산 - 5줄
production = [1200, 1150, 1300, 1180, 1250]
total = 0
for x in production:
    total += x
average = total / len(production)

# NumPy로 평균 계산 - 1줄!
import numpy as np
production = np.array([1200, 1150, 1300, 1180, 1250])
average = production.mean()
```

> **속도**: NumPy는 Python 리스트보다 최대 **100배 빠름**

---

# NumPy vs Pandas 역할

## 언제 무엇을 사용하나?

| 구분 | NumPy | Pandas |
|------|-------|--------|
| **데이터 형태** | 동일 타입 배열 | 표 형태 (다양한 타입) |
| **주 용도** | 수치 계산, 행렬 연산 | 데이터 분석, 조작 |
| **핵심 객체** | ndarray | DataFrame, Series |
| **제조 예시** | 센서값 계산 | 생산일지 분석 |

> **둘 다 필수!** NumPy → 빠른 계산, Pandas → 데이터 관리

---

# NumPy 소개

## NumPy = Numerical Python

| 특징 | 설명 |
|------|------|
| **다차원 배열** | ndarray 객체로 행렬 연산 |
| **벡터화 연산** | 반복문 없이 빠른 계산 |
| **브로드캐스팅** | 크기가 다른 배열 간 연산 |
| **통계 함수** | mean, std, max, min 등 내장 |
| **과학 계산** | 선형대수, 푸리에 변환 |

```python
import numpy as np
print(np.__version__)  # 버전 확인
```

---

# NumPy 배열 생성

## 다양한 방법으로 배열 만들기

```python
import numpy as np

# 1. 리스트에서 생성
temps = np.array([82, 85, 88, 95, 84])

# 2. 2차원 배열 (행렬)
sensor_matrix = np.array([
    [82, 45, 1.2],  # 온도, 습도, 압력
    [85, 48, 1.1],
    [88, 52, 1.3]
])

# 3. 특수 배열
zeros = np.zeros(5)           # [0, 0, 0, 0, 0]
ones = np.ones(5)             # [1, 1, 1, 1, 1]
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 균등 간격 5개
```

---

# NumPy 배열 속성

## 배열 정보 확인하기

```python
data = np.array([[1, 2, 3], [4, 5, 6]])

print(f"차원: {data.ndim}")        # 2
print(f"형상: {data.shape}")       # (2, 3) - 2행 3열
print(f"크기: {data.size}")        # 6개 요소
print(f"타입: {data.dtype}")       # int64
```

```
┌─────────────────────────┐
│  배열 시각화              │
│  [[1, 2, 3],            │
│   [4, 5, 6]]            │
│                         │
│  shape = (2, 3)         │
│  2행 × 3열               │
└─────────────────────────┘
```

---

# NumPy 인덱싱과 슬라이싱

## 원하는 데이터 접근하기

```python
arr = np.array([10, 20, 30, 40, 50])

# 인덱싱 (0부터 시작)
print(arr[0])     # 10 (첫 번째)
print(arr[-1])    # 50 (마지막)

# 슬라이싱
print(arr[1:4])   # [20, 30, 40]
print(arr[:3])    # [10, 20, 30]
print(arr[2:])    # [30, 40, 50]

# 조건 인덱싱 (불리언 마스크)
print(arr[arr > 25])  # [30, 40, 50]
```

---

# NumPy 2차원 배열 접근

## 행/열 선택하기

```python
data = np.array([
    [82, 45, 1.2],   # 0행: 온도, 습도, 압력
    [85, 48, 1.1],   # 1행
    [88, 52, 1.3]    # 2행
])

# 단일 요소
print(data[0, 0])     # 82 (0행 0열)
print(data[1, 2])     # 1.1 (1행 2열)

# 행 선택
print(data[0, :])     # [82, 45, 1.2] (0행 전체)

# 열 선택
print(data[:, 0])     # [82, 85, 88] (온도 열)
```

---

# NumPy 벡터화 연산

## 반복문 없이 빠른 계산

```python
temps = np.array([82, 85, 88, 95, 84])

# 스칼라 연산 (모든 요소에 적용)
print(temps + 10)     # [92, 95, 98, 105, 94]
print(temps * 1.1)    # [90.2, 93.5, ...]
print(temps - 80)     # [2, 5, 8, 15, 4]

# 단위 변환: 섭씨 → 화씨
fahrenheit = temps * 9/5 + 32

# 배열 간 연산
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)          # [5, 7, 9]
print(a * b)          # [4, 10, 18]
```

---

# NumPy 통계 함수

## 제조 데이터 기본 통계

```python
production = np.array([1200, 1150, 1300, 1180, 1250])

print(f"합계: {np.sum(production)}")        # 6080
print(f"평균: {np.mean(production)}")       # 1216.0
print(f"중앙값: {np.median(production)}")   # 1200.0
print(f"표준편차: {np.std(production):.2f}") # 53.45
print(f"분산: {np.var(production):.2f}")    # 2856.0
print(f"최대값: {np.max(production)}")      # 1300
print(f"최소값: {np.min(production)}")      # 1150
print(f"범위: {np.ptp(production)}")        # 150 (max-min)
```

---

# NumPy 조건 필터링

## 원하는 데이터만 추출

```python
temps = np.array([82, 85, 88, 95, 84, 91, 86])

# 조건 마스크
mask = temps > 90
print(mask)  # [False, False, False, True, False, True, False]

# 조건에 맞는 값 추출
high_temps = temps[temps > 90]
print(f"90도 초과: {high_temps}")  # [95, 91]
print(f"개수: {len(high_temps)}")  # 2

# 조건 카운트
count = np.sum(temps > 90)  # True = 1로 계산
print(f"90도 초과 횟수: {count}")  # 2

# 조건에 따른 값 변경
adjusted = np.where(temps > 90, temps - 5, temps)
print(adjusted)  # [82, 85, 88, 90, 84, 86, 86]
```

---

# Pandas 소개

## 표 형태 데이터의 표준 도구

| 특징 | 설명 |
|------|------|
| **DataFrame** | Excel처럼 행/열로 구성된 2D 데이터 |
| **Series** | 1차원 데이터 (DataFrame의 한 열) |
| **파일 지원** | CSV, Excel, SQL, JSON 등 |
| **데이터 조작** | 필터링, 정렬, 그룹화, 병합 |
| **통계 분석** | 기술통계, 집계 함수 |

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
    "제품코드": ["SM-001", "SM-002", "SM-003", "SM-004", "SM-005"],
    "생산량": [1200, 1150, 1300, 1180, 1250],
    "불량수": [24, 35, 26, 42, 25],
    "라인": [1, 2, 1, 2, 1]
}

df = pd.DataFrame(data)
print(df)
```

```
  제품코드  생산량  불량수  라인
0  SM-001   1200     24    1
1  SM-002   1150     35    2
2  SM-003   1300     26    1
3  SM-004   1180     42    2
4  SM-005   1250     25    1
```

---

# DataFrame 기본 탐색

## 데이터 파악하기

```python
# CSV 파일 읽기
df = pd.read_csv("production_data.csv", encoding="utf-8")

# 기본 탐색
df.head()        # 처음 5행
df.tail()        # 마지막 5행
df.shape         # (행수, 열수)
df.columns       # 열 이름 목록
df.dtypes        # 각 열의 데이터 타입
df.info()        # 전체 정보 요약
df.describe()    # 수치형 열 통계

# 결측치 확인
df.isnull().sum()
```

---

# DataFrame 열 선택

## 원하는 열만 추출

```python
# 단일 열 선택 - Series 반환
production = df["생산량"]
print(type(production))  # pandas.Series

# 다중 열 선택 - DataFrame 반환
subset = df[["제품코드", "생산량", "불량수"]]
print(type(subset))  # pandas.DataFrame

# 열 접근 방법 비교
df["생산량"]      # 대괄호 표기 (권장)
df.생산량        # 점 표기 (열 이름에 공백 없을 때)
```

> **주의**: 단일 열은 Series, 다중 열은 DataFrame!

---

# DataFrame 행 선택

## loc과 iloc의 차이

```python
# loc: 라벨 기반 선택 (끝 포함)
df.loc[0]        # 0번 인덱스 행
df.loc[0:2]      # 0, 1, 2번 행 (끝 포함!)
df.loc[0, "생산량"]  # 0번 행의 "생산량" 값

# iloc: 정수 인덱스 기반 (끝 미포함)
df.iloc[0]       # 0번 행
df.iloc[0:2]     # 0, 1번 행 (끝 미포함!)
df.iloc[0, 1]    # 0번 행, 1번 열 값
```

```
loc[0:2]  → 0, 1, 2번 행 (라벨 기반, 끝 포함)
iloc[0:2] → 0, 1번 행 (정수 기반, 끝 미포함)
```

---

# DataFrame 조건 필터링

## 원하는 데이터만 추출

```python
# 단일 조건
high_prod = df[df["생산량"] > 1200]

# 복합 조건 (AND: &, OR: |)
# 주의: 각 조건을 괄호로 감싸야 함!
filtered = df[(df["생산량"] > 1100) & (df["라인"] == 1)]

# OR 조건
line_12 = df[(df["라인"] == 1) | (df["라인"] == 2)]

# isin() 활용
line_12 = df[df["라인"].isin([1, 2])]

# 문자열 조건
a_products = df[df["제품코드"].str.startswith("SM")]
```

---

# DataFrame 새 열 추가

## 파생 변수 만들기

```python
# 불량률 계산
df["불량률"] = df["불량수"] / df["생산량"]

# 퍼센트로 표시
df["불량률_%"] = (df["불량률"] * 100).round(2)

# 양품수 계산
df["양품수"] = df["생산량"] - df["불량수"]

# 조건에 따른 등급 분류
def classify(rate):
    if rate <= 0.02:
        return "A"
    elif rate <= 0.03:
        return "B"
    else:
        return "C"

df["등급"] = df["불량률"].apply(classify)
```

---

# DataFrame 그룹별 집계

## groupby로 그룹 분석

```python
# 라인별 평균 생산량
df.groupby("라인")["생산량"].mean()

# 라인별 여러 통계
df.groupby("라인").agg({
    "생산량": ["count", "mean", "sum"],
    "불량수": ["mean", "sum"],
    "불량률": "mean"
})

# 등급별 집계
df.groupby("등급")[["생산량", "불량률"]].mean()

# 복수 그룹
df.groupby(["라인", "등급"])["생산량"].mean()
```

---

# DataFrame 정렬

## sort_values로 정렬하기

```python
# 단일 열 오름차순
df.sort_values("생산량")

# 단일 열 내림차순
df.sort_values("생산량", ascending=False)

# 다중 열 정렬
df.sort_values(["라인", "생산량"], ascending=[True, False])

# 인덱스 정렬
df.sort_index()

# 상위 N개 추출
df.nlargest(3, "생산량")  # 생산량 상위 3개
df.nsmallest(3, "불량률")  # 불량률 하위 3개
```

---

# 결측치 처리

## 누락된 데이터 다루기

```python
# 결측치 확인
df.isnull().sum()        # 열별 결측치 개수
df.isnull().any()        # 결측치 있는 열
df[df.isnull().any(axis=1)]  # 결측치 있는 행

# 결측치 제거
df.dropna()              # 결측치 있는 행 제거
df.dropna(subset=["생산량"])  # 특정 열 기준

# 결측치 채우기
df.fillna(0)             # 0으로 채우기
df.fillna(df.mean())     # 평균으로 채우기
df["생산량"].fillna(df["생산량"].median())  # 중앙값
```

---

# CSV 파일 읽기/쓰기

## 가장 많이 사용하는 기능

```python
# CSV 읽기
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", encoding="utf-8")
df = pd.read_csv("data.csv", encoding="cp949")  # 한글

# 옵션
df = pd.read_csv("data.csv",
    header=0,           # 첫 번째 행이 헤더
    index_col="날짜",   # 인덱스 열 지정
    parse_dates=["날짜"], # 날짜 파싱
    na_values=["NA", ""]  # 결측치로 인식할 값
)

# CSV 저장
df.to_csv("output.csv", index=False, encoding="utf-8")
```

---

# 이론 정리

## 핵심 포인트

### NumPy
- **배열(ndarray)**: 빠른 수치 계산
- **벡터화 연산**: 반복문 없이 한 번에 계산
- **통계 함수**: mean, std, max, min, sum
- **조건 필터링**: arr[arr > 값]

### Pandas
- **DataFrame**: 표 형태 데이터
- **CSV 읽기/쓰기**: read_csv, to_csv
- **데이터 탐색**: head, info, describe
- **필터링/집계**: 조건 선택, groupby

---

# - 실습편 -

## 3차시

**제조 데이터 분석 실습**

---

# 실습 개요

## 센서 데이터와 생산 데이터 분석

### 실습 목표
- NumPy로 센서 데이터 통계 분석
- Pandas로 생산 데이터 다루기
- 불량률 계산과 라인별 분석
- 이상 데이터 탐지

### 실습 환경
```python
import numpy as np
import pandas as pd
```

---

# 실습 1: NumPy 센서 데이터 분석

## 24시간 온도 모니터링

```python
import numpy as np

# 24시간 온도 데이터
temps = np.array([
    82, 84, 85, 87, 88, 90, 92, 95,
    93, 91, 89, 86, 85, 84, 83, 82,
    81, 80, 79, 80, 81, 82, 83, 84
])

print(f"평균 온도: {temps.mean():.1f}도")
print(f"최고 온도: {temps.max()}도 ({temps.argmax()}시)")
print(f"최저 온도: {temps.min()}도 ({temps.argmin()}시)")
print(f"표준편차: {temps.std():.2f}")

# 경고 (90도 초과) 분석
warnings = temps[temps > 90]
print(f"경고 횟수: {len(warnings)}회")
print(f"경고 시간대: {np.where(temps > 90)[0]}시")
```

---

# 실습 2: DataFrame 생성 및 탐색

## 일주일 생산 데이터

```python
import pandas as pd

data = {
    "날짜": pd.date_range("2024-12-09", periods=7),
    "라인": [1, 2, 1, 2, 1, 2, 1],
    "생산량": [1200, 1150, 1300, 1180, 1250, 1220, 1280],
    "불량수": [24, 35, 26, 42, 25, 30, 22]
}

df = pd.DataFrame(data)
print(df)
print(f"\n데이터 크기: {df.shape}")
print(f"\n기본 통계:\n{df.describe()}")
```

---

# 실습 3: 불량률 계산

## 파생 변수 추가

```python
# 불량률 계산
df["불량률"] = df["불량수"] / df["생산량"]
df["불량률_%"] = (df["불량률"] * 100).round(2)

# 양품수
df["양품수"] = df["생산량"] - df["불량수"]

# 등급 분류
df["등급"] = df["불량률"].apply(
    lambda x: "A" if x <= 0.02 else ("B" if x <= 0.03 else "C")
)

print(df[["날짜", "라인", "생산량", "불량수", "불량률_%", "등급"]])
```

---

# 실습 4: 조건 필터링

## 원하는 데이터만 추출

```python
# 불량률 3% 초과
high_defect = df[df["불량률"] > 0.03]
print("=== 불량률 3% 초과 ===")
print(high_defect[["날짜", "라인", "불량률_%"]])

# 1번 라인 데이터
line1 = df[df["라인"] == 1]
print("\n=== 1번 라인 ===")
print(line1[["날짜", "생산량", "불량률_%"]])

# 복합 조건: A등급 AND 생산량 1200 이상
excellent = df[(df["등급"] == "A") & (df["생산량"] >= 1200)]
print("\n=== A등급 & 생산량 1200+ ===")
print(excellent)
```

---

# 실습 5: 그룹별 집계

## 라인별 성과 분석

```python
# 라인별 평균
line_stats = df.groupby("라인").agg({
    "생산량": ["count", "mean", "sum"],
    "불량수": "sum",
    "불량률": "mean"
}).round(3)

print("=== 라인별 통계 ===")
print(line_stats)

# 등급별 분포
print("\n=== 등급별 건수 ===")
print(df["등급"].value_counts())

# 최우수/개선 필요 라인
best = df.groupby("라인")["불량률"].mean().idxmin()
worst = df.groupby("라인")["불량률"].mean().idxmax()
print(f"\n최우수 라인: {best}번")
print(f"개선 필요: {worst}번")
```

---

# 실습 6: 종합 보고서

## 일일 품질 리포트

```python
print("=" * 50)
print("         주간 품질 리포트")
print("=" * 50)
print(f"분석 기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
print("-" * 50)
print(f"총 생산량: {df['생산량'].sum():,}개")
print(f"총 불량수: {df['불량수'].sum():,}개")
print(f"평균 불량률: {df['불량률'].mean()*100:.2f}%")
print("-" * 50)
print("라인별 성과:")
for line in df["라인"].unique():
    line_data = df[df["라인"] == line]
    rate = line_data["불량률"].mean() * 100
    print(f"  {line}번 라인: {rate:.2f}%")
print("-" * 50)
print(f"A등급: {(df['등급']=='A').sum()}건")
print(f"B등급: {(df['등급']=='B').sum()}건")
print(f"C등급: {(df['등급']=='C').sum()}건")
print("=" * 50)
```

---

# 자주 하는 실수

## 주의할 점

```python
# 1. 단일 열 vs 다중 열 선택
df["생산량"]      # Series 반환
df[["생산량"]]    # DataFrame 반환

# 2. 조건 연결에서 and/or 대신 &/| 사용
df[(df["A"] > 1) & (df["B"] < 2)]   # 정답
df[(df["A"] > 1) and (df["B"] < 2)] # 에러!

# 3. 조건마다 괄호로 감싸기 필수
df[(조건1) & (조건2)]   # 정답
df[조건1 & 조건2]       # 에러 가능

# 4. loc vs iloc 혼동
df.loc[0:2]    # 0, 1, 2번 행 (끝 포함)
df.iloc[0:2]   # 0, 1번 행 (끝 미포함)
```

---

# 실습 정리

## 핵심 체크포인트

### NumPy
- [ ] np.array()로 배열 생성
- [ ] 통계 함수: mean, max, min, std
- [ ] 조건 필터링: arr[arr > 값]
- [ ] argmax(), argmin()로 위치 찾기

### Pandas
- [ ] pd.DataFrame()으로 테이블 생성
- [ ] pd.read_csv()로 파일 읽기
- [ ] df.head(), df.info(), df.describe()
- [ ] df.groupby()로 그룹별 집계
- [ ] 새 열 추가: df["새열"] = 계산식

---

# 다음 차시 예고

## 4차시: 공개 데이터셋 확보 및 데이터 생태계 이해

### 학습 내용
- 공공데이터포털, AI 허브, Kaggle, UCI
- 데이터 플랫폼별 특성과 활용법
- 데이터셋 구조와 용어 정리
- 실제 데이터 다운로드 및 탐색

> 오늘 배운 Pandas로 실제 공개 데이터를 분석해봅니다!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **NumPy**: 빠른 수치 계산, 통계 함수, 조건 필터링
2. **Pandas**: 표 데이터, CSV 파일, 그룹 집계
3. **실무 적용**: 불량률 계산, 라인별 분석, 품질 보고서

### 추가 학습 자료
- Pandas 공식 문서: https://pandas.pydata.org/docs/
- NumPy 공식 문서: https://numpy.org/doc/

---

# 감사합니다

## 3차시: 제조 데이터 다루기 기초

**수고하셨습니다!**
**다음 시간에 공개 데이터셋을 탐색해봅시다!**
