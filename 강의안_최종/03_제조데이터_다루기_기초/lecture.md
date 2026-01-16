# 3차시: 제조 데이터 다루기 기초

## 학습 목표

본 차시를 마치면 다음을 수행할 수 있음:

1. **NumPy 배열**의 개념과 기본 연산을 이해함
2. **Pandas DataFrame**으로 표 형태 데이터를 다룸
3. **CSV 파일**을 불러오고 제조 데이터 탐색을 수행함
4. **실무 분석**: 불량률 계산, 라인별 분석을 수행함

---

## 강의 구성

| 순서 | 내용 | 시간 |
|------|------|------|
| Part 1 | NumPy 소개 및 배열 연산 | 5분 |
| Part 2 | Pandas DataFrame 기초 | 5분 |
| Part 3 | 실습: 센서 데이터 분석 | 8분 |
| Part 4 | 실습: 생산 데이터 분석 | 7분 |
| Part 5 | 정리 및 Q&A | 5분 |

---

## Part 1: NumPy 소개 및 배열 연산

### 개념 설명

**왜 NumPy와 Pandas인가?**

Python 기본 리스트 대신 NumPy를 사용하면 코드가 간결해지고 속도가 빨라짐.

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

> NumPy는 Python 리스트보다 최대 100배 빠름

**NumPy vs Pandas 역할**

| 구분 | NumPy | Pandas |
|------|-------|--------|
| 데이터 형태 | 동일 타입 배열 | 표 형태 (다양한 타입) |
| 주 용도 | 수치 계산, 행렬 연산 | 데이터 분석, 조작 |
| 핵심 객체 | ndarray | DataFrame, Series |
| 제조 예시 | 센서값 계산 | 생산일지 분석 |

> 둘 다 필수임. NumPy는 빠른 계산, Pandas는 데이터 관리에 사용함

**NumPy 특징**

| 특징 | 설명 |
|------|------|
| 다차원 배열 | ndarray 객체로 행렬 연산 |
| 벡터화 연산 | 반복문 없이 빠른 계산 |
| 브로드캐스팅 | 크기가 다른 배열 간 연산 |
| 통계 함수 | mean, std, max, min 등 내장 |
| 과학 계산 | 선형대수, 푸리에 변환 |

### 실습 코드

#### 1-1. 배열 생성

```python
import numpy as np

# 1. 리스트에서 생성
temps = np.array([82, 85, 88, 95, 84])
print(f"온도 데이터: {temps}")
print(f"데이터 타입: {temps.dtype}")
print(f"배열 크기: {temps.shape}")
print(f"요소 개수: {temps.size}")

# 2. 2차원 배열 (행렬)
sensor_matrix = np.array([
    [82, 45, 1.2],  # 온도, 습도, 압력
    [85, 48, 1.1],
    [88, 52, 1.3]
])
print(f"\n센서 행렬:\n{sensor_matrix}")
print(f"형상 (shape): {sensor_matrix.shape}")  # (3, 3) = 3행 3열

# 3. 특수 배열
print(f"\nzeros(5): {np.zeros(5)}")
print(f"ones(5): {np.ones(5)}")
print(f"arange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"linspace(0, 1, 5): {np.linspace(0, 1, 5)}")
```

**결과 해설**: `np.array()`로 리스트를 배열로 변환함. `.shape`는 배열의 형태(행, 열)를, `.dtype`는 데이터 타입을 반환함. `zeros`, `ones`, `arange`, `linspace`는 특정 패턴의 배열을 생성하는 함수임.

#### 1-2. 인덱싱과 슬라이싱

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"원본 배열: {arr}")

# 인덱싱 (0부터 시작)
print(f"arr[0] (첫 번째): {arr[0]}")
print(f"arr[-1] (마지막): {arr[-1]}")

# 슬라이싱
print(f"arr[1:4]: {arr[1:4]}")     # 인덱스 1~3
print(f"arr[:3]: {arr[:3]}")       # 처음~2번
print(f"arr[5:]: {arr[5:]}")       # 5번~끝

# 조건 인덱싱 (불리언 마스크)
mask = arr > 40
print(f"arr > 40: {mask}")
print(f"arr[arr > 40]: {arr[mask]}")
```

**결과 해설**: 배열 인덱싱은 0부터 시작함. 슬라이싱은 `[시작:끝]` 형태로 끝 인덱스는 포함하지 않음. 조건 인덱싱은 True인 위치의 값만 추출함.

#### 1-3. 2차원 배열 접근

```python
data = np.array([
    [82, 45, 1.2],   # 0행: 온도, 습도, 압력
    [85, 48, 1.1],   # 1행
    [88, 52, 1.3]    # 2행
])
print(f"원본:\n{data}")

# 단일 요소
print(f"data[0, 0] (0행 0열): {data[0, 0]}")
print(f"data[1, 2] (1행 2열): {data[1, 2]}")

# 행 선택
print(f"data[0, :] (0행 전체): {data[0, :]}")

# 열 선택
print(f"data[:, 0] (온도 열): {data[:, 0]}")
```

**결과 해설**: 2차원 배열은 `[행, 열]` 형태로 접근함. `:`는 전체를 의미함. `data[:, 0]`은 모든 행의 0번째 열(온도)을 추출함.

#### 1-4. 벡터화 연산

```python
temps = np.array([82, 85, 88, 95, 84])
print(f"원본 온도: {temps}")

# 스칼라 연산 (모든 요소에 적용)
print(f"temps + 10: {temps + 10}")
print(f"temps * 1.1: {temps * 1.1}")
print(f"temps - 80 (편차): {temps - 80}")

# 단위 변환: 섭씨 -> 화씨
fahrenheit = temps * 9/5 + 32
print(f"화씨 온도: {np.round(fahrenheit, 1)}")

# 배열 간 연산
production = np.array([1200, 1150, 1300, 1180, 1250])
defects = np.array([24, 35, 26, 42, 25])
defect_rates = defects / production
print(f"생산량: {production}")
print(f"불량수: {defects}")
print(f"불량률: {np.round(defect_rates, 4)}")
```

**결과 해설**: 벡터화 연산은 반복문 없이 배열 전체에 연산을 적용함. 스칼라(단일 값)와 배열의 연산, 배열 간 연산 모두 가능함. 코드가 간결해지고 실행 속도도 빨라짐.

#### 1-5. 통계 함수

```python
production = np.array([1200, 1150, 1300, 1180, 1250, 1320, 1100])
print(f"생산량 데이터: {production}")

# 기본 통계
print(f"합계 (sum): {np.sum(production)}")
print(f"평균 (mean): {np.mean(production):.2f}")
print(f"중앙값 (median): {np.median(production)}")
print(f"표준편차 (std): {np.std(production):.2f}")
print(f"분산 (var): {np.var(production):.2f}")
print(f"최대값 (max): {np.max(production)}")
print(f"최소값 (min): {np.min(production)}")
print(f"범위 (ptp): {np.ptp(production)}")

# 위치 찾기
print(f"최대값 위치: {np.argmax(production)}")
print(f"최소값 위치: {np.argmin(production)}")
```

**결과 해설**: NumPy는 다양한 통계 함수를 제공함. `ptp`(peak to peak)는 최대값과 최소값의 차이임. `argmax`, `argmin`은 최대/최소값의 인덱스(위치)를 반환함.

#### 1-6. 조건 필터링과 처리

```python
temps = np.array([82, 85, 88, 95, 84, 91, 86])
print(f"온도 데이터: {temps}")

# 조건 마스크
mask = temps > 90
print(f"temps > 90: {mask}")

# 조건에 맞는 값 추출
high_temps = temps[temps > 90]
print(f"90도 초과: {high_temps}")
print(f"개수: {len(high_temps)}")

# 조건 카운트
count = np.sum(temps > 90)  # True = 1로 계산
print(f"90도 초과 횟수: {count}")

# np.where: 조건에 따른 값 변경
status = np.where(temps > 90, "경고", "정상")
print(f"상태: {status}")

# 온도 조정 (90도 초과 시 5도 감소)
adjusted = np.where(temps > 90, temps - 5, temps)
print(f"조정 후: {adjusted}")
```

**결과 해설**: 조건 필터링으로 원하는 데이터만 추출할 수 있음. `np.where(조건, 참일때, 거짓일때)`는 조건에 따라 다른 값을 반환하는 함수로, 데이터 변환에 유용함.

---

## Part 2: Pandas DataFrame 기초

### 개념 설명

**Pandas 특징**

| 특징 | 설명 |
|------|------|
| DataFrame | Excel처럼 행/열로 구성된 2D 데이터 |
| Series | 1차원 데이터 (DataFrame의 한 열) |
| 파일 지원 | CSV, Excel, SQL, JSON 등 |
| 데이터 조작 | 필터링, 정렬, 그룹화, 병합 |
| 통계 분석 | 기술통계, 집계 함수 |

```
┌─────────────────────────────────────────────────────────────┐
│                    DataFrame 구조                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│         제품코드    생산량    불량수    라인                   │
│     0   SM-001     1200      24       1      <- 행(row)      │
│     1   SM-002     1150      35       2                      │
│     2   SM-003     1300      26       1                      │
│                                                              │
│         열(column) ^                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**loc vs iloc 비교**

```
┌─────────────────────────────────────────────────────────────┐
│  loc[0:2]  -> 0, 1, 2번 행 (라벨 기반, 끝 포함)              │
│  iloc[0:2] -> 0, 1번 행 (정수 기반, 끝 미포함)               │
└─────────────────────────────────────────────────────────────┘
```

### 실습 코드

#### 2-1. DataFrame 생성

```python
import pandas as pd

# 딕셔너리에서 DataFrame 만들기
data = {
    "제품코드": ["SM-001", "SM-002", "SM-003", "SM-004", "SM-005"],
    "생산량": [1200, 1150, 1300, 1180, 1250],
    "불량수": [24, 35, 26, 42, 25],
    "라인": [1, 2, 1, 2, 1]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print(f"\n크기: {df.shape}")
print(f"열 이름: {list(df.columns)}")
```

**결과**:
```
  제품코드  생산량  불량수  라인
0  SM-001   1200     24    1
1  SM-002   1150     35    2
2  SM-003   1300     26    1
3  SM-004   1180     42    2
4  SM-005   1250     25    1
```

**결과 해설**: 딕셔너리의 키가 열 이름이 되고, 값 리스트가 각 열의 데이터가 됨. `.shape`는 (행수, 열수)를 반환함.

#### 2-2. 날짜 포함 DataFrame

```python
weekly_data = {
    "날짜": pd.date_range("2024-12-09", periods=7),
    "라인": [1, 2, 1, 2, 1, 2, 1],
    "생산량": [1200, 1150, 1300, 1180, 1250, 1220, 1280],
    "불량수": [24, 35, 26, 42, 25, 30, 22],
    "가동시간": [8.5, 8.0, 9.0, 7.5, 8.5, 8.0, 9.0]
}
df_week = pd.DataFrame(weekly_data)
print(df_week)
```

**결과 해설**: `pd.date_range()`로 연속적인 날짜를 생성할 수 있음. `periods=7`은 7일 동안의 날짜를 생성함.

#### 2-3. 기본 탐색 메서드

```python
# CSV 파일 읽기 (예시)
# df = pd.read_csv("production_data.csv", encoding="utf-8")

# 기본 탐색
print(df.head())        # 처음 5행
print(df.tail())        # 마지막 5행
print(df.shape)         # (행수, 열수)
print(df.columns)       # 열 이름 목록
print(df.dtypes)        # 각 열의 데이터 타입
print(df.info())        # 전체 정보 요약
print(df.describe())    # 수치형 열 통계

# 결측치 확인
print(df.isnull().sum())
```

**결과 해설**: `head()`와 `tail()`로 데이터의 앞뒤를 확인함. `info()`는 열별 데이터 타입과 결측치 정보를, `describe()`는 수치형 열의 기술통계를 제공함.

#### 2-4. 열 선택

```python
# 단일 열 선택 - Series 반환
production = df["생산량"]
print(type(production))  # pandas.Series

# 다중 열 선택 - DataFrame 반환
subset = df[["제품코드", "생산량", "불량수"]]
print(type(subset))  # pandas.DataFrame
```

**결과 해설**: 단일 열 선택 시 `df["열이름"]`은 Series를 반환하고, 다중 열 선택 시 `df[["열1", "열2"]]`는 DataFrame을 반환함. 대괄호의 개수에 주의해야 함.

#### 2-5. 행 선택 (loc vs iloc)

```python
# loc: 라벨 기반 선택 (끝 포함)
print(df.loc[0])           # 0번 인덱스 행
print(df.loc[0:2])         # 0, 1, 2번 행 (끝 포함!)
print(df.loc[0, "생산량"])  # 0번 행의 "생산량" 값

# iloc: 정수 인덱스 기반 (끝 미포함)
print(df.iloc[0])          # 0번 행
print(df.iloc[0:2])        # 0, 1번 행 (끝 미포함!)
print(df.iloc[0, 1])       # 0번 행, 1번 열 값
```

**결과 해설**: `loc`은 라벨(이름) 기반으로 선택하며 끝 인덱스를 포함함. `iloc`은 정수 인덱스 기반으로 선택하며 끝 인덱스를 포함하지 않음. 이 차이를 명확히 이해해야 함.

#### 2-6. 조건 필터링

```python
# 단일 조건
high_prod = df[df["생산량"] > 1200]
print("생산량 1200 초과:")
print(high_prod)

# 복합 조건 (AND: &, OR: |)
# 주의: 각 조건을 괄호로 감싸야 함!
filtered = df[(df["생산량"] > 1100) & (df["라인"] == 1)]
print("\n생산량 1100 초과 AND 1번 라인:")
print(filtered)

# OR 조건
line_12 = df[(df["라인"] == 1) | (df["라인"] == 2)]

# isin() 활용
line_12 = df[df["라인"].isin([1, 2])]
print("\n1, 2번 라인:")
print(line_12)
```

**결과 해설**: 조건 필터링 시 `and`, `or` 대신 `&`, `|`를 사용해야 함. 각 조건은 반드시 괄호로 감싸야 함. `isin()`은 여러 값 중 하나에 해당하는지 확인할 때 유용함.

#### 2-7. 새 열 추가

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
print(df)
```

**결과 해설**: 새 열은 `df["새열이름"] = 값`으로 추가함. `apply()` 함수를 사용하면 각 행에 함수를 적용하여 새로운 값을 생성할 수 있음.

#### 2-8. 그룹별 집계

```python
# 라인별 평균 생산량
print(df.groupby("라인")["생산량"].mean())

# 라인별 여러 통계
line_stats = df.groupby("라인").agg({
    "생산량": ["count", "mean", "sum"],
    "불량수": ["mean", "sum"],
    "불량률": "mean"
}).round(3)
print(line_stats)

# 등급별 집계
print(df.groupby("등급")[["생산량", "불량률"]].mean())

# 최우수/개선 필요 라인
best = df.groupby("라인")["불량률"].mean().idxmin()
worst = df.groupby("라인")["불량률"].mean().idxmax()
print(f"최우수 라인: {best}번")
print(f"개선 필요: {worst}번")
```

**결과 해설**: `groupby()`는 지정한 열 기준으로 데이터를 그룹화함. `agg()`로 여러 집계 함수를 동시에 적용할 수 있음. `idxmin()`, `idxmax()`는 최소/최대값의 인덱스(그룹명)를 반환함.

---

## Part 3: 실습 - 센서 데이터 분석

### 실습 코드

#### 3-1. 24시간 온도 모니터링

```python
import numpy as np

# 24시간 온도 데이터
temps = np.array([
    82, 84, 85, 87, 88, 90, 92, 95,
    93, 91, 89, 86, 85, 84, 83, 82,
    81, 80, 79, 80, 81, 82, 83, 84
])

print("=== 24시간 온도 분석 ===")
print(f"평균 온도: {temps.mean():.1f}도")
print(f"최고 온도: {temps.max()}도 ({temps.argmax()}시)")
print(f"최저 온도: {temps.min()}도 ({temps.argmin()}시)")
print(f"표준편차: {temps.std():.2f}")

# 경고 (90도 초과) 분석
warnings = temps[temps > 90]
print(f"\n경고 횟수: {len(warnings)}회")
print(f"경고 시간대: {np.where(temps > 90)[0]}시")
```

**결과 해설**: `argmax()`와 `argmin()`으로 최고/최저 온도가 발생한 시간을 찾음. `np.where(조건)[0]`은 조건을 만족하는 인덱스 배열을 반환함.

#### 3-2. 2차원 센서 데이터 분석

```python
# 축별 통계
data_2d = np.array([
    [100, 200, 150],  # 라인1: 오전, 오후, 야간
    [120, 180, 160],  # 라인2
    [110, 190, 140]   # 라인3
])
print("데이터:")
print(data_2d)
print(f"axis=0 (열별, 시간대별) 평균: {np.mean(data_2d, axis=0)}")
print(f"axis=1 (행별, 라인별) 평균: {np.mean(data_2d, axis=1)}")
```

**결과 해설**: `axis=0`은 열 방향(세로)으로 계산하여 각 열의 평균을 반환함. `axis=1`은 행 방향(가로)으로 계산하여 각 행의 평균을 반환함.

---

## Part 4: 실습 - 생산 데이터 분석

### 실습 코드

#### 4-1. 실제 데이터셋 활용 (Tips 데이터)

```python
import pandas as pd
import seaborn as sns

# seaborn의 tips 데이터셋 로드
tips = sns.load_dataset('tips')
print("[Tips 데이터셋 - 레스토랑 팁 데이터]")
print("변수: total_bill(총액), tip(팁), sex(성별), smoker(흡연여부), day(요일), time(시간), size(인원)")

# 기본 탐색
print("\n처음 5행:")
print(tips.head())

print("\n기본 통계:")
print(tips.describe())

print("\n데이터 정보:")
tips.info()
```

**결과 해설**: `seaborn` 라이브러리의 `load_dataset()`으로 실제 공개 데이터셋을 쉽게 불러올 수 있음. Tips 데이터셋은 레스토랑에서 수집된 244건의 식사 기록임.

#### 4-2. 파생 변수 추가

```python
# 팁 비율 계산
tips_analysis = tips.copy()
tips_analysis["tip_rate"] = tips_analysis["tip"] / tips_analysis["total_bill"]
tips_analysis["tip_pct"] = (tips_analysis["tip_rate"] * 100).round(2)

# 1인당 금액 계산
tips_analysis["per_person"] = (tips_analysis["total_bill"] / tips_analysis["size"]).round(2)

# 등급 분류
def classify_tip(rate):
    if rate >= 0.20:
        return "A"
    elif rate >= 0.15:
        return "B"
    else:
        return "C"

tips_analysis["grade"] = tips_analysis["tip_rate"].apply(classify_tip)

print(tips_analysis[["total_bill", "tip", "tip_rate", "tip_pct", "grade"]].head(10))
```

**결과 해설**: `.copy()`로 원본 데이터를 보존하고 작업함. 팁 비율, 1인당 금액 등 파생 변수를 추가하여 더 깊은 분석이 가능함.

#### 4-3. 그룹별 집계

```python
# 요일별 평균
print("요일별 평균 total_bill:")
day_avg = tips.groupby("day")["total_bill"].mean()
print(day_avg)

# 요일별 상세 통계
print("\n요일별 상세 통계:")
day_stats = tips.groupby("day").agg({
    "total_bill": ["count", "mean", "sum"],
    "tip": ["mean", "sum"],
}).round(2)
print(day_stats)

# 성별 집계
print("\n성별 평균:")
sex_stats = tips.groupby("sex")[["total_bill", "tip"]].mean().round(2)
print(sex_stats)

# 최고/최저 팁 요일
day_tip = tips.groupby("day")["tip"].mean()
best_day = day_tip.idxmax()
worst_day = day_tip.idxmin()
print(f"\n최고 팁 요일: {best_day} (평균 팁: ${day_tip[best_day]:.2f})")
print(f"최저 팁 요일: {worst_day} (평균 팁: ${day_tip[worst_day]:.2f})")
```

**결과 해설**: `groupby()`와 `agg()`를 조합하면 그룹별로 다양한 통계를 한 번에 계산할 수 있음. 제조 데이터에서는 라인별, 시간대별, 제품별 분석에 활용됨.

#### 4-4. 결측치 처리

```python
# 결측치가 있는 샘플 데이터
df_missing = pd.DataFrame({
    "온도": [85, 88, np.nan, 92, 87],
    "습도": [45, np.nan, 52, np.nan, 48],
    "압력": [101, 102, 100, 103, 101]
})
print("결측치 있는 데이터:")
print(df_missing)

# 결측치 확인
print("\n결측치 개수:")
print(df_missing.isnull().sum())

# 결측치 있는 행
print("\n결측치 있는 행:")
print(df_missing[df_missing.isnull().any(axis=1)])

# 결측치 제거
print("\ndropna() - 결측치 있는 행 제거:")
print(df_missing.dropna())

# 평균으로 채우기
print("\n열 평균으로 채우기:")
df_filled = df_missing.fillna(df_missing.mean())
print(df_filled)
```

**결과 해설**: `isnull().sum()`으로 열별 결측치 개수를 확인함. `dropna()`는 결측치가 있는 행을 삭제하고, `fillna()`는 지정한 값으로 결측치를 채움. 평균이나 중앙값으로 채우는 것이 일반적임.

#### 4-5. 정렬

```python
# 단일 열 오름차순
tips_sorted = tips.sort_values("total_bill")
print("total_bill 오름차순:")
print(tips_sorted[["total_bill", "tip"]].head())

# 단일 열 내림차순
tips_sorted = tips.sort_values("total_bill", ascending=False)
print("\ntotal_bill 내림차순:")
print(tips_sorted[["total_bill", "tip"]].head())

# 다중 열 정렬
tips_sorted = tips.sort_values(["day", "total_bill"], ascending=[True, False])
print("\nday 오름차순 -> total_bill 내림차순:")
print(tips_sorted[["day", "total_bill", "tip"]].head(10))

# 상위/하위 N개
print("\ntotal_bill 상위 5개:")
print(tips.nlargest(5, "total_bill")[["total_bill", "tip"]])
```

**결과 해설**: `sort_values()`로 데이터를 정렬함. `ascending=False`로 내림차순 정렬, 다중 열 정렬도 가능함. `nlargest()`, `nsmallest()`로 상위/하위 N개를 빠르게 추출할 수 있음.

---

## Part 5: 종합 분석 보고서

### 실습 코드

```python
# 종합 보고서 출력
print("=" * 60)
print("             레스토랑 팁 분석 보고서")
print("=" * 60)
print(f"분석 건수: {len(tips_analysis)}건")
print("-" * 60)

print("\n[전체 현황]")
print(f"  총 매출액: ${tips_analysis['total_bill'].sum():,.2f}")
print(f"  총 팁 수입: ${tips_analysis['tip'].sum():,.2f}")
print(f"  평균 팁 비율: {tips_analysis['tip_rate'].mean()*100:.2f}%")
print(f"  평균 테이블 인원: {tips_analysis['size'].mean():.1f}명")
print(f"  평균 1인당 금액: ${tips_analysis['per_person'].mean():.2f}")

print("\n[요일별 성과]")
day_summary = tips_analysis.groupby("day").agg({
    "total_bill": "sum",
    "tip_rate": "mean",
    "size": "mean"
}).round(3)
for day in tips_analysis["day"].unique():
    day_data = day_summary.loc[day]
    print(f"  {day}: 매출 ${day_data['total_bill']:,.2f}, "
          f"팁비율 {day_data['tip_rate']*100:.2f}%, "
          f"평균인원 {day_data['size']:.1f}명")

print("\n[등급별 분포]")
grade_counts = tips_analysis["grade"].value_counts().sort_index()
for grade, count in grade_counts.items():
    pct = count / len(tips_analysis) * 100
    print(f"  {grade}등급: {count}건 ({pct:.1f}%)")

print("=" * 60)
```

**결과 해설**: NumPy와 Pandas를 활용하여 데이터를 분석하고 보고서 형태로 출력함. 전체 현황, 그룹별 성과, 등급 분포 등 다양한 관점에서 데이터를 요약함.

---

## 자주 하는 실수

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

## 핵심 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    3차시 핵심 요약                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. NumPy 기본                                               │
│     - np.array(): 배열 생성                                  │
│     - 벡터화 연산: arr + 10, arr * 2                         │
│     - 통계: mean, std, max, min, sum                         │
│     - 조건 필터링: arr[arr > 값]                             │
│     - np.where(조건, 참, 거짓): 조건부 값 변경               │
│                                                              │
│  2. Pandas DataFrame                                         │
│     - pd.DataFrame(dict): 테이블 생성                        │
│     - pd.read_csv(): CSV 파일 읽기                           │
│     - df.to_csv(): CSV 파일 저장                             │
│                                                              │
│  3. 데이터 탐색                                              │
│     - head(), tail(): 앞/뒤 데이터                           │
│     - info(): 데이터 타입, 결측치                            │
│     - describe(): 기술 통계                                  │
│                                                              │
│  4. 열/행 선택                                               │
│     - df['열']: Series 반환                                  │
│     - df[['열1', '열2']]: DataFrame 반환                     │
│     - df.loc[]: 라벨 기반 (끝 포함)                          │
│     - df.iloc[]: 정수 기반 (끝 미포함)                       │
│                                                              │
│  5. 조건 필터링                                              │
│     - df[조건]: 조건에 맞는 행                               │
│     - (조건1) & (조건2): AND 조건                            │
│     - (조건1) | (조건2): OR 조건                             │
│                                                              │
│  6. 그룹 집계                                                │
│     - df.groupby('열'): 그룹 생성                            │
│     - .agg({'열': ['함수1', '함수2']}): 다중 집계            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 다음 차시 예고

### 4차시: 공개 데이터셋 확보 및 데이터 생태계 이해

- 공공데이터포털, AI 허브, Kaggle, UCI
- 데이터 플랫폼별 특성과 활용법
- 데이터셋 구조와 용어 정리
- 실제 데이터 다운로드 및 탐색

> 오늘 배운 Pandas로 실제 공개 데이터를 분석해봄
