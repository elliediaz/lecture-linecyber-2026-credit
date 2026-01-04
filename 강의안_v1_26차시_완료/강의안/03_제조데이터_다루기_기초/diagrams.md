# [3차시] 제조 데이터 다루기 기초 - 다이어그램

## 1. NumPy와 Pandas 개요

```mermaid
flowchart TD
    subgraph 데이터분석["Python 데이터 분석 도구"]
        A[NumPy] --> B[수치 계산]
        A --> C[배열 연산]
        A --> D[통계 함수]

        E[Pandas] --> F[표 형태 데이터]
        E --> G[파일 읽기/쓰기]
        E --> H[그룹별 집계]
    end

    I[Python 리스트] --> |느림, 복잡| A
    A --> |기반| E
    E --> J[데이터 분석 결과]
```

## 2. NumPy 배열 구조

```mermaid
flowchart LR
    subgraph 1차원["1차원 배열 (Vector)"]
        A1["[82, 85, 88, 95, 84]"]
    end

    subgraph 2차원["2차원 배열 (Matrix)"]
        B1["[[82, 45, 1.2],"]
        B2[" [85, 48, 1.1],"]
        B3[" [88, 52, 1.3]]"]
    end

    C[np.array] --> 1차원
    C --> 2차원
```

## 3. NumPy 벡터화 연산

```mermaid
flowchart TD
    A["temperatures = [82, 85, 88, 95, 84]"]

    subgraph Python리스트["Python 리스트 방식"]
        B1["for i in range(len(data)):"]
        B2["    result[i] = data[i] + 10"]
        B3["5번 반복 실행"]
    end

    subgraph NumPy방식["NumPy 방식"]
        C1["temps + 10"]
        C2["한 번에 모든 요소 처리"]
        C3["C 언어로 최적화"]
    end

    A --> Python리스트
    A --> NumPy방식

    Python리스트 --> D["느림"]
    NumPy방식 --> E["100배 빠름"]
```

## 4. Pandas DataFrame 구조

```mermaid
flowchart TB
    subgraph DataFrame["DataFrame 구조"]
        direction TB
        H["열 이름 (columns)"]
        H --> R0["인덱스 0 | SM-001 | 1200 | 24"]
        R0 --> R1["인덱스 1 | SM-002 | 1150 | 35"]
        R1 --> R2["인덱스 2 | SM-003 | 1300 | 26"]
    end

    D["딕셔너리"] --> |pd.DataFrame| DataFrame
    F["CSV 파일"] --> |pd.read_csv| DataFrame
```

## 5. DataFrame 열/행 선택

```mermaid
flowchart LR
    subgraph 원본["DataFrame"]
        A["제품코드 | 생산량 | 불량수 | 라인"]
    end

    A --> |"df['생산량']"| B["Series<br>단일 열"]
    A --> |"df[['생산량','불량수']]"| C["DataFrame<br>다중 열"]
    A --> |"df.loc[0]"| D["Series<br>단일 행"]
    A --> |"df.iloc[0:2]"| E["DataFrame<br>여러 행"]
```

## 6. 조건 필터링 흐름

```mermaid
flowchart TD
    A[DataFrame] --> B["조건 생성<br>df['불량률'] > 0.03"]
    B --> C["불리언 마스크<br>[False, True, False, True, False]"]
    C --> D["필터링 적용<br>df[조건]"]
    D --> E["조건 만족 행만 추출"]

    subgraph 복합조건["복합 조건"]
        F["(조건1) & (조건2)"] --> G["AND 연산"]
        H["(조건1) | (조건2)"] --> I["OR 연산"]
    end
```

## 7. groupby 집계 과정

```mermaid
flowchart TD
    A["원본 DataFrame<br>5개 행"] --> B["groupby('라인')"]

    B --> C["라인 1 그룹<br>3개 행"]
    B --> D["라인 2 그룹<br>2개 행"]

    C --> E["집계 함수 적용<br>mean(), sum()"]
    D --> E

    E --> F["라인별 통계 결과"]
```

## 8. 데이터 분석 워크플로우

```mermaid
flowchart LR
    A[CSV 파일] --> B[read_csv]
    B --> C[DataFrame]

    C --> D["탐색<br>head, info, describe"]
    D --> E["전처리<br>결측치, 새 열"]
    E --> F["분석<br>필터링, groupby"]
    F --> G["결과<br>to_csv"]
```

## 9. NumPy 통계 함수

```mermaid
mindmap
  root((NumPy<br>통계함수))
    중심 경향
      np.mean 평균
      np.median 중앙값
    산포도
      np.std 표준편차
      np.var 분산
    범위
      np.max 최대값
      np.min 최소값
    집계
      np.sum 합계
      np.prod 곱
```

## 10. 강의 구조

```mermaid
gantt
    title 3차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)     :a1, 00:00, 2m
    NumPy와 Pandas 필요성 :a2, after a1, 2m
    NumPy 배열과 연산    :a3, after a2, 2m
    NumPy 통계 함수      :a4, after a3, 1m
    Pandas DataFrame    :a5, after a4, 2m
    CSV와 데이터 탐색    :a6, after a5, 1m

    section 실습편
    실습 소개           :b1, after a6, 2m
    NumPy 배열 다루기   :b2, after b1, 3m
    DataFrame 생성     :b3, after b2, 3m
    불량률 계산        :b4, after b3, 3m
    조건 필터링        :b5, after b4, 3m
    그룹별 집계        :b6, after b5, 3m
    종합 예제          :b7, after b6, 2m

    section 정리
    자주 하는 실수      :c1, after b7, 1m
    요약 및 예고        :c2, after c1, 2m
```
