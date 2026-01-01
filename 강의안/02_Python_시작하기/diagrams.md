# [2차시] Python 시작하기 - 다이어그램

## 1. Python 개발 환경 구조

```mermaid
flowchart TD
    subgraph 개발환경["Python 개발 환경"]
        A[Anaconda] --> B[Python]
        A --> C[패키지 관리자<br>conda/pip]
        A --> D[가상환경]
        A --> E[Jupyter Notebook]
    end

    B --> F[NumPy]
    B --> G[Pandas]
    B --> H[Matplotlib]
    B --> I[Scikit-learn]

    E --> J[웹 브라우저에서<br>코드 작성/실행]
```

## 2. Python 5대 자료형

```mermaid
mindmap
  root((Python<br>자료형))
    정수 int
      생산량 1000
      라인 번호 3
      개수, 횟수
    실수 float
      온도 85.7
      불량률 0.025
      측정값
    문자열 str
      제품명
      상태 정상
      텍스트
    리스트 list
      순서 있음
      대괄호 사용
      여러 데이터 모음
    딕셔너리 dict
      키-값 쌍
      중괄호 사용
      이름표 붙은 데이터
```

## 3. 조건문 흐름

```mermaid
flowchart TD
    A[불량률 입력] --> B{불량률 <= 1%?}
    B -->|Yes| C[등급 A<br>우수]
    B -->|No| D{불량률 <= 3%?}
    D -->|Yes| E[등급 B<br>양호]
    D -->|No| F{불량률 <= 5%?}
    F -->|Yes| G[등급 C<br>주의]
    F -->|No| H[등급 D<br>개선 필요]

    C --> I[결과 출력]
    E --> I
    G --> I
    H --> I
```

## 4. 반복문 동작 원리

```mermaid
flowchart LR
    subgraph 리스트["temperatures = [82, 85, 88, 95, 84]"]
        A1[82] --> A2[85] --> A3[88] --> A4[95] --> A5[84]
    end

    subgraph 반복["for temp in temperatures:"]
        B1["1회차<br>temp=82"] --> B2["2회차<br>temp=85"]
        B2 --> B3["3회차<br>temp=88"]
        B3 --> B4["4회차<br>temp=95"]
        B4 --> B5["5회차<br>temp=84"]
    end

    A1 -.-> B1
    A2 -.-> B2
    A3 -.-> B3
    A4 -.-> B4
    A5 -.-> B5
```

## 5. 리스트 인덱싱

```mermaid
flowchart TB
    subgraph 리스트["production = [1200, 1150, 1300, 1180, 1250]"]
        direction LR
        I0["[0]<br>1200"] --- I1["[1]<br>1150"] --- I2["[2]<br>1300"] --- I3["[3]<br>1180"] --- I4["[4]<br>1250"]
    end

    subgraph 음수인덱스["음수 인덱스"]
        direction LR
        N5["[-5]"] --- N4["[-4]"] --- N3["[-3]"] --- N2["[-2]"] --- N1["[-1]"]
    end

    I0 -.-> N5
    I1 -.-> N4
    I2 -.-> N3
    I3 -.-> N2
    I4 -.-> N1
```

## 6. 딕셔너리 구조

```mermaid
flowchart LR
    subgraph 딕셔너리["sensor_data = {...}"]
        K1["온도"] --> V1["85.2"]
        K2["습도"] --> V2["45"]
        K3["압력"] --> V3["1.2"]
        K4["상태"] --> V4["정상"]
    end

    A["sensor_data['온도']"] --> V1
    B["sensor_data['상태']"] --> V4
```

## 7. 프로그램 실행 흐름

```mermaid
sequenceDiagram
    participant U as 사용자
    participant J as Jupyter Notebook
    participant P as Python 인터프리터
    participant O as 출력 결과

    U->>J: 코드 입력
    U->>J: Shift+Enter
    J->>P: 코드 전달
    P->>P: 코드 실행
    P->>O: 결과 생성
    O->>J: 결과 표시
    J->>U: 결과 확인
```

## 8. 강의 구조

```mermaid
gantt
    title 2차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)     :a1, 00:00, 2m
    왜 Python인가       :a2, after a1, 2m
    개발 환경 구축       :a3, after a2, 2m
    자료형 5가지        :a4, after a3, 2m
    조건문/반복문       :a5, after a4, 2m

    section 실습편
    실습 소개           :b1, after a5, 2m
    변수와 계산         :b2, after b1, 3m
    조건문 품질 판정    :b3, after b2, 4m
    반복문 온도 분석    :b4, after b3, 4m
    딕셔너리 설비 관리  :b5, after b4, 3m
    종합 예제           :b6, after b5, 3m

    section 정리
    자주 하는 실수      :c1, after b6, 1m
    요약 및 예고        :c2, after c1, 2m
```

## 9. 자료형 비교

```mermaid
flowchart TB
    subgraph 숫자형["숫자 데이터"]
        INT["정수 int<br>1000, -5, 0"]
        FLOAT["실수 float<br>3.14, 0.025"]
    end

    subgraph 문자형["텍스트 데이터"]
        STR["문자열 str<br>'Hello', '정상'"]
    end

    subgraph 컬렉션["데이터 모음"]
        LIST["리스트 list<br>[1, 2, 3]<br>순서 O, 수정 O"]
        DICT["딕셔너리 dict<br>{'키': '값'}<br>이름표로 접근"]
    end

    INT --> |"연산 가능"| FLOAT
    LIST --> |"인덱스로 접근"| STR
    DICT --> |"키로 접근"| STR
```

## 10. 온도 모니터링 실습 흐름

```mermaid
flowchart TD
    A[온도 리스트 입력] --> B[경고 기준 설정<br>threshold = 90]
    B --> C[첫 번째 온도 확인]
    C --> D{temp > 90?}
    D -->|Yes| E["[경고] 출력"]
    D -->|No| F["[정상] 출력"]
    E --> G{다음 온도 있음?}
    F --> G
    G -->|Yes| C
    G -->|No| H[평균 온도 계산]
    H --> I[결과 출력]
```
