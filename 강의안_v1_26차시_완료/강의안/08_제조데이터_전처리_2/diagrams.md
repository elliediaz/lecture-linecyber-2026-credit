# [8차시] 제조 데이터 전처리 (2) - 다이어그램

## 1. 전처리 워크플로우 (Part 2)

```mermaid
flowchart LR
    A[결측치/이상치 처리] --> B[스케일링]
    B --> C[인코딩]
    C --> D[모델 입력 준비]

    style A fill:#e0e0e0
    style B fill:#bbdefb
    style C fill:#bbdefb
    style D fill:#c8e6c9
```

## 2. 스케일링이 필요한 이유

```mermaid
flowchart TD
    subgraph 문제["스케일 차이 문제"]
        A["온도: 80~100<br>(범위 20)"]
        B["생산량: 1000~1500<br>(범위 500)"]
        C["습도: 40~80<br>(범위 40)"]
    end

    D["범위 차이<br>25배!"]

    A --> D
    B --> D
    C --> D

    D --> E["큰 변수가<br>모델 지배"]
    D --> F["거리 계산<br>왜곡"]
    D --> G["학습 속도<br>저하"]
```

## 3. 표준화 (StandardScaler)

```mermaid
flowchart LR
    subgraph 입력["원본 데이터"]
        A["온도<br>평균: 85<br>표준편차: 5"]
    end

    subgraph 공식["Z-score 변환"]
        B["Z = (값 - 평균) / 표준편차"]
    end

    subgraph 출력["변환 결과"]
        C["온도<br>평균: 0<br>표준편차: 1"]
    end

    A --> B --> C
```

## 4. 정규화 (MinMaxScaler)

```mermaid
flowchart LR
    subgraph 입력["원본 데이터"]
        A["온도<br>최소: 75<br>최대: 95"]
    end

    subgraph 공식["Min-Max 변환"]
        B["X' = (X - min) / (max - min)"]
    end

    subgraph 출력["변환 결과"]
        C["온도<br>최소: 0<br>최대: 1"]
    end

    A --> B --> C
```

## 5. 스케일링 방법 비교

```mermaid
flowchart TD
    A[스케일링 선택] --> B{데이터 특성?}

    B --> |"정규분포<br>이상치 적음"| C["StandardScaler<br>(표준화)"]
    B --> |"신경망 모델<br>이미지 데이터"| D["MinMaxScaler<br>(정규화)"]
    B --> |"이상치 많음"| E["RobustScaler<br>(중앙값 기준)"]

    C --> F["평균 0<br>표준편차 1"]
    D --> G["범위 [0, 1]"]
    E --> H["IQR 기준"]
```

## 6. 범주형 데이터의 문제

```mermaid
flowchart LR
    subgraph 범주형["범주형 데이터"]
        A["라인: A, B, C"]
        B["등급: 상, 중, 하"]
        C["불량유형: 스크래치, 찍힘"]
    end

    D[["ML 모델"]]

    E["❌ 숫자만<br>입력 가능"]

    A --> D
    B --> D
    C --> D
    D --> E

    F["인코딩<br>필요!"]
    E --> F
```

## 7. 레이블 인코딩

```mermaid
flowchart LR
    subgraph 입력["원본"]
        A["상"]
        B["중"]
        C["하"]
    end

    D[["LabelEncoder"]]

    subgraph 출력["변환"]
        E["0"]
        F["1"]
        G["2"]
    end

    A --> D --> E
    B --> D --> F
    C --> D --> G

    H["⚠️ 크기 관계 발생<br>0 < 1 < 2"]
```

## 8. 원-핫 인코딩

```mermaid
flowchart LR
    subgraph 입력["원본"]
        A["라인 A"]
        B["라인 B"]
        C["라인 C"]
    end

    D[["OneHotEncoder"]]

    subgraph 출력["변환"]
        E["[1, 0, 0]"]
        F["[0, 1, 0]"]
        G["[0, 0, 1]"]
    end

    A --> D --> E
    B --> D --> F
    C --> D --> G
```

## 9. 인코딩 선택 가이드

```mermaid
flowchart TD
    A[범주형 데이터] --> B{순서가 있는가?}

    B --> |예| C["등급, 학년, 크기<br>(상/중/하, S/M/L)"]
    B --> |아니오| D["라인, 색상, 지역<br>(A/B/C, 빨강/파랑)"]

    C --> E["레이블 인코딩<br>LabelEncoder"]
    D --> F{고유값 개수?}

    F --> |"적음 (<10)"| G["원-핫 인코딩<br>OneHotEncoder"]
    F --> |"많음 (>10)"| H["타겟/빈도 인코딩"]

    E --> I["0, 1, 2..."]
    G --> J["[1,0,0], [0,1,0]..."]
```

## 10. fit과 transform의 차이

```mermaid
flowchart TD
    subgraph 학습["학습 데이터"]
        A["fit()<br>통계량 학습<br>(평균, 표준편차 등)"]
        B["transform()<br>변환 적용"]
        A --> B
        C["fit_transform()<br>= fit + transform"]
    end

    subgraph 테스트["테스트 데이터"]
        D["transform()만!<br>학습된 통계량으로 변환"]
    end

    A -.-> |"학습된 통계량 전달"| D

    E["⚠️ 테스트에 fit 적용 시<br>데이터 누수 발생!"]
```

## 11. 데이터 누수 방지

```mermaid
flowchart LR
    subgraph 올바른["✅ 올바른 방법"]
        A1["학습 데이터"] --> B1["fit_transform()"]
        C1["테스트 데이터"] --> D1["transform()"]
    end

    subgraph 잘못된["❌ 잘못된 방법"]
        A2["전체 데이터"] --> B2["fit_transform()"]
        B2 --> C2["학습/테스트 분리"]
        C2 --> D2["데이터 누수!"]
    end
```

## 12. 종합 전처리 파이프라인

```mermaid
flowchart TD
    A[원본 데이터] --> B[결측치 처리]
    B --> C[이상치 처리]

    C --> D{데이터 유형?}

    D --> |수치형| E[스케일링]
    D --> |범주형-순서O| F[레이블 인코딩]
    D --> |범주형-순서X| G[원-핫 인코딩]

    E --> H[전처리 완료]
    F --> H
    G --> H

    H --> I[모델 학습 준비 완료]
```

## 13. 강의 구조

```mermaid
gantt
    title 8차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)        :a1, 00:00, 2m
    스케일링 필요성         :a2, after a1, 1.5m
    표준화 (StandardScaler) :a3, after a2, 1.5m
    정규화 (MinMaxScaler)   :a4, after a3, 1m
    범주형 데이터           :a5, after a4, 1m
    레이블 인코딩           :a6, after a5, 1m
    원-핫 인코딩           :a7, after a6, 1.5m

    section 실습편
    실습 소개             :b1, after a7, 2m
    데이터 준비            :b2, after b1, 2m
    스케일 차이 확인        :b3, after b2, 2m
    표준화 적용            :b4, after b3, 2m
    정규화 적용            :b5, after b4, 2m
    레이블 인코딩          :b6, after b5, 2m
    원-핫 인코딩           :b7, after b6, 2m
    역변환               :b8, after b7, 2m

    section 정리
    핵심 요약             :c1, after b8, 1.5m
    주의사항/예고          :c2, after c1, 1.5m
```
