# [10차시] 머신러닝 소개와 문제 유형 - 다이어그램

## 1. 전통적 프로그래밍 vs 머신러닝

```mermaid
flowchart LR
    subgraph 전통["전통적 프로그래밍"]
        A1[데이터] --> B1[규칙]
        B1 --> C1[출력]
        D1["개발자가<br>규칙 작성"]
    end

    subgraph ML["머신러닝"]
        A2[데이터] --> B2[학습]
        E2[정답] --> B2
        B2 --> C2[규칙 발견]
        C2 --> F2[예측]
    end
```

## 2. 머신러닝 핵심 개념

```mermaid
flowchart LR
    A["특성(X)<br>온도, 습도, 속도"] --> B["모델<br>(학습된 패턴)"]
    B --> C["타겟(y)<br>불량 여부"]

    D["학습 데이터"] --> E["fit()"]
    E --> B
```

## 3. 머신러닝의 3가지 유형

```mermaid
flowchart TD
    A[머신러닝] --> B[지도학습]
    A --> C[비지도학습]
    A --> D[강화학습]

    B --> B1["정답 있음<br>분류, 회귀"]
    C --> C1["정답 없음<br>군집화, 이상탐지"]
    D --> D1["보상 기반<br>게임, 로봇"]

    style B fill:#bbdefb
```

## 4. 지도학습: 분류 vs 회귀

```mermaid
flowchart TD
    A[지도학습] --> B[분류]
    A --> C[회귀]

    B --> B1["범주 예측"]
    B --> B2["예: 불량/정상, A/B/C등급"]
    B --> B3["~인가요?"]

    C --> C1["숫자 예측"]
    C --> C2["예: 생산량, 불량률"]
    C --> C3["얼마나?"]
```

## 5. 분류 문제 예시

```mermaid
flowchart LR
    subgraph 입력["입력 (특성)"]
        A["온도: 85°C"]
        B["습도: 50%"]
        C["속도: 100"]
    end

    D[["분류 모델"]]

    subgraph 출력["출력 (범주)"]
        E["정상"]
        F["불량"]
    end

    입력 --> D
    D --> E
    D -.-> F

    style E fill:#c8e6c9
```

## 6. 회귀 문제 예시

```mermaid
flowchart LR
    subgraph 입력["입력 (특성)"]
        A["온도: 85°C"]
        B["습도: 50%"]
        C["속도: 100"]
    end

    D[["회귀 모델"]]

    subgraph 출력["출력 (숫자)"]
        E["1,247개"]
    end

    입력 --> D
    D --> E

    style E fill:#bbdefb
```

## 7. 분류 vs 회귀 구분법

```mermaid
flowchart TD
    A[문제] --> B{"출력이 뭔가요?"}

    B --> |"범주<br>(예/아니오, A/B/C)"| C[분류]
    B --> |"숫자<br>(1,247개, 3.5%)"| D[회귀]

    E["~인가요?"] --> C
    F["얼마나?"] --> D
```

## 8. sklearn 기본 패턴

```mermaid
flowchart TD
    A["1. 모델 생성<br>model = ModelName()"] --> B["2. 학습<br>model.fit(X_train, y_train)"]
    B --> C["3. 예측<br>y_pred = model.predict(X_test)"]
    C --> D["4. 평가<br>score = model.score(X_test, y_test)"]

    E["모든 sklearn 모델이<br>이 패턴을 따름!"]
```

## 9. 학습/테스트 분리

```mermaid
flowchart TD
    A[전체 데이터] --> B["train_test_split()"]

    B --> C["학습 데이터<br>(80%)"]
    B --> D["테스트 데이터<br>(20%)"]

    C --> E["모델 학습<br>fit()"]
    D --> F["성능 평가<br>score()"]

    G["처음 보는 데이터로<br>평가해야 진짜 실력!"]
```

## 10. 데이터 분리 중요성

```mermaid
flowchart LR
    subgraph 잘못된["❌ 잘못된 방법"]
        A1["전체 데이터로<br>학습 + 평가"]
        B1["외웠는지<br>이해했는지 모름"]
    end

    subgraph 올바른["✅ 올바른 방법"]
        A2["학습 데이터로<br>학습"]
        B2["테스트 데이터로<br>평가"]
        C2["일반화 성능 확인"]
    end
```

## 11. 머신러닝 전체 워크플로우

```mermaid
flowchart TD
    A[1. 문제 정의] --> B["분류? 회귀?"]
    B --> C[2. 데이터 준비]
    C --> D[3. 전처리]
    D --> E[4. 학습/테스트 분리]
    E --> F[5. 모델 학습]
    F --> G[6. 예측]
    G --> H[7. 평가]
    H --> I{성능 만족?}
    I --> |아니오| J[모델/데이터 개선]
    J --> F
    I --> |예| K[배포]
```

## 12. 제조 현장 문제 유형

```mermaid
mindmap
  root((제조 AI))
    분류 문제
      불량/정상 판정
      품질 등급 분류
      설비 상태 진단
      고장 유형 분류
    회귀 문제
      생산량 예측
      불량률 예측
      설비 수명 예측
      에너지 소비 예측
```

## 13. 강의 구조

```mermaid
gantt
    title 10차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (Part III 시작)     :a1, 00:00, 2m
    머신러닝이란?           :a2, after a1, 1.5m
    핵심 용어              :a3, after a2, 1m
    ML의 종류             :a4, after a3, 1m
    분류 vs 회귀           :a5, after a4, 2m
    sklearn 소개          :a6, after a5, 1.5m
    학습/테스트 분리        :a7, after a6, 1m

    section 실습편
    실습 소개             :b1, after a7, 2m
    데이터 생성            :b2, after b1, 2m
    특성/타겟 분리         :b3, after b2, 2m
    학습/테스트 분리        :b4, after b3, 2m
    분류 모델             :b5, after b4, 3m
    회귀 모델             :b6, after b5, 3m
    문제 유형 구분         :b7, after b6, 2m

    section 정리
    핵심 요약             :c1, after b7, 1.5m
    다음 차시 예고         :c2, after c1, 1.5m
```

## 14. Part II → Part III 연결

```mermaid
flowchart LR
    subgraph PartII["Part II에서 배운 것"]
        A[데이터 분석]
        B[시각화]
        C[전처리]
        D[EDA]
    end

    subgraph PartIII["Part III에서 활용"]
        E[특성 선택]
        F[모델 평가 이해]
        G[데이터 준비]
        H[결과 해석]
    end

    A --> E
    B --> F
    C --> G
    D --> H

    I["Part II가 있어야<br>Part III가 가능!"]
```
