# [18차시] 딥러닝 입문: 신경망 기초 - 다이어그램

## 1. 머신러닝과 딥러닝

```mermaid
flowchart TD
    A["머신러닝"]

    A --> B["선형회귀"]
    A --> C["의사결정트리"]
    A --> D["랜덤포레스트"]
    A --> E["딥러닝"]

    E --> E1["MLP"]
    E --> E2["CNN"]
    E --> E3["RNN"]

    style E fill:#c8e6c9
```

## 2. 생물학적 뉴런

```mermaid
flowchart LR
    subgraph 입력["수상돌기"]
        A1["신호1"]
        A2["신호2"]
        A3["신호3"]
    end

    B["세포체<br>(처리/판단)"]
    C["축삭돌기<br>(출력)"]

    입력 --> B --> C
```

## 3. 인공 뉴런

```mermaid
flowchart LR
    subgraph 입력["입력"]
        X1["x₁"]
        X2["x₂"]
        X3["x₃"]
    end

    subgraph 가중치["× 가중치"]
        W1["×w₁"]
        W2["×w₂"]
        W3["×w₃"]
    end

    S["Σ + b<br>합산"]
    F["f(x)<br>활성화"]
    Y["y<br>출력"]

    X1 --> W1 --> S
    X2 --> W2 --> S
    X3 --> W3 --> S
    S --> F --> Y
```

## 4. 가중치와 편향

```mermaid
flowchart TD
    subgraph 입력["입력"]
        A1["온도: 85"]
        A2["습도: 50"]
        A3["속도: 100"]
    end

    subgraph 가중치["가중치 (중요도)"]
        B1["0.3"]
        B2["0.2"]
        B3["0.5"]
    end

    C["합산 + 편향"]
    D["25.5 + 10 + 50 + 0.1 = 85.6"]

    입력 --> 가중치 --> C --> D
```

## 5. 활성화 함수 종류

```mermaid
flowchart LR
    subgraph ReLU["ReLU"]
        A1["음수 → 0"]
        A2["양수 → 그대로"]
    end

    subgraph Sigmoid["Sigmoid"]
        B1["0 ~ 1 범위"]
        B2["확률 출력"]
    end

    subgraph Tanh["Tanh"]
        C1["-1 ~ 1 범위"]
        C2["중심이 0"]
    end
```

## 6. ReLU 함수

```mermaid
flowchart TD
    A["ReLU(x) = max(0, x)"]

    A --> B["x < 0 → 0"]
    A --> C["x ≥ 0 → x"]

    D["단순하지만 효과적!"]

    B --> D
    C --> D
```

## 7. 신경망 층 구조

```mermaid
flowchart LR
    subgraph Input["입력층"]
        I1["○"]
        I2["○"]
        I3["○"]
    end

    subgraph Hidden["은닉층"]
        H1["○"]
        H2["○"]
        H3["○"]
        H4["○"]
    end

    subgraph Output["출력층"]
        O1["○"]
    end

    Input --> Hidden --> Output
```

## 8. 심층 신경망 (DNN)

```mermaid
flowchart LR
    subgraph L0["입력층"]
        A1["○"]
        A2["○"]
        A3["○"]
    end

    subgraph L1["은닉층1"]
        B1["○"]
        B2["○"]
        B3["○"]
    end

    subgraph L2["은닉층2"]
        C1["○"]
        C2["○"]
        C3["○"]
    end

    subgraph L3["출력층"]
        D1["○"]
    end

    L0 --> L1 --> L2 --> L3

    E["Deep = 층이 깊다"]
```

## 9. 순전파

```mermaid
flowchart LR
    A["입력<br>x"] --> B["층1"] --> C["층2"] --> D["예측<br>ŷ"]
    D --> E["손실 계산<br>L = (y - ŷ)²"]
```

## 10. 역전파

```mermaid
flowchart RL
    A["손실<br>L"]
    B["기울기<br>∂L/∂w"]
    C["가중치<br>업데이트"]

    A --> B --> C
    C --> D["w = w - η·∂L/∂w"]
```

## 11. 학습 과정

```mermaid
flowchart TD
    A["1. 입력 데이터"]
    B["2. 순전파<br>(예측)"]
    C["3. 손실 계산"]
    D["4. 역전파<br>(기울기)"]
    E["5. 가중치 업데이트"]
    F["반복"]

    A --> B --> C --> D --> E --> F
    F --> A
```

## 12. 경사하강법

```mermaid
flowchart TD
    A["시작점<br>(랜덤 가중치)"]
    B["경사 계산"]
    C["경사 방향으로 이동"]
    D["최소점 도달?"]
    E["완료"]

    A --> B --> C --> D
    D -->|"아니오"| B
    D -->|"예"| E
```

## 13. 손실 함수

```mermaid
flowchart LR
    subgraph 회귀["회귀 문제"]
        A1["MSE"]
        A2["(y - ŷ)²"]
    end

    subgraph 분류["분류 문제"]
        B1["Cross-Entropy"]
        B2["-y·log(ŷ)"]
    end
```

## 14. 딥러닝 vs 머신러닝

```mermaid
flowchart TD
    subgraph ML["머신러닝"]
        A1["특성 엔지니어링<br>사람이 설계"]
        A2["적은 데이터 가능"]
        A3["CPU 가능"]
    end

    subgraph DL["딥러닝"]
        B1["특성 자동 학습"]
        B2["많은 데이터 필요"]
        B3["GPU 권장"]
    end
```

## 15. 제조 현장 적용

```mermaid
flowchart TD
    A["제조 AI 문제"]

    A --> B["테이블 데이터"]
    A --> C["이미지 데이터"]

    B --> B1["센서 값, 생산량"]
    B --> B2["ML 권장<br>(RandomForest)"]

    C --> C1["외관 검사, 결함"]
    C --> C2["딥러닝 권장<br>(CNN)"]
```

## 16. 뉴런 계산 예시

```mermaid
flowchart LR
    subgraph 입력["입력"]
        X["[85, 50, 100]"]
    end

    subgraph 계산["계산"]
        W["×[0.3, 0.2, 0.5]"]
        B["+0.1"]
        R["ReLU"]
    end

    subgraph 출력["출력"]
        Y["85.6"]
    end

    입력 --> W --> B --> R --> 출력
```

## 17. 층 연결

```mermaid
flowchart LR
    A["입력<br>3개"]
    B["층1<br>4뉴런"]
    C["층2<br>1뉴런"]
    D["출력"]

    A -->|"3×4 가중치"| B
    B -->|"4×1 가중치"| C
    C --> D
```

## 18. 강의 구조

```mermaid
gantt
    title 18차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    딥러닝이란?              :a2, after a1, 1.5m
    인공 뉴런               :a3, after a2, 2m
    활성화 함수              :a4, after a3, 1.5m
    신경망 구조              :a5, after a4, 1.5m
    학습 과정               :a6, after a5, 1.5m

    section 실습편
    실습 소개               :b1, after a6, 2m
    뉴런 계산               :b2, after b1, 2m
    활성화 함수              :b3, after b2, 4m
    뉴런/층 클래스           :b4, after b3, 6m
    2층 신경망              :b5, after b4, 3m
    손실 계산               :b6, after b5, 2m

    section 정리
    핵심 요약               :c1, after b6, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((딥러닝<br>기초))
    뉴런
      입력 × 가중치
      + 편향
      활성화 함수
    층
      입력층
      은닉층
      출력층
    활성화
      ReLU
      Sigmoid
      비선형성
    학습
      순전파
      역전파
      경사하강법
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>NumPy 구현"]
    B["다음<br>Keras 사용"]
    C["실습<br>품질 예측"]

    A --> B --> C
```
