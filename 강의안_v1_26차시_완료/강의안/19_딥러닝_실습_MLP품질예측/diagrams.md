# [19차시] 딥러닝 실습: MLP로 품질 예측 - 다이어그램

## 1. Keras 위치

```mermaid
flowchart TD
    A["TensorFlow"]
    B["Keras<br>(고수준 API)"]
    C["저수준 연산"]

    A --> B
    A --> C

    style B fill:#c8e6c9
```

## 2. MLP 구조

```mermaid
flowchart LR
    subgraph Input["입력층 (3)"]
        I1["온도"]
        I2["습도"]
        I3["속도"]
    end

    subgraph H1["은닉층1 (16)"]
        H1a["○ ○ ○ ○"]
        H1b["○ ○ ○ ○"]
        H1c["○ ○ ○ ○"]
        H1d["○ ○ ○ ○"]
    end

    subgraph H2["은닉층2 (8)"]
        H2a["○ ○ ○ ○"]
        H2b["○ ○ ○ ○"]
    end

    subgraph Output["출력층 (1)"]
        O1["불량확률"]
    end

    Input --> H1 --> H2 --> Output
```

## 3. Sequential 모델 코드

```mermaid
flowchart TD
    A["Sequential([...])"]

    A --> B["Dense(16, relu)"]
    B --> C["Dense(8, relu)"]
    C --> D["Dense(1, sigmoid)"]
```

## 4. Dense 층

```mermaid
flowchart LR
    subgraph 입력["입력 (3개)"]
        X1["x₁"]
        X2["x₂"]
        X3["x₃"]
    end

    subgraph Dense["Dense(8, relu)"]
        N1["○"]
        N2["○"]
        N3["○"]
        N4["○ ..."]
    end

    subgraph 출력["출력 (8개)"]
        Y["8개 값"]
    end

    입력 --> Dense --> 출력
```

## 5. 활성화 함수 선택

```mermaid
flowchart TD
    A["활성화 함수 선택"]

    A --> B["은닉층"]
    A --> C["출력층"]

    B --> B1["relu (대부분)"]

    C --> C1["sigmoid<br>(이진 분류)"]
    C --> C2["softmax<br>(다중 분류)"]
    C --> C3["linear<br>(회귀)"]
```

## 6. 모델 구축 흐름

```mermaid
flowchart TD
    A["1. Sequential 생성"]
    B["2. Dense 층 추가"]
    C["3. compile"]
    D["4. fit"]
    E["5. predict"]

    A --> B --> C --> D --> E
```

## 7. compile 구성

```mermaid
flowchart LR
    subgraph compile["model.compile()"]
        A["optimizer<br>'adam'"]
        B["loss<br>'binary_crossentropy'"]
        C["metrics<br>['accuracy']"]
    end
```

## 8. 손실 함수 선택

```mermaid
flowchart TD
    A["문제 유형"]

    A --> B["이진 분류"]
    A --> C["다중 분류"]
    A --> D["회귀"]

    B --> B1["binary_crossentropy"]
    C --> C1["categorical_crossentropy"]
    D --> D1["mse"]
```

## 9. fit 구성

```mermaid
flowchart LR
    subgraph fit["model.fit()"]
        A["X_train, y_train"]
        B["epochs=50"]
        C["batch_size=32"]
        D["validation_split=0.2"]
    end
```

## 10. Epoch와 Batch

```mermaid
flowchart TD
    A["데이터 1000개"]
    B["batch_size=100"]
    C["1 Epoch = 10번 업데이트"]
    D["epochs=10"]
    E["총 100번 업데이트"]

    A --> B --> C --> D --> E
```

## 11. 학습 과정

```mermaid
flowchart LR
    subgraph Epoch1["Epoch 1"]
        A1["Batch 1"] --> A2["Batch 2"] --> A3["..."] --> A4["Batch N"]
    end

    subgraph Epoch2["Epoch 2"]
        B1["Batch 1"] --> B2["..."]
    end

    Epoch1 --> Epoch2
```

## 12. 학습 곡선

```mermaid
flowchart LR
    subgraph Loss["손실 곡선"]
        L["Loss ↓"]
    end

    subgraph Acc["정확도 곡선"]
        A["Accuracy ↑"]
    end
```

## 13. 과대적합 감지

```mermaid
flowchart TD
    subgraph 정상["정상 학습"]
        A1["Train ↓"]
        A2["Val ↓"]
    end

    subgraph 과대적합["과대적합"]
        B1["Train ↓"]
        B2["Val ↑"]
    end

    style 과대적합 fill:#ffcdd2
```

## 14. 데이터 전처리

```mermaid
flowchart TD
    A["원본 데이터"]
    B["StandardScaler"]
    C["정규화된 데이터<br>(평균 0, 분산 1)"]
    D["딥러닝 모델"]

    A --> B --> C --> D
```

## 15. 예측 과정

```mermaid
flowchart LR
    A["X_test"]
    B["model.predict()"]
    C["확률<br>(0~1)"]
    D["> 0.5"]
    E["0 또는 1"]

    A --> B --> C --> D --> E
```

## 16. 평가 흐름

```mermaid
flowchart TD
    A["예측값 y_pred"]
    B["실제값 y_test"]
    C["classification_report"]
    D["정밀도, 재현율, F1"]

    A --> C
    B --> C
    C --> D
```

## 17. ML vs DL 비교

```mermaid
flowchart TD
    subgraph ML["머신러닝 (RF)"]
        A1["특성 엔지니어링 필요"]
        A2["작은 데이터 OK"]
        A3["해석 가능"]
    end

    subgraph DL["딥러닝 (MLP)"]
        B1["자동 특성 학습"]
        B2["많은 데이터 필요"]
        B3["블랙박스"]
    end
```

## 18. 강의 구조

```mermaid
gantt
    title 19차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    Keras 소개              :a2, after a1, 1.5m
    Sequential 모델          :a3, after a2, 2m
    컴파일/학습              :a4, after a3, 2m
    학습 곡선               :a5, after a4, 1.5m
    손실 함수               :a6, after a5, 1m

    section 실습편
    실습 소개               :b1, after a6, 2m
    데이터 생성/전처리        :b2, after b1, 4m
    모델 구축               :b3, after b2, 4m
    학습/시각화             :b4, after b3, 5m
    평가/비교               :b5, after b4, 4m

    section 정리
    핵심 요약               :c1, after b5, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((Keras<br>MLP))
    모델 구축
      Sequential
      Dense
      activation
    컴파일
      optimizer adam
      loss
      metrics
    학습
      fit
      epochs
      batch_size
    평가
      predict
      학습 곡선
      과대적합 감지
```

## 20. 제조 품질 예측 적용

```mermaid
flowchart TD
    A["입력: 온도, 습도, 속도"]
    B["MLP 모델"]
    C["출력: 불량 확률"]
    D{"> 0.5?"}
    E["정상"]
    F["불량 예측"]

    A --> B --> C --> D
    D -->|"아니오"| E
    D -->|"예"| F
```
