# [20차시] 딥러닝 입문: 신경망 기초 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["18차시<br>시계열 예측"]
    B["20차시<br>신경망 기초"]
    C["20차시<br>MLP 실습"]

    A --> B --> C

    B --> B1["인공 뉴런"]
    B --> B2["신경망 구조"]
    B --> B3["순전파/역전파"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["20차시: 딥러닝 입문"]

    A --> B["대주제 1<br>인공 뉴런"]
    A --> C["대주제 2<br>신경망 구조"]
    A --> D["대주제 3<br>순전파/역전파"]

    B --> B1["가중치, 편향<br>활성화 함수"]
    C --> C1["입력층, 은닉층<br>출력층"]
    D --> D1["경사하강법<br>손실 함수"]

    style A fill:#1e40af,color:#fff
```

## 3. ML vs DL 비교

```mermaid
flowchart TD
    A["예측 모델"]

    A --> B["머신러닝"]
    B --> B1["특성 엔지니어링<br>필요"]
    B --> B2["RandomForest<br>XGBoost"]

    A --> C["딥러닝"]
    C --> C1["특성 자동 학습"]
    C --> C2["신경망<br>CNN, RNN"]

    style A fill:#1e40af,color:#fff
```

## 4. 생물학적 뉴런

```mermaid
flowchart LR
    A["수상돌기<br>(입력)"]
    B["세포체<br>(처리)"]
    C["축삭<br>(전달)"]
    D["시냅스<br>(출력)"]

    A --> B --> C --> D

    style B fill:#1e40af,color:#fff
```

## 5. 인공 뉴런 구조

```mermaid
flowchart LR
    X1["x₁"] --> |"w₁"| S["Σ + b"]
    X2["x₂"] --> |"w₂"| S
    X3["x₃"] --> |"w₃"| S

    S --> F["f(z)"]
    F --> Y["출력"]

    style S fill:#1e40af,color:#fff
    style F fill:#059669,color:#fff
```

## 6. 퍼셉트론 수식

```mermaid
flowchart TD
    A["퍼셉트론"]

    A --> B["선형 조합"]
    B --> B1["z = Σwᵢxᵢ + b"]

    A --> C["활성화 함수"]
    C --> C1["y = f(z)"]

    A --> D["최종 출력"]
    D --> D1["분류 또는 값"]

    style A fill:#1e40af,color:#fff
```

## 7. 가중치와 편향

```mermaid
flowchart TD
    A["파라미터"]

    A --> B["가중치 (W)"]
    B --> B1["입력의 중요도"]
    B --> B2["학습으로 조절"]

    A --> C["편향 (b)"]
    C --> C1["활성화 기준점"]
    C --> C2["절편 역할"]

    style A fill:#1e40af,color:#fff
```

## 8. 활성화 함수 종류

```mermaid
flowchart TD
    A["활성화 함수"]

    A --> B["Sigmoid"]
    B --> B1["0~1 출력"]
    B --> B2["이진 분류"]

    A --> C["ReLU"]
    C --> C1["max(0, z)"]
    C --> C2["은닉층 표준"]

    A --> D["Softmax"]
    D --> D1["합 = 1"]
    D --> D2["다중 분류"]

    style A fill:#1e40af,color:#fff
```

## 9. Sigmoid 함수

```mermaid
flowchart TD
    A["Sigmoid"]

    A --> B["공식"]
    B --> B1["σ(z) = 1/(1+e⁻ᶻ)"]

    A --> C["특징"]
    C --> C1["출력: 0~1"]
    C --> C2["확률 해석 가능"]

    A --> D["단점"]
    D --> D1["기울기 소실"]

    style A fill:#1e40af,color:#fff
```

## 10. ReLU 함수

```mermaid
flowchart TD
    A["ReLU"]

    A --> B["공식"]
    B --> B1["f(z) = max(0, z)"]

    A --> C["장점"]
    C --> C1["계산 빠름"]
    C --> C2["기울기 소실 적음"]

    A --> D["단점"]
    D --> D1["음수 = 0<br>(Dying ReLU)"]

    style A fill:#1e40af,color:#fff
```

## 11. Tanh 함수

```mermaid
flowchart TD
    A["Tanh"]

    A --> B["공식"]
    B --> B1["(eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)"]

    A --> C["특징"]
    C --> C1["출력: -1~1"]
    C --> C2["0 중심"]

    A --> D["용도"]
    D --> D1["RNN 은닉 상태"]

    style A fill:#1e40af,color:#fff
```

## 12. Softmax 함수

```mermaid
flowchart TD
    A["Softmax"]

    A --> B["입력"]
    B --> B1["z₁, z₂, z₃"]

    A --> C["출력"]
    C --> C1["p₁ + p₂ + p₃ = 1"]
    C --> C2["각각 확률"]

    A --> D["용도"]
    D --> D1["다중 클래스 분류"]

    style A fill:#1e40af,color:#fff
```

## 13. 활성화 함수 선택

```mermaid
flowchart TD
    A["어떤 활성화 함수?"]

    A --> B["은닉층"]
    B --> B1["✅ ReLU"]
    B --> B2["LeakyReLU"]

    A --> C["출력층 - 이진분류"]
    C --> C1["✅ Sigmoid"]

    A --> D["출력층 - 다중분류"]
    D --> D1["✅ Softmax"]

    A --> E["출력층 - 회귀"]
    E --> E1["없음 (linear)"]

    style A fill:#1e40af,color:#fff
```

## 14. 신경망 층 구조

```mermaid
flowchart LR
    A["입력층<br>Input"]
    B["은닉층 1<br>Hidden"]
    C["은닉층 2<br>Hidden"]
    D["출력층<br>Output"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#1e40af,color:#fff
    style C fill:#1e40af,color:#fff
    style D fill:#dcfce7
```

## 15. 입력층

```mermaid
flowchart TD
    A["입력층"]

    A --> B["역할"]
    B --> B1["데이터 입력"]

    A --> C["노드 수"]
    C --> C1["= 특성 개수"]

    A --> D["예시"]
    D --> D1["4개 특성<br>→ 4개 노드"]

    style A fill:#1e40af,color:#fff
```

## 16. 은닉층

```mermaid
flowchart TD
    A["은닉층"]

    A --> B["역할"]
    B --> B1["특성 변환"]
    B --> B2["패턴 학습"]

    A --> C["개수"]
    C --> C1["1개 이상"]
    C --> C2["깊을수록 복잡"]

    A --> D["노드 수"]
    D --> D1["하이퍼파라미터"]

    style A fill:#1e40af,color:#fff
```

## 17. 출력층

```mermaid
flowchart TD
    A["출력층"]

    A --> B["회귀"]
    B --> B1["1개 노드"]
    B --> B2["활성화 없음"]

    A --> C["이진 분류"]
    C --> C1["1개 노드"]
    C --> C2["Sigmoid"]

    A --> D["다중 분류"]
    D --> D1["클래스 수 노드"]
    D --> D2["Softmax"]

    style A fill:#1e40af,color:#fff
```

## 18. MLP 구조 예시

```mermaid
flowchart LR
    subgraph 입력층
        I1["x₁"]
        I2["x₂"]
        I3["x₃"]
        I4["x₄"]
    end

    subgraph 은닉층
        H1["h₁"]
        H2["h₂"]
        H3["h₃"]
    end

    subgraph 출력층
        O1["y"]
    end

    I1 & I2 & I3 & I4 --> H1 & H2 & H3
    H1 & H2 & H3 --> O1
```

## 19. 파라미터 수 계산

```mermaid
flowchart TD
    A["파라미터 수"]

    A --> B["공식"]
    B --> B1["(이전 노드 × 현재 노드)<br>+ 편향"]

    A --> C["예시: 4→8→1"]
    C --> C1["층1: 4×8 + 8 = 40"]
    C --> C2["층2: 8×1 + 1 = 9"]
    C --> C3["총: 49개"]

    style A fill:#1e40af,color:#fff
```

## 20. 깊은 신경망

```mermaid
flowchart TD
    A["깊은 신경망"]

    A --> B["장점"]
    B --> B1["복잡한 패턴 학습"]
    B --> B2["계층적 특성 추출"]

    A --> C["단점"]
    C --> C1["학습 어려움"]
    C --> C2["과적합 위험"]
    C --> C3["계산량 증가"]

    style A fill:#1e40af,color:#fff
```

## 21. 순전파 개념

```mermaid
flowchart LR
    A["입력 X"]
    B["층 1"]
    C["층 2"]
    D["출력 ŷ"]

    A --> |"W₁, b₁"| B
    B --> |"활성화"| B
    B --> |"W₂, b₂"| C
    C --> |"활성화"| D

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 22. 순전파 수식

```mermaid
flowchart TD
    A["순전파 계산"]

    A --> B["층 1"]
    B --> B1["z₁ = X·W₁ + b₁"]
    B --> B2["a₁ = ReLU(z₁)"]

    A --> C["층 2"]
    C --> C1["z₂ = a₁·W₂ + b₂"]
    C --> C2["ŷ = Sigmoid(z₂)"]

    style A fill:#1e40af,color:#fff
```

## 23. 손실 함수 종류

```mermaid
flowchart TD
    A["손실 함수"]

    A --> B["회귀"]
    B --> B1["MSE"]
    B --> B2["MAE"]

    A --> C["이진 분류"]
    C --> C1["Binary<br>Cross-Entropy"]

    A --> D["다중 분류"]
    D --> D1["Categorical<br>Cross-Entropy"]

    style A fill:#1e40af,color:#fff
```

## 24. MSE 손실

```mermaid
flowchart TD
    A["MSE"]

    A --> B["공식"]
    B --> B1["L = (1/n)Σ(y-ŷ)²"]

    A --> C["특징"]
    C --> C1["제곱으로 패널티"]
    C --> C2["미분 가능"]

    A --> D["용도"]
    D --> D1["회귀 문제"]

    style A fill:#1e40af,color:#fff
```

## 25. Cross-Entropy 손실

```mermaid
flowchart TD
    A["Cross-Entropy"]

    A --> B["이진"]
    B --> B1["-[y·log(ŷ) +<br>(1-y)·log(1-ŷ)]"]

    A --> C["다중"]
    C --> C1["-Σyᵢ·log(ŷᵢ)"]

    A --> D["특징"]
    D --> D1["확률 분포 비교"]

    style A fill:#1e40af,color:#fff
```

## 26. 역전파 개념

```mermaid
flowchart RL
    A["출력층"]
    B["은닉층"]
    C["입력층"]

    A --> |"∂L/∂W₂"| B
    B --> |"∂L/∂W₁"| C

    A --> A1["손실 계산"]
    B --> B1["기울기 전파"]

    style A fill:#fecaca
```

## 27. 역전파 흐름

```mermaid
flowchart TD
    A["1. 순전파"]
    B["2. 손실 계산"]
    C["3. 기울기 계산<br>(역전파)"]
    D["4. 가중치 업데이트"]
    E["5. 반복"]

    A --> B --> C --> D --> E
    E --> |"에포크 반복"| A

    style C fill:#1e40af,color:#fff
```

## 28. 연쇄 법칙

```mermaid
flowchart TD
    A["연쇄 법칙"]

    A --> B["핵심"]
    B --> B1["∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w"]

    A --> C["역할"]
    C --> C1["깊은 층의 기울기 계산"]

    A --> D["구현"]
    D --> D1["출력→입력 방향"]

    style A fill:#1e40af,color:#fff
```

## 29. 경사하강법

```mermaid
flowchart TD
    A["경사하강법"]

    A --> B["아이디어"]
    B --> B1["기울기 반대 방향으로<br>조금씩 이동"]

    A --> C["수식"]
    C --> C1["W = W - η·∂L/∂W"]

    A --> D["η (학습률)"]
    D --> D1["이동 폭 조절"]

    style A fill:#1e40af,color:#fff
```

## 30. 학습률 영향

```mermaid
flowchart TD
    A["학습률"]

    A --> B["너무 큼"]
    B --> B1["발산"]
    B --> B2["최적점 지나침"]

    A --> C["적절함"]
    C --> C1["안정적 수렴"]

    A --> D["너무 작음"]
    D --> D1["학습 느림"]
    D --> D2["지역 최소점"]

    style C fill:#dcfce7
    style B fill:#fecaca
    style D fill:#fef3c7
```

## 31. 옵티마이저 종류

```mermaid
flowchart TD
    A["옵티마이저"]

    A --> B["SGD"]
    B --> B1["기본 경사하강법"]

    A --> C["Momentum"]
    C --> C1["관성 추가"]

    A --> D["Adam"]
    D --> D1["✅ 가장 많이 사용"]
    D --> D2["적응적 학습률"]

    style A fill:#1e40af,color:#fff
    style D fill:#dcfce7
```

## 32. 에포크와 배치

```mermaid
flowchart TD
    A["학습 단위"]

    A --> B["에포크"]
    B --> B1["전체 데이터 1회 학습"]

    A --> C["배치"]
    C --> C1["한 번에 학습하는<br>데이터 개수"]

    A --> D["반복"]
    D --> D1["배치 수 = 데이터 / 배치 크기"]

    style A fill:#1e40af,color:#fff
```

## 33. 배치 크기 영향

```mermaid
flowchart TD
    A["배치 크기"]

    A --> B["작음 (32)"]
    B --> B1["잦은 업데이트"]
    B --> B2["노이즈 많음"]
    B --> B3["일반화 좋음"]

    A --> C["큼 (256)"]
    C --> C1["안정적 업데이트"]
    C --> C2["메모리 많이 필요"]
    C --> C3["빠른 학습"]

    style A fill:#1e40af,color:#fff
```

## 34. 학습 과정 시각화

```mermaid
flowchart TD
    A["학습 시작"]
    B["높은 손실"]
    C["손실 감소"]
    D["수렴"]
    E["학습 완료"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 35. XOR 문제

```mermaid
flowchart TD
    A["XOR 문제"]

    A --> B["입력"]
    B --> B1["(0,0)→0"]
    B --> B2["(0,1)→1"]
    B --> B3["(1,0)→1"]
    B --> B4["(1,1)→0"]

    A --> C["특징"]
    C --> C1["선형 분리 불가"]
    C --> C2["신경망으로 해결"]

    style A fill:#1e40af,color:#fff
```

## 36. XOR 신경망 구조

```mermaid
flowchart LR
    subgraph 입력
        I1["x₁"]
        I2["x₂"]
    end

    subgraph 은닉
        H1["h₁"]
        H2["h₂"]
    end

    subgraph 출력
        O["y"]
    end

    I1 & I2 --> H1 & H2
    H1 & H2 --> O
```

## 37. 기울기 소실

```mermaid
flowchart TD
    A["기울기 소실"]

    A --> B["문제"]
    B --> B1["깊은 층일수록<br>기울기가 0에 가까워짐"]

    A --> C["원인"]
    C --> C1["Sigmoid/Tanh<br>기울기 최대 0.25"]

    A --> D["해결"]
    D --> D1["ReLU 사용"]
    D --> D2["배치 정규화"]

    style A fill:#fecaca
```

## 38. 과적합 방지

```mermaid
flowchart TD
    A["과적합 방지"]

    A --> B["드롭아웃"]
    B --> B1["무작위 노드 끄기"]

    A --> C["조기 종료"]
    C --> C1["검증 손실<br>증가시 중단"]

    A --> D["정규화"]
    D --> D1["L2 가중치 규제"]

    style A fill:#1e40af,color:#fff
```

## 39. 실습 흐름

```mermaid
flowchart TD
    A["1. 활성화 함수 구현"]
    B["2. 순전파 구현"]
    C["3. 손실 계산"]
    D["4. 역전파 구현"]
    E["5. 학습 루프"]
    F["6. 결과 확인"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 40. 핵심 정리

```mermaid
flowchart TD
    A["20차시 핵심"]

    A --> B["인공 뉴런"]
    B --> B1["가중치×입력 + 편향<br>활성화 함수"]

    A --> C["신경망 구조"]
    C --> C1["입력→은닉→출력<br>파라미터 계산"]

    A --> D["학습"]
    D --> D1["순전파→손실→역전파<br>경사하강법"]

    style A fill:#1e40af,color:#fff
```

## 41. 다음 차시 연결

```mermaid
flowchart LR
    A["20차시<br>신경망 기초"]
    B["20차시<br>MLP 실습"]

    A --> B

    A --> A1["원리 이해"]
    B --> B1["Keras 구현"]
    B --> B2["품질 예측"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
