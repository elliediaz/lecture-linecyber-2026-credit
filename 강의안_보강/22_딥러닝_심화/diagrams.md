# [22차시] 딥러닝 심화 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["20차시<br>MLP 실습"]
    B["22차시<br>딥러닝 심화"]
    C["22차시<br>모델 해석"]

    A --> B --> C

    B --> B1["CNN"]
    B --> B2["RNN/LSTM"]
    B --> B3["고급 아키텍처"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["22차시: 딥러닝 심화"]

    A --> B["대주제 1<br>CNN"]
    A --> C["대주제 2<br>RNN"]
    A --> D["대주제 3<br>고급 아키텍처"]

    B --> B1["합성곱, 풀링<br>이미지 처리"]
    C --> C1["LSTM, GRU<br>시계열 처리"]
    D --> D1["ResNet, Transformer<br>최신 기술"]

    style A fill:#1e40af,color:#fff
```

## 3. 딥러닝 아키텍처 분류

```mermaid
flowchart TD
    A["딥러닝 아키텍처"]

    A --> B["MLP"]
    B --> B1["정형 데이터"]

    A --> C["CNN"]
    C --> C1["이미지"]

    A --> D["RNN/LSTM"]
    D --> D1["시계열"]

    A --> E["Transformer"]
    E --> E1["텍스트, 시퀀스"]

    style A fill:#1e40af,color:#fff
```

## 4. CNN 전체 구조

```mermaid
flowchart TD
    A["입력 이미지"]
    B["Conv2D"]
    C["MaxPooling"]
    D["Conv2D"]
    E["MaxPooling"]
    F["Flatten"]
    G["Dense"]
    H["출력"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#dbeafe
    style H fill:#dcfce7
```

## 5. 합성곱 연산

```mermaid
flowchart LR
    A["입력 이미지<br>28×28"]
    B["필터<br>3×3"]
    C["특징 맵<br>26×26"]

    A --> |"합성곱"| C
    B --> |"적용"| C

    style A fill:#dbeafe
    style B fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 6. 필터 종류

```mermaid
flowchart TD
    A["CNN 필터"]

    A --> B["수평 엣지"]
    B --> B1["가로 경계선 탐지"]

    A --> C["수직 엣지"]
    C --> C1["세로 경계선 탐지"]

    A --> D["대각 엣지"]
    D --> D1["대각선 탐지"]

    A --> E["코너"]
    E --> E1["모서리 탐지"]

    style A fill:#1e40af,color:#fff
```

## 7. 풀링 연산

```mermaid
flowchart LR
    A["4×4 특징 맵"]
    B["2×2 Max Pooling"]
    C["2×2 출력"]

    A --> B --> C

    style B fill:#1e40af,color:#fff
```

## 8. MaxPooling 효과

```mermaid
flowchart TD
    A["MaxPooling"]

    A --> B["크기 축소"]
    B --> B1["계산량 감소"]

    A --> C["위치 불변성"]
    C --> C1["약간의 이동에 강건"]

    A --> D["과적합 방지"]
    D --> D1["특징 요약"]

    style A fill:#1e40af,color:#fff
```

## 9. Conv2D 파라미터

```mermaid
flowchart TD
    A["Conv2D"]

    A --> B["filters"]
    B --> B1["필터 개수<br>(출력 채널)"]

    A --> C["kernel_size"]
    C --> C1["필터 크기<br>(3, 3)"]

    A --> D["padding"]
    D --> D1["same: 크기 유지<br>valid: 줄어듦"]

    A --> E["strides"]
    E --> E1["이동 간격"]

    style A fill:#1e40af,color:#fff
```

## 10. CNN 계층적 특징

```mermaid
flowchart TD
    A["얕은 층"]
    B["중간 층"]
    C["깊은 층"]

    A --> |"엣지, 코너"| B
    B --> |"텍스처, 패턴"| C
    C --> |"객체, 부품"| D["출력"]

    style A fill:#dbeafe
    style C fill:#dcfce7
```

## 11. CNN 활용 분야

```mermaid
flowchart TD
    A["CNN 활용"]

    A --> B["제조업"]
    B --> B1["불량 이미지 검출"]
    B --> B2["표면 결함 탐지"]

    A --> C["의료"]
    C --> C1["X-ray, MRI 분석"]

    A --> D["자율주행"]
    D --> D1["객체 인식"]

    style A fill:#1e40af,color:#fff
```

## 12. 전이학습

```mermaid
flowchart TD
    A["전이학습"]

    A --> B["사전 학습 모델"]
    B --> B1["VGG16, ResNet<br>ImageNet"]

    A --> C["특징 추출기"]
    C --> C1["가중치 동결"]

    A --> D["새 분류기"]
    D --> D1["우리 문제에 맞게<br>학습"]

    style A fill:#1e40af,color:#fff
```

## 13. 전이학습 장점

```mermaid
flowchart TD
    A["전이학습 장점"]

    A --> B["적은 데이터"]
    B --> B1["수백 장으로 가능"]

    A --> C["빠른 학습"]
    C --> C1["분 단위"]

    A --> D["높은 성능"]
    D --> D1["사전 지식 활용"]

    style A fill:#1e40af,color:#fff
```

## 14. RNN 구조

```mermaid
flowchart LR
    subgraph t0
        X0["x₀"]
        H0["h₀"]
    end

    subgraph t1
        X1["x₁"]
        H1["h₁"]
    end

    subgraph t2
        X2["x₂"]
        H2["h₂"]
    end

    subgraph t3
        X3["x₃"]
        H3["h₃"]
    end

    X0 --> H0
    H0 --> H1
    X1 --> H1
    H1 --> H2
    X2 --> H2
    H2 --> H3
    X3 --> H3
    H3 --> Y["출력"]
```

## 15. RNN 은닉 상태

```mermaid
flowchart TD
    A["RNN 은닉 상태"]

    A --> B["역할"]
    B --> B1["과거 정보 기억"]
    B --> B2["문맥 유지"]

    A --> C["수식"]
    C --> C1["h_t = f(W·h_{t-1} + U·x_t)"]

    A --> D["문제"]
    D --> D1["긴 시퀀스에서<br>기울기 소실"]

    style A fill:#1e40af,color:#fff
```

## 16. LSTM 구조

```mermaid
flowchart TD
    A["LSTM"]

    A --> B["Forget Gate"]
    B --> B1["버릴 정보 결정"]

    A --> C["Input Gate"]
    C --> C1["저장할 정보 결정"]

    A --> D["Output Gate"]
    D --> D1["출력할 정보 결정"]

    A --> E["Cell State"]
    E --> E1["장기 기억 저장소"]

    style A fill:#1e40af,color:#fff
```

## 17. LSTM vs 기본 RNN

```mermaid
flowchart TD
    A["시퀀스 처리"]

    A --> B["기본 RNN"]
    B --> B1["짧은 시퀀스 OK"]
    B --> B2["긴 시퀀스 X"]

    A --> C["LSTM"]
    C --> C1["긴 시퀀스 OK"]
    C --> C2["장기 의존성 학습"]

    style C fill:#dcfce7
```

## 18. GRU 구조

```mermaid
flowchart TD
    A["GRU"]

    A --> B["Reset Gate"]
    B --> B1["이전 상태 무시 정도"]

    A --> C["Update Gate"]
    C --> C1["새 정보 반영 정도"]

    A --> D["특징"]
    D --> D1["LSTM 간소화"]
    D --> D2["파라미터 적음"]

    style A fill:#1e40af,color:#fff
```

## 19. LSTM 파라미터

```mermaid
flowchart TD
    A["LSTM 파라미터"]

    A --> B["units"]
    B --> B1["은닉 상태 크기"]

    A --> C["return_sequences"]
    C --> C1["True: 모든 시점"]
    C --> C2["False: 마지막만"]

    A --> D["dropout"]
    D --> D1["입력 드롭아웃"]

    style A fill:#1e40af,color:#fff
```

## 20. RNN 활용 분야

```mermaid
flowchart TD
    A["RNN 활용"]

    A --> B["제조업"]
    B --> B1["시계열 예측"]
    B --> B2["이상 탐지"]

    A --> C["금융"]
    C --> C1["주가 예측"]

    A --> D["자연어"]
    D --> D1["번역, 챗봇"]

    style A fill:#1e40af,color:#fff
```

## 21. 시퀀스 데이터 변환

```mermaid
flowchart LR
    A["원본 시계열"]
    B["시퀀스 생성"]
    C["X: (samples, seq_len, features)"]
    D["y: 다음 값"]

    A --> B --> C
    B --> D

    style A fill:#dbeafe
    style C fill:#dcfce7
```

## 22. ResNet Skip Connection

```mermaid
flowchart TD
    A["입력 x"]
    B["Conv"]
    C["BN + ReLU"]
    D["Conv"]
    E["BN"]
    F["+ (덧셈)"]
    G["ReLU"]
    H["출력 y"]

    A --> B --> C --> D --> E --> F --> G --> H
    A --> |"Skip"| F

    style F fill:#1e40af,color:#fff
```

## 23. ResNet 효과

```mermaid
flowchart TD
    A["ResNet"]

    A --> B["문제 해결"]
    B --> B1["깊은 네트워크 학습 가능"]
    B --> B2["기울기 직접 전달"]

    A --> C["결과"]
    C --> C1["152층까지 학습"]
    C --> C2["ImageNet 우승"]

    style A fill:#1e40af,color:#fff
```

## 24. Attention 개념

```mermaid
flowchart TD
    A["Attention"]

    A --> B["아이디어"]
    B --> B1["중요한 부분에 집중"]

    A --> C["예시"]
    C --> C1["'먹었다' 예측 시<br>'사과를'에 집중"]

    A --> D["가중치"]
    D --> D1["동적으로 결정"]

    style A fill:#1e40af,color:#fff
```

## 25. Transformer 구조

```mermaid
flowchart TD
    A["입력 임베딩"]
    B["Self-Attention"]
    C["Feed-Forward"]
    D["반복 N번"]
    E["출력"]

    A --> B --> C --> D --> E

    style B fill:#1e40af,color:#fff
```

## 26. Transformer 특징

```mermaid
flowchart TD
    A["Transformer"]

    A --> B["RNN 없음"]
    B --> B1["Attention만 사용"]

    A --> C["병렬 처리"]
    C --> C1["빠른 학습"]

    A --> D["활용"]
    D --> D1["GPT, BERT 기반"]

    style A fill:#1e40af,color:#fff
```

## 27. 대형 언어 모델

```mermaid
flowchart TD
    A["LLM"]

    A --> B["GPT-3"]
    B --> B1["1,750억 파라미터"]

    A --> C["GPT-4"]
    C --> C1["멀티모달"]

    A --> D["LLaMA"]
    D --> D1["오픈소스"]

    style A fill:#1e40af,color:#fff
```

## 28. Diffusion 모델

```mermaid
flowchart LR
    A["노이즈"]
    B["노이즈 제거"]
    C["이미지"]

    A --> |"역과정 학습"| B
    B --> C

    style A fill:#dbeafe
    style C fill:#dcfce7
```

## 29. CNN 아키텍처 진화

```mermaid
flowchart TD
    A["1998: LeNet"]
    B["2012: AlexNet"]
    C["2014: VGG"]
    D["2015: ResNet"]
    E["2019: EfficientNet"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 30. 아키텍처 선택 가이드

```mermaid
flowchart TD
    A["데이터 유형"]

    A --> B["정형 데이터"]
    B --> B1["MLP, ML"]

    A --> C["이미지"]
    C --> C1["CNN"]

    A --> D["시계열"]
    D --> D1["LSTM, GRU"]

    A --> E["텍스트"]
    E --> E1["Transformer"]

    style A fill:#1e40af,color:#fff
```

## 31. 제조업 딥러닝 로드맵

```mermaid
flowchart TD
    A["1단계: 정형 데이터"]
    B["2단계: 이미지"]
    C["3단계: 시계열"]
    D["4단계: 멀티모달"]

    A --> |"MLP"| B
    B --> |"CNN"| C
    C --> |"LSTM"| D

    A --> A1["센서 기반 예측 ✅"]
    B --> B1["불량 이미지 검출"]
    C --> C1["생산량 예측"]
    D --> D1["센서+이미지 통합"]

    style A fill:#dcfce7
```

## 32. 프레임워크 비교

```mermaid
flowchart TD
    A["딥러닝 프레임워크"]

    A --> B["TensorFlow/Keras"]
    B --> B1["산업 표준<br>배포 용이"]

    A --> C["PyTorch"]
    C --> C1["연구 표준<br>직관적"]

    A --> D["JAX"]
    D --> D1["고성능"]

    style A fill:#1e40af,color:#fff
```

## 33. Edge AI

```mermaid
flowchart TD
    A["Edge AI"]

    A --> B["개념"]
    B --> B1["기기에서 직접 AI 실행"]

    A --> C["장점"]
    C --> C1["실시간 처리"]
    C --> C2["네트워크 불필요"]
    C --> C3["데이터 보안"]

    style A fill:#1e40af,color:#fff
```

## 34. 핵심 정리

```mermaid
flowchart TD
    A["22차시 핵심"]

    A --> B["CNN"]
    B --> B1["이미지 특징 추출<br>전이학습"]

    A --> C["RNN/LSTM"]
    C --> C1["시계열 처리<br>장기 의존성"]

    A --> D["고급"]
    D --> D1["ResNet, Transformer<br>LLM, Diffusion"]

    style A fill:#1e40af,color:#fff
```

## 35. 다음 차시 연결

```mermaid
flowchart LR
    A["22차시<br>딥러닝 심화"]
    B["22차시<br>모델 해석"]

    A --> B

    A --> A1["CNN, RNN"]
    B --> B1["Feature Importance"]
    B --> B2["Permutation Importance"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
