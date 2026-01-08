---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [22차시] 딥러닝 심화

## CNN, RNN, 그리고 고급 아키텍처

---

# 학습 목표

1. **CNN**(합성곱 신경망)의 구조와 원리를 이해한다
2. **RNN**(순환 신경망)의 특징과 활용을 파악한다
3. **고급 아키텍처**의 개념을 살펴본다

---

# 지난 시간 복습

- **Keras로 MLP 구현**: Sequential, Dense, Dropout
- **학습 개선**: EarlyStopping, BatchNormalization
- **품질 예측**: 센서 데이터 → 불량 여부 분류

**오늘**: MLP를 넘어서 다양한 딥러닝 아키텍처 탐구

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | CNN (합성곱 신경망) |
| 대주제 2 | 10분 | RNN (순환 신경망) |
| 대주제 3 | 8분 | 고급 아키텍처 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## CNN (합성곱 신경망)

---

# 왜 CNN인가?

**이미지 처리의 한계**:
- MLP로 이미지 처리 → 파라미터 폭발
- 28×28 이미지 → 784개 입력
- 첫 은닉층 128노드 → 100,000개 파라미터!

**CNN의 해결책**:
- 지역적 패턴 학습 (필터)
- 파라미터 공유
- 위치 불변성

---

# CNN 핵심 개념

```
[입력 이미지]
    ↓
[합성곱층 Conv2D] ← 필터로 특징 추출
    ↓
[풀링층 MaxPooling] ← 크기 축소
    ↓
[평탄화 Flatten]
    ↓
[완전연결층 Dense]
    ↓
[출력]
```

---

# 합성곱(Convolution) 연산

```
필터 (3×3):        입력 이미지 일부:
[1  0 -1]          [10 20 30]
[1  0 -1]    *     [40 50 60]  = 합성곱 결과
[1  0 -1]          [70 80 90]

계산:
1×10 + 0×20 + (-1)×30 +
1×40 + 0×50 + (-1)×60 +
1×70 + 0×80 + (-1)×90 = -60
```

---

# 필터의 역할

| 필터 유형 | 학습하는 패턴 |
|----------|--------------|
| 수평 엣지 | 가로 경계선 |
| 수직 엣지 | 세로 경계선 |
| 대각 엣지 | 대각선 경계 |
| 코너 | 모서리 패턴 |

**CNN은 필터 값을 자동으로 학습!**

---

# 특징 맵 (Feature Map)

```
입력 이미지 (28×28)
    ↓
필터 1 적용 → 특징 맵 1 (수평 엣지)
필터 2 적용 → 특징 맵 2 (수직 엣지)
필터 3 적용 → 특징 맵 3 (대각 패턴)
...
32개 필터 → 32개 특징 맵
```

---

# 풀링 (Pooling)

**목적**: 크기 축소, 계산량 감소, 위치 불변성

```
2×2 Max Pooling:

[1 3 | 2 4]      [3 | 4]
[5 6 | 7 8]  →   [7 | 8]

4×4 → 2×2로 축소
```

**최댓값을 선택** → 가장 강한 특징 유지

---

# Keras에서 CNN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential([
    # 합성곱 블록 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # 합성곱 블록 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 분류기
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

# Conv2D 파라미터

```python
Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
       padding='same', strides=(1, 1))
```

| 파라미터 | 의미 |
|---------|------|
| filters | 필터 개수 (출력 채널) |
| kernel_size | 필터 크기 (3×3) |
| padding | 'same': 크기 유지, 'valid': 줄어듦 |
| strides | 필터 이동 간격 |

---

# CNN 활용 분야

| 분야 | 응용 |
|-----|------|
| 제조업 | **불량 이미지 검출**, 표면 결함 탐지 |
| 의료 | X-ray, MRI 분석, 세포 분류 |
| 자율주행 | 객체 인식, 차선 감지 |
| 보안 | 얼굴 인식, 지문 인식 |

---

# 제조업 CNN 예시

```python
# 불량 이미지 분류 모델
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])
```

---

# CNN 아키텍처 진화

| 모델 | 년도 | 특징 |
|-----|-----|------|
| LeNet | 1998 | 최초의 CNN |
| AlexNet | 2012 | ImageNet 우승, 딥러닝 부흥 |
| VGG | 2014 | 깊은 구조 (16-19층) |
| ResNet | 2015 | Skip Connection |
| EfficientNet | 2019 | 효율적인 스케일링 |

---

# 전이학습 (Transfer Learning)

**아이디어**: 미리 학습된 모델 재활용

```python
from tensorflow.keras.applications import VGG16

# 사전 학습된 VGG16 로드
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))

# 가중치 동결
base_model.trainable = False

# 새로운 분류기 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')  # 불량 분류
])
```

---

# 전이학습 장점

| 항목 | 직접 학습 | 전이학습 |
|-----|----------|----------|
| 데이터 필요량 | 수만~수십만 | 수백~수천 |
| 학습 시간 | 수 시간~일 | 수 분~시간 |
| 성능 | 데이터 의존 | 높은 기본 성능 |

**적은 데이터로도 좋은 성능!**

---

<!-- _class: lead -->
# 대주제 2
## RNN (순환 신경망)

---

# 왜 RNN인가?

**MLP의 한계**:
- 입력이 고정 크기
- 순서 정보 무시

**시계열/순차 데이터**:
- 센서 값의 **시간 순서** 중요
- 텍스트의 **단어 순서** 중요
- 이전 상태가 현재에 영향

---

# RNN 핵심 개념

```
    t=0      t=1      t=2      t=3
     ↓        ↓        ↓        ↓
    [h0] → [h1] → [h2] → [h3] → 출력
     ↑        ↑        ↑        ↑
    x0       x1       x2       x3
```

**은닉 상태(h)가 다음 시점으로 전달**

---

# RNN 수식

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
y_t = W_hy × h_t
```

| 기호 | 의미 |
|-----|------|
| h_t | 현재 은닉 상태 |
| h_{t-1} | 이전 은닉 상태 |
| x_t | 현재 입력 |
| W | 가중치 행렬 |

---

# 기본 RNN 문제점

**기울기 소실 (Vanishing Gradient)**:
- 긴 시퀀스에서 먼 과거 정보 손실
- 역전파 시 기울기가 점점 작아짐

**해결책**:
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

---

# LSTM 구조

```
                    ┌──────────┐
    forget gate ──→ │ Cell     │ ←── input gate
                    │ State    │
    output gate ──→ └────┬─────┘
                         ↓
                    [hidden state]
```

**게이트로 정보 흐름 제어**

---

# LSTM 게이트

| 게이트 | 역할 |
|-------|------|
| Forget Gate | 이전 정보 중 버릴 것 결정 |
| Input Gate | 새 정보 중 저장할 것 결정 |
| Output Gate | 출력할 정보 결정 |

**셀 상태**: 장기 기억 저장소

---

# Keras에서 LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # LSTM 층 (시퀀스 입력)
    LSTM(64, input_shape=(sequence_length, n_features),
         return_sequences=True),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    # 출력층
    Dense(1)  # 다음 값 예측
])

model.compile(optimizer='adam', loss='mse')
```

---

# LSTM 파라미터

```python
LSTM(units=64, return_sequences=True, dropout=0.2)
```

| 파라미터 | 의미 |
|---------|------|
| units | 은닉 상태 크기 |
| return_sequences | True: 모든 시점 출력 |
| dropout | 입력 드롭아웃 비율 |
| recurrent_dropout | 순환 드롭아웃 비율 |

---

# GRU (Gated Recurrent Unit)

**LSTM 간소화 버전**:
- 게이트 2개 (Reset, Update)
- 파라미터 적음, 빠른 학습
- LSTM과 비슷한 성능

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(64, input_shape=(seq_len, n_features)),
    Dense(1)
])
```

---

# RNN 활용 분야

| 분야 | 응용 |
|-----|------|
| 제조업 | **시계열 예측**, 설비 이상 탐지 |
| 금융 | 주가 예측, 거래 패턴 분석 |
| 자연어 | 번역, 챗봇, 감성 분석 |
| 음성 | 음성 인식, 음악 생성 |

---

# 제조업 LSTM 예시

```python
# 시계열 생산량 예측
model = Sequential([
    LSTM(128, input_shape=(24, 5),  # 24시간, 5개 특성
         return_sequences=True),
    Dropout(0.2),

    LSTM(64),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)  # 다음 시점 생산량
])

model.compile(optimizer='adam', loss='mse',
              metrics=['mae'])
```

---

# 데이터 형태 변환

```python
# 시계열 → 시퀀스 데이터
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 예: 과거 24시간 → 다음 1시간 예측
X, y = create_sequences(sensor_data, seq_length=24)
# X.shape: (samples, 24, n_features)
```

---

# LSTM vs 전통 시계열

| 항목 | ARIMA | LSTM |
|-----|-------|------|
| 다변량 처리 | 어려움 | 쉬움 |
| 비선형 패턴 | 제한적 | 강함 |
| 해석 | 쉬움 | 어려움 |
| 데이터 필요량 | 적음 | 많음 |

---

<!-- _class: lead -->
# 대주제 3
## 고급 아키텍처

---

# 딥러닝 아키텍처 발전

```
1998: LeNet (CNN 시작)
    ↓
2012: AlexNet (딥러닝 부흥)
    ↓
2014: VGG, GoogLeNet
    ↓
2015: ResNet (Skip Connection)
    ↓
2017: Transformer (Attention)
    ↓
2022: GPT-4, Diffusion Models
```

---

# ResNet: Skip Connection

**문제**: 깊은 네트워크 학습 어려움

**해결**: 입력을 출력에 더하기

```
    x ─────────────┐
    ↓              │
[Conv] → [BN] → [ReLU] → [+] → y
    ↓                    ↑
[Conv] → [BN] ──────────┘

y = F(x) + x  (Residual)
```

---

# ResNet 효과

| 네트워크 깊이 | 일반 CNN | ResNet |
|-------------|----------|--------|
| 20층 | 학습 가능 | 학습 가능 |
| 50층 | 어려움 | 학습 가능 |
| 152층 | 불가능 | 학습 가능 |

**기울기가 직접 전달 → 깊은 네트워크 가능**

---

# Attention 메커니즘

**아이디어**: 중요한 부분에 집중

```
입력: "나는 사과를 먹었다"

"먹었다"를 예측할 때:
- "나는" → 낮은 attention
- "사과를" → 높은 attention ←
```

**문맥에 따라 가중치 동적 결정**

---

# Transformer

**특징**:
- RNN 없이 Attention만으로 시퀀스 처리
- 병렬 처리 가능 → 빠른 학습
- GPT, BERT의 기반

**구조**:
- Self-Attention
- Multi-Head Attention
- Feed-Forward Network

---

# Transformer 아키텍처

```
[입력 임베딩]
     ↓
[Self-Attention]  ← 문맥 파악
     ↓
[Feed-Forward]
     ↓
(반복 N번)
     ↓
[출력]
```

---

# 대형 언어 모델 (LLM)

| 모델 | 파라미터 수 | 특징 |
|-----|-----------|------|
| BERT | 3억 | 양방향 문맥 |
| GPT-3 | 1,750억 | 생성 능력 |
| GPT-4 | 1조+ (추정) | 멀티모달 |
| LLaMA | 70억~700억 | 오픈소스 |

---

# Diffusion Models

**이미지 생성 모델**:
- DALL-E 2
- Stable Diffusion
- Midjourney

**원리**:
1. 이미지에 노이즈 추가
2. 노이즈 제거 과정 학습
3. 노이즈 → 이미지 생성

---

# 제조업 적용 가능성

| 기술 | 제조업 활용 |
|-----|------------|
| CNN | 불량 이미지 검출, 표면 검사 |
| RNN/LSTM | 시계열 예측, 이상 탐지 |
| Transformer | 설비 로그 분석, 문서 처리 |
| Diffusion | 합성 불량 이미지 생성 (데이터 증강) |

---

# AutoML과 NAS

**AutoML**: 자동 머신러닝
- 하이퍼파라미터 자동 탐색
- 특성 엔지니어링 자동화

**NAS**: Neural Architecture Search
- 최적 아키텍처 자동 탐색
- EfficientNet이 NAS로 설계됨

---

# Edge AI

**개념**: 기기에서 직접 AI 실행

```
[클라우드 AI]         [Edge AI]
센서 → 클라우드 → 결과    센서 → 기기 → 결과
    (지연 발생)             (즉시)
```

**제조업 적용**:
- 실시간 불량 검출
- 네트워크 없이 동작
- 데이터 보안

---

# 딥러닝 프레임워크 비교

| 프레임워크 | 특징 |
|-----------|------|
| TensorFlow/Keras | 산업 표준, 배포 용이 |
| PyTorch | 연구 표준, 직관적 |
| JAX | 고성능, 함수형 |
| ONNX | 프레임워크 간 호환 |

---

# 학습 방향 제안

```
기초
├── MLP (Dense)        ← 20차시에서 완료!
├── CNN (이미지)       ← 필요시 심화
└── RNN/LSTM (시계열)  ← 필요시 심화

심화
├── Transformer (NLP)
├── 전이학습
└── 배포 (TensorFlow Serving)
```

---

<!-- _class: lead -->
# 핵심 정리

---

# 아키텍처 선택 가이드

| 데이터 유형 | 권장 아키텍처 |
|-----------|--------------|
| 정형 데이터 (테이블) | MLP, ML |
| 이미지 | CNN |
| 시계열/순차 | LSTM, GRU |
| 텍스트 | Transformer, BERT |
| 다양한 입력 | 멀티모달 모델 |

---

# 오늘 배운 내용

1. **CNN (합성곱 신경망)**
   - 이미지 특징 자동 추출
   - Conv2D → Pooling → Dense
   - 전이학습으로 효율화

2. **RNN (순환 신경망)**
   - 시계열/순차 데이터 처리
   - LSTM, GRU로 장기 의존성 학습

3. **고급 아키텍처**
   - ResNet, Transformer, Diffusion
   - 제조업 적용 가능성

---

# 제조업 딥러닝 로드맵

```
1단계: 정형 데이터 (MLP)
   └─ 센서 데이터 품질 예측 ✅ 완료

2단계: 이미지 데이터 (CNN)
   └─ 불량 이미지 분류

3단계: 시계열 데이터 (LSTM)
   └─ 생산량/품질 예측

4단계: 멀티모달
   └─ 센서 + 이미지 통합
```

---

# 다음 차시 예고

## [22차시] 모델 해석과 변수별 영향력 분석

- Feature Importance
- Permutation Importance
- SHAP 개념 소개

---

<!-- _class: lead -->
# 수고하셨습니다!

## 질문: CNN/RNN을 어디에 적용할 수 있을까요?
