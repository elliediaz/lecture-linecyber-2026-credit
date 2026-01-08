"""
[22차시] 딥러닝 심화 - 실습 코드

학습 목표:
1. CNN(합성곱 신경망)의 구조와 원리를 이해한다
2. RNN(순환 신경망)의 특징과 활용을 파악한다
3. 고급 아키텍처의 개념을 살펴본다

실습 환경: Python 3.8+, TensorFlow 2.x, NumPy
"""

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow/Keras 임포트
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Dropout, Flatten,
        Conv2D, MaxPooling2D,
        LSTM, GRU, SimpleRNN
    )
    print(f"TensorFlow 버전: {tf.__version__}")
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow가 설치되지 않았습니다.")
    KERAS_AVAILABLE = False

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[22차시] 딥러닝 심화")
print("=" * 60)

# ============================================================
# 1. 합성곱 연산 이해
# ============================================================
print("\n" + "=" * 60)
print("1. 합성곱 연산 이해")
print("=" * 60)

def convolution_2d(image, kernel):
    """
    2D 합성곱 연산 (단순 구현)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# 예시 이미지 (5x5)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# 수직 엣지 필터
vertical_edge = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

# 수평 엣지 필터
horizontal_edge = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=float)

print("\n입력 이미지 (5x5):")
print(image)

print("\n수직 엣지 필터:")
print(vertical_edge)

output_v = convolution_2d(image, vertical_edge)
print("\n수직 엣지 적용 결과:")
print(output_v)

output_h = convolution_2d(image, horizontal_edge)
print("\n수평 엣지 적용 결과:")
print(output_h)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Input Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(vertical_edge, cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title('Vertical Edge Filter')
axes[0, 1].axis('off')

axes[0, 2].imshow(output_v, cmap='RdBu')
axes[0, 2].set_title('Vertical Edge Result')
axes[0, 2].axis('off')

axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title('Input Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(horizontal_edge, cmap='RdBu', vmin=-1, vmax=1)
axes[1, 1].set_title('Horizontal Edge Filter')
axes[1, 1].axis('off')

axes[1, 2].imshow(output_h, cmap='RdBu')
axes[1, 2].set_title('Horizontal Edge Result')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('convolution_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  → 'convolution_demo.png' 저장됨")

# ============================================================
# 2. MaxPooling 이해
# ============================================================
print("\n" + "=" * 60)
print("2. MaxPooling 이해")
print("=" * 60)

def max_pooling_2d(image, pool_size=2):
    """
    2D Max Pooling (단순 구현)
    """
    h, w = image.shape
    output_h = h // pool_size
    output_w = w // pool_size
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output

# 예시
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print("\n특징 맵 (4x4):")
print(feature_map)

pooled = max_pooling_2d(feature_map, pool_size=2)
print("\nMax Pooling (2x2) 결과:")
print(pooled)

print("\n설명:")
print("  - 왼쪽 상단 [1,3,5,6] → max = 6")
print("  - 오른쪽 상단 [2,4,7,8] → max = 8")
print("  - 크기: 4x4 → 2x2 (1/4로 축소)")

# ============================================================
# 3. CNN 모델 예시
# ============================================================
print("\n" + "=" * 60)
print("3. CNN 모델 예시")
print("=" * 60)

if KERAS_AVAILABLE:
    # 이미지 분류 CNN
    cnn_model = Sequential([
        # 합성곱 블록 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        # 합성곱 블록 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # 합성곱 블록 3
        Conv2D(64, (3, 3), activation='relu'),

        # 분류기
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    print("\nCNN 모델 구조:")
    cnn_model.summary()

    # 파라미터 계산 설명
    print("\n파라미터 계산:")
    print("  Conv2D(32, 3x3): 1×3×3×32 + 32 = 320")
    print("  Conv2D(64, 3x3): 32×3×3×64 + 64 = 18,496")
    print("  Conv2D(64, 3x3): 64×3×3×64 + 64 = 36,928")
    print("  Dense(64): 576×64 + 64 = 36,928")
    print("  Dense(10): 64×10 + 10 = 650")
else:
    print("Keras를 사용할 수 없습니다.")

# ============================================================
# 4. RNN 구조 이해
# ============================================================
print("\n" + "=" * 60)
print("4. RNN 구조 이해")
print("=" * 60)

class SimpleRNNCell:
    """
    단순 RNN 셀 (NumPy 구현)
    """
    def __init__(self, input_size, hidden_size):
        # 가중치 초기화
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        """
        순전파
        h_t = tanh(Wxh * x + Whh * h_prev + bh)
        """
        h = np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h

    def process_sequence(self, X):
        """
        전체 시퀀스 처리
        X: (seq_len, input_size)
        """
        seq_len = X.shape[0]
        h = np.zeros((1, self.hidden_size))  # 초기 은닉 상태

        hidden_states = []
        for t in range(seq_len):
            x_t = X[t:t+1, :]  # (1, input_size)
            h = self.forward(x_t, h)
            hidden_states.append(h.copy())

        return np.vstack(hidden_states)

# RNN 예시
np.random.seed(42)
rnn_cell = SimpleRNNCell(input_size=3, hidden_size=4)

# 시퀀스 입력 (5 시점, 3 특성)
X_seq = np.random.randn(5, 3)
print("\n입력 시퀀스 (5 시점 × 3 특성):")
print(X_seq.round(2))

# 시퀀스 처리
hidden_states = rnn_cell.process_sequence(X_seq)
print("\n은닉 상태 (5 시점 × 4 은닉):")
print(hidden_states.round(3))

print("\n설명:")
print("  - 각 시점의 은닉 상태가 다음 시점으로 전달됨")
print("  - 마지막 은닉 상태에 전체 시퀀스 정보가 누적됨")

# ============================================================
# 5. LSTM 셀 개념
# ============================================================
print("\n" + "=" * 60)
print("5. LSTM 셀 개념")
print("=" * 60)

print("""
LSTM (Long Short-Term Memory) 구조:

┌─────────────────────────────────────────────┐
│                Cell State (C)               │
│        (장기 기억 - 컨베이어 벨트)           │
└─────────────────────────────────────────────┘
       ↑ forget    ↑ input         │ output
       │           │               ↓
┌──────┴───┐ ┌─────┴─────┐ ┌───────┴───────┐
│ Forget   │ │  Input    │ │    Output     │
│  Gate    │ │   Gate    │ │     Gate      │
│ (버릴것) │ │ (저장할것)│ │  (출력할것)   │
└──────────┘ └───────────┘ └───────────────┘
       ↑           ↑               ↑
       └───────────┴───────────────┘
                   │
            h(t-1), x(t)

게이트 수식:
- Forget: f = σ(Wf·[h, x] + bf)
- Input:  i = σ(Wi·[h, x] + bi)
- Output: o = σ(Wo·[h, x] + bo)
- Cell:   C = f*C + i*tanh(Wc·[h, x] + bc)
- Hidden: h = o*tanh(C)
""")

# ============================================================
# 6. Keras RNN 모델 예시
# ============================================================
print("\n" + "=" * 60)
print("6. Keras RNN 모델 예시")
print("=" * 60)

if KERAS_AVAILABLE:
    # 시계열 예측 LSTM
    lstm_model = Sequential([
        # 첫 번째 LSTM 층 (시퀀스 출력)
        LSTM(64, input_shape=(24, 5), return_sequences=True),
        Dropout(0.2),

        # 두 번째 LSTM 층
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # 출력층
        Dense(16, activation='relu'),
        Dense(1)  # 다음 값 예측
    ])

    print("\nLSTM 모델 구조:")
    lstm_model.summary()

    print("\n입력 형태 설명:")
    print("  input_shape=(24, 5)")
    print("  - 24: 시퀀스 길이 (예: 24시간)")
    print("  - 5: 특성 수 (예: 온도, 압력, 속도, 습도, 진동)")

    print("\nreturn_sequences 설명:")
    print("  - True: 모든 시점의 출력 반환 (24, 64)")
    print("  - False: 마지막 시점만 반환 (32,)")

    # GRU 모델
    gru_model = Sequential([
        GRU(64, input_shape=(24, 5), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dense(1)
    ])

    print("\nGRU 모델 구조:")
    gru_model.summary()

    print("\nLSTM vs GRU 파라미터 비교:")
    lstm_params = lstm_model.count_params()
    gru_params = gru_model.count_params()
    print(f"  LSTM: {lstm_params:,} 파라미터")
    print(f"  GRU:  {gru_params:,} 파라미터")
    print(f"  차이: GRU가 {(lstm_params - gru_params):,} 적음")

# ============================================================
# 7. 시퀀스 데이터 생성
# ============================================================
print("\n" + "=" * 60)
print("7. 시퀀스 데이터 생성")
print("=" * 60)

def create_sequences(data, seq_length):
    """
    시계열 데이터를 시퀀스로 변환
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 예시 시계열 데이터
np.random.seed(42)
time_series = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)

print("\n원본 시계열 데이터:")
print(f"  길이: {len(time_series)}")
print(f"  처음 10개: {time_series[:10].round(2)}")

# 시퀀스 생성
seq_length = 10
X_seq, y_seq = create_sequences(time_series, seq_length)

print(f"\n시퀀스 생성 (seq_length={seq_length}):")
print(f"  X shape: {X_seq.shape}")
print(f"  y shape: {y_seq.shape}")

print("\n예시:")
print(f"  X[0] (과거 10개): {X_seq[0].round(2)}")
print(f"  y[0] (다음 1개): {y_seq[0]:.2f}")

# RNN 입력 형태로 변환
X_rnn = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
print(f"\nRNN 입력 형태: {X_rnn.shape}")
print("  (samples, timesteps, features)")

# ============================================================
# 8. ResNet Skip Connection 개념
# ============================================================
print("\n" + "=" * 60)
print("8. ResNet Skip Connection 개념")
print("=" * 60)

print("""
ResNet의 핵심: Skip Connection (Residual Learning)

일반 네트워크:
    x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → y
    y = F(x)

ResNet:
    x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → (+) → [ReLU] → y
                                              ↑
    x ─────────────────────────────────────────┘
    y = F(x) + x  (잔차 학습)

효과:
1. 기울기가 직접 전달 → 깊은 네트워크 학습 가능
2. 152층까지도 학습 가능
3. 최소한 입력만큼은 출력 (성능 하한 보장)
""")

# Skip Connection 시뮬레이션
def residual_block(x, F):
    """잔차 블록 시뮬레이션"""
    return F + x  # y = F(x) + x

x = 10  # 입력
F = 2   # 변환 (배워야 할 것)

# 일반 네트워크: y = F(x) = 12를 직접 학습
# ResNet: F(x) = 2만 학습하면 y = 2 + 10 = 12

print("\n수치 예시:")
print(f"  입력 x: {x}")
print(f"  변환 F(x): {F}")
print(f"  출력 y = F(x) + x: {residual_block(x, F)}")
print("\n  → 작은 변화만 학습하면 되므로 더 쉬움!")

# ============================================================
# 9. Attention 메커니즘 개념
# ============================================================
print("\n" + "=" * 60)
print("9. Attention 메커니즘 개념")
print("=" * 60)

def simple_attention(query, keys, values):
    """
    단순 Attention 예시

    query: 현재 질문/상태 (1, d)
    keys: 비교 대상들 (n, d)
    values: 가져올 값들 (n, d)
    """
    # 1. Score 계산 (내적)
    scores = query @ keys.T  # (1, n)

    # 2. Softmax로 가중치 변환
    weights = np.exp(scores) / np.sum(np.exp(scores))

    # 3. 가중 평균
    output = weights @ values  # (1, d)

    return output, weights

# 예시: 번역에서 Attention
# "나는 사과를 먹었다" → "I ate an apple"
# "ate" 예측 시 어디에 집중?

np.random.seed(42)

# 단어 임베딩 (4차원)
words = {
    '나는': np.array([0.1, 0.2, 0.3, 0.1]),
    '사과를': np.array([0.8, 0.1, 0.2, 0.9]),
    '먹었다': np.array([0.2, 0.9, 0.8, 0.2])
}

keys = np.vstack(list(words.values()))  # (3, 4)
values = keys.copy()

# "ate"를 예측할 때의 query (동사 관련)
query = np.array([[0.3, 0.8, 0.7, 0.3]])  # 동사 성격

output, weights = simple_attention(query, keys, values)

print("\nAttention 예시 (번역):")
print("  입력: '나는 사과를 먹었다'")
print("  예측 대상: 'ate'")
print("\n  Attention 가중치:")
for word, w in zip(words.keys(), weights[0]):
    bar = '█' * int(w * 20)
    print(f"    {word}: {w:.3f} {bar}")

print("\n  → '먹었다'에 가장 높은 attention!")

# ============================================================
# 10. 아키텍처 비교
# ============================================================
print("\n" + "=" * 60)
print("10. 아키텍처 비교")
print("=" * 60)

print("""
딥러닝 아키텍처 비교:

| 아키텍처    | 데이터 유형  | 핵심 연산        | 제조업 활용             |
|------------|-------------|-----------------|------------------------|
| MLP        | 정형        | 행렬 곱         | 품질 예측, 분류         |
| CNN        | 이미지      | 합성곱          | 불량 이미지 검출        |
| RNN/LSTM   | 시계열      | 순환 연결        | 생산량 예측, 이상 탐지   |
| Transformer| 시퀀스      | Self-Attention  | 로그 분석, 문서 처리    |

선택 기준:
1. 정형 데이터 (센서 값) → MLP 또는 ML (RandomForest)
2. 이미지 데이터 → CNN (전이학습 권장)
3. 시계열 데이터 → LSTM/GRU
4. 텍스트 데이터 → Transformer (BERT, GPT)
""")

# ============================================================
# 11. 전이학습 예시 코드
# ============================================================
print("\n" + "=" * 60)
print("11. 전이학습 예시 코드")
print("=" * 60)

if KERAS_AVAILABLE:
    print("""
# 전이학습 코드 예시 (실행하지 않음)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# 1. 사전 학습된 모델 로드
base_model = VGG16(
    weights='imagenet',      # ImageNet 가중치
    include_top=False,       # 분류기 제외
    input_shape=(224, 224, 3)
)

# 2. 가중치 동결 (학습 안 함)
base_model.trainable = False

# 3. 새 분류기 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])

# 4. 컴파일 및 학습
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 학습 (적은 데이터로 가능!)
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)
""")

    print("\n전이학습 장점:")
    print("  - 수백 장의 이미지로도 학습 가능")
    print("  - 빠른 학습 (몇 분 이내)")
    print("  - 높은 기본 성능 (ImageNet 지식 활용)")

# ============================================================
# 12. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("12. 핵심 정리")
print("=" * 60)

print("""
[22차시 핵심 정리]

1. CNN (합성곱 신경망)
   - 이미지 데이터 처리에 특화
   - 합성곱: 필터로 지역 패턴 추출
   - 풀링: 크기 축소, 위치 불변성
   - 전이학습: 적은 데이터로 좋은 성능

2. RNN (순환 신경망)
   - 시계열/순차 데이터 처리
   - 은닉 상태로 과거 정보 기억
   - LSTM: 장기 의존성 학습 (게이트 메커니즘)
   - GRU: LSTM 간소화, 비슷한 성능

3. 고급 아키텍처
   - ResNet: Skip Connection으로 깊은 네트워크
   - Transformer: Attention으로 병렬 처리
   - LLM: 수천억 파라미터 언어 모델

4. 아키텍처 선택
   - 정형 데이터 → MLP, ML
   - 이미지 → CNN
   - 시계열 → LSTM, GRU
   - 텍스트 → Transformer
""")

print("\n다음 차시 예고: 모델 해석 (Feature Importance)")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
