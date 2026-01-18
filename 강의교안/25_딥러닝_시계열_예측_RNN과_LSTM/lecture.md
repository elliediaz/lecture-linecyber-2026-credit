# 25차시: 딥러닝 응용 - 시계열 예측 RNN과 LSTM

## 학습 목표

| 목표 | 설명 |
|------|------|
| RNN 구조 이해 | 순환 신경망의 구조와 시퀀스 데이터 처리 원리를 이해합니다 |
| LSTM/GRU 원리 | 게이트 구조로 장기 의존성 문제를 해결하는 방법을 학습합니다 |
| Keras 구현 | LSTM 모델을 Keras로 구성하고 학습합니다 |
| 센서 예측 | Predictive Maintenance AI4I 2020 데이터로 센서값 예측을 수행합니다 |

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|---------|------|------|
| Predictive Maintenance AI4I 2020 | UCI ML Repository / Kaggle | 제조 센서 시계열 예측 |

```python
# 데이터셋 다운로드
# https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

# 주요 컬럼:
# - Air temperature [K]: 공기 온도
# - Process temperature [K]: 공정 온도
# - Rotational speed [rpm]: 회전 속도
# - Torque [Nm]: 토크
# - Tool wear [min]: 공구 마모
```

---

## 수학적 배경

### RNN 기본 수식

순환 신경망(RNN)의 은닉 상태 업데이트:

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

| 기호 | 의미 |
|------|------|
| $h_t$ | 시점 t의 은닉 상태 (hidden state) |
| $h_{t-1}$ | 이전 시점의 은닉 상태 |
| $x_t$ | 시점 t의 입력 벡터 |
| $W_h$ | 은닉 상태에 대한 가중치 행렬 |
| $W_x$ | 입력에 대한 가중치 행렬 |
| $b$ | 편향 벡터 |
| $\tanh$ | 하이퍼볼릭 탄젠트 활성화 함수 (-1 ~ 1) |

### LSTM 게이트 수식

**Forget Gate (잊기 게이트)**:
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**Input Gate (입력 게이트)**:
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**Cell State 업데이트**:
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**Output Gate (출력 게이트)**:
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

| 기호 | 의미 |
|------|------|
| $f_t$ | Forget gate 출력 (0~1, 잊을 비율) |
| $i_t$ | Input gate 출력 (0~1, 저장할 비율) |
| $o_t$ | Output gate 출력 (0~1, 출력할 비율) |
| $C_t$ | Cell state (장기 기억) |
| $\tilde{C}_t$ | 후보 셀 상태 |
| $\sigma$ | 시그모이드 함수 (0~1) |
| $\odot$ | 요소별 곱셈 (element-wise) |

### GRU 게이트 수식

**Reset Gate**:
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

**Update Gate**:
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**Hidden State 업데이트**:
$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

## 강의 구성 (25분)

| 파트 | 주제 | 시간 |
|:----:|------|:----:|
| Part 1 | RNN 기초 - 순환 구조와 시퀀스 처리 | 7분 |
| Part 2 | LSTM과 GRU - 장기 의존성 해결 | 8분 |
| Part 3 | 제조 센서 시계열 예측 실습 | 8분 |
| 정리 | 핵심 요약 | 2분 |

---

## Part 1: RNN 기초 - 순환 구조와 시퀀스 처리 (7분)

### 1.1 시퀀스 데이터의 특성

시퀀스 데이터는 순서가 중요한 데이터입니다. 일반 데이터와의 차이점을 이해하는 것이 RNN 학습의 출발점입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 일반 데이터 vs 시퀀스 데이터 비교
print("=" * 60)
print("일반 데이터 vs 시퀀스 데이터 비교")
print("=" * 60)

# 일반 데이터 예시 (순서 무관)
general_data = {
    'feature_1': [1.2, 2.3, 3.1],
    'feature_2': [4.5, 5.6, 6.2],
    'label': ['A', 'B', 'A']
}
print("\n일반 데이터 (MLP 입력):")
print("  - 각 샘플이 독립적")
print("  - 순서를 바꿔도 동일한 결과")
print(f"  예: {general_data}")

# 시퀀스 데이터 예시 (순서 중요)
sequence_data = [
    [25.3, 1500, 40.2],  # t=0
    [25.8, 1520, 41.1],  # t=1
    [26.5, 1535, 42.3],  # t=2
    [27.9, 1548, 44.8],  # t=3
]
print("\n시퀀스 데이터 (RNN 입력):")
print("  - 이전 시점이 현재에 영향")
print("  - 순서를 바꾸면 의미 변화")
print("  예: 센서 시계열")
for i, row in enumerate(sequence_data):
    print(f"    t={i}: 온도={row[0]}°C, 회전수={row[1]}rpm, 토크={row[2]}Nm")
```

**시퀀스 데이터의 종류**:

| 유형 | 예시 | 특징 |
|------|------|------|
| 시계열 | 센서 데이터, 주가 | 시간 간격이 일정 |
| 자연어 | 문장, 문서 | 단어 순서가 의미 결정 |
| 음성 | 음파, 음악 | 연속적 신호 |
| 동영상 | 프레임 시퀀스 | 시간에 따른 이미지 변화 |

---

### 1.2 RNN의 순환 구조

RNN은 이전 시점의 정보를 "은닉 상태(hidden state)"를 통해 다음 시점으로 전달합니다.

```python
def simple_rnn_forward(x_sequence, W_h, W_x, b):
    """
    단순 RNN의 순전파 구현

    수식: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)

    Parameters:
        x_sequence: 입력 시퀀스 (seq_len, input_dim)
        W_h: 은닉 상태 가중치 (hidden_dim, hidden_dim)
        W_x: 입력 가중치 (hidden_dim, input_dim)
        b: 편향 (hidden_dim,)

    Returns:
        hidden_states: 모든 시점의 은닉 상태
        outputs: 각 시점의 출력
    """
    seq_len = len(x_sequence)
    hidden_dim = W_h.shape[0]

    # 초기 은닉 상태 (0으로 초기화)
    h_prev = np.zeros(hidden_dim)
    hidden_states = []

    print("RNN 순전파 과정:")
    print("-" * 50)

    for t, x_t in enumerate(x_sequence):
        # h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
        h_t = np.tanh(np.dot(W_h, h_prev) + np.dot(W_x, x_t) + b)

        print(f"t={t}: x_t={x_t:.2f}, h_{t-1}={h_prev[0]:.3f} -> h_t={h_t[0]:.3f}")

        hidden_states.append(h_t)
        h_prev = h_t

    return np.array(hidden_states)

# 예시: 간단한 1차원 RNN
np.random.seed(42)
hidden_dim = 2
input_dim = 1

W_h = np.random.randn(hidden_dim, hidden_dim) * 0.5
W_x = np.random.randn(hidden_dim, input_dim) * 0.5
b = np.zeros(hidden_dim)

# 센서 온도 시퀀스 (단순화)
temperature_sequence = np.array([[25.3], [25.8], [26.5], [27.9], [30.2]])

print("\n입력 시퀀스 (온도):", temperature_sequence.flatten())
hidden_states = simple_rnn_forward(temperature_sequence, W_h, W_x, b)
print("\n최종 은닉 상태:", hidden_states[-1])
print("-> 전체 시퀀스의 정보가 압축되어 있음")
```

---

### 1.3 RNN 은닉 상태의 역할 시각화

```python
# RNN 구조 다이어그램
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: RNN 셀 내부 구조
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# RNN 셀 박스
rect = plt.Rectangle((2, 2), 6, 6, fill=False, edgecolor='black', linewidth=2)
ax1.add_patch(rect)
ax1.text(5, 5, 'RNN Cell\n$h_t = tanh(W_h h_{t-1} + W_x x_t + b)$',
         ha='center', va='center', fontsize=10)

# 입력 화살표
ax1.annotate('', xy=(2, 5), xytext=(0, 5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax1.text(0.5, 5.5, '$x_t$', fontsize=12, ha='center')

# 이전 은닉 상태 화살표
ax1.annotate('', xy=(5, 2), xytext=(5, 0),
            arrowprops=dict(arrowstyle='->', lw=2))
ax1.text(5.5, 0.5, '$h_{t-1}$', fontsize=12, ha='center')

# 출력 화살표
ax1.annotate('', xy=(10, 5), xytext=(8, 5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax1.text(9.5, 5.5, '$h_t$', fontsize=12, ha='center')

# 다음으로 전달 (순환)
ax1.annotate('', xy=(5, 10), xytext=(5, 8),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax1.text(6, 9.5, '다음 시점으로', fontsize=10, color='red')

ax1.set_title('RNN 셀 내부 구조', fontsize=14)
ax1.axis('off')

# 오른쪽: 시퀀스 처리 과정 (Unrolled)
ax2 = axes[1]
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 10)

# 3개 시점 표현
for i, t in enumerate([0, 1, 2]):
    x_pos = 2 + i * 4

    # RNN 셀 박스
    rect = plt.Rectangle((x_pos, 3), 2.5, 4, fill=True,
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x_pos + 1.25, 5, f'RNN\nt={t}', ha='center', va='center', fontsize=10)

    # 입력
    ax2.annotate('', xy=(x_pos + 1.25, 3), xytext=(x_pos + 1.25, 1),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax2.text(x_pos + 1.25, 0.5, f'$x_{t}$', fontsize=11, ha='center')

    # 출력
    ax2.annotate('', xy=(x_pos + 1.25, 9), xytext=(x_pos + 1.25, 7),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax2.text(x_pos + 1.25, 9.5, f'$h_{t}$', fontsize=11, ha='center')

    # 은닉 상태 전달 (순환)
    if i < 2:
        ax2.annotate('', xy=(x_pos + 4, 5), xytext=(x_pos + 2.5, 5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax2.set_title('RNN 시퀀스 처리 (Unrolled View)', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig('rnn_structure.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### 1.4 기울기 소실 문제

긴 시퀀스에서 RNN이 겪는 핵심 문제를 이해합니다.

```python
def demonstrate_vanishing_gradient():
    """
    기울기 소실 문제 시뮬레이션

    역전파 시 기울기: dL/dh_0 = dL/dh_T * prod(dh_t/dh_{t-1})
    tanh의 도함수 최대값이 1이므로, 여러 번 곱하면 0에 수렴
    """
    print("=" * 60)
    print("기울기 소실 문제 시뮬레이션")
    print("=" * 60)

    # tanh의 도함수: 1 - tanh(x)^2, 최대값 1 (x=0일 때)
    # 실제로는 대부분 1보다 작은 값

    sequence_lengths = [10, 50, 100, 200]
    gradient_scales = [0.9, 0.7, 0.5]  # 시뮬레이션용 평균 기울기 크기

    print("\n시퀀스 길이별 기울기 크기 변화:")
    print("-" * 50)

    for scale in gradient_scales:
        print(f"\n평균 기울기 크기 = {scale}:")
        for seq_len in sequence_lengths:
            # 기울기가 seq_len번 곱해짐
            final_gradient = scale ** seq_len
            print(f"  시퀀스 길이 {seq_len:3d}: {final_gradient:.2e}")

    # 시각화
    plt.figure(figsize=(10, 5))

    seq_lens = list(range(1, 101))
    for scale in [0.9, 0.7, 0.5, 0.3]:
        gradients = [scale ** t for t in seq_lens]
        plt.plot(seq_lens, gradients, label=f'기울기 스케일={scale}', linewidth=2)

    plt.xlabel('시퀀스 길이 (시점 수)', fontsize=12)
    plt.ylabel('초기 시점에 도달하는 기울기 크기', fontsize=12)
    plt.title('기울기 소실 문제: 시퀀스가 길어질수록 초기 정보의 영향력 감소', fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vanishing_gradient.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n결론:")
    print("  - 긴 시퀀스에서 초기 정보의 기울기가 거의 0에 수렴")
    print("  - 초기 입력이 학습에 거의 영향을 주지 못함")
    print("  - 해결책: LSTM, GRU (게이트 메커니즘)")

demonstrate_vanishing_gradient()
```

**기울기 소실의 실제 영향**:

| 시퀀스 길이 | 문제 | 결과 |
|------------|------|------|
| 10 미만 | 경미 | 기본 RNN으로 충분 |
| 10~50 | 중간 | LSTM/GRU 권장 |
| 50 이상 | 심각 | LSTM/GRU 필수 |

---

## Part 2: LSTM과 GRU - 장기 의존성 해결 (8분)

### 2.1 LSTM 셀 구조

LSTM은 3개의 게이트와 셀 상태를 통해 장기 의존성 문제를 해결합니다.

```python
def lstm_cell_forward(x_t, h_prev, C_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    """
    LSTM 셀의 순전파 구현

    Parameters:
        x_t: 현재 입력 (input_dim,)
        h_prev: 이전 은닉 상태 (hidden_dim,)
        C_prev: 이전 셀 상태 (hidden_dim,)
        Wf, Wi, Wc, Wo: 각 게이트의 가중치
        bf, bi, bc, bo: 각 게이트의 편향

    Returns:
        h_t: 현재 은닉 상태
        C_t: 현재 셀 상태
    """
    # 입력과 이전 은닉 상태 결합
    concat = np.concatenate([h_prev, x_t])

    # Forget Gate: 이전 기억 중 잊을 비율 결정
    f_t = sigmoid(np.dot(Wf, concat) + bf)

    # Input Gate: 새 정보 중 저장할 비율 결정
    i_t = sigmoid(np.dot(Wi, concat) + bi)

    # 후보 셀 상태: 저장할 새 정보
    C_tilde = np.tanh(np.dot(Wc, concat) + bc)

    # 셀 상태 업데이트: 잊기 + 새 정보 추가
    C_t = f_t * C_prev + i_t * C_tilde

    # Output Gate: 출력할 비율 결정
    o_t = sigmoid(np.dot(Wo, concat) + bo)

    # 은닉 상태: 셀 상태에서 필요한 부분만 출력
    h_t = o_t * np.tanh(C_t)

    return h_t, C_t, {'f': f_t, 'i': i_t, 'o': o_t, 'C_tilde': C_tilde}

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# LSTM 동작 시뮬레이션
print("=" * 60)
print("LSTM 게이트 동작 시뮬레이션")
print("=" * 60)

np.random.seed(42)
hidden_dim = 4
input_dim = 2

# 가중치 초기화 (간소화)
concat_dim = hidden_dim + input_dim
Wf = np.random.randn(hidden_dim, concat_dim) * 0.1
Wi = np.random.randn(hidden_dim, concat_dim) * 0.1
Wc = np.random.randn(hidden_dim, concat_dim) * 0.1
Wo = np.random.randn(hidden_dim, concat_dim) * 0.1
bf = np.zeros(hidden_dim)
bi = np.zeros(hidden_dim)
bc = np.zeros(hidden_dim)
bo = np.zeros(hidden_dim)

# 시퀀스 처리
sequence = np.array([
    [0.5, 0.3],   # 정상 온도
    [0.6, 0.4],   # 약간 상승
    [0.9, 0.8],   # 급격한 상승 (이상 신호)
    [0.7, 0.5],   # 다시 하강
])

h_t = np.zeros(hidden_dim)
C_t = np.zeros(hidden_dim)

print("\n시점별 게이트 출력:")
print("-" * 60)

for t, x_t in enumerate(sequence):
    h_t, C_t, gates = lstm_cell_forward(x_t, h_t, C_t, Wf, Wi, Wc, Wo, bf, bi, bc, bo)
    print(f"t={t}: 입력={x_t}")
    print(f"      Forget={gates['f'].mean():.3f}, Input={gates['i'].mean():.3f}, Output={gates['o'].mean():.3f}")
    print(f"      h_t 평균={h_t.mean():.3f}, C_t 평균={C_t.mean():.3f}")
```

---

### 2.2 LSTM 게이트 역할 상세 설명

```python
def explain_lstm_gates():
    """
    LSTM 게이트의 역할을 제조 센서 예시로 설명
    """
    print("=" * 60)
    print("LSTM 게이트 역할 (제조 센서 예시)")
    print("=" * 60)

    gates_explanation = {
        'Forget Gate': {
            '역할': '이전 기억 중 잊을 것 결정',
            '출력 범위': '0 (완전 삭제) ~ 1 (완전 유지)',
            '제조 예시': '정상 작동 중 온도 데이터는 유지, 비정상 노이즈는 삭제'
        },
        'Input Gate': {
            '역할': '새 정보 중 저장할 것 결정',
            '출력 범위': '0 (무시) ~ 1 (강하게 저장)',
            '제조 예시': '급격한 온도 상승은 강하게 저장, 미세한 변동은 약하게'
        },
        'Output Gate': {
            '역할': '셀 상태에서 출력할 것 결정',
            '출력 범위': '0 (출력 안 함) ~ 1 (전부 출력)',
            '제조 예시': '현재 예측에 필요한 정보만 선별하여 출력'
        }
    }

    for gate_name, info in gates_explanation.items():
        print(f"\n{gate_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

explain_lstm_gates()
```

**LSTM 게이트 동작 비유**:

| 게이트 | 비유 | 제조 시나리오 |
|--------|------|--------------|
| Forget | "무엇을 잊을까?" | 3일 전 정상 온도는 잊어도 됨 |
| Input | "무엇을 기억할까?" | 오늘 온도 급등은 반드시 기억 |
| Output | "무엇을 말할까?" | 고장 예측에 필요한 정보만 출력 |

---

### 2.3 LSTM vs GRU 비교

```python
def compare_lstm_gru():
    """
    LSTM과 GRU의 구조 및 특성 비교
    """
    print("=" * 60)
    print("LSTM vs GRU 비교")
    print("=" * 60)

    comparison = {
        '게이트 수': {'LSTM': '3개 (Forget, Input, Output)', 'GRU': '2개 (Reset, Update)'},
        '상태 종류': {'LSTM': '셀 상태 + 은닉 상태', 'GRU': '은닉 상태만'},
        '파라미터 수': {'LSTM': '4 * d * (d + n + 1)', 'GRU': '3 * d * (d + n + 1)'},
        '학습 속도': {'LSTM': '상대적 느림', 'GRU': '상대적 빠름'},
        '장기 의존성': {'LSTM': '매우 강함', 'GRU': '강함'},
        '권장 용도': {'LSTM': '복잡한 장기 패턴', 'GRU': '일반적 시퀀스'}
    }

    print("\n항목별 비교:")
    print("-" * 60)
    print(f"{'항목':<15} {'LSTM':<25} {'GRU':<25}")
    print("-" * 60)

    for item, values in comparison.items():
        print(f"{item:<15} {values['LSTM']:<25} {values['GRU']:<25}")

    # 파라미터 수 계산 예시
    print("\n파라미터 수 계산 예시 (hidden_dim=64, input_dim=5):")
    d, n = 64, 5

    lstm_params = 4 * d * (d + n + 1)  # 4개 가중치 행렬
    gru_params = 3 * d * (d + n + 1)   # 3개 가중치 행렬

    print(f"  LSTM: 4 * 64 * (64 + 5 + 1) = {lstm_params:,}")
    print(f"  GRU:  3 * 64 * (64 + 5 + 1) = {gru_params:,}")
    print(f"  GRU는 LSTM 대비 {100 - gru_params/lstm_params*100:.0f}% 파라미터 감소")

compare_lstm_gru()
```

---

### 2.4 GRU 구현

```python
def gru_cell_forward(x_t, h_prev, Wr, Wz, Wh, br, bz, bh):
    """
    GRU 셀의 순전파 구현

    Parameters:
        x_t: 현재 입력
        h_prev: 이전 은닉 상태
        Wr, Wz, Wh: Reset, Update, Hidden 가중치
        br, bz, bh: 편향

    Returns:
        h_t: 현재 은닉 상태
    """
    concat = np.concatenate([h_prev, x_t])

    # Reset Gate: 이전 상태 얼마나 무시할지
    r_t = sigmoid(np.dot(Wr, concat) + br)

    # Update Gate: 새 정보와 이전 정보 비율
    z_t = sigmoid(np.dot(Wz, concat) + bz)

    # Reset된 은닉 상태로 후보 계산
    concat_reset = np.concatenate([r_t * h_prev, x_t])
    h_tilde = np.tanh(np.dot(Wh, concat_reset) + bh)

    # 최종 은닉 상태: 이전 상태와 후보의 가중 합
    h_t = (1 - z_t) * h_prev + z_t * h_tilde

    return h_t, {'r': r_t, 'z': z_t}

print("\nGRU 동작 원리:")
print("  Reset Gate (r): 이전 상태를 얼마나 '리셋'할지")
print("  Update Gate (z): 새 정보와 이전 정보의 '혼합 비율'")
print("  h_t = (1-z)*h_{t-1} + z*h_tilde")
print("       └── 이전 유지 ──┘ └── 새로 추가 ──┘")
```

---

## Part 3: 제조 센서 시계열 예측 실습 (8분)

### 3.1 환경 설정 및 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(f"TensorFlow 버전: {tf.__version__}")

# 데이터 로드
df = pd.read_csv('ai4i2020.csv')

print("\n데이터셋 정보:")
print(f"  샘플 수: {len(df):,}")
print(f"  컬럼: {list(df.columns)}")
```

---

### 3.2 데이터 탐색 및 전처리

```python
# 센서 컬럼 선택
sensor_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

print("센서 데이터 통계:")
print(df[sensor_cols].describe())

# 시각화: 센서 데이터 분포
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    ax = axes[i]

    # 시계열 플롯 (처음 500개만)
    ax.plot(df[col].values[:500], alpha=0.7)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel('시점')
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')  # 빈 서브플롯

plt.suptitle('센서 데이터 시계열', fontsize=14)
plt.tight_layout()
plt.savefig('sensor_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### 3.3 시퀀스 데이터 생성

```python
def create_sequences(data, seq_length):
    """
    시계열 데이터를 LSTM 입력 형태로 변환 (Sliding Window)

    Parameters:
        data: 원본 시계열 (n_samples, n_features)
        seq_length: 입력 시퀀스 길이 (과거 몇 시점을 볼지)

    Returns:
        X: (n_samples - seq_length, seq_length, n_features)
        y: (n_samples - seq_length, n_features)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 과거 seq_length 시점의 데이터
        X.append(data[i:i+seq_length])
        # 다음 시점 예측 대상
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[sensor_cols])

print(f"정규화 전 범위: {df[sensor_cols].min().min():.1f} ~ {df[sensor_cols].max().max():.1f}")
print(f"정규화 후 범위: {scaled_data.min():.3f} ~ {scaled_data.max():.3f}")

# 시퀀스 생성
SEQ_LENGTH = 10  # 과거 10개 시점 사용
n_features = len(sensor_cols)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

print(f"\n시퀀스 데이터 형태:")
print(f"  X shape: {X.shape}  (샘플 수, 시퀀스 길이, 특성 수)")
print(f"  y shape: {y.shape}  (샘플 수, 특성 수)")

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # 시계열이므로 셔플 안 함
)

print(f"\n학습 데이터: {X_train.shape[0]:,} 샘플")
print(f"테스트 데이터: {X_test.shape[0]:,} 샘플")
```

---

### 3.4 Sliding Window 시각화

```python
# Sliding Window 개념 시각화
fig, ax = plt.subplots(figsize=(12, 6))

# 원본 시계열 (처음 20개만)
sample_data = scaled_data[:20, 0]  # 첫 번째 센서
ax.plot(range(20), sample_data, 'b-o', label='원본 시계열', markersize=8)

# 윈도우 표시 (seq_length=5 예시)
window_size = 5
colors = ['red', 'green', 'purple']

for i, color in enumerate(colors):
    start = i * 2
    # 입력 윈도우
    ax.axvspan(start, start + window_size - 1, alpha=0.2, color=color)
    # 예측 대상
    ax.scatter(start + window_size, sample_data[start + window_size],
               s=200, color=color, marker='*', zorder=5)
    ax.annotate(f'예측{i+1}', (start + window_size, sample_data[start + window_size] + 0.05),
                fontsize=10, ha='center')

ax.set_xlabel('시점', fontsize=12)
ax.set_ylabel('정규화된 값', fontsize=12)
ax.set_title('Sliding Window 방식: 과거 N개 시점으로 다음 시점 예측', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sliding_window.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSliding Window 설명:")
print(f"  - 윈도우 크기: {SEQ_LENGTH} (과거 {SEQ_LENGTH}개 시점 사용)")
print(f"  - 예측 대상: 다음 1개 시점의 모든 센서값")
print(f"  - 총 생성 샘플: {len(X):,}개")
```

---

### 3.5 LSTM 모델 구성

```python
def create_lstm_model(seq_length, n_features):
    """
    제조 센서 예측용 LSTM 모델 생성

    구조:
    - LSTM(64, return_sequences=True): 첫 번째 LSTM 층
    - Dropout(0.2): 과적합 방지
    - LSTM(32): 두 번째 LSTM 층
    - Dense(16, relu): 은닉층
    - Dense(n_features): 출력층 (모든 센서 예측)

    Parameters:
        seq_length: 입력 시퀀스 길이
        n_features: 특성(센서) 수

    Returns:
        model: 컴파일된 Keras 모델
    """
    model = Sequential([
        # 첫 번째 LSTM 층: 모든 시점의 출력 반환
        LSTM(64, return_sequences=True,
             input_shape=(seq_length, n_features),
             name='lstm_1'),
        Dropout(0.2, name='dropout_1'),

        # 두 번째 LSTM 층: 마지막 시점만 출력
        LSTM(32, name='lstm_2'),

        # 완전 연결층
        Dense(16, activation='relu', name='dense_1'),
        Dense(n_features, name='output')  # 회귀: 활성화 함수 없음
    ])

    return model

# 모델 생성
model = create_lstm_model(SEQ_LENGTH, n_features)
model.summary()
```

**모델 구조 요약**:

| 층 | 출력 형태 | 파라미터 수 | 설명 |
|----|----------|------------|------|
| lstm_1 | (10, 64) | 17,920 | 모든 시점 출력 |
| dropout_1 | (10, 64) | 0 | 20% 드롭아웃 |
| lstm_2 | (32,) | 12,416 | 마지막 시점만 |
| dense_1 | (16,) | 528 | 특징 추출 |
| output | (5,) | 85 | 5개 센서 예측 |

---

### 3.6 모델 컴파일 및 학습

```python
# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error (회귀)
    metrics=['mae']  # Mean Absolute Error
)

print("컴파일 설정:")
print("  - Optimizer: Adam")
print("  - Loss: MSE (평균 제곱 오차)")
print("  - Metrics: MAE (평균 절대 오차)")

# 콜백 설정
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 모델 학습
print("\n모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n학습 완료!")
print(f"  실제 에포크 수: {len(history.history['loss'])}")
print(f"  최종 학습 MSE: {history.history['loss'][-1]:.6f}")
print(f"  최종 검증 MSE: {history.history['val_loss'][-1]:.6f}")
```

---

### 3.7 학습 곡선 시각화

```python
# 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선
axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_title('Loss Curve (MSE)', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE 곡선
axes[1].plot(history.history['mae'], label='Train', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation', linewidth=2)
axes[1].set_title('MAE Curve', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### 3.8 예측 및 평가

```python
# 테스트 데이터 예측
y_pred = model.predict(X_test, verbose=0)

# 성능 평가 (정규화된 스케일)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("테스트 데이터 성능 (정규화 스케일):")
print(f"  MSE: {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE: {mae:.6f}")

# 역정규화 (원래 스케일)
y_pred_orig = scaler.inverse_transform(y_pred)
y_test_orig = scaler.inverse_transform(y_test)

# 센서별 성능
print("\n센서별 MAE (원래 스케일):")
for i, col in enumerate(sensor_cols):
    sensor_mae = mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i])
    print(f"  {col}: {sensor_mae:.4f}")
```

---

### 3.9 예측 결과 시각화

```python
# 예측 vs 실제 비교 시각화
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# 처음 100개 시점만 표시
n_plot = 100

for i, col in enumerate(sensor_cols):
    ax = axes[i]
    ax.plot(y_test_orig[:n_plot, i], label='실제', alpha=0.8, linewidth=1.5)
    ax.plot(y_pred_orig[:n_plot, i], label='예측', alpha=0.8, linewidth=1.5)
    ax.set_title(col, fontsize=12)
    ax.set_xlabel('시점')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')

plt.suptitle('LSTM 센서값 예측 결과 (실제 vs 예측)', fontsize=14)
plt.tight_layout()
plt.savefig('lstm_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### 3.10 예측 오차 분석

```python
# 예측 오차 분석
errors = y_pred_orig - y_test_orig

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    ax = axes[i]
    ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{col}\n평균 오차: {errors[:, i].mean():.4f}', fontsize=11)
    ax.set_xlabel('예측 오차')
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')

plt.suptitle('센서별 예측 오차 분포', fontsize=14)
plt.tight_layout()
plt.savefig('prediction_error_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 오차 통계
print("\n예측 오차 통계:")
print("-" * 60)
print(f"{'센서':<30} {'평균 오차':>12} {'표준편차':>12}")
print("-" * 60)
for i, col in enumerate(sensor_cols):
    mean_err = errors[:, i].mean()
    std_err = errors[:, i].std()
    print(f"{col:<30} {mean_err:>12.4f} {std_err:>12.4f}")
```

---

### 3.11 GRU 모델 비교

```python
def create_gru_model(seq_length, n_features):
    """
    GRU 기반 센서 예측 모델
    LSTM 대비 파라미터가 적고 학습이 빠름
    """
    model = Sequential([
        GRU(64, return_sequences=True,
            input_shape=(seq_length, n_features),
            name='gru_1'),
        Dropout(0.2),
        GRU(32, name='gru_2'),
        Dense(16, activation='relu'),
        Dense(n_features)
    ])
    return model

# GRU 모델 생성 및 학습
model_gru = create_gru_model(SEQ_LENGTH, n_features)
model_gru.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nGRU 모델 학습...")
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

# 성능 비교
y_pred_gru = model_gru.predict(X_test, verbose=0)
mse_gru = mean_squared_error(y_test, y_pred_gru)

print("\nLSTM vs GRU 성능 비교:")
print(f"  LSTM MSE: {mse:.6f}")
print(f"  GRU MSE:  {mse_gru:.6f}")

# 파라미터 수 비교
lstm_params = model.count_params()
gru_params = model_gru.count_params()
print(f"\n파라미터 수:")
print(f"  LSTM: {lstm_params:,}")
print(f"  GRU:  {gru_params:,} ({100 - gru_params/lstm_params*100:.1f}% 감소)")
```

---

### 3.12 실시간 예측 시뮬레이션

```python
def simulate_realtime_prediction(model, scaler, initial_sequence, n_steps=10):
    """
    실시간 센서 예측 시뮬레이션

    Parameters:
        model: 학습된 LSTM/GRU 모델
        scaler: 정규화에 사용된 스케일러
        initial_sequence: 초기 시퀀스 (seq_length, n_features)
        n_steps: 예측할 미래 시점 수

    Returns:
        predictions: 예측된 센서값 시퀀스
    """
    current_seq = initial_sequence.copy()
    predictions = []

    print("실시간 예측 시뮬레이션:")
    print("-" * 50)

    for step in range(n_steps):
        # 현재 시퀀스로 다음 시점 예측
        pred = model.predict(current_seq.reshape(1, -1, current_seq.shape[-1]), verbose=0)
        predictions.append(pred[0])

        # 시퀀스 업데이트 (sliding window)
        current_seq = np.vstack([current_seq[1:], pred])

        # 원래 스케일로 변환하여 출력
        pred_orig = scaler.inverse_transform(pred)
        print(f"  Step {step+1}: 온도={pred_orig[0, 0]:.1f}K, "
              f"회전수={pred_orig[0, 2]:.0f}rpm, "
              f"토크={pred_orig[0, 3]:.1f}Nm")

    return np.array(predictions)

# 시뮬레이션 실행
initial_seq = X_test[0]  # 첫 번째 테스트 시퀀스 사용
future_predictions = simulate_realtime_prediction(model, scaler, initial_seq, n_steps=5)
```

---

### 3.13 이상 탐지 응용

```python
def detect_anomaly_with_prediction(model, scaler, X_test, y_test, threshold_percentile=95):
    """
    예측 오차 기반 이상 탐지

    Parameters:
        model: 학습된 모델
        scaler: 스케일러
        X_test: 테스트 입력
        y_test: 테스트 정답
        threshold_percentile: 이상 판단 임계값 백분위수

    Returns:
        anomaly_indices: 이상으로 판단된 시점 인덱스
    """
    # 예측
    y_pred = model.predict(X_test, verbose=0)

    # 예측 오차 계산 (유클리드 거리)
    errors = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))

    # 임계값 설정
    threshold = np.percentile(errors, threshold_percentile)

    # 이상 탐지
    anomaly_indices = np.where(errors > threshold)[0]

    print(f"\n이상 탐지 결과 (임계값 상위 {100-threshold_percentile}%):")
    print(f"  임계값: {threshold:.6f}")
    print(f"  이상 탐지 개수: {len(anomaly_indices)} / {len(errors)}")
    print(f"  이상 비율: {len(anomaly_indices)/len(errors)*100:.2f}%")

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.plot(errors, alpha=0.7, linewidth=0.5)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'임계값 ({threshold:.4f})')
    plt.scatter(anomaly_indices, errors[anomaly_indices], color='red', s=30, label='이상 탐지')
    plt.xlabel('시점')
    plt.ylabel('예측 오차')
    plt.title('LSTM 예측 오차 기반 이상 탐지')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.show()

    return anomaly_indices, errors

anomaly_idx, prediction_errors = detect_anomaly_with_prediction(
    model, scaler, X_test, y_test, threshold_percentile=95
)
```

---

### 3.14 모델 저장

```python
# 모델 저장
model.save('sensor_prediction_lstm.keras')
print("LSTM 모델 저장 완료: sensor_prediction_lstm.keras")

model_gru.save('sensor_prediction_gru.keras')
print("GRU 모델 저장 완료: sensor_prediction_gru.keras")

# 스케일러 저장
import joblib
joblib.dump(scaler, 'sensor_scaler.pkl')
print("스케일러 저장 완료: sensor_scaler.pkl")

# 단일 시퀀스 예측 함수
def predict_next_sensor_values(model, scaler, recent_data, seq_length=10):
    """
    최근 센서 데이터로 다음 시점 예측

    Parameters:
        model: 학습된 LSTM/GRU 모델
        scaler: 정규화 스케일러
        recent_data: 최근 seq_length 시점의 센서값 (seq_length, n_features)
        seq_length: 시퀀스 길이

    Returns:
        prediction: 다음 시점 예측값 (원래 스케일)
    """
    # 정규화
    scaled_data = scaler.transform(recent_data)

    # 예측
    scaled_pred = model.predict(scaled_data.reshape(1, seq_length, -1), verbose=0)

    # 역정규화
    prediction = scaler.inverse_transform(scaled_pred)

    return prediction[0]

print("\n예측 함수 사용 예시:")
print("  prediction = predict_next_sensor_values(model, scaler, recent_10_samples)")
```

---

## 핵심 정리

### 오늘 배운 내용

| 주제 | 핵심 내용 |
|------|----------|
| RNN 수식 | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ |
| 기울기 소실 | 긴 시퀀스에서 초기 정보 영향력 감소 |
| LSTM Forget | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ |
| LSTM Input | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| LSTM Output | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ |
| LSTM Cell | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| GRU Update | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$ |
| Sliding Window | 과거 N개 시점으로 다음 시점 예측 |

### 핵심 코드

```python
# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# LSTM 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(n_features)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, validation_split=0.2,
          callbacks=[EarlyStopping(patience=10)])
```

### 체크리스트

| 항목 | 확인 |
|------|------|
| RNN 순환 구조와 은닉 상태 역할 이해 | [ ] |
| 기울기 소실 문제와 해결 필요성 인식 | [ ] |
| LSTM 3가지 게이트 (Forget, Input, Output) 역할 설명 | [ ] |
| GRU와 LSTM 차이점 비교 | [ ] |
| create_sequences 함수로 시퀀스 데이터 준비 | [ ] |
| Keras LSTM 모델 구성 및 학습 | [ ] |
| 예측 결과 시각화 및 오차 분석 | [ ] |

---

## 다음 차시 예고

### [26차시] 딥러닝 모델 해석과 파인튜닝

- 학습된 모델의 예측 근거 해석
- SHAP/LIME을 활용한 변수 중요도 분석
- 모델 파인튜닝 및 성능 최적화 기법
- 전이 학습의 기초 개념

---

## [참고자료] 심화 학습

> 선택 학습 - 25분 초과 콘텐츠

### A. Bidirectional LSTM

양방향 LSTM은 시퀀스를 순방향과 역방향 모두 처리하여 더 풍부한 문맥을 학습합니다.

```python
from tensorflow.keras.layers import Bidirectional

model_bi = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(SEQ_LENGTH, n_features)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(16, activation='relu'),
    Dense(n_features)
])

print("Bidirectional LSTM 구조:")
model_bi.summary()
```

**Bidirectional LSTM 특징**:

| 항목 | 설명 |
|------|------|
| 구조 | 순방향 + 역방향 LSTM 병렬 처리 |
| 출력 | 양방향 출력 연결 (hidden_dim x 2) |
| 장점 | 과거와 미래 문맥 모두 활용 |
| 단점 | 실시간 예측에 부적합 (미래 정보 필요) |

---

### B. Attention 메커니즘

Attention은 시퀀스의 모든 시점에서 중요한 부분에 가중치를 부여합니다.

```python
from tensorflow.keras.layers import Attention, Concatenate

# 간단한 Self-Attention 적용
def create_lstm_with_attention(seq_length, n_features):
    """
    Attention 메커니즘이 적용된 LSTM 모델
    """
    from tensorflow.keras.layers import Input, Attention, Concatenate
    from tensorflow.keras.models import Model

    inputs = Input(shape=(seq_length, n_features))

    # LSTM 인코더
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    # Self-Attention
    attention_out = Attention()([lstm_out, lstm_out])

    # 최종 LSTM
    final_lstm = LSTM(32)(attention_out)

    # 출력
    dense = Dense(16, activation='relu')(final_lstm)
    outputs = Dense(n_features)(dense)

    model = Model(inputs, outputs)
    return model

print("\nAttention LSTM 적용 효과:")
print("  - 중요한 시점에 더 큰 가중치 부여")
print("  - 긴 시퀀스에서도 핵심 정보 유지")
print("  - 해석 가능성 향상 (attention weight 시각화)")
```

---

### C. 다변량 다중 출력 예측

여러 미래 시점을 동시에 예측하는 모델입니다.

```python
def create_multi_step_sequences(data, input_length, output_length):
    """
    다중 시점 예측을 위한 시퀀스 생성

    Parameters:
        data: 원본 시계열
        input_length: 입력 시퀀스 길이
        output_length: 출력(예측) 시퀀스 길이

    Returns:
        X: (n_samples, input_length, n_features)
        y: (n_samples, output_length, n_features)
    """
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i+input_length])
        y.append(data[i+input_length:i+input_length+output_length])
    return np.array(X), np.array(y)

# 다중 시점 예측 모델
def create_multi_step_model(input_length, output_length, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_length, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dense(output_length * n_features),
        # 출력을 (output_length, n_features) 형태로 reshape
        tf.keras.layers.Reshape((output_length, n_features))
    ])
    return model

print("\n다중 시점 예측 예시:")
print("  입력: 과거 10개 시점")
print("  출력: 미래 5개 시점 예측")
print("  활용: 중기 설비 상태 예측")
```

---

### D. 시계열 교차 검증

시계열 데이터의 시간 순서를 유지하는 교차 검증 방법입니다.

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(X, y, model_fn, n_splits=5):
    """
    시계열 교차 검증

    Parameters:
        X: 입력 데이터
        y: 타겟 데이터
        model_fn: 모델 생성 함수
        n_splits: 분할 수

    Returns:
        scores: 각 폴드의 성능 점수
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    print(f"\n시계열 {n_splits}-Fold 교차 검증:")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 모델 생성 및 학습
        model = model_fn()
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, verbose=0,
                 validation_data=(X_val, y_val),
                 callbacks=[EarlyStopping(patience=5)])

        # 평가
        mse = model.evaluate(X_val, y_val, verbose=0)
        scores.append(mse)
        print(f"  Fold {fold+1}: MSE = {mse:.6f}")

    print("-" * 50)
    print(f"  평균 MSE: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

    return scores

# 교차 검증 실행
# scores = time_series_cross_validation(
#     X, y,
#     lambda: create_lstm_model(SEQ_LENGTH, n_features),
#     n_splits=5
# )
```

---

### E. 학습률 스케줄링

학습 과정에서 학습률을 동적으로 조정하여 성능을 향상시킵니다.

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# 1. ReduceLROnPlateau: 검증 손실이 개선되지 않으면 학습률 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # 학습률을 50%로 감소
    patience=5,      # 5 에포크 동안 개선 없으면
    min_lr=1e-6,     # 최소 학습률
    verbose=1
)

# 2. Cosine Annealing (코사인 감쇠)
def cosine_annealing(epoch, lr, total_epochs=50):
    return lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

cosine_scheduler = LearningRateScheduler(
    lambda epoch: cosine_annealing(epoch, 0.001, 50)
)

print("\n학습률 스케줄링 기법:")
print("  1. ReduceLROnPlateau: 검증 손실 정체 시 학습률 감소")
print("  2. Cosine Annealing: 주기적으로 학습률 감소/증가")
print("  3. Warmup + Decay: 초기 워밍업 후 점진적 감소")
```

---

### F. 전체 파이프라인 코드

```python
# 완전한 LSTM 시계열 예측 파이프라인
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
df = pd.read_csv('ai4i2020.csv')
sensor_cols = ['Air temperature [K]', 'Process temperature [K]',
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# 2. 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[sensor_cols])

# 3. 시퀀스 생성
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 4. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 5. 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(sensor_cols))),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(len(sensor_cols))
])

# 6. 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# 8. 평가
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 MSE: {test_loss:.6f}, MAE: {test_mae:.6f}")

# 9. 저장
model.save('sensor_prediction_lstm.keras')
```
