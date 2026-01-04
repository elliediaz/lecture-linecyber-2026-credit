# [18차시] 딥러닝 입문: 신경망 기초 - 실습 코드

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("18차시: 딥러닝 입문 - 신경망 기초")
print("인공지능의 핵심, 신경망을 이해합니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 뉴런 계산
# ============================================================
print("=" * 50)
print("실습 1: 뉴런 계산")
print("=" * 50)

# 입력 (온도, 습도, 속도)
inputs = np.array([85, 50, 100])

# 가중치 (각 입력의 중요도)
weights = np.array([0.3, 0.2, 0.5])

# 편향
bias = 0.1

# 뉴런 계산: 입력 × 가중치 + 편향
z = np.dot(inputs, weights) + bias

print(f"입력: {inputs}")
print(f"가중치: {weights}")
print(f"편향: {bias}")
print(f"\n계산: 85×0.3 + 50×0.2 + 100×0.5 + 0.1")
print(f"     = 25.5 + 10 + 50 + 0.1")
print(f"     = {z}")
print()


# ============================================================
# 실습 2: 활성화 함수
# ============================================================
print("=" * 50)
print("실습 2: 활성화 함수")
print("=" * 50)

# ReLU 함수
def relu(x):
    """음수는 0, 양수는 그대로"""
    return np.maximum(0, x)

# Sigmoid 함수
def sigmoid(x):
    """0~1 범위로 변환"""
    return 1 / (1 + np.exp(-x))

# Tanh 함수
def tanh(x):
    """-1~1 범위로 변환"""
    return np.tanh(x)

# 테스트
test_values = np.array([-2, -1, 0, 1, 2, 5])
print("입력값에 대한 활성화 함수 출력:")
print(f"입력:    {test_values}")
print(f"ReLU:    {relu(test_values)}")
print(f"Sigmoid: {np.round(sigmoid(test_values), 3)}")
print(f"Tanh:    {np.round(tanh(test_values), 3)}")

# z값에 적용
print(f"\nz = {z}에 적용:")
print(f"ReLU({z:.1f}) = {relu(z):.1f}")
print(f"Sigmoid({z:.1f}) = {sigmoid(z):.6f}")
print()


# ============================================================
# 실습 3: 활성화 함수 시각화
# ============================================================
print("=" * 50)
print("실습 3: 활성화 함수 시각화")
print("=" * 50)

x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ReLU
axes[0].plot(x, relu(x), linewidth=2, color='blue')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('ReLU: max(0, x)', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1, 5)

# Sigmoid
axes[1].plot(x, sigmoid(x), linewidth=2, color='red')
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Sigmoid: 1/(1+e^(-x))', fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].grid(True, alpha=0.3)

# Tanh
axes[2].plot(x, tanh(x), linewidth=2, color='green')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_title('Tanh: tanh(x)', fontsize=12)
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('활성화_함수.png', dpi=100)
plt.close()
print("활성화 함수 그래프 저장: 활성화_함수.png")
print()


# ============================================================
# 실습 4: 단일 뉴런 클래스
# ============================================================
print("=" * 50)
print("실습 4: 단일 뉴런 클래스")
print("=" * 50)

class Neuron:
    """단일 뉴런 구현"""

    def __init__(self, n_inputs):
        # 가중치 초기화 (작은 랜덤값)
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, inputs):
        """순전파: 입력 → 출력"""
        # 가중합 + 편향
        z = np.dot(inputs, self.weights) + self.bias
        # 활성화 함수 적용
        return relu(z)

# 뉴런 생성 (입력 3개)
np.random.seed(42)
neuron = Neuron(3)

print(f"가중치: {neuron.weights}")
print(f"편향: {neuron.bias}")

# 순전파
inputs = np.array([85, 50, 100])
output = neuron.forward(inputs)
print(f"\n입력: {inputs}")
print(f"출력: {output:.4f}")
print()


# ============================================================
# 실습 5: 층 (Layer) 구현
# ============================================================
print("=" * 50)
print("실습 5: 층 (Layer) 구현")
print("=" * 50)

class Layer:
    """뉴런들의 집합 = 층"""

    def __init__(self, n_inputs, n_neurons):
        # 가중치 행렬: (입력 개수, 뉴런 개수)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        # 편향 벡터: (뉴런 개수)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        """순전파"""
        z = np.dot(inputs, self.weights) + self.biases
        return relu(z)

# 층 생성 (입력 3개 → 뉴런 4개)
np.random.seed(42)
layer = Layer(3, 4)

print(f"가중치 행렬 크기: {layer.weights.shape}")
print(f"편향 벡터 크기: {layer.biases.shape}")

# 순전파
inputs = np.array([85, 50, 100])
output = layer.forward(inputs)
print(f"\n입력: {inputs} (3개)")
print(f"출력: {output} (4개)")
print()


# ============================================================
# 실습 6: 2층 신경망
# ============================================================
print("=" * 50)
print("실습 6: 2층 신경망")
print("=" * 50)

np.random.seed(42)

# 2층 신경망 구성
# 입력(3) → 은닉층(4) → 출력(1)
layer1 = Layer(3, 4)  # 입력층 → 은닉층
layer2 = Layer(4, 1)  # 은닉층 → 출력층

# 순전파
inputs = np.array([85, 50, 100])

print("순전파 과정:")
print(f"1. 입력: {inputs}")

hidden = layer1.forward(inputs)
print(f"2. 은닉층 출력: {hidden}")

output = layer2.forward(hidden)
print(f"3. 최종 출력: {output}")
print()


# ============================================================
# 실습 7: 여러 샘플 처리 (배치)
# ============================================================
print("=" * 50)
print("실습 7: 여러 샘플 처리 (배치)")
print("=" * 50)

np.random.seed(42)

# 5개 샘플 데이터
X = np.array([
    [85, 50, 100],
    [90, 45, 110],
    [80, 55, 95],
    [88, 48, 105],
    [82, 52, 98]
])

print(f"입력 데이터 크기: {X.shape} (5개 샘플, 3개 특성)")

# 신경망 통과
layer1 = Layer(3, 4)
layer2 = Layer(4, 1)

hidden = layer1.forward(X)
output = layer2.forward(hidden)

print(f"은닉층 출력 크기: {hidden.shape}")
print(f"최종 출력 크기: {output.shape}")
print(f"\n각 샘플의 예측값:")
for i, pred in enumerate(output):
    print(f"  샘플 {i+1}: {pred[0]:.4f}")
print()


# ============================================================
# 실습 8: 손실 함수
# ============================================================
print("=" * 50)
print("실습 8: 손실 함수")
print("=" * 50)

def mse_loss(y_true, y_pred):
    """Mean Squared Error (평균 제곱 오차)"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """Mean Absolute Error (평균 절대 오차)"""
    return np.mean(np.abs(y_true - y_pred))

# 예측값과 실제값
y_pred = np.array([0.8, 0.3, 0.9, 0.2, 0.7])
y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

print("예측값:", y_pred)
print("실제값:", y_true)
print()

mse = mse_loss(y_true, y_pred)
mae = mae_loss(y_true, y_pred)

print(f"MSE (평균 제곱 오차): {mse:.4f}")
print(f"MAE (평균 절대 오차): {mae:.4f}")
print("\n★ 손실이 작아지도록 가중치를 조정하는 게 학습!")
print()


# ============================================================
# 실습 9: 완전한 MLP 클래스
# ============================================================
print("=" * 50)
print("실습 9: 완전한 MLP 클래스")
print("=" * 50)

class MLP:
    """다층 퍼셉트론 (Multi-Layer Perceptron)"""

    def __init__(self, layer_sizes):
        """
        layer_sizes: 각 층의 뉴런 수 리스트
        예: [3, 4, 2, 1] → 입력 3, 은닉층1 4, 은닉층2 2, 출력 1
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)

    def forward(self, X):
        """순전파"""
        output = X
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            print(f"  층 {i+1} 출력 크기: {output.shape}")
        return output

# MLP 생성: 입력(3) → 은닉(4) → 은닉(2) → 출력(1)
np.random.seed(42)
mlp = MLP([3, 4, 2, 1])

print(f"MLP 구조: 3 → 4 → 2 → 1")
print(f"층 개수: {len(mlp.layers)}")

# 순전파
X = np.array([[85, 50, 100]])
print(f"\n입력 크기: {X.shape}")
print("순전파 과정:")
output = mlp.forward(X)
print(f"최종 출력: {output}")
print()


# ============================================================
# 실습 10: 딥러닝 vs 머신러닝 비교
# ============================================================
print("=" * 50)
print("실습 10: 딥러닝 vs 머신러닝 비교")
print("=" * 50)

print("""
┌────────────────────────────────────────────────────────┐
│           딥러닝 vs 머신러닝 비교                        │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┬──────────────┬──────────────┐        │
│  │    항목      │   머신러닝    │   딥러닝     │        │
│  ├──────────────┼──────────────┼──────────────┤        │
│  │ 특성 엔지니어링 │ 사람이 설계  │ 자동 학습   │        │
│  │ 데이터 양     │ 적어도 OK    │ 많이 필요   │        │
│  │ 연산 장치     │ CPU 가능     │ GPU 권장    │        │
│  │ 해석 가능성   │ 상대적 쉬움  │ 블랙박스    │        │
│  │ 적합한 데이터 │ 테이블 데이터 │ 이미지/텍스트 │       │
│  └──────────────┴──────────────┴──────────────┘        │
│                                                         │
│  ★ 제조 현장에서는?                                     │
│    - 센서 데이터, 생산량 → RandomForest (ML)            │
│    - 외관 검사, 결함 탐지 → CNN (딥러닝)                │
│                                                         │
└────────────────────────────────────────────────────────┘
""")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              딥러닝 기초 핵심 정리                      │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 뉴런 (Neuron)                                       │
│     y = f(w₁x₁ + w₂x₂ + ... + b)                      │
│     입력 × 가중치 + 편향 → 활성화 함수                  │
│                                                        │
│  ▶ 활성화 함수                                         │
│     - ReLU: max(0, x) ← 가장 많이 사용!                │
│     - Sigmoid: 1/(1+e^(-x)), 0~1 범위                  │
│     - 비선형성 추가 → 복잡한 패턴 학습                  │
│                                                        │
│  ▶ 층 (Layer)                                          │
│     - 입력층: 데이터 받음                               │
│     - 은닉층: 특성 학습                                 │
│     - 출력층: 결과 출력                                 │
│                                                        │
│  ▶ 심층 신경망 (DNN)                                   │
│     - 은닉층 2개 이상 = "Deep" Learning                │
│                                                        │
│  ▶ 학습 과정                                           │
│     순전파 → 손실 계산 → 역전파 → 가중치 업데이트       │
│     ★ 손실이 작아지도록 가중치 조정!                    │
│                                                        │
│  ▶ 실무 적용                                           │
│     테이블 데이터 → ML (RandomForest)                  │
│     이미지 데이터 → 딥러닝 (CNN)                       │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 딥러닝 실습 - MLP로 품질 예측
""")

print("=" * 60)
print("18차시 실습 완료!")
print("인공지능의 핵심, 신경망을 이해했습니다!")
print("=" * 60)
