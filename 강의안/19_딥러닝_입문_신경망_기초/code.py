"""
[19차시] 딥러닝 입문: 신경망 기초 - 실습 코드
학습목표: 신경망 기본 개념 이해, 활성화 함수, 순전파 구현
"""

import numpy as np
import matplotlib.pyplot as plt

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 활성화 함수
# ============================================================
print("=" * 50)
print("1. 활성화 함수 (Activation Functions)")
print("=" * 50)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# 시각화
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# ReLU
axes[0].plot(x, relu(x), linewidth=2)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('ReLU: max(0, x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)

# Sigmoid
axes[1].plot(x, sigmoid(x), linewidth=2, color='orange')
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Sigmoid: 1/(1+e^-x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True, alpha=0.3)

# Tanh
axes[2].plot(x, tanh(x), linewidth=2, color='green')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_title('Tanh')
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ activation_functions.png 저장됨")

print("\n활성화 함수 특징:")
print("  - ReLU: 음수 → 0, 양수 → 그대로. 가장 많이 사용")
print("  - Sigmoid: 0~1 사이. 확률 출력에 적합")
print("  - Tanh: -1~1 사이. 중심이 0")

# ============================================================
# 2. 단일 뉴런 시뮬레이션
# ============================================================
print("\n" + "=" * 50)
print("2. 단일 뉴런 시뮬레이션")
print("=" * 50)

# 입력 (제조 데이터: 온도, 습도, 속도)
inputs = np.array([85, 50, 100])
print(f"입력 (x): {inputs}")
print(f"  온도=85, 습도=50, 속도=100")

# 가중치 (학습되는 값)
weights = np.array([0.02, 0.01, 0.05])
print(f"\n가중치 (w): {weights}")
print(f"  온도 가중치=0.02, 습도 가중치=0.01, 속도 가중치=0.05")

# 편향
bias = -6.0
print(f"\n편향 (b): {bias}")

# 계산
z = np.dot(inputs, weights) + bias
print(f"\n▶ 선형 조합: z = Σ(x_i × w_i) + b")
print(f"  z = {inputs[0]}×{weights[0]} + {inputs[1]}×{weights[1]} + {inputs[2]}×{weights[2]} + {bias}")
print(f"  z = {inputs[0]*weights[0]:.1f} + {inputs[1]*weights[1]:.1f} + {inputs[2]*weights[2]:.1f} + {bias}")
print(f"  z = {z:.2f}")

# 활성화 함수 적용
output_relu = relu(z)
output_sigmoid = sigmoid(z)

print(f"\n▶ 활성화 함수 적용:")
print(f"  ReLU(z) = {output_relu:.2f}")
print(f"  Sigmoid(z) = {output_sigmoid:.4f}")

# ============================================================
# 3. 신경망 층 시뮬레이션
# ============================================================
print("\n" + "=" * 50)
print("3. 간단한 신경망 (2층)")
print("=" * 50)

np.random.seed(42)

# 입력층: 3개 뉴런
X = np.array([[85, 50, 100],
              [90, 60, 110],
              [80, 45, 95]])
print(f"입력 (3개 샘플, 3개 특성):\n{X}")

# 은닉층: 4개 뉴런
W1 = np.random.randn(3, 4) * 0.01  # 입력→은닉
b1 = np.zeros(4)

print(f"\n은닉층 가중치 W1 shape: {W1.shape} (3 입력 → 4 은닉)")

# 출력층: 1개 뉴런
W2 = np.random.randn(4, 1) * 0.01  # 은닉→출력
b2 = np.zeros(1)

print(f"출력층 가중치 W2 shape: {W2.shape} (4 은닉 → 1 출력)")

# 순전파 (Forward Pass)
print("\n▶ 순전파 (Forward Pass):")

# 은닉층
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)
print(f"  1. 입력 → 은닉층: Z1 shape = {Z1.shape}")
print(f"     ReLU 적용 후 A1 shape = {A1.shape}")

# 출력층
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
print(f"  2. 은닉층 → 출력층: Z2 shape = {Z2.shape}")
print(f"     Sigmoid 적용 후 A2 shape = {A2.shape}")

print(f"\n최종 출력 (확률):\n{A2.flatten()}")

# ============================================================
# 4. 손실 함수
# ============================================================
print("\n" + "=" * 50)
print("4. 손실 함수 (Loss Function)")
print("=" * 50)

# 예측값과 실제값
y_pred = np.array([0.8, 0.6, 0.9])
y_true = np.array([1.0, 0.0, 1.0])

print(f"예측값: {y_pred}")
print(f"실제값: {y_true}")

# MSE (회귀용)
mse = np.mean((y_true - y_pred) ** 2)
print(f"\n▶ MSE (Mean Squared Error):")
print(f"  = (1/n) × Σ(y_true - y_pred)²")
print(f"  = {mse:.4f}")

# Binary Cross Entropy (이진 분류용)
epsilon = 1e-15  # 0으로 나누기 방지
bce = -np.mean(y_true * np.log(y_pred + epsilon) +
               (1 - y_true) * np.log(1 - y_pred + epsilon))
print(f"\n▶ Binary Cross Entropy (분류용):")
print(f"  = -mean(y×log(p) + (1-y)×log(1-p))")
print(f"  = {bce:.4f}")

# ============================================================
# 5. 경사하강법 시각화
# ============================================================
print("\n" + "=" * 50)
print("5. 경사하강법 시각화")
print("=" * 50)

# 손실 함수 (예시: 2차 함수)
def loss_function(w):
    return (w - 3) ** 2 + 1

def loss_gradient(w):
    return 2 * (w - 3)

# 경사하강법 시뮬레이션
w = 0.0  # 초기 가중치
learning_rate = 0.1
history = [w]

for i in range(20):
    grad = loss_gradient(w)
    w = w - learning_rate * grad
    history.append(w)

print(f"초기 가중치: {history[0]:.2f}")
print(f"최종 가중치: {history[-1]:.2f} (최적값: 3.0)")
print(f"학습률: {learning_rate}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실 함수와 경로
w_range = np.linspace(-1, 6, 100)
axes[0].plot(w_range, loss_function(w_range), linewidth=2)
axes[0].scatter(history, [loss_function(h) for h in history],
                c=range(len(history)), cmap='Reds', s=50, zorder=5)
axes[0].set_xlabel('Weight (w)')
axes[0].set_ylabel('Loss')
axes[0].set_title('Gradient Descent Path')
axes[0].grid(True, alpha=0.3)

# 가중치 변화
axes[1].plot(history, 'o-', linewidth=2)
axes[1].axhline(y=3, color='r', linestyle='--', label='Optimal (w=3)')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Weight (w)')
axes[1].set_title('Weight Update Over Iterations')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n▶ gradient_descent.png 저장됨")

# ============================================================
# 6. 딥러닝 vs 머신러닝 비교
# ============================================================
print("\n" + "=" * 50)
print("6. 딥러닝 vs 머신러닝")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────┐
│              딥러닝 vs 머신러닝 비교                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  구분        │  머신러닝          │  딥러닝              │
│  ───────────────────────────────────────────────────    │
│  데이터 양   │  적어도 가능       │  많을수록 좋음       │
│  특성 추출   │  직접 설계         │  자동 학습          │
│  해석 가능성 │  비교적 쉬움       │  블랙박스           │
│  학습 시간   │  빠름              │  오래 걸림          │
│  적합한 문제 │  테이블 데이터     │  이미지, 텍스트     │
│  주요 모델   │  RF, XGBoost       │  MLP, CNN, RNN      │
│                                                          │
└─────────────────────────────────────────────────────────┘
""")

# ============================================================
# 7. 핵심 요약
# ============================================================
print("=" * 50)
print("7. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                 신경망 기초 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 뉴런의 계산                                         │
│     z = Σ(x_i × w_i) + b     (선형 조합)               │
│     a = f(z)                  (활성화 함수)            │
│                                                        │
│  ▶ 활성화 함수                                         │
│     - ReLU: max(0, x) → 은닉층에서 사용               │
│     - Sigmoid: 1/(1+e^-x) → 이진 분류 출력            │
│                                                        │
│  ▶ 학습 과정                                           │
│     1. 순전파: 입력 → 예측                             │
│     2. 손실 계산: 예측 vs 실제                         │
│     3. 역전파: 가중치 조정 방향 계산                    │
│     4. 업데이트: w = w - lr × gradient                │
│                                                        │
│  ▶ 경사하강법: 손실이 줄어드는 방향으로 가중치 조정    │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: Keras로 MLP 구현하기
""")
