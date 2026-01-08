"""
[20차시] 딥러닝 입문: 신경망 기초 - 실습 코드

학습 목표:
1. 인공 뉴런의 구조와 동작 원리를 이해한다
2. 신경망의 층 구조와 파라미터를 계산한다
3. NumPy로 순전파와 역전파를 구현한다

실습 환경: Python 3.8+, NumPy, Matplotlib
데이터셋: sklearn digits (8x8 손글씨 숫자)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[20차시] 딥러닝 입문: 신경망 기초")
print("=" * 60)

# ============================================================
# 1. 활성화 함수 구현 및 시각화
# ============================================================
print("\n" + "=" * 60)
print("1. 활성화 함수 구현 및 시각화")
print("=" * 60)

def sigmoid(z):
    """시그모이드 함수: 출력 범위 0~1"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """시그모이드 미분: 역전파에서 사용"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU 함수: 음수는 0, 양수는 그대로"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU 미분: 양수면 1, 음수면 0"""
    return (z > 0).astype(float)

def tanh(z):
    """Tanh 함수: 출력 범위 -1~1"""
    return np.tanh(z)

def tanh_derivative(z):
    """Tanh 미분"""
    return 1 - np.tanh(z)**2

def softmax(z):
    """Softmax 함수: 다중 클래스 분류용"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 안정성
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 활성화 함수 시각화
print("\n활성화 함수 시각화:")
z = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sigmoid
axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
axes[0, 0].plot(z, sigmoid_derivative(z), 'b--', linewidth=1.5, label='Sigmoid 미분')
axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].set_title('Sigmoid: 0~1 출력, 이진 분류 출력층')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('f(z)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ReLU
axes[0, 1].plot(z, relu(z), 'r-', linewidth=2, label='ReLU')
axes[0, 1].plot(z, relu_derivative(z), 'r--', linewidth=1.5, label='ReLU 미분')
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].set_title('ReLU: max(0, z), 은닉층 표준')
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('f(z)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Tanh
axes[1, 0].plot(z, tanh(z), 'g-', linewidth=2, label='Tanh')
axes[1, 0].plot(z, tanh_derivative(z), 'g--', linewidth=1.5, label='Tanh 미분')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Tanh: -1~1 출력, 0 중심')
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('f(z)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 비교
axes[1, 1].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
axes[1, 1].plot(z, relu(z)/5, 'r-', linewidth=2, label='ReLU/5')
axes[1, 1].plot(z, tanh(z), 'g-', linewidth=2, label='Tanh')
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_title('활성화 함수 비교')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('f(z)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 'activation_functions.png' 저장됨")

# ============================================================
# 2. 실제 데이터 로드: sklearn digits
# ============================================================
print("\n" + "=" * 60)
print("2. 실제 데이터 로드: sklearn digits")
print("=" * 60)

# digits 데이터 로드 (8x8 손글씨 숫자)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"\n[Digits 데이터셋]")
print(f"  샘플 수: {X_digits.shape[0]}")
print(f"  특성 수: {X_digits.shape[1]} (8x8 픽셀)")
print(f"  클래스: {np.unique(y_digits)} (0-9 숫자)")
print(f"  데이터 범위: {X_digits.min():.1f} ~ {X_digits.max():.1f}")

# 샘플 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y_digits[i]}')
    ax.axis('off')
plt.suptitle('Digits 샘플 이미지 (8x8 픽셀)', fontsize=14)
plt.tight_layout()
plt.savefig('digits_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 'digits_samples.png' 저장됨")

# ============================================================
# 3. 단일 뉴런 (퍼셉트론) 구현
# ============================================================
print("\n" + "=" * 60)
print("3. 단일 뉴런 (퍼셉트론) 구현")
print("=" * 60)

class Perceptron:
    """
    단일 뉴런 (퍼셉트론)
    y = f(Sigma wi*xi + b)
    """
    def __init__(self, n_inputs, activation='sigmoid'):
        """
        Args:
            n_inputs: 입력 개수
            activation: 활성화 함수 ('sigmoid', 'relu', 'tanh')
        """
        # 가중치 초기화 (Xavier 초기화)
        self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        self.bias = 0.0

        # 활성화 함수 설정
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'relu':
            self.activation = relu
        elif activation == 'tanh':
            self.activation = tanh
        else:
            self.activation = lambda x: x  # linear

    def forward(self, X):
        """순전파: z = Xw + b, y = f(z)"""
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def __repr__(self):
        return f"Perceptron(weights_shape={self.weights.shape}, bias={self.bias:.4f})"

# 퍼셉트론 예시 - digits 데이터 첫 5개
print("\n단일 뉴런 예시 (digits 데이터):")
neuron = Perceptron(n_inputs=64, activation='sigmoid')  # 64 픽셀 입력
print(f"  가중치 shape: {neuron.weights.shape}")
print(f"  편향: {neuron.bias}")

# 입력 데이터 (정규화)
X_sample = X_digits[:5] / 16.0  # 0-1 범위로 정규화
output = neuron.forward(X_sample)
print(f"\n  입력 shape: {X_sample.shape}")
print(f"  출력: {output}")

# ============================================================
# 4. 파라미터 수 계산
# ============================================================
print("\n" + "=" * 60)
print("4. 파라미터 수 계산")
print("=" * 60)

def count_parameters(layers):
    """
    신경망 파라미터 수 계산

    Args:
        layers: 각 층의 노드 수 리스트 [입력, 은닉1, 은닉2, ..., 출력]

    Returns:
        총 파라미터 수, 층별 파라미터 수
    """
    total_params = 0
    layer_params = []

    for i in range(len(layers) - 1):
        # 가중치: 이전 층 x 현재 층
        weights = layers[i] * layers[i+1]
        # 편향: 현재 층
        biases = layers[i+1]
        # 층별 총 파라미터
        params = weights + biases

        layer_params.append({
            'layer': f'층 {i+1} ({layers[i]}->{layers[i+1]})',
            'weights': weights,
            'biases': biases,
            'total': params
        })
        total_params += params

    return total_params, layer_params

# 예시 1: digits 분류 (간단한 구조)
print("\n예시 1: 입력(64) -> 은닉(32) -> 출력(10)")
layers1 = [64, 32, 10]
total1, details1 = count_parameters(layers1)
for d in details1:
    print(f"  {d['layer']}: 가중치 {d['weights']}, 편향 {d['biases']}, 합계 {d['total']}")
print(f"  총 파라미터: {total1}개")

# 예시 2: 더 깊은 구조
print("\n예시 2: 입력(64) -> 은닉(128) -> 은닉(64) -> 출력(10)")
layers2 = [64, 128, 64, 10]
total2, details2 = count_parameters(layers2)
for d in details2:
    print(f"  {d['layer']}: 가중치 {d['weights']}, 편향 {d['biases']}, 합계 {d['total']}")
print(f"  총 파라미터: {total2}개")

# ============================================================
# 5. 2층 신경망 (MLP) 구현
# ============================================================
print("\n" + "=" * 60)
print("5. 2층 신경망 (MLP) 구현")
print("=" * 60)

class SimpleNeuralNetwork:
    """
    2층 신경망 (입력 -> 은닉 -> 출력)
    순전파, 역전파, 학습 구현
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Args:
            input_size: 입력층 노드 수
            hidden_size: 은닉층 노드 수
            output_size: 출력층 노드 수
            learning_rate: 학습률
        """
        self.lr = learning_rate

        # 가중치 초기화 (He 초기화 for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 학습 기록
        self.loss_history = []

    def forward(self, X):
        """순전파"""
        # 은닉층: z1 = XW1 + b1, a1 = ReLU(z1)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # 출력층: z2 = a1W2 + b2, a2 = Sigmoid(z2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """Binary Cross-Entropy 손실"""
        epsilon = 1e-8  # log(0) 방지
        loss = -np.mean(
            y_true * np.log(y_pred + epsilon) +
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss

    def backward(self, X, y_true):
        """역전파: 기울기 계산"""
        m = X.shape[0]  # 샘플 수

        # 출력층 기울기
        dz2 = self.a2 - y_true  # Sigmoid + BCE 미분 결과
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # 은닉층 기울기 (연쇄 법칙)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2):
        """경사하강법으로 가중치 업데이트"""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000, verbose=True):
        """학습 루프"""
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)

            # 손실 계산
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # 역전파
            dW1, db1, dW2, db2 = self.backward(X, y)

            # 가중치 업데이트
            self.update(dW1, db1, dW2, db2)

            # 진행 상황 출력
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                accuracy = self.evaluate(X, y)
                print(f"  Epoch {epoch+1:4d}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")

        return self.loss_history

    def predict(self, X, threshold=0.5):
        """예측 (이진 분류)"""
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)

    def evaluate(self, X, y):
        """정확도 계산"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# ============================================================
# 6. XOR 문제 학습
# ============================================================
print("\n" + "=" * 60)
print("6. XOR 문제 학습")
print("=" * 60)

print("\nXOR 문제:")
print("  (0, 0) -> 0")
print("  (0, 1) -> 1")
print("  (1, 0) -> 1")
print("  (1, 1) -> 0")
print("\n-> 선형 분리 불가능! 신경망으로 해결")

# XOR 데이터
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# 신경망 생성 및 학습
np.random.seed(42)
nn_xor = SimpleNeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.5
)

print("\n학습 시작...")
print(f"  구조: 입력(2) -> 은닉(4) -> 출력(1)")
print(f"  학습률: 0.5")
print(f"  에포크: 2000")
print()

loss_history = nn_xor.train(X_xor, y_xor, epochs=2000, verbose=True)

# 결과 확인
print("\n학습 결과:")
y_pred_prob = nn_xor.forward(X_xor)
y_pred = nn_xor.predict(X_xor)

for i in range(len(X_xor)):
    print(f"  입력: {X_xor[i]} -> 예측: {y_pred_prob[i, 0]:.4f} -> 분류: {y_pred[i, 0]} (정답: {y_xor[i, 0]})")

print(f"\n최종 정확도: {nn_xor.evaluate(X_xor, y_xor):.2%}")

# 손실 곡선 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('XOR 학습 손실 곡선')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 결정 경계 시각화
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
Z = nn_xor.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
plt.colorbar(label='Probability')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.ravel(),
            cmap='RdYlBu', s=200, edgecolors='black', linewidths=2)
plt.title('XOR 결정 경계')
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.savefig('xor_learning.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  -> 'xor_learning.png' 저장됨")

# ============================================================
# 7. Digits 이진 분류 (0 vs 나머지)
# ============================================================
print("\n" + "=" * 60)
print("7. Digits 이진 분류 (숫자 0 vs 나머지)")
print("=" * 60)

# 이진 분류 데이터 준비
y_binary = (y_digits == 0).astype(int).reshape(-1, 1)  # 0이면 1, 아니면 0

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\n[데이터 분할]")
print(f"  학습: {len(X_train)}개, 테스트: {len(X_test)}개")
print(f"  학습 클래스 비율: {y_train.mean():.2%} (숫자 0)")
print(f"  테스트 클래스 비율: {y_test.mean():.2%} (숫자 0)")

# 신경망 생성 및 학습
np.random.seed(42)
nn_digits = SimpleNeuralNetwork(
    input_size=64,
    hidden_size=32,
    output_size=1,
    learning_rate=0.1
)

print("\n학습 시작...")
print(f"  구조: 입력(64) -> 은닉(32) -> 출력(1)")
loss_history_digits = nn_digits.train(X_train, y_train, epochs=500, verbose=True)

# 평가
train_acc = nn_digits.evaluate(X_train, y_train)
test_acc = nn_digits.evaluate(X_test, y_test)
print(f"\n학습 정확도: {train_acc:.2%}")
print(f"테스트 정확도: {test_acc:.2%}")

# ============================================================
# 8. 학습률 비교 실험
# ============================================================
print("\n" + "=" * 60)
print("8. 학습률 비교 실험")
print("=" * 60)

learning_rates = [0.01, 0.1, 0.5, 1.0]
loss_histories = {}

print("\n학습률별 XOR 학습:")
for lr in learning_rates:
    np.random.seed(42)
    nn = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=4,
        output_size=1,
        learning_rate=lr
    )
    loss_histories[lr] = nn.train(X_xor, y_xor, epochs=500, verbose=False)
    final_acc = nn.evaluate(X_xor, y_xor)
    print(f"  lr={lr}: 최종 손실={loss_histories[lr][-1]:.4f}, 정확도={final_acc:.2%}")

# 학습률 비교 시각화
plt.figure(figsize=(10, 5))
for lr in learning_rates:
    plt.plot(loss_histories[lr], label=f'lr={lr}')
plt.title('학습률에 따른 손실 변화')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  -> 'learning_rate_comparison.png' 저장됨")

# ============================================================
# 9. 은닉층 크기 비교 실험
# ============================================================
print("\n" + "=" * 60)
print("9. 은닉층 크기 비교 실험")
print("=" * 60)

hidden_sizes = [2, 4, 8, 16]
results = {}

print("\n은닉층 크기별 XOR 학습:")
for hs in hidden_sizes:
    np.random.seed(42)
    nn = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=hs,
        output_size=1,
        learning_rate=0.5
    )
    loss_hist = nn.train(X_xor, y_xor, epochs=1000, verbose=False)
    final_acc = nn.evaluate(X_xor, y_xor)
    total_params, _ = count_parameters([2, hs, 1])
    results[hs] = {
        'loss': loss_hist[-1],
        'accuracy': final_acc,
        'params': total_params
    }
    print(f"  은닉={hs}: 파라미터={total_params}, 손실={loss_hist[-1]:.4f}, 정확도={final_acc:.2%}")

# ============================================================
# 10. 손실 함수 구현
# ============================================================
print("\n" + "=" * 60)
print("10. 손실 함수 구현")
print("=" * 60)

def mse_loss(y_true, y_pred):
    """Mean Squared Error (회귀용)"""
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-8):
    """Binary Cross-Entropy (이진 분류용)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-8):
    """Categorical Cross-Entropy (다중 분류용)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 손실 함수 예시
print("\n손실 함수 예시:")

# 회귀
y_true_reg = np.array([100, 150, 200])
y_pred_reg = np.array([105, 145, 210])
print(f"\n회귀 (MSE):")
print(f"  실제: {y_true_reg}")
print(f"  예측: {y_pred_reg}")
print(f"  MSE: {mse_loss(y_true_reg, y_pred_reg):.2f}")

# 이진 분류
y_true_bin = np.array([1, 0, 1, 1])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.6])
print(f"\n이진 분류 (BCE):")
print(f"  실제: {y_true_bin}")
print(f"  예측: {y_pred_bin}")
print(f"  BCE: {binary_crossentropy_loss(y_true_bin, y_pred_bin):.4f}")

# 다중 분류
y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one-hot
y_pred_cat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
print(f"\n다중 분류 (CCE):")
print(f"  실제: {y_true_cat}")
print(f"  예측: {y_pred_cat}")
print(f"  CCE: {categorical_crossentropy_loss(y_true_cat, y_pred_cat):.4f}")

# ============================================================
# 11. 경사하강법 시각화
# ============================================================
print("\n" + "=" * 60)
print("11. 경사하강법 시각화")
print("=" * 60)

def simple_loss(w):
    """간단한 2D 손실 함수: L = w1^2 + w2^2"""
    return w[0]**2 + w[1]**2

def simple_gradient(w):
    """손실 함수의 기울기"""
    return np.array([2*w[0], 2*w[1]])

# 경사하강법 실행
w = np.array([4.0, 3.0])  # 시작점
lr = 0.3
path = [w.copy()]

for _ in range(20):
    grad = simple_gradient(w)
    w = w - lr * grad
    path.append(w.copy())

path = np.array(path)

# 시각화
fig, ax = plt.subplots(figsize=(8, 6))

# 손실 함수 등고선
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=8, linewidth=2, label='경사하강법 경로')
ax.plot(path[0, 0], path[0, 1], 'go', markersize=15, label='시작점')
ax.plot(0, 0, 'b*', markersize=20, label='최적점')

ax.set_title('경사하강법 시각화')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 'gradient_descent.png' 저장됨")

# ============================================================
# 12. 가중치 초기화 방법
# ============================================================
print("\n" + "=" * 60)
print("12. 가중치 초기화 방법")
print("=" * 60)

def initialize_weights(input_size, output_size, method='xavier'):
    """
    가중치 초기화 방법
    """
    if method == 'zero':
        # 절대 사용 금지! 모든 뉴런이 같은 값을 학습
        return np.zeros((input_size, output_size))

    elif method == 'random':
        # 간단한 랜덤 초기화
        return np.random.randn(input_size, output_size) * 0.01

    elif method == 'xavier':
        # Xavier (Glorot) 초기화: Sigmoid, Tanh에 적합
        return np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

    elif method == 'he':
        # He 초기화: ReLU에 적합
        return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

    else:
        raise ValueError(f"Unknown method: {method}")

# 초기화 방법 비교
methods = ['zero', 'random', 'xavier', 'he']
print("\n가중치 초기화 비교 (64->32 층):")

for method in methods:
    W = initialize_weights(64, 32, method)
    print(f"\n{method:8s}: 평균={W.mean():.4f}, 표준편차={W.std():.4f}")
    print(f"         최소={W.min():.4f}, 최대={W.max():.4f}")

# ============================================================
# 13. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("13. 핵심 정리")
print("=" * 60)

print("""
[20차시 핵심 정리]

1. 인공 뉴런 (퍼셉트론)
   - 구조: z = Sigma(wi*xi) + b, y = f(z)
   - 가중치(W): 입력의 중요도
   - 편향(b): 활성화 기준점
   - 활성화 함수: 비선형성 추가

2. 활성화 함수
   - Sigmoid: 0~1, 이진 분류 출력층
   - ReLU: max(0, z), 은닉층 표준
   - Softmax: 합=1, 다중 분류 출력층

3. 신경망 구조
   - 입력층 -> 은닉층(들) -> 출력층
   - 파라미터 수 = (이전 노드 x 현재 노드) + 편향

4. 순전파/역전파
   - 순전파: 입력 -> 출력 계산
   - 손실: 예측과 정답의 차이
   - 역전파: 손실의 기울기 계산 (연쇄 법칙)
   - 경사하강법: W = W - lr * gradient

5. 학습 하이퍼파라미터
   - 학습률(lr): 이동 폭 (보통 0.001~0.1)
   - 에포크: 전체 데이터 반복 횟수
   - 배치: 한 번에 학습하는 샘플 수

6. 사용한 데이터셋
   - sklearn digits: 8x8 손글씨 숫자 (0-9)
   - 1797개 샘플, 64개 특성
   - MNIST보다 작아서 교육용으로 적합
""")

print("\n다음 차시 예고: Keras로 MLP 구현, 품질 예측 실습")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
