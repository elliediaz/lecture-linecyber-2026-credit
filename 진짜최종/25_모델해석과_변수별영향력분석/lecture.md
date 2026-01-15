# 25차시: 딥러닝 모델 해석과 학습 추적

## 학습 목표

1. **블랙박스 문제**와 딥러닝 모델 해석의 필요성을 이해합니다
2. **Saliency Map**으로 입력 픽셀별 중요도를 시각화합니다
3. **Grad-CAM**으로 CNN 모델의 판단 근거를 히트맵으로 확인합니다
4. **TensorBoard**와 **Keras Callbacks**로 학습 과정을 모니터링합니다
5. **Weights & Biases**로 실험을 체계적으로 추적하고 관리합니다

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 5분 | 딥러닝 모델 해석의 필요성 |
| Part 2 | 8분 | Grad-CAM과 Saliency Map |
| Part 3 | 7분 | 학습 추적과 모니터링 |
| Part 4 | 5분 | Weights & Biases 활용 |

**총 25분**

---

# Part 1: 딥러닝 모델 해석의 필요성 (5분)

## 1.1 블랙박스 문제란?

딥러닝 모델은 놀라운 성능을 보여주지만, 그 내부 동작을 이해하기 어렵다는 문제가 있습니다. 이를 **블랙박스(Black Box) 문제**라고 합니다.

```
딥러닝 모델의 블랙박스 특성:

입력 데이터           딥러닝 모델 내부              출력
                 ┌─────────────────────┐
[이미지] ────────>│   수백만 개 파라미터   │────────> [예측: 고양이]
                 │   복잡한 비선형 변환   │
                 │        ???          │
                 └─────────────────────┘
                       무슨 일이?
```

**블랙박스 문제의 핵심**:
- 입력과 출력의 관계를 인간이 직관적으로 이해하기 어려움
- "왜 이런 예측을 했는가?"라는 질문에 답하기 어려움
- 모델의 판단 근거를 설명할 수 없음

---

## 1.2 해석 가능한 AI (XAI)의 필요성

**XAI (eXplainable AI)**: 인공지능의 판단 과정을 인간이 이해할 수 있도록 설명하는 기술

### 해석이 필요한 실제 상황들

| 분야 | 상황 | 해석 필요 이유 |
|:----:|:----:|:-------------:|
| 의료 | 암 진단 AI | 왜 암으로 판정했는지 의사가 확인 필요 |
| 제조 | 불량 검출 | 어떤 결함 때문인지 작업자에게 설명 |
| 금융 | 대출 심사 | 거절 시 법적 설명 의무 존재 |
| 자율주행 | 사고 발생 | 차량이 왜 그런 판단을 했는지 분석 |
| 채용 | AI 면접 평가 | 탈락 사유를 지원자에게 설명 |

### 해석의 세 가지 목적

1. **신뢰 구축**: 모델의 판단 근거를 알아야 신뢰할 수 있습니다
2. **디버깅**: 모델이 잘못 학습했는지 확인할 수 있습니다
3. **개선**: 어떤 특성이 중요한지 알면 데이터/모델을 개선할 수 있습니다

---

## 1.3 ML 해석 기법 vs DL 해석 기법

이전 차시(18차시)에서 다룬 ML 모델 해석 기법과 오늘 배울 DL 특화 기법의 차이를 정리합니다.

### ML 모델 해석 기법 (복습)

| 기법 | 설명 | 대상 |
|:----:|:----:|:----:|
| Feature Importance | 변수별 중요도 순위 | 트리 기반 모델 |
| SHAP Values | 각 변수의 예측 기여도 | 모든 모델 |
| Permutation Importance | 변수 셔플 후 성능 변화 | 모든 모델 |
| Partial Dependence Plot | 변수와 예측의 관계 | 모든 모델 |

### DL 특화 해석 기법 (이번 차시)

| 기법 | 설명 | 대상 |
|:----:|:----:|:----:|
| Saliency Map | 입력 픽셀별 기울기 | 모든 DNN |
| Grad-CAM | 활성화 맵 기반 히트맵 | CNN |
| Attention 시각화 | 어텐션 가중치 분석 | Transformer |
| Activation Maximization | 뉴런 활성화 최대화 입력 | CNN |

**핵심 차이**: ML은 "변수별" 중요도, DL은 "입력 영역별" 기여도

---

## 1.4 딥러닝 해석 기법의 분류

```
딥러닝 해석 기법 분류

├── 기울기 기반 (Gradient-based)
│   ├── Saliency Map (Vanilla Gradient)
│   ├── Grad-CAM / Grad-CAM++
│   ├── Integrated Gradients
│   └── SmoothGrad
│
├── 섭동 기반 (Perturbation-based)
│   ├── LIME (Local Interpretable Model-agnostic)
│   ├── Occlusion Sensitivity
│   └── RISE (Randomized Input Sampling)
│
└── 활성화 기반 (Activation-based)
    ├── CAM (Class Activation Map)
    ├── Feature Visualization
    └── Network Dissection
```

이번 차시에서는 가장 실용적인 **Saliency Map**과 **Grad-CAM**에 집중합니다.

---

## 1.5 해석 기법 선택 가이드

| 모델 유형 | 데이터 유형 | 권장 기법 |
|:--------:|:----------:|:--------:|
| CNN | 이미지 | **Grad-CAM** (영역 단위) |
| CNN | 이미지 | Saliency Map (픽셀 단위) |
| MLP | 테이블 | SHAP, Permutation |
| Transformer | 텍스트/이미지 | Attention 시각화 |
| 모든 모델 | 모든 데이터 | LIME |

**실무 팁**: CNN 이미지 분류에는 Grad-CAM이 가장 직관적이고 실용적입니다.

---

# Part 2: Grad-CAM과 Saliency Map (8분)

## 2.1 Saliency Map 개념

**Saliency Map (현저성 맵)**은 입력 이미지의 각 픽셀이 예측 결과에 미치는 영향을 시각화합니다.

### 수학적 정의

$$\text{Saliency}(x_i) = \left| \frac{\partial y_c}{\partial x_i} \right|$$

- $y_c$: 특정 클래스 $c$에 대한 출력 스코어
- $x_i$: 입력 이미지의 $i$번째 픽셀
- $\frac{\partial y_c}{\partial x_i}$: 픽셀 $x_i$가 미세하게 변할 때 출력이 얼마나 변하는지

### 직관적 해석

- **기울기가 크다** = 해당 픽셀을 조금 바꾸면 예측이 크게 변함 = 중요한 픽셀
- **기울기가 작다** = 해당 픽셀을 바꿔도 예측이 거의 변하지 않음 = 중요하지 않은 픽셀

---

## 2.2 Saliency Map 구현

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_map(model, image, class_idx):
    """
    Saliency Map 계산

    Parameters:
    -----------
    model : tf.keras.Model
        학습된 분류 모델
    image : np.ndarray
        입력 이미지, shape: (1, H, W, C)
    class_idx : int
        관심 클래스 인덱스

    Returns:
    --------
    saliency : np.ndarray
        정규화된 Saliency Map, shape: (H, W)
    """
    # 텐서로 변환
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # 기울기 계산을 위한 GradientTape
    with tf.GradientTape() as tape:
        # 입력 텐서를 추적 대상으로 등록
        tape.watch(image_tensor)

        # 순전파
        predictions = model(image_tensor)

        # 타겟 클래스의 출력 스코어
        target_score = predictions[0, class_idx]

    # 입력에 대한 기울기 계산
    gradients = tape.gradient(target_score, image_tensor)

    # 절대값 취하고 채널 방향으로 최대값 선택
    # RGB 3채널 중 가장 큰 기울기를 사용
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = saliency.numpy()[0]

    # 0~1로 정규화
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

    return saliency


def visualize_saliency(original_image, saliency_map, pred_class, true_class=None):
    """
    Saliency Map 시각화

    Parameters:
    -----------
    original_image : np.ndarray
        원본 이미지 (H, W, C), 값 범위 0-1
    saliency_map : np.ndarray
        계산된 Saliency Map (H, W)
    pred_class : str
        예측 클래스 이름
    true_class : str, optional
        실제 클래스 이름
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    title = f'Original Image'
    if true_class:
        title += f'\nTrue: {true_class}'
    axes[0].set_title(title, fontsize=12)
    axes[0].axis('off')

    # Saliency Map (열 지도)
    im = axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 오버레이
    axes[2].imshow(original_image)
    axes[2].imshow(saliency_map, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay\nPred: {pred_class}', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('saliency_map_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saliency Map 저장 완료: saliency_map_visualization.png")
```

---

## 2.3 Saliency Map 실행 예제

```python
# 예제: 사전학습된 모델로 Saliency Map 생성
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. 모델 로드
print("VGG16 모델 로드 중...")
model = VGG16(weights='imagenet')

# 2. 이미지 로드 및 전처리
img_path = 'sample_cat.jpg'  # 테스트 이미지 경로
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_array_expanded.copy())

# 3. 예측
predictions = model.predict(img_preprocessed, verbose=0)
pred_class_idx = np.argmax(predictions[0])
pred_class_name = decode_predictions(predictions, top=1)[0][0][1]
pred_confidence = decode_predictions(predictions, top=1)[0][0][2]

print(f"예측 결과: {pred_class_name} (확률: {pred_confidence:.2%})")

# 4. Saliency Map 계산
saliency = compute_saliency_map(model, img_preprocessed, pred_class_idx)

# 5. 시각화
original_for_display = img_array / 255.0  # 0-1 범위로 정규화
visualize_saliency(original_for_display, saliency, pred_class_name)
```

---

## 2.4 CAM과 Grad-CAM 개념

### CAM (Class Activation Map)

CAM은 **마지막 합성곱 층의 특성 맵(Feature Map)**을 가중 합산하여 히트맵을 생성합니다.

$$\text{CAM}_c = \sum_{k} w_k^c \cdot A_k$$

- $A_k$: $k$번째 특성 맵
- $w_k^c$: 클래스 $c$에 대한 출력층의 가중치

**제한**: Global Average Pooling (GAP) 구조가 필수

### Grad-CAM (Gradient-weighted CAM)

Grad-CAM은 **기울기를 가중치로 사용**하여 CAM을 일반화합니다.

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$\text{Grad-CAM}_c = \text{ReLU}\left( \sum_k \alpha_k^c \cdot A_k \right)$$

**장점**: 모든 CNN 아키텍처에 적용 가능

---

## 2.5 Grad-CAM 동작 원리

```
Grad-CAM 계산 과정:

1. 순전파: 입력 이미지를 모델에 통과
   입력 → [Conv1] → [Conv2] → ... → [마지막 Conv] → [FC] → 출력
                                        ↓
                                   Feature Maps
                                   (H' x W' x K개)

2. 역전파: 출력에서 마지막 Conv까지 기울기 계산
   ∂y/∂A (각 Feature Map의 각 픽셀에 대한 기울기)

3. 가중치 계산: 기울기를 공간 방향으로 평균 (Global Average)
   α_k = (1/Z) * Σᵢ Σⱼ (∂y/∂A_ij^k)

4. 가중 합산: Feature Maps를 α_k로 가중 합산
   Grad-CAM = ReLU(Σ_k α_k * A_k)

5. 업샘플링: 원본 이미지 크기로 확대
```

---

## 2.6 Grad-CAM 구현

```python
import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM 히트맵 생성

    Parameters:
    -----------
    model : tf.keras.Model
        CNN 모델
    image : np.ndarray
        전처리된 입력 이미지, shape: (1, H, W, C)
    last_conv_layer_name : str
        마지막 합성곱 층의 이름
    pred_index : int, optional
        타겟 클래스 인덱스. None이면 최대 확률 클래스

    Returns:
    --------
    heatmap : np.ndarray
        정규화된 Grad-CAM 히트맵
    """
    # 마지막 Conv 층 출력과 모델 출력을 동시에 얻는 모델 생성
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 기울기 계산
    with tf.GradientTape() as tape:
        # 순전파: Conv 출력과 최종 예측 동시 획득
        conv_outputs, predictions = grad_model(image)

        # 타겟 클래스 결정
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # 타겟 클래스의 출력 스코어
        class_channel = predictions[:, pred_index]

    # 마지막 Conv 층에 대한 기울기 계산
    grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling으로 가중치(α) 계산
    # grads shape: (1, H, W, K) -> pooled_grads shape: (K,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 가중 합산
    conv_outputs = conv_outputs[0]  # (H, W, K)
    # 각 Feature Map을 해당 가중치로 곱하고 합산
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)  # (H, W)

    # ReLU 적용 (양수만 유지)
    heatmap = tf.maximum(heatmap, 0)

    # 정규화 (0~1)
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def apply_gradcam_overlay(original_image, heatmap, alpha=0.4):
    """
    Grad-CAM 히트맵을 원본 이미지에 오버레이

    Parameters:
    -----------
    original_image : np.ndarray
        원본 이미지, shape: (H, W, C), 값 범위 0-255
    heatmap : np.ndarray
        Grad-CAM 히트맵
    alpha : float
        오버레이 투명도 (0~1)

    Returns:
    --------
    superimposed : np.ndarray
        오버레이된 이미지
    heatmap_colored : np.ndarray
        컬러 히트맵
    """
    # 히트맵을 원본 이미지 크기로 리사이즈
    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    # 0-255 범위로 스케일링
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # JET 컬러맵 적용 (파랑->초록->노랑->빨강)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 원본 이미지와 오버레이
    superimposed = heatmap_colored * alpha + original_image * (1 - alpha)
    superimposed = np.uint8(superimposed)

    return superimposed, heatmap_colored
```

---

## 2.7 Grad-CAM 시각화 함수

```python
def visualize_gradcam(model, image_array, image_preprocessed,
                      last_conv_layer_name, class_names=None):
    """
    Grad-CAM 결과 종합 시각화

    Parameters:
    -----------
    model : tf.keras.Model
        학습된 CNN 모델
    image_array : np.ndarray
        원본 이미지 배열 (H, W, C), 값 범위 0-255
    image_preprocessed : np.ndarray
        전처리된 이미지 (1, H, W, C)
    last_conv_layer_name : str
        마지막 합성곱 층 이름
    class_names : list, optional
        클래스 이름 리스트
    """
    # 예측
    predictions = model.predict(image_preprocessed, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_prob = predictions[0][pred_idx]

    if class_names:
        pred_name = class_names[pred_idx]
    else:
        pred_name = f"Class {pred_idx}"

    print(f"예측: {pred_name} (확률: {pred_prob:.2%})")

    # Grad-CAM 히트맵 생성
    heatmap = make_gradcam_heatmap(
        model, image_preprocessed, last_conv_layer_name, pred_idx
    )

    # 오버레이 생성
    original_uint8 = image_array.astype(np.uint8)
    superimposed, heatmap_colored = apply_gradcam_overlay(
        original_uint8, heatmap, alpha=0.4
    )

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 원본 이미지
    axes[0].imshow(original_uint8)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Grad-CAM 히트맵 (원본 해상도)
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Feature Map Resolution)', fontsize=14)
    axes[1].axis('off')

    # 컬러 히트맵 (확대)
    axes[2].imshow(heatmap_colored)
    axes[2].set_title('Colored Heatmap\n(Upsampled)', fontsize=14)
    axes[2].axis('off')

    # 오버레이
    axes[3].imshow(superimposed)
    axes[3].set_title(f'Overlay\nPrediction: {pred_name}\n({pred_prob:.1%})', fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Grad-CAM 저장 완료: gradcam_visualization.png")

    return heatmap, superimposed
```

---

## 2.8 Grad-CAM 전체 실습 코드

```python
"""
Grad-CAM 전체 실습 코드
CIFAR-10 데이터셋으로 CNN 학습 후 Grad-CAM 적용
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import cv2

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# 1. 데이터 준비
# =============================================================================
print("데이터 로드 중...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 클래스 이름
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 정규화
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# 원-핫 인코딩
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# =============================================================================
# 2. CNN 모델 정의
# =============================================================================
def create_cnn_model():
    """Grad-CAM 적용을 위한 CNN 모델"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(32, 32, 3), name='conv1'),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2), name='pool1'),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        MaxPooling2D((2, 2), name='pool2'),
        Dropout(0.25),

        # Block 3 - Grad-CAM 타겟 층
        Conv2D(128, (3, 3), activation='relu', padding='same',
               name='conv5_gradcam_target'),
        MaxPooling2D((2, 2), name='pool3'),

        # 분류기
        Flatten(name='flatten'),
        Dense(256, activation='relu', name='fc1'),
        Dropout(0.5),
        Dense(10, activation='softmax', name='predictions')
    ])

    return model

print("\n모델 생성 중...")
model = create_cnn_model()
model.summary()

# =============================================================================
# 3. 모델 컴파일 및 학습
# =============================================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n모델 학습 중...")
history = model.fit(
    X_train_norm, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(X_test_norm, y_test_cat, verbose=0)
print(f"\n테스트 정확도: {test_acc:.2%}")

# =============================================================================
# 4. Grad-CAM 적용
# =============================================================================
print("\nGrad-CAM 시각화...")

# 테스트 이미지 선택
test_idx = 42
test_image = X_test_norm[test_idx:test_idx+1]  # (1, 32, 32, 3)
test_image_display = X_test[test_idx]  # (32, 32, 3), 0-255
true_label = CLASS_NAMES[y_test[test_idx][0]]

# Grad-CAM 실행
LAST_CONV_LAYER = 'conv5_gradcam_target'
heatmap, overlay = visualize_gradcam(
    model=model,
    image_array=test_image_display,
    image_preprocessed=test_image,
    last_conv_layer_name=LAST_CONV_LAYER,
    class_names=CLASS_NAMES
)

print(f"실제 레이블: {true_label}")
```

---

## 2.9 Saliency Map vs Grad-CAM 비교

```python
def compare_saliency_gradcam(model, image, image_preprocessed,
                              last_conv_layer_name, class_idx, class_name):
    """
    Saliency Map과 Grad-CAM 결과 비교
    """
    # Saliency Map 계산
    saliency = compute_saliency_map(model, image_preprocessed, class_idx)

    # Grad-CAM 계산
    heatmap = make_gradcam_heatmap(model, image_preprocessed,
                                    last_conv_layer_name, class_idx)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 상단: Saliency Map
    axes[0, 0].imshow(image.astype(np.uint8))
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(saliency, cmap='hot')
    axes[0, 1].set_title('Saliency Map', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image.astype(np.uint8))
    axes[0, 2].imshow(saliency, cmap='jet', alpha=0.5)
    axes[0, 2].set_title('Saliency Overlay', fontsize=12)
    axes[0, 2].axis('off')

    # 하단: Grad-CAM
    axes[1, 0].imshow(image.astype(np.uint8))
    axes[1, 0].set_title(f'Prediction: {class_name}', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap, cmap='jet')
    axes[1, 1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1, 1].axis('off')

    # Grad-CAM 오버레이
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    axes[1, 2].imshow(image.astype(np.uint8))
    axes[1, 2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[1, 2].set_title('Grad-CAM Overlay', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle('Saliency Map vs Grad-CAM Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('saliency_vs_gradcam.png', dpi=150, bbox_inches='tight')
    plt.show()

# 비교 실행 예시
# compare_saliency_gradcam(model, test_image_display, test_image,
#                          LAST_CONV_LAYER, pred_idx, pred_name)
```

### 비교 요약

| 특성 | Saliency Map | Grad-CAM |
|:----:|:-----------:|:--------:|
| 해상도 | 입력과 동일 (픽셀 수준) | 특성 맵 크기 (낮음) |
| 노이즈 | 많음 (세밀한 패턴) | 적음 (부드러운 영역) |
| 해석 | 어떤 픽셀이 중요한가 | 어떤 영역이 중요한가 |
| 계산량 | 낮음 | 중간 |
| 적용 범위 | 모든 미분 가능 모델 | CNN 특화 |

**실무 권장**: 먼저 Grad-CAM으로 전체 영역을 파악하고, 필요시 Saliency Map으로 세부 분석

---

# Part 3: 학습 추적과 모니터링 (7분)

## 3.1 학습 모니터링의 필요성

딥러닝 모델 학습 시 다양한 지표를 추적해야 합니다:

| 추적 항목 | 확인 목적 |
|:--------:|:--------:|
| Loss 곡선 | 학습 진행 상태, 수렴 여부 |
| Accuracy 곡선 | 성능 개선 추이 |
| Train vs Val 차이 | 과적합 여부 |
| 기울기 분포 | Vanishing/Exploding Gradient |
| 가중치 분포 | 초기화 적절성, 학습 진행 |
| 학습률 변화 | 스케줄링 동작 확인 |

---

## 3.2 TensorBoard 소개

**TensorBoard**는 TensorFlow의 공식 시각화 도구입니다.

### 주요 기능

```
TensorBoard 기능 구성

┌─────────────────────────────────────────────┐
│               TensorBoard                    │
├─────────────────────────────────────────────┤
│  Scalars     │ 수치 지표의 변화 그래프       │
│  Graphs      │ 모델 구조 시각화             │
│  Histograms  │ 가중치/기울기 분포 변화      │
│  Distributions│ 분포의 시간별 변화          │
│  Images      │ 입력/출력 이미지 확인        │
│  Text        │ 텍스트 로그                  │
│  Projector   │ 임베딩 시각화                │
│  Profiler    │ 성능 프로파일링              │
└─────────────────────────────────────────────┘
```

---

## 3.3 TensorBoard Callback 설정

```python
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os

# 로그 디렉토리 설정 (타임스탬프 포함)
log_dir = os.path.join(
    "logs", "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

# TensorBoard 콜백 생성
tensorboard_callback = TensorBoard(
    log_dir=log_dir,            # 로그 저장 경로
    histogram_freq=1,           # 매 에포크마다 히스토그램 기록
    write_graph=True,           # 모델 그래프 저장
    write_images=True,          # 가중치를 이미지로 저장
    update_freq='epoch',        # 업데이트 주기 ('batch' 또는 'epoch')
    profile_batch=(10, 15)      # 프로파일링할 배치 범위
)

print(f"TensorBoard 로그 디렉토리: {log_dir}")

# 모델 학습에 콜백 적용
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback],
    verbose=1
)
```

---

## 3.4 TensorBoard 실행

### 터미널에서 실행

```bash
# 터미널에서 TensorBoard 서버 시작
tensorboard --logdir=logs/fit --port=6006

# 브라우저에서 접속
# http://localhost:6006
```

### Jupyter Notebook에서 실행

```python
# TensorBoard 확장 로드
%load_ext tensorboard

# 인라인으로 TensorBoard 표시
%tensorboard --logdir logs/fit

# 또는 새 탭에서 열기
from tensorboard import notebook
notebook.start("--logdir logs/fit")
```

---

## 3.5 Keras Callbacks 종합

Keras는 학습 중 특정 시점에 자동 실행되는 다양한 콜백을 제공합니다.

```python
from tensorflow.keras.callbacks import (
    TensorBoard,          # 시각화 로깅
    EarlyStopping,        # 조기 종료
    ModelCheckpoint,      # 모델 저장
    ReduceLROnPlateau,    # 학습률 감소
    CSVLogger,            # CSV 로깅
    LearningRateScheduler,# 학습률 스케줄링
    TerminateOnNaN,       # NaN 발생 시 종료
    LambdaCallback        # 커스텀 함수 실행
)
```

---

## 3.6 EarlyStopping - 과적합 방지

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',         # 모니터링할 지표
    patience=10,                # 개선 없이 기다릴 에포크 수
    min_delta=0.001,            # 개선으로 인정할 최소 변화량
    mode='min',                 # 'min': 감소, 'max': 증가, 'auto': 자동
    restore_best_weights=True,  # 최적 가중치로 복원
    verbose=1                   # 종료 시 메시지 출력
)

# 사용 예시
model.fit(
    X_train, y_train,
    epochs=100,  # 최대 에포크 (실제로는 조기 종료됨)
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# 실제 학습된 에포크 수 확인
print(f"실제 학습 에포크: {early_stopping.stopped_epoch}")
```

### EarlyStopping 동작 원리

```
Epoch 1:  val_loss = 0.5000
Epoch 2:  val_loss = 0.4500  (개선)
Epoch 3:  val_loss = 0.4200  (개선)
Epoch 4:  val_loss = 0.4180  (개선, min_delta=0.001 이상)
Epoch 5:  val_loss = 0.4175  (개선 미미, patience 카운트 시작)
Epoch 6:  val_loss = 0.4180  (개선 없음, patience=1)
Epoch 7:  val_loss = 0.4200  (개선 없음, patience=2)
...
Epoch 15: val_loss = 0.4500  (개선 없음, patience=10)
→ 학습 종료, Epoch 4의 가중치로 복원
```

---

## 3.7 ModelCheckpoint - 모델 저장

```python
from tensorflow.keras.callbacks import ModelCheckpoint

# 최적 모델만 저장
checkpoint_best = ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,        # 최고 성능만 저장
    save_weights_only=False,    # 전체 모델 저장
    mode='max',
    verbose=1
)

# 매 에포크마다 저장 (파일명에 에포크 포함)
checkpoint_epoch = ModelCheckpoint(
    filepath='models/model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras',
    save_freq='epoch'
)

# 학습
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_best, checkpoint_epoch]
)

# 저장된 최적 모델 로드
best_model = tf.keras.models.load_model('models/best_model.keras')
```

---

## 3.8 ReduceLROnPlateau - 학습률 자동 감소

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # 학습률을 절반으로 감소
    patience=5,           # 5 에포크 동안 개선 없으면 감소
    min_lr=1e-6,          # 최소 학습률
    min_delta=0.001,      # 개선 기준
    verbose=1
)

# 사용 예시
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]
)
```

### 동작 예시

```
Epoch 1-5:   lr = 0.001, val_loss 감소
Epoch 6-10:  lr = 0.001, val_loss 정체
Epoch 11:    lr → 0.0005 (자동 감소)
Epoch 12-16: lr = 0.0005, val_loss 감소
Epoch 17-21: lr = 0.0005, val_loss 정체
Epoch 22:    lr → 0.00025 (자동 감소)
...
```

---

## 3.9 CSVLogger - 학습 기록 저장

```python
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False  # True면 기존 파일에 추가
)

# 학습
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[csv_logger]
)

# 로그 파일 확인
import pandas as pd
log_df = pd.read_csv('training_log.csv')
print(log_df.head())

# 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(log_df['epoch'], log_df['loss'], label='Train Loss')
axes[0].plot(log_df['epoch'], log_df['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()

axes[1].plot(log_df['epoch'], log_df['accuracy'], label='Train Acc')
axes[1].plot(log_df['epoch'], log_df['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

## 3.10 복합 콜백 사용 예시

```python
import datetime
import os

# 로그 및 모델 디렉토리 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/fit/{timestamp}"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# 콜백 리스트 구성
callbacks = [
    # 1. TensorBoard - 시각화
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    ),

    # 2. EarlyStopping - 조기 종료
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # 3. ModelCheckpoint - 최적 모델 저장
    ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # 4. ReduceLROnPlateau - 학습률 감소
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    # 5. CSVLogger - CSV 로깅
    CSVLogger(
        filename=f'logs/training_{timestamp}.csv'
    )
]

# 학습 실행
print(f"TensorBoard 로그: {log_dir}")
print(f"모델 저장 경로: {model_dir}/best_model.keras")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n학습 완료!")
print(f"최종 학습 에포크: {len(history.history['loss'])}")
```

---

## 3.11 커스텀 콜백 작성

```python
from tensorflow.keras.callbacks import Callback
import time

class CustomTrainingCallback(Callback):
    """
    학습 중 커스텀 로직을 실행하는 콜백
    """

    def __init__(self, target_accuracy=0.95):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.start_time = None

    def on_train_begin(self, logs=None):
        """학습 시작 시 호출"""
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print("학습을 시작합니다.")
        print(f"목표 정확도: {self.target_accuracy:.1%}")
        print(f"{'='*50}\n")

    def on_epoch_begin(self, epoch, logs=None):
        """각 에포크 시작 시 호출"""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """각 에포크 종료 시 호출"""
        epoch_time = time.time() - self.epoch_start_time

        val_acc = logs.get('val_accuracy', 0)
        train_acc = logs.get('accuracy', 0)

        # 과적합 경고
        if train_acc - val_acc > 0.1:
            print(f"  [경고] 과적합 징후 감지! (Train-Val 차이: {train_acc-val_acc:.2%})")

        # 목표 달성 확인
        if val_acc >= self.target_accuracy:
            print(f"\n목표 정확도 달성! (val_accuracy: {val_acc:.2%})")

    def on_train_end(self, logs=None):
        """학습 종료 시 호출"""
        total_time = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"학습 완료! 총 소요 시간: {total_time:.1f}초")
        print(f"최종 검증 정확도: {logs.get('val_accuracy', 0):.2%}")
        print(f"{'='*50}\n")

# 사용
custom_callback = CustomTrainingCallback(target_accuracy=0.90)
model.fit(X_train, y_train, callbacks=[custom_callback], ...)
```

---

# Part 4: Weights & Biases 활용 (5분)

## 4.1 Weights & Biases (W&B) 소개

**Weights & Biases**는 머신러닝 실험 추적 및 협업을 위한 MLOps 플랫폼입니다.

### 주요 기능

| 기능 | 설명 |
|:----:|:----:|
| **Experiments** | 실험 추적, 메트릭 로깅, 비교 |
| **Artifacts** | 데이터셋, 모델의 버전 관리 |
| **Sweeps** | 하이퍼파라미터 자동 탐색 |
| **Reports** | 결과 문서화 및 팀 공유 |
| **Alerts** | 학습 완료/이상 발생 시 알림 |

### TensorBoard와의 차이

| 항목 | TensorBoard | Weights & Biases |
|:----:|:----------:|:----------------:|
| 설치 | 로컬 | 클라우드 기반 |
| 팀 협업 | 제한적 | 강력함 |
| 실험 비교 | 기본 | 고급 기능 |
| 하이퍼파라미터 탐색 | 없음 | Sweeps |
| 가격 | 무료 | 무료/유료 |

---

## 4.2 W&B 설치 및 초기화

```bash
# 설치
pip install wandb
```

```python
import wandb

# 로그인 (최초 1회, API 키 필요)
# wandb.ai에서 무료 계정 생성 후 API 키 확인
wandb.login()

# 프로젝트 초기화
wandb.init(
    project="cifar10-classification",  # 프로젝트 이름
    name="cnn-experiment-001",         # 실험 이름 (선택)
    config={                           # 하이퍼파라미터 기록
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "dropout": 0.5
    },
    tags=["baseline", "cnn"]           # 태그 (선택)
)

print(f"W&B Run URL: {wandb.run.url}")
```

---

## 4.3 W&B 메트릭 로깅

### 기본 로깅

```python
import wandb

wandb.init(project="my-project")

# 학습 루프에서 메트릭 로깅
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()

    # 메트릭 로깅
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.learning_rate.numpy()
    })

# 학습 종료
wandb.finish()
```

### Keras 통합

```python
from wandb.integration.keras import WandbCallback

wandb.init(project="keras-project", config={"epochs": 30})

# Keras fit에 WandbCallback 추가
history = model.fit(
    X_train, y_train,
    epochs=wandb.config.epochs,
    validation_data=(X_val, y_val),
    callbacks=[
        WandbCallback(
            save_model=True,              # 모델 자동 저장
            save_weights_only=False,      # 전체 모델 저장
            log_weights=True,             # 가중치 히스토그램
            log_gradients=True            # 기울기 히스토그램
        )
    ]
)

wandb.finish()
```

---

## 4.4 이미지 및 시각화 로깅

```python
import wandb
import matplotlib.pyplot as plt
import numpy as np

# 예측 결과 이미지 로깅
def log_predictions(model, X_test, y_test, class_names, num_samples=10):
    """예측 결과를 W&B에 이미지로 로깅"""

    predictions = model.predict(X_test[:num_samples])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:num_samples].flatten()

    images = []
    for i in range(num_samples):
        # 이미지와 예측 결과를 하나의 figure로 생성
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(X_test[i])

        pred_name = class_names[pred_classes[i]]
        true_name = class_names[true_classes[i]]
        color = 'green' if pred_classes[i] == true_classes[i] else 'red'

        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}",
                     color=color, fontsize=10)
        ax.axis('off')

        # W&B Image 객체로 변환
        images.append(wandb.Image(fig, caption=f"Sample {i}"))
        plt.close(fig)

    # 로깅
    wandb.log({"predictions": images})

# Grad-CAM 결과 로깅
wandb.log({
    "gradcam/heatmap": wandb.Image(heatmap, caption="Grad-CAM Heatmap"),
    "gradcam/overlay": wandb.Image(overlay, caption="Grad-CAM Overlay")
})
```

---

## 4.5 하이퍼파라미터 탐색 (Sweeps)

```python
import wandb

# Sweep 설정 정의
sweep_config = {
    'method': 'bayes',  # 탐색 방법: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'dropout': {
            'min': 0.1,
            'max': 0.5
        },
        'hidden_units': {
            'values': [64, 128, 256, 512]
        }
    }
}

# Sweep 생성
sweep_id = wandb.sweep(sweep_config, project="hp-optimization")

# 학습 함수 정의
def train_sweep():
    wandb.init()
    config = wandb.config

    # config 값으로 모델 생성
    model = create_model(
        hidden_units=config.hidden_units,
        dropout=config.dropout
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 학습
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=config.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[WandbCallback()]
    )

    wandb.finish()

# Sweep 실행 (10회 시도)
wandb.agent(sweep_id, function=train_sweep, count=10)
```

---

## 4.6 모델 아티팩트 관리

```python
import wandb

# 모델을 Artifact로 저장
def save_model_artifact(model, artifact_name, metadata=None):
    """학습된 모델을 W&B Artifact로 저장"""

    # 로컬에 모델 저장
    model_path = f'{artifact_name}.keras'
    model.save(model_path)

    # Artifact 생성
    artifact = wandb.Artifact(
        name=artifact_name,
        type='model',
        description='Trained CNN model for CIFAR-10',
        metadata=metadata or {}
    )

    # 파일 추가
    artifact.add_file(model_path)

    # W&B에 업로드
    wandb.log_artifact(artifact)
    print(f"Artifact '{artifact_name}' 저장 완료")

    return artifact

# 사용 예시
save_model_artifact(
    model=model,
    artifact_name='cifar10-cnn-v1',
    metadata={
        'accuracy': 0.92,
        'epochs': 50,
        'architecture': 'CNN'
    }
)

# Artifact 불러오기
def load_model_artifact(artifact_name, version='latest'):
    """W&B Artifact에서 모델 로드"""

    artifact = wandb.use_artifact(f'{artifact_name}:{version}')
    artifact_dir = artifact.download()

    model = tf.keras.models.load_model(f'{artifact_dir}/{artifact_name}.keras')
    return model

# loaded_model = load_model_artifact('cifar10-cnn-v1')
```

---

## 4.7 W&B 실험 비교

```python
# 여러 실험을 순차적으로 실행하고 비교
experiments = [
    {"lr": 0.01, "batch": 32, "name": "high-lr"},
    {"lr": 0.001, "batch": 32, "name": "medium-lr"},
    {"lr": 0.0001, "batch": 32, "name": "low-lr"},
]

for exp in experiments:
    wandb.init(
        project="lr-comparison",
        name=exp["name"],
        config={
            "learning_rate": exp["lr"],
            "batch_size": exp["batch"]
        },
        reinit=True  # 동일 스크립트에서 여러 run 허용
    )

    # 모델 학습
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=exp["lr"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=exp["batch"],
        validation_data=(X_val, y_val),
        callbacks=[WandbCallback()]
    )

    wandb.finish()

# W&B 대시보드에서 세 실험을 한눈에 비교 가능
print("wandb.ai에서 실험 비교를 확인하세요!")
```

---

# 핵심 정리

## 오늘 배운 내용

### Part 1: 딥러닝 모델 해석의 필요성
- 블랙박스 문제와 XAI (해석 가능한 AI)의 중요성
- ML 해석 기법(Feature Importance, SHAP)과 DL 해석 기법의 차이
- 딥러닝 특화 해석 기법 분류 (기울기 기반, 섭동 기반, 활성화 기반)

### Part 2: Grad-CAM과 Saliency Map
- Saliency Map: 입력 픽셀별 기울기로 중요도 시각화
- Grad-CAM: 마지막 Conv 층의 활성화 맵을 기울기로 가중 합산
- 두 기법의 비교와 사용 시나리오

### Part 3: 학습 추적과 모니터링
- TensorBoard로 학습 과정 시각화 (Scalars, Histograms, Graphs)
- Keras Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- 복합 콜백 구성과 커스텀 콜백 작성

### Part 4: Weights & Biases
- W&B 초기화, 메트릭 로깅, 이미지 로깅
- Sweeps를 통한 하이퍼파라미터 자동 탐색
- Artifacts로 모델 버전 관리

---

## 도구 선택 가이드

| 상황 | 권장 도구 |
|:----:|:--------:|
| CNN 이미지 분류 해석 | **Grad-CAM** |
| 픽셀 수준 세부 분석 | Saliency Map |
| 로컬에서 빠른 확인 | TensorBoard |
| 여러 실험 체계적 관리 | **Weights & Biases** |
| 팀 협업/공유 | Weights & Biases |
| 하이퍼파라미터 탐색 | W&B Sweeps |
| 과적합 방지 | EarlyStopping |
| 최적 모델 자동 저장 | ModelCheckpoint |

---

## 실습 체크리스트

- [ ] 딥러닝 블랙박스 문제와 XAI의 필요성을 설명할 수 있다
- [ ] Saliency Map의 수학적 원리를 이해한다
- [ ] Grad-CAM 코드를 작성하고 결과를 해석할 수 있다
- [ ] TensorBoard 콜백을 설정하고 웹에서 확인할 수 있다
- [ ] EarlyStopping, ModelCheckpoint를 적절히 활용할 수 있다
- [ ] W&B로 실험을 추적하고 여러 실험을 비교할 수 있다
- [ ] 복합 콜백을 구성하여 학습 파이프라인을 자동화할 수 있다

---

## 다음 차시 예고

### 26차시: 모델 저장과 실무 배포 준비

- 모델 저장 형식: SavedModel, ONNX, TFLite
- 모델 경량화: Quantization, Pruning, Knowledge Distillation
- 추론 최적화와 서빙 준비
- 실무 배포를 위한 체크리스트

**학습된 모델을 실제 서비스에 적용하는 방법을 배웁니다**

---

# 부록: 전체 통합 실습 코드

```python
"""
25차시 전체 통합 실습: 딥러닝 모델 해석과 학습 추적
- Grad-CAM 시각화
- TensorBoard 모니터링
- Weights & Biases 실험 추적
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.datasets import cifar10

# W&B (설치된 경우)
try:
    import wandb
    from wandb.integration.keras import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B integration.")

# =============================================================================
# 설정
# =============================================================================
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
USE_WANDB = False  # W&B 사용 여부

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 디렉토리 설정
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = f"logs/fit/{timestamp}"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# =============================================================================
# 2. 모델 정의
# =============================================================================
print("\n" + "=" * 60)
print("2. CNN 모델 정의")
print("=" * 60)

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(32, 32, 3), name='conv1'),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same',
               name='conv5_gradcam'),  # Grad-CAM 타겟
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax', name='predictions')
    ])
    return model

model = create_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# =============================================================================
# 3. 콜백 설정
# =============================================================================
print("\n" + "=" * 60)
print("3. 콜백 설정")
print("=" * 60)

callbacks = [
    TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_DIR}/best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    CSVLogger(f'{LOG_DIR}/training.csv')
]

# W&B 콜백 추가 (선택)
if USE_WANDB and WANDB_AVAILABLE:
    wandb.init(
        project="cifar10-gradcam",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        }
    )
    callbacks.append(WandbCallback(save_model=True))

print(f"TensorBoard 로그: {LOG_DIR}")
print(f"모델 저장 경로: {MODEL_DIR}/best_model.keras")

# =============================================================================
# 4. 모델 학습
# =============================================================================
print("\n" + "=" * 60)
print("4. 모델 학습")
print("=" * 60)

history = model.fit(
    X_train_norm, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(X_test_norm, y_test_cat, verbose=0)
print(f"\n테스트 정확도: {test_acc:.2%}")

# =============================================================================
# 5. Grad-CAM 시각화
# =============================================================================
print("\n" + "=" * 60)
print("5. Grad-CAM 시각화")
print("=" * 60)

def make_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

# 테스트 이미지로 Grad-CAM 생성
test_idx = 42
test_image = X_test_norm[test_idx:test_idx+1]
test_image_display = X_test[test_idx]
true_label = CLASS_NAMES[y_test[test_idx][0]]

# 예측
pred = model.predict(test_image, verbose=0)
pred_idx = np.argmax(pred[0])
pred_label = CLASS_NAMES[pred_idx]
pred_prob = pred[0][pred_idx]

print(f"실제 레이블: {true_label}")
print(f"예측 레이블: {pred_label} ({pred_prob:.2%})")

# Grad-CAM 생성
heatmap = make_gradcam_heatmap(model, test_image, 'conv5_gradcam', pred_idx)

# 시각화
heatmap_resized = cv2.resize(heatmap, (32, 32))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(test_image_display)
axes[0].set_title(f'Original\nTrue: {true_label}')
axes[0].axis('off')

axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')

axes[2].imshow(test_image_display)
axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
axes[2].set_title(f'Overlay\nPred: {pred_label} ({pred_prob:.1%})')
axes[2].axis('off')

axes[3].barh(CLASS_NAMES, pred[0])
axes[3].set_title('Predictions')
axes[3].set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f'{LOG_DIR}/gradcam_result.png', dpi=150)
plt.show()

print(f"\nGrad-CAM 결과 저장: {LOG_DIR}/gradcam_result.png")

# =============================================================================
# 6. 학습 곡선 시각화
# =============================================================================
print("\n" + "=" * 60)
print("6. 학습 곡선 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{LOG_DIR}/learning_curves.png', dpi=150)
plt.show()

print(f"학습 곡선 저장: {LOG_DIR}/learning_curves.png")

# W&B 종료
if USE_WANDB and WANDB_AVAILABLE:
    wandb.log({
        "test_accuracy": test_acc,
        "gradcam": wandb.Image(f'{LOG_DIR}/gradcam_result.png')
    })
    wandb.finish()

# =============================================================================
# 완료
# =============================================================================
print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
print(f"TensorBoard 실행: tensorboard --logdir={LOG_DIR}")
print(f"최적 모델: {MODEL_DIR}/best_model.keras")
print(f"학습 로그: {LOG_DIR}/training.csv")
```

---

# 참고 자료

## 추천 학습 자료

1. **Grad-CAM 논문**
   - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
   - https://arxiv.org/abs/1610.02391

2. **TensorBoard 공식 문서**
   - https://www.tensorflow.org/tensorboard

3. **Weights & Biases 문서**
   - https://docs.wandb.ai/

4. **Keras Callbacks 가이드**
   - https://keras.io/api/callbacks/

## 핵심 용어 정리

| 용어 | 설명 |
|:----:|:----:|
| XAI | 해석 가능한 인공지능 |
| Saliency Map | 입력 픽셀별 기울기 시각화 |
| Grad-CAM | 기울기 가중 활성화 맵 |
| TensorBoard | TensorFlow 시각화 도구 |
| Callback | 학습 중 자동 실행 함수 |
| EarlyStopping | 조기 종료 콜백 |
| W&B | Weights & Biases, MLOps 플랫폼 |
| Artifact | 데이터셋/모델의 버전 관리 단위 |
| Sweep | 하이퍼파라미터 자동 탐색 |

---

# 수고하셨습니다!

**오늘의 핵심 메시지**:

1. 딥러닝 모델도 해석 가능합니다 (XAI)
2. Grad-CAM으로 CNN의 판단 근거를 시각적으로 확인합니다
3. TensorBoard와 Callbacks로 학습을 체계적으로 관리합니다
4. W&B로 실험을 추적하고 팀과 협업합니다

**다음 시간**: 모델 저장과 실무 배포 준비
