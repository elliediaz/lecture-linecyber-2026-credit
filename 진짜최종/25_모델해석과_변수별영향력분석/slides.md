---
marp: true
theme: default
paginate: true
backgroundColor: #fff
math: mathjax
---

# 25차시: 딥러닝 모델 해석과 학습 추적
## Grad-CAM, TensorBoard, Weights & Biases

**학습 목표**
- 딥러닝 모델의 블랙박스 문제와 해석 필요성을 이해합니다
- Grad-CAM과 Saliency Map으로 CNN 판단 근거를 시각화합니다
- TensorBoard와 Callbacks로 학습 과정을 모니터링합니다
- Weights & Biases로 실험을 체계적으로 관리합니다

---

# 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 5분 | 딥러닝 모델 해석의 필요성 |
| Part 2 | 8분 | Grad-CAM과 Saliency Map |
| Part 3 | 7분 | 학습 추적과 모니터링 |
| Part 4 | 5분 | Weights & Biases 활용 |

**총 25분**

---

# Part 1: 딥러닝 모델 해석의 필요성
## 블랙박스 vs 해석 가능성

---

## 1.1 블랙박스 문제란?

**딥러닝 모델의 특성**:

```
입력 (이미지)  →  [ 신경망 내부 ]  →  출력 (예측)
                      ?
                 무슨 일이 일어났지?
```

- 수백만 개의 파라미터
- 복잡한 비선형 변환
- 인간이 직관적으로 이해하기 어려움

**문제**: "왜 이런 예측을 했는가?"에 답하기 어려움

---

## 1.2 해석이 필요한 이유

| 상황 | 해석 필요성 |
|:----:|:----------:|
| 의료 진단 | 왜 암으로 판정했는지 설명 필요 |
| 제조 불량 판정 | 어떤 결함 때문인지 근거 제시 |
| 금융 심사 | 거절 사유 법적 설명 의무 |
| 자율주행 | 사고 발생 시 원인 분석 |

**신뢰성**: 모델의 판단 근거를 알아야 신뢰할 수 있습니다

---

## 1.3 ML vs DL 해석 기법 비교

| 구분 | ML 해석 기법 | DL 특화 해석 기법 |
|:----:|:-----------:|:----------------:|
| 대상 | 테이블 데이터 | 이미지, 시퀀스 |
| 방법 | Feature Importance | Grad-CAM |
| | SHAP Values | Saliency Map |
| | Permutation | Activation Map |
| 관점 | 변수별 중요도 | 입력 영역별 기여도 |

**이번 차시**: 딥러닝 특화 해석 기법에 집중합니다

---

## 1.4 딥러닝 해석 기법의 분류

```
딥러닝 해석 기법
├── 기울기 기반 (Gradient-based)
│   ├── Saliency Map
│   ├── Grad-CAM
│   └── Integrated Gradients
├── 섭동 기반 (Perturbation-based)
│   ├── LIME
│   └── Occlusion
└── 활성화 기반 (Activation-based)
    ├── CAM
    └── Feature Visualization
```

---

## 1.5 해석 기법 선택 가이드

| 모델 유형 | 권장 기법 | 특징 |
|:--------:|:--------:|:----:|
| CNN (이미지) | **Grad-CAM** | 히트맵으로 관심 영역 표시 |
| 일반 DNN | Saliency Map | 입력 픽셀별 중요도 |
| Transformer | Attention 시각화 | 어텐션 가중치 분석 |
| 모든 모델 | LIME | 모델 무관 해석 |

**CNN 실습에 Grad-CAM이 가장 적합합니다**

---

# Part 2: Grad-CAM과 Saliency Map
## CNN 모델의 시각적 해석

---

## 2.1 Saliency Map이란?

**정의**: 입력 이미지의 각 픽셀이 예측에 미치는 영향을 시각화

$$\text{Saliency}(x_i) = \left| \frac{\partial y_c}{\partial x_i} \right|$$

- $y_c$: 클래스 $c$에 대한 출력 스코어
- $x_i$: 입력 이미지의 $i$번째 픽셀
- 기울기의 절대값이 클수록 중요한 픽셀

---

## 2.2 Saliency Map 직관적 이해

```
원본 이미지              Saliency Map
┌─────────────┐         ┌─────────────┐
│             │         │   밝음=중요  │
│   고양이    │   →     │     ■■■     │
│             │         │    ■   ■    │
│             │         │     얼굴    │
└─────────────┘         └─────────────┘

"예측을 바꾸려면 어디를 수정해야 하는가?"
→ 밝은 영역을 수정하면 예측이 크게 변함
```

---

## 2.3 Saliency Map 구현

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_map(model, image, class_idx):
    """
    Saliency Map 계산

    Parameters:
    - model: 학습된 CNN 모델
    - image: 입력 이미지 (1, H, W, C)
    - class_idx: 타겟 클래스 인덱스

    Returns:
    - saliency: 정규화된 Saliency Map
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        target_score = predictions[0, class_idx]

    # 입력에 대한 기울기 계산
    gradients = tape.gradient(target_score, image_tensor)

    # 절대값 취하고 채널 방향으로 최대값
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = saliency.numpy()[0]

    # 정규화 (0~1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency
```

---

## 2.4 Saliency Map 시각화

```python
def visualize_saliency(original_image, saliency_map, title="Saliency Map"):
    """Saliency Map을 원본 이미지와 함께 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Saliency Map
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')

    # 오버레이
    axes[2].imshow(original_image)
    axes[2].imshow(saliency_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('saliency_map.png', dpi=150)
    plt.show()
```

---

## 2.5 CAM (Class Activation Map)

**CAM**: 마지막 합성곱 층의 특성 맵을 가중 합산

$$\text{CAM}_c = \sum_k w_k^c \cdot A_k$$

- $A_k$: $k$번째 특성 맵 (Feature Map)
- $w_k^c$: 클래스 $c$에 대한 가중치
- Global Average Pooling 직전 층 사용

**제한**: GAP 구조가 필요함

---

## 2.6 Grad-CAM이란?

**Grad-CAM**: 기울기 가중 활성화 맵

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$\text{Grad-CAM}_c = \text{ReLU}\left( \sum_k \alpha_k^c \cdot A_k \right)$$

- 기울기를 가중치로 사용 (CAM의 일반화)
- 모든 CNN 아키텍처에 적용 가능
- 특정 층의 활성화 맵 시각화

---

## 2.7 Grad-CAM 직관적 이해

```
CNN 모델 내부 구조

입력 → [Conv층들] → [마지막Conv] → [FC층] → 출력
                         ↓
                   Feature Maps (A)
                   14x14 x 512개

Grad-CAM 과정:
1. 출력 → 마지막 Conv까지 역전파
2. 각 Feature Map의 중요도(α) 계산
3. 가중 합산 → 히트맵 생성
4. 원본 크기로 업샘플링
```

---

## 2.8 Grad-CAM 구현 (1/2)

```python
import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM 히트맵 생성

    Parameters:
    - model: CNN 모델
    - image: 전처리된 입력 이미지
    - last_conv_layer_name: 마지막 합성곱 층 이름
    - pred_index: 타겟 클래스 (None이면 최대 확률 클래스)
    """
    # 마지막 Conv 층 출력과 모델 출력을 얻는 모델 생성
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 기울기 계산
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
```

---

## 2.9 Grad-CAM 구현 (2/2)

```python
    # (계속)
    # 마지막 Conv 층에 대한 기울기
    grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling으로 가중치 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 가중 합산
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU 적용 및 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()
```

---

## 2.10 Grad-CAM 시각화

```python
def apply_gradcam(original_image, heatmap, alpha=0.4):
    """Grad-CAM 히트맵을 원본 이미지에 오버레이"""

    # 히트맵을 원본 크기로 리사이즈
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # 컬러맵 적용 (JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 오버레이
    superimposed = heatmap_colored * alpha + original_image * (1 - alpha)
    superimposed = np.uint8(superimposed)

    return superimposed, heatmap_colored
```

---

## 2.11 전체 Grad-CAM 실습 코드

```python
# 예시: 사전학습된 VGG16으로 Grad-CAM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# 모델 로드
model = VGG16(weights='imagenet')

# 이미지 로드 및 전처리
img_path = 'sample_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_array.copy())

# 예측
preds = model.predict(img_preprocessed)
pred_class = np.argmax(preds[0])
print(f"예측: {decode_predictions(preds, top=1)[0][0]}")

# Grad-CAM 생성
heatmap = make_gradcam_heatmap(model, img_preprocessed, 'block5_conv3', pred_class)

# 시각화
original = img_array[0].astype('uint8')
superimposed, _ = apply_gradcam(original, heatmap)
```

---

## 2.12 Grad-CAM 결과 해석

```
원본 이미지         Grad-CAM 히트맵         오버레이
┌───────────┐      ┌───────────┐      ┌───────────┐
│           │      │   빨강=   │      │           │
│   개(Dog) │  →   │   관심영역│  →   │  개 얼굴  │
│           │      │           │      │  강조     │
└───────────┘      └───────────┘      └───────────┘

해석:
- 빨간색/노란색: 예측에 크게 기여한 영역
- 파란색: 영향이 적은 영역
- "모델이 개의 얼굴을 보고 판단했다"
```

---

## 2.13 Saliency vs Grad-CAM 비교

| 특성 | Saliency Map | Grad-CAM |
|:----:|:-----------:|:--------:|
| 해상도 | 픽셀 수준 (높음) | 특성 맵 수준 (낮음) |
| 노이즈 | 많음 | 적음 |
| 해석 용이성 | 세부적 | 영역 단위 |
| 계산 비용 | 낮음 | 중간 |
| 적용 범위 | 모든 DNN | CNN 특화 |

**실무 권장**: Grad-CAM으로 전체 파악, Saliency로 세부 확인

---

# Part 3: 학습 추적과 모니터링
## TensorBoard와 Keras Callbacks

---

## 3.1 학습 모니터링의 필요성

**학습 중 확인해야 할 것들**:

| 항목 | 확인 목적 |
|:----:|:--------:|
| 손실 곡선 | 학습 진행 상태 |
| 정확도 곡선 | 성능 개선 확인 |
| 과적합 여부 | train/val 차이 |
| 기울기 분포 | vanishing/exploding |
| 가중치 분포 | 초기화 적절성 |

**TensorBoard**: 이 모든 것을 시각적으로 제공

---

## 3.2 TensorBoard란?

```
TensorFlow의 시각화 도구

┌──────────────────────────────────────┐
│              TensorBoard              │
├──────────────────────────────────────┤
│  Scalars  │ 손실, 정확도 곡선         │
│  Graphs   │ 모델 구조 시각화          │
│  Histograms│ 가중치/기울기 분포       │
│  Images   │ 입력/출력 이미지          │
│  Profiles │ 성능 프로파일링           │
└──────────────────────────────────────┘
```

**웹 대시보드로 실시간 모니터링**

---

## 3.3 TensorBoard 기본 설정

```python
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 로그 디렉토리 설정 (시간 기반)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard 콜백 생성
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,      # 매 에포크마다 히스토그램 기록
    write_graph=True,      # 모델 그래프 저장
    write_images=True,     # 가중치를 이미지로 저장
    update_freq='epoch',   # 업데이트 주기
    profile_batch=2        # 프로파일링할 배치
)

# 모델 학습에 콜백 적용
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)
```

---

## 3.4 TensorBoard 실행

```bash
# 터미널에서 실행
tensorboard --logdir=logs/fit

# 브라우저에서 접속
# http://localhost:6006
```

**Jupyter Notebook에서 실행**:

```python
# 매직 명령어로 인라인 표시
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

---

## 3.5 TensorBoard 주요 탭

**Scalars 탭**:
- loss, accuracy 등 수치 지표의 변화
- train vs validation 비교
- 여러 실험 동시 비교 가능

```
Loss 그래프          Accuracy 그래프
   │                    │
   │ ↘train             │      ↗train
   │   ↘                │    ↗
   │     val            │  val
   └────────→          └────────→
     epoch               epoch
```

---

## 3.6 Histograms와 Distributions

**가중치/기울기 분포 모니터링**:

```python
# histogram_freq=1 설정 시 기록됨
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1  # 매 에포크마다 히스토그램
)
```

**확인 사항**:
- 가중치가 0으로 수렴하지 않는지 (Dead neurons)
- 기울기가 폭발/소실하지 않는지
- 층별 활성화 분포의 변화

---

## 3.7 Keras Callbacks 개요

```python
from tensorflow.keras.callbacks import (
    TensorBoard,      # 시각화
    EarlyStopping,    # 조기 종료
    ModelCheckpoint,  # 모델 저장
    ReduceLROnPlateau,# 학습률 감소
    CSVLogger,        # CSV 로깅
    LearningRateScheduler  # 학습률 스케줄링
)
```

**콜백**: 학습 중 특정 시점에 자동 실행되는 함수

---

## 3.8 EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',    # 모니터링할 지표
    patience=10,           # 개선 없이 기다릴 에포크 수
    min_delta=0.001,       # 개선으로 인정할 최소 변화량
    mode='min',            # 'min'(손실) 또는 'max'(정확도)
    restore_best_weights=True  # 최적 가중치 복원
)

model.fit(
    X_train, y_train,
    epochs=100,  # 최대 에포크
    callbacks=[early_stopping]
)
```

**과적합 방지**: validation 성능이 개선되지 않으면 자동 중단

---

## 3.9 ModelCheckpoint

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # 저장 경로
    monitor='val_accuracy',       # 모니터링 지표
    save_best_only=True,          # 최고 성능만 저장
    save_weights_only=False,      # 전체 모델 저장
    mode='max',                   # 최대값 기준
    verbose=1                     # 저장 시 메시지 출력
)

# 에포크별 저장
checkpoint_epoch = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.keras',
    save_freq='epoch'  # 매 에포크마다 저장
)
```

---

## 3.10 CSVLogger

```python
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False  # 기존 파일에 추가 여부
)

# 학습 실행
model.fit(X_train, y_train, callbacks=[csv_logger])

# 로그 확인
import pandas as pd
log_df = pd.read_csv('training_log.csv')
print(log_df.tail())
```

**CSV 출력 예시**:
```
epoch,loss,accuracy,val_loss,val_accuracy
0,0.6543,0.6234,0.5432,0.6543
1,0.4321,0.7654,0.4567,0.7234
...
```

---

## 3.11 ReduceLROnPlateau

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # 모니터링 지표
    factor=0.5,            # 학습률 감소 비율
    patience=5,            # 기다릴 에포크 수
    min_lr=1e-6,           # 최소 학습률
    verbose=1              # 감소 시 메시지 출력
)

# 정체기에 학습률 자동 감소
# val_loss가 5 에포크 동안 개선되지 않으면
# learning_rate = learning_rate * 0.5
```

---

## 3.12 복합 콜백 사용

```python
# 여러 콜백을 함께 사용
callbacks = [
    TensorBoard(log_dir='./logs'),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5),
    CSVLogger('training.csv')
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks
)
```

---

## 3.13 커스텀 콜백 작성

```python
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    """학습 중 커스텀 로직 실행"""

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n=== Epoch {epoch + 1} 시작 ===")

    def on_epoch_end(self, epoch, logs=None):
        # 특정 조건에서 알림
        if logs.get('val_accuracy') > 0.95:
            print("목표 정확도 달성!")

    def on_batch_end(self, batch, logs=None):
        # 매 배치 후 실행
        pass

# 사용
model.fit(X_train, y_train, callbacks=[CustomCallback()])
```

---

# Part 4: 외부 API 연동 - Weights & Biases
## 실험 추적과 팀 협업

---

## 4.1 Weights & Biases (W&B) 소개

```
Weights & Biases = MLOps 플랫폼

┌─────────────────────────────────────┐
│              W&B 기능               │
├─────────────────────────────────────┤
│  Experiments  │ 실험 추적/비교      │
│  Artifacts    │ 데이터셋/모델 버전  │
│  Sweeps       │ 하이퍼파라미터 튜닝 │
│  Reports      │ 결과 문서화/공유    │
│  Alerts       │ 학습 완료 알림      │
└─────────────────────────────────────┘
```

**무료 플랜**: 개인/학술 사용 무료

---

## 4.2 W&B 설치 및 로그인

```bash
# 설치
pip install wandb

# 로그인 (API 키 필요)
wandb login
```

```python
import wandb

# 프로젝트 초기화
wandb.init(
    project="my-ml-project",
    name="experiment-001",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "architecture": "CNN"
    }
)
```

---

## 4.3 W&B 기본 로깅

```python
import wandb

# 초기화
wandb.init(project="digit-classification")

# 학습 루프에서 로깅
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss, val_acc = evaluate()

    # 메트릭 로깅
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# 학습 종료
wandb.finish()
```

---

## 4.4 Keras와 W&B 통합

```python
import wandb
from wandb.integration.keras import WandbCallback

# W&B 초기화
wandb.init(
    project="keras-cnn",
    config={
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epochs": 30
    }
)

# Keras 모델 학습에 WandbCallback 추가
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=wandb.config.epochs,
    callbacks=[
        WandbCallback(save_model=True)  # 모델도 자동 저장
    ]
)

wandb.finish()
```

---

## 4.5 하이퍼파라미터 로깅

```python
# 설정값 로깅
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 32
config.dropout = 0.3
config.hidden_units = 128

# 또는 초기화 시 전달
wandb.init(
    project="hp-search",
    config={
        "model_type": "CNN",
        "conv_layers": 3,
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "pool_size": 2,
        "dense_units": 256,
        "dropout": 0.5
    }
)

# config 값 사용
model.add(Dense(wandb.config.dense_units, activation='relu'))
model.add(Dropout(wandb.config.dropout))
```

---

## 4.6 이미지 로깅

```python
# 예측 결과 이미지 로깅
import matplotlib.pyplot as plt

def log_predictions(model, X_test, y_test, num_samples=10):
    predictions = model.predict(X_test[:num_samples])

    images = []
    for i in range(num_samples):
        fig, ax = plt.subplots()
        ax.imshow(X_test[i])
        ax.set_title(f"True: {y_test[i]}, Pred: {predictions[i].argmax()}")
        ax.axis('off')

        # W&B Image 객체로 변환
        images.append(wandb.Image(fig, caption=f"Sample {i}"))
        plt.close(fig)

    # 로깅
    wandb.log({"predictions": images})

# Grad-CAM 히트맵 로깅
wandb.log({
    "gradcam": wandb.Image(superimposed, caption="Grad-CAM Visualization")
})
```

---

## 4.7 W&B 대시보드 활용

```
W&B 웹 대시보드

┌────────────────────────────────────────────┐
│  Project: keras-cnn                         │
├────────────────────────────────────────────┤
│  Runs:                                      │
│  ├── exp-001 (lr=0.001) - acc: 0.92        │
│  ├── exp-002 (lr=0.01)  - acc: 0.87        │
│  └── exp-003 (lr=0.005) - acc: 0.94        │
│                                             │
│  Charts:      [loss] [accuracy] [custom]    │
│  Compare:     [ ] exp-001  [ ] exp-003     │
│  Reports:     Create Report                 │
└────────────────────────────────────────────┘
```

---

## 4.8 실험 비교 기능

```python
# 여러 실험을 자동으로 비교
# 각 실험에 다른 설정 적용

learning_rates = [0.01, 0.001, 0.0001]

for lr in learning_rates:
    wandb.init(
        project="lr-comparison",
        name=f"lr-{lr}",
        config={"learning_rate": lr},
        reinit=True  # 반복 실험 허용
    )

    model = create_model()
    model.compile(optimizer=Adam(learning_rate=lr), ...)
    model.fit(..., callbacks=[WandbCallback()])

    wandb.finish()

# 웹 대시보드에서 한눈에 비교 가능
```

---

## 4.9 모델 아티팩트 저장

```python
# 학습된 모델을 W&B Artifact로 저장
artifact = wandb.Artifact(
    name='trained-model',
    type='model',
    description='CNN for image classification',
    metadata={
        "accuracy": final_accuracy,
        "epochs_trained": num_epochs
    }
)

# 모델 파일 추가
model.save('model.keras')
artifact.add_file('model.keras')

# 아티팩트 로깅
wandb.log_artifact(artifact)

# 나중에 불러오기
artifact = wandb.use_artifact('trained-model:latest')
artifact_dir = artifact.download()
loaded_model = tf.keras.models.load_model(f'{artifact_dir}/model.keras')
```

---

## 4.10 Sweep (하이퍼파라미터 탐색)

```python
# Sweep 설정 정의
sweep_config = {
    'method': 'bayes',  # 베이지안 최적화
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16, 32, 64]},
        'dropout': {'min': 0.1, 'max': 0.5}
    }
}

# Sweep 생성
sweep_id = wandb.sweep(sweep_config, project="hp-sweep")

def train():
    wandb.init()
    config = wandb.config
    # config 값으로 모델 학습
    ...

# Sweep 실행 (10회 시도)
wandb.agent(sweep_id, function=train, count=10)
```

---

## 4.11 팀 협업 기능

```
W&B Teams 기능

1. 프로젝트 공유
   - 팀 멤버 초대
   - 권한 관리 (view/edit/admin)

2. Report 작성
   - 실험 결과 문서화
   - 그래프/테이블 포함
   - 마크다운 지원

3. 알림 설정
   - 학습 완료 시 Slack/Email 알림
   - 특정 조건 달성 시 알림
```

---

## 4.12 TensorBoard vs W&B 비교

| 기능 | TensorBoard | Weights & Biases |
|:----:|:----------:|:----------------:|
| 설치 | 로컬 | 클라우드 기반 |
| 실시간 모니터링 | O | O |
| 실험 비교 | 제한적 | 강력함 |
| 팀 협업 | X | O |
| 하이퍼파라미터 탐색 | X | O (Sweeps) |
| 모델 버전 관리 | X | O (Artifacts) |
| 가격 | 무료 | 무료/유료 |

**권장**: 개인 학습은 TensorBoard, 팀 프로젝트는 W&B

---

# 핵심 정리

---

## 오늘 배운 내용

1. **딥러닝 해석의 필요성**
   - 블랙박스 문제와 해석 가능한 AI (XAI)
   - ML과 DL 해석 기법의 차이

2. **Grad-CAM과 Saliency Map**
   - 입력 영역별 기여도 시각화
   - CNN 모델의 판단 근거 확인

3. **TensorBoard와 Callbacks**
   - 학습 곡선 실시간 모니터링
   - EarlyStopping, ModelCheckpoint 등 자동화

4. **Weights & Biases**
   - 실험 추적 및 비교
   - 하이퍼파라미터 튜닝과 팀 협업

---

## 도구 선택 가이드

| 상황 | 권장 도구 |
|:----:|:--------:|
| CNN 해석이 필요할 때 | Grad-CAM |
| 픽셀 수준 분석 | Saliency Map |
| 로컬에서 빠르게 확인 | TensorBoard |
| 여러 실험 체계적 관리 | Weights & Biases |
| 팀 협업 프로젝트 | Weights & Biases |
| 하이퍼파라미터 탐색 | W&B Sweeps |

---

## 체크리스트

- [ ] 딥러닝 블랙박스 문제를 설명할 수 있다
- [ ] Saliency Map의 원리를 이해한다
- [ ] Grad-CAM 코드를 작성하고 결과를 해석할 수 있다
- [ ] TensorBoard 콜백을 설정하고 실행할 수 있다
- [ ] EarlyStopping, ModelCheckpoint를 활용할 수 있다
- [ ] W&B로 실험을 추적하고 비교할 수 있다

---

## 다음 차시 예고

### 26차시: 모델 저장과 실무 배포 준비

- 모델 저장 형식 (SavedModel, ONNX, TFLite)
- 모델 경량화 기법 (Quantization, Pruning)
- 추론 최적화와 서빙 준비
- 실무 배포를 위한 체크리스트

**학습된 모델을 실제 서비스에 적용하는 방법을 배웁니다**

---

# 부록: 전체 실습 코드

---

## A.1 CNN 모델 + Grad-CAM 전체 코드

```python
"""
딥러닝 모델 해석 - Grad-CAM 전체 실습
MNIST 또는 CIFAR-10 데이터셋 사용
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import datetime

# GPU 메모리 증가 방지
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## A.2 데이터 로드 및 전처리

```python
# CIFAR-10 데이터셋 로드
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 정규화
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 레이블 원-핫 인코딩
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## A.3 CNN 모델 정의

```python
def create_cnn_model():
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

        # Block 3 (Grad-CAM 타겟)
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_gradcam'),
        MaxPooling2D((2, 2), name='pool3'),

        # Classifier
        Flatten(),
        Dense(256, activation='relu', name='fc1'),
        Dropout(0.5),
        Dense(10, activation='softmax', name='output')
    ])

    return model

model = create_cnn_model()
model.summary()
```

---

## A.4 콜백 설정 및 학습

```python
# 로그 디렉토리
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 콜백 설정
callbacks = [
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_cifar_model.keras', save_best_only=True)
]

# 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks
)
```

---

## A.5 Grad-CAM 함수 정의

```python
def make_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    """Grad-CAM 히트맵 생성"""
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
```

---

## A.6 Grad-CAM 시각화 실행

```python
import cv2

# 테스트 이미지 선택
idx = 42
test_image = X_test[idx:idx+1]
true_label = class_names[y_test[idx][0]]

# 예측
pred = model.predict(test_image)
pred_label = class_names[np.argmax(pred)]

# Grad-CAM 히트맵 생성
heatmap = make_gradcam_heatmap(model, test_image, 'conv5_gradcam')

# 시각화
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 원본
axes[0].imshow(X_test[idx])
axes[0].set_title(f'Original\nTrue: {true_label}')
axes[0].axis('off')

# 히트맵
axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')

# 히트맵 리사이즈 및 오버레이
heatmap_resized = cv2.resize(heatmap, (32, 32))
axes[2].imshow(X_test[idx])
axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
axes[2].set_title(f'Overlay\nPred: {pred_label}')
axes[2].axis('off')

# 예측 확률
axes[3].barh(class_names, pred[0])
axes[3].set_title('Prediction Probabilities')
axes[3].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('gradcam_result.png', dpi=150)
plt.show()
```

---

## A.7 W&B 통합 버전

```python
import wandb
from wandb.integration.keras import WandbCallback

# W&B 초기화
wandb.init(
    project="cifar10-gradcam",
    config={
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "epochs": 50,
        "batch_size": 64,
        "optimizer": "adam"
    }
)

# 학습 (W&B 콜백 추가)
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        WandbCallback(save_model=True),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Grad-CAM 이미지 로깅
wandb.log({
    "gradcam_example": wandb.Image(
        'gradcam_result.png',
        caption=f"True: {true_label}, Pred: {pred_label}"
    )
})

wandb.finish()
```

---

# 수고하셨습니다!

**오늘의 핵심 메시지**:

1. 딥러닝 모델도 해석 가능합니다 (XAI)
2. Grad-CAM으로 CNN의 판단 근거를 시각화합니다
3. TensorBoard와 Callbacks로 학습을 체계적으로 관리합니다
4. W&B로 실험을 추적하고 팀과 협업합니다

**다음 시간**: 모델 저장과 실무 배포 준비
