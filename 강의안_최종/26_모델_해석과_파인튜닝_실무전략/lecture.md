# [26차시] 모델 해석과 파인튜닝 실무 전략

## 학습 목표

| 번호 | 목표 |
|:----:|------|
| 1 | Grad-CAM으로 CNN 모델의 판단 근거를 시각화합니다 |
| 2 | TensorBoard와 Callbacks로 학습 과정을 추적합니다 |
| 3 | joblib과 Pipeline으로 모델을 저장하고 관리합니다 |
| 4 | 전이학습과 파인튜닝으로 사전학습 모델을 활용합니다 |

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 6분 | 딥러닝 모델 해석 (Grad-CAM) |
| Part 2 | 5분 | 학습 추적 (TensorBoard, Callbacks) |
| Part 3 | 6분 | 모델 저장과 Pipeline |
| Part 4 | 6분 | 파인튜닝과 전이학습 기초 |
| 정리 | 2분 | 배포 전 체크리스트 |

**총 25분**

---

# Part 1: 딥러닝 모델 해석 - Grad-CAM (6분)

## 1.1 블랙박스 문제란?

딥러닝 모델은 높은 성능을 보여주지만, 그 내부 동작을 이해하기 어렵다는 문제가 있습니다. 이를 **블랙박스(Black Box) 문제**라고 합니다.

```
딥러닝 모델의 블랙박스 특성:

입력 데이터           딥러닝 모델 내부              출력
                 ┌─────────────────────┐
[이미지] ────────>│   수백만 개 파라미터   │────────> [예측: 불량]
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

## 1.2 해석이 필요한 이유

**XAI (eXplainable AI)**: 인공지능의 판단 과정을 인간이 이해할 수 있도록 설명하는 기술

### 해석이 필요한 실제 상황들

| 분야 | 상황 | 해석 필요 이유 |
|:----:|:----:|:-------------:|
| 의료 | 암 진단 AI | 왜 암으로 판정했는지 의사가 확인 필요 |
| 제조 | 불량 검출 | 어떤 결함 때문인지 작업자에게 설명 |
| 금융 | 대출 심사 | 거절 시 법적 설명 의무 존재 |
| 자율주행 | 사고 발생 | 차량이 왜 그런 판단을 했는지 분석 |

### 해석의 세 가지 목적

1. **신뢰 구축**: 모델의 판단 근거를 알아야 신뢰할 수 있습니다
2. **디버깅**: 모델이 잘못 학습했는지 확인할 수 있습니다
3. **개선**: 어떤 특성이 중요한지 알면 데이터/모델을 개선할 수 있습니다

---

## 1.3 Grad-CAM 개념

**Grad-CAM (Gradient-weighted Class Activation Map)**: 마지막 합성곱 층의 기울기를 가중치로 사용하여 히트맵을 생성하는 기법입니다.

### Grad-CAM 동작 원리

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

## 1.4 Grad-CAM 구현

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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 가중 합산
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU 적용 (양수만 유지)
    heatmap = tf.maximum(heatmap, 0)

    # 정규화 (0~1)
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()
```

---

## 1.5 Grad-CAM 시각화

```python
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

    return superimposed
```

---

## 1.6 Grad-CAM 시각화 함수

```python
import matplotlib.pyplot as plt

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
    superimposed = apply_gradcam_overlay(original_uint8, heatmap, alpha=0.4)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_uint8)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Grad-CAM 히트맵
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[1].axis('off')

    # 오버레이
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Overlay - {pred_name} ({pred_prob:.1%})', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    return heatmap, superimposed
```

---

## 1.7 제조업 적용 예시

```python
# 불량 검출 모델에 Grad-CAM 적용
def analyze_defect_prediction(model, defect_image, last_conv_layer):
    """불량 판정 시 Grad-CAM으로 근거 시각화"""

    # 전처리
    img_preprocessed = preprocess_image(defect_image)

    # Grad-CAM 생성
    heatmap = make_gradcam_heatmap(model, img_preprocessed, last_conv_layer)

    # 시각화
    overlay = apply_gradcam_overlay(defect_image, heatmap)

    # 결과 분석
    # - 빨간색 영역: 불량으로 판단한 근거 영역
    # - 품질 담당자가 AI 판단 검증 가능

    return overlay, heatmap

# 사용 예시
# overlay, heatmap = analyze_defect_prediction(model, defect_img, 'conv5')
# print("스크래치 영역을 보고 불량으로 판단")
```

**제조업 활용 장점**:
- 오탐 원인 분석 가능
- 모델 신뢰도 향상
- 품질 개선 포인트 발견

---

# Part 2: 학습 추적 - TensorBoard와 Callbacks (5분)

## 2.1 학습 모니터링의 필요성

딥러닝 모델 학습 시 다양한 지표를 추적해야 합니다:

| 추적 항목 | 확인 목적 |
|:--------:|:--------:|
| Loss 곡선 | 학습 진행 상태, 수렴 여부 |
| Accuracy 곡선 | 성능 개선 추이 |
| Train vs Val 차이 | 과적합 여부 |
| 기울기 분포 | Vanishing/Exploding Gradient |
| 학습률 변화 | 스케줄링 동작 확인 |

---

## 2.2 TensorBoard 설정

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
    update_freq='epoch'         # 업데이트 주기 ('batch' 또는 'epoch')
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

## 2.3 TensorBoard 실행

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
```

---

## 2.4 EarlyStopping - 과적합 방지

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
...
Epoch 15: val_loss = 0.4500  (개선 없음, patience=10)
→ 학습 종료, Epoch 4의 가중치로 복원
```

---

## 2.5 ModelCheckpoint - 모델 저장

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

## 2.6 ReduceLROnPlateau - 학습률 자동 감소

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
```

---

## 2.7 복합 콜백 사용 예시

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
        patience=10,
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

# Part 3: 모델 저장과 Pipeline (6분)

## 3.1 왜 모델을 저장해야 하는가?

학습에는 시간이 오래 걸리지만, 예측은 빠릅니다.

```
학습: 1시간 (데이터 로드 -> 학습 -> 튜닝)
예측: 1초
```

모델 저장의 이점:
- 매번 학습하는 시간 낭비를 방지합니다
- 학습된 모델을 파일로 저장합니다
- 필요할 때 로드해서 바로 사용 가능합니다

---

## 3.2 모델 저장 방법 비교

| 방법 | 장점 | 단점 |
|-----|------|------|
| **joblib** | 대용량 배열 효율적, sklearn 권장 | Python 전용 |
| **pickle** | Python 표준 | 대용량 비효율 |
| **ONNX** | 언어 중립, 배포 최적화 | 변환 필요 |

sklearn 모델은 joblib 사용을 권장합니다.

---

## 3.3 joblib 저장/로드

```python
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 저장 디렉토리 생성
MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"학습 정확도: {model.score(X_train, y_train):.2%}")

# 2. 모델 저장
model_path = os.path.join(MODEL_DIR, 'quality_model.pkl')
joblib.dump(model, model_path)
print(f"모델 저장: {model_path}")

# 파일 크기 확인
model_size = os.path.getsize(model_path)
print(f"모델 크기: {model_size/1024:.1f} KB")

# 3. 모델 로드
loaded_model = joblib.load(model_path)

# 4. 검증
print(f"로드 후 정확도: {loaded_model.score(X_test, y_test):.2%}")
```

---

## 3.4 압축 저장

```python
# 압축 저장
compressed_path = os.path.join(MODEL_DIR, 'model_compressed.pkl.gz')
joblib.dump(model, compressed_path, compress=3)

compressed_size = os.path.getsize(compressed_path)
print(f"압축 전: {model_size/1024:.1f} KB")
print(f"압축 후: {compressed_size/1024:.1f} KB")
print(f"압축률: {(1 - compressed_size/model_size)*100:.1f}%")
```

### 압축 옵션

| 레벨 | 특징 |
|------|------|
| 0 | 압축 없음 (빠름, 용량 큼) |
| 3 | 일반적인 선택 |
| 9 | 최대 압축 (느림, 용량 작음) |

대용량 모델은 압축 저장을 권장합니다.

---

## 3.5 Pipeline이 필요한 이유

문제 상황:
```python
# 학습 시
scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 배포 시
X_new_scaled = scaler.transform(X_new)  # scaler도 저장해야 합니다
prediction = model.predict(X_new_scaled)
```

전처리 객체를 따로 관리해야 하는 번거로움이 발생합니다.

---

## 3.6 Pipeline이란?

전처리 + 모델을 하나로 묶은 객체입니다.

```
[원본 데이터] -> [전처리] -> [모델] -> [예측]
                +-------- Pipeline --------+
```

Pipeline 장점:
- 한 번에 저장/로드 가능합니다
- 데이터 누수를 방지합니다
- 코드가 간결해집니다

---

## 3.7 Pipeline 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # 1단계: 스케일링
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # 2단계: 모델
])

# 학습 (자동으로 순서대로 처리)
pipeline.fit(X_train, y_train)

# 예측 (자동으로 스케일링 후 예측)
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)

print(f"학습 정확도: {train_acc:.2%}")
print(f"테스트 정확도: {test_acc:.2%}")
```

---

## 3.8 Pipeline 저장/로드

```python
# 저장 (전처리 + 모델 통째로)
pipeline_path = os.path.join(MODEL_DIR, 'quality_pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"Pipeline 저장: {pipeline_path}")

# 로드
loaded_pipeline = joblib.load(pipeline_path)

# 검증
loaded_acc = loaded_pipeline.score(X_test, y_test)
print(f"로드 후 정확도: {loaded_acc:.2%}")

# 새 데이터 예측 (전처리 자동 적용)
prediction = loaded_pipeline.predict(X_new)
```

전처리 객체를 따로 관리할 필요가 없습니다.

---

## 3.9 메타데이터 관리

```python
import json
from datetime import datetime
import sklearn

# 메타데이터 기록
metadata = {
    'model_name': 'quality_predictor',
    'version': 'v1.0.0',
    'created_at': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'features': feature_cols,
    'target': 'quality',
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test)
}

# 모델 번들 저장
model_bundle = {
    'pipeline': pipeline,
    'metadata': metadata
}

bundle_path = os.path.join(MODEL_DIR, 'model_bundle.pkl')
joblib.dump(model_bundle, bundle_path)
print(f"모델 번들 저장: {bundle_path}")

# 메타데이터 JSON 별도 저장 (가독성)
metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"메타데이터 저장: {metadata_path}")
```

메타데이터를 함께 저장하면 모델 이력 관리가 용이합니다.

---

# Part 4: 파인튜닝과 전이학습 기초 (6분)

## 4.1 전이학습이란?

**전이학습 (Transfer Learning)**: 다른 문제에서 학습한 지식을 재활용하는 기법

```
기존 방식:
[데이터 1만 장] → [처음부터 학습] → [내 모델]

전이학습:
[ImageNet 100만 장] → [사전학습 모델] → [내 데이터로 미세조정] → [고성능 모델]
```

### 전이학습의 장점

1. **적은 데이터**: 수백 장으로도 좋은 성능 달성
2. **빠른 학습**: 이미 학습된 특징 활용
3. **안정적 초기화**: 검증된 가중치에서 시작

---

## 4.2 사전학습 모델 활용

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

# 사전학습된 VGG16 로드 (ImageNet 가중치)
base_model = VGG16(
    weights='imagenet',     # 사전학습 가중치
    include_top=False,      # 분류 층 제외
    input_shape=(224, 224, 3)
)

# 기존 층 동결 (가중치 고정)
base_model.trainable = False

print(f"VGG16 층 수: {len(base_model.layers)}")
print(f"학습 가능 파라미터: {base_model.trainable_variables}")
```

---

## 4.3 새로운 분류 층 추가

```python
# 새 분류 층 추가
num_classes = 2  # 예: 정상/불량

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 학습: 새 분류 층만 학습됨
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=3)]
)
```

---

## 4.4 파인튜닝: 일부 층 학습

```python
# 기본 학습 후, 일부 층 해제하여 미세조정
base_model.trainable = True

# 마지막 4개 층만 학습 허용
for layer in base_model.layers[:-4]:
    layer.trainable = False

# 학습 가능한 층 확인
trainable_layers = [l.name for l in base_model.layers if l.trainable]
print(f"학습 가능 층: {trainable_layers}")

# 낮은 학습률로 재학습 (기존 가중치 보존)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 파인튜닝
history_finetune = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=2)]
)
```

---

## 4.5 전이학습 전략

| 전략 | 데이터 양 | 유사도 | 방법 |
|:----:|:--------:|:------:|:----:|
| 특성 추출 | 적음 | 높음 | base 동결, 새 층만 학습 |
| 파인튜닝 | 중간 | 중간 | 상위 층 일부 학습 |
| 전체 학습 | 많음 | 낮음 | 전체 층 학습 |

### 데이터 양/유사도에 따른 전략 선택

```
         데이터 많음
              │
    전체 학습 ─┼─ 파인튜닝
              │
    특성 추출 ─┼─ 파인튜닝
              │
         데이터 적음
    유사도 낮음 ─────── 유사도 높음
```

---

## 4.6 제조업 적용: 불량 검출

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

# ResNet50 사전학습 모델
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 불량 검출 모델
defect_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])

defect_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 적은 데이터(수백 장)로도 높은 성능 달성 가능
print("ImageNet 사전학습 모델로 주조 불량 검출 가능")
```

---

## 4.7 다양한 사전학습 모델

```python
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101,
    MobileNetV2,
    EfficientNetB0
)

# 모델별 특징
models_info = {
    'VGG16': {'params': '138M', 'top1_acc': '71.3%', 'size': '528MB'},
    'ResNet50': {'params': '25.6M', 'top1_acc': '74.9%', 'size': '98MB'},
    'MobileNetV2': {'params': '3.5M', 'top1_acc': '71.3%', 'size': '14MB'},
    'EfficientNetB0': {'params': '5.3M', 'top1_acc': '77.1%', 'size': '29MB'}
}

# 제조업 추천: MobileNetV2 (경량), EfficientNetB0 (성능/효율 균형)
```

| 모델 | 파라미터 | 용량 | 추천 상황 |
|:----:|:--------:|:----:|:--------:|
| VGG16 | 138M | 528MB | 학습용, 해석 용이 |
| ResNet50 | 25.6M | 98MB | 범용, 성능 우수 |
| MobileNetV2 | 3.5M | 14MB | 경량, 엣지 배포 |
| EfficientNetB0 | 5.3M | 29MB | 효율적, 권장 |

---

# 정리: 배포 전 체크리스트

## 체크리스트 항목

| 영역 | 확인 항목 |
|------|----------|
| **모델** | 성능 기록, 버전 태깅, Grad-CAM 검증 |
| **데이터** | 입력 형식, 범위 검증, 전처리 포함 |
| **환경** | requirements.txt, 의존성 버전 |
| **테스트** | 단위/통합 테스트, 예측 시간 측정 |

---

## 핵심 정리

### Part 1: Grad-CAM
- CNN의 판단 근거를 히트맵으로 시각화
- 마지막 Conv 층의 기울기 활용
- 제조업: 불량 판정 근거 확인

### Part 2: TensorBoard와 Callbacks
- TensorBoard로 학습 곡선 모니터링
- EarlyStopping: 과적합 방지
- ModelCheckpoint: 최고 모델 저장
- ReduceLROnPlateau: 학습률 자동 조정

### Part 3: joblib과 Pipeline
- joblib.dump()/load()로 모델 저장/로드
- Pipeline으로 전처리 + 모델 통합
- 메타데이터 함께 저장

### Part 4: 전이학습과 파인튜닝
- 사전학습 모델(VGG, ResNet) 활용
- 층 동결 후 새 분류 층 추가
- 파인튜닝: 일부 층만 학습

---

## 핵심 코드 요약

```python
# Grad-CAM
heatmap = make_gradcam_heatmap(model, img, 'last_conv')
overlay = apply_gradcam_overlay(original, heatmap)

# Callbacks
callbacks = [
    TensorBoard(log_dir='./logs'),
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Pipeline 저장
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.pkl')

# 전이학습
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False
model = Sequential([base_model, Flatten(), Dense(2, activation='softmax')])
```

---

## 다음 차시 예고

### 27차시: ML 모델 서비스 배포

학습 내용:
- Streamlit으로 웹앱 만들기
- FastAPI로 REST API 구축
- 모델 서빙과 실시간 예측
- 배포 환경 구성

학습된 모델을 실제 서비스로 제공하는 방법을 학습합니다.
