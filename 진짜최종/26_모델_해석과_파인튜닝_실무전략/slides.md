---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [26차시] 모델 해석과 파인튜닝 실무 전략

## Grad-CAM, Callbacks, Pipeline, 전이학습

---

# 학습 목표

1. **Grad-CAM**으로 CNN 모델의 판단 근거를 시각화합니다
2. **TensorBoard**와 **Callbacks**로 학습 과정을 추적합니다
3. **joblib**과 **Pipeline**으로 모델을 저장하고 관리합니다
4. **전이학습**과 **파인튜닝**으로 사전학습 모델을 활용합니다

---

# 지난 시간 복습

- **RNN/LSTM**: 시계열 데이터 처리와 시퀀스 모델링
- **Sequential 모델**: LSTM 셀을 활용한 예측
- **과적합 방지**: Dropout, Early Stopping

**오늘**: 모델을 해석하고, 저장하며, 재활용하는 방법

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 6분 | 딥러닝 모델 해석 (Grad-CAM) |
| Part 2 | 5분 | 학습 추적 (TensorBoard, Callbacks) |
| Part 3 | 6분 | 모델 저장과 Pipeline |
| Part 4 | 6분 | 파인튜닝과 전이학습 기초 |
| 정리 | 2분 | 배포 전 체크리스트 |

---

<!-- _class: lead -->
# Part 1
## 딥러닝 모델 해석: Grad-CAM

---

# 블랙박스 문제란?

**딥러닝 모델의 특성**:

```
입력 (이미지)  →  [ 신경망 내부 ]  →  출력 (예측)
                      ?
                 무슨 일이 일어났지?
```

- 수백만 개의 파라미터
- 복잡한 비선형 변환
- **"왜 이렇게 판단했는가?"**에 답하기 어려움

---

# 해석이 필요한 이유

| 상황 | 해석 필요성 |
|:----:|:----------:|
| 의료 진단 | 왜 암으로 판정했는지 설명 필요 |
| 제조 불량 판정 | **어느 부위를 보고 불량으로 판단했는가?** |
| 금융 심사 | 거절 사유 법적 설명 의무 |
| 자율주행 | 사고 발생 시 원인 분석 |

**신뢰성**: 모델의 판단 근거를 알아야 신뢰할 수 있습니다

---

# Grad-CAM이란?

**Grad-CAM**: 기울기 가중 활성화 맵 (Gradient-weighted Class Activation Map)

```
CNN 모델 내부 구조

입력 → [Conv층들] → [마지막Conv] → [FC층] → 출력
                         ↓
                   Feature Maps
                   14x14 x 512개

"출력 → 마지막 Conv까지 역전파하여 중요도 계산"
```

---

# Grad-CAM 원리

**핵심 아이디어**: 마지막 합성곱 층의 기울기를 가중치로 사용

1. **순전파**: 입력 이미지를 모델에 통과
2. **역전파**: 타겟 클래스에서 마지막 Conv까지 기울기 계산
3. **가중치 계산**: 기울기를 공간 방향으로 평균
4. **히트맵 생성**: Feature Maps를 가중 합산
5. **업샘플링**: 원본 이미지 크기로 확대

---

# Grad-CAM 구현 (1/2)

```python
import tensorflow as tf

def make_gradcam_heatmap(model, img, last_conv_layer_name):
    """Grad-CAM 히트맵 생성"""
    # 마지막 Conv 층 출력과 모델 출력을 얻는 모델 생성
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 기울기 계산
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
```

---

# Grad-CAM 구현 (2/2)

```python
    # (계속)
    # 마지막 Conv 층에 대한 기울기
    grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling으로 가중치 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 가중 합산
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]

    # ReLU 적용 및 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()
```

---

# Grad-CAM 시각화

```python
import cv2
import numpy as np

def apply_gradcam(original_image, heatmap, alpha=0.4):
    """Grad-CAM 히트맵을 원본 이미지에 오버레이"""
    # 히트맵을 원본 크기로 리사이즈
    heatmap = cv2.resize(heatmap, (original_image.shape[1],
                                    original_image.shape[0]))

    # 컬러맵 적용 (JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 오버레이
    superimposed = heatmap_colored * alpha + original_image * (1 - alpha)
    return np.uint8(superimposed)
```

---

# Grad-CAM 결과 해석

```
원본 이미지         Grad-CAM 히트맵         오버레이
┌───────────┐      ┌───────────┐      ┌───────────┐
│           │      │   빨강=   │      │           │
│   부품    │  →   │   관심영역│  →   │  불량부위 │
│           │      │           │      │   강조    │
└───────────┘      └───────────┘      └───────────┘

해석:
- 빨간색/노란색: 예측에 크게 기여한 영역
- 파란색: 영향이 적은 영역
```

---

# 제조업 적용 사례

**불량 판정 시 활용**:

```python
# 불량 판정 근거 시각화
heatmap = make_gradcam_heatmap(model, defect_image, 'conv5')
overlay = apply_gradcam(original, heatmap)

# 결과: "스크래치 영역을 보고 불량으로 판단"
# 품질 담당자가 AI 판단 근거 확인 가능
```

**장점**:
- 오탐 원인 분석
- 모델 신뢰도 향상
- 품질 개선 포인트 발견

---

<!-- _class: lead -->
# Part 2
## 학습 추적: TensorBoard와 Callbacks

---

# 학습 모니터링의 필요성

**학습 중 확인해야 할 것들**:

| 항목 | 확인 목적 |
|:----:|:--------:|
| 손실 곡선 | 학습 진행 상태 |
| 정확도 곡선 | 성능 개선 확인 |
| 과적합 여부 | train/val 차이 |
| 기울기 분포 | vanishing/exploding |

**TensorBoard**: 이 모든 것을 시각적으로 제공

---

# TensorBoard 설정

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 로그 디렉토리 설정 (시간 기반)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard 콜백 생성
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,      # 매 에포크마다 히스토그램 기록
    write_graph=True,      # 모델 그래프 저장
    update_freq='epoch'    # 업데이트 주기
)

# 모델 학습에 콜백 적용
model.fit(X_train, y_train, callbacks=[tensorboard_callback])
```

---

# TensorBoard 실행

**터미널에서 실행**:
```bash
tensorboard --logdir=logs/fit
# 브라우저에서 http://localhost:6006 접속
```

**Jupyter Notebook에서 실행**:
```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

---

# EarlyStopping: 과적합 방지

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',         # 모니터링할 지표
    patience=10,                # 개선 없이 기다릴 에포크 수
    min_delta=0.001,            # 개선으로 인정할 최소 변화량
    restore_best_weights=True   # 최적 가중치 복원
)

model.fit(
    X_train, y_train,
    epochs=100,  # 최대 에포크
    callbacks=[early_stopping]
)
```

**과적합 방지**: validation 성능이 개선되지 않으면 자동 중단

---

# ModelCheckpoint: 최고 모델 저장

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # 저장 경로
    monitor='val_accuracy',       # 모니터링 지표
    save_best_only=True,          # 최고 성능만 저장
    mode='max',                   # 최대값 기준
    verbose=1                     # 저장 시 메시지 출력
)

# 에포크별 저장
checkpoint_epoch = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.keras',
    save_freq='epoch'
)
```

---

# ReduceLROnPlateau: 학습률 자동 조정

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # 모니터링 지표
    factor=0.5,            # 학습률 감소 비율 (절반으로)
    patience=5,            # 기다릴 에포크 수
    min_lr=1e-6,           # 최소 학습률
    verbose=1
)

# 정체기에 학습률 자동 감소
# val_loss가 5 에포크 동안 개선되지 않으면
# learning_rate = learning_rate * 0.5
```

---

# 복합 콜백 사용

```python
# 여러 콜백을 함께 사용
callbacks = [
    TensorBoard(log_dir='./logs'),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=5
    )
]

history = model.fit(X_train, y_train, callbacks=callbacks)
```

---

<!-- _class: lead -->
# Part 3
## 모델 저장과 Pipeline

---

# 왜 모델을 저장해야 하는가?

**학습의 문제점**:
```
학습: 1시간 (데이터 로드 -> 학습 -> 튜닝)
예측: 1초
```

- 매번 학습하면 시간 낭비
- 학습된 모델을 파일로 저장
- 필요할 때 로드해서 바로 사용

---

# joblib 저장/로드

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. 모델 저장
joblib.dump(model, 'quality_model.pkl')

# 3. 모델 로드
loaded_model = joblib.load('quality_model.pkl')

# 바로 예측 사용
prediction = loaded_model.predict(X_new)
```

---

# Pipeline이 필요한 이유

**문제 상황**:
```python
# 학습 시
scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 배포 시
X_new_scaled = scaler.transform(X_new)  # scaler도 저장해야 함
prediction = model.predict(X_new_scaled)
```

**전처리 객체를 따로 관리해야 하는 번거로움**

---

# Pipeline 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # 1단계: 스케일링
    ('model', RandomForestClassifier()) # 2단계: 모델
])

# 학습 (자동으로 순서대로 처리)
pipeline.fit(X_train, y_train)

# 저장 (전처리 + 모델 통째로)
joblib.dump(pipeline, 'quality_pipeline.pkl')
```

---

# Pipeline 예측

```python
# 로드
loaded_pipeline = joblib.load('quality_pipeline.pkl')

# 예측 (전처리 + 모델 한 번에)
prediction = loaded_pipeline.predict(X_new)

# 내부적으로:
# 1. scaler.transform(X_new)
# 2. model.predict(X_scaled)
```

**단일 객체로 전체 워크플로우 관리**

---

# 메타데이터 관리

```python
import json
from datetime import datetime

# 메타데이터 기록
metadata = {
    'model_name': 'quality_predictor',
    'version': 'v1.0.0',
    'created_at': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'train_accuracy': 0.95,
    'test_accuracy': 0.92,
    'features': feature_cols
}

# 저장
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

<!-- _class: lead -->
# Part 4
## 파인튜닝과 전이학습 기초

---

# 전이학습이란?

**전이학습 (Transfer Learning)**: 다른 문제에서 학습한 지식을 재활용

```
기존 방식:
[데이터 1만 장] → [처음부터 학습] → [내 모델]

전이학습:
[ImageNet 100만 장] → [사전학습 모델] → [내 데이터로 미세조정] → [고성능 모델]
```

**장점**:
- 적은 데이터로도 좋은 성능
- 학습 시간 단축
- 안정적인 초기 가중치

---

# 사전학습 모델 활용

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 사전학습된 VGG16 로드 (ImageNet 가중치)
base_model = VGG16(
    weights='imagenet',    # 사전학습 가중치
    include_top=False,     # 분류 층 제외
    input_shape=(224, 224, 3)
)

# 기존 층 동결 (가중치 고정)
base_model.trainable = False
```

---

# 새로운 분류 층 추가

```python
# 새 분류 층 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습: 새 분류 층만 학습됨
model.fit(X_train, y_train, epochs=10)
```

---

# 파인튜닝: 일부 층 학습

```python
# 기본 학습 후, 일부 층 해제하여 미세조정
base_model.trainable = True

# 마지막 4개 층만 학습 허용
for layer in base_model.layers[:-4]:
    layer.trainable = False

# 낮은 학습률로 재학습
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)
```

---

# 전이학습 전략

| 전략 | 데이터 양 | 유사도 | 방법 |
|:----:|:--------:|:------:|:----:|
| 특성 추출 | 적음 | 높음 | base 동결, 새 층만 학습 |
| 파인튜닝 | 중간 | 중간 | 상위 층 일부 학습 |
| 전체 학습 | 많음 | 낮음 | 전체 층 학습 |

**제조업 예시**:
- ImageNet 모델 → 부품 이미지 분류
- 의료 영상 모델 → X-ray 결함 검출

---

# 제조업 적용: 주조 불량 검출

```python
from tensorflow.keras.applications import ResNet50

# ResNet50 사전학습 모델
base_model = ResNet50(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
base_model.trainable = False

# 불량 검출 모델
defect_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 정상/불량
])

# 적은 데이터(수백 장)로도 높은 성능 달성 가능
```

---

<!-- _class: lead -->
# 정리
## 배포 전 체크리스트

---

# 배포 전 체크리스트

| 영역 | 확인 항목 |
|------|----------|
| **모델** | 성능 기록, 버전 태깅, Grad-CAM 검증 |
| **데이터** | 입력 형식, 범위 검증, 전처리 포함 |
| **환경** | requirements.txt, 의존성 버전 |
| **테스트** | 단위/통합 테스트, 예측 시간 측정 |

---

# 오늘 배운 내용

1. **Grad-CAM**: CNN의 판단 근거 시각화
   - 마지막 Conv 층 기울기 활용
   - 불량 판정 근거 확인

2. **TensorBoard와 Callbacks**: 학습 과정 추적
   - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

3. **joblib과 Pipeline**: 모델 저장 및 관리
   - 전처리 + 모델 통합
   - 메타데이터 기록

4. **전이학습과 파인튜닝**: 사전학습 모델 활용
   - 적은 데이터로 고성능 달성

---

# 핵심 코드 요약

```python
# Grad-CAM
heatmap = make_gradcam_heatmap(model, img, 'last_conv')

# Callbacks
callbacks = [EarlyStopping(patience=10),
             ModelCheckpoint('best.keras')]

# Pipeline 저장
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('model', RandomForestClassifier())])
joblib.dump(pipeline, 'pipeline.pkl')

# 전이학습
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False
```

---

# 체크리스트

- [ ] Grad-CAM으로 모델 판단 근거를 시각화할 수 있다
- [ ] TensorBoard로 학습 과정을 모니터링할 수 있다
- [ ] EarlyStopping, ModelCheckpoint 콜백을 활용할 수 있다
- [ ] joblib으로 모델을 저장하고 로드할 수 있다
- [ ] Pipeline으로 전처리와 모델을 통합할 수 있다
- [ ] 사전학습 모델을 활용한 전이학습을 수행할 수 있다

---

# 다음 차시 예고

## [27차시] ML 모델 서비스 배포

- Streamlit으로 웹앱 만들기
- FastAPI로 REST API 구축
- 모델 서빙과 실시간 예측
- 배포 환경 구성

**학습된 모델을 실제 서비스로 제공하는 방법을 배웁니다**

---

<!-- _class: lead -->
# 수고하셨습니다

## 모델을 해석하고, 저장하고, 재활용하세요!
