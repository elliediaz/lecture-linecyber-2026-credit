# [19차시] 딥러닝 실습: MLP로 품질 예측 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("19차시: 딥러닝 실습 - MLP로 품질 예측")
print("Keras로 첫 딥러닝 모델을 만들어봅니다!")
print("=" * 60)
print(f"\nTensorFlow 버전: {tf.__version__}")
print()


# ============================================================
# 실습 1: 제조 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 1000

# 입력 특성: 온도, 습도, 속도
temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)

# 불량 확률 (온도, 습도 영향)
defect_prob = 0.05 + 0.03 * (temperature - 80) / 5 + 0.02 * (humidity - 40) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

# 데이터셋 구성
X = np.column_stack([temperature, humidity, speed])
y = defect

print(f"데이터 크기: {X.shape}")
print(f"불량 비율: {y.mean():.1%}")
print(f"불량: {y.sum()}개, 정상: {len(y) - y.sum()}개")
print()


# ============================================================
# 실습 2: 데이터 분할 및 정규화
# ============================================================
print("=" * 50)
print("실습 2: 데이터 분할 및 정규화")
print("=" * 50)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}개, Test: {len(X_test)}개")

# 정규화 (딥러닝에서 중요!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n정규화 전 평균: {X_train.mean(axis=0)}")
print(f"정규화 후 평균: {X_train_scaled.mean(axis=0).round(2)}")
print(f"정규화 후 표준편차: {X_train_scaled.std(axis=0).round(2)}")
print()


# ============================================================
# 실습 3: MLP 모델 구축
# ============================================================
print("=" * 50)
print("실습 3: MLP 모델 구축")
print("=" * 50)

# Sequential 모델 생성
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),  # 입력층 → 은닉층1
    Dense(8, activation='relu'),                      # 은닉층1 → 은닉층2
    Dense(1, activation='sigmoid')                    # 은닉층2 → 출력층
])

print("모델 구조:")
print("-" * 50)
model.summary()
print()


# ============================================================
# 실습 4: 모델 컴파일
# ============================================================
print("=" * 50)
print("실습 4: 모델 컴파일")
print("=" * 50)

model.compile(
    optimizer='adam',           # 최적화 알고리즘
    loss='binary_crossentropy', # 이진 분류 손실 함수
    metrics=['accuracy']        # 평가 지표
)

print("컴파일 설정:")
print(f"  - optimizer: adam")
print(f"  - loss: binary_crossentropy")
print(f"  - metrics: accuracy")
print()


# ============================================================
# 실습 5: 모델 학습
# ============================================================
print("=" * 50)
print("실습 5: 모델 학습")
print("=" * 50)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\n학습 완료!")
print(f"최종 Train Loss: {history.history['loss'][-1]:.4f}")
print(f"최종 Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"최종 Val Loss: {history.history['val_loss'][-1]:.4f}")
print(f"최종 Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()


# ============================================================
# 실습 6: 학습 곡선 시각화
# ============================================================
print("=" * 50)
print("실습 6: 학습 곡선 시각화")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss 곡선
axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_title('Loss 곡선')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy 곡선
axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[1].set_title('Accuracy 곡선')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('학습_곡선.png', dpi=100)
plt.close()
print("학습 곡선 저장: 학습_곡선.png")

# 과대적합 여부 확인
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
if final_val_loss > final_train_loss * 1.2:
    print("⚠️ 과대적합 의심: Val Loss > Train Loss * 1.2")
else:
    print("✅ 과대적합 없음: 정상 학습")
print()


# ============================================================
# 실습 7: 예측 및 평가
# ============================================================
print("=" * 50)
print("실습 7: 예측 및 평가")
print("=" * 50)

# 예측 (확률)
y_prob = model.predict(X_test_scaled, verbose=0)

# 이진 분류 (0.5 기준)
y_pred = (y_prob > 0.5).astype(int).flatten()

# 정확도
mlp_accuracy = accuracy_score(y_test, y_pred)
print(f"MLP 정확도: {mlp_accuracy:.3f}")

# 상세 리포트
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))


# ============================================================
# 실습 8: 혼동 행렬
# ============================================================
print("=" * 50)
print("실습 8: 혼동 행렬")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬:")
print(f"          예측")
print(f"          정상  불량")
print(f"실제 정상  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"     불량  {cm[1,0]:4d}  {cm[1,1]:4d}")
print()


# ============================================================
# 실습 9: RandomForest와 비교
# ============================================================
print("=" * 50)
print("실습 9: RandomForest와 비교")
print("=" * 50)

# RandomForest 학습 (정규화 불필요)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("성능 비교:")
print("-" * 40)
print(f"{'모델':<20} {'정확도':<15}")
print("-" * 40)
print(f"{'MLP (Keras)':<20} {mlp_accuracy:<15.3f}")
print(f"{'RandomForest':<20} {rf_accuracy:<15.3f}")
print("-" * 40)

if mlp_accuracy > rf_accuracy:
    print("\n★ MLP가 더 좋음")
elif rf_accuracy > mlp_accuracy:
    print("\n★ RandomForest가 더 좋음")
else:
    print("\n★ 두 모델 성능 동일")

print("\n💡 테이블 데이터에서는 ML과 DL 성능이 비슷하거나 ML이 나을 수 있음")
print()


# ============================================================
# 실습 10: 모델 저장
# ============================================================
print("=" * 50)
print("실습 10: 모델 저장")
print("=" * 50)

# 모델 저장
model.save('mlp_defect_model.keras')
print("모델 저장: mlp_defect_model.keras")

# 모델 로드 (예시)
# loaded_model = keras.models.load_model('mlp_defect_model.keras')
print()


# ============================================================
# 실습 11: 새 데이터 예측
# ============================================================
print("=" * 50)
print("실습 11: 새 데이터 예측")
print("=" * 50)

# 새로운 제조 조건
new_data = np.array([
    [90, 60, 105],  # 높은 온도, 높은 습도
    [80, 45, 100],  # 정상 조건
    [95, 55, 110]   # 매우 높은 온도
])

# 정규화
new_data_scaled = scaler.transform(new_data)

# 예측
new_probs = model.predict(new_data_scaled, verbose=0)

print("새 데이터 예측:")
print("-" * 50)
for i, (data, prob) in enumerate(zip(new_data, new_probs)):
    label = "불량" if prob > 0.5 else "정상"
    print(f"조건 {i+1}: 온도={data[0]:.0f}, 습도={data[1]:.0f}, 속도={data[2]:.0f}")
    print(f"         불량 확률: {prob[0]:.1%} → 예측: {label}")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              Keras MLP 핵심 정리                       │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 모델 구축                                           │
│     model = Sequential([                               │
│         Dense(16, activation='relu', input_shape=(3,)),│
│         Dense(8, activation='relu'),                   │
│         Dense(1, activation='sigmoid')                 │
│     ])                                                 │
│                                                        │
│  ▶ 컴파일                                              │
│     model.compile(                                     │
│         optimizer='adam',                              │
│         loss='binary_crossentropy',                    │
│         metrics=['accuracy']                           │
│     )                                                  │
│                                                        │
│  ▶ 학습                                                │
│     history = model.fit(                               │
│         X_train, y_train,                              │
│         epochs=50, batch_size=32,                      │
│         validation_split=0.2                           │
│     )                                                  │
│                                                        │
│  ▶ 예측                                                │
│     y_prob = model.predict(X_test)                     │
│     y_pred = (y_prob > 0.5).astype(int)                │
│                                                        │
│  ★ 데이터 정규화 필수! (StandardScaler)                │
│  ★ 학습 곡선으로 과대적합 감지!                         │
│  ★ 테이블 데이터는 ML과 비교 필요!                      │
│                                                        │
│  결과:                                                 │
│    MLP 정확도: {mlp_accuracy:.3f}                                │
│    RandomForest 정확도: {rf_accuracy:.3f}                        │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: AI API의 이해와 활용
""")

print("=" * 60)
print("19차시 실습 완료!")
print("Keras로 첫 딥러닝 모델을 만들었습니다!")
print("=" * 60)
