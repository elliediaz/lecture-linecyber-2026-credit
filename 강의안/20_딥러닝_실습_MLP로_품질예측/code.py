"""
[20차시] 딥러닝 실습: MLP로 품질 예측 - 실습 코드
학습목표: Keras로 MLP 구현, 모델 학습/평가, 학습 곡선 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# TensorFlow/Keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 메시지 최소화

try:
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ TensorFlow/Keras가 설치되지 않았습니다.")
    print("   설치: pip install tensorflow")

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 제조 데이터 생성
# ============================================================
print("=" * 50)
print("1. 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 1000

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)
pressure = np.random.normal(1.0, 0.1, n_samples)

# 불량 여부 (비선형 관계)
defect_prob = 0.1 + 0.03 * (temperature - 85) / 5 + 0.02 * np.abs(humidity - 50) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '압력': pressure,
    '불량여부': defect
})

print(df.head())
print(f"\n데이터 크기: {len(df)}")
print(f"불량 비율: {df['불량여부'].mean():.1%}")

# ============================================================
# 2. 데이터 준비
# ============================================================
print("\n" + "=" * 50)
print("2. 데이터 준비")
print("=" * 50)

X = df[['온도', '습도', '속도', '압력']]
y = df['불량여부']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")

# 스케일링 (신경망에 필수!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n▶ 스케일링 완료 (StandardScaler)")
print(f"   스케일링 전 평균: {X_train.mean().mean():.2f}")
print(f"   스케일링 후 평균: {X_train_scaled.mean():.2f}")

if KERAS_AVAILABLE:
    # ============================================================
    # 3. MLP 모델 구축
    # ============================================================
    print("\n" + "=" * 50)
    print("3. MLP 모델 구축 (Keras)")
    print("=" * 50)

    # Sequential API로 모델 생성
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),  # 입력층 + 은닉층1
        Dense(32, activation='relu'),                     # 은닉층2
        Dense(16, activation='relu'),                     # 은닉층3
        Dense(1, activation='sigmoid')                    # 출력층 (이진 분류)
    ])

    # 모델 요약
    print("\n▶ 모델 구조:")
    model.summary()

    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("\n▶ 모델 컴파일 완료")
    print("   optimizer: adam")
    print("   loss: binary_crossentropy")
    print("   metrics: accuracy")

    # ============================================================
    # 4. 모델 학습
    # ============================================================
    print("\n" + "=" * 50)
    print("4. 모델 학습")
    print("=" * 50)

    # Early Stopping 설정
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # 학습
    print("\n▶ 학습 시작...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    print(f"\n▶ 학습 완료! (총 {len(history.history['loss'])} 에폭)")

    # ============================================================
    # 5. 학습 곡선 시각화
    # ============================================================
    print("\n" + "=" * 50)
    print("5. 학습 곡선")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 곡선
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 곡선
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("▶ learning_curve.png 저장됨")

    # ============================================================
    # 6. 모델 평가
    # ============================================================
    print("\n" + "=" * 50)
    print("6. 모델 평가")
    print("=" * 50)

    # 테스트 데이터 평가
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n▶ MLP 테스트 결과:")
    print(f"   손실: {loss:.4f}")
    print(f"   정확도: {accuracy:.1%}")

    # 예측
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n▶ 분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=['정상', '불량']))

    # ============================================================
    # 7. 랜덤포레스트와 비교
    # ============================================================
    print("\n" + "=" * 50)
    print("7. MLP vs RandomForest 비교")
    print("=" * 50)

    # 랜덤포레스트 학습
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)  # 스케일링 불필요
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print(f"\n{'모델':<20} {'테스트 정확도':<15}")
    print("-" * 35)
    print(f"{'MLP (Keras)':<20} {accuracy:.1%}")
    print(f"{'RandomForest':<20} {rf_accuracy:.1%}")

    if accuracy > rf_accuracy:
        print("\n→ MLP가 더 높은 정확도!")
    else:
        print("\n→ RandomForest가 더 높은 정확도!")
        print("   (테이블 데이터에서는 ML이 효과적일 수 있음)")

else:
    print("\n⚠️ TensorFlow가 없어서 딥러닝 실습을 건너뜁니다.")
    print("   대신 RandomForest로 비교 모델 학습...")

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"\nRandomForest 정확도: {rf_accuracy:.1%}")

# ============================================================
# 8. 새로운 데이터 예측 예시
# ============================================================
print("\n" + "=" * 50)
print("8. 새로운 데이터 예측")
print("=" * 50)

# 새 데이터
new_data = np.array([[90, 55, 105, 1.05],   # 높은 온도
                     [82, 48, 98, 0.98]])    # 정상 범위

new_data_scaled = scaler.transform(new_data)

if KERAS_AVAILABLE:
    new_pred_proba = model.predict(new_data_scaled, verbose=0)
    print("▶ MLP 예측:")
    for i, (data, prob) in enumerate(zip(new_data, new_pred_proba)):
        status = "불량 위험" if prob > 0.5 else "정상"
        print(f"   샘플 {i+1}: 온도={data[0]}, 습도={data[1]} → {status} (확률: {prob[0]:.1%})")

# ============================================================
# 9. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("9. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│               Keras MLP 핵심 정리                       │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 모델 구축                                           │
│     model = Sequential([                              │
│         Dense(64, activation='relu', input_shape=(4,)),
│         Dense(32, activation='relu'),                 │
│         Dense(1, activation='sigmoid')                │
│     ])                                                 │
│                                                        │
│  ▶ 컴파일                                              │
│     model.compile(                                    │
│         optimizer='adam',                             │
│         loss='binary_crossentropy',                   │
│         metrics=['accuracy']                          │
│     )                                                  │
│                                                        │
│  ▶ 학습                                                │
│     history = model.fit(X_train, y_train,             │
│                        epochs=50,                      │
│                        validation_split=0.2,           │
│                        callbacks=[early_stop])         │
│                                                        │
│  ▶ 평가                                                │
│     loss, acc = model.evaluate(X_test, y_test)        │
│                                                        │
│  ★ 스케일링 필수! StandardScaler 사용                   │
│  ★ 학습 곡선으로 과대적합 확인                          │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: Part IV 시작 - AI API의 이해와 활용
""")
