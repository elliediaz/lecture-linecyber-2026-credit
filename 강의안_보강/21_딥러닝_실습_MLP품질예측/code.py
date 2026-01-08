"""
[21차시] 딥러닝 실습: MLP로 품질 예측 - 실습 코드

학습 목표:
1. Keras 프레임워크의 기본 사용법을 익힌다
2. MLP 모델을 구현하고 학습시킨다
3. 학습 결과를 분석하고 개선한다

실습 환경: Python 3.8+, TensorFlow 2.x, scikit-learn
데이터셋: Breast Cancer Wisconsin (sklearn 내장)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# TensorFlow/Keras 임포트
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 숨기기

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    print(f"TensorFlow 버전: {tf.__version__}")
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow가 설치되지 않았습니다.")
    print("설치: pip install tensorflow")
    KERAS_AVAILABLE = False

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[21차시] 딥러닝 실습: MLP로 품질 예측")
print("=" * 60)

# ============================================================
# 1. Breast Cancer 데이터 로드
# ============================================================
print("\n" + "=" * 60)
print("1. Breast Cancer 데이터 로드")
print("=" * 60)

# Breast Cancer 데이터 로드
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# DataFrame으로 변환
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

print("\n[Breast Cancer 데이터셋]")
print(f"  총 샘플 수: {len(df)}")
print(f"  특성 수: {len(cancer.feature_names)}")
print(f"  클래스: {cancer.target_names} (0=악성, 1=양성)")
print(f"  양성(benign) 비율: {y.mean():.2%}")

print("\n특성 목록 (상위 10개):")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  {i+1}. {name}")
print("  ...")

print("\n데이터 통계:")
print(df.describe().round(2).iloc[:, :5])

# ============================================================
# 2. 데이터 전처리
# ============================================================
print("\n" + "=" * 60)
print("2. 데이터 전처리")
print("=" * 60)

# 특성과 타겟 분리
feature_cols = cancer.feature_names.tolist()

print("\n정규화 전:")
print(f"  mean radius 범위: {X[:, 0].min():.1f} ~ {X[:, 0].max():.1f}")
print(f"  mean area 범위: {X[:, 3].min():.1f} ~ {X[:, 3].max():.1f}")

# 정규화 (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n정규화 후:")
print(f"  mean radius 범위: {X_scaled[:, 0].min():.2f} ~ {X_scaled[:, 0].max():.2f}")
print(f"  mean area 범위: {X_scaled[:, 3].min():.2f} ~ {X_scaled[:, 3].max():.2f}")
print(f"  평균: {X_scaled.mean():.6f}, 표준편차: {X_scaled.std():.4f}")

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n데이터 분할:")
print(f"  학습 세트: {len(X_train)}개")
print(f"  테스트 세트: {len(X_test)}개")
print(f"  학습 양성 비율: {y_train.mean():.2%}")
print(f"  테스트 양성 비율: {y_test.mean():.2%}")

# ============================================================
# 3. 기본 MLP 모델 생성
# ============================================================
print("\n" + "=" * 60)
print("3. 기본 MLP 모델 생성")
print("=" * 60)

if KERAS_AVAILABLE:
    # 기본 모델 생성
    model_basic = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 컴파일
    model_basic.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n모델 구조:")
    model_basic.summary()

    # 파라미터 수 계산
    print("\n파라미터 수 계산:")
    print(f"  입력(30) -> Dense(64): 30x64 + 64 = {30*64 + 64}")
    print(f"  Dense(64) -> Dense(32): 64x32 + 32 = {64*32 + 32}")
    print(f"  Dense(32) -> Dense(1): 32x1 + 1 = {32*1 + 1}")
    print(f"  총 파라미터: {30*64+64 + 64*32+32 + 32*1+1}")
else:
    print("Keras를 사용할 수 없습니다.")

# ============================================================
# 4. Dropout과 BatchNormalization 추가
# ============================================================
print("\n" + "=" * 60)
print("4. Dropout과 BatchNormalization 추가")
print("=" * 60)

if KERAS_AVAILABLE:
    # 개선된 모델
    model = Sequential([
        # 첫 번째 은닉층
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        # 두 번째 은닉층
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # 세 번째 은닉층
        Dense(16, activation='relu'),
        Dropout(0.1),

        # 출력층
        Dense(1, activation='sigmoid')
    ])

    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n개선된 모델 구조:")
    model.summary()

# ============================================================
# 5. 콜백 설정
# ============================================================
print("\n" + "=" * 60)
print("5. 콜백 설정")
print("=" * 60)

if KERAS_AVAILABLE:
    # EarlyStopping: 검증 손실이 개선되지 않으면 조기 종료
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # ModelCheckpoint: 최적 모델 저장
    checkpoint = ModelCheckpoint(
        'best_cancer_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    # ReduceLROnPlateau: 학습 정체 시 학습률 감소
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    callbacks = [early_stop, checkpoint, reduce_lr]
    print("콜백 설정 완료:")
    print("  - EarlyStopping: patience=15")
    print("  - ModelCheckpoint: best_cancer_model.keras")
    print("  - ReduceLROnPlateau: factor=0.5, patience=5")

# ============================================================
# 6. 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("6. 모델 학습")
print("=" * 60)

if KERAS_AVAILABLE:
    print("\n학습 시작...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n학습 완료!")
    print(f"  실제 에포크 수: {len(history.history['loss'])}")
    print(f"  최종 학습 손실: {history.history['loss'][-1]:.4f}")
    print(f"  최종 검증 손실: {history.history['val_loss'][-1]:.4f}")

# ============================================================
# 7. 학습 곡선 시각화
# ============================================================
print("\n" + "=" * 60)
print("7. 학습 곡선 시각화")
print("=" * 60)

if KERAS_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 손실 곡선
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 정확도 곡선
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 'learning_curves.png' 저장됨")

# ============================================================
# 8. 모델 평가
# ============================================================
print("\n" + "=" * 60)
print("8. 모델 평가")
print("=" * 60)

if KERAS_AVAILABLE:
    # 테스트 세트 평가
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n테스트 결과:")
    print(f"  손실: {loss:.4f}")
    print(f"  정확도: {accuracy:.2%}")

    # 예측
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    # AUC 점수
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"  AUC: {auc_score:.4f}")

    # 분류 보고서
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=['악성(Malignant)', '양성(Benign)']))

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n혼동 행렬:")
    print(f"  TN (악성->악성): {cm[0, 0]}")
    print(f"  FP (악성->양성): {cm[0, 1]}")
    print(f"  FN (양성->악성): {cm[1, 0]}")
    print(f"  TP (양성->양성): {cm[1, 1]}")

# ============================================================
# 9. 혼동 행렬 시각화
# ============================================================
print("\n" + "=" * 60)
print("9. 혼동 행렬 시각화")
print("=" * 60)

if KERAS_AVAILABLE:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    # 레이블
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Malignant', 'Predicted: Benign'])
    ax.set_yticklabels(['Actual: Malignant', 'Actual: Benign'])

    # 값 표시
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          fontsize=20, fontweight='bold')

    ax.set_title('Confusion Matrix', fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 'confusion_matrix.png' 저장됨")

# ============================================================
# 10. 모델 저장 및 로드
# ============================================================
print("\n" + "=" * 60)
print("10. 모델 저장 및 로드")
print("=" * 60)

if KERAS_AVAILABLE:
    # 모델 저장
    model.save('cancer_model.keras')
    print("모델 저장 완료: cancer_model.keras")

    # 모델 로드
    loaded_model = load_model('cancer_model.keras')
    print("모델 로드 완료")

    # 로드한 모델로 예측
    y_pred_loaded = loaded_model.predict(X_test[:5], verbose=0)
    print("\n로드한 모델 예측 (처음 5개):")
    for i in range(5):
        status = "양성(Benign)" if y_pred_loaded[i, 0] > 0.5 else "악성(Malignant)"
        actual = "양성(Benign)" if y_test[i] == 1 else "악성(Malignant)"
        print(f"  샘플 {i+1}: 예측={y_pred_loaded[i, 0]:.4f} ({status}), 실제={actual}")

# ============================================================
# 11. 하이퍼파라미터 실험
# ============================================================
print("\n" + "=" * 60)
print("11. 하이퍼파라미터 실험")
print("=" * 60)

if KERAS_AVAILABLE:
    # 다양한 구조 테스트
    architectures = [
        {'hidden': [32], 'dropout': [0.2]},
        {'hidden': [64, 32], 'dropout': [0.3, 0.2]},
        {'hidden': [128, 64, 32], 'dropout': [0.3, 0.2, 0.1]},
    ]

    results = []
    print("\n다양한 구조 테스트:")

    for i, arch in enumerate(architectures):
        # 모델 생성
        model_exp = Sequential()
        model_exp.add(Dense(arch['hidden'][0], activation='relu',
                           input_shape=(X_train.shape[1],)))
        model_exp.add(Dropout(arch['dropout'][0]))

        for j in range(1, len(arch['hidden'])):
            model_exp.add(Dense(arch['hidden'][j], activation='relu'))
            model_exp.add(Dropout(arch['dropout'][j]))

        model_exp.add(Dense(1, activation='sigmoid'))

        model_exp.compile(optimizer='adam', loss='binary_crossentropy',
                         metrics=['accuracy'])

        # 빠른 학습
        history_exp = model_exp.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # 평가
        _, acc = model_exp.evaluate(X_test, y_test, verbose=0)
        results.append({
            'structure': str(arch['hidden']),
            'accuracy': acc,
            'val_loss': min(history_exp.history['val_loss'])
        })
        print(f"  구조 {arch['hidden']}: 정확도={acc:.2%}")

    # 최적 구조
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n최적 구조: {best_result['structure']}")
    print(f"  정확도: {best_result['accuracy']:.2%}")

# ============================================================
# 12. L2 정규화 적용
# ============================================================
print("\n" + "=" * 60)
print("12. L2 정규화 적용")
print("=" * 60)

if KERAS_AVAILABLE:
    # L2 정규화 모델
    model_l2 = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],),
              kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model_l2.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("L2 정규화 모델 학습...")
    history_l2 = model_l2.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )

    _, acc_l2 = model_l2.evaluate(X_test, y_test, verbose=0)
    print(f"L2 정규화 모델 정확도: {acc_l2:.2%}")

# ============================================================
# 13. 배치 크기 비교
# ============================================================
print("\n" + "=" * 60)
print("13. 배치 크기 비교")
print("=" * 60)

if KERAS_AVAILABLE:
    batch_sizes = [16, 32, 64, 128]
    batch_results = []

    print("배치 크기별 학습:")
    for bs in batch_sizes:
        model_bs = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model_bs.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['accuracy'])

        history_bs = model_bs.fit(
            X_train, y_train,
            epochs=30,
            batch_size=bs,
            validation_split=0.2,
            verbose=0
        )

        _, acc = model_bs.evaluate(X_test, y_test, verbose=0)
        batch_results.append({'batch_size': bs, 'accuracy': acc})
        print(f"  batch_size={bs}: 정확도={acc:.2%}")

# ============================================================
# 14. 새로운 데이터 예측
# ============================================================
print("\n" + "=" * 60)
print("14. 새로운 데이터 예측")
print("=" * 60)

if KERAS_AVAILABLE:
    # 테스트 데이터에서 샘플 추출 (새로운 환자 시뮬레이션)
    sample_indices = [0, 10, 50]

    print("\n새로운 환자 예측 결과:")
    print("-" * 60)
    for idx in sample_indices:
        # 원본 데이터로 정보 표시
        sample_original = X[idx + len(X_train)]  # 테스트 데이터 인덱스
        sample_scaled = X_test[idx:idx+1]

        prediction = model.predict(sample_scaled, verbose=0)
        pred_class = "양성(Benign)" if prediction[0, 0] > 0.5 else "악성(Malignant)"
        actual_class = "양성(Benign)" if y_test[idx] == 1 else "악성(Malignant)"

        print(f"환자 {idx+1}:")
        print(f"  mean radius: {sample_original[0]:.2f}")
        print(f"  mean texture: {sample_original[1]:.2f}")
        print(f"  mean area: {sample_original[3]:.2f}")
        print(f"  예측 확률: {prediction[0, 0]:.2%} -> {pred_class}")
        print(f"  실제: {actual_class}")
        print()

# ============================================================
# 15. 임계값 조정
# ============================================================
print("\n" + "=" * 60)
print("15. 임계값 조정")
print("=" * 60)

if KERAS_AVAILABLE:
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("임계값별 성능:")

    for thresh in thresholds:
        y_pred_thresh = (y_pred_prob > thresh).astype(int).ravel()

        # 정밀도, 재현율 계산
        tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
        fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
        fn = np.sum((y_pred_thresh == 0) & (y_test == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"  임계값={thresh}: 정밀도={precision:.2%}, 재현율={recall:.2%}, F1={f1:.2%}")

# ============================================================
# 16. ML vs DL 비교
# ============================================================
print("\n" + "=" * 60)
print("16. ML vs DL 비교")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)

print("\n모델 비교:")
print(f"  Logistic Regression: {lr_acc:.2%}")
print(f"  Random Forest: {rf_acc:.2%}")
if KERAS_AVAILABLE:
    print(f"  MLP (Keras): {accuracy:.2%}")

print("\n결론:")
if KERAS_AVAILABLE:
    if accuracy > rf_acc and accuracy > lr_acc:
        print("  -> 이 데이터에서는 MLP가 가장 좋은 성능!")
    else:
        print("  -> 정형 데이터에서는 전통적 ML도 좋은 성능")
        print("  -> 데이터 크기, 특성에 따라 적절한 모델 선택 필요")

# ============================================================
# 17. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("17. 핵심 정리")
print("=" * 60)

print("""
[21차시 핵심 정리]

1. Keras 기초
   - Sequential: 층을 순서대로 쌓는 모델
   - Dense: 완전연결층 (units, activation)
   - compile: optimizer, loss, metrics 설정

2. MLP 구현 워크플로우
   (1) 데이터 준비 (정규화 필수!)
   (2) 모델 생성 (Sequential + Dense)
   (3) 컴파일 (optimizer='adam')
   (4) 학습 (fit + callbacks)
   (5) 평가 (evaluate, predict)

3. 과적합 방지
   - Dropout: 무작위 노드 비활성화
   - BatchNormalization: 층 출력 정규화
   - EarlyStopping: 적절한 시점 종료
   - L2 정규화: 가중치 크기 제한

4. 학습 곡선 해석
   - Train 손실 감소, Val 손실 감소: 정상 학습
   - Train 손실 감소, Val 손실 증가: 과적합 -> 조기 종료

5. 하이퍼파라미터
   - 은닉층: 1~3개, 노드: 32~256
   - Dropout: 0.2~0.5
   - 학습률: 0.0001~0.01
   - 배치 크기: 32~128

6. 사용한 데이터셋
   - Breast Cancer Wisconsin (sklearn)
   - 569개 샘플, 30개 특성
   - 이진 분류: 악성(Malignant) vs 양성(Benign)
""")

print("\n다음 차시 예고: 딥러닝 심화 (CNN, RNN)")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
