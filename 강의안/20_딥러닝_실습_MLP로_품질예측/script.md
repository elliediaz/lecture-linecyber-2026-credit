# [20차시] 딥러닝 실습: MLP로 품질 예측 - 강사 스크립트

## 강의 정보
- **차시**: 20차시 (25분)
- **유형**: 실습 중심
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 지난 시간 복습 [1.5분]

> 안녕하세요, 20차시를 시작하겠습니다.
>
> 지난 시간에 신경망의 기초 개념을 배웠습니다. 뉴런, 층, 활성화 함수, 경사하강법의 원리를 이해했죠.
>
> 오늘은 **Keras**를 사용해서 직접 신경망을 구현해봅니다. 제조 데이터로 품질 예측 모델을 만들어볼게요.

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, Keras로 MLP 모델을 구축합니다.
> 둘째, 모델을 학습하고 평가합니다.
> 셋째, 학습 곡선을 해석합니다.

---

## 전개 (19분)

### 섹션 1: Keras 소개와 모델 구축 (7min)

#### Keras란 [2min]

> **Keras**는 TensorFlow의 고수준 API입니다. 복잡한 딥러닝 코드를 간단하게 작성할 수 있어요.
>
> ```python
> from tensorflow import keras
> from tensorflow.keras import Sequential
> from tensorflow.keras.layers import Dense
> ```
>
> 이미 설치되어 있으면 바로 사용할 수 있습니다.

#### Sequential API [3min]

> *(코드 시연)*
>
> ```python
> model = Sequential([
>     Dense(64, activation='relu', input_shape=(4,)),
>     Dense(32, activation='relu'),
>     Dense(1, activation='sigmoid')
> ])
> ```
>
> Sequential은 층을 순차적으로 쌓는 방법이에요.
>
> - 첫 번째 Dense: 64개 뉴런, ReLU 활성화, 입력 4개(온도, 습도, 속도, 압력)
> - 두 번째 Dense: 32개 뉴런, ReLU
> - 마지막 Dense: 1개 뉴런, Sigmoid (불량 확률 출력)

#### 모델 컴파일 [2min]

> *(코드 시연)*
>
> ```python
> model.compile(
>     optimizer='adam',
>     loss='binary_crossentropy',
>     metrics=['accuracy']
> )
> ```
>
> **optimizer**: 가중치 업데이트 방법. adam이 가장 많이 사용돼요.
>
> **loss**: 손실 함수. 이진 분류라서 binary_crossentropy를 써요.
>
> **metrics**: 학습 중 볼 지표. 정확도를 추가했습니다.

---

### 섹션 2: 학습과 평가 (7min)

#### 데이터 전처리 [2min]

> 신경망은 **스케일링이 중요**합니다!
>
> ```python
> from sklearn.preprocessing import StandardScaler
>
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)
> X_test_scaled = scaler.transform(X_test)
> ```
>
> 특성 범위가 다르면 학습이 불안정해요. 표준화로 평균 0, 표준편차 1로 맞춰줍니다.

#### 모델 학습 [3min]

> *(코드 시연)*
>
> ```python
> history = model.fit(
>     X_train_scaled, y_train,
>     epochs=50,
>     batch_size=32,
>     validation_split=0.2
> )
> ```
>
> **epochs=50**: 전체 데이터를 50번 반복 학습
> **batch_size=32**: 한 번에 32개씩 처리
> **validation_split=0.2**: 학습 데이터의 20%를 검증용으로 분리
>
> 학습 중에 loss와 accuracy가 출력돼요. 점점 좋아지는 것을 확인할 수 있습니다.

#### 모델 평가 [2min]

> *(코드 시연)*
>
> ```python
> loss, accuracy = model.evaluate(X_test_scaled, y_test)
> print(f"테스트 정확도: {accuracy:.1%}")
> ```
>
> 테스트 데이터로 최종 성능을 확인합니다.

---

### 섹션 3: 학습 곡선과 과대적합 (5min)

#### 학습 곡선 시각화 [2min]

> *(코드 시연)*
>
> ```python
> plt.plot(history.history['loss'], label='Train')
> plt.plot(history.history['val_loss'], label='Validation')
> plt.legend()
> plt.show()
> ```
>
> 학습 곡선(Learning Curve)은 에폭에 따른 손실 변화를 보여줘요.
>
> Train과 Validation이 같이 내려가면 좋은 거예요.

#### 과대적합 탐지 [2min]

> Train Loss는 계속 내려가는데 Validation Loss가 다시 올라가면 **과대적합**입니다.
>
> 모델이 학습 데이터를 외워버린 거예요.
>
> 해결책으로 **Early Stopping**을 사용합니다.
>
> ```python
> early_stop = EarlyStopping(
>     monitor='val_loss',
>     patience=10,
>     restore_best_weights=True
> )
> ```
>
> Validation Loss가 10에폭 동안 개선되지 않으면 학습을 멈춰요.

#### ML과 비교 [1min]

> 같은 데이터로 랜덤포레스트와 비교해보면, 테이블 데이터에서는 비슷한 성능이 나오는 경우가 많아요.
>
> 딥러닝이 항상 최선은 아닙니다. 데이터와 문제에 맞는 방법을 선택하세요.

---

## 정리 (3분)

### 핵심 내용 요약 [1.5min]

> 오늘 배운 핵심 내용을 정리하면:
>
> 1. **Sequential**: 층을 순차적으로 쌓기
> 2. **Dense**: Fully Connected 층
> 3. **compile**: optimizer, loss, metrics 설정
> 4. **fit**: 학습 (epochs, batch_size)
> 5. **evaluate**: 테스트 평가
> 6. **학습 곡선**: 과대적합 탐지
> 7. **EarlyStopping**: 과대적합 방지
>
> Part III가 끝났습니다! 다음부터는 실무 활용을 배워요.

### 다음 차시 예고 [1min]

> 다음 21차시부터 **Part IV: AI 활용**이 시작됩니다.
>
> 만든 모델을 실무에서 어떻게 활용하는지 배웁니다. AI API, 웹앱, 모델 서빙 등을 다룹니다.

### 마무리 인사 [0.5분]

> Part III 완료를 축하드립니다! 수고하셨습니다!

---

## 강의 노트

### 예상 질문
1. "GPU가 없어도 되나요?"
   → 작은 모델은 CPU로 충분. 큰 모델은 Google Colab 활용

2. "에폭을 얼마로 해야 하나요?"
   → Early Stopping 사용 권장. 100 정도로 설정하고 자동 중단

3. "layers.Dense 대신 keras.layers.Dense?"
   → 같은 거. import 방식에 따라 다름

### 시간 조절 팁
- 시간 부족: ML과 비교 부분 생략
- 시간 여유: Dropout 층 추가 실습
