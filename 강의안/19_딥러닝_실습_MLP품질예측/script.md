# [19차시] 딥러닝 실습: MLP로 품질 예측 - 강사 스크립트

## 강의 정보
- **차시**: 19차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 19차시를 시작하겠습니다.
>
> 지난 시간에 딥러닝의 기초를 배웠습니다. 뉴런, 층, 활성화 함수, 순전파와 역전파를 배웠죠.
>
> 오늘은 실제 딥러닝 프레임워크인 **Keras**를 사용해서 품질 예측 모델을 만들어봅니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, Keras로 신경망을 구축합니다.
> 둘째, MLP 모델을 학습합니다.
> 셋째, 제조 품질 예측에 적용합니다.

---

### 핵심 내용 (8분)

#### Keras 소개 [1.5min]

> **Keras**는 딥러닝을 쉽게 만들 수 있는 고수준 API입니다.
>
> TensorFlow 안에 포함되어 있어요.
>
> ```python
> from tensorflow import keras
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense
> ```
>
> 몇 줄의 코드로 신경망을 만들 수 있어서 입문자에게 좋습니다.

#### Sequential 모델 [2min]

> Keras에서 모델을 만드는 가장 쉬운 방법은 **Sequential**입니다.
>
> 층을 순차적으로 쌓는 방식이에요.
>
> ```python
> model = Sequential([
>     Dense(8, activation='relu', input_shape=(3,)),
>     Dense(4, activation='relu'),
>     Dense(1, activation='sigmoid')
> ])
> ```
>
> **Dense**는 완전 연결층이에요. 8, 4, 1은 각 층의 뉴런 개수입니다.
>
> 마지막에 sigmoid를 쓰면 0과 1 사이 값이 나와서 이진 분류에 적합해요.

#### 컴파일과 학습 [2min]

> 모델을 만들었으면 **compile**로 설정하고 **fit**으로 학습해요.
>
> ```python
> model.compile(
>     optimizer='adam',
>     loss='binary_crossentropy',
>     metrics=['accuracy']
> )
>
> model.fit(X_train, y_train, epochs=50, batch_size=32)
> ```
>
> **optimizer**는 adam을 많이 씁니다. 학습률을 자동 조절해줘요.
>
> **epochs**는 전체 데이터를 몇 번 볼지, **batch_size**는 한 번에 몇 개씩 볼지예요.

#### 학습 곡선 [1.5min]

> 학습 과정을 보려면 **학습 곡선**을 그려요.
>
> Loss가 내려가고 Accuracy가 올라가면 잘 학습되는 거예요.
>
> 중요한 건 Train과 Validation을 함께 보는 겁니다.
>
> Train 손실은 내려가는데 Val 손실이 올라가면 **과대적합**이에요. 이럴 땐 조기 중단하거나 모델을 단순화해야 합니다.

#### 손실 함수 선택 [1min]

> 문제 유형에 따라 손실 함수가 달라요.
>
> - 이진 분류: binary_crossentropy
> - 다중 분류: categorical_crossentropy
> - 회귀: mse
>
> 오늘은 불량/정상 이진 분류니까 binary_crossentropy를 씁니다.

---

## 실습편 (15-20분)

### 실습 소개 [2min]

> 이제 실습 시간입니다. Keras로 제조 불량 예측 모델을 만듭니다.
>
> **실습 목표**입니다.
> 1. Sequential로 MLP를 구축합니다.
> 2. 불량 여부를 이진 분류합니다.
> 3. 학습 곡선을 분석합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import tensorflow as tf
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense
> ```

### 실습 1: 데이터 생성 [2min]

> 첫 번째 실습입니다. 제조 불량 데이터를 생성합니다.
>
> 온도, 습도, 속도를 입력으로 하고 불량 여부를 예측합니다.
>
> 온도가 높고 습도가 높을수록 불량 확률이 높게 설정했어요.

### 실습 2: 데이터 전처리 [2min]

> 두 번째 실습입니다. 딥러닝에서 중요한 **정규화**를 합니다.
>
> ```python
> from sklearn.preprocessing import StandardScaler
>
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)
> X_test_scaled = scaler.transform(X_test)
> ```
>
> 입력값의 스케일이 다르면 학습이 잘 안 돼요. 평균 0, 분산 1로 맞춰줍니다.

### 실습 3: 모델 구축 [3min]

> 세 번째 실습입니다. Sequential로 모델을 만듭니다.
>
> ```python
> model = Sequential([
>     Dense(16, activation='relu', input_shape=(3,)),
>     Dense(8, activation='relu'),
>     Dense(1, activation='sigmoid')
> ])
> ```
>
> 입력 3개, 은닉층 16뉴런, 은닉층 8뉴런, 출력 1개입니다.
>
> model.summary()로 구조를 확인할 수 있어요.

### 실습 4: 컴파일 [1min]

> 네 번째 실습입니다. compile로 설정합니다.
>
> ```python
> model.compile(
>     optimizer='adam',
>     loss='binary_crossentropy',
>     metrics=['accuracy']
> )
> ```

### 실습 5: 학습 [3min]

> 다섯 번째 실습입니다. fit으로 학습합니다.
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
> validation_split=0.2로 학습 데이터의 20%를 검증용으로 씁니다.
>
> history에 학습 기록이 저장돼요.

### 실습 6: 학습 곡선 [2min]

> 여섯 번째 실습입니다. 학습 곡선을 그립니다.
>
> Loss가 내려가고 있나요? Train과 Val이 비슷하게 움직이나요?
>
> Val 손실이 올라가면 과대적합입니다.

### 실습 7: 예측 및 평가 [2min]

> 일곱 번째 실습입니다. predict로 예측하고 평가합니다.
>
> ```python
> y_prob = model.predict(X_test_scaled)
> y_pred = (y_prob > 0.5).astype(int)
> ```
>
> classification_report로 정밀도, 재현율을 확인합니다.

### 실습 8: RF와 비교 [2min]

> 마지막 실습입니다. RandomForest와 비교해봅니다.
>
> 테이블 데이터에서는 딥러닝이 항상 좋은 건 아니에요. 오히려 ML이 더 나을 수 있어요.
>
> 데이터와 문제에 맞게 선택하는 게 중요합니다.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **Keras**로 딥러닝 모델을 쉽게 만들 수 있습니다.
>
> **Sequential**로 층을 순차적으로 쌓고, **Dense**로 완전 연결층을 만들어요.
>
> **compile**에서 optimizer, loss, metrics를 설정하고, **fit**으로 학습합니다.
>
> **학습 곡선**을 보면서 과대적합을 감지하세요.

#### 다음 차시 예고 [1min]

> 다음 20차시에서는 **AI API의 이해와 활용**을 배웁니다.
>
> 직접 모델을 만드는 것도 좋지만, 만들어진 AI를 API로 활용하는 것도 중요해요. 외부 AI 서비스를 호출하는 방법을 알아봅니다.

#### 마무리 [0.5min]

> Keras로 첫 딥러닝 모델을 만들었습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- TensorFlow/Keras 설치 확인

### 주의사항
- 데이터 정규화 필수 강조
- 학습 곡선 해석 방법 설명
- ML vs DL 비교 (만능 아님)

### 예상 질문
1. "TensorFlow 설치가 안 돼요"
   → pip install tensorflow, 버전 확인

2. "Loss가 안 내려가요"
   → 학습률 조정, 층 구조 변경, 에폭 증가

3. "RandomForest가 더 좋은데요?"
   → 테이블 데이터에서는 정상. 딥러닝은 이미지/텍스트에서 강점

4. "은닉층을 몇 개 쌓아야 하나요?"
   → 정답 없음. 작게 시작해서 늘려보기. 과대적합 주의
