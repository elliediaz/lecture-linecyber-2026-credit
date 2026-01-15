# 23차시: 딥러닝 응용 - 오토인코더 기반 이상 탐지

## 학습 목표

1. **오토인코더**의 구조와 원리를 이해합니다
2. **Keras**로 오토인코더 모델을 구성합니다
3. **재구성 오차**를 활용한 이상 탐지 기법을 익힙니다

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 8분 | 오토인코더 개념과 수학적 배경 |
| 대주제 2 | 10분 | Keras로 오토인코더 구현 |
| 대주제 3 | 5분 | 재구성 오차 기반 이상 탐지 |
| 정리 | 2분 | 핵심 요약 |

---

## 지난 시간 복습

- **Keras로 MLP 구현**: Sequential, Dense, Dropout
- **역전파 알고리즘**: 경사 하강법으로 가중치 업데이트
- **과적합 방지**: Dropout, Early Stopping, Batch Normalization

**오늘**: 비지도 학습 기반 이상 탐지 - 오토인코더

---

# 대주제 1: 오토인코더 개념과 수학적 배경

## 1.1 오토인코더란?

**정의**: 입력을 압축 후 복원하여 **입력 자체를 재구성**하는 신경망

**특징**:
- 비지도 학습 (레이블 불필요)
- 차원 축소와 특징 추출
- 이상 탐지에 활용

---

## 1.2 오토인코더 구조

```mermaid
flowchart LR
    subgraph 인코더
        X[입력 X] --> H1[은닉층]
        H1 --> Z[잠재 벡터 z]
    end
    subgraph 디코더
        Z --> H2[은닉층]
        H2 --> X_hat[출력 X_hat]
    end
```

- **인코더**: 입력을 저차원으로 압축합니다
- **잠재 공간 (Latent Space)**: 핵심 특징만 남긴 표현입니다
- **디코더**: 저차원에서 원래 차원으로 복원합니다

---

## 1.3 오토인코더 목적 함수

**목표**: 입력과 출력의 차이를 최소화합니다

$$\mathcal{L} = \min_{\theta} ||X - \hat{X}||^2$$

| 기호 | 의미 |
|-----|------|
| $X$ | 원본 입력 |
| $\hat{X}$ | 재구성된 출력 |
| $\theta$ | 모델 파라미터 (가중치, 편향) |
| $\|\|\cdot\|\|^2$ | L2 노름 (유클리드 거리의 제곱) |

---

## 1.4 인코더 수식

**입력을 잠재 공간으로 변환합니다**

$$z = f_{\text{enc}}(X) = \sigma(W_e X + b_e)$$

| 기호 | 의미 |
|-----|------|
| $z \in \mathbb{R}^k$ | 잠재 벡터 (Latent Vector, k차원) |
| $W_e \in \mathbb{R}^{k \times n}$ | 인코더 가중치 행렬 |
| $b_e \in \mathbb{R}^k$ | 인코더 편향 벡터 |
| $\sigma$ | 활성화 함수 (ReLU 등) |

---

## 1.5 디코더 수식

**잠재 공간에서 입력을 복원합니다**

$$\hat{X} = f_{\text{dec}}(z) = \sigma(W_d z + b_d)$$

| 기호 | 의미 |
|-----|------|
| $\hat{X} \in \mathbb{R}^n$ | 재구성된 출력 (n차원) |
| $W_d \in \mathbb{R}^{n \times k}$ | 디코더 가중치 행렬 |
| $b_d \in \mathbb{R}^n$ | 디코더 편향 벡터 |

---

## 1.6 재구성 오차 (Reconstruction Error)

**손실 함수**: MSE (Mean Squared Error)

$$L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(X_i - \hat{X}_i)^2$$

**벡터 형태**:
$$L = \frac{1}{N}\sum_{j=1}^{N}||X^{(j)} - \hat{X}^{(j)}||^2$$

**의미**:
- 낮은 오차 = 정상 데이터 (학습 패턴과 유사합니다)
- 높은 오차 = 이상 데이터 (학습 패턴과 상이합니다)

---

## 1.7 이상 탐지 원리

```
[학습 단계]
정상 데이터만으로 학습 -> 정상 패턴 기억

[탐지 단계]
새 데이터 입력 -> 재구성 -> 오차 계산
  |
  +-- 오차 작음 -> 정상
  +-- 오차 큼   -> 이상
```

---

## 1.8 임계값 설정

**통계적 임계값 설정**

$$\theta = \mu + k\sigma$$

| 기호 | 의미 | 권장값 |
|-----|------|--------|
| $\mu$ | 재구성 오차 평균 | - |
| $\sigma$ | 재구성 오차 표준편차 | - |
| $k$ | 민감도 상수 | 2 ~ 3 |

- $k=2$: 약 95% 신뢰구간
- $k=3$: 약 99.7% 신뢰구간

---

## 1.9 왜 오토인코더로 이상 탐지?

| 방법 | 장점 | 단점 |
|-----|------|------|
| 규칙 기반 | 해석 용이 | 복잡한 패턴 불가 |
| 통계 기반 | 간단함 | 다변량 어려움 |
| **오토인코더** | **복잡한 패턴 학습** | 학습 필요 |

**핵심**: 정상 패턴을 학습하여 비정상을 탐지합니다

---

## 1.10 오토인코더 변형

| 변형 | 특징 | 용도 |
|-----|------|------|
| Vanilla AE | 기본 구조 | 차원 축소, 이상 탐지 |
| Denoising AE | 노이즈 추가 학습 | 노이즈 제거, 강건성 |
| Variational AE | 확률적 잠재 공간 | 데이터 생성 |
| Sparse AE | 희소 제약 | 특징 추출 |

---

# 대주제 2: Keras로 오토인코더 구현

## 2.1 실습 데이터셋: AI4I 2020 Predictive Maintenance

**UCI Machine Learning Repository 제공 데이터셋**

| 항목 | 내용 |
|-----|------|
| 데이터 출처 | UCI ML Repository |
| 샘플 수 | 10,000개 |
| 특성 수 | 6개 (센서 측정값) |
| 고장 유형 | TWF, HDF, PWF, OSF, RNF |
| 목적 | 설비 예지 정비 (Predictive Maintenance) |

---

## 2.2 데이터셋 특성 설명

| 특성 | 설명 | 단위 |
|-----|------|------|
| Air temperature | 공기 온도 | K |
| Process temperature | 공정 온도 | K |
| Rotational speed | 회전 속도 | rpm |
| Torque | 토크 | Nm |
| Tool wear | 공구 마모 | min |
| Machine failure | 고장 여부 | 0/1 |

---

## 2.3 기본 오토인코더 구조

```
입력 (6) -> 인코더 -> 잠재 (2) -> 디코더 -> 출력 (6)
           [6->4->2]            [2->4->6]
```

**차원 관계**: 입력 = 출력 > 잠재 공간

---

## 2.4 Sequential 모델로 구현

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 입력 차원: 6, 잠재 차원: 2
autoencoder = Sequential([
    # 인코더
    Dense(4, activation='relu', input_shape=(6,)),
    Dense(2, activation='relu'),  # 잠재 공간
    # 디코더
    Dense(4, activation='relu'),
    Dense(6, activation='sigmoid')  # 출력
])
```

---

## 2.5 Functional API로 구현

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 입력 정의
input_layer = Input(shape=(6,))

# 인코더
encoded = Dense(4, activation='relu')(input_layer)
latent = Dense(2, activation='relu')(encoded)

# 디코더
decoded = Dense(4, activation='relu')(latent)
output = Dense(6, activation='sigmoid')(decoded)

# 모델 생성
autoencoder = Model(input_layer, output)
```

---

## 2.6 인코더/디코더 분리

```python
# 인코더만 추출
encoder = Model(input_layer, latent)

# 잠재 벡터 추출 가능
latent_features = encoder.predict(X_test)
```

**활용**: 차원 축소, 특징 추출

---

## 2.7 컴파일 설정

```python
autoencoder.compile(
    optimizer='adam',
    loss='mse'  # 재구성 오차
)
```

| 설정 | 값 | 이유 |
|-----|-----|------|
| optimizer | adam | 적응적 학습률 |
| loss | mse | 재구성 오차 최소화 |

---

## 2.8 학습 방법

```python
history = autoencoder.fit(
    X_train, X_train,  # 입력 = 타겟
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=10)]
)
```

**핵심**: `y = X` (자기 자신을 예측합니다)

---

## 2.9 잠재 공간 크기 선택

| 입력 차원 | 잠재 차원 | 압축률 |
|----------|----------|--------|
| 6 | 2 | 67% |
| 30 | 8 | 73% |
| 100 | 20 | 80% |

**원칙**: 충분히 압축하되, 정보 손실 최소화합니다

---

## 2.10 활성화 함수 선택

| 위치 | 함수 | 이유 |
|-----|------|------|
| 은닉층 | ReLU | 비선형성 |
| 출력층 | sigmoid | 0~1 정규화 데이터 |
| 출력층 | linear | 정규화 안 된 데이터 |

---

# 대주제 3: 재구성 오차 기반 이상 탐지

## 3.1 재구성 오차 계산

```python
# 재구성
X_pred = autoencoder.predict(X_test)

# 샘플별 오차 (MSE)
reconstruction_error = np.mean((X_test - X_pred) ** 2, axis=1)
```

---

## 3.2 임계값 결정

```python
# 통계적 임계값
mu = np.mean(reconstruction_error)
sigma = np.std(reconstruction_error)

threshold = mu + 3 * sigma  # 99.7% 신뢰구간
```

---

## 3.3 이상 판정

```python
# 이상 여부 판정
is_anomaly = reconstruction_error > threshold

# 결과 확인
print(f"정상: {(~is_anomaly).sum()}개")
print(f"이상: {is_anomaly.sum()}개")
```

---

## 3.4 AI4I 데이터셋 고장 유형

| 고장 코드 | 설명 | 원인 |
|---------|------|------|
| TWF | Tool Wear Failure | 공구 마모 |
| HDF | Heat Dissipation Failure | 열 방출 실패 |
| PWF | Power Failure | 전력 이상 |
| OSF | Overstrain Failure | 과부하 |
| RNF | Random Failure | 무작위 고장 |

---

## 3.5 성능 평가 지표

| 지표 | 수식 | 의미 |
|-----|------|------|
| Precision | $\frac{TP}{TP+FP}$ | 이상 예측 중 실제 이상 비율 |
| Recall | $\frac{TP}{TP+FN}$ | 실제 이상 중 탐지 비율 |
| F1-Score | $2 \cdot \frac{P \cdot R}{P+R}$ | Precision과 Recall의 조화 평균 |
| AUC-ROC | - | 분류 성능 종합 지표 |

---

# 핵심 정리

## 오토인코더 수식 요약

| 구성요소 | 수식 |
|---------|------|
| 목적 함수 | $\min \|\|X - \hat{X}\|\|^2$ |
| 인코더 | $z = \sigma(W_e X + b_e)$ |
| 디코더 | $\hat{X} = \sigma(W_d z + b_d)$ |
| 재구성 오차 | $L = \frac{1}{n}\sum_{i=1}^{n}(X_i - \hat{X}_i)^2$ |
| 임계값 | $\theta = \mu + k\sigma$ |

---

## 오늘 배운 내용

1. **오토인코더 구조**
   - 인코더-잠재공간-디코더
   - 입력 = 출력 (자기 자신 복원)

2. **Keras 구현**
   - Sequential 또는 Functional API
   - loss='mse', 입력을 타겟으로 학습

3. **이상 탐지**
   - 재구성 오차 계산
   - 임계값 기반 판정

---

## 체크리스트

- [ ] 오토인코더 구조 이해 (인코더-잠재공간-디코더)
- [ ] 재구성 오차 개념 이해 ($L = ||X - \hat{X}||^2$)
- [ ] Keras로 오토인코더 구현
- [ ] 학습: `model.fit(X, X)` 형태
- [ ] 임계값 설정: $\theta = \mu + k\sigma$
- [ ] 이상 판정: 오차 > 임계값

---

## 다음 차시 예고

### [24차시] CNN 기반 이미지 불량 검출

- 합성곱 신경망(CNN) 구조
- Conv2D, MaxPooling2D 레이어
- 이미지 데이터 전처리
- 제조 현장 불량 이미지 분류
