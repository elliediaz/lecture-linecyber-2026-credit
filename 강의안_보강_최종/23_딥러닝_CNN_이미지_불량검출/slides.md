---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [23차시] 딥러닝 응용: CNN 기반 이미지 불량 검출

## 합성곱 신경망으로 제품 외관 검사하기

---

# 학습 목표

1. **합성곱 신경망(CNN)**의 구조와 원리를 이해한다
2. **Keras**로 CNN 모델을 구성한다
3. **제품 외관 이미지** 기반 불량 분류를 수행한다

---

# 지난 시간 복습

- **CNN 개념**: 합성곱, 풀링, 특징 맵
- **RNN/LSTM**: 시계열 데이터 처리
- **고급 아키텍처**: ResNet, Transformer

**오늘**: CNN을 활용한 실제 이미지 분류 실습

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 8분 | CNN 수학적 배경 |
| 대주제 2 | 10분 | Keras CNN 모델 구성 |
| 대주제 3 | 5분 | 이미지 불량 분류 실습 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## CNN 수학적 배경

---

# CNN의 핵심 구조

```
입력 이미지
    |
[합성곱층 Conv2D]     <- 필터로 특징 추출
    |
[풀링층 MaxPooling]   <- 공간 크기 축소
    |
(반복)
    |
[평탄화 Flatten]      <- 1차원 변환
    |
[완전연결층 Dense]    <- 분류
    |
출력
```

---

# 합성곱 연산 (Convolution)

입력 이미지 $I$와 커널(필터) $K$의 합성곱:

$$
(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)
$$

| 기호 | 의미 |
|------|------|
| $I$ | 입력 이미지 |
| $K$ | 커널(필터) |
| $(i,j)$ | 출력 위치 |
| $(m,n)$ | 커널 내 위치 |

---

# 합성곱 연산 예시

```
입력 (3x3):           커널 (2x2):
[1  2  3]             [1  0]
[4  5  6]             [0  1]
[7  8  9]

출력 (2x2):
(1,1): 1x1 + 2x0 + 4x0 + 5x1 = 6
(1,2): 2x1 + 3x0 + 5x0 + 6x1 = 8
(2,1): 4x1 + 5x0 + 7x0 + 8x1 = 12
(2,2): 5x1 + 6x0 + 8x0 + 9x1 = 14

결과: [6  8 ]
      [12 14]
```

---

# 특성 맵 크기 계산

출력 크기 공식:

$$
O = \frac{W - K + 2P}{S} + 1
$$

| 기호 | 의미 | 예시 |
|------|------|------|
| $W$ | 입력 크기 | 28 |
| $K$ | 커널 크기 | 3 |
| $P$ | 패딩 | 0 |
| $S$ | 스트라이드 | 1 |
| $O$ | 출력 크기 | (28-3+0)/1+1 = 26 |

---

# 패딩과 스트라이드

**패딩 (Padding)**:
- `valid`: 패딩 없음 (크기 감소)
- `same`: 입력과 동일 크기 유지

**스트라이드 (Stride)**:
- 필터 이동 간격
- stride=2: 출력 크기 절반

```
stride=1: [1][2][3][4][5] -> 5-3+1 = 3 출력
stride=2: [1][2][3][4][5] -> (5-3)/2+1 = 2 출력
```

---

# 풀링 연산 (Max Pooling)

영역 $R_{ij}$ 내 최댓값 선택:

$$
y_{ij} = \max_{(m,n) \in R_{ij}} x_{mn}
$$

```
2x2 Max Pooling:

입력 (4x4):          출력 (2x2):
[1 3 | 2 4]          [3  4]
[5 6 | 7 8]    ->    [10 12]
---------
[9 10| 11 12]
[2 1 | 3  2]
```

**효과**: 크기 축소, 위치 불변성, 계산량 감소

---

# CNN 파라미터 계산

**Conv2D 파라미터 수**:
$$
\text{params} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
$$

예시: Conv2D(32, (3,3)), 입력 채널 1개
- $(3 \times 3 \times 1 + 1) \times 32 = 320$

**Dense 파라미터 수**:
$$
\text{params} = (n_{in} + 1) \times n_{out}
$$

---

# 계층적 특징 학습

```
입력 이미지
    |
[Conv1] -> 저수준 특징 (엣지, 코너)
    |
[Conv2] -> 중간 특징 (텍스처, 패턴)
    |
[Conv3] -> 고수준 특징 (형태, 객체 부분)
    |
[Dense] -> 분류 결정
```

**깊어질수록 추상적인 특징 학습**

---

<!-- _class: lead -->
# 대주제 2
## Keras CNN 모델 구성

---

# CNN 모델 아키텍처

```mermaid
graph LR
    A[입력 28x28x1] --> B[Conv2D 32]
    B --> C[MaxPool 2x2]
    C --> D[Conv2D 64]
    D --> E[MaxPool 2x2]
    E --> F[Conv2D 64]
    F --> G[Flatten]
    G --> H[Dense 64]
    H --> I[Dense 10]
```

---

# Conv2D 층 구성

```python
Conv2D(filters, kernel_size, activation, padding, strides)
```

| 파라미터 | 의미 | 권장값 |
|---------|------|--------|
| filters | 필터 개수 | 32, 64, 128 |
| kernel_size | 필터 크기 | (3,3) |
| activation | 활성화 함수 | 'relu' |
| padding | 패딩 방식 | 'same' |
| strides | 이동 간격 | (1,1) |

---

# MaxPooling2D 층

```python
MaxPooling2D(pool_size=(2, 2))
```

**효과**:
- 공간 크기 절반 축소
- 파라미터 없음 (학습 대상 X)
- 위치 변화에 강건

**대안**: AveragePooling2D (평균값 사용)

---

# Flatten과 Dense

**Flatten**:
- 다차원 특성 맵을 1차원으로 변환
- (7, 7, 64) -> (3136,)

**Dense**:
- 완전 연결층
- 분류 수행

```python
Flatten(),
Dense(64, activation='relu'),
Dropout(0.5),
Dense(10, activation='softmax')
```

---

# 이미지 데이터 형태

```
Keras CNN 입력 형태: (samples, height, width, channels)

예시:
- 흑백 이미지: (1000, 28, 28, 1)
- 컬러 이미지: (1000, 224, 224, 3)

정규화:
- 픽셀값 범위: 0~255 -> 0~1
- X = X / 255.0
```

---

# 다중 클래스 분류 설정

| 항목 | 설정 |
|------|------|
| 출력층 | Dense(클래스 수, activation='softmax') |
| 손실 함수 | 'sparse_categorical_crossentropy' |
| 라벨 형태 | 정수 (0, 1, 2, ...) |

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

# 데이터 증강 (Data Augmentation)

**목적**: 학습 데이터 다양화, 과적합 방지

| 기법 | 설명 |
|------|------|
| 회전 | rotation_range=20 |
| 이동 | width/height_shift_range=0.1 |
| 확대/축소 | zoom_range=0.1 |
| 좌우 반전 | horizontal_flip=True |

---

<!-- _class: lead -->
# 대주제 3
## 이미지 불량 분류 실습

---

# 실습 데이터: Fashion-MNIST

| 항목 | 내용 |
|------|------|
| 이미지 크기 | 28 x 28 픽셀 (흑백) |
| 학습 데이터 | 60,000장 |
| 테스트 데이터 | 10,000장 |
| 클래스 수 | 10개 (의류 종류) |

**제조 시나리오**: 의류 외관 검사 시스템

---

# 클래스 구성

| 라벨 | 품목 | 라벨 | 품목 |
|------|------|------|------|
| 0 | T-shirt | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

# 모델 성능 평가

**혼동 행렬 (Confusion Matrix)**:
- 실제 vs 예측 클래스 비교
- 오분류 패턴 파악

**분류 보고서**:
- Precision: 예측 중 정답 비율
- Recall: 실제 중 예측 성공 비율
- F1-score: 조화 평균

---

# 제조업 적용 시나리오

```
제품 외관 검사 시스템:

[카메라]
    |
[이미지 전처리]
    |
[CNN 모델]
    |
[정상/불량 분류]
    |
[자동 선별]
```

**불량 유형**: 스크래치, 오염, 변형, 균열

---

<!-- _class: lead -->
# 핵심 정리

---

# CNN 수학 공식 요약

| 연산 | 공식 |
|------|------|
| 합성곱 | $(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)$ |
| Max Pooling | $y_{ij} = \max_{(m,n) \in R_{ij}} x_{mn}$ |
| 출력 크기 | $O = \frac{W - K + 2P}{S} + 1$ |

---

# CNN 구성 요약

```
1. 특징 추출부
   +-- Conv2D (필터로 특징 추출)
   +-- MaxPooling2D (크기 축소)
   +-- (반복)

2. 분류부
   +-- Flatten (1차원 변환)
   +-- Dense + Dropout (분류)
   +-- Dense(softmax) (출력)
```

---

# 오늘 배운 내용

1. **CNN 수학적 배경**
   - 합성곱 연산과 특성 맵 크기 계산
   - 풀링 연산의 역할

2. **Keras CNN 구현**
   - Conv2D, MaxPooling2D, Flatten
   - 다중 클래스 분류 설정

3. **이미지 불량 분류**
   - Fashion-MNIST 데이터셋 활용
   - 모델 평가 및 시각화

---

# 체크리스트

- [ ] 합성곱 연산 수식 이해
- [ ] 특성 맵 크기 계산 공식 적용
- [ ] Conv2D, MaxPooling2D 층 구성
- [ ] 이미지 데이터 전처리 (정규화, reshape)
- [ ] 다중 클래스 분류 모델 학습
- [ ] 혼동 행렬로 성능 분석

---

# 다음 차시 예고

## [24차시] 모델 저장과 실무 배포 준비

- 학습된 모델 저장/로드
- 실무 배포를 위한 체크리스트
- 추론 최적화

---

<!-- _class: lead -->
# 수고하셨습니다!

## 질문: CNN을 활용한 품질 검사 아이디어는?
