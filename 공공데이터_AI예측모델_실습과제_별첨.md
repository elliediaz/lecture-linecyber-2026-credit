# 실습과제 및 프로젝트 별첨

## 과목명: 공공데이터를 활용한 AI 이해와 예측 모델 구축

---

## 실습과제 목록

| 차시 | 과제명 | 난이도 |
|------|--------|--------|
| 3 | 실습과제 1: 공개 데이터셋 탐색 | 초급 |
| 8 | 실습과제 2-초급: t-검정 실습 | 초급 |
| 10 | 실습과제 2-중급: 카이제곱 검정 실습 | 중급 |
| 13 | 실습과제 3: 시계열 데이터 전처리 | 초급 |
| 17 | 실습과제 4-초급: 분류 모델 구축 | 초급 |
| 19 | 실습과제 4-중급: PCA 적용 분류 모델 | 중급 |
| 24 | 실습과제 5: 역전파 구현 | 중급 |
| 28 | 실습과제 6-초급: CNN 이미지 분류 | 초급 |
| 31 | 실습과제 6-중급: LSTM 시계열 예측 | 중급 |
| 34 | 실습과제 7: GAN 이미지 생성 | 중급 |
| 40 | 최종 프로젝트 | 종합 |

---

## 실습과제 1: 공개 데이터셋 탐색 (차시 3)

### 난이도: 초급

### 과제 목표
- 주요 공개 데이터 플랫폼(Kaggle, AI Hub, 공공데이터포털)을 직접 탐색한다
- 관심 분야의 데이터셋을 선정하고 구조를 파악한다
- 데이터 분석 프로젝트의 첫 단계인 데이터 이해 역량을 함양한다

### 과제 내용

**[필수 수행 사항]**
1. Kaggle(https://kaggle.com)에 회원가입 후 데이터셋 탐색
2. 관심 있는 데이터셋 1개 선정 및 다운로드
3. 다음 항목을 포함한 데이터 탐색 보고서 작성

**[보고서 포함 항목]**
- 데이터셋 이름 및 출처 URL
- 데이터셋 선정 이유 (어떤 문제를 해결하고 싶은가?)
- 데이터셋 구조 분석
  - 전체 행(샘플) 수, 열(피처) 수
  - 각 열의 데이터 타입 (수치형/범주형)
  - 결측치 존재 여부
- 기초 통계량 (`df.describe()` 결과 캡처)
- 예상되는 분석/모델링 방향

### 제출물
- Jupyter Notebook 파일 (.ipynb) 또는 PDF 보고서
- 파일명: `실습과제1_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터셋 선정의 적절성 | 20% |
| 구조 분석의 정확성 | 40% |
| 기초 통계량 해석 | 20% |
| 보고서 완성도 | 20% |

### 참고 코드
```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('your_dataset.csv')

# 기본 정보 확인
print("데이터 크기:", df.shape)
print("\n데이터 타입:")
print(df.dtypes)
print("\n결측치 확인:")
print(df.isnull().sum())
print("\n기초 통계량:")
print(df.describe())
```

---

## 실습과제 2-초급: t-검정 실습 (차시 8)

### 난이도: 초급

### 과제 목표
- 가설검정의 기본 절차를 이해하고 직접 수행한다
- t-검정을 통해 두 집단 간 평균 차이를 통계적으로 검증한다

### 과제 내용

**[데이터셋]**
- UCI Iris 데이터셋 (sklearn.datasets.load_iris)

**[수행 사항]**
1. Iris 데이터셋 로드 및 탐색
2. 두 품종(setosa vs versicolor) 선택
3. 꽃잎 길이(petal length)에 대한 독립표본 t-검정 수행
4. 결과 해석 및 보고서 작성

**[분석 절차]**
- 귀무가설(H0): 두 품종의 꽃잎 길이 평균은 같다
- 대립가설(H1): 두 품종의 꽃잎 길이 평균은 다르다
- 유의수준: α = 0.05

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 파일명: `실습과제2초급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 전처리 정확성 | 20% |
| t-검정 수행 정확성 | 30% |
| 결과 해석의 정확성 | 30% |
| 코드 및 문서 품질 | 20% |

### 참고 코드
```python
from sklearn.datasets import load_iris
from scipy import stats
import pandas as pd

# 데이터 로드
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 두 품종 데이터 분리 (0: setosa, 1: versicolor)
setosa_petal = df[df['species'] == 0]['petal length (cm)']
versicolor_petal = df[df['species'] == 1]['petal length (cm)']

# t-검정 수행
t_stat, p_value = stats.ttest_ind(setosa_petal, versicolor_petal)

print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

# 결과 해석
if p_value < 0.05:
    print("결론: 유의수준 0.05에서 귀무가설을 기각합니다.")
    print("두 품종의 꽃잎 길이 평균은 통계적으로 유의미하게 다릅니다.")
else:
    print("결론: 유의수준 0.05에서 귀무가설을 기각할 수 없습니다.")
```

---

## 실습과제 2-중급: 카이제곱 검정 실습 (차시 10)

### 난이도: 중급

### 과제 목표
- 범주형 변수 간의 독립성을 검정하는 카이제곱 검정을 수행한다
- 공공데이터를 활용하여 실제 분석을 경험한다

### 과제 내용

**[데이터셋]**
- 공공데이터포털 또는 AI Hub에서 설문조사, 사회조사 등 범주형 변수가 포함된 데이터셋 선택
- 예시: 국민건강영양조사, 사회조사 마이크로데이터 등

**[수행 사항]**
1. 공공데이터에서 범주형 변수 2개 선정
2. 교차표(Crosstab) 작성
3. 카이제곱 독립성 검정 수행
4. 결과 해석 및 시각화

**[분석 예시]**
- 연구 질문: "성별과 직업 만족도 사이에 관련성이 있는가?"
- 귀무가설(H0): 성별과 직업 만족도는 독립이다
- 대립가설(H1): 성별과 직업 만족도는 독립이 아니다

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 사용한 공공데이터 출처 명시
- 파일명: `실습과제2중급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 공공데이터 활용 적절성 | 20% |
| 교차표 및 검정 수행 | 30% |
| 결과 해석 및 시각화 | 30% |
| 연구 질문의 의미성 | 20% |

### 참고 코드
```python
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# 공공데이터 로드 (예시)
df = pd.read_csv('survey_data.csv')

# 교차표 생성
crosstab = pd.crosstab(df['성별'], df['직업만족도'])
print("교차표:")
print(crosstab)

# 카이제곱 검정
chi2, p_value, dof, expected = chi2_contingency(crosstab)

print(f"\n카이제곱 통계량: {chi2:.4f}")
print(f"자유도: {dof}")
print(f"p-value: {p_value:.6f}")
print(f"\n기대빈도:")
print(pd.DataFrame(expected,
                   index=crosstab.index,
                   columns=crosstab.columns).round(2))

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('성별 vs 직업만족도 교차표')
plt.show()
```

---

## 실습과제 3: 시계열 데이터 전처리 (차시 13)

### 난이도: 초급

### 과제 목표
- 시계열 데이터의 특성을 이해하고 전처리를 수행한다
- 공공데이터포털의 기상 데이터를 활용하여 실습한다

### 과제 내용

**[데이터셋]**
- 공공데이터포털(data.go.kr) 기상청 일별 기상관측 데이터
- 또는 AI Hub 기상 데이터

**[수행 사항]**
1. 최소 1년치 일별 기온 데이터 다운로드
2. 날짜 컬럼을 datetime 인덱스로 변환
3. 결측치 처리 (보간법 적용)
4. 7일 이동평균 계산
5. 원본 데이터와 이동평균을 함께 시각화
6. 월별 평균 기온 리샘플링 및 막대그래프 시각화

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 시각화 결과 이미지 포함
- 파일명: `실습과제3_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 로드 및 전처리 | 30% |
| datetime 변환 및 인덱싱 | 20% |
| 이동평균 및 리샘플링 | 30% |
| 시각화 품질 | 20% |

### 참고 코드
```python
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('weather_data.csv')

# datetime 인덱스 설정
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 결측치 보간
df['temperature'] = df['temperature'].interpolate(method='linear')

# 7일 이동평균
df['temp_ma7'] = df['temperature'].rolling(window=7).mean()

# 시각화
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['temperature'], label='일별 기온', alpha=0.5)
ax.plot(df.index, df['temp_ma7'], label='7일 이동평균', color='red')
ax.set_xlabel('날짜')
ax.set_ylabel('기온 (°C)')
ax.legend()
ax.set_title('일별 기온 및 이동평균')
plt.show()

# 월별 평균
monthly = df['temperature'].resample('M').mean()
monthly.plot(kind='bar', figsize=(10, 5))
plt.title('월별 평균 기온')
plt.ylabel('기온 (°C)')
plt.show()
```

---

## 실습과제 4-초급: 분류 모델 구축 (차시 17)

### 난이도: 초급

### 과제 목표
- Scikit-learn의 기본 워크플로우를 익힌다
- 분류 모델을 학습하고 평가한다

### 과제 내용

**[데이터셋]**
- UCI Wine 데이터셋 (sklearn.datasets.load_wine)

**[수행 사항]**
1. 데이터 로드 및 탐색
2. 학습/테스트 데이터 분할 (test_size=0.2)
3. 의사결정트리 분류 모델 학습
4. 테스트 데이터로 예측 및 정확도 평가
5. 5-Fold 교차검증 수행
6. 결과 보고서 작성

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 파일명: `실습과제4초급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 분할 정확성 | 20% |
| 모델 학습 및 예측 | 30% |
| 교차검증 수행 | 30% |
| 결과 해석 | 20% |

### 참고 코드
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 로드
wine = load_wine()
X, y = wine.data, wine.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(f"테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 교차검증
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\n5-Fold 교차검증 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

---

## 실습과제 4-중급: PCA 적용 분류 모델 (차시 19)

### 난이도: 중급

### 과제 목표
- PCA를 활용한 차원축소의 효과를 실험적으로 확인한다
- 차원축소 전후 모델 성능을 비교 분석한다

### 과제 내용

**[데이터셋]**
- Kaggle에서 고차원 분류 데이터셋 선택
- 예시: MNIST digits, Fashion-MNIST, 또는 피처가 많은 tabular 데이터

**[수행 사항]**
1. 데이터 로드 및 전처리 (스케일링 필수)
2. 원본 데이터로 분류 모델 학습 및 성능 측정
3. PCA 적용 (설명 분산 비율 95% 기준)
4. 축소된 데이터로 동일 모델 학습 및 성능 측정
5. 주성분 개수에 따른 성능 변화 그래프
6. 결과 비교 분석 보고서

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 파일명: `실습과제4중급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 전처리 및 스케일링 | 20% |
| PCA 적용 및 설명분산 분석 | 25% |
| 모델 성능 비교 실험 | 30% |
| 분석 및 해석의 깊이 | 25% |

### 참고 코드
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 원본 데이터 모델
model_original = RandomForestClassifier(random_state=42)
model_original.fit(X_train_scaled, y_train)
score_original = model_original.score(X_test_scaled, y_test)

# PCA 적용 (설명분산 95%)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"원본 차원: {X_scaled.shape[1]} → PCA 후: {X_pca.shape[1]}")
print(f"설명된 분산 비율: {sum(pca.explained_variance_ratio_):.4f}")

# 주성분 개수별 성능 변화
n_components_range = range(1, min(50, X_scaled.shape[1]))
scores = []
for n in n_components_range:
    pca_temp = PCA(n_components=n)
    X_pca_temp = pca_temp.fit_transform(X_scaled)
    X_train_pca, X_test_pca, y_train_split, y_test_split = train_test_split(
        X_pca_temp, y, test_size=0.2, random_state=42
    )
    model_temp = RandomForestClassifier(random_state=42)
    model_temp.fit(X_train_pca, y_train_split)
    scores.append(model_temp.score(X_test_pca, y_test_split))

plt.plot(n_components_range, scores)
plt.xlabel('주성분 개수')
plt.ylabel('정확도')
plt.title('주성분 개수에 따른 분류 성능')
plt.show()
```

---

## 실습과제 5: 역전파 구현 (차시 24)

### 난이도: 중급

### 과제 목표
- 역전파 알고리즘의 수학적 원리를 코드로 구현한다
- 신경망 학습의 내부 동작을 깊이 이해한다

### 과제 내용

**[수행 사항]**
1. 2층 신경망(입력-은닉-출력) 클래스 구현
2. 순전파(forward) 메서드 구현
3. 역전파(backward) 메서드 구현
4. 경사하강법을 통한 가중치 업데이트
5. XOR 문제 또는 간단한 분류 문제 학습
6. 손실 함수 변화 시각화

**[구현 요구사항]**
- NumPy만 사용 (sklearn, TensorFlow 등 사용 금지)
- Sigmoid 활성화 함수 사용
- MSE 손실 함수 사용
- 각 단계별 수식을 주석으로 설명

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 파일명: `실습과제5_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 순전파 구현 정확성 | 25% |
| 역전파 구현 정확성 | 35% |
| 학습 및 수렴 확인 | 20% |
| 코드 가독성 및 주석 | 20% |

### 참고 코드 (뼈대)
```python
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        # 가중치 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # TODO: 순전파 구현
        # z1 = X @ W1 + b1
        # a1 = sigmoid(z1)
        # z2 = a1 @ W2 + b2
        # y_pred = sigmoid(z2)
        pass

    def backward(self, X, y, y_pred):
        # TODO: 역전파 구현
        # 출력층 기울기 계산
        # 은닉층 기울기 계산
        # 가중치 업데이트
        pass

    def train(self, X, y, epochs=10000):
        losses = []
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)
            # 손실 계산
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            # 역전파
            self.backward(X, y, y_pred)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return losses

# XOR 문제
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 학습
net = TwoLayerNet(2, 4, 1, lr=0.5)
losses = net.train(X, y, epochs=10000)
```

---

## 실습과제 6-초급: CNN 이미지 분류 (차시 28)

### 난이도: 초급

### 과제 목표
- CNN 모델을 구성하고 이미지 분류를 수행한다
- Fashion-MNIST 데이터셋으로 의류 분류 모델을 구축한다

### 과제 내용

**[데이터셋]**
- Fashion-MNIST (keras.datasets.fashion_mnist)
- 10개 의류 클래스: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**[수행 사항]**
1. 데이터 로드 및 전처리 (정규화, reshape)
2. CNN 모델 구성 (Conv2D, MaxPooling2D, Dense)
3. 모델 학습 (epochs=10)
4. 테스트 정확도 평가
5. 학습 곡선(loss, accuracy) 시각화
6. 예측 결과 샘플 시각화

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 파일명: `실습과제6초급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 전처리 | 20% |
| CNN 모델 구성 | 30% |
| 모델 학습 및 평가 | 30% |
| 시각화 | 20% |

### 참고 코드
```python
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 데이터 로드
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 전처리
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 클래스 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# CNN 모델 구성
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습
history = model.fit(X_train, y_train, epochs=10,
                    validation_split=0.1, batch_size=64)

# 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'테스트 정확도: {test_acc:.4f}')

# 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy'); axes[1].legend()
plt.show()
```

---

## 실습과제 6-중급: LSTM 시계열 예측 (차시 31)

### 난이도: 중급

### 과제 목표
- LSTM 모델을 활용하여 시계열 예측을 수행한다
- 공공데이터를 활용하여 실제 예측 문제를 해결한다

### 과제 내용

**[데이터셋]**
- 공공데이터포털 기상 데이터 (일별 기온) 또는
- AI Hub 에너지 소비량 데이터 또는
- 한국거래소 주가 데이터

**[수행 사항]**
1. 최소 2년치 일별 데이터 수집
2. 시계열 데이터 전처리 (스케일링, 시퀀스 생성)
3. LSTM 모델 구성 및 학습
4. 미래 값 예측 (7일 또는 30일)
5. 실제값 vs 예측값 시각화
6. 평가지표(RMSE, MAE) 계산

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 사용 데이터 출처 명시
- 파일명: `실습과제6중급_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| 데이터 수집 및 전처리 | 25% |
| 시퀀스 데이터 생성 | 20% |
| LSTM 모델 구성 및 학습 | 30% |
| 평가 및 시각화 | 25% |

### 참고 코드
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 시퀀스 데이터 생성 함수
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# 데이터 스케일링
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 시퀀스 생성
lookback = 30
X, y = create_sequences(data_scaled, lookback)
X = X.reshape(-1, lookback, 1)

# 학습/테스트 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM 모델
model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(lookback, 1)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_split=0.1, verbose=1)

# 예측
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# 평가
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
```

---

## 실습과제 7: GAN 이미지 생성 (차시 34)

### 난이도: 중급

### 과제 목표
- GAN의 구조와 학습 원리를 이해한다
- 간단한 GAN으로 손글씨 숫자 이미지를 생성한다

### 과제 내용

**[데이터셋]**
- MNIST (keras.datasets.mnist)

**[수행 사항]**
1. Generator 모델 구성
2. Discriminator 모델 구성
3. GAN 학습 루프 구현
4. 에포크별 생성 이미지 저장
5. 학습 과정 시각화 (Loss 변화)
6. 최종 생성 이미지 품질 평가

### 제출물
- Jupyter Notebook 파일 (.ipynb)
- 생성된 이미지 샘플 포함
- 파일명: `실습과제7_학번_이름.ipynb`

### 평가 기준
| 항목 | 배점 |
|------|------|
| Generator 구현 | 25% |
| Discriminator 구현 | 25% |
| 학습 루프 구현 | 30% |
| 생성 이미지 품질 및 시각화 | 20% |

### 참고 코드 (뼈대)
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generator
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),
        layers.Dense(784, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN 학습
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# GAN 모델 (학습 시 discriminator 고정)
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated = generator(gan_input)
gan_output = discriminator(generated)
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(0.0002, 0.5))

# 학습 루프 구현 (TODO)
```

---

## 최종 프로젝트 (차시 40)

### 난이도: 종합

### 프로젝트 목표
- 전 과정에서 학습한 내용을 종합하여 AI 예측 모델을 구축한다
- 공개 데이터셋을 활용하여 실무형 프로젝트를 완성한다
- 구축한 모델을 웹 서비스로 구현한다

### 프로젝트 요구사항

**[필수 요구사항]**
1. 공개 데이터셋(AI Hub, Kaggle, 공공데이터포털 등) 활용
2. 데이터 전처리 및 탐색적 데이터 분석(EDA)
3. 적절한 AI 모델 선택 및 학습
4. 모델 평가 및 성능 개선
5. Streamlit 또는 FastAPI를 활용한 서비스 구현
6. 최종 보고서 및 발표

### 난이도별 주제 예시

**[초급]**
- 공공데이터 기반 날씨/교통량/판매량 예측
- 사용 기술: Scikit-learn, Pandas, Matplotlib
- 서비스: Streamlit 대시보드

**[중급]**
- 시계열 예측 (에너지 소비량, 주가 등)
- 이미지 분류 (의료영상, 제조 불량 탐지)
- 사용 기술: TensorFlow/Keras, LSTM 또는 CNN
- 서비스: Streamlit + 예측 API

**[심화]**
- 텍스트 분류/감성 분석 (뉴스, 리뷰 등)
- 객체 탐지 (YOLO 활용)
- 사용 기술: Transformer, Pre-trained models
- 서비스: FastAPI 배포

### 평가 기준

| 평가 항목 | 배점 | 세부 내용 |
|----------|------|----------|
| 문제 정의의 명확성 | 10% | 해결하고자 하는 문제와 목표가 명확하게 정의되었는가 |
| 데이터 전처리 및 분석의 적절성 | 20% | 결측치, 이상치 처리, 스케일링 등 전처리가 적절하게 수행되었는가 |
| 모델 선택 및 학습의 타당성 | 30% | 문제 유형에 적합한 모델이 선택되고 학습되었는가 |
| 모델 평가 및 해석 | 20% | 적절한 평가지표 사용, 결과 해석 및 개선점 도출 |
| 서비스 구현 | 10% | 모델을 활용한 간단한 서비스가 구현되었는가 |
| 보고서 및 발표 | 10% | 문서화 품질, 발표 논리성 |

### 제출물
1. **프로젝트 소스코드**: Jupyter Notebook 또는 Python 파일
2. **최종 보고서** (10페이지 내외)
   - 프로젝트 개요 및 목표
   - 데이터 설명 및 전처리 과정
   - 모델 설계 및 학습 과정
   - 결과 분석 및 해석
   - 서비스 구현 내용
   - 결론 및 향후 발전 방향
3. **서비스 데모**: 실행 가능한 Streamlit/FastAPI 코드
4. **발표 자료**: PPT (10분 내외 발표)

### 일정
- 주제 선정 및 데이터 확보: 37~38차시
- 모델 개발 및 서비스 구현: 39차시
- 프로젝트 발표: 40차시 이후 (별도 일정)

### Streamlit 서비스 예시 구조
```python
# app.py
import streamlit as st
import pickle
import pandas as pd

st.title('AI 예측 서비스')

# 사이드바: 사용자 입력
st.sidebar.header('입력 값 설정')
feature1 = st.sidebar.slider('특성 1', 0.0, 10.0, 5.0)
feature2 = st.sidebar.slider('특성 2', 0.0, 10.0, 5.0)

# 모델 로드
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# 예측
if st.button('예측하기'):
    input_data = [[feature1, feature2]]
    prediction = model.predict(input_data)
    st.success(f'예측 결과: {prediction[0]}')

# 데이터 시각화
st.subheader('데이터 분포')
# 시각화 코드
```

---

## 참고 데이터셋 목록

| 분류 | 데이터셋명 | 출처 | URL |
|------|-----------|------|-----|
| 분류 | Iris | UCI | https://archive.ics.uci.edu/dataset/53/iris |
| 분류 | Wine | UCI | https://archive.ics.uci.edu/dataset/109/wine |
| 분류 | Breast Cancer | sklearn | sklearn.datasets.load_breast_cancer |
| 이미지 | MNIST | Keras | keras.datasets.mnist |
| 이미지 | Fashion-MNIST | Keras | keras.datasets.fashion_mnist |
| 시계열 | 기상 관측 | 공공데이터포털 | https://data.go.kr |
| 시계열 | 서울시 대기질 | 서울열린데이터 | https://data.seoul.go.kr |
| 텍스트 | 네이버 영화 리뷰 | AI Hub | https://aihub.or.kr |
| 객체탐지 | COCO | COCO | https://cocodataset.org |
| 다양한 | Kaggle Datasets | Kaggle | https://kaggle.com/datasets |

---

*본 문서는 K디지털기초역량훈련 "공공데이터를 활용한 AI 이해와 예측 모델 구축" 과정의 실습과제 및 프로젝트 별첨입니다.*
