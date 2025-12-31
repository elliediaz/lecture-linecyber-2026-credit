# [ 강의계획서 ]

| 항목 | 내용 |
|------|------|
| **과목명** | 공공데이터를 활용한 AI 이해와 예측 모델 구축 |
| **운영 교·강사명** | (담당 강사명) |
| **연도/학기** | 2026 |

---

## 강의개요

본 과정은 AI 기술 개발 분야에 진입하려는 비전공자와 입문자를 대상으로, 공공데이터를 활용하여 기초적인 AI 예측 모델을 구현하는 실무 역량을 배양한다. "문제정의 → 데이터 수집·분석 → 모델링 → 평가"의 전 과정을 단계별 실습 훈련으로 구성하여, AI S/W 개발 과정 전반을 이해하고 실제 데이터를 활용한 예측 모델을 직접 구축할 수 있도록 한다. 파이썬 프로그래밍 환경에서 NumPy, Pandas, Scikit-learn, TensorFlow/Keras 등 핵심 라이브러리를 활용하며, AI Hub·Kaggle·UCI Repository 등 다양한 공개 데이터셋을 통해 실제 분석 및 모델 학습을 수행한다.

---

## 사전필요지식

- 기본적인 컴퓨터 활용 능력 (파일 관리, 인터넷 검색 등)
- 프로그래밍 경험이 없어도 수강 가능 (파이썬 기초부터 시작)
- 고등학교 수준의 수학 기초 (함수, 그래프 해석 등)

---

## 훈련핵심역량

- 파이썬 기반 데이터 분석 및 시각화 역량
- 통계적 데이터 검증 및 전처리 기법 적용 역량
- 머신러닝·딥러닝 모델 설계 및 학습 역량
- 공개 데이터셋을 활용한 실무형 AI 모델 구축 역량
- AI 모델의 서비스화 및 API 연동 기초 역량

---

## 사후취득지식

- 파이썬 프로그래밍 및 데이터 분석 도구(NumPy, Pandas, Matplotlib) 활용 능력
- 기술통계, 가설검정(t-test, ANOVA, 카이제곱) 등 통계적 데이터 분석 능력
- 데이터 전처리(결측치, 이상치, 스케일링, 인코딩) 기법 적용 능력
- Scikit-learn 기반 머신러닝 모델(SVM, 의사결정트리, PCA 등) 구현 능력
- CNN, RNN/LSTM, GAN 등 딥러닝 모델의 구조 이해 및 구현 능력
- 트랜스포머 구조와 LLM의 기본 원리 이해
- Streamlit, FastAPI를 활용한 AI 모델 서비스화 기초 능력

---

## 주교재

- **주교재**:
  - 박해선, 『혼자 공부하는 머신러닝+딥러닝』, 한빛미디어, 2024
  - 오렐리앙 제롱, 『핸즈온 머신러닝 (3판)』, 한빛미디어, 2023

- **부교재/참고자료**:
  - AI Hub (https://aihub.or.kr) 공개 데이터셋
  - Kaggle (https://kaggle.com) 데이터셋 및 노트북
  - UCI Machine Learning Repository (https://archive.ics.uci.edu)
  - 공공데이터포털 (https://data.go.kr)
  - TensorFlow/Keras 공식 문서 및 튜토리얼

---

## 차시별 강의내용

### Part I. 환경 구축 및 기초 (1~4차시)

---

#### 차시 1

| 항목 | 내용 |
|------|------|
| **차시명** | 과정 오리엔테이션 및 AI 개발 환경 소개 |
| **학습목표** | - AI 기초체력과정의 전체 구조와 학습 로드맵을 이해한다 |
| | - 파이썬이 AI/데이터 분석에서 사용되는 이유를 설명한다 |
| | - 개발 환경(Anaconda, Jupyter Notebook, VS Code)의 역할을 파악한다 |
| **학습내용** | - 과정 소개: "문제정의 → 분석 → 모델링 → 평가" 학습 구조 |
| | - 파이썬 언어의 특징과 AI 생태계에서의 위치 |
| | - Anaconda 설치 및 가상환경 생성 실습 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

---

#### 차시 2

| 항목 | 내용 |
|------|------|
| **차시명** | 파이썬 기초 문법과 Jupyter Notebook 활용 |
| **학습목표** | - 파이썬 기본 자료형(int, float, str, list, dict)을 구분하고 활용한다 |
| | - 조건문과 반복문을 사용하여 간단한 로직을 구현한다 |
| | - Jupyter Notebook에서 셀 단위 실행과 마크다운을 활용한다 |
| **학습내용** | - 변수 선언, 자료형 변환, 기본 연산자 |
| | - if-else 조건문, for/while 반복문 실습 |
| | - Jupyter Notebook 단축키와 마크다운 문서화 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

---

#### 차시 3

| 항목 | 내용 |
|------|------|
| **차시명** | 공개 데이터셋 탐색과 활용 방법 |
| **학습목표** | - AI Hub, Kaggle, UCI Repository 등 주요 공개 데이터 플랫폼을 탐색한다 |
| | - 데이터셋 다운로드 및 API 키 발급 절차를 수행한다 |
| | - CSV, JSON 등 데이터 형식의 특징을 이해한다 |
| **학습내용** | - 공공데이터포털, AI Hub 데이터셋 검색 및 다운로드 |
| | - Kaggle 계정 생성 및 Kaggle API를 통한 데이터 다운로드 |
| | - UCI Repository의 대표 데이터셋(Iris, Wine, Boston Housing 등) 소개 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 1]** Kaggle에서 관심 데이터셋 1개 다운로드 후 구조 파악 보고서 작성 |

---

#### 차시 4

| 항목 | 내용 |
|------|------|
| **차시명** | NumPy와 Pandas 기초 - 데이터 다루기의 시작 |
| **학습목표** | - NumPy 배열(ndarray)의 생성과 기본 연산을 수행한다 |
| | - Pandas DataFrame을 생성하고 데이터를 조회·필터링한다 |
| | - CSV 파일을 불러와 기본 정보를 확인한다 |
| **학습내용** | - NumPy: 배열 생성(arange, linspace), 형상 변환(reshape), 브로드캐스팅 |
| | - Pandas: Series와 DataFrame, read_csv(), head(), info(), describe() |
| | - 인덱싱과 슬라이싱, loc/iloc를 활용한 데이터 접근 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

---

### Part II. 데이터 분석 및 통계 기초 (5~18차시)

---

#### 차시 5

| 항목 | 내용 |
|------|------|
| **차시명** | 기술통계량의 이해 (1) - 중심경향 측도 |
| **학습목표** | - 평균, 중앙값, 최빈값의 개념과 차이를 설명한다 |
| | - 각 중심경향 측도가 적합한 데이터 상황을 판단한다 |
| | - Pandas와 NumPy로 중심경향 측도를 계산한다 |
| **학습내용** | - 평균(mean): 산술평균의 수학적 정의 ∑x/n와 코드 구현 |
| | - 중앙값(median): 정렬 후 중앙 위치값, 이상치에 강건한 특성 |
| | - 최빈값(mode): 범주형 데이터에서의 활용 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 중심경향 측도 계산
import numpy as np
import pandas as pd

data = [23, 25, 27, 28, 29, 30, 31, 150]  # 이상치 포함 데이터
print(f"평균: {np.mean(data):.2f}")      # 이상치 영향 큼
print(f"중앙값: {np.median(data):.2f}")  # 이상치에 강건
print(f"최빈값: {pd.Series(data).mode().values}")
```

---

#### 차시 6

| 항목 | 내용 |
|------|------|
| **차시명** | 기술통계량의 이해 (2) - 산포도와 분포 |
| **학습목표** | - 분산, 표준편차, 범위, IQR의 개념을 이해한다 |
| | - 데이터의 퍼짐 정도를 정량적으로 표현한다 |
| | - 왜도(skewness)와 첨도(kurtosis)로 분포 형태를 파악한다 |
| **학습내용** | - 분산(variance): σ² = Σ(x-μ)²/n 수식과 np.var() 구현 |
| | - 표준편차(std): 분산의 제곱근, 원래 단위로 해석 |
| | - 사분위수와 IQR(Q3-Q1)을 활용한 이상치 탐지 기준 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 산포도 계산 및 시각화
import matplotlib.pyplot as plt

data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15
print(f"분산: {np.var(data):.2f}")
print(f"표준편차: {np.std(data):.2f}")

# 박스플롯으로 분포 확인
plt.boxplot(data)
plt.title('데이터 분포 박스플롯')
plt.show()
```

---

#### 차시 7

| 항목 | 내용 |
|------|------|
| **차시명** | 확률분포의 기초 - 정규분포와 표준화 |
| **학습목표** | - 정규분포의 특성과 68-95-99.7 규칙을 설명한다 |
| | - Z-점수(표준화)의 의미와 계산 방법을 이해한다 |
| | - scipy.stats를 활용하여 확률을 계산한다 |
| **학습내용** | - 정규분포 N(μ, σ²): 종 모양 곡선, 평균 중심 대칭 |
| | - 표준정규분포 Z = (X - μ) / σ: 평균 0, 표준편차 1로 변환 |
| | - scipy.stats.norm을 활용한 CDF, PDF, PPF 계산 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 정규분포와 표준화
from scipy import stats

# 평균 70, 표준편차 10인 시험 점수 분포
mu, sigma = 70, 10
score = 85

# Z-점수 계산
z_score = (score - mu) / sigma
print(f"85점의 Z-점수: {z_score}")

# 85점 이상일 확률
prob = 1 - stats.norm.cdf(z_score)
print(f"85점 이상일 확률: {prob:.4f} ({prob*100:.2f}%)")
```

---

#### 차시 8

| 항목 | 내용 |
|------|------|
| **차시명** | 가설검정의 기초 - t-검정 |
| **학습목표** | - 귀무가설과 대립가설의 개념을 구분한다 |
| | - p-value의 의미와 유의수준(α)의 관계를 이해한다 |
| | - 독립표본 t-검정을 수행하고 결과를 해석한다 |
| **학습내용** | - 가설검정의 논리: 귀무가설 기각 여부 판단 |
| | - 1종 오류(α)와 2종 오류(β)의 개념 |
| | - scipy.stats.ttest_ind()를 활용한 두 집단 평균 비교 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 2-초급]** UCI Iris 데이터셋에서 두 품종 간 꽃잎 길이 차이 t-검정 수행 |

```python
# 실습 예시: 독립표본 t-검정
from scipy import stats

# A반과 B반의 시험 점수
class_A = [78, 82, 85, 79, 88, 91, 76, 84]
class_B = [72, 75, 69, 78, 71, 73, 77, 70]

# t-검정 수행
t_stat, p_value = stats.ttest_ind(class_A, class_B)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("유의수준 0.05에서 두 반의 평균 점수는 유의미하게 다릅니다.")
```

---

#### 차시 9

| 항목 | 내용 |
|------|------|
| **차시명** | 분산분석(ANOVA)의 이해 |
| **학습목표** | - ANOVA가 t-검정과 다른 점(3개 이상 집단 비교)을 설명한다 |
| | - F-통계량과 집단 간/집단 내 변동의 관계를 이해한다 |
| | - 일원분산분석(One-way ANOVA)을 수행하고 해석한다 |
| **학습내용** | - ANOVA의 기본 가정: 정규성, 등분산성, 독립성 |
| | - F-통계량 = 집단간 분산 / 집단내 분산 |
| | - scipy.stats.f_oneway()를 활용한 ANOVA 실습 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 일원분산분석
from scipy import stats

# 세 가지 학습법에 따른 성적
method_A = [85, 88, 90, 87, 86]
method_B = [78, 82, 80, 79, 81]
method_C = [92, 95, 91, 94, 93]

# ANOVA 수행
f_stat, p_value = stats.f_oneway(method_A, method_B, method_C)
print(f"F-통계량: {f_stat:.4f}")
print(f"p-value: {p_value:.6f}")
# → 학습법에 따른 성적 차이가 유의미한지 판단
```

---

#### 차시 10

| 항목 | 내용 |
|------|------|
| **차시명** | 카이제곱 검정과 범주형 데이터 분석 |
| **학습목표** | - 카이제곱 검정의 적용 상황(범주형 변수 간 관계)을 이해한다 |
| | - 적합도 검정과 독립성 검정을 구분한다 |
| | - 교차표(crosstab)를 작성하고 카이제곱 검정을 수행한다 |
| **학습내용** | - 카이제곱 통계량: χ² = Σ(관측값-기대값)²/기대값 |
| | - pd.crosstab()을 활용한 교차표 생성 |
| | - scipy.stats.chi2_contingency()로 독립성 검정 수행 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 2-중급]** 공공데이터(설문조사 등)를 활용한 범주형 변수 간 독립성 검정 보고서 |

```python
# 실습 예시: 카이제곱 독립성 검정
import pandas as pd
from scipy.stats import chi2_contingency

# 성별과 제품 선호도 교차표
data = pd.DataFrame({
    '성별': ['남', '남', '여', '여', '남', '여', '남', '여'] * 20,
    '선호제품': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B'] * 20
})
crosstab = pd.crosstab(data['성별'], data['선호제품'])
print(crosstab)

chi2, p_value, dof, expected = chi2_contingency(crosstab)
print(f"\n카이제곱 통계량: {chi2:.4f}, p-value: {p_value:.4f}")
```

---

#### 차시 11

| 항목 | 내용 |
|------|------|
| **차시명** | 데이터 전처리 (1) - 결측치와 이상치 처리 |
| **학습목표** | - 결측치의 유형(MCAR, MAR, MNAR)을 구분한다 |
| | - 결측치 처리 방법(삭제, 대체, 예측)을 상황에 맞게 적용한다 |
| | - IQR 방법과 Z-score 방법으로 이상치를 탐지한다 |
| **학습내용** | - isnull(), dropna(), fillna()를 활용한 결측치 처리 |
| | - 평균/중앙값 대체, 최빈값 대체, 보간법(interpolate) |
| | - 이상치 탐지: IQR(Q1-1.5*IQR ~ Q3+1.5*IQR), Z-score(|z|>3) |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 결측치 및 이상치 처리
import pandas as pd
import numpy as np

# 결측치 포함 데이터
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, 50]
})

# 결측치 확인 및 처리
print("결측치 확인:\n", df.isnull().sum())
df_filled = df.fillna(df.mean())  # 평균으로 대체
print("\n평균 대체 후:\n", df_filled)

# IQR 기반 이상치 탐지
Q1, Q3 = df['B'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['B'] < Q1-1.5*IQR) | (df['B'] > Q3+1.5*IQR)]
```

---

#### 차시 12

| 항목 | 내용 |
|------|------|
| **차시명** | 데이터 전처리 (2) - 스케일링과 인코딩 |
| **학습목표** | - 피처 스케일링(정규화, 표준화)의 필요성을 이해한다 |
| | - MinMaxScaler와 StandardScaler를 적절히 선택한다 |
| | - 범주형 변수의 인코딩(Label, One-Hot)을 수행한다 |
| **학습내용** | - 정규화(Min-Max): (x-min)/(max-min), 0~1 범위로 변환 |
| | - 표준화(Z-score): (x-μ)/σ, 평균 0, 표준편차 1로 변환 |
| | - LabelEncoder, OneHotEncoder, pd.get_dummies() 활용 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 스케일링과 인코딩
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import pandas as pd

# 스케일링
data = [[100, 0.1], [200, 0.2], [300, 0.3]]
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print("표준화 결과:\n", scaled)

# 원-핫 인코딩
df = pd.DataFrame({'도시': ['서울', '부산', '대구', '서울']})
encoded = pd.get_dummies(df['도시'], prefix='도시')
print("\n원-핫 인코딩:\n", encoded)
```

---

#### 차시 13

| 항목 | 내용 |
|------|------|
| **차시명** | 시계열 데이터의 이해와 전처리 |
| **학습목표** | - 시계열 데이터의 구성요소(추세, 계절성, 잔차)를 파악한다 |
| | - 시계열 데이터의 인덱싱과 리샘플링을 수행한다 |
| | - 이동평균과 차분을 통해 데이터를 정상화한다 |
| **학습내용** | - pd.to_datetime(), DatetimeIndex 활용 |
| | - resample()을 통한 일별→주별→월별 집계 |
| | - 이동평균(rolling), 차분(diff)을 통한 추세 제거 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 3]** 공공데이터포털 기상 데이터를 활용한 시계열 전처리 및 시각화 |

```python
# 실습 예시: 시계열 데이터 처리
import pandas as pd
import numpy as np

# 시계열 데이터 생성
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

# 7일 이동평균
ts_ma = ts.rolling(window=7).mean()

# 차분 (1차)
ts_diff = ts.diff()

print("원본 데이터 통계:\n", ts.describe())
```

---

#### 차시 14

| 항목 | 내용 |
|------|------|
| **차시명** | 텍스트 데이터의 이해와 전처리 |
| **학습목표** | - 텍스트 데이터 전처리의 주요 단계를 나열한다 |
| | - 토큰화, 불용어 제거, 정규화를 수행한다 |
| | - TF-IDF를 이용하여 텍스트를 수치화한다 |
| **학습내용** | - 텍스트 정제: 특수문자 제거, 소문자 변환, 정규표현식 |
| | - 형태소 분석(konlpy), 불용어(stopwords) 처리 |
| | - Bag of Words, TF-IDF 벡터화 (sklearn.feature_extraction) |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 텍스트 전처리와 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# 텍스트 정제
texts = ["AI는 인공지능입니다!", "머신러닝과 딥러닝은 AI의 하위 분야입니다."]
cleaned = [re.sub(r'[^\w\s]', '', t) for t in texts]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned)
print("어휘 사전:", vectorizer.get_feature_names_out())
print("TF-IDF 행렬:\n", tfidf_matrix.toarray())
```

---

#### 차시 15

| 항목 | 내용 |
|------|------|
| **차시명** | 상관분석과 회귀분석 기초 |
| **학습목표** | - 피어슨 상관계수의 의미와 해석 방법을 이해한다 |
| | - 단순선형회귀의 원리(최소제곱법)를 설명한다 |
| | - sklearn으로 선형회귀 모델을 학습하고 예측한다 |
| **학습내용** | - 상관계수 r: -1 ~ +1, 선형 관계의 강도와 방향 |
| | - 회귀직선 y = wx + b, 손실함수 MSE = Σ(y-ŷ)²/n |
| | - LinearRegression() 모델 학습, fit(), predict(), score() |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 상관분석과 선형회귀
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.0, 5.8, 8.1, 9.9])

# 상관계수
corr = np.corrcoef(X.flatten(), y)[0, 1]
print(f"상관계수: {corr:.4f}")

# 선형회귀
model = LinearRegression()
model.fit(X, y)
print(f"기울기(w): {model.coef_[0]:.4f}, 절편(b): {model.intercept_:.4f}")
print(f"R² 점수: {model.score(X, y):.4f}")
```

---

#### 차시 16

| 항목 | 내용 |
|------|------|
| **차시명** | Classification vs Regression - 머신러닝의 두 축 |
| **학습목표** | - 분류(Classification)와 회귀(Regression)의 차이를 구분한다 |
| | - 문제 유형에 따른 적절한 알고리즘을 선택한다 |
| | - 평가지표(정확도, RMSE 등)의 적용 기준을 이해한다 |
| **학습내용** | - 분류: 이산적 클래스 예측 (스팸/정상, 품종 분류 등) |
| | - 회귀: 연속적 수치 예측 (가격, 온도, 수요량 등) |
| | - 분류 평가: Accuracy, Precision, Recall, F1-score |
| | - 회귀 평가: MSE, RMSE, MAE, R² |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

---

#### 차시 17

| 항목 | 내용 |
|------|------|
| **차시명** | Scikit-learn 기초 - 모델 학습 파이프라인 |
| **학습목표** | - train_test_split을 활용하여 데이터를 분할한다 |
| | - Scikit-learn의 fit-predict 패턴을 이해한다 |
| | - 교차검증(Cross-Validation)의 필요성과 방법을 익힌다 |
| **학습내용** | - 학습/검증/테스트 데이터 분할의 의미 |
| | - cross_val_score()를 활용한 K-Fold 교차검증 |
| | - 과적합(Overfitting)과 과소적합(Underfitting) 개념 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 4-초급]** UCI Wine 데이터셋을 활용한 품종 분류 모델 학습 및 평가 |

```python
# 실습 예시: 데이터 분할과 교차검증
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 데이터 로드 및 분할
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 모델 학습 및 평가
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print(f"테스트 정확도: {model.score(X_test, y_test):.4f}")

# 5-Fold 교차검증
cv_scores = cross_val_score(model, iris.data, iris.target, cv=5)
print(f"교차검증 평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

---

#### 차시 18

| 항목 | 내용 |
|------|------|
| **차시명** | SVM과 의사결정트리 실습 |
| **학습목표** | - SVM의 마진 최대화 원리를 설명한다 |
| | - 의사결정트리의 분할 기준(지니 계수, 엔트로피)을 이해한다 |
| | - 두 알고리즘을 실제 데이터에 적용하고 비교한다 |
| **학습내용** | - SVM: 초평면, 서포트 벡터, 커널 트릭(RBF, polynomial) |
| | - 의사결정트리: 정보 이득, 가지치기(pruning), 해석 용이성 |
| | - sklearn의 SVC, DecisionTreeClassifier 비교 실습 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: SVM vs 의사결정트리 비교
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 유방암 데이터셋
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
print(f"SVM 정확도: {svm.score(X_test, y_test):.4f}")

# 의사결정트리
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
print(f"의사결정트리 정확도: {tree.score(X_test, y_test):.4f}")
```

---

#### 차시 19 (Part II 마무리)

| 항목 | 내용 |
|------|------|
| **차시명** | PCA와 차원축소 기법 |
| **학습목표** | - 차원의 저주와 차원축소의 필요성을 설명한다 |
| | - PCA의 원리(공분산 행렬, 고유벡터)를 이해한다 |
| | - PCA를 적용하여 데이터를 시각화하고 모델 성능을 비교한다 |
| **학습내용** | - 고차원 데이터의 문제점: 계산량, 과적합, 시각화 어려움 |
| | - PCA: 분산을 최대한 보존하는 새로운 축으로 투영 |
| | - explained_variance_ratio_로 주성분 선택 기준 결정 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 4-중급]** Kaggle 데이터셋에 PCA 적용 후 분류 모델 성능 비교 보고서 |

```python
# 실습 예시: PCA 차원축소
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 64차원 손글씨 데이터
digits = load_digits()
print(f"원본 차원: {digits.data.shape}")  # (1797, 64)

# PCA로 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(digits.data)
print(f"축소 후 차원: {X_pca.shape}")  # (1797, 2)
print(f"설명된 분산 비율: {sum(pca.explained_variance_ratio_):.4f}")

# 시각화
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='tab10', alpha=0.5)
plt.colorbar()
plt.title('PCA로 축소된 손글씨 숫자 데이터')
plt.show()
```

---

### Part III. AI 기초 (20~32차시)

---

#### 차시 20

| 항목 | 내용 |
|------|------|
| **차시명** | 인공지능·머신러닝·딥러닝 개념 정리 |
| **학습목표** | - AI, ML, DL의 포함 관계와 역사적 발전을 설명한다 |
| | - 지도학습, 비지도학습, 강화학습을 구분한다 |
| | - 딥러닝이 기존 머신러닝과 다른 점을 이해한다 |
| **학습내용** | - AI > 머신러닝 > 딥러닝의 계층 구조 |
| | - 지도학습: 레이블이 있는 데이터로 학습 (분류, 회귀) |
| | - 비지도학습: 레이블 없이 패턴 발견 (클러스터링, 차원축소) |
| | - 딥러닝: 다층 신경망을 통한 자동 특징 추출 |
| **이론/실습** | 이론 |
| **과제/프로젝트** | - |

---

#### 차시 21

| 항목 | 내용 |
|------|------|
| **차시명** | AI를 위한 수학 기초 (1) - 선형대수 |
| **학습목표** | - 벡터와 행렬의 기본 연산을 수행한다 |
| | - 행렬곱의 의미와 신경망에서의 역할을 이해한다 |
| | - NumPy로 선형대수 연산을 구현한다 |
| **학습내용** | - 벡터: 방향과 크기, 내적(dot product), 코사인 유사도 |
| | - 행렬: 덧셈, 스칼라곱, 행렬곱(@ 연산자), 전치(transpose) |
| | - 신경망에서 입력×가중치 행렬곱 → 출력 계산 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 선형대수 기초
import numpy as np

# 벡터 내적 (코사인 유사도의 분자)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
print(f"내적: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# 행렬곱 (신경망의 핵심 연산)
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 입력
W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2x3 가중치
output = X @ W  # 3x3 출력
print(f"행렬곱 결과:\n{output}")
```

---

#### 차시 22

| 항목 | 내용 |
|------|------|
| **차시명** | AI를 위한 수학 기초 (2) - 미분과 경사하강법 |
| **학습목표** | - 미분의 기하학적 의미(기울기)를 이해한다 |
| | - 경사하강법의 원리와 학습률의 역할을 설명한다 |
| | - 간단한 함수에 경사하강법을 적용하여 최솟값을 찾는다 |
| **학습내용** | - 미분: f'(x) = lim(Δx→0) [f(x+Δx) - f(x)] / Δx |
| | - 경사하강법: x_new = x_old - η * ∂L/∂x (η: 학습률) |
| | - 손실함수(Loss Function)와 최적화의 관계 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 경사하강법으로 최솟값 찾기
import numpy as np
import matplotlib.pyplot as plt

# f(x) = x^2 의 최솟값 찾기
def f(x): return x ** 2
def df(x): return 2 * x  # 미분

x = 10.0  # 시작점
lr = 0.1  # 학습률
history = [x]

for _ in range(20):
    x = x - lr * df(x)  # 경사하강법 업데이트
    history.append(x)

print(f"최종 x: {x:.6f}, f(x): {f(x):.6f}")

# 시각화
plt.plot(history, [f(h) for h in history], 'bo-')
plt.xlabel('iteration'); plt.ylabel('f(x)')
plt.title('경사하강법으로 x^2의 최솟값 찾기')
plt.show()
```

---

#### 차시 23

| 항목 | 내용 |
|------|------|
| **차시명** | 신경망의 기본 구조 - 퍼셉트론 |
| **학습목표** | - 퍼셉트론의 구조(입력, 가중치, 활성화)를 설명한다 |
| | - 단층 퍼셉트론의 한계(XOR 문제)를 이해한다 |
| | - 다층 퍼셉트론(MLP)이 비선형 문제를 해결하는 원리를 파악한다 |
| **학습내용** | - 퍼셉트론: y = activation(Σ(w_i * x_i) + b) |
| | - 활성화 함수: step, sigmoid, ReLU |
| | - XOR 문제와 은닉층의 필요성 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 퍼셉트론 구현
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

# AND 게이트 학습
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(2)
# 간단한 학습 루프 생략 - 실제 수업에서 구현
```

---

#### 차시 24

| 항목 | 내용 |
|------|------|
| **차시명** | 역전파(Backpropagation) 알고리즘 |
| **학습목표** | - 순전파와 역전파의 차이를 설명한다 |
| | - 연쇄법칙(Chain Rule)이 역전파에서 적용되는 방식을 이해한다 |
| | - 간단한 신경망에서 역전파를 직접 계산한다 |
| **학습내용** | - 순전파: 입력 → 출력 방향으로 값 계산 |
| | - 역전파: 출력 → 입력 방향으로 기울기 전파 |
| | - 연쇄법칙: ∂L/∂w = ∂L/∂y * ∂y/∂z * ∂z/∂w |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 5]** 2층 신경망의 역전파 과정을 직접 계산하고 코드로 구현 |

```python
# 실습 예시: 간단한 역전파 구현
import numpy as np

# 순전파
def forward(x, w1, w2):
    z1 = x @ w1
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid
    z2 = a1 @ w2
    y_pred = z2
    return z1, a1, z2, y_pred

# 역전파 (MSE 손실 기준)
def backward(x, y, z1, a1, y_pred, w1, w2):
    m = x.shape[0]
    # 출력층 기울기
    dz2 = (y_pred - y) / m
    dw2 = a1.T @ dz2
    # 은닉층 기울기
    da1 = dz2 @ w2.T
    dz1 = da1 * a1 * (1 - a1)  # sigmoid 미분
    dw1 = x.T @ dz1
    return dw1, dw2
```

---

#### 차시 25

| 항목 | 내용 |
|------|------|
| **차시명** | 기울기 소실 문제와 활성화 함수 |
| **학습목표** | - 기울기 소실(Vanishing Gradient) 문제의 원인을 설명한다 |
| | - ReLU가 sigmoid보다 깊은 신경망에서 유리한 이유를 이해한다 |
| | - 다양한 활성화 함수(ReLU, Leaky ReLU, ELU)를 비교한다 |
| **학습내용** | - Sigmoid/Tanh: 출력 범위 제한 → 기울기가 0에 가까워짐 |
| | - ReLU: f(x) = max(0, x), 양수에서 기울기 1 유지 |
| | - Dying ReLU 문제와 Leaky ReLU, ELU의 해결책 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 활성화 함수 비교
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# 활성화 함수들
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01 * x)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, func, name in zip(axes, [sigmoid, tanh, relu, leaky_relu],
                          ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']):
    ax.plot(x, func)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(name)
plt.tight_layout()
plt.show()
```

---

#### 차시 26

| 항목 | 내용 |
|------|------|
| **차시명** | TensorFlow/Keras 기초와 첫 번째 딥러닝 모델 |
| **학습목표** | - TensorFlow와 Keras의 관계를 이해한다 |
| | - Sequential 모델로 간단한 신경망을 구성한다 |
| | - compile, fit, evaluate의 역할을 파악한다 |
| **학습내용** | - Keras: 고수준 딥러닝 API, TensorFlow 백엔드 |
| | - Sequential([Dense, Activation, ...]) 구조 |
| | - compile(optimizer, loss, metrics), fit(X, y, epochs, batch_size) |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: Keras로 첫 신경망 구축
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 모델 구성
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3개 클래스
])

# 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
print(f"테스트 정확도: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
```

---

#### 차시 27

| 항목 | 내용 |
|------|------|
| **차시명** | CNN 기초 - 합성곱과 풀링 |
| **학습목표** | - 합성곱(Convolution) 연산의 원리를 이해한다 |
| | - 필터(커널)가 특징을 추출하는 방식을 설명한다 |
| | - 풀링(Pooling)의 역할과 종류를 파악한다 |
| **학습내용** | - 합성곱: 필터를 슬라이딩하며 지역 특징 추출 |
| | - 스트라이드(stride), 패딩(padding)의 의미 |
| | - MaxPooling, AveragePooling: 공간 크기 축소, 과적합 방지 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 합성곱 연산 직접 구현
import numpy as np

# 5x5 이미지
image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [2, 1, 0, 1, 2],
    [0, 1, 2, 1, 0]
])

# 3x3 에지 감지 필터
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# 합성곱 수행 (valid padding, stride=1)
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        output[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)

print("합성곱 결과:\n", output)
```

---

#### 차시 28

| 항목 | 내용 |
|------|------|
| **차시명** | CNN 실습 - MNIST 손글씨 분류 |
| **학습목표** | - CNN 모델을 Keras로 구성하고 학습한다 |
| | - Conv2D, MaxPooling2D, Flatten, Dense 레이어를 연결한다 |
| | - 학습 결과를 시각화하고 성능을 평가한다 |
| **학습내용** | - MNIST 데이터셋: 28x28 그레이스케일, 10개 숫자 클래스 |
| | - 데이터 전처리: 정규화(0~1), reshape((28, 28, 1)) |
| | - CNN 구조: Conv → Pool → Conv → Pool → Flatten → Dense |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 6-초급]** Fashion-MNIST 데이터셋으로 의류 분류 CNN 모델 구축 |

```python
# 실습 예시: MNIST CNN 분류
from tensorflow import keras
from tensorflow.keras import layers

# 데이터 로드 및 전처리
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
print(f"테스트 정확도: {model.evaluate(X_test, y_test)[1]:.4f}")
```

---

#### 차시 29

| 항목 | 내용 |
|------|------|
| **차시명** | YOLO를 활용한 객체 탐지 이해 |
| **학습목표** | - 이미지 분류와 객체 탐지의 차이를 설명한다 |
| | - YOLO(You Only Look Once)의 작동 원리를 이해한다 |
| | - 사전 학습된 YOLO 모델을 활용하여 객체 탐지를 수행한다 |
| **학습내용** | - 객체 탐지: 위치(Bounding Box) + 클래스 분류 |
| | - YOLO: 그리드 기반 1-Stage Detector, 실시간 처리 가능 |
| | - Ultralytics YOLOv8 활용 실습 (사전 학습 모델) |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: YOLOv8 객체 탐지 (Ultralytics 라이브러리)
# pip install ultralytics

from ultralytics import YOLO

# 사전 학습된 모델 로드
model = YOLO('yolov8n.pt')  # nano 버전

# 이미지에서 객체 탐지
results = model('path/to/image.jpg')

# 결과 시각화
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"클래스: {model.names[int(box.cls)]}, 신뢰도: {box.conf:.2f}")
    result.show()  # 탐지 결과 이미지 표시
```

---

#### 차시 30

| 항목 | 내용 |
|------|------|
| **차시명** | RNN 기초 - 순환신경망의 이해 |
| **학습목표** | - 순차 데이터(시계열, 텍스트)의 특성을 설명한다 |
| | - RNN의 순환 구조와 은닉 상태(hidden state)를 이해한다 |
| | - RNN의 한계(장기 의존성 문제)를 파악한다 |
| **학습내용** | - 순차 데이터: 순서가 중요한 데이터 (시계열, 문장 등) |
| | - RNN 구조: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t) |
| | - BPTT(Backpropagation Through Time)와 기울기 소실/폭발 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 간단한 RNN 셀 구현
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, inputs, h_prev):
        """inputs: (input_size, 1), h_prev: (hidden_size, 1)"""
        h_next = np.tanh(self.Wxh @ inputs + self.Whh @ h_prev + self.bh)
        return h_next

# 시퀀스 처리
rnn = SimpleRNN(input_size=10, hidden_size=20)
h = np.zeros((20, 1))  # 초기 은닉 상태
for t in range(5):  # 5개 타임스텝
    x = np.random.randn(10, 1)
    h = rnn.forward(x, h)
    print(f"t={t}, h shape: {h.shape}")
```

---

#### 차시 31

| 항목 | 내용 |
|------|------|
| **차시명** | LSTM과 시계열 예측 실습 |
| **학습목표** | - LSTM의 게이트 구조(망각, 입력, 출력)를 이해한다 |
| | - LSTM이 장기 의존성 문제를 해결하는 원리를 설명한다 |
| | - AI Hub 또는 공공 시계열 데이터로 예측 모델을 구축한다 |
| **학습내용** | - LSTM 셀: Cell State(장기 메모리) + Hidden State(단기 메모리) |
| | - 망각 게이트, 입력 게이트, 출력 게이트의 역할 |
| | - Keras LSTM 레이어를 활용한 시계열 예측 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 6-중급]** 공공데이터(기온, 주가 등) LSTM 예측 모델 구축 및 평가 보고서 |

```python
# 실습 예시: LSTM 시계열 예측
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 시계열 데이터 생성 (사인 함수)
time = np.arange(0, 100, 0.1)
data = np.sin(time)

# 시퀀스 데이터 준비 (lookback=10)
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(data)
X = X.reshape(-1, 10, 1)  # (samples, timesteps, features)

# LSTM 모델
model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(10, 1)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:800], y[:800], epochs=20, batch_size=32, verbose=0)
print(f"검증 MSE: {model.evaluate(X[800:], y[800:], verbose=0):.6f}")
```

---

#### 차시 32

| 항목 | 내용 |
|------|------|
| **차시명** | 비지도학습 (1) - 클러스터링 (K-means) |
| **학습목표** | - 비지도학습의 개념과 활용 사례를 설명한다 |
| | - K-means 알고리즘의 작동 원리를 이해한다 |
| | - 엘보우 방법으로 최적 클러스터 수를 결정한다 |
| **학습내용** | - 비지도학습: 레이블 없이 데이터 구조 파악 |
| | - K-means: 중심점 초기화 → 할당 → 업데이트 반복 |
| | - 엘보우 방법: 관성(inertia) 변화율로 K 결정 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: K-means 클러스터링
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 데이터 생성
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# 엘보우 방법
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias, 'bo-')
plt.xlabel('K'); plt.ylabel('Inertia')
plt.title('엘보우 방법'); plt.show()

# 최적 K=4로 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('K-means 클러스터링 결과'); plt.show()
```

---

#### 차시 33

| 항목 | 내용 |
|------|------|
| **차시명** | 비지도학습 (2) - 오토인코더 |
| **학습목표** | - 오토인코더의 구조(인코더-잠재공간-디코더)를 이해한다 |
| | - 차원 축소와 이상 탐지에서의 활용을 설명한다 |
| | - Keras로 간단한 오토인코더를 구현한다 |
| **학습내용** | - 오토인코더: 입력을 재구성하는 신경망 |
| | - 잠재 공간(Latent Space): 압축된 특징 표현 |
| | - 이상 탐지: 재구성 오차가 큰 샘플 = 이상치 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: 오토인코더
from tensorflow import keras
from tensorflow.keras import layers

# MNIST 데이터
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# 오토인코더 구성
encoder = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu')  # 잠재 공간 (32차원)
])

decoder = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(32,)),
    layers.Dense(784, activation='sigmoid')
])

autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, validation_data=(X_test, X_test))
```

---

#### 차시 34

| 항목 | 내용 |
|------|------|
| **차시명** | GAN 기초와 실습 |
| **학습목표** | - GAN(Generative Adversarial Network)의 구조를 이해한다 |
| | - 생성자(Generator)와 판별자(Discriminator)의 역할을 설명한다 |
| | - 간단한 GAN으로 이미지를 생성한다 |
| **학습내용** | - GAN: 생성자 vs 판별자의 적대적 학습 |
| | - 생성자: 랜덤 노이즈 → 가짜 이미지 생성 |
| | - 판별자: 진짜/가짜 이미지 구분 |
| | - 학습 과정: 두 네트워크의 균형있는 학습 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | **[실습과제 7]** MNIST GAN으로 손글씨 숫자 이미지 생성 실습 |

```python
# 실습 예시: 간단한 GAN 구조
from tensorflow import keras
from tensorflow.keras import layers

# 생성자 (Generator)
def build_generator(latent_dim=100):
    return keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28, 1))
    ])

# 판별자 (Discriminator)
def build_discriminator():
    return keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

generator = build_generator()
discriminator = build_discriminator()
# 학습 루프는 수업에서 구현
```

---

#### 차시 35

| 항목 | 내용 |
|------|------|
| **차시명** | 트랜스포머 구조의 이해 |
| **학습목표** | - Self-Attention 메커니즘의 원리를 이해한다 |
| | - 트랜스포머의 Encoder-Decoder 구조를 설명한다 |
| | - RNN 대비 트랜스포머의 장점(병렬 처리)을 파악한다 |
| **학습내용** | - Attention: Query, Key, Value 개념 |
| | - Self-Attention: 문장 내 단어들 간의 관계 파악 |
| | - 트랜스포머: 인코더 블록, 디코더 블록, Multi-Head Attention |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: Self-Attention 직접 구현
import numpy as np

def self_attention(Q, K, V):
    """
    Q, K, V: (seq_len, d_k) 형태의 행렬
    """
    d_k = Q.shape[-1]
    # Attention Score = Q @ K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax로 가중치 계산
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    # 가중 합
    output = weights @ V
    return output, weights

# 예시: 문장 "I love AI" (3개 단어, 임베딩 차원 4)
seq_len, d_k = 3, 4
Q = K = V = np.random.randn(seq_len, d_k)
output, attention_weights = self_attention(Q, K, V)
print("Attention 가중치:\n", attention_weights)
```

---

#### 차시 36

| 항목 | 내용 |
|------|------|
| **차시명** | LLM과 임베딩의 이해 |
| **학습목표** | - 대규모 언어 모델(LLM)의 발전 과정을 설명한다 |
| | - 단어/문장 임베딩의 개념과 활용을 이해한다 |
| | - 사전 학습(Pre-training)과 미세조정(Fine-tuning)을 구분한다 |
| **학습내용** | - GPT, BERT 등 LLM의 기본 구조 |
| | - Word2Vec, FastText, Sentence Transformers 임베딩 |
| | - 임베딩 공간에서의 유사도 계산 (코사인 유사도) |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: Sentence Transformers를 활용한 문장 임베딩
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "인공지능은 미래 기술입니다.",
    "AI는 우리 삶을 변화시킬 것입니다.",
    "오늘 날씨가 좋습니다."
]

embeddings = model.encode(sentences)
print(f"임베딩 차원: {embeddings.shape}")  # (3, 384)

# 코사인 유사도 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_01 = cosine_similarity(embeddings[0], embeddings[1])
sim_02 = cosine_similarity(embeddings[0], embeddings[2])
print(f"문장 0-1 유사도: {sim_01:.4f}")  # 높음 (AI 관련)
print(f"문장 0-2 유사도: {sim_02:.4f}")  # 낮음 (주제 다름)
```

---

### Part IV. 서비스 확장 (37~40차시)

---

#### 차시 37

| 항목 | 내용 |
|------|------|
| **차시명** | Streamlit을 활용한 웹 앱 구축 |
| **학습목표** | - Streamlit의 기본 구조와 실행 방법을 이해한다 |
| | - 학습된 모델을 Streamlit 앱에 연동한다 |
| | - 사용자 입력을 받아 예측 결과를 표시한다 |
| **학습내용** | - Streamlit 설치 및 기본 컴포넌트(text, slider, button) |
| | - st.file_uploader()로 사용자 데이터 업로드 |
| | - 저장된 모델 로드 및 예측 결과 시각화 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: Streamlit 앱 (app.py)
import streamlit as st
import pickle
import numpy as np

st.title('붓꽃 품종 예측 앱')

# 사용자 입력
sepal_length = st.slider('꽃받침 길이', 4.0, 8.0, 5.0)
sepal_width = st.slider('꽃받침 너비', 2.0, 4.5, 3.0)
petal_length = st.slider('꽃잎 길이', 1.0, 7.0, 4.0)
petal_width = st.slider('꽃잎 너비', 0.1, 2.5, 1.0)

# 모델 로드 및 예측
if st.button('예측하기'):
    # model = pickle.load(open('iris_model.pkl', 'rb'))
    # features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # prediction = model.predict(features)
    st.success(f'예측 품종: Setosa (예시)')

# 실행: streamlit run app.py
```

---

#### 차시 38

| 항목 | 내용 |
|------|------|
| **차시명** | FastAPI 기초와 모델 API 구축 |
| **학습목표** | - REST API의 기본 개념(GET, POST)을 이해한다 |
| | - FastAPI로 간단한 API 서버를 구축한다 |
| | - 학습된 모델을 API로 서빙하는 방법을 익힌다 |
| **학습내용** | - FastAPI 설치, 라우팅, 요청/응답 처리 |
| | - Pydantic을 활용한 데이터 검증 |
| | - 모델 로드 및 /predict 엔드포인트 구현 |
| **이론/실습** | 이론 + 실습 |
| **과제/프로젝트** | - |

```python
# 실습 예시: FastAPI 모델 서빙 (main.py)
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 모델 로드 (서버 시작 시 1회)
# model = pickle.load(open('iris_model.pkl', 'rb'))

@app.get("/")
def root():
    return {"message": "Iris 품종 예측 API"}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]])
    # prediction = model.predict(features)
    return {"prediction": "setosa", "probability": 0.95}

# 실행: uvicorn main:app --reload
```

---

#### 차시 39

| 항목 | 내용 |
|------|------|
| **차시명** | 모델 배포와 서비스화 사례 분석 |
| **학습목표** | - 모델 배포의 주요 방식(Cloud, Edge, Serverless)을 비교한다 |
| | - MLOps의 기본 개념과 필요성을 이해한다 |
| | - 실제 AI 서비스 사례를 분석한다 |
| **학습내용** | - 배포 환경: AWS, GCP, Azure, On-premise |
| | - 컨테이너(Docker)와 서버리스(Lambda) 개념 소개 |
| | - MLOps: 모델 버전 관리, 모니터링, 재학습 파이프라인 |
| | - 사례: 네이버 클로바, 카카오 AI, 제조 현장 AI 적용 사례 |
| **이론/실습** | 이론 (사례 분석) |
| **과제/프로젝트** | - |

---

#### 차시 40

| 항목 | 내용 |
|------|------|
| **차시명** | 과정 종합 및 최종 프로젝트 안내 |
| **학습목표** | - 전 과정의 핵심 개념을 복습하고 정리한다 |
| | - 최종 프로젝트의 주제와 평가 기준을 이해한다 |
| | - AI 분야의 추가 학습 방향을 파악한다 |
| **학습내용** | - Part I~IV 핵심 개념 요약 |
| | - 최종 프로젝트 주제 선정 및 진행 방법 안내 |
| | - AI 개발자 진로, 자격증, 추가 학습 리소스 소개 |
| **이론/실습** | 이론 |
| **과제/프로젝트** | **[최종 프로젝트]** 공개 데이터셋을 활용한 AI 예측 모델 및 웹 서비스 구축 |

---

## 실습과제 및 프로젝트 정리

### 실습과제 목록

| 차시 | 과제명 | 난이도 | 내용 |
|------|--------|--------|------|
| 3 | 실습과제 1 | 초급 | Kaggle 데이터셋 다운로드 및 구조 파악 보고서 |
| 8 | 실습과제 2 | 초급 | UCI Iris 데이터 t-검정 수행 |
| 10 | 실습과제 2 | 중급 | 공공데이터 범주형 변수 카이제곱 검정 |
| 13 | 실습과제 3 | 초급 | 기상 데이터 시계열 전처리 및 시각화 |
| 17 | 실습과제 4 | 초급 | UCI Wine 품종 분류 모델 |
| 19 | 실습과제 4 | 중급 | PCA 적용 후 분류 성능 비교 |
| 24 | 실습과제 5 | 중급 | 2층 신경망 역전파 직접 구현 |
| 28 | 실습과제 6 | 초급 | Fashion-MNIST CNN 분류 |
| 31 | 실습과제 6 | 중급 | 공공 시계열 데이터 LSTM 예측 |
| 34 | 실습과제 7 | 중급 | MNIST GAN 이미지 생성 |

### 최종 프로젝트

**주제**: 공개 데이터셋을 활용한 AI 예측 모델 및 웹 서비스 구축

**평가 기준**:
1. 문제 정의의 명확성 (10%)
2. 데이터 전처리 및 분석의 적절성 (20%)
3. 모델 선택 및 학습의 타당성 (30%)
4. 모델 평가 및 해석 (20%)
5. 서비스 구현(Streamlit/FastAPI) (10%)
6. 보고서 및 발표 (10%)

**난이도별 주제 예시**:

- **초급**: 공공데이터 기반 분류/회귀 예측 (날씨, 교통량, 판매량 등)
- **중급**: 시계열 예측 또는 이미지 분류 + Streamlit 대시보드
- **심화**: 텍스트 분류/감성분석 + FastAPI 배포 또는 YOLO 객체 탐지 앱

---

## 참고 데이터셋 목록

| 분류 | 데이터셋명 | 출처 | 활용 차시 |
|------|-----------|------|----------|
| 분류 | Iris | UCI | 17, 26 |
| 분류 | Wine | UCI | 17 |
| 분류 | Breast Cancer | sklearn | 18 |
| 이미지 | MNIST | Keras | 28 |
| 이미지 | Fashion-MNIST | Keras | 28 |
| 시계열 | 기상청 날씨 데이터 | 공공데이터포털 | 13, 31 |
| 시계열 | 서울시 대기질 | 공공데이터포털 | 31 |
| 텍스트 | 네이버 영화 리뷰 | AI Hub | 14, 36 |
| 객체탐지 | COCO (Pre-trained) | COCO | 29 |

---

*본 강의계획서는 K디지털기초역량훈련 AI 기초체력과정 양식에 맞추어 작성되었습니다.*
