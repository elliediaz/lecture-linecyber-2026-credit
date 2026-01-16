---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 10차시'
footer: '공공데이터를 활용한 AI 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
  table { font-size: 0.9em; }
  .highlight { background-color: #fef3c7; padding: 10px; border-radius: 8px; }
  .important { background-color: #fee2e2; padding: 10px; border-radius: 8px; }
  .tip { background-color: #d1fae5; padding: 10px; border-radius: 8px; }
---

# 제조 데이터 전처리 (2)

## 10차시 | Part II. 기초 수리와 데이터 분석

**스케일링, 인코딩, 특성 엔지니어링**

---

# 지난 시간 복습

## 8차시에서 배운 내용

### 결측치 처리
- `isnull()`, `fillna()`, `dropna()`
- 수치형: 평균/중앙값, 범주형: 최빈값

### 이상치 처리
- IQR: Q1 - 1.5×IQR ~ Q3 + 1.5×IQR
- Z-score: |Z| > 2 또는 3
- Clipping, 제거, 플래그

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **스케일링(정규화, 표준화)의 필요성을 이해**한다 |
| 2 | **범주형 데이터 인코딩 방법**을 적용한다 |
| 3 | **sklearn 전처리 도구**를 활용한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  스케일링   │ → │   인코딩    │ → │ sklearn 도구 │
└─────────────┘    └─────────────┘    └─────────────┘
   표준화             레이블            Pipeline
   정규화             원-핫             전처리 흐름
   RobustScaler       특성 조합          역변환
```

---

<!-- _class: lead -->

# Part 1

## 스케일링(정규화, 표준화)의 필요성 이해

---

# 왜 스케일링이 필요한가?

## 제조 데이터의 스케일 차이

```
온도:     80 ~ 100    (범위 20)
생산량:   1000 ~ 1500 (범위 500)
습도:     40 ~ 80     (범위 40)
불량률:   0.01 ~ 0.05 (범위 0.04)
```

### 문제점
- 스케일이 큰 변수가 모델에 **과도한 영향**
- 거리 기반 알고리즘에서 **왜곡**
- 경사하강법 **수렴 속도 저하**

---

# 스케일링이 필요한 알고리즘

## 스케일 영향을 받는 알고리즘

| 알고리즘 | 스케일링 필요성 | 이유 |
|----------|:-------------:|------|
| KNN | **필수** | 거리 계산에 직접 영향 |
| SVM | **필수** | 서포트 벡터 결정에 영향 |
| 선형회귀 | 권장 | 계수 해석, 수렴 속도 |
| 신경망 | **필수** | 활성화 함수, 경사하강법 |
| 의사결정나무 | 불필요 | 분기점 기반 (순위만 사용) |
| 랜덤포레스트 | 불필요 | 트리 기반 앙상블 |

---

# 스케일링 미적용 시 문제

## 거리 기반 알고리즘 예시

```python
# 두 제품 간 거리 계산 (스케일링 없이)
제품A: [온도=80, 생산량=1000]
제품B: [온도=90, 생산량=1200]

거리 = √((90-80)² + (1200-1000)²)
     = √(100 + 40000)
     = √40100 ≈ 200.25
```

### 문제점
- 온도 차이(10) vs 생산량 차이(200)
- **생산량이 거리를 지배** → 온도 영향력 무시됨

---

# 표준화 (Standardization)

## Z-score 변환

### 공식
$$Z = \frac{X - \mu}{\sigma}$$

- μ: 평균 (mean)
- σ: 표준편차 (std)

### 결과
- **평균 = 0**
- **표준편차 = 1**
- 값의 범위: 대략 -3 ~ +3 (정규분포 가정)

---

# StandardScaler 사용법

## sklearn 구현

```python
from sklearn.preprocessing import StandardScaler

# 1. 스케일러 생성
scaler = StandardScaler()

# 2. 학습 데이터로 fit (평균, 표준편차 학습)
scaler.fit(X_train)

# 3. 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 같은 기준으로!

# 또는 한번에
X_train_scaled = scaler.fit_transform(X_train)
```

---

# 표준화 예시

## 실제 데이터 변환

| 원본 온도 | Z-score |
|:---------:|:-------:|
| 70 | -2.0 |
| 80 | -1.0 |
| 90 | 0.0 |
| 100 | 1.0 |
| 110 | 2.0 |

> 평균=90, 표준편차=10인 경우
> Z = (100 - 90) / 10 = 1.0

---

# 정규화 (Min-Max Scaling)

## 0~1 범위로 변환

### 공식
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

### 결과
- **최솟값 → 0**
- **최댓값 → 1**
- 모든 값: 0 ~ 1 범위

---

# MinMaxScaler 사용법

## sklearn 구현

```python
from sklearn.preprocessing import MinMaxScaler

# 1. 스케일러 생성
scaler = MinMaxScaler()

# 2. 학습 데이터로 fit (min, max 학습)
scaler.fit(X_train)

# 3. 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# feature_range 옵션으로 범위 조절 가능
scaler = MinMaxScaler(feature_range=(0, 1))  # 기본값
scaler = MinMaxScaler(feature_range=(-1, 1)) # -1~1 범위
```

---

# 정규화 예시

## 실제 데이터 변환

| 원본 온도 | Min-Max |
|:---------:|:-------:|
| 70 (min) | 0.00 |
| 80 | 0.25 |
| 90 | 0.50 |
| 100 | 0.75 |
| 110 (max) | 1.00 |

> min=70, max=110
> X_scaled = (90 - 70) / (110 - 70) = 20/40 = 0.5

---

# RobustScaler

## 이상치에 강건한 스케일링

### 공식
$$X_{scaled} = \frac{X - Q2}{Q3 - Q1} = \frac{X - median}{IQR}$$

### 특징
- 중앙값(Q2)과 IQR 사용
- **이상치 영향 최소화**
- 평균/표준편차 대신 중앙값/IQR 사용

---

# RobustScaler 사용법

## sklearn 구현

```python
from sklearn.preprocessing import RobustScaler

# 이상치가 있는 데이터
X = [[100], [80], [90], [85], [500]]  # 500은 이상치

# RobustScaler 적용
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
# 이상치(500)가 있어도 다른 값들이 적절히 스케일링됨
```

---

# 스케일링 방법 비교

## 상황별 선택 가이드

| 방법 | 결과 | 사용 시점 | 이상치 |
|------|------|----------|:------:|
| **StandardScaler** | 평균0, std1 | 일반적인 ML | 민감 |
| **MinMaxScaler** | [0, 1] | 신경망, 이미지 | 매우 민감 |
| **RobustScaler** | 중앙값 기준 | 이상치 多 | 강건 |

<div class="tip">

**권장 순서**: StandardScaler → 이상치 있으면 RobustScaler → 신경망이면 MinMaxScaler

</div>

---

# 스케일링 시각화

## 변환 전후 분포 비교

```
원본 데이터:
온도     [70 ─────●───── 110]
생산량   [900 ────────────●────────────── 1500]

StandardScaler 후:
온도     [-3 ───●─── +3]
생산량   [-3 ───●─── +3]

MinMaxScaler 후:
온도     [0 ───●─── 1]
생산량   [0 ───●─── 1]
```

---

# 스케일링 주의사항

## 핵심 원칙

<div class="important">

### 데이터 누수 방지
```python
# ❌ 잘못된 방법
scaler.fit(전체_데이터)  # 테스트 정보가 학습에 포함!

# ✅ 올바른 방법
scaler.fit(X_train)      # 학습 데이터만으로 fit
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 같은 기준으로 변환
```

</div>

> fit은 학습 데이터만! transform은 모든 데이터에!

---

<!-- _class: lead -->

# Part 2

## 범주형 데이터 인코딩 방법

---

# 범주형 데이터란?

## 숫자가 아닌 데이터

### 제조 현장의 범주형 데이터 예시
```python
라인:      ['A', 'B', 'C']
등급:      ['상', '중', '하']
불량유형:  ['스크래치', '찍힘', '변색', '기타']
장비:      ['1호기', '2호기', '3호기']
작업자:    ['김', '이', '박', '최']
```

### 문제점
- 대부분의 ML 모델은 **숫자만 입력 가능**
- 범주를 숫자로 변환해야 함 → **인코딩**

---

# 범주형 데이터 유형

## 명목형 vs 순서형

### 명목형 (Nominal)
- 순서/크기 관계 **없음**
- 예: 라인(A, B, C), 불량유형, 색상

### 순서형 (Ordinal)
- 순서/크기 관계 **있음**
- 예: 등급(상, 중, 하), 학력, 만족도(1~5)

---

# 레이블 인코딩 (Label Encoding)

## 범주를 숫자로 매핑

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['등급_encoded'] = le.fit_transform(df['등급'])
```

### 결과
```
등급 → 등급_encoded
하   → 0
상   → 1
중   → 2
```

<div class="important">

**주의**: 알파벳순 정렬! (가나다순)

</div>

---

# 레이블 인코딩 특징

## 장점과 단점

### 장점
- 구현이 간단
- 메모리 효율적 (컬럼 1개 유지)
- 트리 기반 모델에서 잘 작동

### 단점
- **숫자 크기에 의미가 생김** (0 < 1 < 2)
- 순서가 없는 범주에 부적합
- 선형 모델에서 왜곡 발생

---

# 레이블 인코딩 적용 시점

## 언제 사용하나?

### 적합한 경우
- **순서가 있는** 범주형 데이터
  - 등급: 하 < 중 < 상
  - 만족도: 1 < 2 < 3 < 4 < 5
- **트리 기반 모델** 사용 시

### 부적합한 경우
- 순서가 없는 범주 (라인 A, B, C)
- 선형 회귀, SVM 등 거리/크기 사용 모델

---

# 원-핫 인코딩 (One-Hot Encoding)

## 이진 벡터로 변환

```python
# pandas 방법 (가장 간편!)
df_encoded = pd.get_dummies(df, columns=['라인'])
```

### 결과
```
라인  →  라인_A  라인_B  라인_C
 A         1       0       0
 B         0       1       0
 C         0       0       1
```

---

# 원-핫 인코딩 원리

## 각 범주를 독립 컬럼으로

```
원본: [A, B, C, A, B]

변환 후:
    라인_A  라인_B  라인_C
0     1      0      0
1     0      1      0
2     0      0      1
3     1      0      0
4     0      1      0
```

- 해당 범주면 1, 아니면 0
- **순서/크기 관계 없음** → 공정한 표현

---

# pd.get_dummies 사용법

## pandas 원-핫 인코딩

```python
import pandas as pd

# 기본 사용
df_encoded = pd.get_dummies(df, columns=['라인', '장비'])

# 접두어 지정
df_encoded = pd.get_dummies(df, columns=['라인'], prefix='line')
# 결과: line_A, line_B, line_C

# drop_first 옵션 (다중공선성 방지)
df_encoded = pd.get_dummies(df, columns=['라인'], drop_first=True)
# 결과: 라인_B, 라인_C (라인_A는 생략)
```

---

# OneHotEncoder (sklearn)

## sklearn 원-핫 인코딩

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['라인']])

# 컬럼명 확인
feature_names = encoder.get_feature_names_out(['라인'])
print(feature_names)  # ['라인_A', '라인_B', '라인_C']

# DataFrame으로 변환
df_encoded = pd.DataFrame(encoded, columns=feature_names)
```

---

# get_dummies vs OneHotEncoder

## 언제 무엇을 쓸까?

| 방법 | 장점 | 단점 | 사용 시점 |
|------|------|------|----------|
| **pd.get_dummies** | 간편함 | 새 범주 처리 어려움 | EDA, 빠른 실험 |
| **OneHotEncoder** | 새 범주 처리 | 코드 복잡 | 프로덕션 |

```python
# OneHotEncoder: 새로운 범주 처리
encoder = OneHotEncoder(handle_unknown='ignore')
# 학습 시 없던 범주 → 모두 0으로 처리
```

---

# 다중공선성 문제

## drop_first가 필요한 이유

```
라인_A + 라인_B + 라인_C = 1 (항상!)
```

### 문제점
- 세 변수가 **완벽한 선형 관계**
- 선형 모델에서 계수 추정 불안정

### 해결책
```python
# 하나의 컬럼 제거
df_encoded = pd.get_dummies(df, columns=['라인'], drop_first=True)
# 라인_A 제거 → 라인_B, 라인_C만 사용
# 라인_B=0, 라인_C=0 이면 라인_A임을 알 수 있음
```

---

# 인코딩 선택 가이드

## 데이터 유형별 추천

| 데이터 유형 | 예시 | 권장 방법 |
|------------|------|----------|
| 순서 있는 범주 | 등급(상/중/하) | 레이블 인코딩 |
| 순서 없는 범주 | 라인(A/B/C) | 원-핫 인코딩 |
| 고유값 많음 | 제품코드 | 타겟/빈도 인코딩 |
| 이진 범주 | 성별(남/여) | 레이블 or 원-핫 |

---

# 고차원 범주 처리

## 고유값이 많을 때

```python
# 제품코드: 1000개 이상의 고유값
# 원-핫 인코딩 시 1000개 컬럼 생성 → 비효율!
```

### 대안
1. **빈도 인코딩**: 범주별 출현 빈도
2. **타겟 인코딩**: 범주별 타겟 평균
3. **그룹화**: 상위 N개만 유지, 나머지 '기타'

```python
# 빈도 인코딩
freq = df['제품코드'].value_counts(normalize=True)
df['제품코드_freq'] = df['제품코드'].map(freq)
```

---

<!-- _class: lead -->

# Part 3

## sklearn 전처리 도구 활용

---

# sklearn 전처리 흐름

## 일반적인 워크플로우

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  fit()   │ →  │transform()│ →  │  모델    │
│(학습데이터)│    │(모든 데이터)│    │ 학습/예측 │
└──────────┘    └──────────┘    └──────────┘
```

### 핵심 메서드
- `fit(X)`: 학습 데이터에서 파라미터 학습
- `transform(X)`: 데이터 변환
- `fit_transform(X)`: fit + transform (학습 데이터용)

---

# 전처리 클래스 구조

## sklearn 일관된 API

```python
from sklearn.preprocessing import StandardScaler

# 모든 전처리 클래스가 동일한 패턴
scaler = StandardScaler()        # 1. 객체 생성
scaler.fit(X_train)              # 2. 학습
X_train_scaled = scaler.transform(X_train)  # 3. 변환
X_test_scaled = scaler.transform(X_test)    # 4. 테스트 변환

# 학습된 파라미터 확인
print(scaler.mean_)      # 학습된 평균
print(scaler.scale_)     # 학습된 표준편차
```

---

# 역변환 (Inverse Transform)

## 원래 값으로 복원

```python
# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 모델 예측 (스케일링된 값으로)
y_pred_scaled = model.predict(X_test_scaled)

# 역변환 (원래 스케일로)
X_original = scaler.inverse_transform(X_scaled)
```

### 활용
- 예측 결과를 원래 단위로 표시
- 해석 가능성 향상

---

# ColumnTransformer

## 컬럼별 다른 전처리 적용

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 컬럼별 전처리 정의
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['온도', '습도', '생산량']),
        ('cat', OneHotEncoder(), ['라인', '장비'])
    ]
)

# 적용
X_processed = preprocessor.fit_transform(df)
```

---

# ColumnTransformer 구조

## 여러 전처리를 한번에

```
입력 데이터
┌───────────────────────────────────┐
│ 온도 | 습도 | 생산량 | 라인 | 장비 │
└───────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│     ColumnTransformer               │
│  ┌─────────────┬─────────────────┐  │
│  │StandardScaler│  OneHotEncoder │  │
│  │ (수치형)     │   (범주형)      │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
          ↓
┌───────────────────────────────────────────────┐
│ 온도_std | 습도_std | 생산량_std | 라인_A | ... │
└───────────────────────────────────────────────┘
```

---

# Pipeline

## 전처리 + 모델을 연결

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 파이프라인 정의
pipe = Pipeline([
    ('preprocessor', preprocessor),  # 전처리
    ('classifier', LogisticRegression())  # 모델
])

# 학습 (전처리 + 모델 학습 자동)
pipe.fit(X_train, y_train)

# 예측 (전처리 + 예측 자동)
y_pred = pipe.predict(X_test)
```

---

# Pipeline 장점

## 왜 Pipeline을 사용하나?

### 1. 코드 간결화
```python
# Pipeline 없이
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Pipeline 사용
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 2. 데이터 누수 방지
### 3. 교차 검증 용이

---

# Pipeline + GridSearchCV

## 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 파이프라인 내 파라미터 접근
param_grid = {
    'preprocessor__num__with_mean': [True, False],
    'classifier__C': [0.1, 1, 10]
}

# GridSearch 적용
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"최적 파라미터: {grid.best_params_}")
```

---

# 전처리 저장/로드

## 모델과 함께 저장

```python
import joblib

# 저장 (전처리 포함 파이프라인)
joblib.dump(pipe, 'model_pipeline.pkl')

# 로드
loaded_pipe = joblib.load('model_pipeline.pkl')

# 바로 예측 가능 (전처리 자동 적용)
y_pred = loaded_pipe.predict(new_data)
```

<div class="tip">

**실무 팁**: 전처리기와 모델을 항상 함께 저장하세요!

</div>

---

# 수치형 전처리 종합

## Imputer + Scaler 조합

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 수치형 전처리 파이프라인
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 결측치
    ('scaler', StandardScaler())                    # 스케일링
])
```

---

# 범주형 전처리 종합

## Imputer + Encoder 조합

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 범주형 전처리 파이프라인
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측치
    ('encoder', OneHotEncoder(handle_unknown='ignore'))    # 인코딩
])
```

---

# 종합 전처리 파이프라인

## 수치형 + 범주형 통합

```python
from sklearn.compose import ColumnTransformer

numeric_features = ['온도', '습도', '생산량']
categorical_features = ['라인', '장비']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 최종 파이프라인
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

---

# 전처리 파이프라인 시각화

## 전체 흐름

```
원본 데이터
    ↓
┌───────────────────────────────────────┐
│         ColumnTransformer            │
│  ┌────────────────┬─────────────────┐ │
│  │ numeric_trans  │ categoric_trans │ │
│  │ ┌──────────┐  │  ┌──────────┐   │ │
│  │ │ Imputer  │  │  │ Imputer  │   │ │
│  │ │ Scaler   │  │  │ Encoder  │   │ │
│  │ └──────────┘  │  └──────────┘   │ │
│  └────────────────┴─────────────────┘ │
└───────────────────────────────────────┘
    ↓
  모델 (Classifier/Regressor)
```

---

<!-- _class: lead -->

# 실습편

## 스케일링, 인코딩, 파이프라인 실습

---

# 실습 개요

## 제조 데이터 전처리 실습

### 실습 목표
1. StandardScaler, MinMaxScaler 적용
2. LabelEncoder, OneHotEncoder 적용
3. Pipeline 구성 및 활용

### 실습 데이터
- 제조 센서 데이터 (온도, 습도, 생산량)
- 범주형 데이터 (라인, 등급)

---

# 실습 1: 데이터 준비

## 샘플 데이터 생성

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(60, 10, n),
    '생산량': np.random.normal(1200, 50, n),
    '라인': np.random.choice(['A', 'B', 'C'], n),
    '등급': np.random.choice(['상', '중', '하'], n),
    '불량': np.random.choice([0, 1], n, p=[0.8, 0.2])
})

print(df.head())
print(df.info())
```

---

# 실습 2: 스케일 차이 확인

## 변수별 범위 비교

```python
print("=== 변수별 통계 ===")
print(df[['온도', '습도', '생산량']].describe().round(2))

print("\n=== 변수별 범위 ===")
for col in ['온도', '습도', '생산량']:
    col_range = df[col].max() - df[col].min()
    print(f"{col}: {df[col].min():.1f} ~ {df[col].max():.1f} (범위: {col_range:.1f})")
```

---

# 실습 3: 표준화 적용

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler

numeric_cols = ['온도', '습도', '생산량']
X = df[numeric_cols].values

# 표준화
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# DataFrame으로 변환
df_std = pd.DataFrame(X_std, columns=numeric_cols)

print("=== 표준화 후 ===")
print(df_std.describe().round(3))
```

---

# 실습 4: 정규화 적용

## MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# 정규화
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# DataFrame으로 변환
df_mm = pd.DataFrame(X_mm, columns=numeric_cols)

print("=== 정규화 후 ===")
print(df_mm.describe().round(3))
```

---

# 실습 5: 스케일링 비교 시각화

## 전/후 분포 비교

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=20, alpha=0.5, label='원본')
    axes[i].hist(df_std[col] * df[col].std() + df[col].mean(),
                 bins=20, alpha=0.5, label='표준화(역변환)')
    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.show()
```

---

# 실습 6: 레이블 인코딩

## 순서형 범주 변환

```python
from sklearn.preprocessing import LabelEncoder

# 등급 인코딩 (순서 있음)
le = LabelEncoder()
df['등급_encoded'] = le.fit_transform(df['등급'])

print("=== 레이블 인코딩 결과 ===")
print(df[['등급', '등급_encoded']].drop_duplicates())
print(f"\n클래스: {le.classes_}")

# 역변환
original = le.inverse_transform([0, 1, 2])
print(f"역변환: {original}")
```

---

# 실습 7: 원-핫 인코딩 (pandas)

## get_dummies 활용

```python
# pandas 방법
df_onehot = pd.get_dummies(df, columns=['라인'], prefix='라인')

print("=== 원-핫 인코딩 결과 ===")
print(df_onehot[['라인_A', '라인_B', '라인_C']].head(10))

# drop_first 옵션
df_onehot_drop = pd.get_dummies(df, columns=['라인'], drop_first=True)
print(f"\ndrop_first 적용 컬럼: {df_onehot_drop.columns.tolist()}")
```

---

# 실습 8: 원-핫 인코딩 (sklearn)

## OneHotEncoder 활용

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
line_encoded = encoder.fit_transform(df[['라인']])

# 컬럼명 생성
feature_names = encoder.get_feature_names_out(['라인'])
df_line_enc = pd.DataFrame(line_encoded, columns=feature_names)

print("=== OneHotEncoder 결과 ===")
print(df_line_enc.head())
print(f"\n카테고리: {encoder.categories_}")
```

---

# 실습 9: 종합 전처리

## 수치형 + 범주형 통합

```python
from sklearn.compose import ColumnTransformer

numeric_features = ['온도', '습도', '생산량']
categorical_features = ['라인']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(df)
print(f"전처리 후 shape: {X_processed.shape}")
```

---

# 실습 10: Pipeline 구성

## 전처리 + 모델 연결

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 데이터 분할
X = df[['온도', '습도', '생산량', '라인']]
y = df['불량']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 파이프라인
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(f"정확도: {pipe.score(X_test, y_test):.3f}")
```

---

# 실습 11: 파이프라인 저장

## 모델 저장 및 로드

```python
import joblib

# 저장
joblib.dump(pipe, 'manufacturing_pipeline.pkl')
print("파이프라인 저장 완료!")

# 로드
loaded_pipe = joblib.load('manufacturing_pipeline.pkl')

# 새 데이터 예측
new_data = pd.DataFrame({
    '온도': [87.5],
    '습도': [62.0],
    '생산량': [1250],
    '라인': ['B']
})

pred = loaded_pipe.predict(new_data)
print(f"예측 결과: {pred[0]} (0: 양품, 1: 불량)")
```

---

# 실습 정리

## 핵심 체크포인트

### 스케일링
- [ ] StandardScaler: 평균0, 표준편차1
- [ ] MinMaxScaler: 0~1 범위
- [ ] fit은 학습 데이터만!

### 인코딩
- [ ] LabelEncoder: 순서 있는 범주
- [ ] OneHotEncoder / get_dummies: 순서 없는 범주

### Pipeline
- [ ] ColumnTransformer: 컬럼별 전처리
- [ ] Pipeline: 전처리 + 모델 연결

---

# 핵심 정리

## 10차시 요약

| 구분 | 핵심 내용 |
|------|----------|
| **스케일링** | StandardScaler(일반), MinMaxScaler(신경망), RobustScaler(이상치) |
| **인코딩** | LabelEncoder(순서O), OneHotEncoder(순서X) |
| **Pipeline** | ColumnTransformer + Pipeline으로 자동화 |
| **주의사항** | fit은 학습 데이터만, 전처리기와 모델 함께 저장 |

---

# 다음 차시 예고

## 10차시: 제조 데이터 탐색 분석 종합

### 학습 내용
- EDA 전체 워크플로우
- 데이터 이해부터 인사이트 도출까지
- 제조 데이터 종합 분석 프로젝트

### 준비물
- 1~10차시 내용 복습
- Python 환경 점검

---

# 감사합니다

## 10차시: 제조 데이터 전처리 (2)

**Q&A**

다음 시간에 EDA 종합 실습으로 만나요!
