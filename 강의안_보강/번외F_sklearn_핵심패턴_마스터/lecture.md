# 번외F: sklearn 핵심 패턴 마스터

## 학습 목표

| 목표 | 설명 |
|------|------|
| sklearn 설계 철학 이해 | 일관된 API 패턴 |
| fit/transform 구분 | 학습과 변환의 차이 |
| Pipeline 활용 | 워크플로우 자동화 |
| ColumnTransformer | 컬럼별 전처리 |

---

## 핵심 개념: 일관성

sklearn의 가장 큰 장점은 **일관성**임. 모든 객체가 동일한 패턴을 따름

```python
# 모델이든, 전처리기든 같은 방식
something.fit(X_train, y_train)
something.predict(X_test)  # 또는 transform(X_test)
```

---

## sklearn 객체 유형

| 객체 유형 | 역할 | 주요 메서드 |
|----------|------|------------|
| Estimator | 데이터로부터 학습 | fit() |
| Predictor | 예측 수행 | predict() |
| Transformer | 데이터 변환 | transform() |

모든 sklearn 객체는 **fit()** 메서드를 가짐

---

# Part 1: 기본 패턴 이해

## 실습 환경 설정

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

print("=" * 60)
print("번외F: sklearn 핵심 패턴 마스터")
print("=" * 60)
```

---

## 데이터 로드 및 분할

```python
# Iris 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

print(f"샘플 수: {X.shape[0]}")
print(f"특성 수: {X.shape[1]}")
print(f"특성 이름: {feature_names}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"학습 데이터: {len(X_train)}건")
print(f"테스트 데이터: {len(X_test)}건")
```

| 구분 | 크기 |
|------|------|
| 학습 데이터 | 120건 (80%) |
| 테스트 데이터 | 30건 (20%) |

---

## fit()이 하는 일

| 객체 | fit()에서 학습하는 것 |
|------|---------------------|
| LinearRegression | 계수 (coefficients) |
| DecisionTree | 분할 규칙 |
| StandardScaler | 평균, 표준편차 |
| MinMaxScaler | 최솟값, 최댓값 |
| OneHotEncoder | 범주 목록 |

---

## fit()과 transform() 분리

```python
# 스케일러 생성
scaler = StandardScaler()

# fit(): 학습 데이터로 평균/표준편차 계산
scaler.fit(X_train)

print(f"[fit() 후 저장된 값]")
print(f"평균: {scaler.mean_}")
print(f"표준편차: {scaler.scale_}")

# transform(): 저장된 값으로 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"학습 데이터 변환 후 평균: {X_train_scaled.mean(axis=0).round(2)}")
print(f"테스트 데이터 변환 후 평균: {X_test_scaled.mean(axis=0).round(2)}")
```

테스트도 학습 기준으로 변환됨 (0이 아닐 수 있음)

---

## 핵심 규칙: fit()은 학습에서만!

| 데이터 | fit() | transform() / predict() |
|--------|-------|------------------------|
| 학습 데이터 | O | O |
| 테스트 데이터 | **X** | O |

**테스트 데이터에 fit() 하면 데이터 누출!**

---

## 잘못된 사용법 vs 올바른 사용법

```python
# ❌ 잘못된 방법
scaler_wrong = StandardScaler()
X_train_wrong = scaler_wrong.fit_transform(X_train)
X_test_wrong = scaler_wrong.fit_transform(X_test)  # 잘못!
# → 둘 다 0이지만 서로 다른 기준!

# ✅ 올바른 방법
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # transform만
# → 학습 기준으로 테스트도 변환!
```

---

## fit_transform() 사용법

```python
# fit_transform(): fit + transform 한 번에
X_train_scaled = scaler.fit_transform(X_train)  # 학습용에서만!
X_test_scaled = scaler.transform(X_test)        # 테스트는 transform만
```

**주의**: fit_transform()은 **학습용에서만** 사용

---

# Part 2: 모델 학습과 예측

## 전체 흐름

```python
# 1. 스케일러를 학습 데이터로 fit_transform
X_train_scaled = scaler.fit_transform(X_train)

# 2. 테스트 데이터는 transform만
X_test_scaled = scaler.transform(X_test)

# 3. 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 4. 예측
y_pred = model.predict(X_test_scaled)

# 5. 평가
accuracy = model.score(X_test_scaled, y_test)
print(f"정확도: {accuracy:.3f}")
```

---

## 수동 관리의 문제점

```python
# 배포 시 이 모든 것을 기억해야 함
scaler = StandardScaler()
encoder = OneHotEncoder()
model = RandomForestClassifier()

# 순서도 맞춰야 함
X_scaled = scaler.transform(X)
X_encoded = encoder.transform(X_cat)
X_final = np.hstack([X_scaled, X_encoded])
y_pred = model.predict(X_final)
```

| 문제점 | 설명 |
|--------|------|
| 복잡성 | 여러 객체 관리 |
| 실수 가능성 | 순서 오류 |
| 배포 어려움 | 모든 객체 저장 필요 |

→ **Pipeline**으로 해결!

---

# Part 3: Pipeline 사용

## Pipeline이란?

여러 단계를 하나로 묶는 객체

| 장점 | 설명 |
|------|------|
| 코드 간결 | 여러 단계를 한 줄로 |
| 순서 보장 | 전처리 순서 자동 |
| 누출 방지 | fit/transform 자동 관리 |
| 배포 용이 | 하나의 객체로 저장 |

---

## Pipeline 기본 사용법

```python
from sklearn.pipeline import Pipeline

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # 1단계: 스케일링
    ('classifier', LogisticRegression(max_iter=200))    # 2단계: 모델
])

# 학습 (내부에서 fit_transform → fit 자동)
pipeline.fit(X_train, y_train)

# 예측 (내부에서 transform → predict 자동)
y_pred = pipeline.predict(X_test)

# 평가
print(f"정확도: {pipeline.score(X_test, y_test):.3f}")
```

---

## Pipeline 내부 동작

### fit() 호출 시:
```
1. scaler.fit_transform(X_train)
2. classifier.fit(X_transformed, y_train)
```

### predict() 호출 시:
```
1. scaler.transform(X_test)
2. classifier.predict(X_transformed)
```

**자동으로 fit/transform을 관리!**

---

## make_pipeline - 더 간단하게

```python
from sklearn.pipeline import make_pipeline

# 이름 자동 생성
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)

# 자동 생성된 이름
print(pipe.named_steps.keys())
# dict_keys(['standardscaler', 'logisticregression'])

# 동일하게 사용
pipe.fit(X_train, y_train)
print(f"정확도: {pipe.score(X_test, y_test):.3f}")
```

---

## Pipeline 단계 접근

```python
# 특정 단계 접근
pipe.named_steps['standardscaler']
pipe.named_steps['logisticregression']

# 학습된 파라미터 확인
print(pipe.named_steps['standardscaler'].mean_)  # 스케일러 평균
print(pipe.named_steps['logisticregression'].coef_)  # 모델 계수
```

---

# Part 4: ColumnTransformer

## 필요성

실제 데이터는 수치형과 범주형이 섞여 있음

| 컬럼 유형 | 필요한 전처리 |
|----------|--------------|
| 수치형 | StandardScaler |
| 범주형 | OneHotEncoder |

→ **ColumnTransformer**로 컬럼별 다른 전처리 적용

---

## Titanic 데이터 준비

```python
# Titanic 데이터 로드
df = sns.load_dataset('titanic')

# 특성과 타겟 분리
X_df = df[['pclass', 'age', 'fare', 'sex', 'embarked']].copy()
y_df = df['survived']

# 결측치 처리
X_df['age'].fillna(X_df['age'].median(), inplace=True)
X_df['embarked'].fillna('S', inplace=True)

# 컬럼 유형 분리
numeric_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

print(f"수치형: {numeric_features}")
print(f"범주형: {categorical_features}")
```

---

## ColumnTransformer 정의

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
```

| 요소 | 설명 |
|------|------|
| 이름 | 식별용 문자열 ('num', 'cat') |
| 변환기 | Transformer 객체 |
| 컬럼리스트 | 적용할 컬럼명 리스트 |

---

## 전체 Pipeline 구성

```python
# 전처리 + 모델 파이프라인
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 데이터 분할
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
)

# 학습
full_pipeline.fit(X_train_df, y_train_df)

# 평가
y_pred_df = full_pipeline.predict(X_test_df)
accuracy_df = accuracy_score(y_test_df, y_pred_df)
print(f"정확도: {accuracy_df:.3f}")
```

---

## 새 데이터로 예측

```python
# 새로운 승객 데이터
new_passenger = pd.DataFrame({
    'pclass': [1, 3],
    'age': [30, 25],
    'fare': [100, 10],
    'sex': ['female', 'male'],
    'embarked': ['C', 'S']
})

# 전처리 + 예측 한 번에!
predictions = full_pipeline.predict(new_passenger)
print(f"생존 예측: {predictions}")
# 0: 사망, 1: 생존
```

전처리 순서, 스케일링 기준 모두 **자동 적용**!

---

# Part 5: GridSearchCV + Pipeline

## 파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 파라미터 그리드 (파이프라인 단계명__파라미터)
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None]
}

# GridSearchCV
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 CV 점수: {grid_search.best_score_:.3f}")
print(f"테스트 정확도: {grid_search.score(X_test_df, y_test_df):.3f}")
```

---

# Part 6: 모델 저장 및 배포

## 저장 및 로드

```python
import joblib

# 최적 모델 저장 (전체 파이프라인 포함)
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'titanic_pipeline.pkl')
print("모델 저장 완료: titanic_pipeline.pkl")

# 모델 로드
loaded_model = joblib.load('titanic_pipeline.pkl')
print("모델 로드 완료")

# 로드된 모델로 예측 (전처리 자동!)
loaded_predictions = loaded_model.predict(new_passenger)
print(f"예측 결과: {loaded_predictions}")
```

전처리기와 모델이 **함께 저장**됨!

---

# sklearn 패턴 요약

## 핵심 규칙 3가지

| 규칙 | 설명 |
|------|------|
| 1 | **fit()은 학습용에서만** |
| 2 | **transform()/predict()는 동일 기준** |
| 3 | **Pipeline으로 묶으면 자동 관리** |

---

## 메서드 사용 정리

| 상황 | 사용 메서드 |
|------|-----------|
| 학습 데이터 전처리 | fit_transform() |
| 테스트 데이터 전처리 | transform() |
| 모델 학습 | fit() |
| 모델 예측 | predict() |
| 모델 평가 | score() |

---

## Pipeline 사용 정리

| Pipeline 메서드 | 내부 동작 |
|----------------|----------|
| fit(X, y) | 각 단계 fit_transform() → 마지막 fit() |
| predict(X) | 각 단계 transform() → 마지막 predict() |
| score(X, y) | 각 단계 transform() → 마지막 score() |

---

# 체크리스트

## 실무 적용 체크리스트

| 체크 | 항목 |
|------|------|
| □ | 데이터 분할 후 전처리 (전처리 후 분할 X) |
| □ | 학습 데이터로만 fit() 호출 |
| □ | 테스트 데이터는 transform()만 |
| □ | Pipeline으로 전처리 + 모델 묶기 |
| □ | ColumnTransformer로 컬럼별 처리 |
| □ | joblib로 전체 파이프라인 저장 |
| □ | 새 데이터는 파이프라인에 바로 입력 |

---

# 핵심 정리

1. **fit()**: 데이터로부터 학습 (학습용에서만!)
2. **transform()**: 학습된 기준으로 변환
3. **predict()**: 학습된 모델로 예측
4. **fit_transform()**: fit + transform (학습용에서만!)
5. **Pipeline**: 전처리 + 모델을 하나로
6. **ColumnTransformer**: 컬럼별 다른 전처리

---

## 다음 학습

- **12차시~**: 개별 모델 심화 학습
- **16차시**: 모델 평가와 교차검증
- **24차시**: 모델 저장과 배포
