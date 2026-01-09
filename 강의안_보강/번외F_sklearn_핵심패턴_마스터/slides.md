---
marp: true
theme: default
paginate: true
---

# 번외F: sklearn 핵심 패턴 마스터

## fit, transform, Pipeline 완전 정복

---

# 학습 목표

1. **sklearn 설계 철학** 이해하기
2. **fit / predict / transform** 완전히 구분하기
3. **Pipeline**으로 워크플로우 자동화하기
4. **ColumnTransformer**로 다른 전처리 적용하기

---

# 왜 sklearn 패턴을 배워야 하는가?

```
sklearn을 "사용"하는 것과 "이해"하는 것은 다르다
```

| 레벨 | 상태 | 문제점 |
|------|------|--------|
| 초급 | 예제 복붙 | 왜 되는지 모름 |
| 중급 | API 이해 | 패턴 적용 가능 |
| **고급** | **설계 철학 이해** | **어떤 상황에도 대응** |

---

# sklearn의 설계 철학

## 일관성 (Consistency)

**모든 객체가 동일한 패턴**을 따름

```python
# 모델이든, 전처리기든 같은 방식!
something.fit(X_train, y_train)
something.predict(X_test)  # 또는 transform(X_test)
```

---

# sklearn의 3가지 핵심 객체

| 객체 유형 | 역할 | 예시 |
|----------|------|------|
| **Estimator** | 데이터로부터 학습 | 모든 sklearn 객체 |
| **Predictor** | 예측 수행 | 분류기, 회귀 모델 |
| **Transformer** | 데이터 변환 | Scaler, Encoder |

모두 **fit()** 메서드를 가짐!

---

# fit() - 가장 중요한 메서드

## "데이터로부터 학습한다"

```python
# 모델: 패턴 학습
model.fit(X_train, y_train)

# 스케일러: 평균, 표준편차 계산
scaler.fit(X_train)

# 인코더: 범주 목록 학습
encoder.fit(X_train)
```

---

# fit()이 하는 일

| 객체 | fit()에서 학습하는 것 |
|------|---------------------|
| LinearRegression | 계수 (coefficients) |
| DecisionTree | 분할 규칙 |
| StandardScaler | 평균, 표준편차 |
| MinMaxScaler | 최솟값, 최댓값 |
| OneHotEncoder | 범주 목록 |
| LabelEncoder | 레이블 매핑 |

---

# predict() vs transform()

## 둘 다 fit() 이후에 호출

| 메서드 | 역할 | 사용 객체 |
|--------|------|----------|
| **predict()** | 예측값 반환 | 분류기, 회귀 모델 |
| **transform()** | 변환된 데이터 반환 | Scaler, Encoder |

---

# predict() 예시

```python
from sklearn.linear_model import LogisticRegression

# 1. 모델 생성
model = LogisticRegression()

# 2. 학습 (패턴 학습)
model.fit(X_train, y_train)

# 3. 예측 (학습된 패턴으로 예측)
y_pred = model.predict(X_test)
```

---

# transform() 예시

```python
from sklearn.preprocessing import StandardScaler

# 1. 스케일러 생성
scaler = StandardScaler()

# 2. 학습 (평균, 표준편차 계산)
scaler.fit(X_train)

# 3. 변환 (계산된 값으로 정규화)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 핵심: fit()은 학습용에서만!

```
⚠️ 가장 중요한 규칙
```

| 데이터 | fit() | transform() / predict() |
|--------|-------|------------------------|
| 학습 데이터 | ✅ O | ✅ O |
| 테스트 데이터 | ❌ X | ✅ O |

**테스트 데이터에 fit() 하면 데이터 누출!**

---

# 왜 테스트에 fit() 하면 안 되는가?

```python
# ❌ 잘못된 방법
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler.fit(X_test)  # 테스트 기준으로 다시 학습!
X_test_scaled = scaler.transform(X_test)
```

→ 학습과 테스트의 **기준이 달라짐**
→ 실제 배포 시 **새 데이터마다 기준이 바뀜**

---

# 올바른 방법

```python
# ✅ 올바른 방법
scaler.fit(X_train)  # 학습용으로만 기준 설정
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 같은 기준!
```

→ 학습과 테스트가 **동일한 기준**
→ 실제 배포 시에도 **일관된 변환**

---

# fit_transform() - 편의 메서드

## fit() + transform() 한 번에

```python
# 이 두 줄을
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# 한 줄로
X_train_scaled = scaler.fit_transform(X_train)
```

**주의**: fit_transform()은 **학습용에서만** 사용!

---

# 전체 흐름 정리

```python
# 1. 학습용: fit_transform() 사용
X_train_scaled = scaler.fit_transform(X_train)

# 2. 테스트용: transform()만 사용
X_test_scaled = scaler.transform(X_test)

# 3. 모델 학습
model.fit(X_train_scaled, y_train)

# 4. 예측
y_pred = model.predict(X_test_scaled)
```

---

# 실습: 기본 패턴 확인

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# 실습: 스케일링 + 모델링

```python
# 스케일러
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # transform만

# 모델
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 평가
print(f"정확도: {model.score(X_test_scaled, y_test):.3f}")
```

---

# 문제점: 수동 관리의 어려움

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

→ **Pipeline**으로 해결!

---

# Pipeline이란?

## 여러 단계를 하나로 묶음

```
[스케일링] → [모델] 을 하나의 객체로
```

| 장점 | 설명 |
|------|------|
| 코드 간결 | 여러 단계를 한 줄로 |
| 순서 보장 | 전처리 순서 자동 |
| 누출 방지 | fit/transform 자동 관리 |
| 배포 용이 | 하나의 객체로 저장 |

---

# Pipeline 기본 사용법

```python
from sklearn.pipeline import Pipeline

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # 1단계
    ('model', LogisticRegression())    # 2단계
])

# 학습 (내부에서 fit_transform → fit 자동)
pipeline.fit(X_train, y_train)

# 예측 (내부에서 transform → predict 자동)
y_pred = pipeline.predict(X_test)
```

---

# Pipeline 내부 동작

## fit() 호출 시

```
1. scaler.fit_transform(X_train)
2. model.fit(X_transformed, y_train)
```

## predict() 호출 시

```
1. scaler.transform(X_test)
2. model.predict(X_transformed)
```

**자동으로 fit/transform을 관리!**

---

# Pipeline 실습

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 파이프라인 생성
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# 한 번에 학습
pipe.fit(X_train, y_train)

# 한 번에 예측
print(f"정확도: {pipe.score(X_test, y_test):.3f}")
```

---

# make_pipeline - 더 간단하게

```python
from sklearn.pipeline import make_pipeline

# 이름 자동 생성
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# 동일하게 사용
pipe.fit(X_train, y_train)
print(f"정확도: {pipe.score(X_test, y_test):.3f}")
```

---

# Pipeline 단계 접근

```python
# 특정 단계 접근
pipe.named_steps['scaler']
pipe.named_steps['logisticregression']

# 학습된 파라미터 확인
print(pipe.named_steps['scaler'].mean_)  # 스케일러 평균
print(pipe.named_steps['logisticregression'].coef_)  # 모델 계수
```

---

# 문제: 다른 전처리가 필요할 때

```python
# 수치형: 스케일링
# 범주형: 인코딩

# 다른 처리가 필요!
```

→ **ColumnTransformer** 사용!

---

# ColumnTransformer란?

## 컬럼별로 다른 변환 적용

```
수치형 컬럼 → StandardScaler
범주형 컬럼 → OneHotEncoder
```

| 장점 | 설명 |
|------|------|
| 유연성 | 컬럼별 다른 처리 |
| 일관성 | 하나의 transformer로 관리 |
| Pipeline 통합 | 전체 워크플로우에 포함 |

---

# ColumnTransformer 기본 구조

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(), ['gender', 'city'])
    ]
)
```

---

# ColumnTransformer 형식

```python
ColumnTransformer([
    (이름, 변환기, 컬럼리스트),
    (이름, 변환기, 컬럼리스트),
    ...
])
```

| 요소 | 설명 |
|------|------|
| 이름 | 식별용 문자열 |
| 변환기 | Transformer 객체 |
| 컬럼리스트 | 적용할 컬럼명 리스트 |

---

# 실습: Titanic 데이터 전처리

```python
import pandas as pd
import seaborn as sns

# 데이터 로드
df = sns.load_dataset('titanic')

# 특성과 타겟 분리
X = df[['pclass', 'age', 'fare', 'sex', 'embarked']].copy()
y = df['survived']

# 결측치 처리 (간단히)
X['age'].fillna(X['age'].median(), inplace=True)
X['embarked'].fillna('S', inplace=True)
```

---

# 컬럼 유형 분리

```python
# 수치형 컬럼
numeric_features = ['age', 'fare']

# 범주형 컬럼
categorical_features = ['pclass', 'sex', 'embarked']
```

---

# ColumnTransformer 정의

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
```

---

# 전체 Pipeline 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 전처리 + 모델 파이프라인
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# 학습 및 평가

```python
# 한 번에 학습
full_pipeline.fit(X_train, y_train)

# 한 번에 예측
y_pred = full_pipeline.predict(X_test)

# 평가
from sklearn.metrics import accuracy_score
print(f"정확도: {accuracy_score(y_test, y_pred):.3f}")
```

---

# Pipeline의 강력함

```python
# 새로운 데이터가 와도 한 줄로 예측!
new_data = pd.DataFrame({
    'pclass': [1],
    'age': [30],
    'fare': [100],
    'sex': ['female'],
    'embarked': ['C']
})

prediction = full_pipeline.predict(new_data)
print(f"생존 예측: {prediction[0]}")
```

전처리 순서, 스케일링 기준 모두 **자동 적용**!

---

# Pipeline + GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# 파라미터 그리드 (파이프라인 단계명__파라미터)
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None]
}

# 그리드 서치
grid_search = GridSearchCV(
    full_pipeline, param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 점수: {grid_search.best_score_:.3f}")
```

---

# 모델 저장 및 배포

```python
import joblib

# 저장 (전체 파이프라인 포함)
joblib.dump(full_pipeline, 'titanic_model.pkl')

# 불러오기
loaded_pipeline = joblib.load('titanic_model.pkl')

# 바로 사용 가능!
prediction = loaded_pipeline.predict(new_data)
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

# 메서드 사용 정리

| 상황 | 사용 메서드 |
|------|-----------|
| 학습 데이터 전처리 | fit_transform() |
| 테스트 데이터 전처리 | transform() |
| 모델 학습 | fit() |
| 모델 예측 | predict() |
| 모델 평가 | score() |

---

# Pipeline 사용 정리

| Pipeline 메서드 | 내부 동작 |
|----------------|----------|
| fit(X, y) | 각 단계 fit_transform() → 마지막 fit() |
| predict(X) | 각 단계 transform() → 마지막 predict() |
| score(X, y) | 각 단계 transform() → 마지막 score() |

---

# 최종 실습: 완전한 ML 워크플로우

```python
# 1. 전처리기 정의
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# 2. 파이프라인 구성
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier())
])

# 3. GridSearch로 최적화
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

# 4. 저장 및 배포
joblib.dump(grid.best_estimator_, 'model.pkl')
```

---

# 핵심 정리

1. **fit()**: 데이터로부터 학습 (학습용에서만!)
2. **transform()**: 학습된 기준으로 변환
3. **predict()**: 학습된 모델로 예측
4. **fit_transform()**: fit + transform (학습용에서만!)
5. **Pipeline**: 전처리 + 모델을 하나로
6. **ColumnTransformer**: 컬럼별 다른 전처리

---

# 다음 학습

- **12차시~**: 개별 모델 심화 학습
- **16차시**: 모델 평가와 교차검증
- **24차시**: 모델 저장과 배포

---

# 참고: sklearn 공식 문서

```
https://scikit-learn.org/stable/
```

- User Guide: 개념 설명
- API Reference: 모든 클래스/함수
- Examples: 실제 사용 예시
