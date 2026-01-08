# [12차시] 머신러닝 소개와 문제 유형

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **머신러닝의 개념**을 전통적 프로그래밍과 비교하여 설명함
2. **지도학습과 비지도학습**을 구분하고 각각의 활용 사례를 이해함
3. **분류와 회귀 문제**를 구분하고 sklearn 기본 패턴으로 구현함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Iris** | sklearn.datasets | 분류 실습 (붓꽃 품종 분류) |
| **California Housing** | sklearn.datasets | 회귀 실습 (주택 가격 예측) |

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 머신러닝의 개념 | 정의, 전통적 프로그래밍과 비교 |
| 2 | 지도학습과 비지도학습 | 학습 유형, 정답 유무 |
| 3 | 분류와 회귀 문제 구분 | 출력 형태, sklearn 기본 패턴 |

---

## 파트 1: 머신러닝의 개념

### 개념 설명

#### 머신러닝(Machine Learning)이란?

명시적으로 프로그래밍하지 않아도 **데이터로부터 스스로 패턴을 학습**하는 알고리즘임.

핵심 아이디어:
- 사람이 규칙을 정하는 것이 아님
- **데이터가 규칙을 알려줌**

#### 전통적 프로그래밍 vs 머신러닝

| 구분 | 전통적 프로그래밍 | 머신러닝 |
|------|------------------|----------|
| **규칙** | 사람이 직접 작성 | 데이터에서 학습 |
| **예시** | `if 온도 > 90: 불량` | 데이터가 알려줌: `온도 > 87.3` |
| **유연성** | 규칙 변경 시 코드 수정 | 새 데이터로 재학습 |
| **복잡성** | 규칙이 많아지면 한계 | 복잡한 패턴도 학습 |

```
전통적 프로그래밍:
[데이터] + [규칙] --> [프로그램] --> [결과]

머신러닝:
[데이터] + [결과] --> [학습] --> [규칙(모델)]
```

#### 핵심 용어 정리

| 용어 | 영어 | 설명 |
|------|------|------|
| **특성** | Feature | 입력 데이터, 예측에 사용하는 정보 |
| **타겟** | Target | 예측하려는 값, 정답 |
| **모델** | Model | 학습된 패턴, 예측 기계 |
| **학습** | Training | 데이터에서 패턴을 찾는 과정 |
| **예측** | Prediction | 새 데이터에 패턴 적용 |

#### 머신러닝이 적합한 경우

| 적합한 상황 | 부적합한 상황 |
|------------|--------------|
| 규칙이 복잡하거나 명확하지 않을 때 | 명확한 규칙이 있을 때 (if-else로 충분) |
| 데이터가 충분할 때 (수백~수천 건 이상) | 데이터가 매우 적을 때 |
| 패턴이 변화할 때 (재학습으로 적응) | 100% 정확도가 필요할 때 |
| 미묘한 판단이 필요할 때 | |

### 실습 코드

#### 전통적 프로그래밍 방식

```python
# 전통적 방식: 사람이 직접 규칙 작성
def traditional_predict(sepal_length, sepal_width, petal_length, petal_width):
    """전통적 방식: 사람이 직접 규칙 작성"""
    if petal_length < 2.5:
        return "setosa"
    elif petal_width < 1.8:
        return "versicolor"
    else:
        return "virginica"

# 테스트
test_cases = [
    (5.0, 3.5, 1.5, 0.2),  # 예상: setosa
    (6.0, 2.8, 4.5, 1.5),  # 예상: versicolor
    (7.0, 3.0, 6.0, 2.5),  # 예상: virginica
]

print("[전통적 방식 - 규칙 기반]")
for sl, sw, pl, pw in test_cases:
    result = traditional_predict(sl, sw, pl, pw)
    print(f"  꽃받침={sl}, 꽃잎={pl} --> {result}")

print("\n문제점:")
print("  - 규칙이 많아지면 관리 어려움")
print("  - 새로운 패턴 발견 어려움")
print("  - 미묘한 경계값 설정 어려움")
```

#### 데이터셋 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing

# Iris 데이터셋 (분류용)
iris = load_iris()
df_clf = pd.DataFrame(iris.data, columns=iris.feature_names)
df_clf['target'] = iris.target
df_clf['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("Iris 데이터셋 로딩 완료")

print(f"\n[Iris 데이터셋 정보]")
print(f"데이터 크기: {df_clf.shape}")
print(f"특성 이름: {iris.feature_names}")
print(f"클래스: {list(iris.target_names)}")
print(f"\n처음 5행:")
print(df_clf.head())
```

```python
# California Housing 데이터셋 (회귀용)
housing = fetch_california_housing()
df_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
df_reg['MedHouseVal'] = housing.target
print("California Housing 데이터셋 로딩 완료")

print(f"\n[California Housing 데이터셋 정보]")
print(f"데이터 크기: {df_reg.shape}")
print(f"특성 이름: {list(housing.feature_names)}")
print(f"\n처음 5행:")
print(df_reg.head())
```

### 결과 해설

- Iris 데이터셋: 150개 샘플, 4개 특성, 3개 클래스(setosa, versicolor, virginica)로 구성됨
- California Housing 데이터셋: 20,640개 샘플, 8개 특성, 연속형 타겟(주택 가격)으로 구성됨
- 두 데이터셋은 각각 분류와 회귀 문제의 대표적인 실습 데이터임

---

## 파트 2: 지도학습과 비지도학습

### 개념 설명

#### 머신러닝의 3가지 유형

```
                머신러닝
                   |
      +------------+------------+
      |            |            |
   지도학습     비지도학습    강화학습
 (Supervised)  (Unsupervised) (Reinforcement)
      |            |            |
   정답 있음    정답 없음     보상 기반
```

#### 지도학습 (Supervised Learning)

정답이 있는 데이터로 학습하는 방식임. "선생님이 정답을 알려주며 가르치는 것"과 유사함.

특징:
- **레이블(정답)**이 있는 데이터 필요
- 입력(X)과 출력(y)의 관계 학습
- 새 입력에 대해 출력 예측

| 문제 | 입력 (특성) | 출력 (타겟) |
|------|------------|------------|
| 불량 분류 | 온도, 습도, 속도 | 불량/정상 |
| 생산량 예측 | 설비, 원료, 인력 | 생산량(숫자) |
| 품질 등급 | 성분, 외관, 강도 | A/B/C 등급 |
| 고장 예측 | 진동, 온도, 가동시간 | 고장 여부 |

#### 비지도학습 (Unsupervised Learning)

정답 없이 패턴을 발견하는 방식임. "정답 없이 스스로 패턴을 찾는 것"임.

특징:
- 레이블(정답) **없음**
- 데이터의 **구조/패턴** 발견
- 군집화, 차원 축소, 이상 탐지

| 문제 | 방법 | 설명 |
|------|------|------|
| **제품 군집화** | 클러스터링 | 유사 제품 그룹 발견 |
| **이상 탐지** | 이상탐지 알고리즘 | 정상 패턴 벗어난 데이터 |
| **차원 축소** | PCA | 변수 개수 줄이기 |

#### 지도학습 vs 비지도학습 비교

| 구분 | 지도학습 | 비지도학습 |
|------|----------|-----------|
| **정답** | 있음 (레이블) | 없음 |
| **목표** | 예측 | 구조 발견 |
| **대표 문제** | 분류, 회귀 | 군집화, 차원축소 |
| **제조 예시** | 불량 예측 | 이상 탐지 |

### 실습 코드

#### 특성과 타겟 분리

```python
from sklearn.model_selection import train_test_split

# 분류용 특성과 타겟
feature_columns_clf = ['sepal length (cm)', 'sepal width (cm)',
                       'petal length (cm)', 'petal width (cm)']
X_clf = df_clf[feature_columns_clf]
y_clf = df_clf['target']

# 회귀용 특성과 타겟 (주요 특성만 선택)
feature_columns_reg = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
X_reg = df_reg[feature_columns_reg]
y_reg = df_reg['MedHouseVal']

print("[분류 데이터 - Iris]")
print(f"  특성 열: {list(X_clf.columns)}")
print(f"  크기: {X_clf.shape}")
print(f"  타겟 클래스 분포:")
for i, name in enumerate(iris.target_names):
    count = (y_clf == i).sum()
    print(f"    - {name}: {count}개 ({count/len(y_clf):.1%})")

print("\n[회귀 데이터 - California Housing]")
print(f"  특성 열: {list(X_reg.columns)}")
print(f"  크기: {X_reg.shape}")
print(f"  타겟 (주택 가격) 통계:")
print(f"    - 평균: ${y_reg.mean()*100000:,.0f}")
print(f"    - 범위: ${y_reg.min()*100000:,.0f} ~ ${y_reg.max()*100000:,.0f}")
```

#### 학습/테스트 데이터 분리

```python
# 분류용 데이터 분리
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf,
    test_size=0.2,      # 20%는 테스트용
    random_state=42,    # 재현성
    stratify=y_clf      # 클래스 비율 유지
)

# 회귀용 데이터 분리
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

print(f"[분류 데이터 분리 결과 (Iris)]")
print(f"  전체 데이터: {len(X_clf)}개")
print(f"  학습 데이터: {len(X_train_clf)}개 ({len(X_train_clf)/len(X_clf):.0%})")
print(f"  테스트 데이터: {len(X_test_clf)}개 ({len(X_test_clf)/len(X_clf):.0%})")

print(f"\n[회귀 데이터 분리 결과 (California Housing)]")
print(f"  전체 데이터: {len(X_reg)}개")
print(f"  학습 데이터: {len(X_train_reg)}개 ({len(X_train_reg)/len(X_reg):.0%})")
print(f"  테스트 데이터: {len(X_test_reg)}개 ({len(X_test_reg)/len(X_reg):.0%})")
```

### 결과 해설

- 학습 데이터(80%)로 모델이 패턴을 학습하고, 테스트 데이터(20%)로 성능을 평가함
- `stratify=y_clf` 옵션은 클래스 비율을 유지하며 분할하여 불균형 문제를 방지함
- 학습에 사용하지 않은 테스트 데이터로 평가해야 실제 성능을 정확히 측정할 수 있음

---

## 파트 3: 분류와 회귀 문제 구분

### 개념 설명

#### 지도학습의 두 가지 문제

```
             지도학습
                |
        +-------+-------+
        |               |
      분류            회귀
 (Classification)  (Regression)
        |               |
    범주 예측        숫자 예측
```

#### 분류 (Classification)

범주(카테고리)를 예측하는 문제임.

```
입력: 온도 85도, 습도 50%, 속도 100
출력: "정상" 또는 "불량"  <-- 범주!
```

특징:
- 출력이 **정해진 범주** 중 하나
- 이산적(discrete) 값
- 확률로 해석 가능

| 문제 | 입력 | 출력 (범주) |
|------|------|-----------|
| 불량 분류 | 센서 데이터 | 불량 / 정상 |
| 품질 등급 | 측정값 | A / B / C |
| 고장 유형 | 진동 패턴 | 모터 / 베어링 / 전기 |

#### 회귀 (Regression)

연속적인 숫자를 예측하는 문제임.

```
입력: 온도 85도, 습도 50%, 속도 100
출력: 1,247개  <-- 숫자!
```

특징:
- 출력이 **연속적인 숫자**
- 어떤 값이든 나올 수 있음
- 오차를 최소화

| 문제 | 입력 | 출력 (숫자) |
|------|------|-----------|
| 생산량 예측 | 설비, 인력 | 1,247개 |
| 불량률 예측 | 공정 조건 | 3.5% |
| 설비 수명 | 사용 이력 | 87일 |

#### 분류 vs 회귀 구분법

| 질문 | 분류 | 회귀 |
|------|:----:|:----:|
| 이 제품 불량인가요? | O | |
| 불량률이 몇 %? | | O |
| 어떤 등급인가요? | O | |
| 생산량이 얼마? | | O |

간단한 구분 팁:
- **분류**: "~인가요?" (예/아니오), "어떤 종류인가요?" (범주 선택)
- **회귀**: "얼마나?", "몇 개?", "몇 %?"

#### sklearn 기본 패턴

모든 sklearn 모델이 동일한 패턴을 따름:

```python
# 1. 모델 생성
model = ModelClass(parameters)

# 2. 학습 (fit)
model.fit(X_train, y_train)

# 3. 예측 (predict)
y_pred = model.predict(X_test)

# 4. 평가 (score)
score = model.score(X_test, y_test)
```

### 실습 코드

#### 분류 모델 (DecisionTreeClassifier)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 모델 생성 및 학습
clf_model = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_model.fit(X_train_clf, y_train_clf)
print("학습 완료")

# 예측
y_pred_clf = clf_model.predict(X_test_clf)
print(f"\n[예측 결과 샘플]")
print(f"  실제: {list(y_test_clf[:10].values)}")
print(f"  예측: {list(y_pred_clf[:10])}")

# 예측을 꽃 이름으로 변환
print(f"\n[예측 결과 (꽃 이름)]")
for i in range(min(5, len(y_pred_clf))):
    actual = iris.target_names[y_test_clf.iloc[i]]
    predicted = iris.target_names[y_pred_clf[i]]
    match = "O" if actual == predicted else "X"
    print(f"  샘플 {i+1}: 실제={actual}, 예측={predicted} [{match}]")

# 평가
accuracy = clf_model.score(X_test_clf, y_test_clf)
print(f"\n[분류 성능]")
print(f"  정확도(Accuracy): {accuracy:.1%}")

# 혼동 행렬
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\n[혼동 행렬]")
print(f"              setosa  versicolor  virginica")
for i, name in enumerate(iris.target_names):
    print(f"  {name:>10}:  {cm[i,0]:5}      {cm[i,1]:5}      {cm[i,2]:5}")
```

#### 회귀 모델 (LinearRegression)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 모델 생성 및 학습
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
print("학습 완료")

# 예측
y_pred_reg = reg_model.predict(X_test_reg)
print(f"\n[예측 결과 샘플 (주택 가격, 단위: $100,000)]")
print(f"  실제: {list(y_test_reg[:5].round(2))}")
print(f"  예측: {list(y_pred_reg[:5].round(2))}")

# 평가
r2 = reg_model.score(X_test_reg, y_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)

print(f"\n[회귀 성능]")
print(f"  R2 점수: {r2:.3f}")
print(f"  MSE: {mse:.3f}")
print(f"  RMSE: {rmse:.3f} (약 ${rmse*100000:,.0f})")

# 회귀 계수
print(f"\n[회귀 계수]")
print(f"  절편: {reg_model.intercept_:.4f}")
for col, coef in zip(feature_columns_reg, reg_model.coef_):
    print(f"  {col}: {coef:.4f}")
```

#### 새 데이터 예측

```python
# 새로운 꽃 데이터 (분류)
new_flowers = pd.DataFrame({
    'sepal length (cm)': [5.0, 6.5, 7.2],
    'sepal width (cm)': [3.5, 2.8, 3.0],
    'petal length (cm)': [1.5, 4.5, 6.0],
    'petal width (cm)': [0.2, 1.5, 2.2]
})

print("[새 꽃 데이터]")
print(new_flowers)

# 분류 예측
flower_predictions = clf_model.predict(new_flowers)
flower_proba = clf_model.predict_proba(new_flowers)

print("\n[꽃 종류 예측 (분류)]")
for i, (pred, proba) in enumerate(zip(flower_predictions, flower_proba)):
    species = iris.target_names[pred]
    print(f"  꽃 {i+1}: {species}")
    print(f"         (확률: setosa {proba[0]:.1%}, versicolor {proba[1]:.1%}, virginica {proba[2]:.1%})")
```

```python
# 새로운 주택 데이터 (회귀)
new_houses = pd.DataFrame({
    'MedInc': [3.0, 5.0, 8.0],      # 중위 소득
    'HouseAge': [20, 15, 5],         # 집 연식
    'AveRooms': [5.0, 6.0, 7.0],     # 평균 방 수
    'AveOccup': [3.0, 2.5, 2.0]      # 평균 거주자 수
})

print("[새 주택 데이터]")
print(new_houses)

# 회귀 예측
price_predictions = reg_model.predict(new_houses)

print("\n[주택 가격 예측 (회귀)]")
for i, pred in enumerate(price_predictions):
    print(f"  주택 {i+1}: ${pred*100000:,.0f}")
```

### 결과 해설

- 분류 모델(DecisionTreeClassifier)은 Iris 데이터에서 높은 정확도(약 96%)를 달성함
- 회귀 모델(LinearRegression)은 California Housing 데이터에서 R2 점수 약 0.5를 기록함
- sklearn의 모든 모델은 fit-predict-score 패턴을 공통으로 사용하여 학습이 용이함

---

## 다양한 모델 비교

### 분류 모델 비교

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

print("[분류 모델 비교 - Iris 데이터셋]")
print(f"{'모델':<20} {'정확도':>10}")
print("-" * 32)

for name, model in classifiers.items():
    model.fit(X_train_clf, y_train_clf)
    accuracy = model.score(X_test_clf, y_test_clf)
    print(f"{name:<20} {accuracy:>10.1%}")
```

### 회귀 모델 비교

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

regressors = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50),
}

print("[회귀 모델 비교 - California Housing 데이터셋]")
print(f"{'모델':<20} {'R2 점수':>10} {'RMSE':>10}")
print("-" * 42)

for name, model in regressors.items():
    model.fit(X_train_reg, y_train_reg)
    r2 = model.score(X_test_reg, y_test_reg)
    y_pred = model.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    print(f"{name:<20} {r2:>10.3f} {rmse:>10.3f}")
```

---

## 핵심 정리

### 머신러닝 핵심 개념

| 개념 | 설명 |
|------|------|
| **머신러닝** | 데이터에서 패턴을 자동 학습 |
| **지도학습** | 정답(레이블) 있는 데이터로 학습 |
| **비지도학습** | 정답 없이 구조/패턴 발견 |
| **분류** | 범주 예측 ("~인가요?") |
| **회귀** | 숫자 예측 ("얼마나?") |

### sklearn 기본 패턴

```python
# 모든 모델 공통 패턴
model = ModelClass(parameters)   # 1. 모델 생성
model.fit(X_train, y_train)      # 2. 학습
y_pred = model.predict(X_test)   # 3. 예측
score = model.score(X_test, y_test)  # 4. 평가
```

### 실제 공개 데이터셋 활용

| 데이터셋 | 문제 유형 | 샘플 수 | 특성 수 |
|----------|----------|--------|--------|
| Iris | 분류 (3종) | 150 | 4 |
| California Housing | 회귀 | 20,640 | 8 |

### 학습/테스트 분리

- 일반적으로 80% 학습, 20% 테스트
- 처음 보는 데이터로 평가해야 실제 성능 확인
- `random_state` 설정으로 재현성 확보
