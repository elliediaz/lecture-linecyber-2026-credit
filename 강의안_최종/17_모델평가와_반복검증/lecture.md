# 16차시: 모델 평가와 반복 검증

## 학습 목표

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | 교차검증의 원리를 이해함 |
| 2 | 과대적합/과소적합을 진단함 |
| 3 | 다양한 평가 지표를 활용함 |

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 교차검증 원리 | K-Fold, cross_val_score |
| 2 | 적합도 진단 | 과대적합, 과소적합, 학습 곡선 |
| 3 | 평가 지표 심화 | 정밀도, 재현율, F1, 혼동 행렬 |

---

## Part 1: 교차검증의 원리

### 1.1 기존 평가 방식의 문제점

단순 학습/테스트 분할의 구조:

```
       전체 데이터
           |
    +------+------+
    |             |
 학습 80%     테스트 20%
```

**문제점**
- 데이터 분할 방식에 따라 결과가 달라짐
- 테스트 세트가 우연히 쉬운 데이터일 수 있음
- 신뢰도가 낮음

---

### 1.2 교차검증 (Cross-Validation) 개념

데이터를 여러 번 나누어 평가하고 평균을 내는 방법임.

**핵심 아이디어**
- 모든 데이터가 한 번씩 테스트에 사용됨
- 여러 번 평가하여 평균과 표준편차를 확인함
- 운에 덜 의존하는 신뢰할 수 있는 평가임

---

### 1.3 K-Fold 교차검증

가장 많이 사용하는 방법임.

```
K = 5 예시:

Fold 1: [테스트] [학습] [학습] [학습] [학습]
Fold 2: [학습] [테스트] [학습] [학습] [학습]
Fold 3: [학습] [학습] [테스트] [학습] [학습]
Fold 4: [학습] [학습] [학습] [테스트] [학습]
Fold 5: [학습] [학습] [학습] [학습] [테스트]

최종 점수 = (점수1 + 점수2 + ... + 점수5) / 5
```

**K-Fold 장점**
| 장점 | 설명 |
|------|------|
| 전체 활용 | 모든 데이터가 한 번씩 검증에 사용됨 |
| 평균 성능 | K번 평가해서 평균 성능을 확인함 |
| 안정성 파악 | 표준편차로 안정성을 파악함 |
| 효율성 | 데이터를 효율적으로 사용함 |

**일반적인 K 값 선택**
| K 값 | 용도 |
|------|------|
| K = 5 | 가장 보편적임 |
| K = 10 | 더 정확하지만 느림 |
| K = 3 | 데이터가 적을 때 사용함 |

---

### 1.4 cross_val_score 사용법

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# 5-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5)

print(f"각 Fold 점수: {scores}")
print(f"평균: {scores.mean():.3f}")
print(f"표준편차: {scores.std():.3f}")
```

**결과 해석**
```
각 Fold 점수: [0.85, 0.87, 0.84, 0.86, 0.85]
평균: 0.854
표준편차: 0.011
```
- 평균 85.4%: 일반적인 성능 수준임
- 표준편차 1.1%: 안정적임 (2~3% 이하면 안정적)
- 표준편차가 크면(5% 이상) 모델이나 데이터 확인이 필요함

---

### 1.5 실습: 단일 분할의 불안정성

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# 데이터 로딩
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("[여러 번 train_test_split 결과]")
scores_single = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores_single.append(score)
    print(f"  시도 {i+1:2d}: {score:.3f}")

print(f"\n점수 범위: {min(scores_single):.3f} ~ {max(scores_single):.3f}")
print(f"표준편차: {np.std(scores_single):.3f}")
print("-> 같은 모델인데 점수가 불안정함!")
```

**결과 해설**
- 동일한 모델임에도 불구하고 분할 방식에 따라 점수가 달라짐
- 단일 분할만으로는 신뢰할 수 있는 평가가 어려움

---

### 1.6 실습: K-Fold 교차검증

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# cross_val_score 사용
model = RandomForestClassifier(n_estimators=50, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)

print("[5-Fold 교차검증 결과]")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")

print(f"\n평균: {cv_scores.mean():.3f}")
print(f"표준편차: {cv_scores.std():.3f}")
print(f"결과: {cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})")
```

---

### 1.7 실습: 수동 K-Fold 구현

```python
from sklearn.model_selection import KFold

print("[수동 K-Fold 구현]")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

manual_scores = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    manual_scores.append(score)
    print(f"  Fold {fold}: 학습 {len(train_idx)}개, 테스트 {len(test_idx)}개 -> {score:.3f}")

print(f"\n평균: {np.mean(manual_scores):.3f} (+/-{np.std(manual_scores):.3f})")
```

---

### 1.8 실습: 여러 모델 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

models = {
    'DecisionTree': DecisionTreeClassifier(max_depth=10),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print(f"{'모델':<20} {'평균':>8} {'표준편차':>8}")
print("-" * 40)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<20} {scores.mean():>8.3f} {scores.std():>8.3f}")
```

**결과 해설**
- 교차검증을 통해 여러 모델을 공정하게 비교할 수 있음
- 평균 점수와 표준편차를 함께 고려하여 모델을 선택함

---

## Part 2: 과대적합/과소적합 진단

### 2.1 과대적합 vs 과소적합 개념

```
        정확도
          ^
        1.0|     ----------- 학습
           |           \
           |            \
           |             - 테스트
        0.0+-----------------> 모델 복잡도
              과소적합   최적   과대적합
```

---

### 2.2 과소적합 (Underfitting)

모델이 너무 단순한 상태임.

| 항목 | 설명 |
|------|------|
| 증상 | 학습 정확도 낮음, 테스트 정확도도 낮음, 둘 다 비슷하게 낮음 |
| 원인 | 모델이 너무 단순함, 특성이 부족함, 학습이 부족함 |
| 해결 | 모델 복잡도 높이기, 특성 추가, 하이퍼파라미터 조정 |

---

### 2.3 과대적합 (Overfitting)

모델이 너무 복잡한 상태임.

| 항목 | 설명 |
|------|------|
| 증상 | 학습 정확도 높음(거의 100%), 테스트 정확도 낮음, 둘 사이 큰 차이 |
| 원인 | 모델이 너무 복잡함, 데이터 부족함, 노이즈까지 학습함 |
| 해결 | 모델 단순화, 데이터 추가, 정규화 적용 |

---

### 2.4 실습: 적합도 진단

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[의사결정나무 - max_depth별 성능]")
print(f"{'max_depth':>10} {'학습 점수':>12} {'테스트 점수':>12} {'갭':>8} {'상태':>10}")
print("-" * 56)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    depth_str = str(depth) if depth else "None"

    if gap > 0.15:
        status = "과대적합"
    elif train_score < 0.7:
        status = "과소적합"
    else:
        status = "적절"

    print(f"{depth_str:>10} {train_score:>12.3f} {test_score:>12.3f} {gap:>8.3f} {status:>10}")
```

**결과 해설**
- max_depth가 낮으면 과소적합 발생 가능함
- max_depth가 없으면(None) 과대적합 위험이 있음
- 학습/테스트 점수 차이(갭)가 0.15 이상이면 과대적합을 의심함

---

### 2.5 학습 곡선 (Learning Curve)

데이터 양에 따른 성능 변화를 확인함.

```
      정확도
        ^
        |  ----------- 학습
        |     /
        |   /
        | /----------- 테스트
        +-----------------> 데이터 양
```

**해석 방법**
- 수렴하면 좋은 모델임
- 학습만 높으면 과대적합임
- 둘 다 낮으면 과소적합임

---

### 2.6 실습: 학습 곡선 시각화

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

model = RandomForestClassifier(n_estimators=50, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    random_state=42
)

print("[학습 곡선 데이터]")
print(f"{'학습 데이터':>12} {'학습 점수':>12} {'검증 점수':>12} {'갭':>8}")
print("-" * 48)

for size, train_mean, val_mean in zip(
    train_sizes,
    train_scores.mean(axis=1),
    val_scores.mean(axis=1)
):
    gap = train_mean - val_mean
    print(f"{size:>12} {train_mean:>12.3f} {val_mean:>12.3f} {gap:>8.3f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='학습', color='blue')
plt.fill_between(train_sizes,
                  train_scores.mean(axis=1) - train_scores.std(axis=1),
                  train_scores.mean(axis=1) + train_scores.std(axis=1),
                  alpha=0.2, color='blue')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='검증', color='orange')
plt.fill_between(train_sizes,
                  val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1),
                  alpha=0.2, color='orange')
plt.xlabel('학습 데이터 수')
plt.ylabel('점수')
plt.title('학습 곡선')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 2.7 검증 곡선 (Validation Curve)

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

param_range = [1, 2, 3, 5, 7, 10, 15, 20]

train_scores_vc, val_scores_vc = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5
)

print("[max_depth별 검증 곡선]")
print(f"{'max_depth':>10} {'학습 점수':>12} {'검증 점수':>12}")
print("-" * 36)

for depth, train_mean, val_mean in zip(
    param_range,
    train_scores_vc.mean(axis=1),
    val_scores_vc.mean(axis=1)
):
    print(f"{depth:>10} {train_mean:>12.3f} {val_mean:>12.3f}")

# 최적 max_depth
best_idx = np.argmax(val_scores_vc.mean(axis=1))
best_depth = param_range[best_idx]
print(f"\n최적 max_depth: {best_depth} (검증 점수: {val_scores_vc.mean(axis=1)[best_idx]:.3f})")
```

---

## Part 3: 다양한 평가 지표

### 3.1 정확도의 한계

불균형 데이터에서 정확도만으로는 부족함.

**예시**: 불량률 1%인 경우
- 전부 "정상"으로 예측해도 정확도 99%임
- 실제로는 불량을 전혀 찾지 못함

---

### 3.2 혼동 행렬 (Confusion Matrix)

|  | 예측: 정상 | 예측: 불량 |
|--|-----------|-----------|
| **실제: 정상** | TN (참 음성) | FP (거짓 양성) |
| **실제: 불량** | FN (거짓 음성) | TP (참 양성) |

| 약어 | 의미 | 설명 |
|------|------|------|
| TN | True Negative | 정상을 정상으로 맞춤 |
| TP | True Positive | 불량을 불량으로 맞춤 |
| FP | False Positive | 정상을 불량으로 틀림 (오탐) |
| FN | False Negative | 불량을 정상으로 틀림 (누락) |

---

### 3.3 정밀도 (Precision)

"불량이라고 예측한 것 중 실제 불량 비율"

$$Precision = \frac{TP}{TP + FP}$$

**중요한 경우**: 오탐 비용이 클 때
- 예: 정상 제품을 버리면 손해

---

### 3.4 재현율 (Recall)

"실제 불량 중 불량으로 예측한 비율"

$$Recall = \frac{TP}{TP + FN}$$

**중요한 경우**: 누락 비용이 클 때
- 예: 불량 제품이 출하되면 큰 문제

---

### 3.5 F1 점수

정밀도와 재현율의 조화 평균임.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**특징**
- 정밀도와 재현율의 균형을 봄
- 둘 다 높아야 F1도 높음
- 불균형 데이터에서 유용함

---

### 3.6 실습: 분류 평가 지표

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 모델 학습 및 예측
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 혼동행렬
cm = confusion_matrix(y_test, y_pred)
print("[혼동행렬]")
print(cm)

# 평가 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n[평가 지표 (weighted average)]")
print(f"정확도 (Accuracy):  {accuracy:.3f}")
print(f"정밀도 (Precision): {precision:.3f}")
print(f"재현율 (Recall):    {recall:.3f}")
print(f"F1 Score:           {f1:.3f}")

# classification_report
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

**결과 예시**
```
              precision    recall  f1-score   support

     class_0       0.95      0.98      0.96        14
     class_1       0.94      0.94      0.94        14
     class_2       1.00      0.88      0.93         8

    accuracy                           0.94        36
   macro avg       0.96      0.93      0.94        36
weighted avg       0.95      0.94      0.94        36
```

---

### 3.7 회귀 평가 지표

| 지표 | 설명 | 특징 |
|------|------|------|
| MSE | 평균 제곱 오차 | 큰 오차 강조 |
| RMSE | MSE의 제곱근 | 원래 단위 |
| MAE | 평균 절대 오차 | 이상치에 강건 |
| R^2 | 결정계수 | 0~1 해석 쉬움 |

---

### 3.8 실습: 회귀 평가 지표

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# 회귀 데이터 로딩
housing = fetch_california_housing()
X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
y_reg = housing.target

# 샘플링
np.random.seed(42)
idx = np.random.choice(len(X_reg), 5000, replace=False)
X_reg = X_reg.iloc[idx]
y_reg = y_reg[idx]

# 분할
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 모델 학습
model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)

# 평가 지표 계산
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("[회귀 평가 지표]")
print(f"MSE:  {mse:.4f} (평균 제곱 오차)")
print(f"RMSE: {rmse:.4f} (MSE의 제곱근)")
print(f"MAE:  {mae:.4f} (평균 절대 오차)")
print(f"R^2:  {r2:.3f} (결정계수, 1에 가까울수록 좋음)")

print("\n[지표 해석]")
print(f"-> 평균적으로 ${rmse*100000:,.0f} 정도의 오차")
print(f"-> 모델이 데이터 변동의 {r2*100:.1f}%를 설명")
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 교차검증 | K번 평가 후 평균 (더 신뢰성 있음) |
| K-Fold | 데이터를 K개로 나눠 순환 평가 |
| 과소적합 | 모델 너무 단순 (둘 다 낮음) |
| 과대적합 | 모델 너무 복잡 (학습만 높음) |
| 정밀도 | 예측 양성 중 실제 양성 비율 |
| 재현율 | 실제 양성 중 예측 양성 비율 |
| F1 | 정밀도와 재현율의 조화 평균 |

---

## 실무 가이드

**권장 사항**
- 항상 교차검증 사용 (최소 5-Fold)
- 표준편차 확인 (안정성 체크)
- 불균형 데이터면 F1, 재현율 주목
- 학습/테스트 점수 차이 모니터링

**주의 사항**
- 단일 분할 결과만 보지 말 것
- 정확도만 보지 말 것 (특히 불균형 데이터)
- 테스트 세트로 파라미터 튜닝하지 말 것

---

## sklearn 주요 함수 요약

```python
from sklearn.model_selection import (
    cross_val_score,    # 교차검증
    learning_curve,     # 학습 곡선
    validation_curve    # 검증 곡선
)
from sklearn.metrics import (
    confusion_matrix,       # 혼동행렬
    classification_report,  # 분류 보고서
    mean_squared_error,     # MSE
    r2_score               # R^2
)
```
