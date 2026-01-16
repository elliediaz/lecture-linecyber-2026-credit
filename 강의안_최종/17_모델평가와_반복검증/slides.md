---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 16차시'
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

# 모델 평가와 반복 검증

## 16차시 | Part III. 문제 중심 모델링 실습

**더 정확하게 모델을 평가합니다!**

---

# 지난 시간 복습

## 14차시에서 배운 것

- **회귀**: 연속적인 숫자 예측
- **선형회귀**: y = wx + b
- **다항회귀**: 곡선 관계 학습
- **R² 점수**: 모델 설명력 평가

<div class="tip">

오늘은 **교차검증**으로 더 신뢰할 수 있는 평가를 합니다!

</div>

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **교차검증의 원리**를 이해한다 |
| 2 | **과대적합/과소적합을 진단**한다 |
| 3 | **다양한 평가 지표**를 활용한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  교차검증   │ →  │  적합도     │ → │  평가지표   │
│  원리       │    │  진단       │    │    심화     │
└─────────────┘    └─────────────┘    └─────────────┘
   K-Fold         과대/과소적합      정밀도, 재현율
   cross_val_score  학습 곡선        혼동 행렬
```

---

<!-- _class: lead -->

# Part 1

## 교차검증의 원리

---

# 기존 평가 방식의 문제

## 단순 학습/테스트 분할

```
       전체 데이터
           │
    ┌──────┴──────┐
    │             │
 학습 80%     테스트 20%
```

### 문제점
- 어떻게 나누느냐에 따라 **결과가 달라짐**
- 테스트 세트가 **운 좋게 쉬운** 데이터일 수 있음
- **신뢰도 낮음**

---

# 교차검증 (Cross-Validation)

## 더 안정적인 평가

> 데이터를 여러 번 나누어 평가하고 **평균**내는 방법

### 핵심 아이디어
- 모든 데이터가 **한 번씩** 테스트에 사용됨
- 여러 번 평가해서 **평균**과 **표준편차** 확인
- 운에 덜 의존하는 **신뢰할 수 있는** 평가

---

# K-Fold 교차검증

## 가장 많이 사용하는 방법

```
K = 5 예시:

Fold 1: [테스트] [학습] [학습] [학습] [학습]
Fold 2: [학습] [테스트] [학습] [학습] [학습]
Fold 3: [학습] [학습] [테스트] [학습] [학습]
Fold 4: [학습] [학습] [학습] [테스트] [학습]
Fold 5: [학습] [학습] [학습] [학습] [테스트]

최종 점수 = (점수1 + 점수2 + ... + 점수5) / 5
```

---

# K-Fold 장점

## 신뢰할 수 있는 평가

<div class="highlight">

### 장점
1. 모든 데이터가 **한 번씩** 검증에 사용됨
2. K번 평가해서 **평균 성능** 확인
3. **표준편차**로 안정성 파악
4. 데이터 효율적 사용

### 일반적인 K 값
- **K = 5**: 가장 보편적
- **K = 10**: 더 정확하지만 느림
- **K = 3**: 데이터 적을 때

</div>

---

# cross_val_score

## sklearn 함수

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

---

# 결과 해석

## 평균과 표준편차

```
각 Fold 점수: [0.85, 0.87, 0.84, 0.86, 0.85]
평균: 0.854
표준편차: 0.011
```

### 해석
- **평균 85.4%**: 일반적인 성능 수준
- **표준편차 1.1%**: 안정적 (2~3% 이하면 안정적)

### 주의
- 표준편차가 **크면** (5% 이상) 불안정
- 모델이나 데이터 확인 필요

---

# scoring 파라미터

## 다양한 평가 지표

```python
# 분류: 정확도 (기본)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# 분류: F1 점수
scores = cross_val_score(model, X, y, cv=5, scoring='f1')

# 회귀: R² 점수 (기본)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# 회귀: MSE (음수로 반환)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

---

# 여러 모델 비교

## 교차검증으로 공정한 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LogisticRegression': LogisticRegression()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

<!-- _class: lead -->

# Part 2

## 과대적합/과소적합 진단

---

# 과대적합 vs 과소적합

## 적합도 문제

```
        정확도
          ↑
        1.0│     ─────────── 학습
           │           \
           │            \
           │             ─ 테스트
        0.0└────────────────→ 모델 복잡도
              과소적합   최적   과대적합
```

---

# 과소적합 (Underfitting)

## 모델이 너무 단순

<div class="important">

### 증상
- 학습 정확도 **낮음**
- 테스트 정확도도 **낮음**
- 둘 다 비슷하게 낮음

### 원인
- 모델이 너무 단순
- 특성이 부족
- 학습 부족

### 해결
- 모델 복잡도 높이기
- 특성 추가
- 하이퍼파라미터 조정

</div>

---

# 과대적합 (Overfitting)

## 모델이 너무 복잡

<div class="important">

### 증상
- 학습 정확도 **높음** (거의 100%)
- 테스트 정확도 **낮음**
- 둘 사이 **큰 차이**

### 원인
- 모델이 너무 복잡
- 데이터 부족
- 노이즈까지 학습

### 해결
- 모델 단순화
- 데이터 추가
- 정규화 적용

</div>

---

# 적합도 진단 코드

## 학습/테스트 점수 비교

```python
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"학습 정확도: {train_score:.1%}")
print(f"테스트 정확도: {test_score:.1%}")
print(f"차이: {train_score - test_score:.1%}")

if train_score - test_score > 0.1:
    print("→ 과대적합 의심!")
elif train_score < 0.7:
    print("→ 과소적합 의심!")
else:
    print("→ 적절한 적합도")
```

---

# 학습 곡선 (Learning Curve)

## 데이터 양에 따른 성능 변화

```
      정확도
        ↑
        │  ───────── 학습
        │     /
        │   /
        │ /──────── 테스트
        └──────────────→ 데이터 양
```

### 해석
- 수렴하면 **좋은 모델**
- 학습만 높으면 **과대적합**
- 둘 다 낮으면 **과소적합**

---

# 학습 곡선 코드

## learning_curve 함수

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=5
)

# 평균 계산
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='학습')
plt.plot(train_sizes, test_mean, 'o-', label='검증')
plt.xlabel('학습 데이터 크기')
plt.ylabel('점수')
plt.legend()
plt.show()
```

---

# 검증 곡선 (Validation Curve)

## 하이퍼파라미터에 따른 성능

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 10, 15]

train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5
)

# 최적 파라미터 찾기
test_mean = test_scores.mean(axis=1)
best_depth = param_range[np.argmax(test_mean)]
print(f"최적 max_depth: {best_depth}")
```

---

<!-- _class: lead -->

# Part 3

## 다양한 평가 지표

---

# 분류 평가 지표

## 정확도만으로는 부족

### 불균형 데이터 문제
- 불량률 1% → 전부 "정상" 예측해도 정확도 99%!

### 추가 지표 필요
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1 점수**

---

# 혼동 행렬 (Confusion Matrix)

## 예측 결과 분류

|  | 예측: 정상 | 예측: 불량 |
|--|-----------|-----------|
| **실제: 정상** | TN (참 음성) | FP (거짓 양성) |
| **실제: 불량** | FN (거짓 음성) | TP (참 양성) |

- **TN**: 정상을 정상으로 맞춤 ✓
- **TP**: 불량을 불량으로 맞춤 ✓
- **FP**: 정상을 불량으로 틀림 (오탐)
- **FN**: 불량을 정상으로 틀림 (누락)

---

# 정밀도 (Precision)

## "불량이라고 예측한 것 중 실제 불량 비율"

$$Precision = \frac{TP}{TP + FP}$$

### 중요한 경우
- **오탐 비용이 클 때**
- 예: 정상 제품을 버리면 손해

```
불량 예측 100개 중
실제 불량 80개 → 정밀도 80%
```

---

# 재현율 (Recall)

## "실제 불량 중 불량으로 예측한 비율"

$$Recall = \frac{TP}{TP + FN}$$

### 중요한 경우
- **누락 비용이 클 때**
- 예: 불량 제품이 출하되면 큰 문제

```
실제 불량 100개 중
불량으로 예측 70개 → 재현율 70%
```

---

# F1 점수

## 정밀도와 재현율의 조화 평균

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 특징
- 정밀도와 재현율의 **균형**
- 둘 다 높아야 F1도 높음
- **불균형 데이터**에서 유용

---

# classification_report

## 모든 지표 한번에

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
```

출력:
```
              precision    recall  f1-score   support

       정상       0.95      0.98      0.96       180
       불량       0.80      0.65      0.72        20

    accuracy                           0.94       200
   macro avg       0.87      0.82      0.84       200
weighted avg       0.94      0.94      0.94       200
```

---

# 회귀 평가 지표

## MSE, RMSE, MAE, R²

| 지표 | 설명 | 특징 |
|------|------|------|
| **MSE** | 평균 제곱 오차 | 큰 오차 강조 |
| **RMSE** | MSE의 제곱근 | 원래 단위 |
| **MAE** | 평균 절대 오차 | 이상치에 강건 |
| **R²** | 결정계수 | 0~1 해석 쉬움 |

---

# 회귀 지표 계산

## sklearn.metrics

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
```

---

<!-- _class: lead -->

# 실습편

## 교차검증과 평가 지표 실습

---

# 실습 목표

## 모델 평가 완전 정복

1. K-Fold 교차검증 적용
2. 여러 모델 비교
3. 학습 곡선 그리기
4. 분류 평가 지표 분석

---

# 실습 1: 교차검증 기본

## cross_val_score 사용

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5)

print(f"각 Fold 점수: {scores}")
print(f"평균: {scores.mean():.3f}")
print(f"표준편차: {scores.std():.3f}")
print(f"95% 신뢰구간: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

---

# 실습 2: 여러 모델 비교

## 공정한 비교

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

---

# 실습 3: 학습 곡선

## 과대적합 진단

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100),
    X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='학습')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='검증')
plt.xlabel('학습 데이터 크기')
plt.ylabel('정확도')
plt.title('학습 곡선')
plt.legend()
plt.grid(True)
plt.show()
```

---

# 실습 4: 분류 평가 지표

## 상세 분석

```python
from sklearn.metrics import confusion_matrix, classification_report

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬:")
print(cm)

# 분류 보고서
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
```

---

# 핵심 정리

## 16차시 요약

| 개념 | 설명 |
|------|------|
| **교차검증** | K번 평가 후 평균 (더 신뢰성 있음) |
| **K-Fold** | 데이터를 K개로 나눠 순환 평가 |
| **과소적합** | 모델 너무 단순 (둘 다 낮음) |
| **과대적합** | 모델 너무 복잡 (학습만 높음) |
| **정밀도** | 예측 양성 중 실제 양성 비율 |
| **재현율** | 실제 양성 중 예측 양성 비율 |
| **F1** | 정밀도와 재현율의 조화 평균 |

---

# 실무 가이드

## 모델 평가 베스트 프랙티스

<div class="highlight">

### 권장 사항
- 항상 **교차검증** 사용 (최소 5-Fold)
- **표준편차** 확인 (안정성 체크)
- 불균형 데이터면 **F1, 재현율** 주목
- 학습/테스트 **점수 차이** 모니터링

### 주의 사항
- 단일 분할 결과만 보지 말 것
- 정확도만 보지 말 것 (특히 불균형 데이터)
- 테스트 세트로 파라미터 튜닝하지 말 것

</div>

---

# 다음 차시 예고

## 16차시: 모델 설정값 최적화

### 학습 내용
- GridSearchCV
- RandomizedSearchCV
- 하이퍼파라미터 튜닝

<div class="tip">

**최적의 파라미터**를 자동으로 찾습니다!

</div>

---

# 감사합니다

## 16차시: 모델 평가와 반복 검증

**신뢰할 수 있는 평가 방법을 배웠습니다!**

Q&A
