---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 15차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 모델 평가와 교차검증

## 15차시 | AI 기초체력훈련 (Pre AI-Campus)

**모델을 제대로 평가하는 방법**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **교차검증**의 개념과 필요성을 이해한다
2. **과대적합/과소적합**을 진단한다
3. **혼동행렬, 정밀도, 재현율**을 해석한다

---

# 지금까지의 평가 방법

## train_test_split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # 한 번만 평가
```

### 문제점
- 운 좋게 쉬운 테스트 데이터가 뽑힐 수 있음
- 운 나쁘게 어려운 데이터가 뽑힐 수 있음
- **한 번의 분할로는 신뢰하기 어려움**

---

# 교차검증이란?

## Cross Validation

> 데이터를 **여러 번 나눠서** 평가하고 **평균**을 계산

```
1회차: [Test] [Train] [Train] [Train] [Train]
2회차: [Train] [Test] [Train] [Train] [Train]
3회차: [Train] [Train] [Test] [Train] [Train]
4회차: [Train] [Train] [Train] [Test] [Train]
5회차: [Train] [Train] [Train] [Train] [Test]

→ 5개 점수의 평균 = 최종 성능
```

> **K-Fold Cross Validation** (K=5)

---

# 왜 교차검증인가?

## 더 신뢰할 수 있는 평가

### train_test_split
```
1회 평가: 85%
→ 진짜 85%? 아니면 운?
```

### 교차검증 (5-Fold)
```
1회: 83%, 2회: 86%, 3회: 84%, 4회: 85%, 5회: 82%
→ 평균: 84% (±1.4%)
→ 더 신뢰할 수 있음!
```

---

# sklearn으로 교차검증

## cross_val_score

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

# 5-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5)

print(f"각 Fold 점수: {scores}")
print(f"평균: {scores.mean():.3f}")
print(f"표준편차: {scores.std():.3f}")
```

### 출력 예시
```
각 Fold 점수: [0.83 0.86 0.84 0.85 0.82]
평균: 0.840
표준편차: 0.014
```

---

# 주요 파라미터

## cv 옵션

```python
# 정수: K-Fold
scores = cross_val_score(model, X, y, cv=5)   # 5-Fold
scores = cross_val_score(model, X, y, cv=10)  # 10-Fold

# 불균형 데이터: StratifiedKFold (기본값)
# 각 Fold에서 클래스 비율 유지

# 회귀 문제
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
```

---

# 과대적합 vs 과소적합

## 모델 복잡도와 성능

```
      │
  오  │   ●                             ●
  차  │    ●                          ●
      │      ●                      ●
      │        ●     최적!        ●
      │          ● ● ● ● ● ● ● ●
      └─────────────────────────────────→
         단순              복잡
         모델              모델

     과소적합           과대적합
    (Underfitting)    (Overfitting)
```

---

# 과소적합 (Underfitting)

## 모델이 너무 단순함

### 증상
- 학습 정확도: **낮음** (70%)
- 테스트 정확도: **낮음** (68%)
- 둘 다 낮음!

### 원인
- 모델이 너무 단순
- 특성이 부족
- 학습 부족

### 해결
- 더 복잡한 모델 사용
- 특성 추가
- max_depth 증가

---

# 과대적합 (Overfitting)

## 모델이 학습 데이터를 외움

### 증상
- 학습 정확도: **높음** (98%)
- 테스트 정확도: **낮음** (75%)
- 차이가 큼!

### 원인
- 모델이 너무 복잡
- 데이터가 부족
- 노이즈까지 학습

### 해결
- 더 단순한 모델 사용
- 데이터 추가
- max_depth 제한, 정규화

---

# 진단 방법

## 학습 vs 테스트 점수 비교

```python
from sklearn.model_selection import cross_val_score

# 학습 데이터 점수
train_scores = cross_val_score(model, X_train, y_train, cv=5)

# 테스트 데이터 점수
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"학습 평균: {train_scores.mean():.3f}")
print(f"테스트: {test_score:.3f}")

# 차이가 크면 과대적합!
```

---

# 혼동행렬 (Confusion Matrix)

## 분류 모델 상세 평가

```
              예측
            정상  불량
      정상 [ 45    5 ]  → 실제 정상 50개
실제
      불량 [ 10   40 ]  → 실제 불량 50개

       TN=45  FP=5
       FN=10  TP=40
```

- **TN (True Negative)**: 정상을 정상으로 예측 ✅
- **FP (False Positive)**: 정상을 불량으로 예측 ❌
- **FN (False Negative)**: 불량을 정상으로 예측 ❌
- **TP (True Positive)**: 불량을 불량으로 예측 ✅

---

# 정밀도와 재현율

## Precision & Recall

### 정밀도 (Precision)
$$Precision = \frac{TP}{TP + FP} = \frac{40}{40+5} = 88.9\%$$
> "불량이라고 예측한 것 중 진짜 불량 비율"

### 재현율 (Recall)
$$Recall = \frac{TP}{TP + FN} = \frac{40}{40+10} = 80.0\%$$
> "실제 불량 중 잡아낸 비율"

---

# 정밀도 vs 재현율

## 상황에 따라 중요도가 다름

| 상황 | 중요한 지표 | 이유 |
|------|------------|------|
| 스팸 메일 필터 | 정밀도 | 중요한 메일을 스팸으로 분류하면 안됨 |
| 암 진단 | 재현율 | 암 환자를 놓치면 안됨 |
| 제조 불량 검출 | 재현율 | 불량품을 놓치면 안됨 |
| 사기 탐지 | 균형 | 둘 다 중요 |

---

# F1 Score

## 정밀도와 재현율의 조화평균

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.3f}")
```

### 특징
- 0~1 사이, 1에 가까울수록 좋음
- 정밀도와 재현율이 **균형**일 때 높음
- 둘 중 하나가 낮으면 F1도 낮아짐

---

# sklearn 분류 리포트

## classification_report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
                            target_names=['정상', '불량']))
```

```
              precision    recall  f1-score   support

        정상       0.82      0.90      0.86        50
        불량       0.89      0.80      0.84        50

    accuracy                           0.85       100
   macro avg       0.85      0.85      0.85       100
weighted avg       0.85      0.85      0.85       100
```

---

# 혼동행렬 시각화

## ConfusionMatrixDisplay

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['정상', '불량'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

---

# 회귀 모델 평가 복습

## 14차시 내용

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# MSE, RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# R²
r2 = r2_score(y_test, y_pred)
```

---

# 평가 지표 정리

## 분류 vs 회귀

| | 분류 | 회귀 |
|--|------|------|
| 기본 지표 | 정확도 (Accuracy) | R² (결정계수) |
| 상세 지표 | 정밀도, 재현율, F1 | MSE, RMSE, MAE |
| 시각화 | 혼동행렬 | 예측 vs 실제 산점도 |
| 교차검증 | cross_val_score | cross_val_score |

---

# 실습 정리

## 모델 평가 체크리스트

```python
# 1. 교차검증
scores = cross_val_score(model, X, y, cv=5)
print(f"CV 평균: {scores.mean():.3f} (±{scores.std():.3f})")

# 2. 과대적합 확인
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train: {train_score:.3f}, Test: {test_score:.3f}")

# 3. 상세 평가 (분류)
print(classification_report(y_test, y_pred))
```

---

# 다음 차시 예고

## 16차시: 하이퍼파라미터 튜닝

- GridSearchCV
- RandomizedSearchCV
- 최적의 파라미터 찾기

> 모델 성능을 **최대로** 끌어올리는 방법!

---

# 감사합니다

## AI 기초체력훈련 15차시

**모델 평가와 교차검증**

신뢰할 수 있는 평가 방법을 배웠습니다!
