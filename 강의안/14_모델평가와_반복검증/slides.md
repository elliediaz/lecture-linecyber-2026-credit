---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 14차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
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
---

# 모델 평가와 반복 검증

## 14차시 | Part III. 문제 중심 모델링 실습

**모델을 제대로 평가하는 방법**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **교차검증**의 개념과 필요성을 이해한다
2. **과대적합/과소적합**을 진단한다
3. **혼동행렬, 정밀도, 재현율**을 해석한다

---

# 지금까지의 평가 방법

## train_test_split의 한계

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

### train_test_split (한 번)
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
```

---

# 과소적합 (Underfitting)

## 모델이 너무 단순함

### 증상
- 학습 정확도: **낮음** (70%)
- 테스트 정확도: **낮음** (68%)
- 둘 다 낮음!

### 해결 방법
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

### 해결 방법
- 더 단순한 모델 사용
- 데이터 추가
- max_depth 제한

---

# 진단 방법

## 학습 vs 테스트 점수 비교

```python
# 학습 데이터 점수
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"학습: {train_score:.3f}")
print(f"테스트: {test_score:.3f}")

# 차이가 크면 과대적합!
if train_score - test_score > 0.1:
    print("⚠️ 과대적합 의심")
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

- **TN**: 정상을 정상으로 예측 ✅
- **FP**: 정상을 불량으로 예측 ❌
- **FN**: 불량을 정상으로 예측 ❌
- **TP**: 불량을 불량으로 예측 ✅

---

# 정밀도와 재현율

## Precision & Recall

### 정밀도 (Precision)
> "불량이라고 예측한 것 중 **진짜 불량** 비율"

$$Precision = \frac{TP}{TP + FP}$$

### 재현율 (Recall)
> "실제 불량 중 **잡아낸** 비율"

$$Recall = \frac{TP}{TP + FN}$$

---

# 정밀도 vs 재현율

## 상황에 따라 중요도가 다름

| 상황 | 중요한 지표 | 이유 |
|------|------------|------|
| 스팸 메일 필터 | 정밀도 | 중요한 메일을 스팸으로 분류하면 안됨 |
| 암 진단 | 재현율 | 암 환자를 놓치면 안됨 |
| **제조 불량 검출** | **재현율** | 불량품을 놓치면 안됨 |

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

---

# 이론 정리

## 모델 평가 핵심

| 항목 | 설명 |
|------|------|
| 교차검증 | 여러 번 나눠서 평가 |
| 과대적합 | 학습 높고 테스트 낮음 |
| 과소적합 | 둘 다 낮음 |
| 혼동행렬 | TN, FP, FN, TP |
| 정밀도 | 예측 정확도 |
| 재현율 | 실제 탐지율 |

---

# - 실습편 -

## 14차시

**교차검증과 분류 평가 실습**

---

# 실습 개요

## 모델 평가 실습

### 목표
- 교차검증으로 성능 평가
- 과대적합 진단
- 혼동행렬과 분류 리포트

### 실습 환경
```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
```

---

# 실습 1: 데이터 준비

## 제조 데이터

```python
np.random.seed(42)
n = 500

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

defect_prob = 0.05 + 0.03*(df['온도']-80)/5
df['불량여부'] = (np.random.random(n) < defect_prob).astype(int)
```

---

# 실습 2: 교차검증

## cross_val_score

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5)

print(f"각 Fold 점수: {scores}")
print(f"평균: {scores.mean():.3f}")
print(f"표준편차: {scores.std():.3f}")
```

---

# 실습 3: 과대적합 진단

## 학습 vs 테스트 비교

```python
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"학습 정확도: {train_score:.1%}")
print(f"테스트 정확도: {test_score:.1%}")
print(f"차이: {(train_score - test_score):.1%}")

if train_score - test_score > 0.1:
    print("⚠️ 과대적합 의심")
else:
    print("✅ 적절한 일반화")
```

---

# 실습 4: 혼동행렬

## confusion_matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("혼동행렬:")
print(cm)

# 시각화
disp = ConfusionMatrixDisplay(cm, display_labels=['정상', '불량'])
disp.plot(cmap='Blues')
plt.show()
```

---

# 실습 5: 정밀도, 재현율, F1

## 상세 평가 지표

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"정밀도: {precision:.3f}")
print(f"재현율: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

---

# 실습 6: 분류 리포트

## classification_report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
                            target_names=['정상', '불량']))
```

```
              precision    recall  f1-score   support

        정상       0.92      0.95      0.93        80
        불량       0.75      0.65      0.70        20

    accuracy                           0.89       100
```

---

# 실습 7: 모델 비교

## 여러 모델 교차검증

```python
from sklearn.tree import DecisionTreeClassifier

models = {
    '의사결정나무': DecisionTreeClassifier(max_depth=5),
    '랜덤포레스트': RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (±{scores.std():.3f})")
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] cross_val_score로 교차검증
- [ ] 학습/테스트 점수로 과대적합 진단
- [ ] confusion_matrix로 혼동행렬 생성
- [ ] precision, recall, f1 계산
- [ ] classification_report로 종합 평가

---

# 다음 차시 예고

## 15차시: 모델 설정값 최적화

### 학습 내용
- GridSearchCV
- RandomizedSearchCV
- 최적의 하이퍼파라미터 찾기

> 모델 성능을 **최대로** 끌어올리는 방법!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **교차검증**: 여러 번 평가 → 신뢰성 확보
2. **과대적합**: 학습 높고 테스트 낮음 → 단순화
3. **과소적합**: 둘 다 낮음 → 복잡화
4. **정밀도/재현율**: 상황에 따라 중요도 다름

---

# 감사합니다

## 14차시: 모델 평가와 반복 검증

**신뢰할 수 있는 평가 방법을 배웠습니다!**
