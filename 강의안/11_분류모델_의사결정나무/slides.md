---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 11차시'
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

# 분류 모델 (1): 의사결정나무

## 11차시 | Part III. 문제 중심 모델링 실습

**첫 번째 AI 모델을 본격적으로 만들어봅시다!**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **의사결정나무**의 원리를 설명한다
2. **DecisionTreeClassifier**로 분류 모델을 구축한다
3. 모델의 **예측과 성능**을 평가한다

---

# 의사결정나무란?

## Decision Tree

> 질문을 통해 **단계적으로 분류**하는 알고리즘

```
            [온도 > 88?]
               /     \
            예         아니오
             |           |
        [습도 > 60?]   → 정상
           /    \
         예     아니오
          |        |
        불량      정상
```

> 마치 **스무고개**처럼 질문으로 분류!

---

# 제조 현장 예시

## 불량 판정 의사결정

```
        [온도 > 87°C?]
              /         \
           예            아니오
            |              |
    [습도 > 55%?]         정상
          /      \
        예       아니오
         |          |
       불량       [속도 < 90?]
                   /    \
                 불량   정상
```

> 현장 경험이 트리 형태로 표현됨

---

# 의사결정나무의 구성 요소

## Tree 구조

```
        ┌─────────────┐
        │   Root Node │ ← 최상위 노드 (첫 질문)
        │  온도 > 88? │
        └──────┬──────┘
          ┌────┴────┐
          ▼         ▼
    ┌─────────┐ ┌───────────┐
    │ Branch  │ │ Leaf Node │ ← 최종 결과
    │습도>60? │ │   정상     │
    └────┬────┘ └───────────┘
      ┌──┴──┐
      ▼     ▼
   ┌────┐ ┌────┐
   │불량│ │정상│  ← Leaf Nodes
   └────┘ └────┘
```

---

# 어떻게 질문을 만들까?

## 정보 이득 (Information Gain)

> "어떤 질문이 데이터를 **가장 잘 나눌까**?"

### 좋은 질문 vs 나쁜 질문

| 질문 | 결과 |
|------|------|
| 온도 > 88? | 왼쪽: 불량 많음 / 오른쪽: 정상 많음 ✅ |
| 습도 > 50? | 양쪽 다 섞여 있음 ❌ |

> 잘 나누는 질문 = **순도(Purity)가 높아지는** 질문

---

# sklearn으로 구현하기

## DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
X = df[['온도', '습도', '속도']]
y = df['불량여부']

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

# 예측하기

## predict 메서드

```python
# 새로운 데이터 예측
new_data = [[90, 55, 100]]  # 온도 90, 습도 55, 속도 100
prediction = model.predict(new_data)

print(f"예측 결과: {'불량' if prediction[0] == 1 else '정상'}")

# 테스트 데이터 전체 예측
y_pred = model.predict(X_test)
print(f"예측 개수: {len(y_pred)}")
```

---

# 예측 확률 보기

## predict_proba 메서드

```python
# 각 클래스에 속할 확률
proba = model.predict_proba([[90, 55, 100]])
print(f"정상 확률: {proba[0][0]:.1%}")
print(f"불량 확률: {proba[0][1]:.1%}")

# 출력 예시:
# 정상 확률: 25.0%
# 불량 확률: 75.0%
```

> 단순히 "불량"이 아니라 **"75% 확률로 불량"** 이라고 알려줌

---

# 모델 성능 평가

## 정확도 (Accuracy)

```python
# 방법 1: score 메서드
accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.1%}")

# 방법 2: accuracy_score 함수
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

### 정확도란?
$$정확도 = \frac{맞은\ 개수}{전체\ 개수}$$

---

# 이론 정리

## 의사결정나무 핵심

| 항목 | 설명 |
|------|------|
| 원리 | 질문으로 단계적 분류 |
| 장점 | 해석 쉬움, 시각화 가능 |
| 단점 | 과대적합 위험 |
| 조절 | max_depth로 깊이 제한 |

---

# - 실습편 -

## 11차시

**불량 분류 모델 구축 실습**

---

# 실습 개요

## 제조 데이터로 불량 예측

### 목표
- 온도, 습도, 속도로 불량 여부 예측
- 의사결정나무 모델 학습
- 성능 평가 및 시각화

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---

# 실습 1: 데이터 생성

## 제조 데이터 준비

```python
np.random.seed(42)
n = 300

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

# 불량 여부 (온도, 습도 영향)
defect_prob = 0.05 + 0.03*(df['온도']-80)/5 + 0.02*(df['습도']-40)/10
df['불량여부'] = (np.random.random(n) < defect_prob).astype(int)

print(f"불량 비율: {df['불량여부'].mean():.1%}")
```

---

# 실습 2: 데이터 분리

## 특성과 타겟, 학습/테스트

```python
# 특성과 타겟
X = df[['온도', '습도', '속도']]
y = df['불량여부']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
```

---

# 실습 3: 모델 학습

## DecisionTreeClassifier

```python
# 모델 생성 (깊이 제한)
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# 학습
model.fit(X_train, y_train)

# 트리 정보
print(f"트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")
```

---

# 실습 4: 예측

## predict와 predict_proba

```python
# 새 데이터 예측
new_data = [[90, 55, 100]]
pred = model.predict(new_data)
proba = model.predict_proba(new_data)

print(f"예측: {'불량' if pred[0]==1 else '정상'}")
print(f"불량 확률: {proba[0][1]:.1%}")

# 테스트 데이터 예측
y_pred = model.predict(X_test)
```

---

# 실습 5: 성능 평가

## 정확도와 과대적합 확인

```python
# 학습 vs 테스트 정확도
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"학습 정확도: {train_acc:.1%}")
print(f"테스트 정확도: {test_acc:.1%}")

# 차이가 크면 과대적합
diff = train_acc - test_acc
if diff > 0.1:
    print("⚠️ 과대적합 의심")
else:
    print("✅ 적절한 일반화")
```

---

# 실습 6: 트리 시각화

## plot_tree 활용

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=['온도', '습도', '속도'],
          class_names=['정상', '불량'],
          filled=True,
          rounded=True)
plt.title('의사결정나무 시각화')
plt.show()
```

> 왜 그렇게 예측했는지 **눈으로 확인** 가능!

---

# 실습 7: 특성 중요도

## 어떤 변수가 중요할까?

```python
importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print(importance)

# 시각화
importance.plot(kind='barh', x='특성', y='중요도')
plt.title('특성 중요도')
plt.show()
```

---

# 실습 8: 하이퍼파라미터 실험

## max_depth 변화

```python
for depth in [1, 2, 3, 5, 10, None]:
    model_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model_temp.fit(X_train, y_train)
    train_acc = model_temp.score(X_train, y_train)
    test_acc = model_temp.score(X_test, y_test)
    print(f"depth={depth}: 학습={train_acc:.1%}, 테스트={test_acc:.1%}")
```

> max_depth가 너무 크면 **과대적합** 발생!

---

# 과대적합 문제

## Overfitting

```
      [깊은 트리]              [적당한 트리]
         /\                       /\
        /  \                     /  \
       /\  /\                  정상  불량
      /\\/\\/\
   ... (너무 복잡!)

   → 학습: 99%              → 학습: 85%
   → 테스트: 70%            → 테스트: 83%
```

> 학습 데이터를 **외웠지만**, 새 데이터에서는 성능 저하

---

# 의사결정나무의 장단점

## 정리

### 장점 ✅
- **해석 쉬움**: 왜 그렇게 예측했는지 설명 가능
- **전처리 간단**: 스케일링 불필요
- **빠른 학습**: 계산이 단순함

### 단점 ❌
- **과대적합 위험**: 트리가 너무 깊어질 수 있음
- **불안정함**: 데이터 변화에 민감
- **성능 한계**: 복잡한 패턴에 한계

---

# 실습 정리

## 핵심 체크포인트

### 모델링 흐름
- [ ] 데이터 준비 (X, y 분리)
- [ ] 학습/테스트 분리
- [ ] 모델 생성 및 학습 (fit)
- [ ] 예측 (predict, predict_proba)
- [ ] 평가 (score, 과대적합 확인)
- [ ] 시각화 (plot_tree, 특성 중요도)

---

# 다음 차시 예고

## 12차시: 분류 모델 (2) - 랜덤포레스트

### 학습 내용
- 랜덤포레스트 원리
- 앙상블 학습이란?
- 의사결정나무의 단점 극복

> 여러 개의 트리를 **숲**처럼 만들면?

---

# 정리 및 Q&A

## 오늘의 핵심

1. **의사결정나무**: 질문으로 단계적 분류
2. **sklearn 사용법**: fit → predict → score
3. **과대적합 주의**: max_depth로 제한
4. **해석 가능**: plot_tree로 시각화

---

# 감사합니다

## 11차시: 분류 모델 (1) - 의사결정나무

**첫 번째 AI 모델을 만들어봤습니다!**
