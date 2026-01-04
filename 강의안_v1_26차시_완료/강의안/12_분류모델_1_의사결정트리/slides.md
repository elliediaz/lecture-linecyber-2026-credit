---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 12차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 분류 모델 (1): 의사결정트리

## 12차시 | AI 기초체력훈련 (Pre AI-Campus)

**첫 번째 AI 모델을 만들어봅시다!**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **의사결정트리**의 원리를 설명한다
2. **DecisionTreeClassifier**로 분류 모델을 구축한다
3. 모델의 **예측과 성능**을 확인한다

---

# 의사결정트리란?

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

> 마치 스무고개처럼 질문으로 분류!

---

# 실생활 예시

## 우산을 가져갈까?

```
        [비가 올 확률 > 50%?]
              /         \
           예            아니오
            |              |
    [외출 시간 > 2시간?]   가져가지 않음
          /      \
        예       아니오
         |          |
     우산 챙김    접이식 우산
```

> 여러분도 매일 의사결정트리를 사용하고 있어요!

---

# 의사결정트리의 구성 요소

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

> "어떤 질문이 데이터를 가장 잘 나눌까?"

### 좋은 질문 vs 나쁜 질문

| 질문 | 결과 |
|------|------|
| 온도 > 88? | 왼쪽: 불량 8개 / 오른쪽: 정상 42개 ✅ |
| 습도 > 50? | 왼쪽: 불량 3개, 정상 22개 / 오른쪽: 불량 5개, 정상 20개 ❌ |

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
# 출력: 예측 결과: 불량

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

# 방법 2: 직접 계산
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.1%}")
```

### 정확도란?
$$정확도 = \frac{맞은 개수}{전체 개수}$$

---

# 트리 시각화

## 모델 해석하기

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=['온도', '습도', '속도'],
          class_names=['정상', '불량'],
          filled=True,
          rounded=True)
plt.title('의사결정트리 시각화')
plt.show()
```

> 왜 그렇게 예측했는지 **눈으로 확인** 가능!

---

# 트리 시각화 결과

## 해석하기

```
              ┌───────────────┐
              │   온도 <= 87.5  │
              │  samples=160   │
              │  불량률=12%     │
              └───────┬───────┘
                 ┌────┴────┐
                 ▼         ▼
          ┌──────────┐ ┌──────────┐
          │ 정상 예측 │ │ 온도>87.5 │
          │ 128개    │ │ 32개     │
          │ 순도 95% │ │ 불량률50%│
          └──────────┘ └──────────┘
```

> "온도가 87.5도 이하면 정상일 가능성 높음"

---

# 하이퍼파라미터

## 트리 조절하기

```python
model = DecisionTreeClassifier(
    max_depth=3,          # 트리 깊이 제한
    min_samples_split=10, # 분할 최소 샘플 수
    min_samples_leaf=5,   # 리프 노드 최소 샘플 수
    random_state=42
)
```

### 주요 파라미터
| 파라미터 | 설명 | 효과 |
|---------|------|------|
| max_depth | 트리 최대 깊이 | 작으면 단순, 크면 복잡 |
| min_samples_split | 분할 최소 샘플 | 크면 분할 억제 |
| min_samples_leaf | 리프 최소 샘플 | 크면 일반화 |

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

### 해결: max_depth 제한!

---

# 과대적합 확인

## 학습 vs 테스트 정확도

```python
# 학습 데이터 정확도
train_acc = model.score(X_train, y_train)

# 테스트 데이터 정확도
test_acc = model.score(X_test, y_test)

print(f"학습 정확도: {train_acc:.1%}")
print(f"테스트 정확도: {test_acc:.1%}")

# 차이가 크면 과대적합!
# 예: 학습 95%, 테스트 70% → 과대적합
```

---

# 의사결정트리의 장단점

## 정리

### 장점 ✅
- **해석이 쉬움**: 왜 그렇게 예측했는지 설명 가능
- **전처리 간단**: 스케일링 불필요
- **빠른 학습**: 계산이 단순함

### 단점 ❌
- **과대적합 위험**: 트리가 너무 깊어질 수 있음
- **불안정함**: 데이터 변화에 민감
- **성능 한계**: 복잡한 패턴 학습에 한계

---

# 실습 정리

## 전체 흐름

```python
# 1. 데이터 준비
X, y = ...

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(...)

# 3. 모델 학습
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 평가
accuracy = model.score(X_test, y_test)
```

---

# 다음 차시 예고

## 13차시: 분류 모델 (2) - 랜덤포레스트

- 랜덤포레스트 원리
- 앙상블 학습이란?
- 의사결정트리의 단점 극복

> 여러 개의 트리를 **숲**처럼 만들면?

---

# 감사합니다

## AI 기초체력훈련 12차시

**분류 모델 (1): 의사결정트리**

첫 번째 AI 모델을 만들어봤습니다!
