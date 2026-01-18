---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 13차시'
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

# 분류 모델 (1) - 의사결정나무

## 13차시 | Part III. 문제 중심 모델링 실습

**첫 번째 분류 모델을 만듭니다!**

---

# 지난 시간 복습

## 11차시에서 배운 것

- **머신러닝**: 데이터에서 패턴 학습
- **지도학습**: 정답 있는 데이터로 학습
- **분류**: 범주 예측 / **회귀**: 숫자 예측
- **sklearn 패턴**: fit → predict → score

<div class="tip">

오늘은 첫 번째 분류 모델, **의사결정나무**를 배웁니다!

</div>

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **의사결정나무의 원리**를 이해한다 |
| 2 | **DecisionTreeClassifier**를 사용한다 |
| 3 | **트리 구조를 시각화하고 해석**한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  의사결정   │ →  │   sklearn   │ →  │  시각화와   │
│  나무 원리  │    │    실습     │    │    해석     │
└─────────────┘    └─────────────┘    └─────────────┘
   질문 기반        모델 학습/예측      트리 구조
   분기 구조        주요 파라미터       특성 중요도
```

---

<!-- _class: lead -->

# Part 1

## 의사결정나무의 원리

---

# 의사결정나무란?

## Decision Tree

> 질문을 통해 데이터를 분류하는 알고리즘
> "스무고개"와 같은 원리!

### 핵심 아이디어
- 예/아니오로 답할 수 있는 질문을 던짐
- 질문에 따라 데이터를 나눔
- 최종적으로 클래스(범주) 결정

---

# 일상 속 의사결정나무

## 과일 분류 예시

```
                "빨간색인가요?"
                      │
            ┌─────────┴─────────┐
           예                  아니오
            │                    │
    "동그란가요?"           "노란색인가요?"
          │                      │
      ┌───┴───┐              ┌───┴───┐
     예       아니오         예       아니오
      │         │            │         │
   사과      딸기        바나나      포도
```

---

# 제조 현장 예시

## 불량품 분류

```
                "온도 > 85도?"
                      │
            ┌─────────┴─────────┐
           예                  아니오
            │                    │
    "습도 > 60%?"              정상
          │
      ┌───┴───┐
     예       아니오
      │         │
   불량       정상
```

---

# 의사결정나무 용어

## 트리 구조

```
        [루트 노드]        ← 첫 번째 질문 (가장 중요한 특성)
           │
    ┌──────┴──────┐
[내부 노드]    [내부 노드]  ← 추가 질문
    │              │
┌───┴───┐    ┌────┴────┐
[리프]  [리프] [리프] [리프]  ← 최종 결정 (클래스)
```

- **루트 노드**: 첫 번째 분기점
- **내부 노드**: 중간 분기점
- **리프 노드**: 최종 결정

---

# 어떤 질문이 좋은 질문인가?

## 불순도 (Impurity)

> 데이터가 얼마나 "섞여있는지" 측정

### 좋은 분할
- 분할 후 각 그룹이 **한 클래스로 통일**
- 불순도가 **낮아짐**

### 나쁜 분할
- 분할 후에도 클래스가 **섞여있음**
- 불순도가 **여전히 높음**

---

# 불순도 예시

## 어떤 분할이 더 좋을까요?

```
[분할 전]
🟢🟢🟢🟢🔴🔴🔴🔴 (50% 정상, 50% 불량)

[분할 A - 좋은 분할]
왼쪽: 🟢🟢🟢🟢    오른쪽: 🔴🔴🔴🔴
      (100% 정상)         (100% 불량)

[분할 B - 나쁜 분할]
왼쪽: 🟢🟢🔴🔴    오른쪽: 🟢🟢🔴🔴
      (50% 정상)          (50% 정상)
```

**분할 A**가 불순도를 더 많이 낮췄습니다!

---

# 지니 불순도 (Gini Impurity)

## 가장 많이 사용하는 불순도 측정

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

### 직관적 이해
- Gini = 0: 완전히 순수 (한 클래스만 있음)
- Gini = 0.5: 가장 불순 (이진 분류에서 50:50)

---

# 지니 불순도 계산 예시

## 두 경우 비교

### Case A: [정상 10개, 불량 0개]
```
Gini = 1 - (10/10)² - (0/10)² = 0
→ 완전히 순수!
```

### Case B: [정상 5개, 불량 5개]
```
Gini = 1 - (5/10)² - (5/10)² = 0.5
→ 가장 불순!
```

### Case C: [정상 8개, 불량 2개]
```
Gini = 1 - (8/10)² - (2/10)² = 0.32
```

---

# 정보 이득 (Information Gain)

## 분할로 얼마나 순수해졌나?

```
정보 이득 = 부모 불순도 - 자식 불순도의 가중 평균
```

### 트리 학습 과정
1. 모든 특성, 모든 분할점 검토
2. **정보 이득이 가장 큰** 분할 선택
3. 반복하여 트리 성장

---

# 트리 학습 과정

## 단계별 설명

```
[1단계] 전체 데이터
        모든 특성 검토 → "온도 > 85" 선택 (정보 이득 최대)

[2단계] 왼쪽 노드 (온도 > 85)
        남은 특성 검토 → "습도 > 60" 선택

[3단계] 오른쪽 노드 (온도 ≤ 85)
        이미 충분히 순수 → 분할 중단

... 반복 ...
```

---

# 과대적합 문제

## 트리가 너무 깊어지면?

<div class="important">

### 과대적합 (Overfitting)
- 학습 데이터에 **너무 맞춤**
- 새 데이터에 **일반화 실패**
- 노이즈까지 학습

</div>

### 해결책
- 트리 깊이 제한 (`max_depth`)
- 최소 샘플 수 설정 (`min_samples_split`)
- 가지치기 (Pruning)

---

# 의사결정나무 장단점

## 정리

| 장점 | 단점 |
|------|------|
| 직관적, 해석 용이 | 과대적합 쉬움 |
| 전처리 불필요 | 불안정 (데이터 변화에 민감) |
| 특성 중요도 제공 | 단일 트리 성능 한계 |
| 범주형/수치형 모두 처리 | 축에 평행한 분할만 가능 |

---

<!-- _class: lead -->

# Part 2

## sklearn으로 의사결정나무 실습

---

# DecisionTreeClassifier

## sklearn 클래스

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성
model = DecisionTreeClassifier(
    criterion='gini',      # 불순도 측정 (gini 또는 entropy)
    max_depth=None,        # 트리 최대 깊이
    min_samples_split=2,   # 분할 최소 샘플 수
    random_state=42        # 재현성
)
```

---

# 주요 파라미터

## 과대적합 방지용

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `max_depth` | 트리 최대 깊이 | None (제한 없음) |
| `min_samples_split` | 분할에 필요한 최소 샘플 | 2 |
| `min_samples_leaf` | 리프에 필요한 최소 샘플 | 1 |
| `max_features` | 분할 시 고려할 최대 특성 수 | None (전체) |

---

# max_depth의 영향

## 깊이에 따른 결정 경계

```
max_depth=1          max_depth=3          max_depth=10
(과소적합)           (적절)               (과대적합)

  ┌───────┐          ┌───┬───┐          ┌─┬─┬─┬─┐
  │   A   │          │ A │   │          │A│B│A│B│
  ├───────┤          ├───┤ B │          ├─┼─┼─┼─┤
  │   B   │          │ B │   │          │B│A│B│A│
  └───────┘          └───┴───┘          └─┴─┴─┴─┘

단순               적절한 복잡도        너무 복잡
```

---

# 실습 데이터 생성

## 제조 불량 분류 데이터

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
    'pressure': np.random.normal(1.0, 0.1, n),
})

# 불량 여부 (온도, 습도에 영향)
defect_prob = 0.1 + 0.03*(df['temperature']-80) + 0.01*(df['humidity']-45)
df['defect'] = (np.random.random(n) < defect_prob).astype(int)
```

---

# 데이터 준비

## 특성/타겟 분리 및 학습/테스트 분할

```python
from sklearn.model_selection import train_test_split

# 특성과 타겟 분리
X = df[['temperature', 'humidity', 'speed', 'pressure']]
y = df['defect']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 클래스 비율 유지
)

print(f"학습: {len(X_train)}개, 테스트: {len(X_test)}개")
```

---

# 모델 학습

## fit 메서드

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성 (깊이 제한)
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# 학습
model.fit(X_train, y_train)

print("학습 완료!")
print(f"트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")
```

---

# 예측

## predict 메서드

```python
# 테스트 데이터 예측
y_pred = model.predict(X_test)

# 확률 예측 (각 클래스에 속할 확률)
y_proba = model.predict_proba(X_test)

print("예측 결과 (처음 10개):")
print(f"실제: {list(y_test[:10])}")
print(f"예측: {list(y_pred[:10])}")

print("\n확률 예측 (처음 5개):")
for i in range(5):
    print(f"  샘플 {i+1}: 정상 {y_proba[i][0]:.1%}, 불량 {y_proba[i][1]:.1%}")
```

---

# 평가

## 정확도와 혼동 행렬

```python
from sklearn.metrics import accuracy_score, confusion_matrix

# 정확도
accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.1%}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print("\n혼동 행렬:")
print(f"           예측: 정상  불량")
print(f"실제 정상: {cm[0,0]:6}  {cm[0,1]:4}")
print(f"실제 불량: {cm[1,0]:6}  {cm[1,1]:4}")
```

---

# max_depth 실험

## 깊이별 성능 비교

```python
# 다양한 깊이 테스트
depths = [1, 2, 3, 5, 10, None]

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    depth_str = str(depth) if depth else "None"
    print(f"깊이 {depth_str:>4}: 학습 {train_acc:.1%}, 테스트 {test_acc:.1%}")
```

---

# max_depth 실험 결과

## 과대적합 관찰

```
깊이    1: 학습 85.5%, 테스트 84.0%  ← 과소적합
깊이    2: 학습 87.3%, 테스트 86.0%
깊이    3: 학습 89.0%, 테스트 87.0%  ← 적절
깊이    5: 학습 92.5%, 테스트 86.0%
깊이   10: 학습 98.0%, 테스트 82.0%  ← 과대적합 시작
깊이 None: 학습 100%, 테스트 78.0%  ← 심한 과대적합
```

<div class="tip">

학습 정확도와 테스트 정확도 차이가 크면 **과대적합**!

</div>

---

<!-- _class: lead -->

# Part 3

## 트리 시각화와 해석

---

# 트리 시각화

## plot_tree 함수

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=['온도', '습도', '속도', '압력'],
    class_names=['정상', '불량'],
    filled=True,        # 색상 채우기
    rounded=True,       # 모서리 둥글게
    fontsize=10
)
plt.title("의사결정나무 시각화")
plt.savefig('decision_tree.png', dpi=150)
plt.show()
```

---

# 트리 노드 해석

## 노드 정보 읽기

```
        ┌─────────────────┐
        │  온도 <= 87.5   │  ← 분할 조건
        │  gini = 0.42    │  ← 불순도
        │  samples = 400  │  ← 샘플 수
        │  value = [320, 80]│  ← 클래스별 샘플 수
        │  class = 정상    │  ← 다수 클래스
        └─────────────────┘
```

---

# 텍스트로 트리 보기

## export_text 함수

```python
from sklearn.tree import export_text

tree_rules = export_text(
    model,
    feature_names=['temperature', 'humidity', 'speed', 'pressure']
)
print(tree_rules)
```

출력 예시:
```
|--- temperature <= 87.50
|   |--- humidity <= 55.00
|   |   |--- class: 정상
|   |--- humidity > 55.00
|   |   |--- temperature <= 84.50
|   |   |   |--- class: 정상
|   |   |--- temperature > 84.50
|   |   |   |--- class: 불량
|--- temperature > 87.50
|   |--- class: 불량
```

---

# 특성 중요도

## feature_importances_

```python
# 특성 중요도 확인
importances = model.feature_importances_
feature_names = ['temperature', 'humidity', 'speed', 'pressure']

# 중요도 순으로 정렬
sorted_idx = np.argsort(importances)[::-1]

print("특성 중요도:")
for idx in sorted_idx:
    print(f"  {feature_names[idx]}: {importances[idx]:.3f}")
```

---

# 특성 중요도 시각화

## 막대 그래프

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.barh(range(len(importances)), importances[sorted_idx])
plt.yticks(range(len(importances)),
           [feature_names[i] for i in sorted_idx])
plt.xlabel('특성 중요도')
plt.title('의사결정나무 - 특성 중요도')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

---

# 특성 중요도 해석

## 실무 활용

```
특성 중요도:
  temperature: 0.650  ← 가장 중요!
  humidity:    0.280
  pressure:    0.050
  speed:       0.020
```

### 인사이트
- **온도**가 불량 판정에 가장 중요한 요소
- **습도**도 유의미한 영향
- 속도와 압력은 영향이 작음

→ 온도 관리에 **집중 투자** 권장

---

# 결정 경계 시각화

## 2개 특성으로 시각화

```python
from sklearn.tree import DecisionTreeClassifier

# 2개 특성만 선택
X_2d = X_train[['temperature', 'humidity']]
model_2d = DecisionTreeClassifier(max_depth=4, random_state=42)
model_2d.fit(X_2d, y_train)

# 결정 경계 그리기
xx, yy = np.meshgrid(
    np.linspace(X_2d['temperature'].min()-2, X_2d['temperature'].max()+2, 200),
    np.linspace(X_2d['humidity'].min()-5, X_2d['humidity'].max()+5, 200)
)
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
```

---

# 결정 경계 시각화 결과

## 축에 평행한 분할

```
     습도
      ↑
   80 ┼─────┬─────────────┐
      │불량 │             │
   60 ┼─────┼──────┬──────┤
      │     │ 불량 │      │
   40 ┼  정상│      │ 정상 │
      │     │      │      │
   20 ┼─────┴──────┴──────┘
      └──────────────────→ 온도
          75    85    95
```

의사결정나무는 **축에 평행한 선**으로만 분할합니다.

---

<!-- _class: lead -->

# 실습편

## 제조 불량 분류 프로젝트

---

# 실습 목표

## 완전한 분류 파이프라인

1. 데이터 로드 및 탐색
2. 특성/타겟 분리
3. 학습/테스트 분할
4. 의사결정나무 학습
5. 평가 및 해석
6. 최적 깊이 찾기

---

# 실습 1: 데이터 탐색

## 기본 통계 확인

```python
import pandas as pd
import numpy as np

# 데이터 생성 (또는 로드)
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
    'pressure': np.random.normal(1.0, 0.1, n),
})
defect_prob = 0.1 + 0.03*(df['temperature']-80) + 0.01*(df['humidity']-45)
df['defect'] = (np.random.random(n) < defect_prob).astype(int)

print(df.describe())
print(f"\n불량률: {df['defect'].mean():.1%}")
```

---

# 실습 2: 데이터 분할

## 80:20 분할

```python
from sklearn.model_selection import train_test_split

X = df[['temperature', 'humidity', 'speed', 'pressure']]
y = df['defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"학습 세트: {len(X_train)}개")
print(f"테스트 세트: {len(X_test)}개")
print(f"학습 불량률: {y_train.mean():.1%}")
print(f"테스트 불량률: {y_test.mean():.1%}")
```

---

# 실습 3: 모델 학습

## 기본 모델

```python
from sklearn.tree import DecisionTreeClassifier

# 기본 모델 (깊이 제한)
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

print(f"트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")
```

---

# 실습 4: 평가

## 다양한 평가 지표

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

# 정확도
print(f"정확도: {model.score(X_test, y_test):.1%}")

# 분류 보고서
print("\n분류 보고서:")
print(classification_report(
    y_test, y_pred,
    target_names=['정상', '불량']
))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬:")
print(cm)
```

---

# 실습 5: 최적 깊이 찾기

## 학습/테스트 정확도 비교

```python
train_scores = []
test_scores = []
depths = range(1, 16)

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# 최적 깊이
best_depth = depths[np.argmax(test_scores)]
best_score = max(test_scores)
print(f"최적 깊이: {best_depth}, 테스트 정확도: {best_score:.1%}")
```

---

# 실습 6: 최적 깊이 시각화

## 학습 곡선

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='학습 정확도')
plt.plot(depths, test_scores, 'o-', label='테스트 정확도')
plt.axvline(x=best_depth, color='red', linestyle='--',
            label=f'최적 깊이={best_depth}')
plt.xlabel('트리 깊이')
plt.ylabel('정확도')
plt.title('깊이별 학습/테스트 정확도')
plt.legend()
plt.grid(True)
plt.savefig('depth_comparison.png', dpi=150)
plt.show()
```

---

# 실습 7: 최종 모델 학습

## 최적 파라미터 적용

```python
# 최적 깊이로 최종 모델
final_model = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=5,
    random_state=42
)
final_model.fit(X_train, y_train)

print(f"최종 모델 정확도: {final_model.score(X_test, y_test):.1%}")

# 특성 중요도
print("\n특성 중요도:")
for name, imp in zip(X.columns, final_model.feature_importances_):
    print(f"  {name}: {imp:.3f}")
```

---

# 실습 8: 새 데이터 예측

## 실제 활용

```python
# 새 제품 데이터
new_data = pd.DataFrame({
    'temperature': [87, 92, 78],
    'humidity': [55, 68, 42],
    'speed': [100, 95, 105],
    'pressure': [1.0, 0.95, 1.05]
})

# 예측
predictions = final_model.predict(new_data)
probabilities = final_model.predict_proba(new_data)

for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
    status = "불량" if pred == 1 else "정상"
    print(f"제품 {i+1}: {status} (불량 확률: {proba[1]:.1%})")
```

---

# 핵심 정리

## 13차시 요약

| 개념 | 설명 |
|------|------|
| **의사결정나무** | 질문 기반으로 데이터 분류 |
| **지니 불순도** | 데이터 섞임 정도 측정 |
| **정보 이득** | 분할로 인한 불순도 감소량 |
| **max_depth** | 과대적합 방지용 깊이 제한 |
| **특성 중요도** | 분류에 기여하는 정도 |

---

# 실무 가이드

## 의사결정나무 사용 시

<div class="highlight">

### 권장 설정
- `max_depth`: 3~10 사이에서 교차검증으로 선택
- `min_samples_leaf`: 전체 데이터의 1~5%
- `class_weight='balanced'`: 불균형 데이터에 유용

### 주의사항
- 단일 트리는 불안정 → **랜덤포레스트** 권장
- 깊이가 너무 깊으면 과대적합
- 축에 평행한 분할만 가능

</div>

---

# 다음 차시 예고

## 13차시: 분류 모델 (2) - 랜덤포레스트

### 학습 내용
- 앙상블 학습의 개념
- 랜덤포레스트 원리 (Bagging)
- 의사결정나무의 약점 보완

<div class="tip">

여러 나무를 모아 **숲**을 만듭니다!

</div>

---

# 감사합니다

## 13차시: 분류 모델 - 의사결정나무

**첫 번째 분류 모델을 완성했습니다!**

Q&A
