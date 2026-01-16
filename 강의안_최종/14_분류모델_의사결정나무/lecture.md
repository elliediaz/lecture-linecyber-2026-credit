# [13차시] 분류 모델 (1) - 의사결정나무

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **의사결정나무의 원리**를 지니 불순도와 정보 이득 개념으로 이해함
2. **DecisionTreeClassifier**를 사용하여 분류 모델을 학습함
3. **트리 구조를 시각화하고 해석**하여 특성 중요도를 분석함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Iris** | sklearn.datasets | 의사결정나무 분류 실습 |

Iris 데이터셋은 붓꽃 3종류(setosa, versicolor, virginica)를 꽃받침/꽃잎 크기로 분류하는 대표적인 분류 문제임.

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 의사결정나무의 원리 | 질문 기반 분류, 불순도, 정보 이득 |
| 2 | sklearn으로 의사결정나무 실습 | DecisionTreeClassifier, 주요 파라미터 |
| 3 | 트리 시각화와 해석 | plot_tree, 특성 중요도 |

---

## 파트 1: 의사결정나무의 원리

### 개념 설명

#### 의사결정나무(Decision Tree)란?

질문을 통해 데이터를 분류하는 알고리즘임. "스무고개"와 같은 원리로 동작함.

핵심 아이디어:
- 예/아니오로 답할 수 있는 질문을 던짐
- 질문에 따라 데이터를 나눔
- 최종적으로 클래스(범주) 결정

#### 일상 속 의사결정나무 예시

```
                "빨간색인가요?"
                      |
            +---------+---------+
           예                  아니오
            |                    |
    "동그란가요?"           "노란색인가요?"
          |                      |
      +---+---+              +---+---+
     예       아니오         예       아니오
      |         |            |         |
   사과      딸기        바나나      포도
```

#### 트리 구조 용어

```
        [루트 노드]        <-- 첫 번째 질문 (가장 중요한 특성)
           |
    +------+------+
[내부 노드]    [내부 노드]  <-- 추가 질문
    |              |
+---+---+    +----+----+
[리프]  [리프] [리프] [리프]  <-- 최종 결정 (클래스)
```

- **루트 노드(Root Node)**: 첫 번째 분기점
- **내부 노드(Internal Node)**: 중간 분기점
- **리프 노드(Leaf Node)**: 최종 결정

#### 불순도(Impurity)란?

데이터가 얼마나 "섞여있는지" 측정하는 지표임.

| 분할 유형 | 설명 |
|----------|------|
| 좋은 분할 | 분할 후 각 그룹이 **한 클래스로 통일**, 불순도가 **낮아짐** |
| 나쁜 분할 | 분할 후에도 클래스가 **섞여있음**, 불순도가 **여전히 높음** |

```
[분할 전]
정상정상정상정상불량불량불량불량 (50% 정상, 50% 불량)

[분할 A - 좋은 분할]
왼쪽: 정상정상정상정상    오른쪽: 불량불량불량불량
      (100% 정상)              (100% 불량)

[분할 B - 나쁜 분할]
왼쪽: 정상정상불량불량    오른쪽: 정상정상불량불량
      (50% 정상)               (50% 정상)
```

#### 지니 불순도(Gini Impurity)

가장 많이 사용하는 불순도 측정 방식임.

```
Gini = 1 - Sum(p_i^2)
```

| Gini 값 | 해석 |
|---------|------|
| 0 | 완전히 순수 (한 클래스만 있음) |
| 0.5 | 가장 불순 (이진 분류에서 50:50) |

계산 예시:
- Case A [정상 10개, 불량 0개]: `Gini = 1 - (10/10)^2 - (0/10)^2 = 0` (완전 순수)
- Case B [정상 5개, 불량 5개]: `Gini = 1 - (5/10)^2 - (5/10)^2 = 0.5` (가장 불순)
- Case C [정상 8개, 불량 2개]: `Gini = 1 - (8/10)^2 - (2/10)^2 = 0.32`

#### 정보 이득(Information Gain)

분할로 얼마나 순수해졌는지 측정함.

```
정보 이득 = 부모 불순도 - 자식 불순도의 가중 평균
```

트리 학습 과정:
1. 모든 특성, 모든 분할점 검토
2. **정보 이득이 가장 큰** 분할 선택
3. 반복하여 트리 성장

#### 과대적합(Overfitting) 문제

트리가 너무 깊어지면 학습 데이터에 과도하게 맞춰지는 문제가 발생함.

| 현상 | 설명 |
|------|------|
| 과대적합 | 학습 데이터에 **너무 맞춤**, 새 데이터에 **일반화 실패**, 노이즈까지 학습 |
| 해결책 | 트리 깊이 제한(`max_depth`), 최소 샘플 수 설정(`min_samples_split`), 가지치기(Pruning) |

#### 의사결정나무 장단점

| 장점 | 단점 |
|------|------|
| 직관적, 해석 용이 | 과대적합 쉬움 |
| 전처리 불필요 | 불안정 (데이터 변화에 민감) |
| 특성 중요도 제공 | 단일 트리 성능 한계 |
| 범주형/수치형 모두 처리 | 축에 평행한 분할만 가능 |

### 실습 코드

#### 데이터 로드 및 탐색

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Iris 데이터셋 로딩
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("Iris 데이터셋 로딩 완료")

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"특성 이름: {list(iris.feature_names)}")
print(f"클래스: {list(iris.target_names)}")

print(f"\n클래스별 샘플 수:")
for i, name in enumerate(iris.target_names):
    count = (df['target'] == i).sum()
    print(f"  {name}: {count}개 ({count/len(df):.1%})")
```

#### 데이터 분할

```python
# 특성과 타겟 분리
feature_columns = list(iris.feature_names)
X = df[feature_columns]
y = df['target']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 클래스 비율 유지
)

print(f"[데이터 분할 결과]")
print(f"학습 데이터: {len(X_train)}개 ({len(X_train)/len(X):.0%})")
print(f"테스트 데이터: {len(X_test)}개 ({len(X_test)/len(X):.0%})")
```

### 결과 해설

- Iris 데이터셋은 150개 샘플, 4개 특성, 3개 클래스로 구성됨
- 각 클래스(setosa, versicolor, virginica)가 50개씩 균등하게 분포함
- `stratify=y` 옵션으로 학습/테스트 분할 시에도 클래스 비율이 유지됨

---

## 파트 2: sklearn으로 의사결정나무 실습

### 개념 설명

#### DecisionTreeClassifier 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `criterion` | 불순도 측정 방식 | 'gini' |
| `max_depth` | 트리 최대 깊이 | None (제한 없음) |
| `min_samples_split` | 분할에 필요한 최소 샘플 | 2 |
| `min_samples_leaf` | 리프에 필요한 최소 샘플 | 1 |
| `max_features` | 분할 시 고려할 최대 특성 수 | None (전체) |

#### max_depth의 영향

```
max_depth=1          max_depth=3          max_depth=10
(과소적합)           (적절)               (과대적합)

  +-------+          +---+---+          +-+-+-+-+
  |   A   |          | A |   |          |A|B|A|B|
  +-------+          +---+ B +          +-+-+-+-+
  |   B   |          | B |   |          |B|A|B|A|
  +-------+          +---+---+          +-+-+-+-+

단순               적절한 복잡도        너무 복잡
```

### 실습 코드

#### 기본 의사결정나무 모델

```python
# 모델 생성 (깊이 제한)
model = DecisionTreeClassifier(
    criterion='gini',      # 불순도 측정 방식
    max_depth=5,           # 트리 최대 깊이
    min_samples_split=5,   # 분할 최소 샘플 수
    min_samples_leaf=2,    # 리프 최소 샘플 수
    random_state=42
)

# 학습
model.fit(X_train, y_train)

print("[모델 학습 완료]")
print(f"트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")
```

#### 예측 및 확률 예측

```python
# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"[예측 결과 샘플 (처음 10개)]")
print(f"실제: {list(y_test[:10].values)}")
print(f"예측: {list(y_pred[:10])}")

print(f"\n[예측 결과 (꽃 이름)]")
for i in range(min(5, len(y_pred))):
    actual = iris.target_names[y_test.iloc[i]]
    predicted = iris.target_names[y_pred[i]]
    match = "O" if actual == predicted else "X"
    print(f"  샘플 {i+1}: 실제={actual}, 예측={predicted} [{match}]")

print(f"\n[확률 예측 (처음 5개)]")
for i in range(5):
    print(f"  샘플 {i+1}: setosa {y_proba[i][0]:.1%}, "
          f"versicolor {y_proba[i][1]:.1%}, virginica {y_proba[i][2]:.1%}")
```

#### 모델 평가

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 정확도
accuracy = model.score(X_test, y_test)
print(f"[정확도]")
print(f"학습 정확도: {model.score(X_train, y_train):.1%}")
print(f"테스트 정확도: {accuracy:.1%}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print(f"\n[혼동 행렬]")
print(f"              setosa  versicolor  virginica")
for i, name in enumerate(iris.target_names):
    print(f"  {name:>10}:  {cm[i,0]:5}      {cm[i,1]:5}      {cm[i,2]:5}")

# 분류 보고서
print(f"\n[분류 보고서]")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### max_depth에 따른 과대적합 실험

```python
depths = list(range(1, 11)) + [None]  # 1~10 + 제한없음
train_scores = []
test_scores = []
n_leaves_list = []

print("[깊이별 성능]")
print(f"{'깊이':>5} {'학습정확도':>10} {'테스트정확도':>12} {'리프수':>8}")
print("-" * 40)

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    n_leaves = dt.get_n_leaves()

    train_scores.append(train_acc)
    test_scores.append(test_acc)
    n_leaves_list.append(n_leaves)

    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:>5} {train_acc:>10.1%} {test_acc:>12.1%} {n_leaves:>8}")

# 최적 깊이 찾기
best_idx = np.argmax(test_scores)
best_depth = depths[best_idx]
print(f"\n최적 깊이: {best_depth if best_depth else 'None'}")
print(f"최고 테스트 정확도: {test_scores[best_idx]:.1%}")
```

### 결과 해설

- max_depth가 너무 낮으면 과소적합, 너무 높으면 과대적합이 발생함
- 학습 정확도는 깊이가 깊을수록 계속 상승하지만, 테스트 정확도는 일정 깊이 이후 하락함
- 학습 정확도와 테스트 정확도 차이가 크면 과대적합 징후임

---

## 파트 3: 트리 시각화와 해석

### 개념 설명

#### 트리 노드 정보 읽기

```
        +------------------+
        |  온도 <= 87.5    |  <-- 분할 조건
        |  gini = 0.42     |  <-- 불순도
        |  samples = 400   |  <-- 샘플 수
        |  value = [320, 80]|  <-- 클래스별 샘플 수
        |  class = 정상     |  <-- 다수 클래스
        +------------------+
```

### 실습 코드

#### 트리 시각화

```python
# 시각화용 모델 (깊이 3으로 제한해서 보기 좋게)
model_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
model_viz.fit(X_train, y_train)

# 트리 시각화
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(
    model_viz,
    feature_names=feature_columns,
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title("Iris 의사결정나무 시각화 (max_depth=3)", fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("트리 시각화 저장: decision_tree_visualization.png")
```

#### 텍스트로 트리 규칙 확인

```python
tree_rules = export_text(
    model_viz,
    feature_names=feature_columns
)
print("[트리 규칙]")
print(tree_rules)
```

출력 예시:
```
|--- petal length (cm) <= 2.45
|   |--- class: setosa
|--- petal length (cm) > 2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: versicolor
|   |   |--- petal length (cm) > 4.95
|   |   |   |--- class: virginica
|   |--- petal width (cm) > 1.75
|   |   |--- class: virginica
```

#### 특성 중요도

```python
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("[특성 중요도]")
for idx in sorted_idx:
    print(f"  {feature_columns[idx]}: {importances[idx]:.3f}")

# 특성 중요도 시각화
plt.figure(figsize=(8, 5))
plt.barh(range(len(importances)), importances[sorted_idx], color='steelblue')
plt.yticks(range(len(importances)),
           [feature_columns[i] for i in sorted_idx])
plt.xlabel('특성 중요도')
plt.title('의사결정나무 - Iris 특성 중요도')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("특성 중요도 시각화 저장: feature_importance.png")
```

#### 결정 경계 시각화 (2D)

```python
# 2개 특성만 사용 (petal length, petal width)
X_2d = X_train[['petal length (cm)', 'petal width (cm)']]
model_2d = DecisionTreeClassifier(max_depth=4, random_state=42)
model_2d.fit(X_2d, y_train)

# 결정 경계 그리기
x_min, x_max = X_2d['petal length (cm)'].min() - 0.5, X_2d['petal length (cm)'].max() + 0.5
y_min, y_max = X_2d['petal width (cm)'].min() - 0.3, X_2d['petal width (cm)'].max() + 0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='black', linewidths=0.5)

# 데이터 포인트
colors_map = {0: 'red', 1: 'yellow', 2: 'blue'}
colors = [colors_map[t] for t in y_train]
plt.scatter(X_2d['petal length (cm)'], X_2d['petal width (cm)'],
            c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('의사결정나무 결정 경계 (Iris: petal 특성)\n빨강=setosa, 노랑=versicolor, 파랑=virginica')
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("결정 경계 시각화 저장: decision_boundary.png")
```

#### 최종 모델 및 새 데이터 예측

```python
# 최적 깊이로 최종 모델
final_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)
final_model.fit(X_train, y_train)

print(f"[최종 모델]")
print(f"깊이: {final_model.get_depth()}")
print(f"리프 수: {final_model.get_n_leaves()}")
print(f"학습 정확도: {final_model.score(X_train, y_train):.1%}")
print(f"테스트 정확도: {final_model.score(X_test, y_test):.1%}")

# 새 데이터 예측
new_flowers = pd.DataFrame({
    'sepal length (cm)': [5.0, 6.5, 7.2, 5.5, 6.8],
    'sepal width (cm)': [3.5, 2.8, 3.0, 2.5, 3.2],
    'petal length (cm)': [1.5, 4.5, 6.0, 4.0, 5.5],
    'petal width (cm)': [0.2, 1.5, 2.2, 1.3, 2.0]
})

print("\n[새 꽃 데이터]")
print(new_flowers)

predictions = final_model.predict(new_flowers)
probabilities = final_model.predict_proba(new_flowers)

print("\n[예측 결과]")
for i in range(len(new_flowers)):
    species = iris.target_names[predictions[i]]
    max_prob = max(probabilities[i]) * 100
    print(f"  꽃 {i+1}: {species} (확신도: {max_prob:.1f}%)")
```

### 결과 해설

- petal length(꽃잎 길이)와 petal width(꽃잎 너비)가 가장 중요한 특성으로 나타남
- setosa는 다른 두 종과 명확히 구분되지만, versicolor와 virginica는 경계가 겹치는 영역이 존재함
- 의사결정나무는 축에 평행한 선으로만 분할하는 특성이 있음

---

## 핵심 정리

### 의사결정나무 핵심 개념

| 개념 | 설명 |
|------|------|
| **의사결정나무** | 질문 기반으로 데이터 분류 |
| **지니 불순도** | 데이터 섞임 정도 측정 (0=순수, 0.5=최대) |
| **정보 이득** | 분할로 인한 불순도 감소량 |
| **max_depth** | 과대적합 방지용 깊이 제한 |
| **특성 중요도** | 분류에 기여하는 정도 |

### 주요 파라미터 가이드

| 파라미터 | 권장값 | 효과 |
|----------|--------|------|
| max_depth | 3~10 | 과대적합 방지 |
| min_samples_split | 2~20 | 분할 조건 |
| min_samples_leaf | 1~10 | 리프 조건 |
| criterion | 'gini' | 불순도 측정 |

### sklearn 사용법

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# 모델 생성 및 학습
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 평가
accuracy = model.score(X_test, y_test)

# 특성 중요도
importances = model.feature_importances_

# 시각화
plot_tree(model, feature_names=..., class_names=..., filled=True)
export_text(model, feature_names=...)
```

### 장단점 요약

| 장점 | 단점 |
|------|------|
| 해석 용이 | 과대적합 쉬움 |
| 전처리 불필요 | 불안정 (데이터 변화에 민감) |
| 특성 중요도 제공 | 축 평행 분할만 가능 |

### 실무 권장사항

- 단일 트리는 불안정하므로 **랜덤포레스트** 사용 권장
- `max_depth`는 교차검증으로 최적값 선택
- `class_weight='balanced'`는 불균형 데이터에 유용
