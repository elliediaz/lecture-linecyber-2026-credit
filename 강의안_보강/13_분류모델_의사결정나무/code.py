"""
[13차시] 분류 모델 (1) - 의사결정나무 - 실습 코드
=================================================

학습 목표:
1. 의사결정나무의 원리를 이해한다
2. DecisionTreeClassifier를 사용한다
3. 트리 구조를 시각화하고 해석한다

실습 내용:
- Iris 데이터셋을 활용한 분류
- DecisionTreeClassifier 학습
- max_depth에 따른 과대적합 실험
- 트리 시각화 및 특성 중요도 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.datasets import load_iris

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Part 1: 데이터 로딩 및 탐색
# ============================================================

print("=" * 60)
print("Part 1: Iris 데이터셋 로딩 및 탐색")
print("=" * 60)

# Iris 데이터셋 로딩
print("\n[Iris 데이터셋 로딩 중...]")
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Iris 데이터셋 로딩 완료!")
except Exception as e:
    print(f"데이터셋 로딩 실패: {e}")
    raise

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"특성 이름: {list(iris.feature_names)}")
print(f"클래스: {list(iris.target_names)}")
print(f"\n처음 5행:")
print(df.head())

print(f"\n기술통계:")
print(df.describe())

print(f"\n클래스별 샘플 수:")
for i, name in enumerate(iris.target_names):
    count = (df['target'] == i).sum()
    print(f"  {name}: {count}개 ({count/len(df):.1%})")


# ============================================================
# Part 2: 데이터 분할
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 데이터 분할")
print("=" * 60)

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

print(f"\n[데이터 분할 결과]")
print(f"학습 데이터: {len(X_train)}개 ({len(X_train)/len(X):.0%})")
print(f"테스트 데이터: {len(X_test)}개 ({len(X_test)/len(X):.0%})")

print(f"\n[클래스별 학습 데이터 비율]")
for i, name in enumerate(iris.target_names):
    train_count = (y_train == i).sum()
    test_count = (y_test == i).sum()
    print(f"  {name}: 학습 {train_count}개, 테스트 {test_count}개")


# ============================================================
# Part 3: 기본 의사결정나무 모델
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 기본 의사결정나무 모델")
print("=" * 60)

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

print("\n[모델 학습 완료]")
print(f"트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"\n[예측 결과 샘플 (처음 10개)]")
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


# ============================================================
# Part 4: 모델 평가
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 모델 평가")
print("=" * 60)

# 정확도
accuracy = model.score(X_test, y_test)
print(f"\n[정확도]")
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


# ============================================================
# Part 5: max_depth에 따른 과대적합 실험
# ============================================================

print("\n" + "=" * 60)
print("Part 5: max_depth에 따른 과대적합 실험")
print("=" * 60)

depths = list(range(1, 11)) + [None]  # 1~10 + 제한없음
train_scores = []
test_scores = []
n_leaves_list = []

print("\n[깊이별 성능]")
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


# ============================================================
# Part 6: 트리 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 트리 시각화")
print("=" * 60)

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
plt.savefig('13_decision_tree_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n트리 시각화 저장: 13_decision_tree_visualization.png")


# ============================================================
# Part 7: 텍스트로 트리 규칙 확인
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 트리 규칙 (텍스트)")
print("=" * 60)

tree_rules = export_text(
    model_viz,
    feature_names=feature_columns
)
print("\n[트리 규칙]")
print(tree_rules)


# ============================================================
# Part 8: 특성 중요도
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 특성 중요도")
print("=" * 60)

importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n[특성 중요도]")
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
plt.savefig('13_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n특성 중요도 시각화 저장: 13_feature_importance.png")


# ============================================================
# Part 9: 깊이별 학습 곡선
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 깊이별 학습 곡선")
print("=" * 60)

# None 제외하고 시각화
depths_plot = list(range(1, 11))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 학습/테스트 정확도
ax1 = axes[0]
ax1.plot(depths_plot, train_scores[:-1], 'o-', label='학습 정확도', color='blue')
ax1.plot(depths_plot, test_scores[:-1], 'o-', label='테스트 정확도', color='orange')
if best_depth and best_depth <= 10:
    ax1.axvline(x=best_depth, color='red', linestyle='--',
                label=f'최적 깊이={best_depth}')
ax1.set_xlabel('트리 깊이')
ax1.set_ylabel('정확도')
ax1.set_title('깊이별 학습/테스트 정확도 (Iris)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 과대적합 갭
ax2 = axes[1]
gap = np.array(train_scores[:-1]) - np.array(test_scores[:-1])
ax2.bar(depths_plot, gap, color='coral')
ax2.set_xlabel('트리 깊이')
ax2.set_ylabel('학습-테스트 정확도 차이')
ax2.set_title('과대적합 정도 (갭이 클수록 과대적합)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_depth_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n깊이별 비교 시각화 저장: 13_depth_comparison.png")


# ============================================================
# Part 10: 결정 경계 시각화 (2D)
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 결정 경계 시각화")
print("=" * 60)

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
plt.savefig('13_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n결정 경계 시각화 저장: 13_decision_boundary.png")


# ============================================================
# Part 11: 최종 모델 및 새 데이터 예측
# ============================================================

print("\n" + "=" * 60)
print("Part 11: 최종 모델 및 새 데이터 예측")
print("=" * 60)

# 최적 깊이로 최종 모델
optimal_depth = best_depth if best_depth and best_depth <= 10 else 5
final_model = DecisionTreeClassifier(
    max_depth=optimal_depth,
    min_samples_leaf=2,
    random_state=42
)
final_model.fit(X_train, y_train)

print(f"\n[최종 모델]")
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


# ============================================================
# Part 12: 종합 대시보드
# ============================================================

print("\n" + "=" * 60)
print("Part 12: 종합 대시보드")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 혼동 행렬
ax1 = axes[0, 0]
y_pred_final = final_model.predict(X_test)
cm_final = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm_final, display_labels=iris.target_names).plot(ax=ax1, cmap='Blues')
ax1.set_title(f"혼동 행렬 (정확도: {final_model.score(X_test, y_test):.1%})")

# 2. 특성 중요도
ax2 = axes[0, 1]
importances_final = final_model.feature_importances_
sorted_idx_final = np.argsort(importances_final)[::-1]
ax2.barh(range(len(importances_final)), importances_final[sorted_idx_final], color='steelblue')
ax2.set_yticks(range(len(importances_final)))
ax2.set_yticklabels([feature_columns[i] for i in sorted_idx_final])
ax2.set_xlabel('중요도')
ax2.set_title('특성 중요도')

# 3. 깊이별 정확도
ax3 = axes[1, 0]
ax3.plot(depths_plot, train_scores[:-1], 'o-', label='학습')
ax3.plot(depths_plot, test_scores[:-1], 'o-', label='테스트')
if best_depth and best_depth <= 10:
    ax3.axvline(x=best_depth, color='red', linestyle='--', label='최적')
ax3.set_xlabel('트리 깊이')
ax3.set_ylabel('정확도')
ax3.set_title('깊이별 성능')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 예측 확률 분포
ax4 = axes[1, 1]
proba_test = final_model.predict_proba(X_test)
max_proba = proba_test.max(axis=1)
ax4.hist(max_proba, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
ax4.set_xlabel('최대 예측 확률 (확신도)')
ax4.set_ylabel('빈도')
ax4.set_title('예측 확신도 분포')
ax4.axvline(x=max_proba.mean(), color='red', linestyle='--', label=f'평균: {max_proba.mean():.2f}')
ax4.legend()

plt.tight_layout()
plt.savefig('13_decision_tree_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n종합 대시보드 저장: 13_decision_tree_dashboard.png")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 13차시 의사결정나무")
print("=" * 60)

print("""
1. 의사결정나무란?
   - 질문 기반으로 데이터를 분류하는 알고리즘
   - "스무고개" 원리
   - 직관적이고 해석 가능

2. Iris 데이터셋
   - 150개 붓꽃 샘플 (3종 x 50개)
   - 4개 특성: 꽃받침/꽃잎의 길이와 너비
   - 분류 연습에 이상적인 데이터

3. 핵심 개념
   - 불순도 (Gini): 데이터 섞임 정도 (0=순수, 0.5=최대)
   - 정보 이득: 분할로 인한 불순도 감소량
   - 가지치기: 과대적합 방지

4. 주요 파라미터
   - max_depth: 트리 최대 깊이 (과대적합 방지)
   - min_samples_split: 분할 최소 샘플 수
   - min_samples_leaf: 리프 최소 샘플 수
   - criterion: 불순도 측정 방식 (gini/entropy)

5. sklearn 사용법
   model = DecisionTreeClassifier(max_depth=5)
   model.fit(X_train, y_train)
   model.predict(X_test)
   model.score(X_test, y_test)
   model.feature_importances_

6. 시각화 및 해석
   - plot_tree(): 트리 구조 시각화
   - export_text(): 규칙 텍스트 출력
   - feature_importances_: 특성 중요도

7. 장단점
   장점: 해석 용이, 전처리 불필요, 특성 중요도
   단점: 과대적합 쉬움, 불안정, 축 평행 분할만
""")

print("\n다음 차시 예고: 14차시 - 분류 모델: 랜덤포레스트")
