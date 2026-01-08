"""
[16차시] 모델 평가와 반복 검증 - 실습 코드
=============================================

학습 목표:
1. 교차검증의 원리를 이해한다
2. 과대적합/과소적합을 진단한다
3. 다양한 평가 지표를 활용한다

실습 내용:
- Wine 데이터셋을 활용한 분류 평가
- California Housing 데이터셋을 활용한 회귀 평가
- K-Fold 교차검증 (cross_val_score)
- 학습 곡선 (learning_curve)
- 검증 곡선 (validation_curve)
- 혼동행렬 및 분류 보고서
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    learning_curve, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.datasets import load_wine, fetch_california_housing

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 데이터 로딩
# ============================================================

print("=" * 60)
print("Part 1: 데이터셋 로딩")
print("=" * 60)

# 분류용 데이터: Wine 데이터셋
print("\n[Wine 데이터셋 로딩 중...]")
try:
    wine = load_wine()
    df_clf = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_clf['target'] = wine.target
    print("Wine 데이터셋 로딩 완료!")
except Exception as e:
    print(f"Wine 데이터셋 로딩 실패: {e}")
    raise

print(f"\n[분류 데이터 - Wine]")
print(f"데이터 크기: {df_clf.shape}")
print(f"클래스: {list(wine.target_names)}")
print(f"클래스별 샘플 수:")
for i, name in enumerate(wine.target_names):
    count = (df_clf['target'] == i).sum()
    print(f"  {name}: {count}개 ({count/len(df_clf):.1%})")

# 회귀용 데이터: California Housing
print("\n[California Housing 데이터셋 로딩 중...]")
try:
    housing = fetch_california_housing()
    df_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
    df_reg['MedHouseVal'] = housing.target
    # 샘플링 (전체 데이터가 크므로 5000개만 사용)
    df_reg = df_reg.sample(n=5000, random_state=42).reset_index(drop=True)
    print("California Housing 데이터셋 로딩 완료!")
except Exception as e:
    print(f"California Housing 데이터셋 로딩 실패: {e}")
    raise

print(f"\n[회귀 데이터 - California Housing (샘플)]")
print(f"데이터 크기: {df_reg.shape}")
print(f"주택 가격 범위: ${df_reg['MedHouseVal'].min()*100000:,.0f} ~ ${df_reg['MedHouseVal'].max()*100000:,.0f}")


# ============================================================
# Part 2: 단일 분할의 불안정성
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 단일 분할의 불안정성")
print("=" * 60)

X_clf = df_clf[wine.feature_names]
y_clf = df_clf['target']

print("\n[여러 번 train_test_split 결과]")
scores_single = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=i
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores_single.append(score)
    print(f"  시도 {i+1:2d}: {score:.3f}")

print(f"\n점수 범위: {min(scores_single):.3f} ~ {max(scores_single):.3f}")
print(f"표준편차: {np.std(scores_single):.3f}")
print("-> 같은 모델인데 점수가 불안정합니다!")


# ============================================================
# Part 3: K-Fold 교차검증
# ============================================================

print("\n" + "=" * 60)
print("Part 3: K-Fold 교차검증")
print("=" * 60)

# cross_val_score 사용
model = RandomForestClassifier(n_estimators=50, random_state=42)
cv_scores = cross_val_score(model, X_clf, y_clf, cv=5)

print("\n[5-Fold 교차검증 결과]")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")

print(f"\n평균: {cv_scores.mean():.3f}")
print(f"표준편차: {cv_scores.std():.3f}")
print(f"결과: {cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})")

# 수동으로 K-Fold 구현
print("\n[수동 K-Fold 구현]")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

manual_scores = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_clf), 1):
    X_train, X_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
    y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    manual_scores.append(score)
    print(f"  Fold {fold}: 학습 {len(train_idx)}개, 테스트 {len(test_idx)}개 -> {score:.3f}")

print(f"\n평균: {np.mean(manual_scores):.3f} (+/-{np.std(manual_scores):.3f})")


# ============================================================
# Part 4: StratifiedKFold (분류용)
# ============================================================

print("\n" + "=" * 60)
print("Part 4: StratifiedKFold (클래스 비율 유지)")
print("=" * 60)

# 불균형 데이터 시뮬레이션
np.random.seed(42)
X_imbalanced = np.random.randn(200, 4)
y_imbalanced = np.array([0]*180 + [1]*20)  # 90% vs 10%

print(f"\n[불균형 데이터]")
print(f"클래스 0: {sum(y_imbalanced==0)} ({sum(y_imbalanced==0)/len(y_imbalanced):.0%})")
print(f"클래스 1: {sum(y_imbalanced==1)} ({sum(y_imbalanced==1)/len(y_imbalanced):.0%})")

# 일반 KFold
print("\n[일반 KFold - 클래스 비율 불균일]")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_imbalanced), 1):
    ratio = y_imbalanced[test_idx].mean()
    print(f"  Fold {fold}: 클래스 1 비율 = {ratio:.1%}")

# StratifiedKFold
print("\n[StratifiedKFold - 클래스 비율 유지]")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skfold.split(X_imbalanced, y_imbalanced), 1):
    ratio = y_imbalanced[test_idx].mean()
    print(f"  Fold {fold}: 클래스 1 비율 = {ratio:.1%}")


# ============================================================
# Part 5: 과대적합/과소적합 실험
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 과대적합/과소적합 실험")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print("\n[의사결정나무 - max_depth별 성능]")
print(f"{'max_depth':>10} {'학습 점수':>12} {'테스트 점수':>12} {'갭':>8} {'상태':>10}")
print("-" * 56)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    if depth is None:
        depth_str = "None"
    else:
        depth_str = str(depth)

    if gap > 0.15:
        status = "과대적합"
    elif train_score < 0.7:
        status = "과소적합"
    else:
        status = "적절"

    print(f"{depth_str:>10} {train_score:>12.3f} {test_score:>12.3f} {gap:>8.3f} {status:>10}")


# ============================================================
# Part 6: 학습 곡선 (Learning Curve)
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 학습 곡선 (Learning Curve)")
print("=" * 60)

model = RandomForestClassifier(n_estimators=50, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    model, X_clf, y_clf, cv=5,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    random_state=42
)

print("\n[학습 곡선 데이터]")
print(f"{'학습 데이터':>12} {'학습 점수':>12} {'검증 점수':>12} {'갭':>8}")
print("-" * 48)

for size, train_mean, val_mean in zip(
    train_sizes,
    train_scores.mean(axis=1),
    val_scores.mean(axis=1)
):
    gap = train_mean - val_mean
    print(f"{size:>12} {train_mean:>12.3f} {val_mean:>12.3f} {gap:>8.3f}")

# 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 적절한 모델 (RandomForest)
ax1 = axes[0]
ax1.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='학습', color='blue')
ax1.fill_between(train_sizes,
                  train_scores.mean(axis=1) - train_scores.std(axis=1),
                  train_scores.mean(axis=1) + train_scores.std(axis=1),
                  alpha=0.2, color='blue')
ax1.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='검증', color='orange')
ax1.fill_between(train_sizes,
                  val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1),
                  alpha=0.2, color='orange')
ax1.set_xlabel('학습 데이터 수')
ax1.set_ylabel('점수')
ax1.set_title('학습 곡선 - RandomForest (Wine)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 과대적합 모델 (DecisionTree, no limit)
model_overfit = DecisionTreeClassifier(random_state=42)
train_sizes2, train_scores2, val_scores2 = learning_curve(
    model_overfit, X_clf, y_clf, cv=5,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    random_state=42
)

ax2 = axes[1]
ax2.plot(train_sizes2, train_scores2.mean(axis=1), 'o-', label='학습', color='blue')
ax2.plot(train_sizes2, val_scores2.mean(axis=1), 'o-', label='검증', color='orange')
ax2.set_xlabel('학습 데이터 수')
ax2.set_ylabel('점수')
ax2.set_title('학습 곡선 - DecisionTree (과대적합)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('16_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n학습 곡선 저장: 16_learning_curves.png")


# ============================================================
# Part 7: 검증 곡선 (Validation Curve)
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 검증 곡선 (Validation Curve)")
print("=" * 60)

param_range = [1, 2, 3, 5, 7, 10, 15, 20]

train_scores_vc, val_scores_vc = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_clf, y_clf,
    param_name="max_depth",
    param_range=param_range,
    cv=5
)

print("\n[max_depth별 검증 곡선]")
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

# 검증 곡선 시각화
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores_vc.mean(axis=1), 'o-', label='학습', color='blue')
plt.fill_between(param_range,
                  train_scores_vc.mean(axis=1) - train_scores_vc.std(axis=1),
                  train_scores_vc.mean(axis=1) + train_scores_vc.std(axis=1),
                  alpha=0.2, color='blue')
plt.plot(param_range, val_scores_vc.mean(axis=1), 'o-', label='검증', color='orange')
plt.fill_between(param_range,
                  val_scores_vc.mean(axis=1) - val_scores_vc.std(axis=1),
                  val_scores_vc.mean(axis=1) + val_scores_vc.std(axis=1),
                  alpha=0.2, color='orange')
plt.axvline(x=best_depth, color='red', linestyle='--', label=f'최적 (depth={best_depth})')
plt.xlabel('max_depth')
plt.ylabel('점수')
plt.title('검증 곡선 - DecisionTreeClassifier (Wine)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('16_validation_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n검증 곡선 저장: 16_validation_curve.png")


# ============================================================
# Part 8: 분류 평가 지표
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 분류 평가 지표")
print("=" * 60)

# 모델 학습 및 예측
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 혼동행렬
cm = confusion_matrix(y_test, y_pred)
print("\n[혼동행렬]")
print(f"              class_0  class_1  class_2")
for i, name in enumerate(wine.target_names):
    print(f"{name:>12}: {cm[i,0]:7}  {cm[i,1]:7}  {cm[i,2]:7}")

# 다중 클래스 평가 지표
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


# ============================================================
# Part 9: 회귀 평가 지표
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 회귀 평가 지표")
print("=" * 60)

X_reg = df_reg[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y_reg = df_reg['MedHouseVal']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 회귀 모델 학습
model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)

# 회귀 평가 지표
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("\n[회귀 평가 지표 (California Housing)]")
print(f"MSE:  {mse:.4f} (평균 제곱 오차)")
print(f"RMSE: {rmse:.4f} (MSE의 제곱근, 약 ${rmse*100000:,.0f})")
print(f"MAE:  {mae:.4f} (평균 절대 오차, 약 ${mae*100000:,.0f})")
print(f"R^2:  {r2:.3f} (결정계수, 1에 가까울수록 좋음)")

print("\n[지표 해석]")
print(f"-> 평균적으로 ${rmse*100000:,.0f} 정도의 오차")
print(f"-> 모델이 데이터 변동의 {r2*100:.1f}%를 설명")


# ============================================================
# Part 10: 모델 비교 (교차검증 활용)
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 모델 비교 (교차검증 활용)")
print("=" * 60)

models = {
    'DecisionTree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'DecisionTree (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest (n=10)': RandomForestClassifier(n_estimators=10, random_state=42),
    'RandomForest (n=50)': RandomForestClassifier(n_estimators=50, random_state=42),
}

print("\n[교차검증으로 모델 비교 (Wine)]")
print(f"{'모델':<25} {'평균':>8} {'표준편차':>8}")
print("-" * 45)

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_clf, y_clf, cv=5)
    results.append({
        'model': name,
        'mean': scores.mean(),
        'std': scores.std()
    })
    print(f"{name:<25} {scores.mean():>8.3f} {scores.std():>8.3f}")

# 최고 모델
best = max(results, key=lambda x: x['mean'])
print(f"\n최고 모델: {best['model']} ({best['mean']:.3f} +/- {best['std']:.3f})")


# ============================================================
# Part 11: 종합 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 11: 종합 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 혼동행렬 히트맵
ax1 = axes[0, 0]
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(wine.target_names, rotation=45)
ax1.set_yticklabels(wine.target_names)
ax1.set_xlabel('예측')
ax1.set_ylabel('실제')
ax1.set_title('혼동행렬 (Wine)')
for i in range(3):
    for j in range(3):
        ax1.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
plt.colorbar(im, ax=ax1)

# 2. 분류 지표 비교
ax2 = axes[0, 1]
metrics = ['정확도', '정밀도', '재현율', 'F1']
values = [accuracy, precision, recall, f1]
colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']
bars = ax2.bar(metrics, values, color=colors)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('점수')
ax2.set_title('분류 평가 지표 (Wine)')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=10)

# 3. 모델별 교차검증 점수
ax3 = axes[1, 0]
model_names = [r['model'].replace(' ', '\n') for r in results]
means = [r['mean'] for r in results]
stds = [r['std'] for r in results]
x_pos = range(len(model_names))
ax3.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, fontsize=8)
ax3.set_ylabel('교차검증 점수')
ax3.set_title('모델별 성능 비교 (Wine)')
ax3.set_ylim(0.7, 1.0)
ax3.grid(True, alpha=0.3, axis='y')

# 4. 회귀: 실제 vs 예측
ax4 = axes[1, 1]
ax4.scatter(y_test_r, y_pred_r, alpha=0.5, s=20)
ax4.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
ax4.set_xlabel('실제 주택 가격')
ax4.set_ylabel('예측 주택 가격')
ax4.set_title(f'회귀 예측 (R^2 = {r2:.3f})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('16_evaluation_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n종합 시각화 저장: 16_evaluation_summary.png")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 16차시 모델 평가와 반복 검증")
print("=" * 60)

print("""
1. 교차검증의 원리
   - 단일 분할은 불안정 (운에 좌우)
   - K-Fold: K번 평가해서 평균
   - cross_val_score(model, X, y, cv=5)
   - 결과: 평균 +/- 표준편차

2. Wine 데이터셋
   - 178개 와인 샘플, 3가지 품종
   - 13개 화학 성분 특성
   - 다중 클래스 분류 연습용

3. 과대적합/과소적합 진단
   - 과대적합: 학습 높고 테스트 낮음 (갭 큼)
   - 과소적합: 둘 다 낮음
   - 학습 곡선: learning_curve()
   - 검증 곡선: validation_curve()

4. 분류 평가 지표
   - 혼동행렬: TP, TN, FP, FN
   - 정밀도: 양성 예측 중 실제 양성
   - 재현율: 실제 양성 중 양성 예측
   - F1: 정밀도와 재현율의 조화평균
   - classification_report() 활용

5. 회귀 평가 지표
   - MSE: 평균 제곱 오차
   - RMSE: MSE 제곱근 (해석 쉬움)
   - MAE: 평균 절대 오차 (이상치 강건)
   - R^2: 결정계수 (1에 가까울수록)

6. sklearn 주요 함수
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
""")

print("\n다음 차시 예고: 17차시 - 모델 설정값 최적화 (GridSearchCV)")
