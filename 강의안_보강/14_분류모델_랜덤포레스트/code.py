"""
[14차시] 분류 모델 (2) - 랜덤포레스트 - 실습 코드
=================================================

학습 목표:
1. 앙상블 학습의 개념을 설명한다
2. 랜덤포레스트의 원리를 이해한다
3. RandomForestClassifier를 사용한다

실습 내용:
- Breast Cancer 데이터셋 활용 (유방암 진단)
- 의사결정나무 vs 랜덤포레스트 비교
- n_estimators 실험
- 특성 중요도 분석
- 안정성 비교 실험
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.datasets import load_breast_cancer
import time

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Part 1: 데이터 로딩
# ============================================================

print("=" * 60)
print("Part 1: Breast Cancer 데이터셋 로딩")
print("=" * 60)

# Breast Cancer 데이터셋 로딩
print("\n[Breast Cancer 데이터셋 로딩 중...]")
try:
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
    print("Breast Cancer 데이터셋 로딩 완료!")
except Exception as e:
    print(f"데이터셋 로딩 실패: {e}")
    raise

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"특성 개수: {len(cancer.feature_names)}")
print(f"클래스: {list(cancer.target_names)}")  # malignant(악성), benign(양성)

print(f"\n주요 특성 (처음 10개):")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  {i+1}. {name}")

print(f"\n클래스별 샘플 수:")
for i, name in enumerate(cancer.target_names):
    count = (df['target'] == i).sum()
    print(f"  {name}: {count}개 ({count/len(df):.1%})")

print(f"\n처음 5행:")
print(df[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'target']].head())


# ============================================================
# Part 2: 데이터 분할
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 데이터 분할")
print("=" * 60)

# 주요 특성 선택 (30개 중 10개 사용)
feature_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                   'mean smoothness', 'mean compactness', 'mean concavity',
                   'mean concave points', 'mean symmetry', 'mean fractal dimension']
X = df[feature_columns]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"\n학습 데이터 악성 비율: {(y_train == 0).mean():.1%}")
print(f"테스트 데이터 악성 비율: {(y_test == 0).mean():.1%}")


# ============================================================
# Part 3: 의사결정나무 vs 랜덤포레스트 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 의사결정나무 vs 랜덤포레스트 비교")
print("=" * 60)

# 의사결정나무
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
start_time = time.time()
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - start_time

dt_train_score = dt_model.score(X_train, y_train)
dt_test_score = dt_model.score(X_test, y_test)

print(f"\n[의사결정나무]")
print(f"  학습 시간: {dt_train_time:.3f}초")
print(f"  학습 정확도: {dt_train_score:.1%}")
print(f"  테스트 정확도: {dt_test_score:.1%}")

# 랜덤포레스트
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

rf_train_score = rf_model.score(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)

print(f"\n[랜덤포레스트 (n_estimators=100)]")
print(f"  학습 시간: {rf_train_time:.3f}초")
print(f"  학습 정확도: {rf_train_score:.1%}")
print(f"  테스트 정확도: {rf_test_score:.1%}")
print(f"  트리 개수: {len(rf_model.estimators_)}")

print(f"\n[비교 결과]")
print(f"  테스트 정확도 차이: {rf_test_score - dt_test_score:+.1%}")


# ============================================================
# Part 4: OOB 점수 활용
# ============================================================

print("\n" + "=" * 60)
print("Part 4: OOB (Out-of-Bag) 점수")
print("=" * 60)

rf_oob = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)

print(f"\n[OOB 점수 활용]")
print(f"  OOB 점수: {rf_oob.oob_score_:.3f}")
print(f"  테스트 점수: {rf_oob.score(X_test, y_test):.3f}")
print(f"  차이: {abs(rf_oob.oob_score_ - rf_oob.score(X_test, y_test)):.3f}")
print("\n  -> OOB 점수가 테스트 점수와 유사합니다!")


# ============================================================
# Part 5: n_estimators 실험
# ============================================================

print("\n" + "=" * 60)
print("Part 5: n_estimators (트리 개수) 실험")
print("=" * 60)

estimators_range = [10, 25, 50, 75, 100, 150, 200, 300]
train_scores = []
test_scores = []
train_times = []

print(f"\n{'트리 개수':>8} {'학습시간':>10} {'학습정확도':>10} {'테스트정확도':>12}")
print("-" * 45)

for n_est in estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)
    train_times.append(train_time)

    print(f"{n_est:>8} {train_time:>10.3f}s {train_score:>10.1%} {test_score:>12.1%}")

best_idx = np.argmax(test_scores)
print(f"\n최적 트리 개수: {estimators_range[best_idx]}")
print(f"최고 테스트 정확도: {test_scores[best_idx]:.1%}")


# ============================================================
# Part 6: 안정성 비교 실험
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 안정성 비교 실험 (10회 반복)")
print("=" * 60)

dt_scores_list = []
rf_scores_list = []

for i in range(10):
    # 데이터 랜덤 분할
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=i)

    # 의사결정나무
    dt = DecisionTreeClassifier(max_depth=10, random_state=i)
    dt.fit(X_tr, y_tr)
    dt_scores_list.append(dt.score(X_te, y_te))

    # 랜덤포레스트
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=i, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_scores_list.append(rf.score(X_te, y_te))

print(f"\n[10회 실험 결과]")
print(f"\n의사결정나무:")
print(f"  점수: {[f'{s:.1%}' for s in dt_scores_list]}")
print(f"  평균: {np.mean(dt_scores_list):.1%}")
print(f"  표준편차: {np.std(dt_scores_list):.3f}")

print(f"\n랜덤포레스트:")
print(f"  점수: {[f'{s:.1%}' for s in rf_scores_list]}")
print(f"  평균: {np.mean(rf_scores_list):.1%}")
print(f"  표준편차: {np.std(rf_scores_list):.3f}")

print(f"\n[안정성 비교]")
if np.std(rf_scores_list) < np.std(dt_scores_list):
    ratio = np.std(dt_scores_list)/np.std(rf_scores_list)
    print(f"  랜덤포레스트 표준편차가 {ratio:.1f}배 작음")
    print(f"  -> 랜덤포레스트가 더 안정적!")
else:
    print(f"  두 모델의 안정성이 비슷함")


# ============================================================
# Part 7: 특성 중요도 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 특성 중요도 비교")
print("=" * 60)

dt_importances = dt_model.feature_importances_
rf_importances = rf_model.feature_importances_

print(f"\n[특성 중요도]")
print(f"{'특성':>25} {'의사결정나무':>12} {'랜덤포레스트':>14}")
print("-" * 55)
for i, col in enumerate(feature_columns):
    print(f"{col:>25} {dt_importances[i]:>12.3f} {rf_importances[i]:>14.3f}")


# ============================================================
# Part 8: 개별 트리 확인
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 개별 트리 확인")
print("=" * 60)

print(f"\n[랜덤포레스트 개별 트리]")
print(f"트리 개수: {len(rf_model.estimators_)}")

# 처음 5개 트리 정보
print("\n처음 5개 트리 정보:")
for i in range(5):
    tree = rf_model.estimators_[i]
    print(f"  트리 {i+1}: 깊이={tree.get_depth()}, 리프 수={tree.get_n_leaves()}")

# 개별 트리 예측 비교
print("\n처음 3개 샘플에 대한 개별 트리 예측:")
sample_data = X_test.iloc[:3]
for i in range(3):
    print(f"\n  샘플 {i+1}:")
    tree_preds = [rf_model.estimators_[j].predict([sample_data.iloc[i]])[0]
                  for j in range(5)]
    ensemble_pred = rf_model.predict([sample_data.iloc[i]])[0]
    actual = y_test.iloc[i]
    print(f"    트리 1-5 예측: {tree_preds}")
    print(f"    앙상블 예측: {ensemble_pred} (실제: {actual})")


# ============================================================
# Part 9: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 모델 비교 (학습/테스트 정확도)
ax1 = axes[0, 0]
x_labels = ['의사결정나무', '랜덤포레스트']
train_accs = [dt_train_score, rf_train_score]
test_accs = [dt_test_score, rf_test_score]
x = np.arange(len(x_labels))
width = 0.35
ax1.bar(x - width/2, train_accs, width, label='학습', color='skyblue')
ax1.bar(x + width/2, test_accs, width, label='테스트', color='coral')
ax1.set_ylabel('정확도')
ax1.set_title('의사결정나무 vs 랜덤포레스트 (Breast Cancer)')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.legend()
ax1.set_ylim(0.85, 1.0)
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    ax1.annotate(f'{train:.1%}', (i - width/2, train + 0.005), ha='center', fontsize=9)
    ax1.annotate(f'{test:.1%}', (i + width/2, test + 0.005), ha='center', fontsize=9)

# 2. n_estimators별 성능
ax2 = axes[0, 1]
ax2.plot(estimators_range, train_scores, 'o-', label='학습', color='blue')
ax2.plot(estimators_range, test_scores, 'o-', label='테스트', color='orange')
ax2.axvline(x=estimators_range[best_idx], color='red', linestyle='--', label='최적')
ax2.set_xlabel('트리 개수 (n_estimators)')
ax2.set_ylabel('정확도')
ax2.set_title('트리 개수별 성능')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 안정성 비교 (박스플롯)
ax3 = axes[1, 0]
ax3.boxplot([dt_scores_list, rf_scores_list], labels=['의사결정나무', '랜덤포레스트'])
ax3.set_ylabel('테스트 정확도')
ax3.set_title('10회 반복 실험 결과 (안정성 비교)')
ax3.grid(True, alpha=0.3)

# 4. 특성 중요도 비교
ax4 = axes[1, 1]
x = np.arange(len(feature_columns))
width = 0.35
ax4.barh(x - width/2, dt_importances, width, label='의사결정나무', color='skyblue')
ax4.barh(x + width/2, rf_importances, width, label='랜덤포레스트', color='coral')
ax4.set_yticks(x)
ax4.set_yticklabels([col.replace('mean ', '') for col in feature_columns], fontsize=8)
ax4.set_xlabel('중요도')
ax4.set_title('특성 중요도 비교')
ax4.legend()

plt.tight_layout()
plt.savefig('14_random_forest_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 14_random_forest_comparison.png")


# ============================================================
# Part 10: 새 데이터 예측
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 새 데이터 예측")
print("=" * 60)

# 새로운 환자 데이터 (가상)
new_patients = pd.DataFrame({
    'mean radius': [14.0, 18.0, 12.0, 20.0, 11.0],
    'mean texture': [18.0, 22.0, 16.0, 25.0, 15.0],
    'mean perimeter': [90.0, 120.0, 78.0, 135.0, 72.0],
    'mean area': [600.0, 1000.0, 450.0, 1200.0, 380.0],
    'mean smoothness': [0.10, 0.12, 0.09, 0.13, 0.08],
    'mean compactness': [0.10, 0.15, 0.08, 0.18, 0.07],
    'mean concavity': [0.08, 0.12, 0.05, 0.15, 0.04],
    'mean concave points': [0.04, 0.08, 0.03, 0.10, 0.02],
    'mean symmetry': [0.18, 0.20, 0.17, 0.22, 0.16],
    'mean fractal dimension': [0.06, 0.07, 0.06, 0.08, 0.05]
})

print("\n[새 환자 데이터 (주요 특성)]")
print(new_patients[['mean radius', 'mean area', 'mean concavity']].to_string())

# 의사결정나무 예측
dt_predictions = dt_model.predict(new_patients)
dt_probabilities = dt_model.predict_proba(new_patients)

# 랜덤포레스트 예측
rf_predictions = rf_model.predict(new_patients)
rf_probabilities = rf_model.predict_proba(new_patients)

print("\n[진단 예측 결과]")
print(f"{'환자':>4} {'DT진단':>12} {'DT확신도':>10} {'RF진단':>12} {'RF확신도':>10}")
print("-" * 55)
for i in range(len(new_patients)):
    dt_diag = cancer.target_names[dt_predictions[i]]
    rf_diag = cancer.target_names[rf_predictions[i]]
    dt_conf = max(dt_probabilities[i]) * 100
    rf_conf = max(rf_probabilities[i]) * 100
    print(f"{i+1:>4} {dt_diag:>12} {dt_conf:>9.1f}% {rf_diag:>12} {rf_conf:>9.1f}%")


# ============================================================
# Part 11: 분류 보고서
# ============================================================

print("\n" + "=" * 60)
print("Part 11: 분류 보고서")
print("=" * 60)

print("\n[의사결정나무 분류 보고서]")
y_pred_dt = dt_model.predict(X_test)
print(classification_report(y_test, y_pred_dt, target_names=cancer.target_names))

print("\n[랜덤포레스트 분류 보고서]")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=cancer.target_names))


# ============================================================
# Part 12: 혼동 행렬 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 12: 혼동 행렬 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 의사결정나무
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt, display_labels=cancer.target_names).plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'의사결정나무 (정확도: {dt_test_score:.1%})')

# 랜덤포레스트
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf, display_labels=cancer.target_names).plot(ax=axes[1], cmap='Greens')
axes[1].set_title(f'랜덤포레스트 (정확도: {rf_test_score:.1%})')

plt.tight_layout()
plt.savefig('14_confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n혼동 행렬 저장: 14_confusion_matrix_comparison.png")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 14차시 랜덤포레스트")
print("=" * 60)

print("""
1. 앙상블 학습이란?
   - 여러 모델의 예측을 결합
   - 집단 지성의 원리
   - 오류가 서로 다르면 상쇄됨

2. Breast Cancer 데이터셋
   - 569개 유방암 진단 샘플
   - 30개 특성 (반경, 질감, 둘레, 면적 등)
   - 2진 분류: 악성(malignant) vs 양성(benign)

3. 배깅 (Bagging)
   - Bootstrap Aggregating
   - 복원 추출로 다양한 데이터셋 생성
   - 모델들을 병렬로 학습

4. 랜덤포레스트 원리
   - 의사결정나무 여러 개 결합
   - 두 가지 랜덤:
     1) 데이터 랜덤 (부트스트랩)
     2) 특성 랜덤 (max_features)
   - 예측: 투표(분류) 또는 평균(회귀)

5. 주요 파라미터
   - n_estimators: 트리 개수 (100~200)
   - max_depth: 트리 깊이 (None 또는 15~20)
   - max_features: 분할 시 특성 수 ('sqrt')
   - n_jobs: CPU 코어 수 (-1)
   - oob_score: OOB 점수 계산 (True)

6. 장단점
   장점: 높은 정확도, 안정성, 과대적합 저항
   단점: 느림, 메모리 사용, 해석 어려움

7. sklearn 사용법
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)
   model.predict(X_test)
   model.score(X_test, y_test)
   model.feature_importances_
""")

print("\n다음 차시 예고: 15차시 - 예측 모델: 선형/다항회귀")
