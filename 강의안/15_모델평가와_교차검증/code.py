"""
[15차시] 모델 평가와 교차검증 - 실습 코드
학습목표: 교차검증, 과대적합/과소적합 진단, 혼동행렬과 평가 지표 이해
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 제조 데이터 생성
# ============================================================
print("=" * 50)
print("1. 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 500

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)

# 불량 여부 (불균형 데이터 - 불량 15%)
defect_prob = 0.1 + 0.02 * (temperature - 85)
defect = (np.random.random(n_samples) < defect_prob).astype(int)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '불량여부': defect
})

print(df.head())
print(f"\n불량 비율: {df['불량여부'].mean():.1%}")

# 데이터 준비
X = df[['온도', '습도', '속도']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 2. 기존 방법 vs 교차검증
# ============================================================
print("\n" + "=" * 50)
print("2. 기존 방법 vs 교차검증")
print("=" * 50)

# 기존 방법: 한 번만 평가
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
single_score = model.score(X_test, y_test)

print(f"▶ 기존 방법 (train_test_split)")
print(f"   테스트 정확도: {single_score:.3f}")
print(f"   → 이 값이 진짜일까요? 운이 좋았던 걸까요?")

# 교차검증
print(f"\n▶ 교차검증 (5-Fold)")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"   각 Fold 점수: {cv_scores.round(3)}")
print(f"   평균: {cv_scores.mean():.3f}")
print(f"   표준편차: {cv_scores.std():.3f}")
print(f"   → 더 신뢰할 수 있는 평가!")

# ============================================================
# 3. K-Fold 상세
# ============================================================
print("\n" + "=" * 50)
print("3. K-Fold 교차검증 상세")
print("=" * 50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("각 Fold의 학습/테스트 데이터 크기:")
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"   Fold {i+1}: 학습 {len(train_idx)}개, 테스트 {len(test_idx)}개")

# ============================================================
# 4. 과대적합 vs 과소적합 진단
# ============================================================
print("\n" + "=" * 50)
print("4. 과대적합 vs 과소적합 진단")
print("=" * 50)

# 다양한 max_depth로 실험
depths = [1, 3, 5, 10, 20, None]
results = []

for depth in depths:
    model = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    diff = train_score - test_score

    depth_str = str(depth) if depth else '무제한'

    # 진단
    if train_score < 0.75 and test_score < 0.75:
        status = "과소적합"
    elif diff > 0.1:
        status = "과대적합"
    else:
        status = "적절"

    results.append({
        'max_depth': depth_str,
        '학습': f'{train_score:.3f}',
        '테스트': f'{test_score:.3f}',
        '차이': f'{diff:.3f}',
        '진단': status
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n★ 학습-테스트 차이가 크면 과대적합!")

# ============================================================
# 5. 혼동행렬 (Confusion Matrix)
# ============================================================
print("\n" + "=" * 50)
print("5. 혼동행렬 (Confusion Matrix)")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 혼동행렬 계산
cm = confusion_matrix(y_test, y_pred)
print("혼동행렬:")
print(cm)

# 각 요소 설명
tn, fp, fn, tp = cm.ravel()
print(f"\n▶ 해석:")
print(f"   TN (True Negative): {tn}개 - 정상을 정상으로 예측 ✅")
print(f"   FP (False Positive): {fp}개 - 정상을 불량으로 예측 ❌")
print(f"   FN (False Negative): {fn}개 - 불량을 정상으로 예측 ❌")
print(f"   TP (True Positive): {tp}개 - 불량을 불량으로 예측 ✅")

# 시각화
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Defect'])
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n▶ confusion_matrix.png 저장됨")

# ============================================================
# 6. 정밀도, 재현율, F1 Score
# ============================================================
print("\n" + "=" * 50)
print("6. 정밀도, 재현율, F1 Score")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"정확도 (Accuracy): {accuracy:.3f}")
print(f"정밀도 (Precision): {precision:.3f}")
print(f"재현율 (Recall): {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# 수식으로 확인
print("\n▶ 수식 확인:")
print(f"   정밀도 = TP/(TP+FP) = {tp}/({tp}+{fp}) = {tp/(tp+fp):.3f}")
print(f"   재현율 = TP/(TP+FN) = {tp}/({tp}+{fn}) = {tp/(tp+fn):.3f}")

# ============================================================
# 7. Classification Report
# ============================================================
print("\n" + "=" * 50)
print("7. 분류 리포트 (Classification Report)")
print("=" * 50)

print(classification_report(y_test, y_pred, target_names=['정상', '불량']))

# ============================================================
# 8. 상황별 지표 선택
# ============================================================
print("\n" + "=" * 50)
print("8. 상황별 중요 지표")
print("=" * 50)

print("""
┌────────────────────────────────────────────────────────┐
│               상황별 중요 지표 선택                      │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ▶ 정밀도(Precision)가 중요한 경우                       │
│     - 스팸 메일 필터 (정상 메일을 스팸으로 분류하면 안됨)  │
│     - 추천 시스템 (잘못된 추천은 신뢰도 하락)            │
│                                                         │
│  ▶ 재현율(Recall)이 중요한 경우                          │
│     - 암 진단 (암 환자를 놓치면 큰 위험)                 │
│     - 제조 불량 검출 (불량품이 출하되면 안됨)            │
│     - 보안 위협 탐지 (공격을 놓치면 안됨)               │
│                                                         │
│  ▶ F1 Score: 둘 다 균형있게 중요할 때                    │
│                                                         │
└────────────────────────────────────────────────────────┘
""")

# ============================================================
# 9. 교차검증으로 여러 지표 계산
# ============================================================
print("=" * 50)
print("9. 교차검증으로 여러 지표 계산")
print("=" * 50)

from sklearn.model_selection import cross_validate

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 여러 지표를 한 번에 계산
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("5-Fold 교차검증 결과 (평균 ± 표준편차):")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"   {metric:<10}: {scores.mean():.3f} (±{scores.std():.3f})")

# ============================================================
# 10. 모델 비교 실습
# ============================================================
print("\n" + "=" * 50)
print("10. 모델 비교 (교차검증 기반)")
print("=" * 50)

models = {
    '의사결정트리': DecisionTreeClassifier(max_depth=5, random_state=42),
    '랜덤포레스트(50)': RandomForestClassifier(n_estimators=50, random_state=42),
    '랜덤포레스트(100)': RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f"{'모델':<20} {'정확도':<15} {'F1 Score':<15}")
print("-" * 50)

for name, model in models.items():
    acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"{name:<20} {acc_scores.mean():.3f} (±{acc_scores.std():.3f})  {f1_scores.mean():.3f} (±{f1_scores.std():.3f})")

# ============================================================
# 11. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("11. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                 모델 평가 핵심 정리                     │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 교차검증                                            │
│     scores = cross_val_score(model, X, y, cv=5)       │
│     print(f"평균: {scores.mean():.3f}")               │
│                                                        │
│  ▶ 과대적합/과소적합 진단                               │
│     - 학습/테스트 점수 비교                             │
│     - 차이가 크면 과대적합                              │
│     - 둘 다 낮으면 과소적합                             │
│                                                        │
│  ▶ 분류 평가 지표                                      │
│     - 혼동행렬: confusion_matrix(y_test, y_pred)       │
│     - 정밀도: precision_score(y_test, y_pred)         │
│     - 재현율: recall_score(y_test, y_pred)            │
│     - F1: f1_score(y_test, y_pred)                    │
│     - 종합: classification_report(y_test, y_pred)     │
│                                                        │
│  ★ 도메인에 따라 중요한 지표 선택!                      │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 하이퍼파라미터 튜닝 (GridSearchCV)
""")
