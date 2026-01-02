# [14차시] 모델 평가와 반복 검증 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    classification_report, accuracy_score
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("14차시: 모델 평가와 반복 검증")
print("모델을 제대로 평가하는 방법을 배웁니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 제조 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 500

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)

# 불량 여부
defect_prob = 0.05 + 0.03 * (temperature - 80) / 5 + 0.02 * (humidity - 40) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '불량여부': defect
})

print(f"데이터 크기: {df.shape}")
print(f"불량 비율: {df['불량여부'].mean():.1%}")

X = df[['온도', '습도', '속도']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
print()


# ============================================================
# 실습 2: 기존 방법 (train_test_split)
# ============================================================
print("=" * 50)
print("실습 2: 기존 평가 방법")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"한 번의 평가 결과: {score:.1%}")
print("\n⚠️ 이 점수가 진짜인지, 운이 좋았던 건지 알기 어려움!")
print()


# ============================================================
# 실습 3: 교차검증
# ============================================================
print("=" * 50)
print("실습 3: 교차검증 (5-Fold)")
print("=" * 50)

# 5-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5)

print("▶ cross_val_score(model, X, y, cv=5)")
print(f"\n각 Fold 점수: {scores}")
print(f"평균: {scores.mean():.3f}")
print(f"표준편차: {scores.std():.3f}")

print(f"\n★ 결론: {scores.mean():.1%} (±{scores.std()*100:.1f}%p)")
print("   → 표준편차가 작으면 안정적인 모델")
print()


# ============================================================
# 실습 4: 다양한 cv 값
# ============================================================
print("=" * 50)
print("실습 4: cv 값에 따른 차이")
print("=" * 50)

for cv in [3, 5, 10]:
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"cv={cv:2d}: 평균={scores.mean():.3f} (±{scores.std():.3f})")

print("\n★ 보통 cv=5 또는 cv=10을 사용")
print()


# ============================================================
# 실습 5: 과대적합 진단
# ============================================================
print("=" * 50)
print("실습 5: 과대적합 진단")
print("=" * 50)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"학습 정확도: {train_score:.1%}")
print(f"테스트 정확도: {test_score:.1%}")
print(f"차이: {(train_score - test_score):.1%}")

diff = train_score - test_score
if diff > 0.1:
    print("\n⚠️ 과대적합 의심! (차이 > 10%p)")
    print("   → max_depth를 줄이거나 데이터를 추가하세요")
elif train_score < 0.7 and test_score < 0.7:
    print("\n⚠️ 과소적합 의심! (둘 다 낮음)")
    print("   → 더 복잡한 모델이나 특성 추가가 필요합니다")
else:
    print("\n✅ 적절한 일반화!")
print()


# ============================================================
# 실습 6: max_depth에 따른 과대적합
# ============================================================
print("=" * 50)
print("실습 6: max_depth에 따른 과대적합")
print("=" * 50)

depths = [1, 3, 5, 10, None]
results = []

for depth in depths:
    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_model.fit(X_train, y_train)

    train_acc = temp_model.score(X_train, y_train)
    test_acc = temp_model.score(X_test, y_test)

    depth_str = str(depth) if depth else '무제한'
    results.append({
        'max_depth': depth_str,
        '학습': f'{train_acc:.1%}',
        '테스트': f'{test_acc:.1%}',
        '차이': f'{(train_acc - test_acc):.1%}'
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\n★ max_depth가 클수록 과대적합 위험 증가")
print()


# ============================================================
# 실습 7: 혼동행렬
# ============================================================
print("=" * 50)
print("실습 7: 혼동행렬")
print("=" * 50)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("혼동행렬:")
print(cm)
print(f"\nTN(정상→정상): {cm[0,0]}")
print(f"FP(정상→불량): {cm[0,1]}")
print(f"FN(불량→정상): {cm[1,0]}  ← 놓친 불량!")
print(f"TP(불량→불량): {cm[1,1]}")

# 시각화
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=['정상', '불량'])
disp.plot(cmap='Blues', ax=ax)
plt.title('혼동행렬')
plt.tight_layout()
plt.show()
print()


# ============================================================
# 실습 8: 정밀도, 재현율, F1 Score
# ============================================================
print("=" * 50)
print("실습 8: 정밀도, 재현율, F1 Score")
print("=" * 50)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"정확도: {accuracy:.3f}")
print(f"정밀도: {precision:.3f}  (불량 예측 중 진짜 불량)")
print(f"재현율: {recall:.3f}  (실제 불량 중 잡아낸 것)")
print(f"F1 Score: {f1:.3f}  (정밀도와 재현율의 조화평균)")

print("\n★ 제조 현장에서는 재현율이 중요!")
print("   → 불량품을 놓치면 고객에게 전달됨")
print()


# ============================================================
# 실습 9: 분류 리포트
# ============================================================
print("=" * 50)
print("실습 9: 분류 리포트")
print("=" * 50)

print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
print()


# ============================================================
# 실습 10: 여러 모델 비교
# ============================================================
print("=" * 50)
print("실습 10: 여러 모델 교차검증 비교")
print("=" * 50)

models = {
    '의사결정나무 (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    '의사결정나무 (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    '랜덤포레스트 (100)': RandomForestClassifier(n_estimators=100, random_state=42),
}

print("모델 비교 (5-Fold 교차검증):")
print("-" * 60)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<25}: {scores.mean():.3f} (±{scores.std():.3f})")
print("-" * 60)
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│               모델 평가와 반복 검증 핵심 정리            │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 교차검증                                            │
│     scores = cross_val_score(model, X, y, cv=5)       │
│     → 여러 번 평가해서 평균과 표준편차 확인             │
│                                                        │
│  ▶ 과대적합 진단                                       │
│     학습 높고 테스트 낮음 → 모델 단순화                 │
│     둘 다 낮음 → 모델 복잡화                           │
│                                                        │
│  ▶ 혼동행렬                                            │
│     TN: 정상→정상, FP: 정상→불량                       │
│     FN: 불량→정상 (놓침!), TP: 불량→불량               │
│                                                        │
│  ▶ 평가 지표                                           │
│     정밀도: 불량 예측 중 진짜 불량                      │
│     재현율: 실제 불량 중 잡아낸 것 (제조에서 중요!)      │
│     F1 Score: 정밀도와 재현율의 조화평균               │
│                                                        │
│  ★ 모델 평가는 한 번이 아니라 여러 번!                  │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 모델 설정값 최적화 (GridSearchCV)
""")

print("=" * 60)
print("14차시 실습 완료!")
print("신뢰할 수 있는 평가 방법을 배웠습니다!")
print("=" * 60)
