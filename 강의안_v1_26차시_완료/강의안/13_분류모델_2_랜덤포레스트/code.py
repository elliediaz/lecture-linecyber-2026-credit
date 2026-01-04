"""
[13차시] 분류 모델 (2): 랜덤포레스트 - 실습 코드
학습목표: 앙상블 학습 이해, RandomForestClassifier 실습, 특성 중요도 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 한글 폰트 설정
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
pressure = np.random.normal(1.0, 0.1, n_samples)  # 추가 특성

# 불량 여부
defect_prob = 0.1 + 0.025 * (temperature - 85) + 0.015 * (humidity - 50) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '압력': pressure,
    '불량여부': defect
})

print(df.head())
print(f"\n데이터 크기: {df.shape}")
print(f"불량 비율: {df['불량여부'].mean():.1%}")

# ============================================================
# 2. 데이터 준비
# ============================================================
print("\n" + "=" * 50)
print("2. 데이터 준비")
print("=" * 50)

X = df[['온도', '습도', '속도', '압력']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# ============================================================
# 3. 의사결정트리 vs 랜덤포레스트 비교
# ============================================================
print("\n" + "=" * 50)
print("3. 의사결정트리 vs 랜덤포레스트")
print("=" * 50)

# 의사결정트리
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 랜덤포레스트
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_acc = rf_model.score(X_train, y_train)
rf_test_acc = rf_model.score(X_test, y_test)

print(f"\n{'모델':<15} {'학습 정확도':<12} {'테스트 정확도':<12}")
print("-" * 40)
print(f"{'의사결정트리':<15} {dt_train_acc:<12.1%} {dt_test_acc:<12.1%}")
print(f"{'랜덤포레스트':<15} {rf_train_acc:<12.1%} {rf_test_acc:<12.1%}")

print("\n★ 랜덤포레스트가 더 안정적인 성능을 보여줍니다!")

# ============================================================
# 4. 랜덤포레스트 상세 실습
# ============================================================
print("\n" + "=" * 50)
print("4. 랜덤포레스트 상세 학습")
print("=" * 50)

# 모델 생성
model = RandomForestClassifier(
    n_estimators=100,     # 트리 100개
    max_depth=10,         # 각 트리 최대 깊이
    min_samples_split=5,  # 분할 최소 샘플
    random_state=42,
    n_jobs=-1             # 병렬 처리
)

# 학습
model.fit(X_train, y_train)
print("▶ 모델 학습 완료!")
print(f"   - 트리 개수: {len(model.estimators_)}")
print(f"   - 특성 개수: {model.n_features_in_}")

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"\n▶ 예측 완료!")
print(f"   첫 5개 예측: {list(y_pred[:5])}")
print(f"   첫 5개 확률 (불량): {[f'{p[1]:.1%}' for p in y_proba[:5]]}")

# ============================================================
# 5. 모델 평가
# ============================================================
print("\n" + "=" * 50)
print("5. 모델 평가")
print("=" * 50)

accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.1%}")

print("\n▶ 분류 리포트")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))

# ============================================================
# 6. 특성 중요도
# ============================================================
print("\n" + "=" * 50)
print("6. 특성 중요도 (Feature Importance)")
print("=" * 50)

importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print(importance.to_string(index=False))

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']
bars = ax.barh(importance['특성'], importance['중요도'], color=colors)
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance')
ax.invert_yaxis()

for bar, val in zip(bars, importance['중요도']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center')

plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n▶ rf_feature_importance.png 저장됨")

# ============================================================
# 7. n_estimators 효과 분석
# ============================================================
print("\n" + "=" * 50)
print("7. 트리 개수(n_estimators)에 따른 성능")
print("=" * 50)

n_trees = [10, 50, 100, 200, 300, 500]
results = []

for n in n_trees:
    temp_model = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
    temp_model.fit(X_train, y_train)
    train_acc = temp_model.score(X_train, y_train)
    test_acc = temp_model.score(X_test, y_test)
    results.append({
        'n_estimators': n,
        '학습 정확도': f'{train_acc:.1%}',
        '테스트 정확도': f'{test_acc:.1%}'
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\n★ 100개 이상에서는 성능 향상이 미미합니다!")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
test_accs = [float(r['테스트 정확도'].strip('%'))/100 for r in results]
ax.plot(n_trees, test_accs, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Trees (n_estimators)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Effect of n_estimators on Performance')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('n_estimators_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ n_estimators_effect.png 저장됨")

# ============================================================
# 8. OOB(Out-of-Bag) 점수
# ============================================================
print("\n" + "=" * 50)
print("8. OOB(Out-of-Bag) 점수")
print("=" * 50)

oob_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    oob_score=True,       # OOB 점수 활성화
    random_state=42
)
oob_model.fit(X_train, y_train)

print(f"OOB 점수: {oob_model.oob_score_:.1%}")
print(f"테스트 정확도: {oob_model.score(X_test, y_test):.1%}")
print("\n★ OOB 점수는 교차검증 없이도 성능을 추정할 수 있습니다!")

# ============================================================
# 9. 개별 트리 확인
# ============================================================
print("\n" + "=" * 50)
print("9. 개별 트리 확인")
print("=" * 50)

print(f"총 트리 개수: {len(model.estimators_)}")
print("\n처음 5개 트리의 개별 예측 (첫 번째 테스트 샘플):")

sample = X_test.iloc[[0]]
individual_preds = []

for i, tree in enumerate(model.estimators_[:5]):
    pred = tree.predict(sample)[0]
    individual_preds.append(pred)
    label = '불량' if pred == 1 else '정상'
    print(f"  트리 {i+1}: {label}")

print(f"\n최종 예측 (다수결): {'불량' if model.predict(sample)[0] == 1 else '정상'}")

# ============================================================
# 10. 의사결정트리 vs 랜덤포레스트 정리
# ============================================================
print("\n" + "=" * 50)
print("10. 의사결정트리 vs 랜덤포레스트 정리")
print("=" * 50)

print("""
┌────────────────────────────────────────────────────────┐
│            의사결정트리 vs 랜덤포레스트                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ▶ 의사결정트리 (DecisionTreeClassifier)                │
│     - 장점: 해석 가능, 빠른 학습                         │
│     - 단점: 불안정, 과대적합 위험                        │
│     - 용도: 모델 설명이 필요할 때                        │
│                                                         │
│  ▶ 랜덤포레스트 (RandomForestClassifier)                │
│     - 장점: 높은 성능, 안정적, 과대적합 방지             │
│     - 단점: 해석 어려움, 메모리 사용                     │
│     - 용도: 실무 대부분의 경우                           │
│                                                         │
├────────────────────────────────────────────────────────┤
│  핵심 코드:                                              │
│  model = RandomForestClassifier(n_estimators=100)       │
│  model.fit(X_train, y_train)                            │
│  model.predict(X_test)                                  │
│  model.feature_importances_  # 특성 중요도              │
│                                                         │
└────────────────────────────────────────────────────────┘

다음 차시: 회귀 모델 (선형회귀, 다항회귀)
""")
