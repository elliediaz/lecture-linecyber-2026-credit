"""
[24차시] 모델 해석과 변수별 영향력 분석 - 실습 코드
제조 AI 과정 | Part IV. AI 서비스화와 활용

학습목표:
1. 모델 해석의 필요성을 이해한다
2. 특성 중요도(Feature Importance)를 분석한다
3. Permutation Importance를 활용한다

실행: python code.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 데이터 준비
# ============================================================
print("=" * 60)
print("[24차시] 모델 해석과 변수별 영향력 분석")
print("제조 AI 과정 | Part IV. AI 서비스화와 활용")
print("=" * 60)
print()
print("=" * 50)
print("1. 제조 데이터 준비")
print("=" * 50)

# 제조 공정 데이터 생성
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n_samples),     # 온도
    'humidity': np.random.normal(50, 8, n_samples),        # 습도
    'speed': np.random.normal(100, 10, n_samples),         # 속도
    'pressure': np.random.normal(1.0, 0.1, n_samples)      # 압력
})

# 불량 여부 (온도와 습도가 주요 영향 요인)
defect_prob = (
    0.1 +
    0.3 * ((data['temperature'] - 85) / 10) +
    0.2 * ((data['humidity'] - 50) / 16) +
    0.1 * ((data['speed'] - 100) / 20) +
    0.05 * (np.random.randn(n_samples))
)
data['defect'] = (defect_prob > 0.3).astype(int)

print(f"데이터 크기: {len(data)}개")
print(f"불량률: {data['defect'].mean():.1%}")
print()
print("데이터 샘플:")
print(data.head())

# ============================================================
# 2. 모델 학습
# ============================================================
print("\n" + "=" * 50)
print("2. 랜덤포레스트 모델 학습")
print("=" * 50)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

feature_names = ['temperature', 'humidity', 'speed', 'pressure']
X = data[feature_names]
y = data['defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"학습 정확도: {train_acc:.3f}")
print(f"테스트 정확도: {test_acc:.3f}")

# ============================================================
# 3. Feature Importance (특성 중요도)
# ============================================================
print("\n" + "=" * 50)
print("3. Feature Importance")
print("=" * 50)

print("""
▶ Feature Importance란?
   - 각 특성이 예측에 얼마나 기여하는지 비율
   - 트리 기반 모델(RandomForest, DecisionTree)에서 제공
   - 학습 과정에서 자동으로 계산됨
""")

# 특성 중요도 확인
importance = model.feature_importances_

print("▶ 특성 중요도:")
for name, imp in zip(feature_names, importance):
    print(f"   {name}: {imp:.3f} ({imp*100:.1f}%)")

# 중요도 합계 확인
print(f"\n   합계: {importance.sum():.3f} (항상 1)")

# 정렬된 결과
importance_df = pd.DataFrame({
    '특성': feature_names,
    '중요도': importance
}).sort_values('중요도', ascending=False)

print("\n▶ 중요도 순위:")
for i, row in importance_df.iterrows():
    print(f"   {row['특성']}: {row['중요도']:.1%}")

# ============================================================
# 4. Feature Importance 시각화
# ============================================================
print("\n" + "=" * 50)
print("4. Feature Importance 시각화")
print("=" * 50)

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))

sorted_idx = np.argsort(importance)
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_idx)))

ax.barh(
    np.array(feature_names)[sorted_idx],
    importance[sorted_idx],
    color=colors
)
ax.set_xlabel('중요도')
ax.set_title('Feature Importance (랜덤포레스트)')

# 값 표시
for i, (idx, imp) in enumerate(zip(sorted_idx, importance[sorted_idx])):
    ax.text(imp + 0.01, i, f'{imp:.1%}', va='center')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("▶ 그래프 저장: feature_importance.png")

# ============================================================
# 5. Permutation Importance
# ============================================================
print("\n" + "=" * 50)
print("5. Permutation Importance")
print("=" * 50)

print("""
▶ Permutation Importance란?
   - 특정 특성의 값을 무작위로 섞음
   - 예측 성능이 얼마나 떨어지는지 측정
   - 많이 떨어지면 → 중요한 특성!
   - Feature Importance보다 신뢰성 높음
""")

from sklearn.inspection import permutation_importance

# Permutation Importance 계산
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

print("▶ Permutation Importance:")
for name, imp_mean, imp_std in zip(
    feature_names,
    result.importances_mean,
    result.importances_std
):
    print(f"   {name}: {imp_mean:.3f} (±{imp_std:.3f})")

# ============================================================
# 6. 두 방법 비교 시각화
# ============================================================
print("\n" + "=" * 50)
print("6. 두 방법 비교")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature Importance
sorted_idx = np.argsort(importance)
axes[0].barh(
    np.array(feature_names)[sorted_idx],
    importance[sorted_idx],
    color='steelblue'
)
axes[0].set_xlabel('중요도')
axes[0].set_title('Feature Importance\n(학습 중 계산)')

# Permutation Importance
perm_sorted_idx = np.argsort(result.importances_mean)
axes[1].barh(
    np.array(feature_names)[perm_sorted_idx],
    result.importances_mean[perm_sorted_idx],
    xerr=result.importances_std[perm_sorted_idx],
    color='coral',
    capsize=5
)
axes[1].set_xlabel('중요도 (성능 감소)')
axes[1].set_title('Permutation Importance\n(학습 후 계산)')

plt.tight_layout()
plt.savefig('importance_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("▶ 그래프 저장: importance_comparison.png")

# 비교 표
print("\n▶ 두 방법 비교:")
print("-" * 50)
print(f"{'특성':<15} {'Feature Imp':>15} {'Permutation Imp':>15}")
print("-" * 50)
for name, fi, pi in zip(
    feature_names,
    importance,
    result.importances_mean
):
    print(f"{name:<15} {fi:>15.3f} {pi:>15.3f}")

# ============================================================
# 7. 의사결정트리 특성 중요도
# ============================================================
print("\n" + "=" * 50)
print("7. 의사결정트리 특성 중요도")
print("=" * 50)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

print("▶ 의사결정트리 특성 중요도:")
for name, imp in zip(feature_names, dt_model.feature_importances_):
    print(f"   {name}: {imp:.3f}")

# ============================================================
# 8. 해석 주의사항
# ============================================================
print("\n" + "=" * 50)
print("8. 해석 시 주의사항")
print("=" * 50)

print("""
▶ 상관된 특성 문제
   - 온도와 습도가 상관관계 높으면 중요도 분산
   - 상관 분석 먼저 수행 권장

▶ 인과관계 ≠ 상관관계
   - 중요도 높음 ≠ 원인
   - 도메인 지식 필요

▶ 모델에 따른 차이
   - 같은 데이터라도 모델마다 중요도 다를 수 있음
   - 여러 모델 비교 권장
""")

# 상관관계 확인
print("▶ 특성 간 상관관계:")
corr = X.corr()
print(corr.round(3))

# ============================================================
# 9. 실무 보고서 형식
# ============================================================
print("\n" + "=" * 50)
print("9. 실무 보고서 형식")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────┐
│           품질 예측 모델 분석 보고서                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ▶ 모델 성능                                        │
│     - 테스트 정확도: {:.1f}%                         │
│                                                      │
│  ▶ 주요 영향 요인 (Feature Importance)              │
│     1. 온도 (temperature): {:.1f}%                  │
│     2. 습도 (humidity): {:.1f}%                     │
│     3. 속도 (speed): {:.1f}%                        │
│     4. 압력 (pressure): {:.1f}%                     │
│                                                      │
│  ▶ 해석                                             │
│     - 온도가 품질에 가장 큰 영향                    │
│     - 85°C 초과 시 불량률 급증                      │
│                                                      │
│  ▶ 권장사항                                         │
│     - 온도 모니터링 시스템 강화                     │
│     - 습도 제어 장치 도입 검토                      │
│                                                      │
└─────────────────────────────────────────────────────┘
""".format(
    test_acc * 100,
    importance[0] * 100,
    importance[1] * 100,
    importance[2] * 100,
    importance[3] * 100
))

# ============================================================
# 10. 핵심 요약
# ============================================================
print("=" * 50)
print("10. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│              특성 중요도 분석 핵심 정리               │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ Feature Importance                                 │
│     model.feature_importances_                        │
│     - 빠름, 학습 중 자동 계산                         │
│     - 트리 기반 모델에서 사용                          │
│                                                        │
│  ▶ Permutation Importance                             │
│     from sklearn.inspection import permutation_importance│
│     result = permutation_importance(model, X, y)      │
│     - 더 신뢰성 높음                                  │
│     - 학습 후 테스트 데이터로 계산                    │
│                                                        │
│  ▶ 해석 주의사항                                       │
│     - 상관된 특성은 중요도 분산                        │
│     - 인과관계 ≠ 상관관계                             │
│     - 도메인 지식과 함께 해석                          │
│                                                        │
│  ▶ 실무 활용                                          │
│     - 중요 요인 파악 → 개선 우선순위 결정             │
│     - 보고서: 숫자 + 비즈니스 인사이트                │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 모델 저장과 실무 배포 준비
""")
