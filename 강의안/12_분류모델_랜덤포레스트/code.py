# [12차시] 분류 모델 (2): 랜덤포레스트 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("12차시: 분류 모델 (2) - 랜덤포레스트")
print("여러 트리가 모여 숲을 이룹니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 제조 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 500  # 랜덤포레스트는 데이터가 많을수록 좋음

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)  # 온도
humidity = np.random.normal(50, 10, n_samples)    # 습도
speed = np.random.normal(100, 15, n_samples)      # 속도

# 불량 여부 (온도와 습도가 높으면 불량 확률 증가)
defect_prob = 0.05 + 0.03 * (temperature - 80) / 5 + 0.02 * (humidity - 40) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '불량여부': defect
})

print("데이터 샘플:")
print(df.head(10))
print(f"\n데이터 크기: {df.shape}")
print(f"불량 비율: {df['불량여부'].mean():.1%}")
print()


# ============================================================
# 실습 2: 데이터 준비
# ============================================================
print("=" * 50)
print("실습 2: 데이터 준비")
print("=" * 50)

# 특성(X)과 타겟(y) 분리
X = df[['온도', '습도', '속도']]
y = df['불량여부']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"\n학습 데이터 불량 비율: {y_train.mean():.1%}")
print(f"테스트 데이터 불량 비율: {y_test.mean():.1%}")
print()


# ============================================================
# 실습 3: 랜덤포레스트 학습
# ============================================================
print("=" * 50)
print("실습 3: 랜덤포레스트 학습")
print("=" * 50)

# 랜덤포레스트 모델 생성
model = RandomForestClassifier(
    n_estimators=100,   # 트리 100개
    random_state=42
)

# 학습
model.fit(X_train, y_train)
print("▶ model.fit(X_train, y_train) - 학습 완료!")

# 모델 정보
print(f"\n트리 개수: {model.n_estimators}")
print(f"특성 개수: {model.n_features_in_}")
print()


# ============================================================
# 실습 4: 의사결정나무 vs 랜덤포레스트 성능 비교
# ============================================================
print("=" * 50)
print("실습 4: 의사결정나무 vs 랜덤포레스트 비교")
print("=" * 50)

# 의사결정나무
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_train_acc = dt.score(X_train, y_train)
dt_test_acc = dt.score(X_test, y_test)

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

print("성능 비교:")
print("-" * 50)
print(f"{'모델':<20} {'학습 정확도':<15} {'테스트 정확도':<15}")
print("-" * 50)
print(f"{'의사결정나무':<20} {dt_train_acc:<15.1%} {dt_test_acc:<15.1%}")
print(f"{'랜덤포레스트':<20} {rf_train_acc:<15.1%} {rf_test_acc:<15.1%}")
print("-" * 50)

if rf_test_acc > dt_test_acc:
    print("\n★ 랜덤포레스트가 더 높은 테스트 정확도!")
print()


# ============================================================
# 실습 5: 특성 중요도 분석
# ============================================================
print("=" * 50)
print("실습 5: 특성 중요도 분석")
print("=" * 50)

importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print("특성 중요도:")
print(importance.to_string(index=False))

# 중요도 시각화
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3B82F6', '#10B981', '#F59E0B']
bars = ax.barh(importance['특성'], importance['중요도'], color=colors)
ax.set_xlabel('중요도')
ax.set_title('랜덤포레스트 특성 중요도')
ax.invert_yaxis()

for bar, val in zip(bars, importance['중요도']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center')

plt.tight_layout()
plt.show()
print()


# ============================================================
# 실습 6: n_estimators에 따른 성능 변화
# ============================================================
print("=" * 50)
print("실습 6: n_estimators에 따른 성능 변화")
print("=" * 50)

n_trees_list = [10, 50, 100, 200, 500]
results = []

for n_trees in n_trees_list:
    model_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model_temp.fit(X_train, y_train)
    train_acc = model_temp.score(X_train, y_train)
    test_acc = model_temp.score(X_test, y_test)
    results.append({
        '트리 개수': n_trees,
        '학습 정확도': f'{train_acc:.1%}',
        '테스트 정확도': f'{test_acc:.1%}'
    })
    print(f"트리 {n_trees:>3}개: 학습={train_acc:.1%}, 테스트={test_acc:.1%}")

print("\n★ 100개 이상은 성능 향상이 미미함 (보통 100개면 충분)")
print()


# ============================================================
# 실습 7: OOB 점수 확인
# ============================================================
print("=" * 50)
print("실습 7: OOB (Out-of-Bag) 점수")
print("=" * 50)

model_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,    # OOB 점수 활성화
    random_state=42
)
model_oob.fit(X_train, y_train)

print(f"OOB 점수: {model_oob.oob_score_:.1%}")
print(f"테스트 점수: {model_oob.score(X_test, y_test):.1%}")
print("\n★ OOB = Bootstrap에서 제외된 데이터로 평가")
print("  → 별도의 검증 세트 없이도 성능 추정 가능")
print()


# ============================================================
# 실습 8: 예측 확률
# ============================================================
print("=" * 50)
print("실습 8: 예측 확률")
print("=" * 50)

# 새 데이터 예측
test_samples = [
    [80, 45, 100],   # 정상적인 조건
    [90, 55, 100],   # 온도 높음
    [92, 65, 90],    # 고온, 고습
    [95, 70, 85],    # 극단적 조건
]

print("예측 결과:")
print("-" * 70)
for sample in test_samples:
    pred = model.predict([sample])
    proba = model.predict_proba([sample])
    result = '불량' if pred[0] == 1 else '정상'
    print(f"온도={sample[0]:5.1f}, 습도={sample[1]:5.1f}, 속도={sample[2]:5.1f}")
    print(f"   → {result} (정상: {proba[0][0]:.1%}, 불량: {proba[0][1]:.1%})")
print()


# ============================================================
# 실습 9: 분류 리포트
# ============================================================
print("=" * 50)
print("실습 9: 분류 리포트")
print("=" * 50)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
print()


# ============================================================
# 실습 10: 의사결정나무 vs 랜덤포레스트 안정성 비교
# ============================================================
print("=" * 50)
print("실습 10: 안정성 비교 (여러 번 학습)")
print("=" * 50)

dt_scores = []
rf_scores = []

for seed in range(10):
    # 데이터 다시 분리 (다른 random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 의사결정나무
    dt_temp = DecisionTreeClassifier(random_state=42)
    dt_temp.fit(X_tr, y_tr)
    dt_scores.append(dt_temp.score(X_te, y_te))

    # 랜덤포레스트
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_tr, y_tr)
    rf_scores.append(rf_temp.score(X_te, y_te))

print("10번 반복 학습 결과:")
print("-" * 50)
print(f"{'모델':<20} {'평균':<15} {'표준편차':<15}")
print("-" * 50)
print(f"{'의사결정나무':<20} {np.mean(dt_scores):<15.1%} {np.std(dt_scores):<15.3f}")
print(f"{'랜덤포레스트':<20} {np.mean(rf_scores):<15.1%} {np.std(rf_scores):<15.3f}")
print("-" * 50)
print("\n★ 랜덤포레스트가 더 안정적 (표준편차가 작음)")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                랜덤포레스트 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 앙상블 학습                                         │
│     여러 모델을 합쳐서 더 좋은 성능                      │
│                                                        │
│  ▶ 랜덤포레스트 = 여러 의사결정나무의 투표               │
│     RandomForestClassifier(n_estimators=100)          │
│                                                        │
│  ▶ 두 가지 랜덤                                        │
│     1. 데이터 랜덤 (Bootstrap 샘플링)                   │
│     2. 특성 랜덤 (노드별 일부 특성만 사용)               │
│                                                        │
│  ▶ 핵심 파라미터                                       │
│     n_estimators: 트리 개수 (기본 100, 보통 충분)       │
│     max_depth: 각 트리의 최대 깊이                     │
│                                                        │
│  ▶ OOB 점수                                           │
│     oob_score=True로 별도 검증 없이 성능 추정           │
│                                                        │
│  ★ 장점: 높은 성능, 안정적, 특성 중요도 제공            │
│  ★ 단점: 해석 어려움, 학습 시간 증가                    │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 예측 모델 - 선형회귀와 다항회귀
""")

print("=" * 60)
print("12차시 실습 완료!")
print("앙상블의 힘을 경험했습니다!")
print("=" * 60)
