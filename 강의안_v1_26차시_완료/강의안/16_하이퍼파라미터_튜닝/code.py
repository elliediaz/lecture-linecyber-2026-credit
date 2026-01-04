"""
[16차시] 하이퍼파라미터 튜닝 - 실습 코드
학습목표: GridSearchCV, RandomizedSearchCV를 활용한 최적 하이퍼파라미터 탐색
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 데이터 준비
# ============================================================
print("=" * 50)
print("1. 데이터 준비")
print("=" * 50)

np.random.seed(42)
n_samples = 500

temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)
pressure = np.random.normal(1.0, 0.1, n_samples)

defect_prob = 0.1 + 0.025 * (temperature - 85) + 0.01 * (humidity - 50) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '압력': pressure,
    '불량여부': defect
})

X = df[['온도', '습도', '속도', '압력']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"불량 비율: {y.mean():.1%}")

# ============================================================
# 2. 하이퍼파라미터란?
# ============================================================
print("\n" + "=" * 50)
print("2. 하이퍼파라미터란?")
print("=" * 50)

print("""
▶ 하이퍼파라미터: 학습 전에 설정하는 값
   - n_estimators: 트리 개수
   - max_depth: 트리 최대 깊이
   - min_samples_split: 분할 최소 샘플 수

▶ 파라미터: 모델이 학습하는 값
   - 가중치 (weights)
   - 절편 (bias)

★ 하이퍼파라미터 값에 따라 성능이 크게 달라집니다!
""")

# ============================================================
# 3. 수동 튜닝 (비효율적인 방법)
# ============================================================
print("=" * 50)
print("3. 수동 튜닝 (비효율적인 방법)")
print("=" * 50)

print("▶ max_depth 값에 따른 성능:")
for depth in [3, 5, 7, 10, 15, 20]:
    model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"   max_depth={depth:2d}: {scores.mean():.3f} (±{scores.std():.3f})")

print("\n★ 이렇게 하나씩 하면 시간이 오래 걸림!")
print("   → GridSearchCV로 자동화!")

# ============================================================
# 4. GridSearchCV
# ============================================================
print("\n" + "=" * 50)
print("4. GridSearchCV")
print("=" * 50)

# 탐색할 파라미터 범위
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

total_combinations = 3 * 4 * 3
print(f"▶ 탐색할 파라미터:")
for key, values in param_grid.items():
    print(f"   {key}: {values}")
print(f"\n▶ 총 조합 수: {total_combinations}개")
print(f"▶ 5-Fold × {total_combinations} = {5 * total_combinations}번 학습")

# GridSearchCV 실행
print("\n▶ GridSearchCV 실행 중...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
print("   완료!")

# 결과 확인
print(f"\n▶ 최적 파라미터: {grid_search.best_params_}")
print(f"▶ 최고 교차검증 점수: {grid_search.best_score_:.4f}")

# 테스트 데이터 평가
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"▶ 테스트 점수: {test_score:.4f}")

# ============================================================
# 5. GridSearchCV 결과 상세 분석
# ============================================================
print("\n" + "=" * 50)
print("5. GridSearchCV 결과 분석")
print("=" * 50)

results = pd.DataFrame(grid_search.cv_results_)
results_summary = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_summary = results_summary.sort_values('rank_test_score').head(10)

print("▶ 상위 10개 조합:")
for idx, row in results_summary.iterrows():
    print(f"   순위 {row['rank_test_score']:2.0f}: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    print(f"           {row['params']}")

# ============================================================
# 6. RandomizedSearchCV
# ============================================================
print("\n" + "=" * 50)
print("6. RandomizedSearchCV")
print("=" * 50)

print("▶ GridSearchCV의 문제:")
print("   - 파라미터가 많으면 조합이 기하급수적으로 증가")
print("   - 예: 4개 파라미터 × 각 5개 값 = 625개 조합!")
print("\n▶ 해결: RandomizedSearchCV")
print("   - 랜덤하게 일부 조합만 시도")

# 파라미터 분포 정의
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

print("\n▶ RandomizedSearchCV 실행 (20개 조합)...")
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 20개 조합만 시도
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("   완료!")

print(f"\n▶ 최적 파라미터: {random_search.best_params_}")
print(f"▶ 최고 교차검증 점수: {random_search.best_score_:.4f}")
print(f"▶ 테스트 점수: {random_search.best_estimator_.score(X_test, y_test):.4f}")

# ============================================================
# 7. Grid vs Random 비교
# ============================================================
print("\n" + "=" * 50)
print("7. GridSearchCV vs RandomizedSearchCV")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────┐
│          GridSearchCV vs RandomizedSearchCV              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  GridSearchCV                                            │
│  - 모든 조합 시도 → 최적값 보장                          │
│  - 조합이 적을 때 사용 (파라미터 2~3개, 값 3~5개)        │
│  - 시간이 오래 걸림                                      │
│                                                          │
│  RandomizedSearchCV                                      │
│  - 랜덤 샘플링 → 근사 최적값                             │
│  - 조합이 많을 때 사용                                   │
│  - 빠름 (n_iter로 횟수 제한)                             │
│                                                          │
└─────────────────────────────────────────────────────────┘
""")

# ============================================================
# 8. 실전 튜닝 워크플로우
# ============================================================
print("=" * 50)
print("8. 실전 튜닝 워크플로우")
print("=" * 50)

print("""
▶ 효율적인 튜닝 전략:

1단계: 핵심 파라미터만 넓은 범위로
   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [5, 10, 15]
   }

2단계: 최적값 근처에서 세밀하게
   param_grid = {
       'n_estimators': [80, 100, 120],
       'max_depth': [8, 10, 12]
   }

3단계: 추가 파라미터 튜닝
   param_grid = {
       'n_estimators': [100],  # 고정
       'max_depth': [10],       # 고정
       'min_samples_split': [2, 5, 10]
   }
""")

# ============================================================
# 9. 최종 모델 평가
# ============================================================
print("=" * 50)
print("9. 최종 모델 평가")
print("=" * 50)

# 최적 모델로 예측
y_pred = best_model.predict(X_test)

print("▶ 최적 모델 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))

# 특성 중요도
print("▶ 특성 중요도:")
importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': best_model.feature_importances_
}).sort_values('중요도', ascending=False)
print(importance.to_string(index=False))

# ============================================================
# 10. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("10. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│              하이퍼파라미터 튜닝 핵심                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ GridSearchCV                                        │
│     param_grid = {'n_estimators': [50, 100, 200]}     │
│     grid = GridSearchCV(model, param_grid, cv=5)      │
│     grid.fit(X_train, y_train)                        │
│                                                        │
│  ▶ 결과 확인                                           │
│     grid.best_params_      → 최적 파라미터             │
│     grid.best_score_       → 최고 CV 점수             │
│     grid.best_estimator_   → 최적 모델                 │
│                                                        │
│  ▶ RandomizedSearchCV                                  │
│     from scipy.stats import randint                   │
│     param_dist = {'max_depth': randint(3, 20)}        │
│     random_search = RandomizedSearchCV(               │
│         model, param_dist, n_iter=20, cv=5)           │
│                                                        │
│  ★ 조합 적으면 Grid, 많으면 Random!                    │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 시계열 데이터 기초
""")
