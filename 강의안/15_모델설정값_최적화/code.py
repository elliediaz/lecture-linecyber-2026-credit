# [15차시] 모델 설정값 최적화 - 실습 코드

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

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("15차시: 모델 설정값 최적화")
print("최적의 하이퍼파라미터를 찾아봅니다!")
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

temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)

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
print(f"학습: {len(X_train)}개, 테스트: {len(X_test)}개")
print()


# ============================================================
# 실습 2: 수동 튜닝
# ============================================================
print("=" * 50)
print("실습 2: 수동 튜닝 (for 루프)")
print("=" * 50)

print("max_depth와 n_estimators 조합 시도:")
print("-" * 50)

manual_results = []
for depth in [3, 5, 10]:
    for n_est in [50, 100, 200]:
        model = RandomForestClassifier(
            max_depth=depth,
            n_estimators=n_est,
            random_state=42
        )
        scores = cross_val_score(model, X, y, cv=5)
        manual_results.append({
            'max_depth': depth,
            'n_estimators': n_est,
            '평균 점수': scores.mean()
        })
        print(f"depth={depth:2d}, n_est={n_est:3d}: {scores.mean():.3f} (±{scores.std():.3f})")

print("\n★ 코드가 복잡하고 관리하기 어려움!")
print("   → GridSearchCV로 자동화!")
print()


# ============================================================
# 실습 3: GridSearchCV
# ============================================================
print("=" * 50)
print("실습 3: GridSearchCV")
print("=" * 50)

# 탐색할 파라미터 정의
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV 생성
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,      # 모든 CPU 코어 사용
    verbose=1       # 진행 상황 출력
)

print("탐색할 조합 수:", 3 * 3 * 3, "개")
print("총 학습 횟수:", 3 * 3 * 3 * 5, "회 (5-Fold)")
print("\n탐색 시작...")

grid_search.fit(X_train, y_train)
print("탐색 완료!")
print()


# ============================================================
# 실습 4: 결과 확인
# ============================================================
print("=" * 50)
print("실습 4: 결과 확인")
print("=" * 50)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차검증 점수: {grid_search.best_score_:.3f}")

# 최적 모델
best_model = grid_search.best_estimator_

# 테스트 점수
test_score = best_model.score(X_test, y_test)
print(f"테스트 점수: {test_score:.3f}")
print()


# ============================================================
# 실습 5: 결과 상세 분석
# ============================================================
print("=" * 50)
print("실습 5: 결과 상세 분석 (cv_results_)")
print("=" * 50)

results = pd.DataFrame(grid_search.cv_results_)

# 주요 컬럼만 선택
cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
top_results = results[cols].sort_values('rank_test_score')

print("상위 5개 조합:")
print("-" * 70)
for idx, row in top_results.head(5).iterrows():
    print(f"순위 {row['rank_test_score']:2.0f}: 점수={row['mean_test_score']:.3f} "
          f"(±{row['std_test_score']:.3f})")
    print(f"         {row['params']}")
print()


# ============================================================
# 실습 6: RandomizedSearchCV
# ============================================================
print("=" * 50)
print("실습 6: RandomizedSearchCV")
print("=" * 50)

# 파라미터 분포 정의
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# RandomizedSearchCV 생성
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # 20개 조합만 시도
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("시도할 조합 수: 20개 (랜덤 선택)")
print("\n탐색 시작...")
random_search.fit(X_train, y_train)
print("탐색 완료!")

print(f"\n최적 파라미터: {random_search.best_params_}")
print(f"최고 교차검증 점수: {random_search.best_score_:.3f}")

# 테스트 점수
random_best_model = random_search.best_estimator_
random_test_score = random_best_model.score(X_test, y_test)
print(f"테스트 점수: {random_test_score:.3f}")
print()


# ============================================================
# 실습 7: Grid vs Random 비교
# ============================================================
print("=" * 50)
print("실습 7: GridSearchCV vs RandomizedSearchCV 비교")
print("=" * 50)

print("비교 결과:")
print("-" * 50)
print(f"{'방법':<25} {'최고 CV 점수':<15} {'테스트 점수':<15}")
print("-" * 50)
print(f"{'GridSearchCV':<25} {grid_search.best_score_:<15.3f} {test_score:<15.3f}")
print(f"{'RandomizedSearchCV':<25} {random_search.best_score_:<15.3f} {random_test_score:<15.3f}")
print("-" * 50)

print("\n★ GridSearchCV: 확실하지만 시간 오래 걸림")
print("★ RandomizedSearchCV: 빠르고 대체로 좋은 결과")
print()


# ============================================================
# 실습 8: 최종 모델 평가
# ============================================================
print("=" * 50)
print("실습 8: 최종 모델 평가")
print("=" * 50)

# GridSearchCV의 최적 모델 사용
y_pred = best_model.predict(X_test)

print("분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
print()


# ============================================================
# 실습 9: 튜닝 전후 비교
# ============================================================
print("=" * 50)
print("실습 9: 튜닝 전후 비교")
print("=" * 50)

# 기본 모델 (튜닝 전)
default_model = RandomForestClassifier(random_state=42)
default_scores = cross_val_score(default_model, X, y, cv=5)
default_model.fit(X_train, y_train)
default_test = default_model.score(X_test, y_test)

# 튜닝된 모델
tuned_scores = cross_val_score(best_model, X, y, cv=5)
tuned_test = best_model.score(X_test, y_test)

print("성능 비교:")
print("-" * 50)
print(f"{'모델':<20} {'CV 점수':<15} {'테스트 점수':<15}")
print("-" * 50)
print(f"{'기본 모델':<20} {default_scores.mean():<15.3f} {default_test:<15.3f}")
print(f"{'튜닝된 모델':<20} {tuned_scores.mean():<15.3f} {tuned_test:<15.3f}")
print("-" * 50)

improvement = (tuned_test - default_test) / default_test * 100
if improvement > 0:
    print(f"\n★ 튜닝으로 {improvement:.1f}% 성능 향상!")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              하이퍼파라미터 튜닝 핵심 정리              │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 하이퍼파라미터                                      │
│     학습 전에 설정하는 값                               │
│     n_estimators, max_depth, min_samples_split 등     │
│                                                        │
│  ▶ GridSearchCV                                        │
│     param_grid = {{'n_estimators': [50, 100]}}         │
│     grid_search = GridSearchCV(model, param_grid, cv=5)│
│     grid_search.fit(X_train, y_train)                  │
│     → 모든 조합 시도, 확실함                           │
│                                                        │
│  ▶ RandomizedSearchCV                                  │
│     n_iter=20 → 랜덤하게 20개 조합만 시도              │
│     → 빠르고 대체로 좋은 결과                          │
│                                                        │
│  ▶ 결과 확인                                           │
│     grid_search.best_params_   : 최적 파라미터         │
│     grid_search.best_score_    : 최고 CV 점수          │
│     grid_search.best_estimator_: 최적 모델             │
│                                                        │
│  ★ 조합 적으면 Grid, 많으면 Random!                    │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 시계열 데이터 기초
""")

print("=" * 60)
print("15차시 실습 완료!")
print("모델 성능을 최대로 끌어올렸습니다!")
print("=" * 60)
