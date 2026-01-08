"""
[17차시] 모델 설정값 최적화 - 실습 코드
=============================================

학습 목표:
1. 하이퍼파라미터의 개념을 이해한다
2. GridSearchCV로 최적값을 탐색한다
3. RandomizedSearchCV로 효율적 탐색을 수행한다

실습 내용:
- Digits 데이터셋을 활용한 손글씨 숫자 분류
- 하이퍼파라미터 영향 실험
- GridSearchCV 기본 사용
- RandomizedSearchCV 기본 사용
- Grid vs Random 비교
- 2단계 탐색 전략
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.datasets import load_digits
from scipy.stats import randint, uniform

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 데이터 로딩
# ============================================================

print("=" * 60)
print("Part 1: Digits 데이터셋 로딩")
print("=" * 60)

# Digits 데이터셋 로딩
print("\n[Digits 데이터셋 로딩 중...]")
try:
    digits = load_digits()
    X = digits.data
    y = digits.target
    print("Digits 데이터셋 로딩 완료!")
except Exception as e:
    print(f"데이터셋 로딩 실패: {e}")
    raise

print(f"\n[데이터 확인]")
print(f"데이터 크기: {X.shape}")
print(f"특성 개수: {X.shape[1]} (8x8 픽셀)")
print(f"클래스: 0-9 숫자 ({len(np.unique(y))}개)")

print(f"\n클래스별 샘플 수:")
for digit in range(10):
    count = (y == digit).sum()
    print(f"  숫자 {digit}: {count}개")

# 데이터 시각화 (샘플)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'숫자: {digits.target[i]}')
    ax.axis('off')
plt.suptitle('Digits 데이터셋 샘플')
plt.tight_layout()
plt.savefig('17_digits_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n샘플 이미지 저장: 17_digits_samples.png")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")


# ============================================================
# Part 2: 하이퍼파라미터의 영향
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 하이퍼파라미터의 영향")
print("=" * 60)

print("\n[max_depth에 따른 성능 변화]")
print(f"{'max_depth':>10} {'학습 점수':>10} {'테스트 점수':>12} {'상태':>10}")
print("-" * 46)

for depth in [1, 2, 3, 5, 7, 10, 15, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    depth_str = str(depth) if depth else "None"

    if gap > 0.15:
        status = "과대적합"
    elif test_score < 0.6:
        status = "과소적합"
    else:
        status = "적절"

    print(f"{depth_str:>10} {train_score:>10.3f} {test_score:>12.3f} {status:>10}")


# ============================================================
# Part 3: 기본 모델 성능 (최적화 전)
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 기본 모델 성능 (최적화 전)")
print("=" * 60)

# 기본 설정 RandomForest
model_default = RandomForestClassifier(random_state=42)
scores_default = cross_val_score(model_default, X_train, y_train, cv=5)

print(f"\n[기본 RandomForest]")
print(f"교차검증 점수: {scores_default.mean():.3f} (+/-{scores_default.std():.3f})")

model_default.fit(X_train, y_train)
test_score_default = model_default.score(X_test, y_test)
print(f"테스트 점수: {test_score_default:.3f}")


# ============================================================
# Part 4: GridSearchCV 기본 사용
# ============================================================

print("\n" + "=" * 60)
print("Part 4: GridSearchCV 기본 사용")
print("=" * 60)

# 탐색 범위 정의
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = 3 * 4 * 3 * 3
print(f"\n[탐색 범위]")
print(f"n_estimators: {param_grid['n_estimators']}")
print(f"max_depth: {param_grid['max_depth']}")
print(f"min_samples_split: {param_grid['min_samples_split']}")
print(f"min_samples_leaf: {param_grid['min_samples_leaf']}")
print(f"총 조합 수: {total_combinations}")
print(f"5-Fold 시 학습 횟수: {total_combinations * 5}")

# GridSearchCV 실행
print("\n[GridSearchCV 실행 중...]")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\n[GridSearchCV 결과]")
print(f"탐색 시간: {grid_time:.1f}초")
print(f"최적 파라미터:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"최고 CV 점수 (F1): {grid_search.best_score_:.3f}")

# 테스트 평가
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
test_acc_grid = accuracy_score(y_test, y_pred_grid)
test_f1_grid = f1_score(y_test, y_pred_grid, average='weighted')
print(f"테스트 정확도: {test_acc_grid:.3f}")
print(f"테스트 F1 점수: {test_f1_grid:.3f}")


# ============================================================
# Part 5: cv_results_ 분석
# ============================================================

print("\n" + "=" * 60)
print("Part 5: cv_results_ 분석")
print("=" * 60)

results_df = pd.DataFrame(grid_search.cv_results_)

print("\n[상위 10개 조합]")
top10 = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
for i, row in top10.iterrows():
    print(f"#{int(row['rank_test_score'])}: {row['mean_test_score']:.3f} (+/-{row['std_test_score']:.3f})")

# max_depth별 평균 점수
print("\n[max_depth별 평균 점수]")
for depth in param_grid['max_depth']:
    mask = results_df['param_max_depth'] == depth
    mean_score = results_df.loc[mask, 'mean_test_score'].mean()
    depth_str = str(depth) if depth else "None"
    print(f"  max_depth={depth_str:>4}: {mean_score:.3f}")


# ============================================================
# Part 6: RandomizedSearchCV 사용
# ============================================================

print("\n" + "=" * 60)
print("Part 6: RandomizedSearchCV 사용")
print("=" * 60)

# 분포 정의
param_distributions = {
    'n_estimators': randint(50, 200),      # 50~199 정수
    'max_depth': [5, 10, 15, 20, None],    # 리스트도 가능
    'min_samples_split': randint(2, 20),   # 2~19 정수
    'min_samples_leaf': randint(1, 10),    # 1~9 정수
    'max_features': ['sqrt', 'log2', None]
}

print("\n[탐색 범위]")
print(f"n_estimators: randint(50, 200)")
print(f"max_depth: {param_distributions['max_depth']}")
print(f"min_samples_split: randint(2, 20)")
print(f"min_samples_leaf: randint(1, 10)")
print(f"max_features: {param_distributions['max_features']}")
print(f"탐색 조합 수: 50 (n_iter)")

# RandomizedSearchCV 실행
print("\n[RandomizedSearchCV 실행 중...]")
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"\n[RandomizedSearchCV 결과]")
print(f"탐색 시간: {random_time:.1f}초")
print(f"최적 파라미터:")
for param, value in random_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"최고 CV 점수 (F1): {random_search.best_score_:.3f}")

# 테스트 평가
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
test_acc_random = accuracy_score(y_test, y_pred_random)
test_f1_random = f1_score(y_test, y_pred_random, average='weighted')
print(f"테스트 정확도: {test_acc_random:.3f}")
print(f"테스트 F1 점수: {test_f1_random:.3f}")


# ============================================================
# Part 7: Grid vs Random 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 7: Grid vs Random 비교")
print("=" * 60)

print("\n[Grid Search vs Random Search 비교]")
print(f"{'항목':<20} {'Grid':>12} {'Random':>12}")
print("-" * 48)
print(f"{'탐색 조합 수':<20} {total_combinations:>12} {50:>12}")
print(f"{'탐색 시간(초)':<20} {grid_time:>12.1f} {random_time:>12.1f}")
print(f"{'최고 CV 점수':<20} {grid_search.best_score_:>12.3f} {random_search.best_score_:>12.3f}")
print(f"{'테스트 정확도':<20} {test_acc_grid:>12.3f} {test_acc_random:>12.3f}")
print(f"{'테스트 F1 점수':<20} {test_f1_grid:>12.3f} {test_f1_random:>12.3f}")

if random_time > 0:
    time_ratio = grid_time / random_time
    print(f"\n-> Random은 Grid 대비 {time_ratio:.1f}배 빠름")
print(f"-> 점수 차이: {abs(test_f1_grid - test_f1_random):.3f}")


# ============================================================
# Part 8: 2단계 탐색 전략
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 2단계 탐색 전략")
print("=" * 60)

# 1단계: Random Search로 넓은 범위 탐색
print("\n[1단계: Random Search - 넓은 범위 탐색]")
param_dist_wide = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 25),
    'min_samples_split': randint(2, 30),
    'min_samples_leaf': randint(1, 15)
}

random_stage1 = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist_wide,
    n_iter=30,
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

random_stage1.fit(X_train, y_train)
print(f"1단계 최적 파라미터: {random_stage1.best_params_}")
print(f"1단계 최고 점수: {random_stage1.best_score_:.3f}")

# 최적값 추출
best_n = random_stage1.best_params_['n_estimators']
best_depth = random_stage1.best_params_['max_depth']
best_split = random_stage1.best_params_['min_samples_split']

# 2단계: Grid Search로 좁은 범위 정밀 탐색
print("\n[2단계: Grid Search - 좁은 범위 정밀 탐색]")
param_grid_narrow = {
    'n_estimators': [max(50, best_n-30), best_n, min(300, best_n+30)],
    'max_depth': [max(5, best_depth-3), best_depth, best_depth+3],
    'min_samples_split': [max(2, best_split-2), best_split, best_split+2],
    'min_samples_leaf': [1, 2]
}

print(f"좁혀진 탐색 범위:")
for param, values in param_grid_narrow.items():
    print(f"  {param}: {values}")

grid_stage2 = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_narrow,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_stage2.fit(X_train, y_train)
print(f"\n2단계 최적 파라미터: {grid_stage2.best_params_}")
print(f"2단계 최고 점수: {grid_stage2.best_score_:.3f}")

# 최종 테스트
best_model_2stage = grid_stage2.best_estimator_
y_pred_2stage = best_model_2stage.predict(X_test)
test_acc_2stage = accuracy_score(y_test, y_pred_2stage)
test_f1_2stage = f1_score(y_test, y_pred_2stage, average='weighted')
print(f"테스트 정확도: {test_acc_2stage:.3f}")
print(f"테스트 F1 점수: {test_f1_2stage:.3f}")


# ============================================================
# Part 9: 다중 평가 지표
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 다중 평가 지표")
print("=" * 60)

param_grid_multi = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, None]
}

grid_multi = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_multi,
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'recall_weighted'],
    refit='f1_weighted',  # 최종 모델 선택 기준
    n_jobs=-1
)

grid_multi.fit(X_train, y_train)

print("\n[다중 평가 지표 결과]")
results_multi = pd.DataFrame(grid_multi.cv_results_)

print(f"\n{'파라미터':<35} {'정확도':>8} {'F1':>8} {'재현율':>8}")
print("-" * 65)

for _, row in results_multi.iterrows():
    params_str = str(row['params'])[:35]
    print(f"{params_str:<35} {row['mean_test_accuracy']:>8.3f} "
          f"{row['mean_test_f1_weighted']:>8.3f} {row['mean_test_recall_weighted']:>8.3f}")

print(f"\n최적 파라미터 (F1 기준): {grid_multi.best_params_}")


# ============================================================
# Part 10: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Grid Search: max_depth별 점수 분포
ax1 = axes[0, 0]
grid_results = pd.DataFrame(grid_search.cv_results_)
for depth in [5, 10, 15]:
    mask = grid_results['param_max_depth'] == depth
    subset = grid_results[mask]
    ax1.scatter([depth]*len(subset), subset['mean_test_score'],
               alpha=0.6, s=50, label=f'depth={depth}')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('CV Score (F1)')
ax1.set_title('GridSearchCV: max_depth별 점수 분포')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 방법별 비교 (막대 그래프)
ax2 = axes[0, 1]
methods = ['기본', 'Grid', 'Random', '2단계']
scores = [
    f1_score(y_test, model_default.predict(X_test), average='weighted'),
    test_f1_grid,
    test_f1_random,
    test_f1_2stage
]
colors = ['gray', '#3b82f6', '#22c55e', '#f59e0b']
bars = ax2.bar(methods, scores, color=colors)
ax2.set_ylabel('테스트 F1 점수')
ax2.set_title('방법별 성능 비교 (Digits)')
ax2.set_ylim(0.9, 1.0)
for bar, score in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{score:.3f}', ha='center', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. 탐색 시간 비교
ax3 = axes[1, 0]
methods_time = ['Grid', 'Random']
times = [grid_time, random_time]
colors_time = ['#3b82f6', '#22c55e']
bars_time = ax3.bar(methods_time, times, color=colors_time)
ax3.set_ylabel('시간 (초)')
ax3.set_title('탐색 시간 비교')
for bar, t in zip(bars_time, times):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{t:.1f}초', ha='center', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 4. 예측 결과 샘플 (혼동 행렬 요약)
ax4 = axes[1, 1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_grid)
im = ax4.imshow(cm, cmap='Blues')
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
ax4.set_xlabel('예측')
ax4.set_ylabel('실제')
ax4.set_title(f'혼동 행렬 (GridSearch 최적 모델)\n정확도: {test_acc_grid:.1%}')
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig('17_hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 17_hyperparameter_tuning.png")


# ============================================================
# Part 11: 최적 모델 저장
# ============================================================

print("\n" + "=" * 60)
print("Part 11: 최적 모델 저장")
print("=" * 60)

import joblib
import json

# 최적 모델 저장
joblib.dump(best_model_grid, '17_best_model.pkl')
print("모델 저장 완료: 17_best_model.pkl")

# 최적 파라미터 저장
best_params_serializable = {k: (v if v is not None else "None")
                           for k, v in grid_search.best_params_.items()}
with open('17_best_params.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_serializable, f, indent=2, ensure_ascii=False)
print("파라미터 저장 완료: 17_best_params.json")

# 모델 불러오기 예시
print("\n[모델 불러오기 예시]")
loaded_model = joblib.load('17_best_model.pkl')
loaded_score = loaded_model.score(X_test, y_test)
print(f"불러온 모델 테스트 점수: {loaded_score:.3f}")


# ============================================================
# Part 12: 분류 보고서
# ============================================================

print("\n" + "=" * 60)
print("Part 12: 최종 분류 보고서")
print("=" * 60)

print("\n[최적 모델 분류 보고서]")
print(classification_report(y_test, y_pred_grid))


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 17차시 모델 설정값 최적화")
print("=" * 60)

print("""
1. 하이퍼파라미터
   - 사람이 미리 설정하는 값
   - 예: max_depth, n_estimators, min_samples_split
   - 성능에 큰 영향 -> 최적화 필요

2. Digits 데이터셋
   - 1,797개 손글씨 숫자 이미지 (8x8 픽셀)
   - 0-9 숫자 분류 (10개 클래스)
   - 다중 클래스 분류 연습용

3. GridSearchCV
   - 모든 조합 철저히 탐색
   - param_grid: 딕셔너리로 범위 정의
   - 장점: 확실함 / 단점: 느림

   grid = GridSearchCV(model, param_grid, cv=5)
   grid.fit(X, y)
   grid.best_params_     # 최적 파라미터
   grid.best_score_      # 최고 점수
   grid.best_estimator_  # 최적 모델

4. RandomizedSearchCV
   - 랜덤 샘플링으로 효율적 탐색
   - param_distributions: 분포로 범위 정의
   - n_iter: 탐색 횟수 제한
   - 장점: 빠름, 연속 분포 / 단점: 운 필요

   from scipy.stats import randint, uniform
   param_dist = {'max_depth': randint(3, 20)}
   rand = RandomizedSearchCV(model, param_dist, n_iter=50)

5. 실무 전략 (2단계)
   1) Random Search: 넓은 범위 빠르게
   2) Grid Search: 좁은 범위 정밀하게

6. 유용한 옵션
   - scoring: 'accuracy', 'f1_weighted', 'recall_weighted', 'r2' 등
   - n_jobs=-1: 병렬 처리 (모든 CPU 코어)
   - refit: 다중 지표 시 최종 선택 기준
""")

print("\n다음 차시 예고: 18차시 - 시계열 데이터 기초")
