# 17차시: 모델 설정값 최적화

## 학습 목표

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | 하이퍼파라미터의 개념을 이해함 |
| 2 | GridSearchCV로 최적값을 탐색함 |
| 3 | RandomizedSearchCV로 효율적 탐색을 수행함 |

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 하이퍼파라미터 개념 | 파라미터 vs 하이퍼파라미터 |
| 2 | GridSearchCV | 모든 조합 철저 탐색 |
| 3 | RandomizedSearchCV | 랜덤 샘플링 효율 탐색 |

---

## Part 1: 하이퍼파라미터의 개념

### 1.1 파라미터 vs 하이퍼파라미터

| 구분 | 파라미터 (Parameter) | 하이퍼파라미터 (Hyperparameter) |
|------|---------------------|-------------------------------|
| 정의 | 모델이 학습으로 찾는 값 | 사람이 미리 설정하는 값 |
| 예시 | 선형회귀의 가중치(w), 절편(b) | max_depth, n_estimators |
| 결정 시점 | 학습 과정에서 자동 결정 | 학습 전에 결정해야 함 |

---

### 1.2 주요 모델별 하이퍼파라미터

| 모델 | 하이퍼파라미터 | 설명 |
|------|---------------|------|
| DecisionTree | max_depth | 트리 최대 깊이 |
| DecisionTree | min_samples_split | 분할 최소 샘플 수 |
| RandomForest | n_estimators | 트리 개수 |
| RandomForest | max_features | 특성 선택 개수 |
| KNN | n_neighbors | 이웃 개수 |
| SVM | C, gamma | 정규화, 커널 파라미터 |

---

### 1.3 왜 최적화가 필요한가?

하이퍼파라미터가 성능에 큰 영향을 미침.

```python
# max_depth = 2: 과소적합
model = DecisionTreeClassifier(max_depth=2)
# 점수: 0.72

# max_depth = 10: 적절
model = DecisionTreeClassifier(max_depth=10)
# 점수: 0.89

# max_depth = None: 과대적합
model = DecisionTreeClassifier(max_depth=None)
# 점수: 0.85 (테스트에서 떨어짐)
```

---

### 1.4 수동 탐색의 한계

**문제점**
| 문제 | 설명 |
|------|------|
| 시간 소모 | 일일이 테스트해야 함 |
| 조합 폭발 | 파라미터가 많으면 경우의 수 급증 |
| 최적값 놓침 | 일부만 테스트하면 좋은 값 못 찾음 |

**예시**: 3개 파라미터, 각 5개 값
- 조합 수: 5 x 5 x 5 = 125가지
- 교차검증 5회면: 625번 학습

---

### 1.5 탐색 방법 비교

| 방법 | 특징 | 장점 | 단점 |
|------|------|------|------|
| 수동 탐색 | 직접 테스트 | 직관적임 | 시간 소모 |
| Grid Search | 모든 조합 | 철저함 | 느림 |
| Random Search | 랜덤 샘플링 | 빠름 | 운 필요 |
| Bayesian | 이전 결과 활용 | 효율적임 | 복잡함 |

---

## Part 2: GridSearchCV로 최적값 탐색

### 2.1 GridSearchCV란?

- **Grid**: 격자 (모든 조합)
- **Search**: 탐색
- **CV**: 교차검증 (Cross-Validation)

**동작 방식**
1. 하이퍼파라미터 후보값 정의
2. 모든 조합 생성
3. 각 조합으로 교차검증
4. 최고 점수 조합 선택

---

### 2.2 GridSearchCV 시각화

```
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

조합 격자:
         max_depth
           3    5    7
min    2  [X]  [X]  [X]
samples 5  [X]  [X]  [X]
split  10  [X]  [X]  [X]

-> 9가지 조합 x 5-Fold = 45회 학습
```

---

### 2.3 GridSearchCV 기본 사용법

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 1. 탐색 범위 정의
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 2. GridSearchCV 생성
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# 3. 탐색 실행
grid_search.fit(X_train, y_train)

# 4. 결과 확인
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 점수: {grid_search.best_score_:.3f}")

# 5. 최적 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

---

### 2.4 GridSearchCV 주요 속성

| 속성 | 설명 |
|------|------|
| `best_params_` | 최적 하이퍼파라미터 딕셔너리 |
| `best_score_` | 교차검증 최고 점수 |
| `best_estimator_` | 최적 파라미터로 학습된 모델 |
| `cv_results_` | 모든 조합의 상세 결과 |

---

### 2.5 cv_results_ 활용

```python
import pandas as pd

results = pd.DataFrame(grid_search.cv_results_)

# 주요 열 확인
print(results[['params', 'mean_test_score', 'rank_test_score']])

# 상위 5개 조합
top5 = results.nsmallest(5, 'rank_test_score')
print(top5[['params', 'mean_test_score']])
```

---

### 2.6 scoring 옵션

| 문제 유형 | scoring 값 | 설명 |
|----------|-----------|------|
| 분류 | 'accuracy' | 정확도 (기본값) |
| 분류 | 'f1' | F1 Score |
| 분류 | 'precision' | 정밀도 |
| 분류 | 'recall' | 재현율 |
| 회귀 | 'r2' | R^2 점수 |
| 회귀 | 'neg_mean_squared_error' | -MSE |

---

### 2.7 다중 평가 지표

```python
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring=['accuracy', 'f1', 'recall'],
    refit='f1'  # 최종 모델 선택 기준
)

# 결과에서 각 지표별 점수 확인
results = pd.DataFrame(grid_search.cv_results_)
print(results[['mean_test_accuracy', 'mean_test_f1', 'mean_test_recall']])
```

---

### 2.8 실습: GridSearchCV 적용

```python
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# 데이터 로딩
digits = load_digits()
X = digits.data
y = digits.target

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 탐색 범위 정의
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = 3 * 4 * 3 * 3
print(f"[탐색 범위]")
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
test_acc_grid = best_model_grid.score(X_test, y_test)
print(f"테스트 정확도: {test_acc_grid:.3f}")
```

---

### 2.9 GridSearchCV의 한계

**문제점**
| 문제 | 설명 |
|------|------|
| 계산 비용 | 조합이 많으면 매우 느림 |
| 차원의 저주 | 파라미터 늘수록 급증 |
| 불연속 탐색 | 격자 사이 값은 못 찾음 |

**예시**
```
4개 파라미터, 각 10개 값
-> 10^4 = 10,000 조합
-> 5-Fold면 50,000번 학습
-> 몇 시간 ~ 며칠 소요
```

---

## Part 3: RandomizedSearchCV로 효율적 탐색

### 3.1 RandomizedSearchCV란?

모든 조합이 아닌 랜덤 샘플링으로 탐색함.

**장점**
| 장점 | 설명 |
|------|------|
| 빠름 | 시간 제어 가능 |
| 연속 분포 | 범위 내 모든 값 탐색 가능 |
| 효율적 | 대부분 좋은 결과 도출 |

---

### 3.2 Grid vs Random 비교

```
Grid Search (격자):           Random Search (랜덤):
    O-O-O-O-O                     o
    O-O-O-O-O                   o   o
    O-O-O-O-O                 o       o
    O-O-O-O-O                   o o
    O-O-O-O-O                     o

25개 점 모두 탐색            10개 점 랜덤 샘플링
-> 오래 걸림                  -> 빠름, 운 좋으면 최적점
```

---

### 3.3 RandomizedSearchCV 사용법

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 파라미터 분포 정의
param_distributions = {
    'n_estimators': randint(50, 200),      # 50~199 정수
    'max_depth': [5, 10, 15, 20, None],    # 리스트도 가능
    'min_samples_split': randint(2, 20),   # 2~19 정수
    'min_samples_leaf': randint(1, 10),    # 1~9 정수
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,           # 50개 조합만 테스트
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1            # 병렬 처리
)

random_search.fit(X_train, y_train)

print(f"최적 파라미터: {random_search.best_params_}")
print(f"최고 점수: {random_search.best_score_:.3f}")
```

---

### 3.4 분포 함수 종류

| 분포 | scipy 함수 | 용도 |
|------|-----------|------|
| 이산 균등 | `randint(a, b)` | 정수 파라미터 |
| 연속 균등 | `uniform(a, b)` | 실수 파라미터 |
| 로그 균등 | `loguniform(a, b)` | 스케일이 큰 파라미터 |
| 리스트 | `[val1, val2, ...]` | 특정 값만 선택 |

---

### 3.5 로그 스케일 탐색

파라미터 범위가 클 때 사용함.

```python
from scipy.stats import loguniform

param_dist = {
    'learning_rate': loguniform(1e-4, 1e-1),
    # 0.0001 ~ 0.1 사이를 로그 균등하게 샘플링
    # -> 0.0001, 0.001, 0.01, 0.1 근처가 골고루 탐색됨
}
```

---

### 3.6 n_iter 선택 가이드

| 상황 | n_iter | 이유 |
|------|--------|------|
| 탐색적 | 10~20 | 빠른 확인 |
| 일반적 | 50~100 | 균형잡힌 탐색 |
| 중요한 모델 | 200+ | 철저한 탐색 |

**경험적 규칙**: 전체 조합의 5~10% 정도면 충분함

---

### 3.7 실습: RandomizedSearchCV 적용

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import time

# 분포 정의
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

print("[탐색 범위]")
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
test_acc_random = best_model_random.score(X_test, y_test)
print(f"테스트 정확도: {test_acc_random:.3f}")
```

---

### 3.8 실습: Grid vs Random 비교

```python
from sklearn.metrics import f1_score

# F1 점수 계산
y_pred_grid = grid_search.best_estimator_.predict(X_test)
y_pred_random = random_search.best_estimator_.predict(X_test)

test_f1_grid = f1_score(y_test, y_pred_grid, average='weighted')
test_f1_random = f1_score(y_test, y_pred_random, average='weighted')

print("\n[Grid vs Random 비교]")
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
```

**결과 해설**
- Random Search가 훨씬 빠르면서 비슷한 성능을 달성함
- 실무에서는 Random Search로 먼저 탐색 후 Grid Search로 정밀 탐색하는 전략이 효과적임

---

### 3.9 실무 전략: 2단계 탐색

```python
# 1단계: Random Search로 넓은 범위 탐색
print("[1단계: Random Search - 넓은 범위 탐색]")
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
test_acc_2stage = best_model_2stage.score(X_test, y_test)
print(f"테스트 정확도: {test_acc_2stage:.3f}")
```

---

### 3.10 병렬 처리 (n_jobs)

```python
# CPU 코어 활용
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1  # 모든 코어 사용
)

# n_jobs 옵션
# -1: 모든 코어
#  1: 단일 코어 (기본값)
#  4: 4개 코어
```

---

### 3.11 최적 모델 저장

```python
import joblib
import json

# 최적 모델 저장
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')
print("모델 저장 완료: best_model.pkl")

# 최적 파라미터 저장
best_params_serializable = {k: (v if v is not None else "None")
                           for k, v in grid_search.best_params_.items()}
with open('best_params.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_serializable, f, indent=2, ensure_ascii=False)
print("파라미터 저장 완료: best_params.json")

# 모델 불러오기
loaded_model = joblib.load('best_model.pkl')
loaded_score = loaded_model.score(X_test, y_test)
print(f"불러온 모델 테스트 점수: {loaded_score:.3f}")
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 하이퍼파라미터 | 사람이 설정하는 값 (max_depth, n_estimators 등) |
| GridSearchCV | 모든 조합 탐색 (철저함, 느림) |
| RandomizedSearchCV | 랜덤 샘플링 (효율적, 빠름) |
| n_iter | Random Search에서 탐색할 조합 수 |
| n_jobs | 병렬 처리 코어 수 (-1이면 모든 코어) |

---

## sklearn 주요 함수 요약

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)
grid.best_params_      # 최적 파라미터
grid.best_score_       # 최고 점수
grid.best_estimator_   # 최적 모델

# RandomizedSearchCV
rand = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5)
rand.fit(X, y)
```

---

## 실무 전략 요약

| 단계 | 방법 | 목적 |
|------|------|------|
| 1단계 | Random Search | 넓은 범위 빠르게 탐색 |
| 2단계 | Grid Search | 좁은 범위 정밀 탐색 |
| 최종 | 테스트 평가 | 최적 모델 성능 확인 |
