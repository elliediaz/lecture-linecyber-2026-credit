---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [17차시] 모델 설정값 최적화

## 공공데이터 AI 예측 모델 개발

---

# 학습 목표

1. **하이퍼파라미터의 개념**을 이해한다
2. **GridSearchCV로 최적값**을 탐색한다
3. **RandomizedSearchCV로 효율적 탐색**을 수행한다

---

# 목차

## 대주제 1: 하이퍼파라미터의 개념
## 대주제 2: GridSearchCV로 최적값 탐색
## 대주제 3: RandomizedSearchCV로 효율적 탐색
## 실습편: 제조 품질 예측 모델 최적화

---

<!-- _class: lead -->
# 대주제 1
## 하이퍼파라미터의 개념을 이해한다

---

# 파라미터 vs 하이퍼파라미터

## 파라미터 (Parameter)
- 모델이 **학습으로 찾는** 값
- 예: 선형회귀의 가중치(w), 절편(b)
- 데이터로부터 자동 결정

## 하이퍼파라미터 (Hyperparameter)
- **사람이 미리 설정**하는 값
- 예: max_depth, n_estimators
- 학습 전에 결정해야 함

---

# 하이퍼파라미터 예시

| 모델 | 하이퍼파라미터 | 설명 |
|------|---------------|------|
| DecisionTree | max_depth | 트리 최대 깊이 |
| DecisionTree | min_samples_split | 분할 최소 샘플 수 |
| RandomForest | n_estimators | 트리 개수 |
| RandomForest | max_features | 특성 선택 개수 |
| KNN | n_neighbors | 이웃 개수 |
| SVM | C, gamma | 정규화, 커널 파라미터 |

---

# 왜 최적화가 필요한가?

## 하이퍼파라미터가 성능에 큰 영향

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

# 수동 탐색의 한계

## 문제점

1. **시간 소모**: 일일이 테스트해야 함
2. **조합 폭발**: 파라미터가 많으면 경우의 수 급증
3. **최적값 놓침**: 일부만 테스트하면 좋은 값 못 찾음

## 예시: 3개 파라미터, 각 5개 값
- 조합 수: 5 × 5 × 5 = **125가지**
- 교차검증 5회면: 625번 학습

---

# 자동 탐색의 필요성

```
수동 탐색:
  max_depth=3 → 0.82
  max_depth=5 → 0.85
  max_depth=7 → 0.87  ← 여기서 멈춤?
  max_depth=10 → 0.89 ← 못 찾음!

자동 탐색 (GridSearchCV):
  모든 조합 자동 테스트
  → 최적값 찾음!
```

---

# 탐색 방법 비교

| 방법 | 특징 | 장점 | 단점 |
|------|------|------|------|
| 수동 탐색 | 직접 테스트 | 직관적 | 시간 소모 |
| Grid Search | 모든 조합 | 철저함 | 느림 |
| Random Search | 랜덤 샘플링 | 빠름 | 운 필요 |
| Bayesian | 이전 결과 활용 | 효율적 | 복잡함 |

---

<!-- _class: lead -->
# 대주제 2
## GridSearchCV로 최적값 탐색

---

# GridSearchCV란?

## 개념
- **Grid**: 격자 (모든 조합)
- **Search**: 탐색
- **CV**: 교차검증 (Cross-Validation)

## 동작 방식
1. 하이퍼파라미터 후보값 정의
2. 모든 조합 생성
3. 각 조합으로 교차검증
4. 최고 점수 조합 선택

---

# GridSearchCV 시각화

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

→ 9가지 조합 × 5-Fold = 45회 학습
```

---

# GridSearchCV 기본 사용법

```python
from sklearn.model_selection import GridSearchCV

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
```

---

# GridSearchCV 실행 및 결과

```python
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

# GridSearchCV 주요 속성

| 속성 | 설명 |
|------|------|
| `best_params_` | 최적 하이퍼파라미터 딕셔너리 |
| `best_score_` | 교차검증 최고 점수 |
| `best_estimator_` | 최적 파라미터로 학습된 모델 |
| `cv_results_` | 모든 조합의 상세 결과 |

---

# cv_results_ 활용

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

# scoring 옵션

| 문제 유형 | scoring 값 | 설명 |
|----------|-----------|------|
| 분류 | 'accuracy' | 정확도 (기본값) |
| 분류 | 'f1' | F1 Score |
| 분류 | 'precision' | 정밀도 |
| 분류 | 'recall' | 재현율 |
| 회귀 | 'r2' | R² 점수 |
| 회귀 | 'neg_mean_squared_error' | -MSE |

---

# 다중 평가 지표

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

# GridSearchCV의 한계

## 문제점
1. **계산 비용**: 조합이 많으면 매우 느림
2. **차원의 저주**: 파라미터 늘수록 급증
3. **불연속 탐색**: 격자 사이 값은 못 찾음

## 예시
```
4개 파라미터, 각 10개 값
→ 10^4 = 10,000 조합
→ 5-Fold면 50,000번 학습
→ 몇 시간 ~ 며칠 소요
```

---

<!-- _class: lead -->
# 대주제 3
## RandomizedSearchCV로 효율적 탐색

---

# RandomizedSearchCV란?

## 개념
- 모든 조합이 아닌 **랜덤 샘플링**
- 지정한 횟수만큼만 테스트

## 장점
1. **빠름**: 시간 제어 가능
2. **연속 분포**: 범위 내 모든 값 탐색 가능
3. **효율적**: 대부분 좋은 결과 도출

---

# Grid vs Random 비교

```
Grid Search (격자):           Random Search (랜덤):
    ●─●─●─●─●                     ○
    ●─●─●─●─●                   ○   ○
    ●─●─●─●─●                 ○       ○
    ●─●─●─●─●                   ○ ○
    ●─●─●─●─●                     ○

25개 점 모두 탐색            10개 점 랜덤 샘플링
→ 오래 걸림                  → 빠름, 운 좋으면 최적점
```

---

# RandomizedSearchCV 사용법

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 파라미터 분포 정의
param_distributions = {
    'max_depth': randint(3, 20),        # 3~19 정수
    'min_samples_split': randint(2, 20), # 2~19 정수
    'min_samples_leaf': randint(1, 10),  # 1~9 정수
    'max_features': uniform(0.5, 0.5)    # 0.5~1.0 실수
}
```

---

# RandomizedSearchCV 실행

```python
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,           # 50개 조합만 테스트
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1            # 병렬 처리
)

random_search.fit(X_train, y_train)

print(f"최적 파라미터: {random_search.best_params_}")
print(f"최고 점수: {random_search.best_score_:.3f}")
```

---

# 분포 함수 종류

| 분포 | scipy 함수 | 용도 |
|------|-----------|------|
| 이산 균등 | `randint(a, b)` | 정수 파라미터 |
| 연속 균등 | `uniform(a, b)` | 실수 파라미터 |
| 로그 균등 | `loguniform(a, b)` | 스케일이 큰 파라미터 |
| 리스트 | `[val1, val2, ...]` | 특정 값만 선택 |

---

# 로그 스케일 탐색

## 언제 사용?
- 파라미터 범위가 클 때
- 예: learning_rate (0.0001 ~ 1)

```python
from scipy.stats import loguniform

param_dist = {
    'learning_rate': loguniform(1e-4, 1e-1),
    # 0.0001 ~ 0.1 사이를 로그 균등하게 샘플링
    # → 0.0001, 0.001, 0.01, 0.1 근처가 골고루 탐색됨
}
```

---

# n_iter 선택 가이드

| 상황 | n_iter | 이유 |
|------|--------|------|
| 탐색적 | 10~20 | 빠른 확인 |
| 일반적 | 50~100 | 균형잡힌 탐색 |
| 중요한 모델 | 200+ | 철저한 탐색 |

## 경험적 규칙
- Grid보다 같거나 더 좋은 결과를 위해
- 전체 조합의 5~10% 정도면 충분

---

# Grid vs Random 실험 비교

```python
# Grid Search: 4×4×4 = 64 조합
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
# 64 조합 × 5-Fold = 320회 학습

# Random Search: 30 조합만
param_dist = {
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 30),
    'min_samples_leaf': randint(1, 15)
}
# 30 조합 × 5-Fold = 150회 학습
# → 절반 시간에 비슷한 결과!
```

---

# 실무 전략

## 2단계 탐색

1. **1단계: Random Search** (넓은 범위, 적은 n_iter)
   - 좋은 영역 파악

2. **2단계: Grid Search** (좁은 범위, 세밀하게)
   - 최적값 정밀 탐색

```python
# 1단계: 넓은 범위
random_search → best: max_depth=8

# 2단계: 좁은 범위
param_grid = {'max_depth': [6, 7, 8, 9, 10]}
```

---

# 병렬 처리 (n_jobs)

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

<!-- _class: lead -->
# 실습편
## 제조 품질 예측 모델 최적화

---

# 실습 개요

## 목표
- 제조 불량 예측 모델의 하이퍼파라미터 최적화
- GridSearchCV와 RandomizedSearchCV 비교

## 데이터
- 제조 공정 센서 데이터
- 특성: 온도, 압력, 진동, 습도
- 타겟: 불량 여부

---

# 실습 1: 데이터 준비

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'pressure': np.random.normal(100, 10, n),
    'vibration': np.random.normal(0.5, 0.1, n),
    'humidity': np.random.normal(50, 10, n)
})

df['defect'] = ((df['temperature'] > 88) |
               (df['vibration'] > 0.6)).astype(int)
```

---

# 실습 2: 기본 모델 성능

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X = df[['temperature', 'pressure', 'vibration', 'humidity']]
y = df['defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 기본 설정 모델
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"기본 모델: {scores.mean():.3f} (±{scores.std():.3f})")
```

---

# 실습 3: GridSearchCV 적용

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
```

---

# 실습 4: GridSearchCV 결과

```python
import time

start = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

print(f"[GridSearchCV 결과]")
print(f"탐색 시간: {grid_time:.1f}초")
print(f"총 조합 수: {len(grid_search.cv_results_['params'])}")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 점수: {grid_search.best_score_:.3f}")

# 테스트 점수
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"테스트 점수: {test_score:.3f}")
```

---

# 실습 5: RandomizedSearchCV 적용

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # 50개만 탐색
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
```

---

# 실습 6: RandomizedSearchCV 결과

```python
start = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start

print(f"[RandomizedSearchCV 결과]")
print(f"탐색 시간: {random_time:.1f}초")
print(f"탐색 조합 수: 50")
print(f"최적 파라미터: {random_search.best_params_}")
print(f"최고 점수: {random_search.best_score_:.3f}")

# 테스트 점수
best_model_r = random_search.best_estimator_
test_score_r = best_model_r.score(X_test, y_test)
print(f"테스트 점수: {test_score_r:.3f}")
```

---

# 실습 7: 결과 비교

```python
print("\n[Grid vs Random 비교]")
print(f"{'항목':<15} {'Grid':>10} {'Random':>10}")
print("-" * 38)
print(f"{'탐색 조합 수':<15} {len(grid_search.cv_results_['params']):>10} {50:>10}")
print(f"{'탐색 시간(초)':<15} {grid_time:>10.1f} {random_time:>10.1f}")
print(f"{'최고 CV 점수':<15} {grid_search.best_score_:>10.3f} {random_search.best_score_:>10.3f}")
print(f"{'테스트 점수':<15} {test_score:>10.3f} {test_score_r:>10.3f}")
```

---

# 실습 8: 결과 시각화

```python
import matplotlib.pyplot as plt

# cv_results를 DataFrame으로
grid_results = pd.DataFrame(grid_search.cv_results_)

# max_depth별 평균 점수
plt.figure(figsize=(10, 6))
for depth in [3, 5, 7, 10]:
    mask = grid_results['param_max_depth'] == depth
    subset = grid_results[mask]
    plt.scatter([depth]*len(subset), subset['mean_test_score'],
               alpha=0.5, label=f'depth={depth}')

plt.xlabel('max_depth')
plt.ylabel('CV Score (F1)')
plt.title('GridSearchCV 결과 분포')
plt.legend()
```

---

# 실습 9: 2단계 탐색

```python
# 1단계: RandomizedSearchCV로 넓은 탐색
random_search_1 = RandomizedSearchCV(
    model, param_dist, n_iter=30, cv=5, random_state=42
)
random_search_1.fit(X_train, y_train)
print(f"1단계 최적: {random_search_1.best_params_}")

# 2단계: 좁은 범위로 GridSearchCV
best_depth = random_search_1.best_params_['max_depth']
param_grid_2 = {
    'max_depth': [best_depth-2, best_depth-1, best_depth,
                  best_depth+1, best_depth+2],
    'n_estimators': [80, 100, 120],
}
grid_search_2 = GridSearchCV(model, param_grid_2, cv=5)
grid_search_2.fit(X_train, y_train)
```

---

# 실습 10: 최종 모델 저장

```python
import joblib

# 최적 모델 저장
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')
print("모델 저장 완료: best_model.pkl")

# 최적 파라미터 저장
import json
with open('best_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f, indent=2)
print("파라미터 저장 완료: best_params.json")
```

---

# 핵심 정리

## 1. 하이퍼파라미터
- 사람이 설정하는 값 (max_depth, n_estimators 등)
- 성능에 큰 영향 → 최적화 필요

## 2. GridSearchCV
- 모든 조합 탐색 (철저함)
- 시간 많이 소요

## 3. RandomizedSearchCV
- 랜덤 샘플링 (효율적)
- 연속 분포 탐색 가능

---

# sklearn 주요 함수

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

# 다음 차시 예고

## 17차시: 시계열 데이터 기초
- 시계열 데이터의 특성
- Python datetime 처리
- Pandas 날짜 인덱스
- 이동평균, 시차 변수

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습 파일: `16_hyperparameter_tuning.py`
