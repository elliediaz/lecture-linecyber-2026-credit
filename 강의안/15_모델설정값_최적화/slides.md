---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 15차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# 모델 설정값 최적화

## 15차시 | Part III. 문제 중심 모델링 실습

**최적의 하이퍼파라미터 찾기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **하이퍼파라미터**의 개념을 이해한다
2. **GridSearchCV**로 최적 설정을 찾는다
3. **RandomizedSearchCV**를 활용한다

---

# 하이퍼파라미터란?

## Hyperparameter

> 모델 **학습 전에 설정**하는 값 (모델이 스스로 학습하지 않음)

### 예시
```python
model = RandomForestClassifier(
    n_estimators=100,     # 트리 개수 → 하이퍼파라미터
    max_depth=10,         # 트리 깊이 → 하이퍼파라미터
    random_state=42
)
```

### 파라미터 vs 하이퍼파라미터
- **파라미터**: 모델이 학습하는 값 (가중치, 절편)
- **하이퍼파라미터**: 우리가 설정하는 값

---

# 왜 튜닝이 필요한가?

## 성능 차이가 크다!

```
max_depth=3  → 정확도 75%
max_depth=10 → 정확도 85%  ← 최적
max_depth=50 → 정확도 78% (과대적합)
```

### 문제
- 어떤 값이 최적인지 모름
- 하이퍼파라미터가 여러 개
- 조합이 너무 많음

> 해결: **자동으로 최적값 찾기!**

---

# 기본 접근: 수동 튜닝

## 직접 실험

```python
# 여러 max_depth 시도
for depth in [3, 5, 7, 10, 15]:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"depth={depth}: {scores.mean():.3f}")
```

### 문제점
- 시간이 오래 걸림
- 조합이 많으면 비현실적
- 코드가 복잡해짐

---

# GridSearchCV

## 모든 조합을 시도

```
         n_estimators
          50   100   200
        ┌─────┬─────┬─────┐
    3   │  ●  │  ●  │  ●  │
        ├─────┼─────┼─────┤
max  5  │  ●  │  ●  │  ●  │
depth   ├─────┼─────┼─────┤
   10   │  ●  │  ●  │  ●  │
        └─────┴─────┴─────┘

9가지 조합 × 5-Fold = 45번 학습
→ 최적 조합 자동 선택!
```

---

# GridSearchCV 코드

## sklearn 구현

```python
from sklearn.model_selection import GridSearchCV

# 1. 탐색할 파라미터 범위 정의
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# 2. GridSearchCV 생성
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 3. 실행
grid_search.fit(X_train, y_train)
```

---

# 결과 확인

## 최적 파라미터

```python
# 최적 파라미터
print(f"최적 파라미터: {grid_search.best_params_}")
# {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}

# 최고 점수
print(f"최고 점수: {grid_search.best_score_:.3f}")
# 0.856

# 최적 모델
best_model = grid_search.best_estimator_

# 테스트 데이터로 평가
test_score = best_model.score(X_test, y_test)
print(f"테스트 점수: {test_score:.3f}")
```

---

# 결과 상세 분석

## cv_results_

```python
# 모든 조합의 결과
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)

# 주요 컬럼만 확인
cols = ['params', 'mean_test_score', 'rank_test_score']
print(results[cols].sort_values('rank_test_score').head())
```

```
      params                              mean_test_score  rank
{'max_depth': 10, 'n_estimators': 100}         0.856       1
{'max_depth': 5, 'n_estimators': 200}          0.848       2
...
```

---

# GridSearchCV의 문제

## 조합 폭발

```python
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],  # 5개
    'max_depth': [3, 5, 7, 10, 15, 20],         # 6개
    'min_samples_split': [2, 5, 10, 15, 20],   # 5개
    'min_samples_leaf': [1, 2, 4, 8]           # 4개
}

# 총 조합: 5 × 6 × 5 × 4 = 600개
# 5-Fold: 600 × 5 = 3,000번 학습!
```

> 시간이 너무 오래 걸림 → **RandomizedSearchCV**

---

# RandomizedSearchCV

## 랜덤 샘플링

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 파라미터 분포 정의
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

# RandomizedSearchCV 생성
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # 20개 조합만 시도
    cv=5,
    random_state=42,
    n_jobs=-1
)
```

---

# Grid vs Random 비교

## 언제 무엇을?

| | GridSearchCV | RandomizedSearchCV |
|--|--------------|-------------------|
| 방식 | 모든 조합 | 랜덤 샘플링 |
| 시간 | 오래 걸림 | 빠름 |
| 최적화 | 확실함 | 근사값 |
| 사용 | 조합 적을 때 | 조합 많을 때 |

### 추천
- 파라미터 2~3개, 값 3~5개: **GridSearchCV**
- 파라미터 많거나 범위 넓음: **RandomizedSearchCV**

---

# 이론 정리

## 하이퍼파라미터 튜닝 핵심

| 개념 | 설명 |
|------|------|
| 하이퍼파라미터 | 학습 전 설정하는 값 |
| GridSearchCV | 모든 조합 시도 |
| RandomizedSearchCV | 랜덤 샘플링 |
| best_params_ | 최적 파라미터 |
| best_estimator_ | 최적 모델 |

---

# - 실습편 -

## 15차시

**GridSearchCV와 RandomizedSearchCV 실습**

---

# 실습 개요

## 최적 모델 찾기

### 목표
- GridSearchCV로 최적 파라미터 탐색
- RandomizedSearchCV로 빠른 탐색
- 최적 모델로 예측

### 실습 환경
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
```

---

# 실습 1: 데이터 준비

## 제조 데이터

```python
np.random.seed(42)
n = 500

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

defect_prob = 0.05 + 0.03*(df['온도']-80)/5
df['불량여부'] = (np.random.random(n) < defect_prob).astype(int)
```

---

# 실습 2: 수동 튜닝

## for 루프로 실험

```python
from sklearn.model_selection import cross_val_score

for depth in [3, 5, 7, 10]:
    for n_est in [50, 100, 200]:
        model = RandomForestClassifier(
            max_depth=depth,
            n_estimators=n_est,
            random_state=42
        )
        scores = cross_val_score(model, X, y, cv=5)
        print(f"depth={depth}, n_est={n_est}: {scores.mean():.3f}")
```

> 코드가 복잡하고 관리하기 어려움

---

# 실습 3: GridSearchCV

## 자동 탐색

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

---

# 실습 4: 결과 확인

## best_params_, best_score_

```python
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차검증 점수: {grid_search.best_score_:.3f}")

# 최적 모델로 테스트
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"테스트 점수: {test_score:.3f}")
```

---

# 실습 5: 결과 상세 분석

## cv_results_

```python
results = pd.DataFrame(grid_search.cv_results_)

# 순위별 정렬
top_results = results[['params', 'mean_test_score', 'rank_test_score']]
print(top_results.sort_values('rank_test_score').head())
```

---

# 실습 6: RandomizedSearchCV

## 빠른 탐색

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

---

# 실습 7: 최종 모델 평가

## classification_report

```python
from sklearn.metrics import classification_report

# 최적 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 상세 평가
print(classification_report(y_test, y_pred,
                            target_names=['정상', '불량']))
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] param_grid 정의
- [ ] GridSearchCV로 자동 탐색
- [ ] best_params_, best_score_ 확인
- [ ] best_estimator_로 최종 모델 사용
- [ ] RandomizedSearchCV 활용

---

# 다음 차시 예고

## 16차시: 시계열 데이터 기초

### 학습 내용
- 시계열 데이터란?
- 날짜/시간 처리 (datetime)
- 시계열 시각화

> 시간에 따라 변하는 데이터를 다룹니다!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **하이퍼파라미터**: 학습 전 설정하는 값
2. **GridSearchCV**: 모든 조합 시도, 확실함
3. **RandomizedSearchCV**: 랜덤 샘플링, 빠름
4. **best_params_**: 최적 파라미터 확인

---

# 감사합니다

## 15차시: 모델 설정값 최적화

**모델 성능을 최대로 끌어올렸습니다!**
