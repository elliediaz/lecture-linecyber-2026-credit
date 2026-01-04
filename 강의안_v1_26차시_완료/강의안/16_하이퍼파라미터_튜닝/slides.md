---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 16차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 하이퍼파라미터 튜닝

## 16차시 | AI 기초체력훈련 (Pre AI-Campus)

**최적의 모델 설정 찾기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **하이퍼파라미터**의 개념을 이해한다
2. **GridSearchCV**로 최적 파라미터를 찾는다
3. **RandomizedSearchCV**를 활용한다

---

# 하이퍼파라미터란?

## Hyperparameter

> 모델 **학습 전에 설정**하는 값
> (모델이 스스로 학습하지 않음)

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
- **하이퍼파라미터**: 우리가 설정하는 값 (n_estimators, max_depth)

---

# 왜 튜닝이 필요한가?

## 성능 차이가 크다!

```
max_depth=3  → 정확도 75%
max_depth=10 → 정확도 85%
max_depth=50 → 정확도 78% (과대적합)
```

### 문제
- 어떤 값이 최적인지 모름
- 하이퍼파라미터가 여러 개
- 조합이 너무 많음

### 해결: 자동으로 최적값 찾기!

---

# 기본 접근: 수동 튜닝

## 직접 실험

```python
# 여러 max_depth 시도
for depth in [3, 5, 7, 10, 15]:
    model = RandomForestClassifier(max_depth=depth)
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
results = pd.DataFrame(grid_search.cv_results_)

# 주요 컬럼
results[['params', 'mean_test_score', 'rank_test_score']]
```

```
      params                          mean_test_score  rank_test_score
{'max_depth': 10, 'n_estimators': 100}    0.856            1
{'max_depth': 5, 'n_estimators': 200}     0.848            2
...
```

---

# GridSearchCV의 문제

## 조합 폭발

```
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

random_search.fit(X_train, y_train)
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

# 주요 하이퍼파라미터

## 랜덤포레스트

| 파라미터 | 설명 | 권장 범위 |
|---------|------|----------|
| n_estimators | 트리 개수 | 50~500 |
| max_depth | 최대 깊이 | 3~20 |
| min_samples_split | 분할 최소 샘플 | 2~20 |
| min_samples_leaf | 리프 최소 샘플 | 1~10 |
| max_features | 특성 선택 | 'sqrt', 'log2' |

---

# 튜닝 팁

## 효율적인 접근

### 1. 중요한 파라미터부터
```python
# 먼저 핵심 파라미터만
param_grid_1 = {'n_estimators': [100, 200], 'max_depth': [5, 10, 15]}

# 최적값 근처에서 세부 튜닝
param_grid_2 = {'n_estimators': [150, 175, 200], 'max_depth': [8, 10, 12]}
```

### 2. 데이터 일부로 먼저 테스트
```python
# 전체 데이터의 일부로 빠르게 탐색
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.3)
grid_search.fit(X_sample, y_sample)
```

---

# 실습: 전체 워크플로우

## 최적 모델 구축

```python
# 1. 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(...)

# 2. 그리드 서치
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=5
)
grid_search.fit(X_train, y_train)

# 3. 최적 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 4. 평가
print(classification_report(y_test, y_pred))
```

---

# 정리

## 핵심 개념

| 개념 | 설명 |
|------|------|
| 하이퍼파라미터 | 학습 전 설정하는 값 |
| GridSearchCV | 모든 조합 시도 |
| RandomizedSearchCV | 랜덤 샘플링 시도 |
| best_params_ | 최적 파라미터 |
| best_estimator_ | 최적 모델 |

---

# 다음 차시 예고

## 17차시: 시계열 데이터 기초

- 시계열 데이터란?
- 날짜/시간 처리 (datetime)
- 시계열 시각화

> 시간에 따라 변하는 데이터를 다룹니다!

---

# 감사합니다

## AI 기초체력훈련 16차시

**하이퍼파라미터 튜닝**

모델 성능을 최대로 끌어올렸습니다!
