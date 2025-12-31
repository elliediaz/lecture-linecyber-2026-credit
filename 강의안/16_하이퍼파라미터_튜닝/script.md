# [16차시] 하이퍼파라미터 튜닝 - 강사 스크립트

## 강의 정보
- **차시**: 16차시 (25분)
- **유형**: 이론 + 실습
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 지난 시간 복습 [1.5분]

> 안녕하세요, 16차시를 시작하겠습니다.
>
> 지난 시간에 교차검증과 다양한 평가 지표를 배웠습니다. 모델을 신뢰성 있게 평가하는 방법을 익혔죠.
>
> 그런데 모델을 만들 때 max_depth, n_estimators 같은 값은 어떻게 정해야 할까요? 오늘은 **최적의 설정값을 자동으로 찾는 방법**을 배웁니다.

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, 하이퍼파라미터의 개념을 이해합니다.
> 둘째, GridSearchCV로 최적 파라미터를 찾습니다.
> 셋째, RandomizedSearchCV를 활용합니다.

---

## 전개 (19분)

### 섹션 1: 하이퍼파라미터 이해 (4분)

#### 하이퍼파라미터란 [2min]

> **하이퍼파라미터(Hyperparameter)**는 모델 학습 전에 우리가 설정하는 값입니다.
>
> 예를 들어 RandomForestClassifier에서:
> - n_estimators=100 → 트리 100개
> - max_depth=10 → 각 트리 깊이 10
>
> 이 값들은 모델이 스스로 학습하지 않아요. **우리가 정해줘야** 합니다.
>
> 반면 선형회귀에서 가중치(w)와 절편(b)은 모델이 학습하는 값이에요. 이것을 **파라미터**라고 합니다.

#### 왜 튜닝이 중요한가 [2min]

> 하이퍼파라미터 값에 따라 성능이 크게 달라집니다.
>
> max_depth=3이면 75%, max_depth=10이면 85%, max_depth=50이면 78%(과대적합)
>
> 문제는 **어떤 값이 최적인지 미리 알 수 없다**는 거예요. 데이터마다 다르거든요.
>
> 그래서 여러 값을 시도해보고 가장 좋은 것을 선택해야 합니다.

---

### 섹션 2: GridSearchCV (8분)

#### 개념 설명 [2min]

> **GridSearchCV**는 모든 조합을 시도하는 방법입니다.
>
> 예를 들어:
> - n_estimators: [50, 100, 200] → 3가지
> - max_depth: [3, 5, 10] → 3가지
>
> 총 9가지 조합을 다 시도하고, 각각 5-Fold 교차검증을 합니다. 그래서 45번 학습해요.
>
> 그 중 가장 좋은 조합을 자동으로 알려줍니다!

#### 코드 실습 [4min]

> *(코드 시연)*
>
> ```python
> from sklearn.model_selection import GridSearchCV
>
> # 탐색할 파라미터 범위
> param_grid = {
>     'n_estimators': [50, 100, 200],
>     'max_depth': [3, 5, 10]
> }
>
> # GridSearchCV 생성
> grid_search = GridSearchCV(
>     estimator=RandomForestClassifier(random_state=42),
>     param_grid=param_grid,
>     cv=5,
>     scoring='accuracy',
>     n_jobs=-1
> )
>
> # 실행
> grid_search.fit(X_train, y_train)
> ```
>
> n_jobs=-1은 모든 CPU 코어를 사용해서 병렬로 처리합니다.

#### 결과 확인 [2min]

> *(코드 시연)*
>
> ```python
> # 최적 파라미터
> print(grid_search.best_params_)
> # {'max_depth': 10, 'n_estimators': 100}
>
> # 최고 점수
> print(grid_search.best_score_)
> # 0.856
>
> # 최적 모델
> best_model = grid_search.best_estimator_
> ```
>
> best_params_에 최적 파라미터가, best_estimator_에 최적으로 학습된 모델이 들어있어요.

---

### 섹션 3: RandomizedSearchCV (4min)

#### GridSearchCV의 문제 [1.5min]

> GridSearchCV는 좋지만 **조합이 많으면 시간이 너무 오래 걸려요**.
>
> 파라미터 4개에 각각 5개씩 값이 있으면 5×5×5×5 = 625개 조합이에요. 5-Fold면 3,125번 학습!
>
> 이럴 때 **RandomizedSearchCV**를 사용합니다.

#### RandomizedSearchCV [2.5min]

> *(코드 시연)*
>
> ```python
> from sklearn.model_selection import RandomizedSearchCV
> from scipy.stats import randint
>
> param_dist = {
>     'n_estimators': randint(50, 300),
>     'max_depth': randint(3, 20)
> }
>
> random_search = RandomizedSearchCV(
>     RandomForestClassifier(random_state=42),
>     param_distributions=param_dist,
>     n_iter=20,  # 20개 조합만 시도
>     cv=5
> )
> random_search.fit(X_train, y_train)
> ```
>
> n_iter=20이면 랜덤하게 20개 조합만 시도해요. 전체를 다 하지 않아도 좋은 결과를 빠르게 얻을 수 있습니다.

---

### 섹션 4: 실전 팁 (3min)

#### 효율적인 튜닝 전략 [2min]

> 튜닝 시간을 줄이는 팁을 알려드릴게요.
>
> 첫째, **중요한 파라미터부터** 튜닝하세요. 랜덤포레스트에서는 n_estimators와 max_depth가 가장 중요해요.
>
> 둘째, **범위를 넓게 → 좁게** 접근하세요. 먼저 [50, 100, 200]으로 대략적인 최적값을 찾고, 그 근처에서 [90, 100, 110]으로 세밀하게 튜닝합니다.
>
> 셋째, 데이터가 크면 **일부 샘플로 먼저 테스트**하세요.

#### 주의사항 [1min]

> GridSearchCV가 찾은 best_score_는 학습 데이터 기준이에요. 반드시 **테스트 데이터로 최종 평가**해야 합니다.
>
> ```python
> test_score = best_model.score(X_test, y_test)
> ```

---

## 정리 (3분)

### 핵심 내용 요약 [1.5분]

> 오늘 배운 핵심 내용을 정리하면:
>
> 1. **하이퍼파라미터**: 학습 전 설정하는 값 (n_estimators, max_depth 등)
> 2. **GridSearchCV**: 모든 조합 시도, 확실하지만 시간 오래 걸림
> 3. **RandomizedSearchCV**: 랜덤 샘플링, 빠름
> 4. **best_params_**: 최적 파라미터
> 5. **best_estimator_**: 최적 모델
>
> 조합이 적으면 GridSearchCV, 많으면 RandomizedSearchCV를 사용하세요!

### 다음 차시 예고 [1min]

> 다음 17차시부터 **시계열 데이터**를 다룹니다.
>
> 날짜와 시간에 따라 변하는 데이터, 예를 들어 일별 생산량, 월별 판매량 같은 데이터를 분석하고 예측해봅니다.

### 마무리 인사 [0.5분]

> 최적의 모델 설정을 자동으로 찾는 방법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 예상 질문
1. "n_iter를 얼마로 해야 하나요?"
   → 보통 20~100. 시간과 성능의 균형

2. "GridSearchCV가 항상 최고인가요?"
   → 조합이 적으면 최고, 많으면 RandomizedSearchCV 추천

3. "여러 모델을 비교할 때도 사용하나요?"
   → 각 모델별로 GridSearchCV 후 최종 비교

### 시간 조절 팁
- 시간 부족: RandomizedSearchCV 간략히
- 시간 여유: cv_results_ 분석, 시각화
