# [15차시] 모델 설정값 최적화 - 강사 스크립트

## 강의 정보
- **차시**: 15차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 15차시를 시작하겠습니다.
>
> 지난 시간에 교차검증과 혼동행렬을 배웠습니다. 모델을 제대로 평가하는 방법을 알았죠.
>
> 오늘은 모델 **성능을 최대로** 끌어올리는 방법을 배웁니다. 바로 **하이퍼파라미터 튜닝**입니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 하이퍼파라미터의 개념을 이해합니다.
> 둘째, GridSearchCV로 최적 설정을 찾습니다.
> 셋째, RandomizedSearchCV를 활용합니다.

---

### 핵심 내용 (8분)

#### 하이퍼파라미터란? [1.5분]

> **하이퍼파라미터**는 모델 학습 전에 우리가 설정하는 값입니다.
>
> 예를 들어 랜덤포레스트의 n_estimators, max_depth 같은 거예요.
>
> ```python
> model = RandomForestClassifier(
>     n_estimators=100,  # 트리 개수
>     max_depth=10       # 트리 깊이
> )
> ```
>
> 모델이 스스로 학습하는 가중치(파라미터)와 다릅니다. 우리가 직접 정해줘야 해요.

#### 왜 튜닝이 필요한가? [1.5min]

> 하이퍼파라미터 값에 따라 성능이 크게 달라집니다.
>
> max_depth=3이면 75%, max_depth=10이면 85%, max_depth=50이면 78%가 나올 수 있어요.
>
> 너무 작으면 과소적합, 너무 크면 과대적합이죠.
>
> 문제는 어떤 값이 최적인지 모른다는 거예요. 그래서 **자동으로 찾는 방법**이 필요합니다.

#### GridSearchCV [2.5min]

> **GridSearchCV**는 모든 조합을 시도해서 최적을 찾습니다.
>
> n_estimators가 [50, 100, 200], max_depth가 [3, 5, 10]이면, 3×3=9가지 조합을 다 해봐요.
>
> 각 조합마다 교차검증을 하니까, 5-Fold면 9×5=45번 학습합니다.
>
> ```python
> from sklearn.model_selection import GridSearchCV
>
> param_grid = {
>     'n_estimators': [50, 100, 200],
>     'max_depth': [3, 5, 10]
> }
>
> grid_search = GridSearchCV(
>     estimator=RandomForestClassifier(random_state=42),
>     param_grid=param_grid,
>     cv=5
> )
> grid_search.fit(X_train, y_train)
> ```
>
> 결과는 best_params_로 최적 파라미터, best_score_로 최고 점수를 확인합니다.

#### RandomizedSearchCV [1.5min]

> GridSearchCV의 문제는 조합이 많으면 시간이 오래 걸린다는 거예요.
>
> 파라미터 4개, 각각 5개 값이면 625개 조합이죠.
>
> **RandomizedSearchCV**는 모든 조합을 다 하지 않고 랜덤하게 일부만 시도합니다.
>
> ```python
> from sklearn.model_selection import RandomizedSearchCV
>
> random_search = RandomizedSearchCV(
>     estimator=model,
>     param_distributions=param_dist,
>     n_iter=20  # 20개 조합만 시도
> )
> ```
>
> 시간이 적게 들고, 보통 좋은 결과를 찾아줍니다.

#### 언제 무엇을? [1min]

> 정리하면:
>
> - **GridSearchCV**: 조합이 적을 때 (파라미터 2~3개, 값 3~5개)
> - **RandomizedSearchCV**: 조합이 많을 때 (파라미터 많거나 범위 넓을 때)
>
> 실무에서는 먼저 RandomizedSearchCV로 대략적인 범위를 찾고, 그 근처에서 GridSearchCV로 세밀하게 조정하기도 합니다.

---

## 실습편 (15-20분)

### 실습 소개 [2분]

> 이제 실습 시간입니다. GridSearchCV와 RandomizedSearchCV를 직접 사용해봅니다.
>
> **실습 목표**입니다.
> 1. GridSearchCV로 최적 파라미터를 찾습니다.
> 2. 결과를 분석합니다.
> 3. RandomizedSearchCV로 빠른 탐색을 해봅니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
> ```

### 실습 1: 데이터 준비 [2min]

> 첫 번째 실습입니다. 제조 데이터를 생성합니다.
>
> 이전과 같은 불량 예측 문제입니다.

### 실습 2: 수동 튜닝 [2min]

> 두 번째 실습입니다. for 루프로 직접 실험해봅니다.
>
> ```python
> for depth in [3, 5, 7, 10]:
>     for n_est in [50, 100, 200]:
>         model = RandomForestClassifier(max_depth=depth, n_estimators=n_est)
>         scores = cross_val_score(model, X, y, cv=5)
>         print(f"depth={depth}, n_est={n_est}: {scores.mean():.3f}")
> ```
>
> 코드가 복잡하고 결과 관리도 어렵죠? 이걸 자동화하는 게 GridSearchCV입니다.

### 실습 3: GridSearchCV [3min]

> 세 번째 실습입니다. GridSearchCV로 자동 탐색합니다.
>
> ```python
> from sklearn.model_selection import GridSearchCV
>
> param_grid = {
>     'n_estimators': [50, 100, 200],
>     'max_depth': [3, 5, 10]
> }
>
> grid_search = GridSearchCV(
>     estimator=RandomForestClassifier(random_state=42),
>     param_grid=param_grid,
>     cv=5,
>     n_jobs=-1
> )
>
> grid_search.fit(X_train, y_train)
> ```
>
> n_jobs=-1은 모든 CPU 코어를 사용한다는 뜻이에요. 속도가 빨라집니다.

### 실습 4: 결과 확인 [2min]

> 네 번째 실습입니다. 결과를 확인합니다.
>
> ```python
> print(f"최적 파라미터: {grid_search.best_params_}")
> print(f"최고 점수: {grid_search.best_score_:.3f}")
>
> best_model = grid_search.best_estimator_
> test_score = best_model.score(X_test, y_test)
> print(f"테스트 점수: {test_score:.3f}")
> ```
>
> best_estimator_가 최적 파라미터로 학습된 모델이에요. 바로 사용할 수 있습니다.

### 실습 5: 결과 상세 분석 [2min]

> 다섯 번째 실습입니다. cv_results_로 모든 조합 결과를 봅니다.
>
> ```python
> results = pd.DataFrame(grid_search.cv_results_)
> cols = ['params', 'mean_test_score', 'rank_test_score']
> print(results[cols].sort_values('rank_test_score').head())
> ```
>
> 어떤 조합이 몇 위인지 한눈에 볼 수 있어요.

### 실습 6: RandomizedSearchCV [2min]

> 여섯 번째 실습입니다. RandomizedSearchCV를 사용해봅니다.
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
>     estimator=RandomForestClassifier(random_state=42),
>     param_distributions=param_dist,
>     n_iter=20,
>     cv=5
> )
>
> random_search.fit(X_train, y_train)
> ```
>
> randint(50, 300)은 50에서 300 사이 랜덤 정수를 뜻해요.

### 실습 7: 최종 모델 평가 [2min]

> 마지막 실습입니다. 최적 모델로 분류 리포트를 출력합니다.
>
> ```python
> from sklearn.metrics import classification_report
>
> best_model = grid_search.best_estimator_
> y_pred = best_model.predict(X_test)
>
> print(classification_report(y_test, y_pred,
>                             target_names=['정상', '불량']))
> ```
>
> 이제 최적화된 모델이 완성됐습니다!

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **하이퍼파라미터**는 학습 전에 우리가 설정하는 값입니다. n_estimators, max_depth 같은 거예요.
>
> **GridSearchCV**는 모든 조합을 시도해서 최적을 찾습니다. 확실하지만 조합이 많으면 오래 걸려요.
>
> **RandomizedSearchCV**는 랜덤하게 일부만 시도합니다. 빠르고 대부분 좋은 결과를 찾아줘요.
>
> **best_params_**로 최적 파라미터, **best_estimator_**로 최적 모델을 얻을 수 있습니다.

#### 다음 차시 예고 [1min]

> 다음 16차시에서는 **시계열 데이터 기초**를 배웁니다.
>
> 제조 현장에서는 시간에 따른 데이터가 많죠. 센서 로그, 생산량 추이 같은 거요. 시계열 데이터를 다루는 방법을 배웁니다.

#### 마무리 [0.5min]

> 모델 성능을 최대로 끌어올리는 방법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)

### 주의사항
- GridSearchCV vs RandomizedSearchCV 차이 명확히 설명
- best_params_, best_estimator_ 사용법 강조
- n_jobs=-1 옵션 소개

### 예상 질문
1. "GridSearchCV가 너무 오래 걸려요"
   → n_jobs=-1로 병렬 처리, 또는 RandomizedSearchCV 사용

2. "n_iter를 몇으로 해야 하나요?"
   → 보통 20~50. 시간과 성능 trade-off

3. "어떤 파라미터를 튜닝해야 하나요?"
   → 문서 참고. RF는 n_estimators, max_depth가 핵심

4. "테스트 점수가 교차검증 점수보다 낮아요"
   → 정상. 교차검증은 학습 데이터에서의 추정치
