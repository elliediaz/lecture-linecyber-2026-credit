# [15차시] 모델 설정값 최적화 - 다이어그램

## 1. 파라미터 vs 하이퍼파라미터

```mermaid
flowchart LR
    subgraph 파라미터["파라미터 (Parameter)"]
        A1["가중치, 절편"]
        B1["모델이 학습"]
        C1["fit() 중에 결정"]
    end

    subgraph 하이퍼파라미터["하이퍼파라미터"]
        A2["n_estimators, max_depth"]
        B2["우리가 설정"]
        C2["fit() 전에 결정"]
    end
```

## 2. 하이퍼파라미터 영향

```mermaid
flowchart TD
    A["max_depth 설정"]

    A --> B["depth=3"]
    A --> C["depth=10"]
    A --> D["depth=50"]

    B --> B1["정확도 75%<br>과소적합"]
    C --> C1["정확도 85%<br>최적!"]
    D --> D1["정확도 78%<br>과대적합"]

    style C1 fill:#c8e6c9
```

## 3. GridSearchCV 개념

```mermaid
flowchart TD
    A["param_grid 정의"]

    subgraph 조합["모든 조합"]
        B1["n_est=50, depth=3"]
        B2["n_est=50, depth=5"]
        B3["n_est=50, depth=10"]
        B4["n_est=100, depth=3"]
        B5["..."]
        B6["n_est=200, depth=10"]
    end

    C["각 조합 교차검증"]
    D["최적 조합 선택"]

    A --> 조합 --> C --> D
```

## 4. GridSearchCV 그리드

```mermaid
flowchart LR
    subgraph 그리드["파라미터 그리드"]
        direction TB
        A["n_estimators"]
        B["[50, 100, 200]"]
        C["max_depth"]
        D["[3, 5, 10]"]
    end

    E["3 × 3 = 9가지 조합"]
    F["× 5-Fold = 45번 학습"]

    그리드 --> E --> F
```

## 5. GridSearchCV 코드 흐름

```mermaid
flowchart TD
    A["1. param_grid 정의"]
    B["2. GridSearchCV 생성"]
    C["3. fit(X_train, y_train)"]
    D["4. best_params_ 확인"]
    E["5. best_estimator_ 사용"]

    A --> B --> C --> D --> E
```

## 6. GridSearchCV 결과

```mermaid
flowchart LR
    subgraph 결과["GridSearchCV 결과"]
        A["best_params_<br>최적 파라미터"]
        B["best_score_<br>최고 교차검증 점수"]
        C["best_estimator_<br>최적 모델"]
        D["cv_results_<br>모든 결과"]
    end
```

## 7. 조합 폭발 문제

```mermaid
flowchart TD
    A["파라미터 4개<br>각 5개 값"]

    B["5 × 5 × 5 × 5 = 625 조합"]
    C["× 5-Fold = 3,125번 학습"]
    D["⚠️ 시간 너무 오래!"]

    A --> B --> C --> D
```

## 8. RandomizedSearchCV

```mermaid
flowchart TD
    subgraph Grid["GridSearchCV"]
        A1["모든 조합"]
        B1["625번 학습"]
        C1["확실"]
    end

    subgraph Random["RandomizedSearchCV"]
        A2["랜덤 선택"]
        B2["20번 학습"]
        C2["빠름"]
    end

    style Random fill:#c8e6c9
```

## 9. Grid vs Random 비교

```mermaid
flowchart LR
    subgraph Grid["GridSearchCV"]
        A1["✅ 확실히 최적 찾음"]
        B1["❌ 시간 오래 걸림"]
        C1["조합 적을 때 사용"]
    end

    subgraph Random["RandomizedSearchCV"]
        A2["✅ 빠름"]
        B2["△ 근사값"]
        C2["조합 많을 때 사용"]
    end
```

## 10. 추천 전략

```mermaid
flowchart TD
    A["튜닝 시작"]

    A --> B{"조합 수?"}
    B --> |"적음<br>(< 50)"| C["GridSearchCV"]
    B --> |"많음<br>(> 100)"| D["RandomizedSearchCV"]

    D --> E["대략적 범위 파악"]
    E --> F["GridSearchCV로 세밀 조정"]
```

## 11. 랜덤포레스트 주요 하이퍼파라미터

```mermaid
mindmap
  root((RandomForest<br>하이퍼파라미터))
    n_estimators
      트리 개수
      50~500
    max_depth
      트리 깊이
      3~20
    min_samples_split
      분할 최소 샘플
      2~20
    min_samples_leaf
      리프 최소 샘플
      1~10
    max_features
      특성 선택
      sqrt, log2
```

## 12. 튜닝 워크플로우

```mermaid
flowchart TD
    A["1. 기본 모델 학습"]
    B["2. 중요 파라미터 선정"]
    C["3. 탐색 범위 정의"]
    D["4. GridSearchCV 또는 RandomizedSearchCV"]
    E["5. best_params_ 확인"]
    F["6. 최적 모델로 평가"]

    A --> B --> C --> D --> E --> F
```

## 13. n_jobs 병렬 처리

```mermaid
flowchart LR
    subgraph 직렬["n_jobs=1 (기본)"]
        A1["조합1"] --> A2["조합2"] --> A3["조합3"]
    end

    subgraph 병렬["n_jobs=-1 (모든 코어)"]
        B1["조합1"]
        B2["조합2"]
        B3["조합3"]
    end

    C["병렬이 훨씬 빠름!"]
    병렬 --> C
```

## 14. 강의 구조

```mermaid
gantt
    title 15차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    하이퍼파라미터란?         :a2, after a1, 1.5m
    왜 튜닝 필요?            :a3, after a2, 1.5m
    GridSearchCV            :a4, after a3, 2.5m
    RandomizedSearchCV       :a5, after a4, 1.5m
    언제 무엇을?             :a6, after a5, 1m

    section 실습편
    실습 소개               :b1, after a6, 2m
    데이터 준비              :b2, after b1, 2m
    수동 튜닝               :b3, after b2, 2m
    GridSearchCV            :b4, after b3, 3m
    결과 확인               :b5, after b4, 2m
    결과 상세 분석           :b6, after b5, 2m
    RandomizedSearchCV       :b7, after b6, 2m
    최종 모델 평가           :b8, after b7, 2m

    section 정리
    핵심 요약               :c1, after b8, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 15. 핵심 요약

```mermaid
mindmap
  root((하이퍼파라미터<br>튜닝))
    하이퍼파라미터
      학습 전 설정
      n_estimators
      max_depth
    GridSearchCV
      모든 조합
      확실함
      조합 적을 때
    RandomizedSearchCV
      랜덤 샘플링
      빠름
      조합 많을 때
    결과
      best_params_
      best_score_
      best_estimator_
```

## 16. 실무 팁

```mermaid
flowchart TD
    A["효율적 튜닝 팁"]

    A --> B["1. 중요 파라미터부터"]
    A --> C["2. 넓은 범위 → 좁은 범위"]
    A --> D["3. 데이터 일부로 먼저 테스트"]
    A --> E["4. n_jobs=-1 병렬 처리"]

    B --> B1["n_estimators, max_depth"]
    C --> C1["Random → Grid"]
    D --> D1["시간 절약"]
    E --> E1["속도 향상"]
```
