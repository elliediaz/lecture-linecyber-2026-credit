# [14차시] 분류 모델 - 랜덤포레스트 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["앙상블<br>학습 개념"]
    B["랜덤포레스트<br>원리"]
    C["sklearn<br>실습"]
    D["14차시:<br>회귀 모델"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. 앙상블 개념

```mermaid
flowchart TD
    A["앙상블<br>Ensemble"]

    A --> B["정의"]
    B --> B1["여러 모델 결합"]
    B --> B2["더 나은 예측"]

    A --> C["비유"]
    C --> C1["집단 지성"]
    C --> C2["전문가 협의"]
    C --> C3["여러 시험 평균"]

    style A fill:#1e40af,color:#fff
```

## 3. 집단 지성

```mermaid
flowchart TD
    A["집단 지성의 힘"]

    A --> B["젤리빈 실험"]
    B --> B1["개인 추측<br>큰 오차"]
    B --> B2["집단 평균<br>실제 값 근접"]

    A --> C["원리"]
    C --> C1["각자 다른 방향 오류"]
    C --> C2["평균내면 상쇄"]

    style A fill:#1e40af,color:#fff
    style B2 fill:#dcfce7
```

## 4. 앙상블 조건

```mermaid
flowchart TD
    A["좋은 앙상블 조건"]

    A --> B["필수 조건"]
    B --> B1["각 모델이<br>어느 정도 정확"]
    B --> B2["서로 다른<br>오류 패턴"]

    A --> C["나쁜 예"]
    C --> C1["똑같은 모델 100개"]
    C --> C2["같은 곳에서 틀림"]

    style B fill:#dcfce7
    style C fill:#fecaca
```

## 5. 앙상블 방법 종류

```mermaid
flowchart TD
    A["앙상블 방법"]

    A --> B["배깅<br>Bagging"]
    A --> C["부스팅<br>Boosting"]
    A --> D["스태킹<br>Stacking"]

    B --> B1["병렬 학습"]
    B --> B2["랜덤포레스트"]

    C --> C1["순차 학습"]
    C --> C2["XGBoost"]

    D --> D1["계층 학습"]
    D --> D2["모델 결합"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
```

## 6. 배깅 vs 부스팅

```mermaid
flowchart LR
    subgraph bagging["배깅 (Bagging)"]
        A1["병렬 학습"]
        A2["독립"]
        A3["분산 감소"]
    end

    subgraph boosting["부스팅 (Boosting)"]
        B1["순차 학습"]
        B2["의존"]
        B3["편향 감소"]
    end

    style bagging fill:#dcfce7
    style boosting fill:#fef3c7
```

## 7. 배깅 과정

```mermaid
flowchart TD
    A["원본 데이터"]

    A --> B["부트스트랩 1"]
    A --> C["부트스트랩 2"]
    A --> D["부트스트랩 3"]

    B --> E["모델 1"]
    C --> F["모델 2"]
    D --> G["모델 3"]

    E --> H["결합<br>(투표/평균)"]
    F --> H
    G --> H

    H --> I["최종 예측"]

    style A fill:#1e40af,color:#fff
    style H fill:#dcfce7
    style I fill:#fef3c7
```

## 8. 부트스트랩 샘플링

```mermaid
flowchart TD
    A["부트스트랩 샘플링"]

    A --> B["원본"]
    B --> B1["[A, B, C, D, E]"]

    A --> C["샘플 1"]
    C --> C1["[A, A, C, D, E]"]

    A --> D["샘플 2"]
    D --> D1["[B, C, C, E, E]"]

    A --> E["샘플 3"]
    E --> E1["[A, B, D, D, D]"]

    style A fill:#1e40af,color:#fff
```

## 9. 부트스트랩 특성

```mermaid
flowchart TD
    A["부트스트랩 특성"]

    A --> B["복원 추출"]
    B --> B1["일부 데이터 중복"]
    B --> B2["일부 데이터 제외"]

    A --> C["제외 비율"]
    C --> C1["약 37% 제외"]
    C --> C2["OOB 데이터"]

    style A fill:#1e40af,color:#fff
    style C2 fill:#fef3c7
```

## 10. 랜덤포레스트 정의

```mermaid
flowchart TD
    A["랜덤포레스트<br>Random Forest"]

    A --> B["구성"]
    B --> B1["의사결정나무<br>여러 개"]
    B --> B2["투표로 결합"]

    A --> C["핵심"]
    C --> C1["나무 하나: 불안정"]
    C --> C2["숲: 안정적"]

    style A fill:#1e40af,color:#fff
    style C2 fill:#dcfce7
```

## 11. 두 가지 랜덤

```mermaid
flowchart TD
    A["랜덤포레스트의 랜덤"]

    A --> B["데이터 랜덤"]
    B --> B1["부트스트랩 샘플링"]
    B --> B2["각 트리 다른 데이터"]

    A --> C["특성 랜덤"]
    C --> C1["분할시 일부 특성만"]
    C --> C2["max_features"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 12. 특성 랜덤 선택

```mermaid
flowchart TD
    A["특성 랜덤 선택"]

    A --> B["전체 특성"]
    B --> B1["온도, 습도,<br>속도, 압력"]

    A --> C["분할 1"]
    C --> C1["[온도, 습도]<br>중 선택"]

    A --> D["분할 2"]
    D --> D1["[속도, 압력]<br>중 선택"]

    A --> E["효과"]
    E --> E1["트리 다양성 증가"]

    style A fill:#1e40af,color:#fff
```

## 13. 랜덤포레스트 학습

```mermaid
flowchart TD
    A["원본 데이터"]

    A --> B["트리 1"]
    B --> B1["부트스트랩"]
    B1 --> B2["특성 랜덤"]
    B2 --> B3["학습"]

    A --> C["트리 2"]
    C --> C1["부트스트랩"]
    C1 --> C2["특성 랜덤"]
    C2 --> C3["학습"]

    A --> D["..."]
    A --> E["트리 n"]

    style A fill:#1e40af,color:#fff
```

## 14. 예측: 다수결 투표

```mermaid
flowchart LR
    A["새 데이터"]

    A --> B["트리 1"]
    A --> C["트리 2"]
    A --> D["트리 3"]

    B --> B1["정상"]
    C --> C1["불량"]
    D --> D1["정상"]

    B1 --> E["다수결<br>2:1"]
    C1 --> E
    D1 --> E

    E --> F["최종: 정상"]

    style E fill:#fef3c7
    style F fill:#dcfce7
```

## 15. 예측: 평균 (회귀)

```mermaid
flowchart LR
    A["새 데이터"]

    A --> B["트리 1"]
    A --> C["트리 2"]
    A --> D["트리 3"]

    B --> B1["1200"]
    C --> C1["1150"]
    D --> D1["1200"]

    B1 --> E["평균"]
    C1 --> E
    D1 --> E

    E --> F["최종: 1183"]

    style E fill:#fef3c7
    style F fill:#dcfce7
```

## 16. OOB 점수

```mermaid
flowchart TD
    A["OOB (Out-of-Bag)"]

    A --> B["개념"]
    B --> B1["학습에 사용 안 된<br>데이터"]

    A --> C["특성"]
    C --> C1["각 트리당 약 37%"]

    A --> D["활용"]
    D --> D1["별도 테스트 없이<br>성능 추정"]

    style A fill:#1e40af,color:#fff
    style D1 fill:#dcfce7
```

## 17. 의사결정나무 vs 랜덤포레스트

```mermaid
flowchart LR
    subgraph dt["의사결정나무"]
        A1["1개 트리"]
        A2["불안정"]
        A3["과대적합 쉬움"]
        A4["해석 용이"]
    end

    subgraph rf["랜덤포레스트"]
        B1["100+ 트리"]
        B2["안정적"]
        B3["과대적합 저항"]
        B4["해석 어려움"]
    end

    style dt fill:#dbeafe
    style rf fill:#dcfce7
```

## 18. 성능 비교

```mermaid
flowchart TD
    A["성능 비교"]

    A --> B["정확도"]
    B --> B1["RF > DT"]

    A --> C["안정성"]
    C --> C1["RF >> DT"]

    A --> D["속도"]
    D --> D1["DT > RF"]

    A --> E["해석"]
    E --> E1["DT > RF"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#dcfce7
```

## 19. 장단점

```mermaid
flowchart TD
    A["랜덤포레스트"]

    A --> B["장점"]
    B --> B1["과대적합 저항"]
    B --> B2["높은 정확도"]
    B --> B3["신뢰할 수 있는<br>특성 중요도"]
    B --> B4["튜닝 없이도 좋음"]

    A --> C["단점"]
    C --> C1["학습 시간 오래"]
    C --> C2["메모리 많이 사용"]
    C --> C3["해석 어려움"]

    style B fill:#dcfce7
    style C fill:#fecaca
```

## 20. RandomForestClassifier

```mermaid
flowchart TD
    A["RandomForestClassifier"]

    A --> B["주요 파라미터"]
    B --> B1["n_estimators: 트리 수"]
    B --> B2["max_depth: 최대 깊이"]
    B --> B3["max_features: 특성 수"]
    B --> B4["n_jobs: CPU 수"]

    A --> C["메서드"]
    C --> C1["fit, predict, score"]
    C --> C2["predict_proba"]
    C --> C3["feature_importances_"]

    style A fill:#1e40af,color:#fff
```

## 21. n_estimators 영향

```mermaid
flowchart LR
    A["트리 10개"]
    B["트리 50개"]
    C["트리 100개"]
    D["트리 200개"]

    A --> A1["낮음"]
    B --> B1["상승"]
    C --> C1["포화"]
    D --> D1["거의 동일"]

    style A fill:#fecaca
    style C fill:#dcfce7
    style D fill:#dcfce7
```

## 22. n_estimators 그래프

```mermaid
flowchart TD
    A["n_estimators 선택"]

    A --> B["10-50"]
    B --> B1["성능 급상승"]

    A --> C["100-200"]
    C --> C1["최적 구간<br>권장"]

    A --> D["300+"]
    D --> D1["성능 거의 동일<br>시간/메모리 낭비"]

    style C fill:#dcfce7
```

## 23. sklearn 사용법

```mermaid
flowchart TD
    A["1. 모델 생성"]
    B["2. 학습 (fit)"]
    C["3. 예측 (predict)"]
    D["4. 평가 (score)"]
    E["5. 특성 중요도"]

    A --> B --> C --> D --> E

    A --> A1["RandomForest<br>Classifier()"]
    B --> B1["model.fit(X, y)"]
    C --> C1["model.predict(X)"]
    D --> D1["model.score(X, y)"]
    E --> E1["feature_importances_"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 24. 코드 예시

```mermaid
flowchart TD
    A["랜덤포레스트 코드"]

    A --> B["생성"]
    B --> B1["RandomForestClassifier<br>(n_estimators=100)"]

    A --> C["학습"]
    C --> C1["model.fit<br>(X_train, y_train)"]

    A --> D["예측"]
    D --> D1["model.predict<br>(X_test)"]

    A --> E["평가"]
    E --> E1["model.score<br>(X_test, y_test)"]

    style A fill:#1e40af,color:#fff
```

## 25. OOB 점수 활용

```mermaid
flowchart TD
    A["OOB 점수 사용"]

    A --> B["설정"]
    B --> B1["oob_score=True"]

    A --> C["학습"]
    C --> C1["model.fit(X, y)"]

    A --> D["확인"]
    D --> D1["model.oob_score_"]

    A --> E["장점"]
    E --> E1["별도 검증 세트<br>불필요"]

    style A fill:#1e40af,color:#fff
```

## 26. 특성 중요도 비교

```mermaid
flowchart LR
    subgraph dt["의사결정나무"]
        A1["특성 A: 0.8"]
        A2["특성 B: 0.2"]
        A3["불안정"]
    end

    subgraph rf["랜덤포레스트"]
        B1["특성 A: 0.6"]
        B2["특성 B: 0.4"]
        B3["안정적"]
    end

    style rf fill:#dcfce7
```

## 27. 안정성 실험

```mermaid
flowchart TD
    A["안정성 실험"]

    A --> B["10회 반복 실험"]

    B --> C["의사결정나무"]
    C --> C1["평균: 85%"]
    C --> C2["표준편차: 3%"]

    B --> D["랜덤포레스트"]
    D --> D1["평균: 88%"]
    D --> D2["표준편차: 1%"]

    style D fill:#dcfce7
```

## 28. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 준비"]
    B["2. DT vs RF 비교"]
    C["3. 안정성 실험"]
    D["4. n_estimators 탐색"]
    E["5. 특성 중요도"]
    F["6. 새 데이터 예측"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 29. 실무 권장 설정

```mermaid
flowchart TD
    A["권장 설정"]

    A --> B["n_estimators"]
    B --> B1["100~200"]

    A --> C["max_depth"]
    C --> C1["None 또는 15~20"]

    A --> D["max_features"]
    D --> D1["'sqrt' (기본값)"]

    A --> E["n_jobs"]
    E --> E1["-1 (전체 CPU)"]

    style A fill:#1e40af,color:#fff
```

## 30. 언제 사용?

```mermaid
flowchart TD
    A["랜덤포레스트 사용 시기"]

    A --> B["빠른 프로토타이핑"]
    B --> B1["튜닝 없이도<br>좋은 성능"]

    A --> C["특성 중요도 분석"]
    C --> C1["신뢰할 수 있는<br>중요도"]

    A --> D["안정적 예측 필요"]
    D --> D1["분산이 낮음"]

    style A fill:#1e40af,color:#fff
```

## 31. 다음 차시 연결

```mermaid
flowchart LR
    A["14차시<br>랜덤포레스트"]
    B["14차시<br>선형/다항회귀"]

    A --> B

    A --> A1["분류"]
    A --> A2["범주 예측"]

    B --> B1["회귀"]
    B --> B2["숫자 예측"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 32. 핵심 정리

```mermaid
flowchart TD
    A["14차시 핵심"]

    A --> B["앙상블"]
    B --> B1["여러 모델 결합"]

    A --> C["랜덤포레스트"]
    C --> C1["배깅 + 특성 랜덤"]

    A --> D["장점"]
    D --> D1["안정적<br>높은 정확도"]

    style A fill:#1e40af,color:#fff
```
