# [13차시] 예측 모델: 선형회귀와 다항회귀 - 다이어그램

## 1. 분류 vs 회귀

```mermaid
flowchart LR
    subgraph 분류["분류 (Classification)"]
        A1["입력 데이터"] --> B1["모델"]
        B1 --> C1["범주<br>불량/정상"]
    end

    subgraph 회귀["회귀 (Regression)"]
        A2["입력 데이터"] --> B2["모델"]
        B2 --> C2["숫자<br>1,247개"]
    end
```

## 2. 선형회귀 원리

```mermaid
flowchart TD
    subgraph 개념["선형회귀"]
        A["y = wx + b"]
        B["w: 기울기 (특성의 영향력)"]
        C["b: 절편 (기본값)"]
    end

    subgraph 예시["제조 현장 예시"]
        D["생산량 = 10×온도 + 3×습도 + 100"]
        E["온도 1도 ↑ → 생산량 10개 ↑"]
    end

    개념 --> 예시
```

## 3. 다중 선형회귀

```mermaid
flowchart LR
    subgraph 입력["입력 특성"]
        X1["온도"]
        X2["습도"]
        X3["속도"]
    end

    B[["선형회귀<br>모델"]]

    subgraph 출력["출력"]
        Y["생산량<br>(숫자)"]
    end

    입력 --> B --> 출력
```

## 4. 회귀 평가 지표

```mermaid
flowchart TD
    A["회귀 평가 지표"]

    A --> B["MSE<br>(Mean Squared Error)"]
    A --> C["RMSE<br>(Root MSE)"]
    A --> D["R²<br>(결정계수)"]

    B --> B1["오차² 평균<br>작을수록 좋음"]
    C --> C1["√MSE<br>원래 단위로 해석"]
    D --> D1["0~1 사이<br>1에 가까울수록 좋음"]

    style D1 fill:#c8e6c9
```

## 5. R² 해석

```mermaid
flowchart LR
    subgraph 나쁨["❌ R² = 0.3"]
        A1["30% 설명"]
        B1["많이 빗나감"]
    end

    subgraph 보통["△ R² = 0.7"]
        A2["70% 설명"]
        B2["괜찮음"]
    end

    subgraph 좋음["✅ R² = 0.95"]
        A3["95% 설명"]
        B3["매우 좋음"]
    end

    style 좋음 fill:#c8e6c9
```

## 6. 선형회귀의 한계

```mermaid
flowchart LR
    subgraph 가능["✅ 선형 관계"]
        A1["직선으로 설명 가능"]
        B1["●  ●<br>  ●  ●<br>●  ●"]
    end

    subgraph 불가능["❌ 비선형 관계"]
        A2["직선으로 설명 불가"]
        B2["●     ●<br>  ●  ●<br>    ●"]
    end
```

## 7. 다항회귀 개념

```mermaid
flowchart TD
    A["선형회귀 한계"] --> B["곡선 관계"]

    B --> C["다항회귀<br>Polynomial Regression"]

    C --> D["2차: y = w₁x + w₂x² + b"]
    C --> E["3차: y = w₁x + w₂x² + w₃x³ + b"]

    F["곡선으로 모델링!"]
    D --> F
    E --> F
```

## 8. PolynomialFeatures 동작

```mermaid
flowchart LR
    subgraph 원본["원본 특성"]
        A["[x₁, x₂]"]
    end

    B[["PolynomialFeatures<br>degree=2"]]

    subgraph 변환["변환된 특성"]
        C["[1, x₁, x₂, x₁², x₁x₂, x₂²]"]
    end

    원본 --> B --> 변환
```

## 9. Pipeline 구조

```mermaid
flowchart LR
    A["X_train"] --> B["PolynomialFeatures"]
    B --> C["LinearRegression"]
    C --> D["예측"]

    subgraph Pipeline
        B
        C
    end
```

## 10. degree에 따른 변화

```mermaid
flowchart TD
    A["degree 선택"]

    A --> B["degree=1"]
    A --> C["degree=2"]
    A --> D["degree=5+"]

    B --> B1["직선<br>과소적합 가능"]
    C --> C1["2차 곡선<br>적절!"]
    D --> D1["복잡한 곡선<br>과대적합 위험"]

    style C1 fill:#c8e6c9
```

## 11. 과대적합 진단

```mermaid
flowchart TD
    A["degree 증가"]

    subgraph 학습["학습 성능"]
        B["R²: 0.9 → 0.95 → 0.99"]
        C["계속 증가"]
    end

    subgraph 테스트["테스트 성능"]
        D["R²: 0.85 → 0.90 → 0.70"]
        E["어느 순간 감소!"]
    end

    A --> 학습
    A --> 테스트

    F["⚠️ 과대적합 신호"]
    E --> F
```

## 12. sklearn 사용 흐름

```mermaid
flowchart TD
    A["1. 데이터 준비<br>X, y 분리"]
    B["2. 학습/테스트 분리<br>train_test_split"]
    C["3. 모델 생성<br>LinearRegression()"]
    D["4. 학습<br>model.fit(X_train, y_train)"]
    E["5. 예측<br>model.predict(X_test)"]
    F["6. 평가<br>r2_score(y_test, y_pred)"]

    A --> B --> C --> D --> E --> F
```

## 13. 회귀 모델 비교

```mermaid
flowchart TD
    A["회귀 모델 선택"]

    A --> B["LinearRegression"]
    A --> C["PolynomialRegression"]
    A --> D["RandomForestRegressor"]

    B --> B1["장점: 해석 쉬움<br>단점: 비선형 어려움"]
    C --> C1["장점: 곡선 표현<br>단점: 과대적합 주의"]
    D --> D1["장점: 높은 성능<br>단점: 해석 어려움"]

    E["먼저 LinearRegression!"]
    B --> E

    style E fill:#c8e6c9
```

## 14. Classifier vs Regressor

```mermaid
flowchart LR
    subgraph 분류["분류 모델"]
        A1["DecisionTreeClassifier"]
        A2["RandomForestClassifier"]
    end

    subgraph 회귀["회귀 모델"]
        B1["DecisionTreeRegressor"]
        B2["RandomForestRegressor"]
    end

    A1 --> |"출력: 범주"| C1["정상/불량"]
    B1 --> |"출력: 숫자"| C2["1,247개"]
```

## 15. 강의 구조

```mermaid
gantt
    title 13차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    분류 vs 회귀             :a2, after a1, 1m
    선형회귀 복습            :a3, after a2, 1.5m
    다중 선형회귀            :a4, after a3, 1.5m
    평가 지표               :a5, after a4, 1.5m
    다항회귀                :a6, after a5, 2.5m

    section 실습편
    실습 소개               :b1, after a6, 2m
    데이터 준비              :b2, after b1, 2m
    선형회귀 학습            :b3, after b2, 2m
    모델 평가               :b4, after b3, 2m
    다항회귀                :b5, after b4, 3m
    degree 실험             :b6, after b5, 2m
    트리 기반 회귀           :b7, after b6, 2m
    새 데이터 예측           :b8, after b7, 2m

    section 정리
    핵심 요약               :c1, after b8, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 16. 핵심 요약

```mermaid
mindmap
  root((선형회귀 &<br>다항회귀))
    선형회귀
      y = wx + b
      직선 관계
      coef_, intercept_
    다항회귀
      곡선 관계
      PolynomialFeatures
      degree 2~3 권장
    평가 지표
      MSE
      RMSE
      R² (1에 가까울수록 좋음)
```
