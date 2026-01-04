# [6차시] 상관분석과 예측의 기초 - 다이어그램

## 1. 상관분석이 필요한 이유

```mermaid
flowchart TD
    A[제조 현장의 질문] --> B["온도가 높으면<br>불량률도 높아지나?"]
    A --> C["가동 시간과<br>생산량은 관련 있나?"]
    A --> D["습도가 품질에<br>영향을 주나?"]

    B --> E[상관분석]
    C --> E
    D --> E

    E --> F[두 변수가 함께<br>변하는 정도 측정]
    F --> G[상관계수 r]
```

## 2. 상관계수 해석

```mermaid
flowchart LR
    subgraph 양의상관["양의 상관 (r > 0)"]
        A["r = +1.0<br>완벽한 양의 상관"]
        B["r = +0.8<br>강한 양의 상관"]
        C["r = +0.5<br>중간 양의 상관"]
    end

    subgraph 음의상관["음의 상관 (r < 0)"]
        D["r = -0.8<br>강한 음의 상관"]
        E["r = -0.5<br>중간 음의 상관"]
        F["r = -1.0<br>완벽한 음의 상관"]
    end

    subgraph 무상관["상관 없음"]
        G["r ≈ 0<br>관련 없음"]
    end
```

## 3. 상관계수 강도 기준

```mermaid
mindmap
  root((상관계수 |r|))
    강한 상관
      0.7 ~ 1.0
      명확한 관계
      예측에 유용
    중간 상관
      0.3 ~ 0.7
      어느 정도 관계
      다른 요인도 영향
    약한 상관
      0.0 ~ 0.3
      관계 미약
      예측 어려움
```

## 4. 상관관계 vs 인과관계

```mermaid
flowchart TD
    subgraph 상관관계["상관관계"]
        A["아이스크림 판매량"] <--> B["익사 사고 수"]
        A1["r = 0.85 (높은 상관)"]
    end

    C["숨겨진 변수: 기온"] --> A
    C --> B

    subgraph 결론["결론"]
        D["상관관계 ≠ 인과관계"]
        E["상관분석은 관계의<br>존재만 알려줌"]
        F["원인은 별도 분석 필요"]
    end
```

## 5. 선형회귀 개념

```mermaid
flowchart TD
    A["독립변수 X<br>(온도)"] --> B["선형회귀 모델"]
    B --> C["종속변수 Y<br>(불량률)"]

    D["Y = β₀ + β₁X"]

    subgraph 구성요소["구성 요소"]
        E["β₀ (절편)<br>X=0일 때 Y값"]
        F["β₁ (기울기)<br>X가 1 증가할 때<br>Y 변화량"]
    end

    D --> 구성요소
```

## 6. 최소제곱법 (OLS)

```mermaid
flowchart TD
    A[데이터 포인트들] --> B["각 점에서 직선까지<br>거리(오차) 계산"]
    B --> C["오차의 제곱합"]
    C --> D["제곱합을 최소화하는<br>β₀, β₁ 찾기"]
    D --> E["최적의 회귀선"]

    subgraph 오차["오차 계산"]
        F["오차 = 실제값 - 예측값"]
        G["Σ(yᵢ - ŷᵢ)² → 최소화"]
    end
```

## 7. R² (결정계수) 해석

```mermaid
flowchart LR
    subgraph R2["R² 해석"]
        A["R² = 0.9<br>90% 설명<br>매우 좋음"]
        B["R² = 0.7<br>70% 설명<br>좋음"]
        C["R² = 0.5<br>50% 설명<br>보통"]
        D["R² = 0.3<br>30% 설명<br>낮음"]
    end

    E["R² = 1 - (잔차제곱합/총제곱합)"]
    E --> R2

    F["의미: 독립변수가<br>종속변수 변동의<br>몇 %를 설명하는가"]
```

## 8. sklearn 선형회귀 흐름

```mermaid
flowchart TD
    A["1. 라이브러리 임포트<br>from sklearn.linear_model<br>import LinearRegression"]
    A --> B["2. 데이터 준비<br>X (2D 배열), y (1D 배열)"]
    B --> C["3. 모델 생성<br>model = LinearRegression()"]
    C --> D["4. 학습<br>model.fit(X, y)"]
    D --> E["5. 결과 확인<br>model.intercept_<br>model.coef_"]
    E --> F["6. 예측<br>model.predict(new_X)"]
    F --> G["7. 평가<br>r2_score(y, y_pred)"]
```

## 9. 회귀선 시각화

```mermaid
flowchart TD
    subgraph 그래프["산점도 + 회귀선"]
        A["● 실제 데이터 (scatter)"]
        B["─ 회귀선 (plot)"]
        C["↕ 잔차 (오차)"]
    end

    D["좋은 모델"] --> E["점들이 직선에<br>가깝게 분포"]
    F["나쁜 모델"] --> G["점들이 직선에서<br>멀리 퍼져 있음"]
```

## 10. 예측 활용

```mermaid
flowchart LR
    A["학습된 모델<br>Y = 1.5 + 0.05X"] --> B["새로운 온도 입력"]
    B --> C["온도 85도"]
    B --> D["온도 90도"]
    B --> E["온도 95도"]

    C --> F["예측: 5.75%"]
    D --> G["예측: 6.0%"]
    E --> H["예측: 6.25%"]

    I["역산도 가능"] --> J["목표 불량률 5%<br>→ 최대 온도 70도"]
```

## 11. 상관분석 vs 회귀분석

```mermaid
flowchart TD
    subgraph 상관분석["상관분석"]
        A["목적: 관계의 강도 측정"]
        B["결과: 상관계수 r"]
        C["방향성: 양방향<br>(X↔Y 대칭)"]
    end

    subgraph 회귀분석["회귀분석"]
        D["목적: 관계 수식화 + 예측"]
        E["결과: 회귀식 Y = β₀ + β₁X"]
        F["방향성: 단방향<br>(X → Y)"]
    end

    G[두 변수 데이터] --> 상관분석
    G --> 회귀분석
```

## 12. 강의 구조

```mermaid
gantt
    title 6차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)        :a1, 00:00, 2m
    상관분석 필요성         :a2, after a1, 1m
    상관계수              :a3, after a2, 2m
    상관 ≠ 인과           :a4, after a3, 1m
    선형회귀              :a5, after a4, 2m
    R²와 sklearn          :a6, after a5, 2m

    section 실습편
    실습 소개             :b1, after a6, 2m
    데이터 준비            :b2, after b1, 2m
    산점도 시각화          :b3, after b2, 2m
    상관계수 계산          :b4, after b3, 2m
    회귀 모델 학습         :b5, after b4, 3m
    회귀선 시각화          :b6, after b5, 2m
    모델 평가             :b7, after b6, 2m
    예측하기              :b8, after b7, 3m

    section 정리
    핵심 요약             :c1, after b8, 1.5m
    주의사항/예고          :c2, after c1, 1.5m
```
