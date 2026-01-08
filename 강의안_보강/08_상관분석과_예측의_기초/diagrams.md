# [8차시] 상관분석과 예측의 기초 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["상관계수<br>r = -1~+1"]
    B["선형회귀<br>Y = β₀ + β₁X"]
    C["sklearn<br>예측 구현"]
    D["8차시:<br>데이터 전처리"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. 상관관계의 종류

```mermaid
flowchart TD
    A["상관관계"]

    A --> B["양의 상관"]
    A --> C["음의 상관"]
    A --> D["무상관"]

    B --> B1["X↑ → Y↑<br>r > 0"]
    C --> C1["X↑ → Y↓<br>r < 0"]
    D --> D1["관계 없음<br>r ≈ 0"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#e2e8f0
```

## 3. 상관계수 범위

```mermaid
flowchart LR
    A["-1"]
    B["-0.7"]
    C["0"]
    D["+0.7"]
    E["+1"]

    A --- B --- C --- D --- E

    A --> A1["완벽한<br>음의 상관"]
    C --> C1["선형 관계<br>없음"]
    E --> E1["완벽한<br>양의 상관"]

    style A fill:#fecaca
    style C fill:#e2e8f0
    style E fill:#dcfce7
```

## 4. 상관계수 해석 기준

```mermaid
flowchart TD
    A["|r| 값"]

    A --> B["0.0 ~ 0.3"]
    A --> C["0.3 ~ 0.5"]
    A --> D["0.5 ~ 0.7"]
    A --> E["0.7 ~ 1.0"]

    B --> B1["약한 상관"]
    C --> C1["보통 상관"]
    D --> D1["중간 상관"]
    E --> E1["강한 상관"]

    style E1 fill:#dcfce7
    style D1 fill:#fef3c7
```

## 5. 상관 vs 인과

```mermaid
flowchart TD
    A["상관관계"]
    B["인과관계"]

    A --> A1["함께 변함"]
    B --> B1["원인 → 결과"]

    A1 --> C["분석으로 확인 가능"]
    B1 --> D["실험/추가분석 필요"]

    style A fill:#dbeafe
    style B fill:#fef3c7
    style C fill:#dcfce7
    style D fill:#fecaca
```

## 6. 허위 상관 예시

```mermaid
flowchart TD
    A["아이스크림 판매량"]
    B["익사 사고 수"]
    C["기온 (숨겨진 변수)"]

    C --> A
    C --> B

    A -.->|"높은 상관"| B

    style C fill:#fef3c7
```

## 7. 상관행렬

```mermaid
flowchart TD
    A["상관행렬"]

    A --> B["df.corr()"]
    B --> C["변수 쌍별 상관계수"]
    C --> D["히트맵 시각화"]

    style A fill:#1e40af,color:#fff
    style D fill:#dcfce7
```

## 8. 회귀분석 용어

```mermaid
flowchart LR
    subgraph 입력["입력 (원인)"]
        X["독립변수<br>X"]
    end

    subgraph 출력["출력 (결과)"]
        Y["종속변수<br>Y"]
    end

    X -->|"회귀 모델"| Y

    style X fill:#dbeafe
    style Y fill:#dcfce7
```

## 9. 단순선형회귀 공식

```mermaid
flowchart TD
    A["Y = β₀ + β₁X"]

    A --> B["Y: 종속변수"]
    A --> C["X: 독립변수"]
    A --> D["β₀: 절편"]
    A --> E["β₁: 기울기"]

    D --> D1["X=0일 때 Y값"]
    E --> E1["X가 1 증가할 때<br>Y 변화량"]

    style A fill:#1e40af,color:#fff
```

## 10. 회귀선의 의미

```mermaid
flowchart TD
    subgraph 산점도["데이터와 회귀선"]
        A["● 데이터 점"]
        B["━ 회귀선"]
    end

    산점도 --> C["오차 최소화"]
    C --> D["최적의 직선"]

    style B fill:#fecaca
    style D fill:#dcfce7
```

## 11. 최소제곱법 (OLS)

```mermaid
flowchart TD
    A["최소제곱법<br>OLS"]

    A --> B["목표"]
    B --> C["Σ(실제값 - 예측값)²"]
    C --> D["→ 최소화"]

    D --> E["최적의 β₀, β₁ 도출"]

    style A fill:#1e40af,color:#fff
    style E fill:#dcfce7
```

## 12. 잔차 개념

```mermaid
flowchart LR
    A["실제값 (y)"]
    B["예측값 (ŷ)"]
    C["잔차 = y - ŷ"]

    A --> C
    B --> C

    C --> D["0에 가까울수록 좋음"]

    style C fill:#fef3c7
    style D fill:#dcfce7
```

## 13. R² (결정계수)

```mermaid
flowchart TD
    A["R² = 1 - SSR/SST"]

    A --> B["범위: 0 ~ 1"]
    A --> C["의미"]

    C --> C1["독립변수가<br>종속변수 변동의<br>몇 %를 설명"]

    style A fill:#1e40af,color:#fff
    style C1 fill:#dcfce7
```

## 14. R² 해석 기준

```mermaid
flowchart TD
    A["R² 값"]

    A --> B["0.9+"]
    A --> C["0.7~0.9"]
    A --> D["0.5~0.7"]
    A --> E["<0.5"]

    B --> B1["매우 좋음"]
    C --> C1["좋음"]
    D --> D1["보통"]
    E --> E1["개선 필요"]

    style B1 fill:#dcfce7
    style C1 fill:#dcfce7
    style D1 fill:#fef3c7
    style E1 fill:#fecaca
```

## 15. 회귀 모델 가정

```mermaid
flowchart TD
    A["선형회귀 가정"]

    A --> B["선형성"]
    A --> C["독립성"]
    A --> D["등분산성"]
    A --> E["정규성"]

    B --> B1["X-Y 직선 관계"]
    C --> C1["오차 간 독립"]
    D --> D1["오차 분산 일정"]
    E --> E1["오차 정규분포"]

    style A fill:#1e40af,color:#fff
```

## 16. sklearn 기본 구조

```mermaid
flowchart TD
    A["sklearn 모델"]

    A --> B["fit(X, y)"]
    A --> C["predict(X_new)"]
    A --> D["score(X, y)"]

    B --> B1["학습"]
    C --> C1["예측"]
    D --> D1["평가"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dbeafe
    style C1 fill:#dcfce7
    style D1 fill:#fef3c7
```

## 17. LinearRegression 사용법

```mermaid
flowchart TD
    A["from sklearn.linear_model<br>import LinearRegression"]
    B["model = LinearRegression()"]
    C["model.fit(X, y)"]
    D["model.predict(X_new)"]

    A --> B --> C --> D

    C --> E["model.intercept_<br>(절편)"]
    C --> F["model.coef_<br>(기울기)"]

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 18. X 데이터 형태 주의

```mermaid
flowchart TD
    A["X 데이터"]

    A --> B{1D vs 2D?}

    B -->|"1D (10,)"| C["에러!"]
    B -->|"2D (10,1)"| D["정상"]

    C --> E["reshape(-1, 1)"]
    E --> D

    style C fill:#fecaca
    style D fill:#dcfce7
```

## 19. 모델 평가 지표

```mermaid
flowchart TD
    A["회귀 평가 지표"]

    A --> B["R²"]
    A --> C["MSE"]
    A --> D["RMSE"]
    A --> E["MAE"]

    B --> B1["결정계수<br>0~1"]
    C --> C1["평균제곱오차"]
    D --> D1["√MSE<br>단위 동일"]
    E --> E1["평균절대오차"]

    style A fill:#1e40af,color:#fff
```

## 20. 잔차 분석

```mermaid
flowchart TD
    A["잔차 분석"]

    A --> B["잔차 플롯"]
    A --> C["잔차 히스토그램"]

    B --> B1["패턴 없어야 함<br>(랜덤 분포)"]
    C --> C1["정규분포<br>따라야 함"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#dcfce7
```

## 21. 예측 활용

```mermaid
flowchart TD
    A["model.predict()"]

    A --> B["순방향 예측"]
    A --> C["역방향 계산"]

    B --> B1["온도 85도<br>→ 불량률?"]
    C --> C1["불량률 5%<br>→ 온도 상한?"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
```

## 22. 다중선형회귀 미리보기

```mermaid
flowchart TD
    subgraph 단순["단순선형회귀"]
        A1["X"]
        A2["Y"]
        A1 --> A2
    end

    subgraph 다중["다중선형회귀"]
        B1["X₁"]
        B2["X₂"]
        B3["X₃"]
        B4["Y"]
        B1 --> B4
        B2 --> B4
        B3 --> B4
    end

    style A2 fill:#dcfce7
    style B4 fill:#dcfce7
```

## 23. 실습 흐름

```mermaid
flowchart TD
    A["데이터 준비"]
    B["산점도 확인"]
    C["상관계수 계산"]
    D["선형회귀 학습"]
    E["회귀선 시각화"]
    F["모델 평가"]
    G["예측 수행"]
    H["잔차 분석"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#dbeafe
    style H fill:#dcfce7
```

## 24. 상관분석 코드

```mermaid
flowchart TD
    subgraph NumPy
        A1["np.corrcoef(X, Y)[0,1]"]
    end

    subgraph Pandas
        B1["df['X'].corr(df['Y'])"]
        B2["df.corr() (상관행렬)"]
    end

    style A1 fill:#dcfce7
    style B1 fill:#dcfce7
```

## 25. 선형회귀 코드

```mermaid
flowchart TD
    A["LinearRegression()"]
    B[".fit(X, y)"]
    C[".predict(X_new)"]
    D["r2_score(y, y_pred)"]

    A --> B --> C
    C --> D

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 26. 상관 → 회귀 → 예측

```mermaid
flowchart LR
    A["상관분석"]
    B["회귀분석"]
    C["예측"]

    A -->|"관계 확인"| B
    B -->|"모델 학습"| C

    A --> A1["r = 0.85"]
    B --> B1["Y = 1.5 + 0.05X"]
    C --> C1["X=90 → Y=6.0"]

    style A fill:#dbeafe
    style B fill:#fef3c7
    style C fill:#dcfce7
```

## 27. 핵심 개념 요약

```mermaid
flowchart TD
    A["8차시 핵심"]

    A --> B["상관계수 r"]
    A --> C["선형회귀"]
    A --> D["R²"]

    B --> B1["-1 ~ +1<br>|r|>0.7: 강한상관"]
    C --> C1["Y = β₀ + β₁X"]
    D --> D1["설명력<br>1에 가까울수록 좋음"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dbeafe
    style C1 fill:#fef3c7
    style D1 fill:#dcfce7
```

## 28. 주의사항

```mermaid
flowchart TD
    A["주의사항"]

    A --> B["상관 ≠ 인과"]
    A --> C["X는 2D 배열"]
    A --> D["잔차 분석 필수"]

    B --> B1["숨겨진 변수 주의"]
    C --> C1["reshape(-1, 1)"]
    D --> D1["패턴 없어야 정상"]

    style A fill:#fecaca
    style B1 fill:#fef3c7
    style C1 fill:#fef3c7
    style D1 fill:#fef3c7
```
