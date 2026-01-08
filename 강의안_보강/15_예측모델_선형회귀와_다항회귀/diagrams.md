# [15차시] 예측 모델 - 선형/다항회귀 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["회귀 문제<br>이해"]
    B["선형회귀"]
    C["다항회귀"]
    D["15차시:<br>모델 평가"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. 분류 vs 회귀

```mermaid
flowchart LR
    subgraph 분류["분류 (Classification)"]
        A1["출력: 범주"]
        A2["정상/불량"]
        A3["~인가요?"]
    end

    subgraph 회귀["회귀 (Regression)"]
        B1["출력: 숫자"]
        B2["1,247개"]
        B3["얼마나?"]
    end

    style 분류 fill:#dbeafe
    style 회귀 fill:#dcfce7
```

## 3. 회귀 문제 예시

```mermaid
flowchart TD
    A["회귀 문제 예시"]

    A --> B["생산량 예측"]
    B --> B1["1,247개"]

    A --> C["불량률 예측"]
    C --> C1["3.2%"]

    A --> D["설비 수명"]
    D --> D1["87일"]

    A --> E["소요 시간"]
    E --> E1["4.5시간"]

    style A fill:#1e40af,color:#fff
```

## 4. 회귀 평가 지표

```mermaid
flowchart TD
    A["회귀 평가 지표"]

    A --> B["MSE"]
    B --> B1["평균 제곱 오차"]
    B --> B2["작을수록 좋음"]

    A --> C["RMSE"]
    C --> C1["MSE의 제곱근"]
    C --> C2["해석 쉬움"]

    A --> D["R² 점수"]
    D --> D1["결정계수"]
    D --> D2["1에 가까울수록 좋음"]

    style A fill:#1e40af,color:#fff
```

## 5. R² 점수 해석

```mermaid
flowchart TD
    A["R² 점수 해석"]

    A --> B["1.0"]
    B --> B1["완벽한 예측"]

    A --> C["0.8+"]
    C --> C1["좋음"]

    A --> D["0.5~0.8"]
    D --> D1["보통"]

    A --> E["0.3 이하"]
    E --> E1["약함"]

    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#fecaca
```

## 6. 선형회귀 수식

```mermaid
flowchart TD
    A["선형회귀"]

    A --> B["수식"]
    B --> B1["y = wx + b"]

    A --> C["구성요소"]
    C --> C1["w: 기울기"]
    C --> C2["b: 절편"]
    C --> C3["x: 입력"]
    C --> C4["y: 예측값"]

    style A fill:#1e40af,color:#fff
```

## 7. 기울기와 절편

```mermaid
flowchart TD
    A["기울기와 절편"]

    A --> B["기울기 (w)"]
    B --> B1["x 1 증가 시<br>y 변화량"]
    B --> B2["w=-3: 온도 1↑<br>생산량 3↓"]

    A --> C["절편 (b)"]
    C --> C1["x=0일 때<br>y 값"]
    C --> C2["기준점"]

    style A fill:#1e40af,color:#fff
```

## 8. 다중 선형회귀

```mermaid
flowchart TD
    A["다중 선형회귀"]

    A --> B["수식"]
    B --> B1["y = w₁x₁ + w₂x₂ +<br>w₃x₃ + b"]

    A --> C["예시"]
    C --> C1["생산량 = 5×속도<br>- 3×온도 + 1000"]

    A --> D["해석"]
    D --> D1["속도 1↑ → 생산 5↑"]
    D --> D2["온도 1↑ → 생산 3↓"]

    style A fill:#1e40af,color:#fff
```

## 9. 최소제곱법

```mermaid
flowchart TD
    A["최소제곱법 (OLS)"]

    A --> B["목표"]
    B --> B1["오차 제곱합 최소화"]

    A --> C["오차"]
    C --> C1["실제값 - 예측값"]

    A --> D["결과"]
    D --> D1["최적의 직선"]

    style A fill:#1e40af,color:#fff
```

## 10. sklearn LinearRegression

```mermaid
flowchart TD
    A["LinearRegression"]

    A --> B["메서드"]
    B --> B1["fit(X, y)"]
    B --> B2["predict(X)"]
    B --> B3["score(X, y)"]

    A --> C["속성"]
    C --> C1["coef_: 계수"]
    C --> C2["intercept_: 절편"]

    style A fill:#1e40af,color:#fff
```

## 11. 선형회귀 흐름

```mermaid
flowchart TD
    A["1. 모델 생성"]
    B["2. 학습 (fit)"]
    C["3. 예측 (predict)"]
    D["4. 평가 (score)"]
    E["5. 계수 확인"]

    A --> B --> C --> D --> E

    A --> A1["LinearRegression()"]
    B --> B1["model.fit(X, y)"]
    C --> C1["model.predict(X)"]
    D --> D1["R² 점수"]
    E --> E1["coef_, intercept_"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 12. 비선형 관계

```mermaid
flowchart TD
    A["비선형 관계"]

    A --> B["문제"]
    B --> B1["데이터가 곡선"]
    B --> B2["직선으로 한계"]

    A --> C["해결"]
    C --> C1["다항회귀"]
    C --> C2["특성 거듭제곱 추가"]

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 13. 다항회귀

```mermaid
flowchart TD
    A["다항회귀"]

    A --> B["2차"]
    B --> B1["y = w₁x + w₂x² + b"]

    A --> C["3차"]
    C --> C1["y = w₁x + w₂x²<br>+ w₃x³ + b"]

    A --> D["특징"]
    D --> D1["곡선 관계 학습"]

    style A fill:#1e40af,color:#fff
```

## 14. PolynomialFeatures

```mermaid
flowchart TD
    A["PolynomialFeatures"]

    A --> B["입력"]
    B --> B1["[x]"]

    A --> C["degree=2"]
    C --> C1["[1, x, x²]"]

    A --> D["degree=3"]
    D --> D1["[1, x, x², x³]"]

    style A fill:#1e40af,color:#fff
```

## 15. 다항회귀 파이프라인

```mermaid
flowchart LR
    A["입력 X"]
    B["PolynomialFeatures"]
    C["LinearRegression"]
    D["예측 y"]

    A --> B --> C --> D

    B --> B1["다항 특성 생성"]
    C --> C1["선형회귀 학습"]

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 16. degree의 영향

```mermaid
flowchart TD
    A["degree (차수)"]

    A --> B["degree=1"]
    B --> B1["직선<br>과소적합 가능"]

    A --> C["degree=2~3"]
    C --> C1["적절한 곡선"]

    A --> D["degree=10+"]
    D --> D1["복잡한 곡선<br>과대적합 위험"]

    style C fill:#dcfce7
    style D fill:#fecaca
```

## 17. 과대적합 징후

```mermaid
flowchart TD
    A["과대적합 징후"]

    A --> B["학습 점수"]
    B --> B1["높음 (0.99)"]

    A --> C["테스트 점수"]
    C --> C1["낮음 (0.60)"]

    A --> D["갭"]
    D --> D1["큰 차이 = 과대적합"]

    style B fill:#dcfce7
    style C fill:#fecaca
    style D fill:#fef3c7
```

## 18. 차수 선택 전략

```mermaid
flowchart TD
    A["차수 선택 전략"]

    A --> B["1. 낮은 차수부터"]
    B --> B1["degree=2 시작"]

    A --> C["2. 성능 비교"]
    C --> C1["학습/테스트 점수"]

    A --> D["3. 과대적합 확인"]
    D --> D1["갭이 크면 낮춤"]

    style A fill:#1e40af,color:#fff
```

## 19. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 생성"]
    B["2. 선형회귀"]
    C["3. 다항회귀"]
    D["4. 차수 비교"]
    E["5. 최종 예측"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 20. 선형 vs 다항 비교

```mermaid
flowchart LR
    subgraph linear["선형회귀"]
        A1["직선"]
        A2["해석 쉬움"]
        A3["단순"]
    end

    subgraph poly["다항회귀"]
        B1["곡선"]
        B2["복잡한 관계"]
        B3["과대적합 주의"]
    end

    style linear fill:#dbeafe
    style poly fill:#dcfce7
```

## 21. 언제 사용?

```mermaid
flowchart TD
    A["모델 선택"]

    A --> B["선형회귀"]
    B --> B1["직선 관계"]
    B --> B2["해석 중요"]
    B --> B3["빠른 학습"]

    A --> C["다항회귀"]
    C --> C1["곡선 관계"]
    C --> C2["degree 2~3"]
    C --> C3["과대적합 주의"]

    style A fill:#1e40af,color:#fff
```

## 22. 예측 시각화

```mermaid
flowchart TD
    A["예측 시각화"]

    A --> B["실제 vs 예측"]
    B --> B1["산점도"]
    B --> B2["대각선에 가까울수록<br>정확"]

    A --> C["잔차 분포"]
    C --> C1["히스토그램"]
    C --> C2["0 근처 집중되면<br>좋음"]

    style A fill:#1e40af,color:#fff
```

## 23. 계수 해석

```mermaid
flowchart TD
    A["계수 해석"]

    A --> B["온도: -3.1"]
    B --> B1["온도 1↑<br>생산량 3.1↓"]

    A --> C["속도: 5.2"]
    C --> C1["속도 1↑<br>생산량 5.2↑"]

    A --> D["인사이트"]
    D --> D1["온도 관리 중요"]

    style A fill:#1e40af,color:#fff
```

## 24. 새 데이터 예측

```mermaid
flowchart TD
    A["새 데이터"]

    A --> B["입력"]
    B --> B1["온도: 85"]
    B --> B2["습도: 50"]
    B --> B3["속도: 100"]

    A --> C["예측"]
    C --> C1["생산량: 1,240개"]

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 25. 핵심 정리

```mermaid
flowchart TD
    A["15차시 핵심"]

    A --> B["회귀"]
    B --> B1["숫자 예측<br>MSE, R²"]

    A --> C["선형회귀"]
    C --> C1["y = wx + b<br>직선"]

    A --> D["다항회귀"]
    D --> D1["곡선 관계<br>과대적합 주의"]

    style A fill:#1e40af,color:#fff
```

## 26. 다음 차시 연결

```mermaid
flowchart LR
    A["15차시<br>선형/다항회귀"]
    B["15차시<br>모델 평가"]

    A --> B

    A --> A1["기본 평가"]
    B --> B1["교차검증"]
    B --> B2["더 정확한 평가"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
