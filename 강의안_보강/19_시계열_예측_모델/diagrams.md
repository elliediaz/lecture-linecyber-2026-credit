# [19차시] 시계열 예측 모델 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["17차시<br>시계열 기초"]
    B["19차시<br>시계열 예측"]
    C["19차시<br>딥러닝 입문"]

    A --> B --> C

    B --> B1["특성 엔지니어링"]
    B --> B2["ML 예측"]
    B --> B3["평가"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["19차시: 시계열 예측 모델"]

    A --> B["대주제 1<br>특성 엔지니어링"]
    A --> C["대주제 2<br>ML 모델 적용"]
    A --> D["대주제 3<br>예측 평가"]

    B --> B1["날짜, Lag, Rolling"]
    C --> C1["RandomForest"]
    D --> D1["MAE, RMSE, MAPE"]

    style A fill:#1e40af,color:#fff
```

## 3. 특성 엔지니어링 종류

```mermaid
flowchart TD
    A["시계열 특성"]

    A --> B["날짜 특성"]
    B --> B1["hour, dayofweek"]

    A --> C["Lag 특성"]
    C --> C1["shift(1), shift(24)"]

    A --> D["Rolling 특성"]
    D --> D1["ma, std, max, min"]

    A --> E["변화 특성"]
    E --> E1["diff, pct_change"]

    A --> F["외부 특성"]
    F --> F1["날씨, 휴일"]

    style A fill:#1e40af,color:#fff
```

## 4. 날짜 특성 추출

```mermaid
flowchart TD
    A["timestamp"]

    A --> B["dt.hour"]
    B --> B1["0~23"]

    A --> C["dt.dayofweek"]
    C --> C1["0=월~6=일"]

    A --> D["dt.month"]
    D --> D1["1~12"]

    A --> E["is_weekend"]
    E --> E1["0 or 1"]

    style A fill:#1e40af,color:#fff
```

## 5. 주기적 인코딩

```mermaid
flowchart TD
    A["hour 인코딩"]

    A --> B["문제"]
    B --> B1["23시와 0시가<br>숫자로 23 차이"]

    A --> C["해결"]
    C --> C1["sin/cos 인코딩"]
    C --> C2["23시와 0시가<br>원형으로 연결"]

    style A fill:#1e40af,color:#fff
```

## 6. Lag 특성 생성

```mermaid
flowchart TD
    A["Lag 특성"]

    A --> B["lag_1"]
    B --> B1["shift(1)<br>1시간 전"]

    A --> C["lag_24"]
    C --> C1["shift(24)<br>24시간 전"]

    A --> D["lag_168"]
    D --> D1["shift(168)<br>7일 전"]

    style A fill:#1e40af,color:#fff
```

## 7. Rolling 특성 생성

```mermaid
flowchart TD
    A["Rolling 특성"]

    A --> B["⚠️ shift 먼저!"]
    B --> B1["shift(1).rolling(24)"]

    A --> C["종류"]
    C --> C1["mean - 이동평균"]
    C --> C2["std - 이동표준편차"]
    C --> C3["max/min - 이동 최대/최소"]

    style B fill:#fecaca
```

## 8. 미래 누출 방지

```mermaid
flowchart TD
    A["Rolling 특성"]

    A --> B["❌ 잘못"]
    B --> B1["rolling(24).mean()"]
    B --> B2["현재 값 포함!"]

    A --> C["✅ 올바름"]
    C --> C1["shift(1).rolling(24).mean()"]
    C --> C2["어제까지만 사용"]

    style B2 fill:#fecaca
    style C2 fill:#dcfce7
```

## 9. ML 예측 흐름

```mermaid
flowchart TD
    A["1. 특성 엔지니어링"]
    B["2. NaN 제거"]
    C["3. 시간 분할"]
    D["4. 모델 학습"]
    E["5. 예측"]
    F["6. 평가"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 10. RandomForest 적용

```mermaid
flowchart TD
    A["RandomForestRegressor"]

    A --> B["장점"]
    B --> B1["비선형 학습"]
    B --> B2["특성 중요도"]
    B --> B3["과적합에 강함"]

    A --> C["단점"]
    C --> C1["순차 의존성 직접 학습 X"]
    C --> C2["→ Lag 특성으로 보완"]

    style A fill:#1e40af,color:#fff
```

## 11. 모델 비교

```mermaid
flowchart TD
    A["시계열 예측 모델"]

    A --> B["RandomForest"]
    B --> B1["강건, 해석 쉬움"]

    A --> C["XGBoost"]
    C --> C1["높은 성능"]

    A --> D["LightGBM"]
    D --> D1["대용량 빠름"]

    style A fill:#1e40af,color:#fff
```

## 12. TimeSeriesSplit

```mermaid
flowchart TD
    A["TimeSeriesSplit"]

    A --> B["Fold 1"]
    B --> B1["학습 | 검증"]

    A --> C["Fold 2"]
    C --> C1["학습___ | 검증"]

    A --> D["Fold 3"]
    D --> D1["학습______ | 검증"]

    style A fill:#1e40af,color:#fff
```

## 13. 다단계 예측

```mermaid
flowchart TD
    A["다단계 예측"]

    A --> B["Recursive"]
    B --> B1["예측값 → 다음 입력"]
    B --> B2["오차 누적"]

    A --> C["Direct"]
    C --> C1["horizon별 별도 모델"]
    C --> C2["독립적"]

    style A fill:#1e40af,color:#fff
```

## 14. 평가 지표

```mermaid
flowchart TD
    A["시계열 평가 지표"]

    A --> B["MAE"]
    B --> B1["평균 절대 오차"]
    B --> B2["해석 쉬움"]

    A --> C["RMSE"]
    C --> C1["평균 제곱근 오차"]
    C --> C2["큰 오차 패널티"]

    A --> D["MAPE"]
    D --> D1["평균 절대 비율 오차"]
    D --> D2["% 단위"]

    style A fill:#1e40af,color:#fff
```

## 15. MAE 개념

```mermaid
flowchart TD
    A["MAE"]

    A --> B["공식"]
    B --> B1["Σ|y - ŷ| / n"]

    A --> C["특징"]
    C --> C1["단위 = 원본 단위"]
    C --> C2["해석 쉬움"]

    A --> D["예시"]
    D --> D1["MAE=50<br>→ 평균 50 오차"]

    style A fill:#1e40af,color:#fff
```

## 16. RMSE 개념

```mermaid
flowchart TD
    A["RMSE"]

    A --> B["공식"]
    B --> B1["√(Σ(y-ŷ)² / n)"]

    A --> C["특징"]
    C --> C1["큰 오차에 패널티"]
    C --> C2["이상치에 민감"]

    A --> D["활용"]
    D --> D1["큰 오차가 중요할 때"]

    style A fill:#1e40af,color:#fff
```

## 17. MAPE 개념

```mermaid
flowchart TD
    A["MAPE"]

    A --> B["공식"]
    B --> B1["Σ|y-ŷ|/y / n × 100"]

    A --> C["특징"]
    C --> C1["% 단위"]
    C --> C2["직관적"]

    A --> D["주의"]
    D --> D1["y=0이면 계산 불가"]

    style A fill:#1e40af,color:#fff
```

## 18. 지표 선택 가이드

```mermaid
flowchart TD
    A["지표 선택"]

    A --> B["일반적"]
    B --> B1["MAE, RMSE"]

    A --> C["큰 오차 중요"]
    C --> C1["RMSE"]

    A --> D["상대적 오차"]
    D --> D1["MAPE"]

    A --> E["비즈니스 보고"]
    E --> E1["MAPE (%)"]

    style A fill:#1e40af,color:#fff
```

## 19. 시각적 평가

```mermaid
flowchart TD
    A["시각적 평가"]

    A --> B["실제 vs 예측"]
    B --> B1["시계열 플롯"]
    B --> B2["전체 흐름 확인"]

    A --> C["잔차 분석"]
    C --> C1["잔차 분포"]
    C --> C2["시간별 잔차"]

    style A fill:#1e40af,color:#fff
```

## 20. 잔차 분석

```mermaid
flowchart TD
    A["잔차 분석"]

    A --> B["잔차 분포"]
    B --> B1["0 중심 분포?"]

    A --> C["시간별 잔차"]
    C --> C1["패턴 없음?"]

    A --> D["좋은 모델"]
    D --> D1["랜덤 잔차"]

    style A fill:#1e40af,color:#fff
```

## 21. 특성 중요도

```mermaid
flowchart TD
    A["특성 중요도"]

    A --> B["높음"]
    B --> B1["lag_1, lag_24"]
    B --> B2["직접적 영향"]

    A --> C["중간"]
    C --> C1["hour, dayofweek"]
    C --> C2["패턴 기여"]

    A --> D["낮음"]
    D --> D1["제거 고려"]

    style A fill:#1e40af,color:#fff
```

## 22. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 생성"]
    B["2. 특성 엔지니어링"]
    C["3. 시간 분할"]
    D["4. 모델 학습"]
    E["5. 예측 및 평가"]
    F["6. 시각화"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 23. 예측 성능 기준

```mermaid
flowchart TD
    A["MAPE 기준"]

    A --> B["< 5%"]
    B --> B1["매우 좋음"]

    A --> C["5~10%"]
    C --> C1["좋음"]

    A --> D["10~20%"]
    D --> D1["보통"]

    A --> E["> 20%"]
    E --> E1["개선 필요"]

    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#fecaca
```

## 24. sklearn 함수 정리

```mermaid
flowchart TD
    A["sklearn 함수"]

    A --> B["모델"]
    B --> B1["RandomForestRegressor"]

    A --> C["분할"]
    C --> C1["TimeSeriesSplit"]

    A --> D["평가"]
    D --> D1["mean_absolute_error"]
    D --> D2["mean_squared_error"]

    style A fill:#1e40af,color:#fff
```

## 25. 예측 구간

```mermaid
flowchart TD
    A["예측 구간"]

    A --> B["점 예측"]
    B --> B1["단일 값"]

    A --> C["구간 예측"]
    C --> C1["상한/하한"]
    C --> C2["불확실성 표현"]

    A --> D["계산"]
    D --> D1["예측 ± 1.96 × std"]

    style A fill:#1e40af,color:#fff
```

## 26. 변화 특성

```mermaid
flowchart TD
    A["변화 특성"]

    A --> B["diff"]
    B --> B1["절대 변화량"]

    A --> C["pct_change"]
    C --> C1["상대 변화율"]

    A --> D["활용"]
    D --> D1["변동성 파악"]

    style A fill:#1e40af,color:#fff
```

## 27. 외부 특성

```mermaid
flowchart TD
    A["외부 특성"]

    A --> B["날씨"]
    B --> B1["온도, 습도"]

    A --> C["캘린더"]
    C --> C1["휴일, 이벤트"]

    A --> D["경제"]
    D --> D1["지표, 환율"]

    style A fill:#1e40af,color:#fff
```

## 28. 핵심 정리

```mermaid
flowchart TD
    A["19차시 핵심"]

    A --> B["특성 엔지니어링"]
    B --> B1["shift(1) 필수"]

    A --> C["ML 예측"]
    C --> C1["RandomForest<br>TimeSeriesSplit"]

    A --> D["평가"]
    D --> D1["MAE, MAPE<br>시각화"]

    style A fill:#1e40af,color:#fff
```

## 29. 과적합 방지

```mermaid
flowchart TD
    A["과적합 방지"]

    A --> B["특성 선택"]
    B --> B1["중요도 낮은 특성 제거"]

    A --> C["교차검증"]
    C --> C1["TimeSeriesSplit"]

    A --> D["정규화"]
    D --> D1["max_depth 제한"]

    style A fill:#1e40af,color:#fff
```

## 30. 다음 차시 연결

```mermaid
flowchart LR
    A["19차시<br>시계열 예측"]
    B["19차시<br>딥러닝 입문"]

    A --> B

    A --> A1["ML 기반 예측"]
    B --> B1["신경망 기초"]
    B --> B2["딥러닝 개념"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 31. 전체 워크플로우

```mermaid
flowchart TD
    A["원본 시계열"]
    B["특성 엔지니어링"]
    C["데이터 분할"]
    D["모델 학습"]
    E["예측"]
    F["평가"]
    G["배포"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style G fill:#dcfce7
```

## 32. 비율 특성

```mermaid
flowchart TD
    A["비율 특성"]

    A --> B["이동평균 대비"]
    B --> B1["value / ma"]

    A --> C["최대값 대비"]
    C --> C1["value / max"]

    A --> D["범위 내 위치"]
    D --> D1["(value-min)/(max-min)"]

    style A fill:#1e40af,color:#fff
```

## 33. 모델 선택 전략

```mermaid
flowchart TD
    A["모델 선택"]

    A --> B["데이터 적음"]
    B --> B1["RandomForest"]

    A --> C["데이터 많음"]
    C --> C1["XGBoost, LightGBM"]

    A --> D["해석 필요"]
    D --> D1["LinearRegression<br>DecisionTree"]

    style A fill:#1e40af,color:#fff
```

## 34. 예측 개선 전략

```mermaid
flowchart TD
    A["예측 개선"]

    A --> B["특성 추가"]
    B --> B1["더 많은 Lag<br>외부 변수"]

    A --> C["모델 튜닝"]
    C --> C1["하이퍼파라미터<br>최적화"]

    A --> D["앙상블"]
    D --> D1["여러 모델 결합"]

    style A fill:#1e40af,color:#fff
```

## 35. 주의사항 정리

```mermaid
flowchart TD
    A["시계열 예측 주의"]

    A --> B["데이터 누출"]
    B --> B1["shift(1) 필수"]

    A --> C["시간 분할"]
    C --> C1["랜덤 분할 금지"]

    A --> D["과적합"]
    D --> D1["교차검증 필수"]

    style A fill:#fecaca
```
