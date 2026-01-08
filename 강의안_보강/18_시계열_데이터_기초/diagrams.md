# [18차시] 시계열 데이터 기초 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["16차시<br>하이퍼파라미터"]
    B["18차시<br>시계열 기초"]
    C["18차시<br>시계열 예측"]

    A --> B --> C

    B --> B1["시계열 특성"]
    B --> B2["datetime 처리"]
    B --> B3["전처리 기법"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["18차시: 시계열 데이터 기초"]

    A --> B["대주제 1<br>시계열 특성"]
    A --> C["대주제 2<br>datetime 처리"]
    A --> D["대주제 3<br>전처리 기법"]

    B --> B1["시간 의존성<br>자기상관<br>계절성"]
    C --> C1["pd.to_datetime<br>dt 접근자"]
    D --> D1["resample<br>rolling<br>shift"]

    style A fill:#1e40af,color:#fff
```

## 3. 일반 데이터 vs 시계열

```mermaid
flowchart LR
    subgraph normal["일반 데이터"]
        A1["행이 독립"]
        A2["랜덤 셔플 가능"]
        A3["랜덤 분할 OK"]
    end

    subgraph timeseries["시계열 데이터"]
        B1["순서가 중요"]
        B2["셔플 금지"]
        B3["시간 기준 분할"]
    end

    style normal fill:#dbeafe
    style timeseries fill:#dcfce7
```

## 4. 시계열 특성 3가지

```mermaid
flowchart TD
    A["시계열 특성"]

    A --> B["시간 의존성"]
    B --> B1["현재 ← 과거"]

    A --> C["자기상관"]
    C --> C1["자기와의 상관"]

    A --> D["계절성"]
    D --> D1["주기적 패턴"]

    style A fill:#1e40af,color:#fff
```

## 5. 시계열 구성요소

```mermaid
flowchart TD
    A["시계열 분해"]

    A --> B["추세 (Trend)"]
    B --> B1["장기 증가/감소"]

    A --> C["계절성 (Seasonality)"]
    C --> C1["주기적 반복"]

    A --> D["잔차 (Residual)"]
    D --> D1["불규칙 변동"]

    style A fill:#1e40af,color:#fff
```

## 6. 추세 예시

```mermaid
flowchart LR
    A["추세 예시"]

    A --> B["상승 추세"]
    B --> B1["설비 개선<br>생산량 증가"]

    A --> C["하락 추세"]
    C --> C1["설비 노후화<br>불량률 증가"]

    style A fill:#1e40af,color:#fff
```

## 7. 계절성 예시

```mermaid
flowchart TD
    A["계절성 예시"]

    A --> B["일별"]
    B --> B1["오전 생산 높음"]

    A --> C["주별"]
    C --> C1["월요일 불량 높음"]

    A --> D["연별"]
    D --> D1["여름 에너지 증가"]

    style A fill:#1e40af,color:#fff
```

## 8. 데이터 분할 비교

```mermaid
flowchart TD
    A["데이터 분할"]

    A --> B["일반 ML"]
    B --> B1["랜덤 분할 OK"]

    A --> C["시계열"]
    C --> C1["시간 기준 분할 필수"]
    C --> C2["과거 → 학습<br>미래 → 테스트"]

    style C1 fill:#dcfce7
```

## 9. 시간 기준 분할

```mermaid
flowchart LR
    A["전체 데이터"]
    B["학습 70%"]
    C["검증 15%"]
    D["테스트 15%"]

    A --> B --> C --> D

    B --> B1["과거"]
    D --> D1["미래"]

    style B fill:#dbeafe
    style D fill:#dcfce7
```

## 10. 데이터 누출 문제

```mermaid
flowchart TD
    A["데이터 누출"]

    A --> B["랜덤 분할"]
    B --> B1["미래 데이터가<br>학습에 포함"]
    B --> B2["성능 과대평가"]
    B --> B3["실전에서 실패"]

    style B3 fill:#fecaca
```

## 11. datetime 모듈

```mermaid
flowchart TD
    A["datetime 모듈"]

    A --> B["datetime"]
    B --> B1["날짜+시간 객체"]

    A --> C["timedelta"]
    C --> C1["시간 차이 계산"]

    A --> D["strptime"]
    D --> D1["문자열 → datetime"]

    A --> E["strftime"]
    E --> E1["datetime → 문자열"]

    style A fill:#1e40af,color:#fff
```

## 12. Pandas 날짜 처리

```mermaid
flowchart TD
    A["Pandas 날짜"]

    A --> B["pd.to_datetime()"]
    B --> B1["문자열 → datetime"]

    A --> C["pd.date_range()"]
    C --> C1["날짜 범위 생성"]

    A --> D["DatetimeIndex"]
    D --> D1["날짜 인덱스"]

    style A fill:#1e40af,color:#fff
```

## 13. dt 접근자

```mermaid
flowchart TD
    A["dt 접근자"]

    A --> B["dt.year"]
    B --> B1["연도"]

    A --> C["dt.month"]
    C --> C1["월"]

    A --> D["dt.day"]
    D --> D1["일"]

    A --> E["dt.hour"]
    E --> E1["시"]

    A --> F["dt.dayofweek"]
    F --> F1["요일 (0=월)"]

    style A fill:#1e40af,color:#fff
```

## 14. DatetimeIndex 장점

```mermaid
flowchart TD
    A["DatetimeIndex"]

    A --> B["기간 슬라이싱"]
    B --> B1["df['2025-01']"]

    A --> C["resample 사용"]
    C --> C1["df.resample('D')"]

    A --> D["시각화 자동"]
    D --> D1["X축 날짜 표시"]

    style A fill:#1e40af,color:#fff
```

## 15. resample 개념

```mermaid
flowchart LR
    A["분별 데이터"]
    B["resample('H')"]
    C["시간별 데이터"]

    A --> B --> C

    B --> B1["집계 방법<br>mean, sum, max"]

    style B fill:#1e40af,color:#fff
```

## 16. resample freq 옵션

```mermaid
flowchart TD
    A["freq 옵션"]

    A --> B["'H'"]
    B --> B1["시간별"]

    A --> C["'D'"]
    C --> C1["일별"]

    A --> D["'W'"]
    D --> D1["주별"]

    A --> E["'M'"]
    E --> E1["월별"]

    style A fill:#1e40af,color:#fff
```

## 17. rolling 개념

```mermaid
flowchart LR
    A["원본 데이터"]
    B["rolling(7)"]
    C["7일 이동평균"]

    A --> B --> C

    B --> B1["윈도우 이동<br>통계 계산"]

    style B fill:#1e40af,color:#fff
```

## 18. rolling 시각화

```mermaid
flowchart TD
    A["원본: 10, 20, 15, 30, 25"]

    A --> B["window=3"]

    B --> C["Day 1-3"]
    C --> C1["(10+20+15)/3 = 15"]

    B --> D["Day 2-4"]
    D --> D1["(20+15+30)/3 = 21.7"]

    B --> E["Day 3-5"]
    E --> E1["(15+30+25)/3 = 23.3"]

    style A fill:#dbeafe
```

## 19. rolling 옵션

```mermaid
flowchart TD
    A["rolling 옵션"]

    A --> B["window"]
    B --> B1["윈도우 크기"]

    A --> C["min_periods"]
    C --> C1["최소 데이터 수<br>NaN 방지"]

    A --> D["center"]
    D --> D1["중앙 정렬"]

    style A fill:#1e40af,color:#fff
```

## 20. shift 개념

```mermaid
flowchart TD
    A["shift(1)"]

    A --> B["원본"]
    B --> B1["[A, B, C, D, E]"]

    A --> C["결과"]
    C --> C1["[NaN, A, B, C, D]"]

    A --> D["의미"]
    D --> D1["1칸 뒤로<br>(과거 값)"]

    style A fill:#1e40af,color:#fff
```

## 21. shift 방향

```mermaid
flowchart TD
    A["shift 방향"]

    A --> B["shift(1)"]
    B --> B1["과거 방향<br>Lag 특성"]

    A --> C["shift(-1)"]
    C --> C1["미래 방향<br>⚠️ 주의!"]

    style B1 fill:#dcfce7
    style C1 fill:#fecaca
```

## 22. Lag 특성 생성

```mermaid
flowchart TD
    A["Lag 특성"]

    A --> B["lag_1"]
    B --> B1["1시간 전 값"]

    A --> C["lag_24"]
    C --> C1["24시간 전 값"]

    A --> D["lag_168"]
    D --> D1["7일 전 값"]

    style A fill:#1e40af,color:#fff
```

## 23. 데이터 누출 방지

```mermaid
flowchart TD
    A["Rolling 특성"]

    A --> B["❌ 잘못된 방법"]
    B --> B1["rolling(7).mean()"]
    B --> B2["오늘 값 포함!"]

    A --> C["✅ 올바른 방법"]
    C --> C1["shift(1).rolling(7).mean()"]
    C --> C2["어제까지만 사용"]

    style B2 fill:#fecaca
    style C2 fill:#dcfce7
```

## 24. 특성 엔지니어링 순서

```mermaid
flowchart LR
    A["원본"]
    B["shift(1)"]
    C["rolling(7)"]
    D["mean()"]
    E["ma_7 특성"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 25. 날짜 특성 추출

```mermaid
flowchart TD
    A["날짜에서 추출"]

    A --> B["hour"]
    B --> B1["시간대 효과"]

    A --> C["dayofweek"]
    C --> C1["요일 효과"]

    A --> D["month"]
    D --> D1["월별 효과"]

    A --> E["is_weekend"]
    E --> E1["주말 효과"]

    style A fill:#1e40af,color:#fff
```

## 26. diff 개념

```mermaid
flowchart TD
    A["diff()"]

    A --> B["원본"]
    B --> B1["[100, 105, 103]"]

    A --> C["diff(1)"]
    C --> C1["[NaN, 5, -2]"]

    A --> D["의미"]
    D --> D1["연속 값의 차이<br>변화량"]

    style A fill:#1e40af,color:#fff
```

## 27. pct_change 개념

```mermaid
flowchart TD
    A["pct_change()"]

    A --> B["원본"]
    B --> B1["[100, 110, 99]"]

    A --> C["결과"]
    C --> C1["[NaN, 0.10, -0.10]"]

    A --> D["의미"]
    D --> D1["변화율<br>10% 증가, 10% 감소"]

    style A fill:#1e40af,color:#fff
```

## 28. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 생성"]
    B["2. datetime 변환"]
    C["3. 날짜 특성 추출"]
    D["4. resample"]
    E["5. rolling, shift"]
    F["6. 시간 분할"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 29. 전처리 기법 요약

```mermaid
flowchart TD
    A["시계열 전처리"]

    A --> B["resample()"]
    B --> B1["주기 변환"]

    A --> C["rolling()"]
    C --> C1["이동 통계"]

    A --> D["shift()"]
    D --> D1["Lag 특성"]

    A --> E["diff()"]
    E --> E1["차분"]

    style A fill:#1e40af,color:#fff
```

## 30. TimeSeriesSplit

```mermaid
flowchart TD
    A["TimeSeriesSplit"]

    A --> B["Fold 1"]
    B --> B1["[Train] [Val]"]

    A --> C["Fold 2"]
    C --> C1["[Train___] [Val]"]

    A --> D["Fold 3"]
    D --> D1["[Train______] [Val]"]

    style A fill:#1e40af,color:#fff
```

## 31. 핵심 정리

```mermaid
flowchart TD
    A["18차시 핵심"]

    A --> B["시계열 특성"]
    B --> B1["순서 중요<br>시간 기준 분할"]

    A --> C["datetime 처리"]
    C --> C1["pd.to_datetime<br>dt 접근자"]

    A --> D["전처리 기법"]
    D --> D1["resample<br>rolling<br>shift"]

    style A fill:#1e40af,color:#fff
```

## 32. shift 필수 규칙

```mermaid
flowchart TD
    A["shift(1) 필수"]

    A --> B["Lag 특성"]
    B --> B1["shift(n)으로 생성"]

    A --> C["Rolling 특성"]
    C --> C1["shift(1) 후 rolling"]

    A --> D["미래 누출 방지"]
    D --> D1["현재/미래 값 제외"]

    style A fill:#fecaca
```

## 33. 다음 차시 연결

```mermaid
flowchart LR
    A["18차시<br>시계열 기초"]
    B["18차시<br>시계열 예측"]

    A --> B

    A --> A1["전처리<br>특성 추출"]
    B --> B1["ML 모델<br>예측 실습"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 34. 제조 데이터 시계열

```mermaid
flowchart TD
    A["제조 시계열"]

    A --> B["센서 온도"]
    B --> B1["초/분 단위"]

    A --> C["생산량"]
    C --> C1["시간/일 단위"]

    A --> D["불량률"]
    D --> D1["일/주 단위"]

    A --> E["설비 로그"]
    E --> E1["이벤트 기반"]

    style A fill:#1e40af,color:#fff
```

## 35. 특성 엔지니어링 전략

```mermaid
flowchart TD
    A["시계열 특성"]

    A --> B["날짜 특성"]
    B --> B1["hour, dayofweek<br>month, is_weekend"]

    A --> C["Lag 특성"]
    C --> C1["shift(1,24,168)"]

    A --> D["Rolling 특성"]
    D --> D1["ma, std, max, min"]

    A --> E["변화 특성"]
    E --> E1["diff, pct_change"]

    style A fill:#1e40af,color:#fff
```

## 36. NaN 처리

```mermaid
flowchart TD
    A["NaN 발생"]

    A --> B["shift(n)"]
    B --> B1["첫 n개 NaN"]

    A --> C["rolling(n)"]
    C --> C1["첫 n-1개 NaN"]

    A --> D["처리 방법"]
    D --> D1["dropna()"]
    D --> D2["fillna()"]

    style A fill:#1e40af,color:#fff
```

## 37. 전체 워크플로우

```mermaid
flowchart TD
    A["시계열 데이터"]
    B["datetime 변환"]
    C["특성 추출"]
    D["전처리 기법"]
    E["NaN 처리"]
    F["시간 기준 분할"]
    G["모델 학습"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style G fill:#dcfce7
```

## 38. 주의사항 정리

```mermaid
flowchart TD
    A["시계열 주의사항"]

    A --> B["분할"]
    B --> B1["시간 기준 필수<br>랜덤 분할 금지"]

    A --> C["특성"]
    C --> C1["shift(1) 먼저<br>미래 누출 방지"]

    A --> D["검증"]
    D --> D1["TimeSeriesSplit<br>순차 검증"]

    style A fill:#fecaca
```
