# [9차시] 제조 데이터 전처리 (1) - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["결측치<br>탐지/처리"]
    B["이상치<br>탐지 방법"]
    C["전처리<br>전략 선택"]
    D["9차시:<br>전처리 2"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. 전처리의 중요성

```mermaid
flowchart TD
    A["데이터 분석 프로젝트"]

    A --> B["전처리<br>60-80%"]
    A --> C["모델링<br>10-20%"]
    A --> D["평가/배포<br>10-20%"]

    B --> B1["결측치 처리"]
    B --> B2["이상치 처리"]
    B --> B3["스케일링/인코딩"]

    style B fill:#fecaca
    style B1 fill:#fef3c7
    style B2 fill:#fef3c7
    style B3 fill:#fef3c7
```

## 3. Garbage In, Garbage Out

```mermaid
flowchart LR
    subgraph 입력["입력 데이터"]
        A1["깨끗한 데이터"]
        A2["더러운 데이터"]
    end

    subgraph 모델["ML 모델"]
        B["동일한 알고리즘"]
    end

    subgraph 출력["결과"]
        C1["좋은 예측"]
        C2["나쁜 예측"]
    end

    A1 --> B --> C1
    A2 --> B --> C2

    style A1 fill:#dcfce7
    style A2 fill:#fecaca
    style C1 fill:#dcfce7
    style C2 fill:#fecaca
```

## 4. 결측치 발생 원인

```mermaid
flowchart TD
    A["결측치<br>(Missing Value)"]

    A --> B["센서 고장"]
    A --> C["통신 오류"]
    A --> D["수동 입력 누락"]
    A --> E["시스템 장애"]
    A --> F["의도적 비공개"]

    style A fill:#1e40af,color:#fff
    style B fill:#fecaca
    style C fill:#fecaca
    style D fill:#fef3c7
    style E fill:#fecaca
    style F fill:#e2e8f0
```

## 5. 결측치 탐지 메서드

```mermaid
flowchart TD
    A["결측치 탐지"]

    A --> B["df.isnull()"]
    A --> C["df.isna()"]
    A --> D["df.info()"]

    B --> B1["각 셀별<br>True/False"]
    C --> C1["isnull()과<br>동일"]
    D --> D1["Non-Null Count<br>확인"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dbeafe
    style D fill:#dcfce7
```

## 6. 결측치 집계

```mermaid
flowchart TD
    A["df.isnull()"]
    B[".sum()"]
    C["컬럼별 결측치 개수"]

    A --> B --> C

    C --> D["/ len(df) * 100"]
    D --> E["결측치 비율 (%)"]

    style A fill:#dbeafe
    style C fill:#dcfce7
    style E fill:#fef3c7
```

## 7. 결측치 비율 기준

```mermaid
flowchart TD
    A["결측치 비율"]

    A --> B["0-5%"]
    A --> C["5-30%"]
    A --> D["30%+"]

    B --> B1["삭제 가능<br>영향 적음"]
    C --> C1["대체 권장<br>적절한 값으로"]
    D --> D1["재검토 필요<br>컬럼 제외 고려"]

    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
    style D1 fill:#fecaca
```

## 8. 결측치 처리 방법

```mermaid
flowchart TD
    A["결측치 처리"]

    A --> B["삭제"]
    A --> C["대체"]

    B --> B1["dropna()"]
    B1 --> B2["행 삭제"]
    B1 --> B3["열 삭제"]

    C --> C1["fillna()"]
    C1 --> C2["고정값"]
    C1 --> C3["통계값"]
    C1 --> C4["전파값"]

    style A fill:#1e40af,color:#fff
    style B fill:#fecaca
    style C fill:#dcfce7
```

## 9. dropna 옵션

```mermaid
flowchart TD
    A["df.dropna()"]

    A --> B["axis=0"]
    A --> C["axis=1"]
    A --> D["how='any'"]
    A --> E["how='all'"]
    A --> F["thresh=n"]

    B --> B1["행 삭제 (기본)"]
    C --> C1["열 삭제"]
    D --> D1["하나라도 NaN이면"]
    E --> E1["모두 NaN일 때만"]
    F --> F1["n개 이상<br>유효값 필요"]

    style A fill:#1e40af,color:#fff
```

## 10. fillna 옵션

```mermaid
flowchart TD
    A["df.fillna()"]

    A --> B["고정값"]
    A --> C["통계값"]
    A --> D["전파"]
    A --> E["보간"]

    B --> B1["fillna(0)"]
    C --> C1["fillna(df.mean())"]
    C --> C2["fillna(df.median())"]
    D --> D1["method='ffill'"]
    D --> D2["method='bfill'"]
    E --> E1["interpolate()"]

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 11. 대체 전략 선택

```mermaid
flowchart TD
    A["데이터 유형"]

    A --> B{"수치형?"}

    B -->|Yes| C{"분포"}
    B -->|No| D["범주형"]

    C -->|정규분포| E["평균 (mean)"]
    C -->|왜곡| F["중앙값 (median)"]

    D --> G["최빈값 (mode)"]

    style E fill:#dcfce7
    style F fill:#dcfce7
    style G fill:#fef3c7
```

## 12. 시계열 결측치 처리

```mermaid
flowchart LR
    A["시계열 데이터"]

    A --> B["ffill"]
    A --> C["bfill"]
    A --> D["interpolate"]

    B --> B1["앞 값으로 채움<br>→ → →"]
    C --> C1["뒤 값으로 채움<br>← ← ←"]
    D --> D1["선형 보간<br>중간값 추정"]

    style A fill:#1e40af,color:#fff
    style D fill:#dcfce7
```

## 13. 이상치란

```mermaid
flowchart TD
    A["이상치<br>(Outlier)"]

    A --> B["정의"]
    A --> C["유형"]

    B --> B1["다른 데이터와<br>동떨어진 값"]

    C --> C1["오류 (Error)"]
    C --> C2["극단값 (Extreme)"]

    C1 --> D["센서 오작동<br>입력 실수"]
    C2 --> E["실제 발생한<br>극단적 상황"]

    style A fill:#1e40af,color:#fff
    style C1 fill:#fecaca
    style C2 fill:#fef3c7
```

## 14. IQR 개념

```mermaid
flowchart LR
    A["최솟값"]
    B["Q1<br>(25%)"]
    C["중앙값<br>(50%)"]
    D["Q3<br>(75%)"]
    E["최댓값"]

    A --- B --- C --- D --- E

    B -.->|"IQR"| D

    style B fill:#dbeafe
    style C fill:#dcfce7
    style D fill:#dbeafe
```

## 15. IQR 이상치 기준

```mermaid
flowchart TD
    A["IQR = Q3 - Q1"]

    A --> B["하한"]
    A --> C["상한"]

    B --> B1["Q1 - 1.5 × IQR"]
    C --> C1["Q3 + 1.5 × IQR"]

    B1 --> D["이 범위를<br>벗어나면"]
    C1 --> D

    D --> E["이상치!"]

    style A fill:#1e40af,color:#fff
    style E fill:#fecaca
```

## 16. IQR 시각화

```mermaid
flowchart LR
    subgraph 이상치["이상치"]
        A["●"]
    end

    subgraph 정상["정상 범위"]
        B["하한"]
        C["━━ Q1 ━━ 중앙값 ━━ Q3 ━━"]
        D["상한"]
    end

    subgraph 이상치2["이상치"]
        E["●"]
    end

    A --- B
    D --- E

    style A fill:#fecaca
    style E fill:#fecaca
    style C fill:#dcfce7
```

## 17. IQR 계산 코드

```mermaid
flowchart TD
    A["quantile(0.25)"]
    B["quantile(0.75)"]
    C["Q3 - Q1"]
    D["Q1 - 1.5*IQR"]
    E["Q3 + 1.5*IQR"]
    F["필터링"]

    A --> C
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F

    F --> G["이상치 추출"]

    style G fill:#dcfce7
```

## 18. Z-score 개념

```mermaid
flowchart TD
    A["Z-score"]

    A --> B["공식"]
    B --> B1["Z = (x - μ) / σ"]

    A --> C["의미"]
    C --> C1["평균에서<br>표준편차 몇 배<br>떨어져 있는가"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dbeafe
```

## 19. Z-score 이상치 기준

```mermaid
flowchart TD
    A["Z-score 기준"]

    A --> B["|Z| > 2"]
    A --> C["|Z| > 3"]

    B --> B1["약 4.6%<br>이상치 판단"]
    C --> C1["약 0.3%<br>극단 이상치"]

    style B fill:#fef3c7
    style C fill:#fecaca
```

## 20. IQR vs Z-score 비교

```mermaid
flowchart TD
    A["이상치 탐지 방법"]

    A --> B["IQR"]
    A --> C["Z-score"]

    B --> B1["장점"]
    B --> B2["단점"]
    B1 --> B3["분포 가정 불필요<br>극단값에 덜 민감"]
    B2 --> B4["통계적 해석 어려움"]

    C --> C1["장점"]
    C --> C2["단점"]
    C1 --> C3["통계적 해석 명확<br>계산 간단"]
    C2 --> C4["정규분포 가정 필요"]

    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 21. 이상치 처리 전략

```mermaid
flowchart TD
    A["이상치 처리"]

    A --> B["제거"]
    A --> C["대체"]
    A --> D["유지"]

    B --> B1["행 삭제"]
    C --> C1["Clipping"]
    C --> C2["경계값 대체"]
    D --> D1["플래그 추가"]
    D --> D2["별도 분석"]

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 22. Clipping 개념

```mermaid
flowchart LR
    subgraph 전["처리 전"]
        A1["-50"]
        A2["25"]
        A3["150"]
    end

    subgraph 후["Clipping 후"]
        B1["10"]
        B2["25"]
        B3["50"]
    end

    A1 -->|"clip(10,50)"| B1
    A2 -->|"유지"| B2
    A3 -->|"clip(10,50)"| B3

    style A1 fill:#fecaca
    style A3 fill:#fecaca
    style B1 fill:#dcfce7
    style B3 fill:#dcfce7
```

## 23. clip 함수

```mermaid
flowchart TD
    A["df['col'].clip()"]

    A --> B["lower=값"]
    A --> C["upper=값"]

    B --> B1["하한 미만 →<br>하한값으로 대체"]
    C --> C1["상한 초과 →<br>상한값으로 대체"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#dcfce7
```

## 24. 플래그 추가

```mermaid
flowchart TD
    A["이상치 플래그"]

    A --> B["is_outlier 컬럼"]
    B --> C["True / False"]

    C --> D["활용"]
    D --> D1["이상치만 필터링"]
    D --> D2["모델 학습 시 제외"]
    D --> D3["별도 분석"]

    style A fill:#1e40af,color:#fff
    style D fill:#dcfce7
```

## 25. 전처리 순서

```mermaid
flowchart TD
    A["1단계: 결측치"]
    B["2단계: 이상치"]
    C["3단계: 추가 전처리"]

    A --> B --> C

    A --> A1["isnull → fillna/dropna"]
    B --> B1["IQR/Z-score → clip/제거"]
    C --> C1["스케일링, 인코딩<br>(9차시)"]

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
```

## 26. 전처리 판단 플로우

```mermaid
flowchart TD
    A["결측치 발견"]

    A --> B{"비율?"}
    B -->|"<5%"| C["삭제"]
    B -->|"5-30%"| D["대체"]
    B -->|">30%"| E["컬럼 제외 고려"]

    D --> F{"수치형?"}
    F -->|Yes| G["mean/median"]
    F -->|No| H["mode"]

    style C fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#fecaca
```

## 27. 이상치 판단 플로우

```mermaid
flowchart TD
    A["이상치 발견"]

    A --> B{"오류인가?"}
    B -->|"명백한 오류"| C["제거/대체"]
    B -->|"실제 극단값"| D["유지/플래그"]
    B -->|"불확실"| E["도메인 전문가<br>확인"]

    C --> F["clip() 또는<br>행 삭제"]
    D --> G["분석에 포함<br>또는 별도 처리"]

    style C fill:#fecaca
    style D fill:#dcfce7
    style E fill:#fef3c7
```

## 28. 전처리 체크리스트

```mermaid
flowchart TD
    A["전처리 체크리스트"]

    A --> B["1. 데이터 형태 확인"]
    A --> C["2. 결측치 현황"]
    A --> D["3. 이상치 탐지"]
    A --> E["4. 분포 확인"]
    A --> F["5. 처리 전략 결정"]
    A --> G["6. 처리 및 검증"]

    B --> B1["df.shape, df.dtypes"]
    C --> C1["df.isnull().sum()"]
    D --> D1["IQR / Z-score"]
    E --> E1["hist(), boxplot()"]

    style A fill:#1e40af,color:#fff
```

## 29. 실습 데이터 구조

```mermaid
flowchart TD
    A["실습 데이터"]

    A --> B["temperature"]
    A --> C["pressure"]
    A --> D["quality"]

    B --> B1["결측: 2개<br>이상치: -50, 150"]
    C --> C1["결측: 1개<br>이상치: 500"]
    D --> D1["결측: 1개<br>범주형"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#fef3c7
    style C1 fill:#fef3c7
    style D1 fill:#dcfce7
```

## 30. 실습 흐름

```mermaid
flowchart TD
    A["데이터 생성"]
    B["결측치 탐지"]
    C["결측치 처리"]
    D["이상치 탐지"]
    E["이상치 처리"]
    F["결과 검증"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 31. 핵심 함수 정리

```mermaid
flowchart TD
    A["전처리 핵심 함수"]

    A --> B["결측치"]
    A --> C["이상치"]
    A --> D["공통"]

    B --> B1["isnull()"]
    B --> B2["dropna()"]
    B --> B3["fillna()"]

    C --> C1["quantile()"]
    C --> C2["clip()"]
    C --> C3["zscore()"]

    D --> D1["info()"]
    D --> D2["describe()"]

    style A fill:#1e40af,color:#fff
```

## 32. 주의사항

```mermaid
flowchart TD
    A["주의사항"]

    A --> B["1. 순서 준수"]
    A --> C["2. 원본 보존"]
    A --> D["3. 문서화"]
    A --> E["4. 검증"]

    B --> B1["결측치 → 이상치"]
    C --> C1["새 컬럼에 결과 저장"]
    D --> D1["왜, 어떻게 처리했는지"]
    E --> E1["처리 전후 비교"]

    style A fill:#fecaca
```

## 33. 다음 차시 연결

```mermaid
flowchart LR
    A["9차시"]
    B["9차시"]

    A --> B

    A --> A1["결측치 처리"]
    A --> A2["이상치 처리"]

    B --> B1["스케일링"]
    B --> B2["인코딩"]
    B --> B3["특성 엔지니어링"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
