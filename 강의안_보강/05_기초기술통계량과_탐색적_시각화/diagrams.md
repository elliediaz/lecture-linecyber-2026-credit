# [5차시] 기초 기술통계량과 탐색적 시각화 - 다이어그램

## 1. 대표값의 종류

```mermaid
flowchart TD
    A["대표값<br>(Central Tendency)"]

    A --> B["평균<br>Mean"]
    A --> C["중앙값<br>Median"]
    A --> D["최빈값<br>Mode"]

    B --> B1["모든 값의 합 ÷ 개수"]
    C --> C1["정렬 후 가운데 값"]
    D --> D1["가장 자주 나타나는 값"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#dcfce7
```

## 2. 평균의 계산

```mermaid
flowchart LR
    A["데이터"] --> B["합계"]
    B --> C["개수로 나눔"]
    C --> D["평균"]

    A --> A1["[85, 86, 84, 87]"]
    B --> B1["342"]
    C --> C1["÷ 4"]
    D --> D1["85.5"]

    style D fill:#dcfce7
```

## 3. 평균의 이상치 민감성

```mermaid
flowchart TD
    subgraph 정상["정상 데이터"]
        N1["85, 86, 84, 87"]
        N2["평균: 85.5"]
    end

    subgraph 이상치["이상치 포함"]
        O1["85, 86, 84, 200"]
        O2["평균: 113.75 ❌"]
    end

    style N2 fill:#dcfce7
    style O2 fill:#fecaca
```

## 4. 중앙값의 강건성

```mermaid
flowchart TD
    subgraph 정상["정상 데이터"]
        N1["84, 85, 86, 87"]
        N2["중앙값: 85.5"]
    end

    subgraph 이상치["이상치 포함"]
        O1["84, 85, 86, 200"]
        O2["중앙값: 85.5 ✓"]
    end

    style N2 fill:#dcfce7
    style O2 fill:#dcfce7
```

## 5. 중앙값 계산 방법

```mermaid
flowchart TD
    A["데이터 정렬"]

    A --> B{홀수 개?}

    B -->|Yes| C["정중앙 값"]
    B -->|No| D["가운데 두 값의 평균"]

    C --> E["[1,2,3,4,5] → 3"]
    D --> F["[1,2,3,4] → (2+3)/2 = 2.5"]

    style A fill:#1e40af,color:#fff
```

## 6. 분포 형태와 대표값

```mermaid
flowchart LR
    subgraph 정규분포
        A1["평균 = 중앙값 = 최빈값"]
    end

    subgraph 오른쪽_꼬리
        B1["평균 > 중앙값 > 최빈값"]
    end

    subgraph 왼쪽_꼬리
        C1["최빈값 > 중앙값 > 평균"]
    end

    style A1 fill:#dcfce7
    style B1 fill:#fef3c7
    style C1 fill:#fef3c7
```

## 7. 산포도의 종류

```mermaid
flowchart TD
    A["산포도<br>(Dispersion)"]

    A --> B["범위<br>Range"]
    A --> C["분산<br>Variance"]
    A --> D["표준편차<br>Std Dev"]
    A --> E["IQR"]

    B --> B1["최대 - 최소"]
    C --> C1["편차 제곱 평균"]
    D --> D1["√분산"]
    E --> E1["Q3 - Q1"]

    style A fill:#1e40af,color:#fff
```

## 8. 분산 계산 과정

```mermaid
flowchart TD
    A["데이터: 84, 85, 86, 87"]
    B["평균: 85.5"]
    C["편차: -1.5, -0.5, 0.5, 1.5"]
    D["제곱: 2.25, 0.25, 0.25, 2.25"]
    E["합계: 5.0"]
    F["분산: 5.0÷3 = 1.67"]

    A --> B --> C --> D --> E --> F

    style F fill:#dcfce7
```

## 9. 표준편차의 의미

```mermaid
flowchart TD
    A["표준편차<br>= √분산"]

    A --> B["단위가 원래대로"]
    A --> C["평균에서의 평균 거리"]

    B --> B1["분산: °C²<br>표준편차: °C"]
    C --> C1["평균 85±3°C<br>→ 82~88°C"]

    style A fill:#1e40af,color:#fff
```

## 10. 68-95-99.7 규칙

```mermaid
flowchart TD
    A["정규분포"]

    A --> B["68%: μ ± 1σ"]
    A --> C["95%: μ ± 2σ"]
    A --> D["99.7%: μ ± 3σ"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 11. 사분위수

```mermaid
flowchart LR
    A["최소"] --> B["Q1<br>(25%)"]
    B --> C["Q2<br>(중앙값)"]
    C --> D["Q3<br>(75%)"]
    D --> E["최대"]

    style B fill:#dbeafe
    style C fill:#dcfce7
    style D fill:#dbeafe
```

## 12. IQR과 이상치 탐지

```mermaid
flowchart TD
    A["IQR = Q3 - Q1"]

    A --> B["이상치 하한"]
    A --> C["이상치 상한"]

    B --> B1["Q1 - 1.5×IQR"]
    C --> C1["Q3 + 1.5×IQR"]

    B1 --> D["하한 미만 → 이상치"]
    C1 --> E["상한 초과 → 이상치"]

    style A fill:#1e40af,color:#fff
    style D fill:#fecaca
    style E fill:#fecaca
```

## 13. EDA의 목적

```mermaid
flowchart TD
    A["탐색적 데이터 분석<br>(EDA)"]

    A --> B["분포 파악"]
    A --> C["이상치 탐지"]
    A --> D["변수 관계 확인"]
    A --> E["인사이트 획득"]

    style A fill:#1e40af,color:#fff
```

## 14. 핵심 시각화 도구

```mermaid
flowchart TD
    A["시각화 도구"]

    A --> B["히스토그램"]
    A --> C["상자그림"]
    A --> D["산점도"]

    B --> B1["분포 확인"]
    C --> C1["이상치, 사분위수"]
    D --> D1["변수 간 관계"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 15. 히스토그램 구조

```mermaid
flowchart TD
    A["히스토그램"]

    A --> B["X축: 값의 구간"]
    A --> C["Y축: 빈도"]
    A --> D["막대: 구간별 개수"]

    style A fill:#1e40af,color:#fff
```

## 16. 히스토그램 해석

```mermaid
flowchart TD
    A["히스토그램 체크포인트"]

    A --> B["중심 위치"]
    A --> C["퍼짐 정도"]
    A --> D["대칭성"]
    A --> E["이상치"]

    B --> B1["어디에 몰려있나?"]
    C --> C1["넓게? 좁게?"]
    D --> D1["좌우 대칭?"]
    E --> E1["동떨어진 막대?"]

    style A fill:#1e40af,color:#fff
```

## 17. 상자그림 구조

```mermaid
flowchart TD
    subgraph BoxPlot
        A["이상치 ○"]
        B["최대 (수염)"]
        C["Q3 ─"]
        D["중앙값 ─"]
        E["Q1 ─"]
        F["최소 (수염)"]
        G["이상치 ○"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

    style D fill:#dcfce7
```

## 18. 상자그림 비교

```mermaid
flowchart LR
    subgraph 라인A["라인 A"]
        A1["작은 상자"]
        A2["안정적"]
    end

    subgraph 라인B["라인 B"]
        B1["중간 상자"]
        B2["보통"]
    end

    subgraph 라인C["라인 C"]
        C1["큰 상자 + 이상치"]
        C2["불안정 ⚠️"]
    end

    style A2 fill:#dcfce7
    style C2 fill:#fecaca
```

## 19. 산점도 패턴

```mermaid
flowchart LR
    subgraph 양의상관["양의 상관"]
        A1["X↑ → Y↑"]
    end

    subgraph 음의상관["음의 상관"]
        B1["X↑ → Y↓"]
    end

    subgraph 상관없음["상관 없음"]
        C1["패턴 없음"]
    end

    style A1 fill:#dcfce7
    style B1 fill:#fef3c7
    style C1 fill:#e2e8f0
```

## 20. describe() 결과

```mermaid
flowchart TD
    A["df.describe()"]

    A --> B["count: 개수"]
    A --> C["mean: 평균"]
    A --> D["std: 표준편차"]
    A --> E["min: 최소"]
    A --> F["25%: Q1"]
    A --> G["50%: 중앙값"]
    A --> H["75%: Q3"]
    A --> I["max: 최대"]

    style A fill:#1e40af,color:#fff
```

## 21. Matplotlib 기본 흐름

```mermaid
flowchart TD
    A["import matplotlib.pyplot as plt"]
    B["plt.figure()"]
    C["plt.plot() / hist() / scatter()"]
    D["plt.xlabel(), ylabel(), title()"]
    E["plt.show()"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 22. 서브플롯 구조

```mermaid
flowchart TD
    A["fig, axes = plt.subplots(2, 2)"]

    A --> B["axes[0,0]<br>히스토그램"]
    A --> C["axes[0,1]<br>상자그림"]
    A --> D["axes[1,0]<br>산점도"]
    A --> E["axes[1,1]<br>시계열"]

    style A fill:#1e40af,color:#fff
```

## 23. 통계량 vs 시각화

```mermaid
flowchart TD
    A["데이터 분석"]

    A --> B["통계량<br>(숫자)"]
    A --> C["시각화<br>(그래프)"]

    B --> B1["평균, 표준편차"]
    C --> C1["히스토그램, 상자그림"]

    B1 --> D["요약"]
    C1 --> D

    D --> E["인사이트"]

    style E fill:#dcfce7
```

## 24. 앤스콤 콰르텟 교훈

```mermaid
flowchart TD
    A["4개 데이터셋"]

    A --> B["평균 같음"]
    A --> C["분산 같음"]
    A --> D["상관계수 같음"]

    B --> E["그래프는<br>완전히 다름!"]
    C --> E
    D --> E

    E --> F["반드시 시각화 필요"]

    style E fill:#fef3c7
    style F fill:#dcfce7
```

## 25. 데이터 확인 순서

```mermaid
flowchart TD
    A["데이터 확보"]
    B["df.shape"]
    C["df.info()"]
    D["df.describe()"]
    E["히스토그램"]
    F["상자그림"]
    G["산점도"]
    H["분석 시작"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#dbeafe
    style H fill:#dcfce7
```

## 26. 이상치 탐지 방법

```mermaid
flowchart TD
    A["이상치 탐지"]

    A --> B["통계적 방법"]
    A --> C["시각적 방법"]

    B --> B1["IQR 기준"]
    B --> B2["Z-score (3σ)"]

    C --> C1["상자그림"]
    C --> C2["히스토그램"]

    style A fill:#1e40af,color:#fff
```

## 27. 5차시 학습 흐름

```mermaid
flowchart LR
    A["대표값<br>평균/중앙값"]
    B["산포도<br>표준편차/IQR"]
    C["시각화<br>히스토그램"]
    D["실습<br>Matplotlib"]
    E["6차시:<br>확률분포"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dcfce7
    style E fill:#1e40af,color:#fff
```

## 28. 핵심 코드 정리

```mermaid
flowchart TD
    subgraph 통계량
        S1["np.mean()"]
        S2["np.median()"]
        S3["np.std()"]
        S4["df.describe()"]
    end

    subgraph 시각화
        V1["plt.hist()"]
        V2["plt.boxplot()"]
        V3["plt.scatter()"]
    end

    style S4 fill:#dcfce7
    style V1 fill:#dcfce7
```
