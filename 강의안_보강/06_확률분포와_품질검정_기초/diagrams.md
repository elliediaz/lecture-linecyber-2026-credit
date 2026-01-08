# [6차시] 확률분포와 품질검정 기초 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["정규분포<br>68-95-99.7"]
    B["Z-score<br>이상치 탐지"]
    C["7차시:<br>통계 검정 실습"]

    A --> B --> C

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#1e40af,color:#fff
```

## 2. 확률분포의 종류

```mermaid
flowchart TD
    A["확률분포"]

    A --> B["연속형"]
    A --> C["이산형"]

    B --> B1["정규분포"]
    B --> B2["균등분포"]
    B --> B3["지수분포"]

    C --> C1["이항분포"]
    C --> C2["포아송분포"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
```

## 3. 정규분포의 특성

```mermaid
flowchart TD
    A["정규분포<br>N(μ, σ²)"]

    A --> B["평균(μ)"]
    A --> C["표준편차(σ)"]

    B --> B1["분포의 중심"]
    C --> C1["퍼짐 정도"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
```

## 4. 정규분포 곡선

```mermaid
flowchart TD
    subgraph 정규분포["정규분포 (Bell Curve)"]
        A["종 모양"]
        B["평균 중심 대칭"]
        C["멀어질수록 확률 감소"]
    end

    정규분포 --> D["대부분의 자연/제조 데이터"]

    style A fill:#dbeafe
    style B fill:#dbeafe
    style C fill:#dbeafe
    style D fill:#dcfce7
```

## 5. 68-95-99.7 규칙

```mermaid
flowchart TD
    A["정규분포 경험적 규칙"]

    A --> B["μ ± 1σ"]
    A --> C["μ ± 2σ"]
    A --> D["μ ± 3σ"]

    B --> B1["68%"]
    C --> C1["95%"]
    D --> D1["99.7%"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
    style D1 fill:#dbeafe
```

## 6. 68-95-99.7 의미

```mermaid
flowchart LR
    subgraph 범위["데이터 범위"]
        A["±1σ: 68%<br>정상 변동"]
        B["±2σ: 95%<br>주의 범위"]
        C["±3σ: 99.7%<br>관리 한계"]
    end

    C --> D["밖: 0.3%<br>이상치!"]

    style A fill:#dcfce7
    style B fill:#fef3c7
    style C fill:#dbeafe
    style D fill:#fecaca
```

## 7. 품질 관리 기준 예시

```mermaid
flowchart TD
    A["생산량 N(1200, 50²)"]

    A --> B["68%: 1150~1250"]
    A --> C["95%: 1100~1300"]
    A --> D["99.7%: 1050~1350"]

    B --> B1["정상 변동"]
    C --> C1["주의 필요"]
    D --> D1["관리 한계"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
    style D1 fill:#fecaca
```

## 8. 이상치란?

```mermaid
flowchart TD
    A["이상치<br>(Outlier)"]

    A --> B["정의"]
    A --> C["원인"]
    A --> D["중요성"]

    B --> B1["다른 데이터와<br>동떨어진 값"]
    C --> C1["측정 오류, 설비 이상,<br>원자재 불량..."]
    D --> D1["문제의 신호<br>빠른 발견 → 빠른 대응"]

    style A fill:#1e40af,color:#fff
    style D1 fill:#dcfce7
```

## 9. Z-score 개념

```mermaid
flowchart LR
    A["Z-score"]
    B["(X - μ) / σ"]
    C["평균에서<br>표준편차 몇 개<br>떨어져 있는가"]

    A --> B --> C

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 10. Z-score 해석

```mermaid
flowchart TD
    A["Z-score 값"]

    A --> B["Z = 0"]
    A --> C["Z = 1"]
    A --> D["Z = 2"]
    A --> E["Z = 3"]

    B --> B1["평균과 같음"]
    C --> C1["상위 16%"]
    D --> D1["상위 2.5%"]
    E --> E1["상위 0.15%"]

    style A fill:#1e40af,color:#fff
    style D1 fill:#fef3c7
    style E1 fill:#fecaca
```

## 11. Z-score 판단 기준

```mermaid
flowchart TD
    A["Z-score 기준"]

    A --> B["|Z| ≤ 1"]
    A --> C["1 < |Z| ≤ 2"]
    A --> D["2 < |Z| ≤ 3"]
    A --> E["|Z| > 3"]

    B --> B1["정상 (68%)"]
    C --> C1["주의 (27%)"]
    D --> D1["경고 (4.3%)"]
    E --> E1["이상치 (0.3%)"]

    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
    style D1 fill:#fef3c7
    style E1 fill:#fecaca
```

## 12. Z-score 계산 과정

```mermaid
flowchart TD
    A["데이터 수집"]
    B["평균(μ) 계산"]
    C["표준편차(σ) 계산"]
    D["Z = (X-μ)/σ"]
    E["기준 초과 확인"]
    F["이상치 판정"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 13. Z-score vs IQR 비교

```mermaid
flowchart TD
    subgraph Z["Z-score 방식"]
        Z1["평균/표준편차 기반"]
        Z2["정규분포 가정"]
        Z3["|Z| > 2 또는 3"]
    end

    subgraph I["IQR 방식"]
        I1["Q1, Q3, IQR 기반"]
        I2["분포 가정 불필요"]
        I3["1.5×IQR 밖"]
    end

    style Z1 fill:#dbeafe
    style I1 fill:#dcfce7
```

## 14. 이상치 탐지 흐름

```mermaid
flowchart TD
    A["데이터 수집"]
    B{분포 형태?}
    C["Z-score 방법"]
    D["IQR 방법"]
    E["이상치 식별"]
    F["원인 조사"]

    A --> B
    B -->|정규분포| C
    B -->|비정규| D
    C --> E
    D --> E
    E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 15. 실습 흐름

```mermaid
flowchart TD
    A["실습 1<br>정규분포 시각화"]
    B["실습 2<br>68-95-99.7 검증"]
    C["실습 3<br>Z-score 계산"]
    D["실습 4<br>이상치 시각화"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 16. 핵심 코드 정리

```mermaid
flowchart TD
    subgraph 정규분포["정규분포"]
        N1["np.random.normal()"]
        N2["데이터 생성"]
    end

    subgraph Zscore["Z-score"]
        Z1["(X - μ) / σ"]
        Z2["stats.zscore()"]
    end

    style N1 fill:#dcfce7
    style Z2 fill:#dcfce7
```

## 17. 6차시 핵심 정리

```mermaid
flowchart TD
    A["6차시 핵심"]

    A --> B["정규분포"]
    A --> C["Z-score"]

    B --> B1["68-95-99.7 규칙<br>±3σ = 이상치 기준"]

    C --> C1["|Z| > 2 주의<br>|Z| > 3 이상치"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
    style C1 fill:#fef3c7
```

## 18. 다음 차시 예고

```mermaid
flowchart TD
    A["7차시: 통계 검정 실습"]

    A --> B["t-검정"]
    A --> C["카이제곱 검정"]
    A --> D["ANOVA"]

    B --> B1["두 그룹 평균 비교"]
    C --> C1["범주형 데이터 분석"]
    D --> D1["3개+ 그룹 비교"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```
