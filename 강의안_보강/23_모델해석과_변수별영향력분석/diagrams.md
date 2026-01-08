# [23차시] 모델 해석과 변수별 영향력 분석 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["21차시<br>딥러닝 심화"]
    B["23차시<br>모델 해석"]
    C["23차시<br>모델 저장"]

    A --> B --> C

    B --> B1["Feature Importance"]
    B --> B2["Permutation Importance"]
    B --> B3["SHAP"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["23차시: 모델 해석"]

    A --> B["대주제 1<br>해석의 필요성"]
    A --> C["대주제 2<br>Feature Importance"]
    A --> D["대주제 3<br>Permutation Importance"]

    B --> B1["신뢰, 디버깅<br>개선, 규제"]
    C --> C1["트리 기반<br>불순도 감소"]
    D --> D1["모델 무관<br>성능 하락"]

    style A fill:#1e40af,color:#fff
```

## 3. 블랙박스 문제

```mermaid
flowchart LR
    A["입력<br>(온도, 압력)"]
    B["모델<br>?"]
    C["예측<br>불량"]

    A --> B --> C

    B --> B1["왜 불량?"]
    B --> B2["뭘 바꾸면?"]

    style B fill:#fecaca
```

## 4. 모델 해석 필요성

```mermaid
flowchart TD
    A["왜 모델 해석?"]

    A --> B["신뢰"]
    B --> B1["의사결정 근거"]

    A --> C["디버깅"]
    C --> C1["잘못된 패턴 발견"]

    A --> D["개선"]
    D --> D1["데이터 수집 방향"]

    A --> E["규제"]
    E --> E1["설명 의무화"]

    style A fill:#1e40af,color:#fff
```

## 5. 해석 가능성 vs 성능

```mermaid
flowchart LR
    A["선형 회귀"]
    B["의사결정나무"]
    C["RandomForest"]
    D["신경망"]

    A --> |"해석 쉬움"| B
    B --> C
    C --> |"해석 어려움"| D

    A --> A1["계수 해석"]
    D --> D1["블랙박스"]

    style A fill:#dcfce7
    style D fill:#fecaca
```

## 6. 전역 vs 지역 해석

```mermaid
flowchart TD
    A["모델 해석"]

    A --> B["전역 해석<br>Global"]
    B --> B1["모델 전체<br>변수 중요도"]
    B --> B2["'온도가 가장 중요'"]

    A --> C["지역 해석<br>Local"]
    C --> C1["개별 예측 설명"]
    C --> C2["'이 샘플은 압력 때문'"]

    style A fill:#1e40af,color:#fff
```

## 7. Feature Importance 원리

```mermaid
flowchart TD
    A["Feature Importance"]

    A --> B["분할 기여도"]
    B --> B1["변수로 분할할 때<br>불순도 감소량"]

    A --> C["합산"]
    C --> C1["모든 트리에서<br>합산"]

    A --> D["정규화"]
    D --> D1["합 = 1"]

    style A fill:#1e40af,color:#fff
```

## 8. Feature Importance 예시

```mermaid
flowchart LR
    A["온도: 0.35"]
    B["압력: 0.25"]
    C["진동: 0.18"]
    D["속도: 0.12"]
    E["습도: 0.10"]

    A --> A1["████████████"]
    B --> B1["████████"]
    C --> C1["██████"]
    D --> D1["████"]
    E --> E1["███"]
```

## 9. Feature Importance 활용

```mermaid
flowchart TD
    A["분석 결과"]

    A --> B["온도 35%"]
    B --> B1["✅ 모니터링 강화"]

    A --> C["압력 25%"]
    C --> C1["✅ 센서 정밀도 개선"]

    A --> D["습도 10%"]
    D --> D1["⚠️ 비용 대비 효과 낮음"]

    style A fill:#1e40af,color:#fff
```

## 10. 상관 변수 문제

```mermaid
flowchart TD
    A["상관 변수 문제"]

    A --> B["온도 ↔ 습도<br>상관 0.9"]

    A --> C["Feature Importance"]
    C --> C1["온도: 0.25"]
    C --> C2["습도: 0.20"]
    C --> C3["분산됨!"]

    A --> D["실제"]
    D --> D1["온도 하나로 충분"]

    style C fill:#fecaca
```

## 11. Permutation Importance 원리

```mermaid
flowchart TD
    A["원본"]
    B["온도 섞기"]
    C["성능 비교"]
    D["중요도"]

    A --> |"정확도 90%"| B
    B --> |"정확도 65%"| C
    C --> |"90% - 65%"| D
    D --> D1["중요도 = 25%"]

    style D fill:#1e40af,color:#fff
```

## 12. Permutation 과정

```mermaid
flowchart LR
    subgraph 원본
        A1["온도"]
        A2["압력"]
        A3["속도"]
    end

    subgraph 섞기
        B1["랜덤"]
        B2["압력"]
        B3["속도"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3

    B1 --> C["성능 측정"]
```

## 13. Permutation Importance 파라미터

```mermaid
flowchart TD
    A["permutation_importance"]

    A --> B["estimator"]
    B --> B1["학습된 모델"]

    A --> C["X, y"]
    C --> C1["테스트 데이터"]

    A --> D["n_repeats"]
    D --> D1["반복 횟수<br>(안정성)"]

    style A fill:#1e40af,color:#fff
```

## 14. Feature vs Permutation

```mermaid
flowchart TD
    A["비교"]

    A --> B["Feature Importance"]
    B --> B1["트리 모델만"]
    B --> B2["학습 데이터 기반"]
    B --> B3["빠름"]

    A --> C["Permutation Importance"]
    C --> C1["모든 모델"]
    C --> C2["테스트 데이터 기반"]
    C --> C3["느림 (n_repeats)"]

    style A fill:#1e40af,color:#fff
```

## 15. 두 방법 함께 사용

```mermaid
flowchart TD
    A["1. Feature Importance"]
    B["2. Permutation Importance"]
    C["3. 비교"]
    D["4. 신뢰도 확인"]

    A --> |"빠른 탐색"| B
    B --> |"검증"| C
    C --> D

    D --> D1["순위 비슷<br>→ 신뢰 ↑"]
    D --> D2["순위 다름<br>→ 상관 변수 확인"]

    style D fill:#1e40af,color:#fff
```

## 16. SHAP 개념

```mermaid
flowchart TD
    A["SHAP"]

    A --> B["원리"]
    B --> B1["게임 이론 기반"]

    A --> C["특징"]
    C --> C1["개별 예측 설명"]
    C --> C2["전역 + 지역"]

    A --> D["출력"]
    D --> D1["변수별 기여도<br>(+/-)"]

    style A fill:#1e40af,color:#fff
```

## 17. SHAP 예시

```mermaid
flowchart TD
    A["샘플 예측: 불량 (0.85)"]

    A --> B["온도 +0.25"]
    B --> B1["높아서 불량 방향 ↑"]

    A --> C["압력 +0.15"]
    C --> C1["높아서 불량 방향 ↑"]

    A --> D["속도 -0.05"]
    D --> D1["낮아서 정상 방향 ↓"]

    style A fill:#fecaca
```

## 18. 모델 해석 워크플로우

```mermaid
flowchart TD
    A["1. 모델 학습"]
    B["2. Feature Importance<br>(빠른 탐색)"]
    C["3. Permutation Importance<br>(검증)"]
    D["4. SHAP<br>(개별 설명, 필요시)"]
    E["5. 비즈니스 인사이트"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 19. 비즈니스 활용

```mermaid
flowchart TD
    A["분석 결과"]

    A --> B["온도 25%"]
    A --> C["압력 18%"]

    B --> D["조치 1<br>모니터링 주기 단축"]
    B --> E["조치 2<br>온도 이상 시 알림"]

    C --> F["조치 3<br>센서 교정 주기 확인"]

    style A fill:#1e40af,color:#fff
```

## 20. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 로드"]
    B["2. 모델 학습"]
    C["3. Feature Importance"]
    D["4. Permutation Importance"]
    E["5. 비교 및 시각화"]
    F["6. 인사이트 도출"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 21. sklearn 함수

```mermaid
flowchart TD
    A["sklearn"]

    A --> B["model.feature_importances_"]
    B --> B1["트리 모델 내장"]

    A --> C["permutation_importance()"]
    C --> C1["모든 모델"]

    A --> D["result"]
    D --> D1[".importances_mean"]
    D --> D2[".importances_std"]

    style A fill:#1e40af,color:#fff
```

## 22. 핵심 정리

```mermaid
flowchart TD
    A["23차시 핵심"]

    A --> B["모델 해석"]
    B --> B1["신뢰, 디버깅<br>개선, 규제"]

    A --> C["Feature Importance"]
    C --> C1["트리 모델<br>빠름"]

    A --> D["Permutation Importance"]
    D --> D1["모든 모델<br>테스트 기반"]

    style A fill:#1e40af,color:#fff
```

## 23. 다음 차시 연결

```mermaid
flowchart LR
    A["23차시<br>모델 해석"]
    B["23차시<br>모델 저장"]

    A --> B

    A --> A1["변수 중요도"]
    B --> B1["joblib 저장"]
    B --> B2["Pipeline"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
