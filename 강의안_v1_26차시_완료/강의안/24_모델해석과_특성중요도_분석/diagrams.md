# [24차시] 모델 해석과 변수별 영향력 분석 - 다이어그램

## 1. 블랙박스 문제

```mermaid
flowchart LR
    A["입력 데이터<br>(온도, 습도, 속도)"]
    B["ML 모델<br>???"]
    C["예측 결과<br>(정상/불량)"]

    A --> B --> C

    style B fill:#fee2e2
```

## 2. 모델 해석의 필요성

```mermaid
flowchart TD
    A["모델 해석"]

    A --> B["신뢰<br>왜 불량인가요?"]
    A --> C["디버깅<br>이상한 패턴?"]
    A --> D["규제<br>설명 의무"]

    style A fill:#dbeafe
```

## 3. 특성 중요도 개념

```mermaid
flowchart LR
    A["Feature Importance"]

    A --> B["온도: 35%"]
    A --> C["습도: 30%"]
    A --> D["속도: 25%"]
    A --> E["압력: 10%"]

    F["합계: 100%"]

    B & C & D & E --> F
```

## 4. 트리 기반 모델과 중요도

```mermaid
flowchart TD
    A["트리 기반 모델"]

    A --> B["RandomForest"]
    A --> C["DecisionTree"]
    A --> D["XGBoost"]

    E["feature_importances_<br>자동 제공"]

    B & C & D --> E
```

## 5. Feature Importance 코드

```mermaid
flowchart TD
    A["model = RandomForestClassifier()"]
    B["model.fit(X_train, y_train)"]
    C["model.feature_importances_"]
    D["[0.35, 0.30, 0.25, 0.10]"]

    A --> B --> C --> D
```

## 6. Permutation Importance 원리

```mermaid
flowchart TD
    A["원본 데이터"]
    B["온도 값을 섞음"]
    C["예측 수행"]
    D["성능 측정"]
    E{성능 변화?}
    F["중요한 특성!"]
    G["덜 중요한 특성"]

    A --> B --> C --> D --> E
    E -->|많이 떨어짐| F
    E -->|거의 변화 없음| G

    style F fill:#dcfce7
    style G fill:#fee2e2
```

## 7. 두 방법 비교

```mermaid
flowchart LR
    subgraph FI["Feature Importance"]
        A1["학습 중 계산"]
        A2["빠름"]
        A3["편향 가능"]
    end

    subgraph PI["Permutation Importance"]
        B1["학습 후 계산"]
        B2["느림"]
        B3["더 신뢰성"]
    end
```

## 8. 제조 데이터 특성 중요도

```mermaid
flowchart LR
    subgraph 중요도["품질 예측 모델"]
        A["온도 ████████ 35%"]
        B["습도 ██████ 30%"]
        C["속도 █████ 25%"]
        D["압력 ██ 10%"]
    end

    E["온도 관리가<br>최우선!"]

    중요도 --> E
```

## 9. 시각화 방법

```mermaid
flowchart TD
    A["특성 중요도 분석"]

    A --> B["막대 그래프<br>plt.barh()"]
    A --> C["정렬<br>np.argsort()"]
    A --> D["색상 구분<br>중요도별 색"]

    E["직관적 비교"]

    B & C & D --> E
```

## 10. 상관된 특성 문제

```mermaid
flowchart LR
    A["온도"]
    B["습도"]
    C["상관관계 0.8"]

    A <--> B
    A --> C
    B --> C

    D["중요도가<br>분산될 수 있음"]

    C --> D

    style D fill:#fef3c7
```

## 11. 인과관계 vs 상관관계

```mermaid
flowchart TD
    A["중요도가 높다"]
    B["원인이다?"]

    A -->|"≠"| B

    C["도메인 지식<br>필요"]

    B --> C

    style B fill:#fee2e2
```

## 12. 분석 워크플로우

```mermaid
flowchart TD
    A["모델 학습"]
    B["Feature Importance<br>빠른 확인"]
    C{중요한 분석?}
    D["Permutation Importance<br>신뢰성 높은 분석"]
    E["결과 해석"]
    F["보고서 작성"]

    A --> B --> C
    C -->|예| D --> E
    C -->|아니오| E
    E --> F
```

## 13. 비즈니스 인사이트 도출

```mermaid
flowchart TD
    A["특성 중요도 결과"]
    B["온도 35% > 습도 30% > 속도 25%"]
    C["해석"]
    D["온도가 품질에<br>가장 큰 영향"]
    E["권장사항"]
    F["온도 모니터링 강화"]

    A --> B --> C --> D --> E --> F
```

## 14. 실무 보고서 구조

```mermaid
flowchart TD
    subgraph 보고서["분석 보고서"]
        A["모델 성능<br>정확도 92%"]
        B["주요 영향 요인<br>온도, 습도, 속도"]
        C["해석<br>85°C 초과 시 불량↑"]
        D["권장사항<br>온도 제어 강화"]
    end
```

## 15. sklearn.inspection 모듈

```mermaid
flowchart LR
    A["sklearn.inspection"]

    A --> B["permutation_importance()"]
    A --> C["PartialDependenceDisplay"]
    A --> D["DecisionBoundaryDisplay"]

    style B fill:#dbeafe
```

## 16. 고급 해석 기법

```mermaid
flowchart TD
    subgraph 기본["기본 (이번 차시)"]
        A["Feature Importance"]
        B["Permutation Importance"]
    end

    subgraph 고급["고급 (향후)"]
        C["SHAP"]
        D["LIME"]
        E["PDP"]
    end

    기본 --> 고급
```

## 17. SHAP 개념

```mermaid
flowchart LR
    A["개별 예측"]
    B["SHAP 값"]
    C["각 특성의<br>기여도"]

    A --> B --> C

    D["왜 이 제품이<br>불량인지 설명"]

    C --> D
```

## 18. 강의 구조

```mermaid
gantt
    title 24차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    블랙박스 문제             :a2, after a1, 2m
    Feature Importance        :a3, after a2, 2m
    제조 예시 해석            :a4, after a3, 1.5m
    Permutation Importance    :a5, after a4, 1.5m
    이론 정리                 :a6, after a5, 1m

    section 실습편
    실습 소개                 :b1, after a6, 1.5m
    데이터 준비               :b2, after b1, 2m
    모델 학습                 :b3, after b2, 2m
    Feature Importance        :b4, after b3, 2m
    중요도 시각화             :b5, after b4, 2m
    Permutation Importance    :b6, after b5, 2m
    두 방법 비교              :b7, after b6, 2m
    결과 해석                 :b8, after b7, 2m

    section 정리
    핵심 요약                 :c1, after b8, 1.5m
    다음 차시 예고             :c2, after c1, 1m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((모델 해석))
    Feature Importance
      model.feature_importances_
      트리 기반 모델
      빠른 확인
    Permutation Importance
      sklearn.inspection
      더 신뢰성
      학습 후 계산
    시각화
      막대 그래프
      정렬해서 표시
    주의사항
      상관관계
      인과관계 구분
      도메인 지식
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>모델 해석<br>(특성 중요도)"]
    B["다음<br>모델 저장<br>(배포 준비)"]
    C["이후<br>종합 실습<br>(프로젝트)"]

    A --> B --> C
```
