# [14차시] 모델 평가와 반복 검증 - 다이어그램

## 1. train_test_split의 한계

```mermaid
flowchart LR
    subgraph 문제["한 번의 분할"]
        A["전체 데이터"] --> B["학습 80%"]
        A --> C["테스트 20%"]
        C --> D["점수: 85%"]
    end

    E["운이 좋았을 수도?<br>운이 나빴을 수도?"]
    D --> E
```

## 2. K-Fold 교차검증

```mermaid
flowchart TD
    A["전체 데이터"] --> B["5등분"]

    B --> C1["Fold 1: [Test][Train][Train][Train][Train]"]
    B --> C2["Fold 2: [Train][Test][Train][Train][Train]"]
    B --> C3["Fold 3: [Train][Train][Test][Train][Train]"]
    B --> C4["Fold 4: [Train][Train][Train][Test][Train]"]
    B --> C5["Fold 5: [Train][Train][Train][Train][Test]"]

    C1 --> D["5개 점수"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D

    D --> E["평균 = 최종 성능"]
```

## 3. 교차검증 결과 해석

```mermaid
flowchart LR
    subgraph 결과["교차검증 결과"]
        A["Fold 1: 83%"]
        B["Fold 2: 86%"]
        C["Fold 3: 84%"]
        D["Fold 4: 85%"]
        E["Fold 5: 82%"]
    end

    F["평균: 84%<br>표준편차: ±1.4%"]
    G["신뢰할 수 있음!"]

    결과 --> F --> G
```

## 4. 과대적합 vs 과소적합

```mermaid
flowchart LR
    subgraph 과소["과소적합"]
        A1["학습: 70%"]
        B1["테스트: 68%"]
        C1["둘 다 낮음"]
    end

    subgraph 적절["적절한 모델"]
        A2["학습: 86%"]
        B2["테스트: 84%"]
        C2["비슷하게 높음"]
    end

    subgraph 과대["과대적합"]
        A3["학습: 98%"]
        B3["테스트: 75%"]
        C3["차이가 큼!"]
    end

    style 적절 fill:#c8e6c9
```

## 5. 과대적합 발생 과정

```mermaid
flowchart TD
    A["모델 복잡도 증가"] --> B["학습 데이터 암기"]
    B --> C["학습 정확도 상승"]
    B --> D["일반화 능력 저하"]
    D --> E["테스트 정확도 하락"]
    E --> F["⚠️ 과대적합"]
```

## 6. 해결 방법

```mermaid
flowchart TD
    subgraph 과소["과소적합 해결"]
        A1["더 복잡한 모델"]
        B1["특성 추가"]
        C1["max_depth 증가"]
    end

    subgraph 과대["과대적합 해결"]
        A2["더 단순한 모델"]
        B2["데이터 추가"]
        C2["max_depth 감소"]
    end
```

## 7. 혼동행렬 구조

```mermaid
flowchart TD
    subgraph 행렬["혼동행렬"]
        A["예측: 정상<br>실제: 정상<br>TN ✅"]
        B["예측: 불량<br>실제: 정상<br>FP ❌"]
        C["예측: 정상<br>실제: 불량<br>FN ❌"]
        D["예측: 불량<br>실제: 불량<br>TP ✅"]
    end
```

## 8. TN, FP, FN, TP 의미

```mermaid
mindmap
  root((혼동행렬))
    TN
      True Negative
      정상 → 정상 예측
      맞음 ✅
    FP
      False Positive
      정상 → 불량 예측
      틀림 ❌
    FN
      False Negative
      불량 → 정상 예측
      틀림 ❌ (놓침!)
    TP
      True Positive
      불량 → 불량 예측
      맞음 ✅
```

## 9. 정밀도와 재현율

```mermaid
flowchart TD
    subgraph 정밀도["정밀도 (Precision)"]
        A1["TP / (TP + FP)"]
        B1["불량 예측 중 진짜 불량"]
        C1["경고 정확도"]
    end

    subgraph 재현율["재현율 (Recall)"]
        A2["TP / (TP + FN)"]
        B2["실제 불량 중 잡아낸 것"]
        C2["탐지율"]
    end
```

## 10. 상황별 중요 지표

```mermaid
flowchart TD
    A["어떤 지표가 중요?"]

    A --> B["스팸 필터"]
    A --> C["암 진단"]
    A --> D["제조 불량"]

    B --> B1["정밀도<br>중요 메일 보호"]
    C --> C1["재현율<br>환자 놓치면 안됨"]
    D --> D1["재현율<br>불량품 놓치면 안됨"]

    style C1 fill:#ffcccc
    style D1 fill:#ffcccc
```

## 11. F1 Score

```mermaid
flowchart LR
    A["정밀도"] --> C["F1 Score"]
    B["재현율"] --> C

    C --> D["조화평균"]
    D --> E["둘 다 중요할 때 사용"]
```

## 12. sklearn 평가 함수

```mermaid
flowchart TD
    A["sklearn.metrics"]

    A --> B["confusion_matrix"]
    A --> C["precision_score"]
    A --> D["recall_score"]
    A --> E["f1_score"]
    A --> F["classification_report"]

    B --> B1["혼동행렬"]
    C --> C1["정밀도"]
    D --> D1["재현율"]
    E --> E1["F1 점수"]
    F --> F1["종합 리포트"]
```

## 13. 분류 vs 회귀 평가 지표

```mermaid
flowchart LR
    subgraph 분류["분류 모델"]
        A1["정확도"]
        B1["정밀도, 재현율"]
        C1["F1 Score"]
        D1["혼동행렬"]
    end

    subgraph 회귀["회귀 모델"]
        A2["R²"]
        B2["MSE, RMSE"]
        C2["MAE"]
        D2["산점도"]
    end
```

## 14. 강의 구조

```mermaid
gantt
    title 14차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    기존 방법 문제            :a2, after a1, 1.5m
    교차검증 개념             :a3, after a2, 2m
    과대적합/과소적합         :a4, after a3, 2m
    혼동행렬                 :a5, after a4, 1.5m
    정밀도/재현율             :a6, after a5, 1m

    section 실습편
    실습 소개               :b1, after a6, 2m
    데이터 준비              :b2, after b1, 2m
    교차검증                :b3, after b2, 2m
    과대적합 진단            :b4, after b3, 2m
    혼동행렬                :b5, after b4, 2m
    상세 지표               :b6, after b5, 2m
    분류 리포트              :b7, after b6, 2m
    모델 비교               :b8, after b7, 2m

    section 정리
    핵심 요약               :c1, after b8, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 15. 모델 평가 체크리스트

```mermaid
flowchart TD
    A["모델 평가 순서"]

    A --> B["1. 교차검증<br>cross_val_score"]
    B --> C["2. 과대적합 확인<br>학습 vs 테스트"]
    C --> D["3. 혼동행렬<br>confusion_matrix"]
    D --> E["4. 상세 평가<br>classification_report"]
    E --> F["5. 비교<br>여러 모델 교차검증"]
```

## 16. 핵심 요약

```mermaid
mindmap
  root((모델 평가))
    교차검증
      여러 번 평가
      평균으로 신뢰성
      cross_val_score
    과대적합
      학습만 높음
      모델 단순화
    과소적합
      둘 다 낮음
      모델 복잡화
    혼동행렬
      TN FP FN TP
    정밀도/재현율
      상황에 따라
      제조: 재현율 중요
```
