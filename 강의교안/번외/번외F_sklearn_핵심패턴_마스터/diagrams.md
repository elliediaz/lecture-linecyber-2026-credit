# 번외F: sklearn 핵심 패턴 마스터 - Mermaid 다이어그램

## 1. sklearn 설계 철학

```mermaid
flowchart LR
    A[sklearn 객체] --> B[fit]
    B --> C[학습 완료]
    C --> D[predict 또는 transform]
```

## 2. sklearn 객체 유형

```mermaid
flowchart TD
    A[sklearn 객체] --> B[Estimator]
    A --> C[Predictor]
    A --> D[Transformer]
    B --> E[fit 메서드]
    C --> F[predict 메서드]
    D --> G[transform 메서드]
```

## 3. fit()이 하는 일

```mermaid
flowchart TD
    A[fit] --> B[모델]
    A --> C[스케일러]
    A --> D[인코더]
    B --> E[패턴/계수 학습]
    C --> F[평균/표준편차 계산]
    D --> G[범주 목록 학습]
```

## 4. predict vs transform

```mermaid
flowchart LR
    A[fit 완료] --> B{객체 유형}
    B -->|모델| C[predict]
    B -->|전처리기| D[transform]
    C --> E[예측값]
    D --> F[변환된 데이터]
```

## 5. 올바른 fit/transform 흐름

```mermaid
flowchart TD
    A[학습 데이터] --> B[fit_transform]
    B --> C[변환된 학습 데이터]
    D[테스트 데이터] --> E[transform만]
    E --> F[변환된 테스트 데이터]
    C --> G[모델 학습]
    F --> H[모델 예측]
```

## 6. 잘못된 방법

```mermaid
flowchart TD
    A[학습 데이터] --> B[fit + transform]
    D[테스트 데이터] --> E[fit + transform]
    B --> F[기준 A]
    E --> G[기준 B]
    F --> H[불일치!]
    G --> H
    style H fill:#f66
```

## 7. fit_transform 설명

```mermaid
flowchart LR
    A[fit_transform] --> B[fit]
    A --> C[transform]
    B --> D[내부 상태 저장]
    C --> E[데이터 변환]
```

## 8. 전체 ML 워크플로우

```mermaid
flowchart LR
    A[원본 데이터] --> B[스케일링]
    B --> C[인코딩]
    C --> D[모델 학습]
    D --> E[예측]
```

## 9. Pipeline 없이 수동 관리

```mermaid
flowchart TD
    A[scaler] --> B[encoder]
    B --> C[model]
    D[배포 시] --> E[순서 기억?]
    D --> F[기준 저장?]
    D --> G[모든 객체 로드?]
    style E fill:#ff9
    style F fill:#ff9
    style G fill:#ff9
```

## 10. Pipeline 구조

```mermaid
flowchart LR
    subgraph Pipeline
        A[Step 1: Scaler] --> B[Step 2: Model]
    end
    C[데이터] --> Pipeline
    Pipeline --> D[예측]
```

## 11. Pipeline fit() 내부 동작

```mermaid
flowchart TD
    A[pipeline.fit] --> B[step1.fit_transform]
    B --> C[step2.fit_transform]
    C --> D[stepN.fit]
    D --> E[학습 완료]
```

## 12. Pipeline predict() 내부 동작

```mermaid
flowchart TD
    A[pipeline.predict] --> B[step1.transform]
    B --> C[step2.transform]
    C --> D[stepN.predict]
    D --> E[예측 결과]
```

## 13. make_pipeline

```mermaid
flowchart LR
    A[make_pipeline] --> B[자동 이름 생성]
    B --> C[standardscaler]
    B --> D[logisticregression]
```

## 14. Pipeline 장점

```mermaid
mindmap
  root((Pipeline))
    코드 간결
      한 줄로 fit
      한 줄로 predict
    순서 보장
      전처리 순서 자동
    누출 방지
      fit/transform 자동
    배포 용이
      하나의 파일로 저장
```

## 15. ColumnTransformer 필요성

```mermaid
flowchart TD
    A[원본 데이터] --> B{컬럼 유형}
    B -->|수치형| C[StandardScaler]
    B -->|범주형| D[OneHotEncoder]
    C --> E[합치기]
    D --> E
    E --> F[모델 입력]
```

## 16. ColumnTransformer 구조

```mermaid
flowchart TD
    subgraph ColumnTransformer
        A[num: StandardScaler] --> D[결과 합침]
        B[cat: OneHotEncoder] --> D
    end
    E[age, fare] --> A
    F[sex, embarked] --> B
```

## 17. 전체 Pipeline + ColumnTransformer

```mermaid
flowchart LR
    subgraph Full Pipeline
        subgraph Preprocessor
            A[num: Scaler]
            B[cat: Encoder]
        end
        C[Classifier]
    end
    D[원본 데이터] --> Preprocessor
    Preprocessor --> C
    C --> E[예측]
```

## 18. GridSearchCV + Pipeline

```mermaid
flowchart TD
    A[GridSearchCV] --> B[Pipeline]
    B --> C[파라미터 조합 시도]
    C --> D[교차 검증]
    D --> E[최적 파라미터]
```

## 19. 모델 저장 및 배포

```mermaid
flowchart LR
    A[Pipeline] --> B[joblib.dump]
    B --> C[model.pkl]
    C --> D[joblib.load]
    D --> E[바로 predict 가능]
```

## 20. 핵심 규칙 3가지

```mermaid
flowchart TD
    A[sklearn 핵심 규칙] --> B[1. fit은 학습에서만]
    A --> C[2. transform은 동일 기준]
    A --> D[3. Pipeline으로 자동 관리]
```

## 21. 메서드 사용 가이드

```mermaid
flowchart TD
    A{데이터 유형} -->|학습| B[fit_transform]
    A -->|테스트| C[transform만]
    D{객체 유형} -->|모델| E[fit + predict]
    D -->|전처리기| F[fit + transform]
```

## 22. sklearn 학습 경로

```mermaid
flowchart LR
    A[번외 D<br>워크플로우] --> B[번외 E<br>특성 공학]
    B --> C[번외 F<br>sklearn 패턴]
    C --> D[12차시~<br>모델 심화]
```
