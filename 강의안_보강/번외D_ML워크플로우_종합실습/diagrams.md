# 번외D: ML 워크플로우 종합 실습 - Mermaid 다이어그램

## 1. ML 워크플로우 전체 흐름

```mermaid
flowchart LR
    A[1. 문제 정의] --> B[2. 데이터 탐색]
    B --> C[3. 전처리]
    C --> D[4. 모델 학습]
    D --> E[5. 모델 평가]
    E --> F[6. 결과 해석]
    F -.-> A
```

## 2. 분류 vs 회귀 판단

```mermaid
flowchart TD
    A[예측 대상 확인] --> B{값의 종류는?}
    B -->|범주형| C[분류 Classification]
    B -->|연속형| D[회귀 Regression]
    C --> E[예: 불량/양품, 생존/사망]
    D --> F[예: 가격, 온도, 수량]
```

## 3. 타이타닉 문제 분석

```mermaid
flowchart TD
    A[타이타닉 생존 예측] --> B[예측 대상: Survived]
    B --> C{가능한 값}
    C --> D[0: 사망]
    C --> E[1: 생존]
    D --> F[이진 분류 문제]
    E --> F
```

## 4. 데이터 탐색 단계

```mermaid
flowchart TD
    A[데이터 로드] --> B[df.shape]
    B --> C[df.info - 타입, 결측치]
    C --> D[df.describe - 통계량]
    D --> E[타겟 분포 확인]
    E --> F[특성별 관계 분석]
```

## 5. 결측치 처리 전략

```mermaid
flowchart TD
    A[결측치 발견] --> B{결측 비율}
    B -->|낮음 5% 이하| C[삭제 가능]
    B -->|중간 5-30%| D[대체값 사용]
    B -->|높음 30% 이상| E[컬럼 제거 고려]
    D --> F{데이터 타입}
    F -->|수치형| G[평균/중앙값]
    F -->|범주형| H[최빈값]
```

## 6. 연속형 결측치 대체 비교

```mermaid
flowchart LR
    A[연속형 결측치] --> B[평균 Mean]
    A --> C[중앙값 Median]
    B --> D[이상치에 민감]
    C --> E[이상치에 강건]
    E --> F[권장]
```

## 7. 범주형 인코딩 방법

```mermaid
flowchart TD
    A[범주형 변수] --> B{인코딩 방법}
    B --> C[Label Encoding]
    B --> D[One-Hot Encoding]
    C --> E[순서 있는 범주에 적합]
    D --> F[순서 없는 범주에 적합]
    D --> G[sex → sex_male]
```

## 8. 데이터 분할 구조

```mermaid
flowchart LR
    A[전체 데이터<br>891건] --> B[학습 데이터<br>712건 80%]
    A --> C[테스트 데이터<br>179건 20%]
    B --> D[모델 학습]
    C --> E[성능 평가]
```

## 9. stratify 옵션 효과

```mermaid
flowchart TD
    A[원본 데이터<br>생존 38%] --> B{stratify=y}
    B -->|사용| C[학습: 38%<br>테스트: 38%]
    B -->|미사용| D[비율 달라질 수 있음]
    C --> E[권장]
```

## 10. 3가지 모델 비교

```mermaid
flowchart TD
    A[모델 선택] --> B[Logistic Regression]
    A --> C[Decision Tree]
    A --> D[Random Forest]
    B --> E[선형 분류, 빠름]
    C --> F[규칙 기반, 직관적]
    D --> G[앙상블, 높은 성능]
```

## 11. 모델 학습 흐름

```mermaid
sequenceDiagram
    participant D as 데이터
    participant M as 모델
    participant R as 결과
    D->>M: fit(X_train, y_train)
    Note over M: 학습 진행
    M->>R: predict(X_test)
    R->>R: 정확도 계산
```

## 12. 혼동 행렬 구조

```mermaid
flowchart TD
    A[혼동 행렬] --> B[TN: 진음성<br>사망 예측 → 실제 사망]
    A --> C[FP: 위양성<br>생존 예측 → 실제 사망]
    A --> D[FN: 위음성<br>사망 예측 → 실제 생존]
    A --> E[TP: 진양성<br>생존 예측 → 실제 생존]
```

## 13. 평가 지표 관계

```mermaid
flowchart TD
    A[예측 결과] --> B[정확도<br>TP+TN / 전체]
    A --> C[정밀도<br>TP / TP+FP]
    A --> D[재현율<br>TP / TP+FN]
    C --> E[F1 Score]
    D --> E
    E --> F[조화 평균]
```

## 14. 상황별 중요 지표

```mermaid
flowchart TD
    A[어떤 지표가 중요?] --> B{상황}
    B -->|불량 놓치면 안됨| C[재현율 Recall]
    B -->|잘못된 경보 비용| D[정밀도 Precision]
    B -->|균형 필요| E[F1 Score]
    C --> F[제조 불량 탐지]
    D --> G[고가 설비 점검]
```

## 15. 특성 중요도 해석

```mermaid
flowchart LR
    A[Random Forest] --> B[feature_importances_]
    B --> C[sex_male: 28%]
    B --> D[fare: 26%]
    B --> E[age: 24%]
    B --> F[pclass: 12%]
    C --> G[가장 중요]
```

## 16. 비즈니스 인사이트 도출

```mermaid
flowchart TD
    A[특성 중요도] --> B[성별이 가장 중요]
    B --> C[여성 우선 구조 정책]
    A --> D[요금/등급 중요]
    D --> E[상위 등급 객실 위치 유리]
    A --> F[나이 중요]
    F --> G[어린이 우선 구조]
```

## 17. 제조업 적용 매핑

```mermaid
flowchart LR
    subgraph 타이타닉
        A[성별]
        B[요금]
        C[나이]
    end
    subgraph 제조업
        D[설비 종류]
        E[온도]
        F[작업 시간]
    end
    A --> D
    B --> E
    C --> F
```

## 18. 전체 코드 흐름

```mermaid
flowchart TD
    A[import 라이브러리] --> B[데이터 로드]
    B --> C[EDA]
    C --> D[결측치 처리]
    D --> E[인코딩]
    E --> F[특성 선택]
    F --> G[train_test_split]
    G --> H[모델 학습]
    H --> I[예측]
    I --> J[평가]
    J --> K[해석]
```

## 19. 체크리스트 마인드맵

```mermaid
mindmap
  root((ML 워크플로우))
    문제 정의
      분류 vs 회귀
      타겟 변수 확인
    데이터 탐색
      shape, info
      describe
      분포 확인
    전처리
      결측치 처리
      인코딩
      특성 선택
    모델링
      여러 모델 비교
      최적 모델 선택
    평가
      정확도
      정밀도, 재현율
      F1 Score
    해석
      특성 중요도
      인사이트 도출
```

## 20. 학습 경로

```mermaid
flowchart LR
    A[번외 D<br>워크플로우 종합] --> B[번외 E<br>특성 공학]
    B --> C[번외 F<br>sklearn 패턴]
    C --> D[12차시~<br>개별 모델 심화]
```
