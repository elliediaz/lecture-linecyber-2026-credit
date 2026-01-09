# 번외E: 특성 공학 기초 실습 - Mermaid 다이어그램

## 1. 특성 공학 전체 흐름

```mermaid
flowchart LR
    A[원본 데이터] --> B[특성 공학]
    B --> C[개선된 특성]
    C --> D[더 좋은 모델]
```

## 2. 특성 공학의 종류

```mermaid
flowchart TD
    A[특성 공학] --> B[수치형 변환]
    A --> C[범주형 인코딩]
    A --> D[특성 생성]
    A --> E[특성 선택]
    B --> B1[로그 변환]
    B --> B2[표준화]
    B --> B3[구간화]
    C --> C1[Label]
    C --> C2[One-Hot]
    C --> C3[빈도]
    C --> C4[타겟]
    D --> D1[비율]
    D --> D2[날짜 분해]
    D --> D3[상호작용]
    E --> E1[상관관계]
    E --> E2[중요도]
```

## 3. 좋은 특성의 조건

```mermaid
flowchart LR
    A[좋은 특성] --> B[정보성<br>타겟과 관련]
    A --> C[독립성<br>중복 없음]
    A --> D[이해 가능<br>해석 가능]
```

## 4. 로그 변환 효과

```mermaid
flowchart LR
    A[원본 분포<br>치우침] --> B[log1p 변환]
    B --> C[변환 후<br>정규분포]
```

## 5. 비율 특성 생성

```mermaid
flowchart TD
    A[total_bill] --> C[tip_ratio]
    B[tip] --> C
    C --> D[tip / total_bill]
```

## 6. 1인당 금액 계산

```mermaid
flowchart TD
    A[total_bill] --> C[per_person]
    B[size] --> C
    C --> D[total_bill / size]
```

## 7. 구간화 방법 비교

```mermaid
flowchart TD
    A[구간화 Binning] --> B[pd.cut]
    A --> C[pd.qcut]
    B --> D[등간격<br>값 범위 균등]
    C --> E[등빈도<br>데이터 수 균등]
```

## 8. 빈도 인코딩 과정

```mermaid
flowchart LR
    A[day 컬럼] --> B[빈도 계산]
    B --> C[Sat: 0.36<br>Sun: 0.31<br>Thur: 0.26]
    C --> D[day_freq 컬럼]
```

## 9. 타겟 인코딩 과정

```mermaid
flowchart LR
    A[day 컬럼] --> B[타겟 평균 계산]
    B --> C[Sat: 2.99<br>Sun: 3.26<br>Thur: 2.77]
    C --> D[day_target 컬럼]
```

## 10. 타겟 인코딩 주의사항

```mermaid
flowchart TD
    A[타겟 인코딩] --> B{학습/테스트 분리}
    B -->|올바름| C[학습 데이터로만<br>평균 계산]
    B -->|잘못됨| D[전체 데이터로<br>평균 계산]
    D --> E[데이터 누출!]
```

## 11. 날짜 특성 추출

```mermaid
flowchart TD
    A[datetime] --> B[year]
    A --> C[month]
    A --> D[dayofweek]
    A --> E[hour]
    A --> F[is_weekend]
```

## 12. 주기적 특성 변환

```mermaid
flowchart LR
    A[month 1-12] --> B[삼각함수 변환]
    B --> C[month_sin]
    B --> D[month_cos]
    C --> E[12월과 1월 가깝게]
    D --> E
```

## 13. 특성 선택 방법

```mermaid
flowchart TD
    A[특성 선택] --> B[상관관계 기반]
    A --> C[중요도 기반]
    A --> D[분산 기반]
    B --> E[타겟과 상관 높은 것]
    C --> F[모델이 중요하다고 판단]
    D --> G[분산 낮은 것 제거]
```

## 14. 상관관계 기반 선택

```mermaid
flowchart LR
    A[특성들] --> B[상관계수 계산]
    B --> C[임계값 필터링]
    C --> D[선택된 특성]
```

## 15. 다중공선성 문제

```mermaid
flowchart TD
    A[total_bill] --> C[높은 상관관계]
    B[log_total_bill] --> C
    C --> D[다중공선성]
    D --> E[하나만 선택]
```

## 16. 성능 비교 실험

```mermaid
flowchart TD
    A[데이터] --> B[기본 특성]
    A --> C[공학 후 특성]
    B --> D[모델 학습]
    C --> E[모델 학습]
    D --> F[RMSE 비교]
    E --> F
```

## 17. 특성 공학 체크리스트

```mermaid
mindmap
  root((특성 공학))
    수치형
      로그 변환
      스케일링
      구간화
    범주형
      One-Hot
      빈도 인코딩
      타겟 인코딩
    생성
      비율 계산
      날짜 분해
      상호작용
    선택
      상관관계
      다중공선성
```

## 18. 전처리 vs 특성 공학

```mermaid
flowchart LR
    A[전처리] --> B[결측치 처리<br>스케일링]
    C[특성 공학] --> D[새 특성 생성<br>변환<br>선택]
    B --> E[깨끗한 데이터]
    D --> F[더 좋은 특성]
```

## 19. Tips 데이터 특성 공학 결과

```mermaid
flowchart TD
    subgraph 원본
        A[total_bill]
        B[tip]
        C[size]
        D[day]
    end
    subgraph 생성
        E[log_total_bill]
        F[tip_ratio]
        G[per_person]
        H[day_freq]
        I[is_weekend]
    end
    A --> E
    A --> F
    B --> F
    A --> G
    C --> G
    D --> H
    D --> I
```

## 20. 학습 경로

```mermaid
flowchart LR
    A[번외 D<br>워크플로우] --> B[번외 E<br>특성 공학]
    B --> C[번외 F<br>sklearn 패턴]
    C --> D[12차시~<br>모델 심화]
```
