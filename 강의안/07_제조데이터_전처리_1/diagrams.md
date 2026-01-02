# [7차시] 제조 데이터 전처리 (1) - 다이어그램

## 1. 데이터 분석 프로젝트 시간 배분

```mermaid
pie title 프로젝트 시간 배분
    "데이터 수집·전처리" : 70
    "모델링" : 15
    "평가·배포" : 15
```

## 2. 전처리가 중요한 이유

```mermaid
flowchart LR
    subgraph 입력["입력"]
        A["좋은 데이터"] --> B["좋은 모델"]
        C["나쁜 데이터"] --> D["나쁜 모델"]
    end

    subgraph 결과["결과"]
        B --> E["정확한 예측"]
        D --> F["잘못된 예측"]
    end

    G["Garbage In, Garbage Out"]
```

## 3. 결측치 발생 원인

```mermaid
mindmap
  root((결측치 발생))
    센서 오류
      통신 두절
      배터리 방전
      장비 고장
    입력 누락
      수기 입력 실수
      필수 항목 미기입
    시스템 장애
      PLC 연결 끊김
      MES 오류
      DB 손상
    정상 결측
      특정 조건만 측정
      옵션 항목
```

## 4. 결측치 탐지 방법

```mermaid
flowchart TD
    A[DataFrame] --> B["df.isnull()"]
    B --> C["불리언 DataFrame"]

    C --> D["df.isnull().sum()<br>열별 결측치 수"]
    C --> E["df.isnull().mean()*100<br>결측 비율"]
    C --> F["df.info()<br>전체 정보"]

    D --> G[결측 현황 파악]
    E --> G
    F --> G
```

## 5. 결측치 처리 전략

```mermaid
flowchart TD
    A[결측치 발견] --> B{결측 비율?}

    B --> |"< 5%"| C[삭제 또는 대체]
    B --> |"5~20%"| D[대체 권장]
    B --> |"> 20%"| E[열 삭제 고려]

    C --> F["dropna()"]
    D --> G["fillna()"]

    G --> H["평균<br>mean()"]
    G --> I["중앙값<br>median()"]
    G --> J["최빈값<br>mode()"]
    G --> K["앞/뒤값<br>ffill/bfill"]
```

## 6. 결측치 대체 방법 선택

```mermaid
flowchart LR
    subgraph 수치형["수치형 데이터"]
        A["이상치 없음"] --> B["평균 사용"]
        C["이상치 있음"] --> D["중앙값 사용"]
    end

    subgraph 범주형["범주형 데이터"]
        E["범주형"] --> F["최빈값 사용"]
    end

    subgraph 시계열["시계열 데이터"]
        G["시계열"] --> H["앞/뒤 값 또는 보간"]
    end
```

## 7. 이상치 발생 원인

```mermaid
mindmap
  root((이상치 발생))
    측정 오류
      센서 고장
      캘리브레이션 불량
      노이즈
    입력 실수
      단위 오류
      오타
      복사 붙여넣기 오류
    실제 극단값
      공정 이탈
      설비 고장
      원자재 불량
    시스템 이상
      리셋 값
      오버플로우
      초기화 오류
```

## 8. IQR 이상치 탐지

```mermaid
flowchart TD
    A[데이터] --> B["Q1 = 25% 백분위수"]
    A --> C["Q3 = 75% 백분위수"]

    B --> D["IQR = Q3 - Q1"]
    C --> D

    D --> E["하한 = Q1 - 1.5×IQR"]
    D --> F["상한 = Q3 + 1.5×IQR"]

    E --> G{값 < 하한?}
    F --> H{값 > 상한?}

    G --> |예| I[이상치]
    H --> |예| I
    G --> |아니오| J[정상]
    H --> |아니오| J
```

## 9. Z-score 이상치 탐지

```mermaid
flowchart TD
    A["Z = (값 - 평균) / 표준편차"] --> B{"|Z| 값은?"}

    B --> |"≤ 2"| C["정상<br>(95% 이내)"]
    B --> |"2 < |Z| ≤ 3"| D["주의<br>(상위/하위 5%)"]
    B --> |"> 3"| E["이상치<br>(상위/하위 0.3%)"]

    C --> F[유지]
    D --> G[모니터링]
    E --> H[처리 필요]
```

## 10. 이상치 처리 전략

```mermaid
flowchart LR
    subgraph 처리방법["처리 방법"]
        A["삭제<br>df[~outliers]"]
        B["클리핑<br>df.clip(lower, upper)"]
        C["대체<br>median으로 대체"]
        D["플래그<br>is_outlier 열 추가"]
    end

    E[이상치 발견] --> F{분석 목적?}

    F --> |"평균 추정"| A
    F --> |"범위 제한"| B
    F --> |"값 보존 필요"| C
    F --> |"별도 분석"| D
```

## 11. 이상치 처리 의사결정

```mermaid
flowchart TD
    A[이상치 발견] --> B{오류인가?}

    B --> |예| C[수정 또는 제거]
    B --> |아니오| D{중요한 신호인가?}

    D --> |예| E[유지 및 별도 분석]
    D --> |아니오| F{분석 목적에 영향?}

    F --> |예| G[클리핑 또는 대체]
    F --> |아니오| H[유지]

    I["항상 도메인 전문가와 상의!"]
```

## 12. 전처리 워크플로우

```mermaid
flowchart TD
    A[원본 데이터] --> B[데이터 탐색]
    B --> C["결측치 탐지<br>isnull().sum()"]

    C --> D{결측치 있음?}
    D --> |예| E[결측치 처리]
    D --> |아니오| F[이상치 탐지]
    E --> F

    F --> G{이상치 있음?}
    G --> |예| H[이상치 처리]
    G --> |아니오| I[전후 비교]
    H --> I

    I --> J[정제된 데이터]
```

## 13. 강의 구조

```mermaid
gantt
    title 7차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)        :a1, 00:00, 2m
    전처리 중요성           :a2, after a1, 1m
    결측치 개념            :a3, after a2, 1.5m
    결측치 처리 전략        :a4, after a3, 1.5m
    이상치 개념            :a5, after a4, 1.5m
    이상치 탐지/처리        :a6, after a5, 1.5m
    주의사항              :a7, after a6, 1m

    section 실습편
    실습 소개             :b1, after a7, 2m
    결측치 데이터 생성      :b2, after b1, 2m
    결측치 탐지            :b3, after b2, 2m
    결측치 처리            :b4, after b3, 2m
    이상치 탐지 (IQR)       :b5, after b4, 3m
    이상치 탐지 (Z-score)   :b6, after b5, 2m
    이상치 처리            :b7, after b6, 2m
    전후 비교             :b8, after b7, 2m

    section 정리
    핵심 요약             :c1, after b8, 1.5m
    주의사항/예고          :c2, after c1, 1.5m
```
