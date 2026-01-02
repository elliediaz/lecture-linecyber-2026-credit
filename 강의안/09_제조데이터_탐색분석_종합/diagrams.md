# [9차시] 제조 데이터 탐색 분석 종합 - 다이어그램

## 1. EDA 5단계 워크플로우

```mermaid
flowchart TD
    A[1. 데이터 개요 파악] --> B[2. 단변량 분석]
    B --> C[3. 이변량 분석]
    C --> D[4. 다변량 분석]
    D --> E[5. 인사이트 도출]

    A --> |"shape, info, describe"| A1[크기, 타입, 통계]
    B --> |"히스토그램, 상자그림"| B1[분포, 이상치]
    C --> |"산점도, 상관계수"| C1[관계, 그룹 비교]
    D --> |"피벗, 히트맵"| D1[복합 패턴]
    E --> |"구체적, 실행 가능"| E1[권고사항]
```

## 2. 1단계: 데이터 개요

```mermaid
flowchart LR
    subgraph 개요[1단계: 개요 파악]
        A["df.shape<br>크기 확인"]
        B["df.dtypes<br>데이터 타입"]
        C["df.head()<br>샘플 보기"]
        D["df.info()<br>요약 정보"]
        E["df.describe()<br>기술통계"]
        F["df.isnull()<br>결측치"]
    end

    G[데이터와 첫 만남!]
    A --> G
    B --> G
    C --> G
    D --> G
    E --> G
    F --> G
```

## 3. 2단계: 단변량 분석

```mermaid
flowchart TD
    A[단변량 분석] --> B{데이터 유형?}

    B --> |수치형| C[분포 분석]
    B --> |범주형| D[빈도 분석]

    C --> C1[히스토그램]
    C --> C2[상자그림]
    C --> C3[기술통계]

    D --> D1["value_counts()"]
    D --> D2[막대그래프]

    C1 --> E[분포 형태 파악]
    C2 --> F[이상치 발견]
    D1 --> G[범주 비율 파악]
```

## 4. 3단계: 이변량 분석

```mermaid
flowchart LR
    subgraph 수치수치["수치 vs 수치"]
        A1[산점도] --> A2[상관계수]
    end

    subgraph 범주수치["범주 vs 수치"]
        B1[그룹별 상자그림] --> B2[평균 비교]
    end

    subgraph 범주범주["범주 vs 범주"]
        C1[교차표] --> C2[히트맵]
    end

    D[이변량 분석]
    D --> 수치수치
    D --> 범주수치
    D --> 범주범주
```

## 5. 4단계: 다변량 분석

```mermaid
flowchart TD
    A[다변량 분석] --> B[상관행렬]
    A --> C[피벗 테이블]
    A --> D[그룹별 집계]

    B --> B1["df.corr()"]
    C --> C1["pivot_table()"]
    D --> D1["groupby().agg()"]

    B1 --> E[변수 간 관계 전체 조망]
    C1 --> F[2차원 교차 분석]
    D1 --> G[다중 조건 집계]
```

## 6. 좋은 인사이트 vs 나쁜 인사이트

```mermaid
flowchart LR
    subgraph 나쁜["❌ 나쁜 예"]
        A1["불량률이 높다"]
        A2["품질을 개선해야 한다"]
        A3["아마 그럴 것 같다"]
    end

    subgraph 좋은["✅ 좋은 예"]
        B1["라인 B 불량률이<br>평균 대비 2%p 높음"]
        B2["온도 85°C 이상 시<br>경고 알람 설정"]
        B3["상관계수 0.42,<br>p<0.05로 유의"]
    end

    C[구체적] --> 좋은
    D[실행 가능] --> 좋은
    E[데이터 기반] --> 좋은
```

## 7. EDA 체크리스트

```mermaid
mindmap
  root((EDA 체크리스트))
    데이터 품질
      결측치 확인
      이상치 확인
      타입 확인
    단변량
      각 변수 분포
      수치형 통계
      범주형 빈도
    이변량
      타겟과의 관계
      상관관계
      그룹별 차이
    다변량
      복합 패턴
      시간적 추이
      교호작용
```

## 8. 분석 시나리오: 품질 문제

```mermaid
flowchart TD
    A[품질관리팀 요청] --> B["불량률 증가 원인 분석"]

    B --> C[1. 데이터 확인]
    C --> D[2. 불량률 분포 확인]
    D --> E[3. 라인별 비교]
    E --> F[4. 시간대별 비교]
    F --> G[5. 온도 영향 분석]
    G --> H[6. 통계 검정]
    H --> I[7. 인사이트 정리]
    I --> J[권고사항 도출]

    J --> K["라인 B 집중 점검"]
    J --> L["온도 모니터링 강화"]
    J --> M["오후 품질관리 강화"]
```

## 9. 불량률 요인 분석 결과

```mermaid
pie title 불량률 영향 요인 (가상)
    "라인 차이" : 40
    "온도 영향" : 30
    "시간대 차이" : 20
    "기타" : 10
```

## 10. Part II 종합

```mermaid
flowchart LR
    subgraph PartII["Part II: 기초 수리와 데이터 분석"]
        A["4차시<br>요약과 시각화"]
        B["5차시<br>확률분포와 검정"]
        C["6차시<br>상관분석과 회귀"]
        D["7차시<br>전처리 1"]
        E["8차시<br>전처리 2"]
        F["9차시<br>EDA 종합"]
    end

    A --> B --> C --> D --> E --> F

    F --> G[Part III: 머신러닝]
```

## 11. EDA 보고서 구조

```mermaid
flowchart TD
    A[EDA 보고서] --> B[1. 분석 개요]
    A --> C[2. 데이터 품질]
    A --> D[3. 주요 발견]
    A --> E[4. 결론 및 권고]

    B --> B1["데이터 설명<br>분석 목적"]
    C --> C1["결측치 현황<br>이상치 처리"]
    D --> D1["핵심 인사이트 3~5개<br>시각화 자료"]
    E --> E1["실행 가능한 제안<br>다음 단계"]
```

## 12. 강의 구조

```mermaid
gantt
    title 9차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/Part II 마무리)  :a1, 00:00, 2m
    EDA란?                    :a2, after a1, 1.5m
    EDA 5단계                 :a3, after a2, 2m
    좋은 인사이트              :a4, after a3, 1.5m
    EDA 체크리스트            :a5, after a4, 1.5m
    Part II 복습              :a6, after a5, 1.5m

    section 실습편
    실습 소개 (시나리오)        :b1, after a6, 2m
    데이터 개요                :b2, after b1, 2m
    단변량 분석               :b3, after b2, 2m
    범주별 분석               :b4, after b3, 2m
    상관관계 분석             :b5, after b4, 2m
    다변량 분석               :b6, after b5, 2m
    통계 검정                 :b7, after b6, 2m
    인사이트 정리             :b8, after b7, 2m

    section 정리
    핵심 요약                 :c1, after b8, 1.5m
    다음 차시 예고             :c2, after c1, 1.5m
```

## 13. Part II → Part III 연결

```mermaid
flowchart LR
    subgraph PartII["Part II 역량"]
        A[데이터 이해]
        B[시각화]
        C[전처리]
        D[EDA]
    end

    subgraph PartIII["Part III 활용"]
        E[특성 선택]
        F[모델 평가]
        G[결과 해석]
        H[개선 방향]
    end

    A --> E
    B --> F
    C --> G
    D --> H

    I["Part II가 없으면<br>Part III도 없다!"]
```
