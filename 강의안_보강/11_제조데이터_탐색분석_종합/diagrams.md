# [11차시] 제조 데이터 탐색 분석 종합 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["EDA<br>전체 흐름"]
    B["인사이트<br>도출"]
    C["종합 분석<br>프로젝트"]
    D["11차시:<br>머신러닝 소개"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. Part II 학습 여정

```mermaid
flowchart TD
    A["Part II: 기초 수리와 데이터 분석"]

    A --> B["5차시: 기술통계/시각화"]
    A --> C["6차시: 확률분포/검정"]
    A --> D["7차시: 상관분석/회귀"]
    A --> E["8차시: 전처리 1"]
    A --> F["9차시: 전처리 2"]
    A --> G["11차시: EDA 종합"]

    style A fill:#1e40af,color:#fff
    style G fill:#dcfce7
```

## 3. EDA란?

```mermaid
flowchart TD
    A["EDA<br>Exploratory Data Analysis"]

    A --> B["목적"]
    B --> B1["데이터 이해"]
    B --> B2["품질 확인"]
    B --> B3["패턴 발견"]
    B --> B4["가설 수립"]

    style A fill:#1e40af,color:#fff
```

## 4. EDA vs 확증적 분석

```mermaid
flowchart LR
    subgraph EDA["EDA (탐색적)"]
        A1["가설 생성"]
        A2["시각화, 요약"]
        A3["무엇이 있나?"]
    end

    subgraph CDA["CDA (확증적)"]
        B1["가설 검증"]
        B2["통계적 검정"]
        B3["가설이 맞나?"]
    end

    EDA -->|"먼저"| CDA

    style EDA fill:#dbeafe
    style CDA fill:#dcfce7
```

## 5. EDA 5단계

```mermaid
flowchart TD
    A["1단계: 데이터 개요"]
    B["2단계: 단변량 분석"]
    C["3단계: 이변량 분석"]
    D["4단계: 다변량 분석"]
    E["5단계: 인사이트 도출"]

    A --> B --> C --> D --> E

    A --> A1["shape, dtypes, head"]
    B --> B1["각 변수 분포"]
    C --> C1["두 변수 관계"]
    D --> D1["여러 변수 동시"]
    E --> E1["발견 정리"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 6. 1단계: 데이터 개요

```mermaid
flowchart TD
    A["1단계: 데이터 개요"]

    A --> B["df.shape"]
    A --> C["df.dtypes"]
    A --> D["df.head()"]
    A --> E["df.info()"]
    A --> F["df.describe()"]
    A --> G["df.isnull().sum()"]

    B --> B1["행, 열 수"]
    C --> C1["데이터 타입"]
    F --> F1["기술통계"]
    G --> G1["결측치 현황"]

    style A fill:#1e40af,color:#fff
```

## 7. 2단계: 단변량 분석

```mermaid
flowchart TD
    A["2단계: 단변량 분석"]

    A --> B["수치형"]
    A --> C["범주형"]

    B --> B1["히스토그램"]
    B --> B2["박스플롯"]
    B --> B3["기술통계"]

    C --> C1["value_counts()"]
    C --> C2["막대그래프"]

    style A fill:#1e40af,color:#fff
```

## 8. 3단계: 이변량 분석

```mermaid
flowchart TD
    A["3단계: 이변량 분석"]

    A --> B["수치 vs 수치"]
    A --> C["범주 vs 수치"]
    A --> D["범주 vs 범주"]

    B --> B1["산점도"]
    B --> B2["상관계수"]

    C --> C1["그룹별 박스플롯"]
    C --> C2["t-검정"]

    D --> D1["교차표"]
    D --> D2["히트맵"]

    style A fill:#1e40af,color:#fff
```

## 9. 4단계: 다변량 분석

```mermaid
flowchart TD
    A["4단계: 다변량 분석"]

    A --> B["상관행렬"]
    A --> C["피벗 테이블"]
    A --> D["그룹별 집계"]

    B --> B1["df.corr()"]
    C --> C1["pivot_table()"]
    D --> D1["groupby().agg()"]

    style A fill:#1e40af,color:#fff
```

## 10. 인사이트 3요소

```mermaid
flowchart TD
    A["좋은 인사이트"]

    A --> B["구체적"]
    A --> C["실행 가능"]
    A --> D["데이터 기반"]

    B --> B1["명확한 수치와 근거"]
    C --> C1["행동으로 연결 가능"]
    D --> D1["통계적 뒷받침"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
    style D fill:#fef3c7
```

## 11. 인사이트 비교

```mermaid
flowchart LR
    subgraph 나쁜["나쁜 인사이트"]
        A1["불량률이 높다"]
        A2["품질 개선 필요"]
        A3["아마 그럴 것 같다"]
    end

    subgraph 좋은["좋은 인사이트"]
        B1["B라인 불량률 5.1%<br>(평균 대비 +1.9%p)"]
        B2["85°C 초과 시<br>알람 설정 권장"]
        B3["r=0.42, p<0.05<br>통계적 유의"]
    end

    style 나쁜 fill:#fecaca
    style 좋은 fill:#dcfce7
```

## 12. 인사이트 도출 프레임워크

```mermaid
flowchart TD
    A["1. 패턴 발견"]
    B["2. 수치화"]
    C["3. 통계 검증"]
    D["4. 원인 추론"]
    E["5. 행동 제안"]

    A --> B --> C --> D --> E

    A --> A1["B라인 불량률 높아 보임"]
    B --> B1["5.1% vs 평균 3.2%"]
    C --> C1["p-value: 0.003"]
    D --> D1["노후 장비? 오후 가동?"]
    E --> E1["장비 점검, 모니터링 강화"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 13. 인사이트 유형

```mermaid
flowchart TD
    A["제조 인사이트 유형"]

    A --> B["영향 요인"]
    A --> C["임계점"]
    A --> D["그룹 차이"]
    A --> E["시간 패턴"]
    A --> F["조합 효과"]

    B --> B1["온도가 42% 설명"]
    C --> C1["85°C 초과 시 급증"]
    D --> D1["A vs B 유의미 차이"]
    E --> E1["오후 15% 높음"]
    F --> F1["고온+고습 3배"]

    style A fill:#1e40af,color:#fff
```

## 14. 가설 수립

```mermaid
flowchart TD
    A["인사이트"]
    B["가설"]
    C["검증"]

    A --> B --> C

    B --> B1["원인이 결과에<br>영향을 미친다"]
    B --> B2["측정 가능"]
    B --> B3["반증 가능"]
    B --> B4["구체적"]

    style A fill:#dbeafe
    style B fill:#fef3c7
    style C fill:#dcfce7
```

## 15. EDA 체크리스트

```mermaid
flowchart TD
    A["EDA 체크리스트"]

    A --> B["필수 항목"]

    B --> B1["데이터 크기/구조"]
    B --> B2["결측치 현황"]
    B --> B3["이상치 현황"]
    B --> B4["각 변수 분포"]
    B --> B5["타겟과의 관계"]
    B --> B6["변수 간 상관"]
    B --> B7["시간적 패턴"]
    B --> B8["그룹별 차이"]

    style A fill:#fecaca
    style B5 fill:#fef3c7
```

## 16. 실습 시나리오

```mermaid
flowchart TD
    A["품질관리팀 요청"]

    A --> B["문제"]
    A --> C["데이터"]
    A --> D["분석 목표"]

    B --> B1["불량률 증가"]
    C --> C1["6개월 생산 데이터<br>1000건"]
    D --> D1["요인 분석<br>라인별/시간대별 차이"]

    style A fill:#1e40af,color:#fff
```

## 17. 실습 데이터 구조

```mermaid
flowchart TD
    A["실습 데이터"]

    A --> B["날짜"]
    A --> C["라인 (A/B/C)"]
    A --> D["시간대 (주간/야간)"]
    A --> E["온도"]
    A --> F["습도"]
    A --> G["생산량"]
    A --> H["불량률"]

    style A fill:#1e40af,color:#fff
    style H fill:#fef3c7
```

## 18. EDA 워크플로우

```mermaid
flowchart TD
    A["1. 데이터 로드"]
    B["2. 품질 확인"]
    C["3. 단변량 분석"]
    D["4. 이변량 분석"]
    E["5. 다변량 분석"]
    F["6. 인사이트 정리"]
    G["7. 보고서 작성"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style G fill:#dcfce7
```

## 19. 대시보드 구성

```mermaid
flowchart TD
    subgraph 대시보드["분석 대시보드"]
        A["불량률 분포"]
        B["라인별 불량률"]
        C["온도 vs 불량률"]
        D["상관행렬"]
    end

    style 대시보드 fill:#fef3c7
```

## 20. 보고서 구조

```mermaid
flowchart TD
    A["EDA 보고서"]

    A --> B["1. 분석 개요"]
    A --> C["2. 데이터 품질"]
    A --> D["3. 주요 발견"]
    A --> E["4. 결론 및 권고"]

    B --> B1["목적, 데이터 설명"]
    C --> C1["결측치, 이상치"]
    D --> D1["인사이트 3-5개<br>시각화"]
    E --> E1["실행 가능한 제안"]

    style A fill:#1e40af,color:#fff
```

## 21. 권고사항 분류

```mermaid
flowchart TD
    A["권고사항"]

    A --> B["즉시 실행"]
    A --> C["단기 개선"]
    A --> D["장기 개선"]

    B --> B1["B라인 긴급 점검"]
    B --> B2["온도 경계값 알람"]

    C --> C1["야간 모니터링 강화"]
    C --> C2["환경 관리 개선"]

    D --> D1["예측 모델 개발"]
    D --> D2["센서 추가 설치"]

    style B fill:#fecaca
    style C fill:#fef3c7
    style D fill:#dcfce7
```

## 22. Part II 역량 정리

```mermaid
flowchart TD
    A["Part II 역량"]

    A --> B["기술통계"]
    A --> C["시각화"]
    A --> D["통계검정"]
    A --> E["상관분석"]
    A --> F["전처리"]
    A --> G["EDA"]

    B --> B1["데이터 요약"]
    C --> C1["분포 확인"]
    D --> D1["차이 검증"]
    E --> E1["관계 파악"]
    F --> F1["데이터 정제"]
    G --> G1["인사이트 도출"]

    style A fill:#1e40af,color:#fff
```

## 23. 시각화 원칙

```mermaid
flowchart TD
    A["효과적인 시각화"]

    A --> B["단순화"]
    A --> C["강조"]
    A --> D["맥락"]

    B --> B1["핵심 메시지 하나"]
    C --> C1["중요 부분 하이라이트"]
    D --> D1["비교 기준 제공"]

    style A fill:#1e40af,color:#fff
```

## 24. 인사이트 우선순위

```mermaid
flowchart TD
    A["우선순위 평가"]

    A --> B["영향력"]
    A --> C["실행 가능성"]
    A --> D["신뢰도"]
    A --> E["비용"]

    B --> B1["타겟에 큰 영향"]
    C --> C1["바로 적용 가능"]
    D --> D1["통계적 유의"]
    E --> E1["개선 비용"]

    style A fill:#1e40af,color:#fff
```

## 25. 분석 결과 정리 형식

```mermaid
flowchart TD
    A["인사이트 문서화"]

    A --> B["제목"]
    A --> C["발견 내용"]
    A --> D["수치적 근거"]
    A --> E["통계적 검증"]
    A --> F["권고 사항"]

    B --> B1["라인 B 불량률 이상"]
    C --> C1["타 라인 대비 높음"]
    D --> D1["5.1% vs 3.2%"]
    E --> E1["p-value: 0.003"]
    F --> F1["장비 점검 실시"]

    style A fill:#1e40af,color:#fff
```

## 26. Part II → Part III 연결

```mermaid
flowchart LR
    A["Part II<br>데이터 분석"]
    B["Part III<br>머신러닝"]

    A --> B

    A --> A1["기술통계"]
    A --> A2["전처리"]
    A --> A3["EDA"]

    B --> B1["분류"]
    B --> B2["회귀"]
    B --> B3["모델 평가"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 27. 핵심 정리

```mermaid
flowchart TD
    A["11차시 핵심"]

    A --> B["EDA 5단계"]
    A --> C["인사이트 3요소"]
    A --> D["체크리스트"]

    B --> B1["개요 → 단변량 →<br>이변량 → 다변량 →<br>인사이트"]
    C --> C1["구체적<br>실행 가능<br>데이터 기반"]
    D --> D1["타겟 관계<br>반드시 확인"]

    style A fill:#1e40af,color:#fff
```

## 28. 다음 차시 예고

```mermaid
flowchart LR
    A["11차시"]
    B["11차시"]

    A --> B

    A --> A1["EDA 종합"]
    B --> B1["머신러닝 소개"]
    B --> B2["지도/비지도학습"]
    B --> B3["분류 vs 회귀"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
