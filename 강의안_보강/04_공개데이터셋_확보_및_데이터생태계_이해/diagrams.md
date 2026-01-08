# [4차시] 공개 데이터셋 확보 및 데이터 생태계 이해 - 다이어그램

## 1. 공공데이터 정의 요소

```mermaid
flowchart TD
    A["공공데이터"]

    A --> B["생성 주체"]
    A --> C["형태"]
    A --> D["목적"]

    B --> B1["정부 부처"]
    B --> B2["지방자치단체"]
    B --> B3["공공기관"]

    C --> C1["CSV/Excel"]
    C --> C2["API"]
    C --> C3["DB"]

    D --> D1["통계"]
    D --> D2["행정"]
    D --> D3["공공 서비스"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dbeafe
    style D fill:#dbeafe
```

## 2. 공공기관의 범위

```mermaid
flowchart LR
    A["공공데이터<br>제공 기관<br>(890+)"]

    A --> B["국가기관"]
    A --> C["지자체"]
    A --> D["공공기관"]
    A --> E["지방공기업"]
    A --> F["특수법인"]

    B --> B1["행정부처"]
    C --> C1["광역/기초"]
    D --> D1["공단/공사"]
    E --> E1["도시공사"]
    F --> F1["KOTRA"]

    style A fill:#1e40af,color:#fff
```

## 3. 공공데이터의 종류

```mermaid
flowchart TD
    A["공공데이터 유형"]

    A --> B["파일 데이터"]
    A --> C["오픈 API"]
    A --> D["표준 데이터"]
    A --> E["링크드 데이터"]

    B --> B1["CSV, Excel, JSON<br>다운로드 방식"]
    C --> C1["REST/SOAP<br>실시간 호출"]
    D --> D1["공통 규격<br>주소, 법인"]
    E --> E1["RDF/LOD<br>지식 그래프"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
```

## 4. 공공데이터 제공 5대 원칙

```mermaid
flowchart TD
    A["공공데이터<br>제공 원칙"]

    A --> B["1️⃣ 무료 제공"]
    A --> C["2️⃣ 기계 판독"]
    A --> D["3️⃣ 최신성 유지"]
    A --> E["4️⃣ 정확성 확보"]
    A --> F["5️⃣ 재이용 허용"]

    B --> B1["실비 외 무료"]
    C --> C1["CSV, JSON 포맷"]
    D --> D1["정기 업데이트"]
    E --> E1["오류 최소화"]
    F --> F1["상업적 이용 가능"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#dcfce7
    style E fill:#dcfce7
    style F fill:#dcfce7
```

## 5. AI 학습 데이터 파이프라인

```mermaid
flowchart LR
    A["공공데이터"] --> D["AI 학습셋"]
    B["민간데이터"] --> D
    C["자체 생성"] --> D

    A --> A1["인프라/통계"]
    B --> B1["기업/고객"]
    C --> C1["센서/실험"]

    D --> E["AI 모델"]

    style A fill:#dbeafe
    style B fill:#fef3c7
    style C fill:#dcfce7
    style D fill:#1e40af,color:#fff
```

## 6. 공공데이터 활용 예시

```mermaid
flowchart TD
    subgraph 공공데이터
        A1["기상 데이터"]
        A2["경기 지수"]
        A3["교통 혼잡도"]
    end

    subgraph AI_프로젝트
        B1["품질 예측"]
        B2["수요 예측"]
        B3["물류 최적화"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3

    style A1 fill:#dbeafe
    style A2 fill:#dbeafe
    style A3 fill:#dbeafe
    style B1 fill:#dcfce7
    style B2 fill:#dcfce7
    style B3 fill:#dcfce7
```

## 7. 국내 4대 데이터 포털

```mermaid
flowchart TD
    A["데이터 포털"]

    A --> B["공공데이터포털<br>data.go.kr"]
    A --> C["AI 허브<br>aihub.or.kr"]
    A --> D["Kaggle<br>kaggle.com"]
    A --> E["통계청 KOSIS"]

    B --> B1["45만+<br>정부 공식"]
    C --> C1["700+<br>AI 학습용"]
    D --> D1["20만+<br>글로벌 ML"]
    E --> E1["100만+<br>공식 통계"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 8. 공공데이터포털 데이터 구성

```mermaid
flowchart TD
    A["공공데이터포털<br>(data.go.kr)"]

    A --> B["파일 데이터<br>38만+"]
    A --> C["오픈 API<br>7만+"]
    A --> D["표준 데이터<br>300+"]

    B --> B1["CSV, Excel<br>다운로드"]
    C --> C1["REST API<br>키 발급"]
    D --> D1["주소, 법인<br>공통 규격"]

    style A fill:#1e40af,color:#fff
```

## 9. AI 허브 데이터 카테고리

```mermaid
flowchart TD
    A["AI 허브"]

    A --> B["시각지능"]
    A --> C["언어지능"]
    A --> D["융합"]

    B --> B1["이미지"]
    B --> B2["영상"]

    C --> C1["텍스트"]
    C --> C2["음성"]

    D --> D1["센서"]
    D --> D2["시계열"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 10. AI 허브 제조 데이터셋

```mermaid
flowchart LR
    A["제조 AI<br>데이터셋"]

    A --> B["공정 이상탐지<br>10만+"]
    A --> C["용접 결함<br>5만+ 이미지"]
    A --> D["설비 진동<br>100만+"]
    A --> E["PCB 불량<br>3만+ 이미지"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#dcfce7
    style E fill:#dcfce7
```

## 11. AI 허브 데이터 패키지 구조

```mermaid
flowchart TD
    A["AI_HUB_DATASET/"]

    A --> B["01_원천데이터/"]
    A --> C["02_라벨링데이터/"]
    A --> D["03_메타데이터/"]
    A --> E["README.md"]

    B --> B1["images/"]
    B --> B2["sensor_data/"]

    C --> C1["annotations.json"]
    C --> C2["labels.csv"]

    D --> D1["metadata.json"]

    style A fill:#1e40af,color:#fff
```

## 12. Kaggle 핵심 기능

```mermaid
flowchart TD
    A["Kaggle"]

    A --> B["Datasets<br>데이터셋"]
    A --> C["Competitions<br>경진대회"]
    A --> D["Notebooks<br>코드 공유"]
    A --> E["Discussions<br>커뮤니티"]

    B --> B1["20만+"]
    C --> C1["상금"]
    D --> D1["바로 실행"]
    E --> E1["Q&A"]

    style A fill:#1e40af,color:#fff
```

## 13. 포털 선택 가이드

```mermaid
flowchart TD
    A{목적?}

    A -->|제조 AI 학습| B["AI 허브"]
    A -->|경제 분석| C["공공데이터포털"]
    A -->|알고리즘 학습| D["UCI/Kaggle"]
    A -->|실시간 연동| E["공공데이터포털 API"]
    A -->|글로벌 트렌드| F["Kaggle"]

    style A fill:#fef3c7
    style B fill:#dcfce7
    style C fill:#dbeafe
    style D fill:#dbeafe
    style E fill:#dbeafe
    style F fill:#dbeafe
```

## 14. 테이블 데이터 구조

```mermaid
flowchart TD
    subgraph 열_Column
        C1["온도"]
        C2["습도"]
        C3["압력"]
        C4["불량여부"]
    end

    subgraph 행_Row
        R1["85.2, 52, 1.01, 0"]
        R2["87.5, 48, 1.03, 0"]
        R3["92.1, 55, 0.98, 1"]
    end

    C1 --> R1
    C2 --> R1
    C3 --> R1
    C4 --> R1

    style C1 fill:#dbeafe
    style C2 fill:#dbeafe
    style C3 fill:#dbeafe
    style C4 fill:#fef3c7
```

## 15. 행(Row) 관련 용어

```mermaid
flowchart LR
    A["행<br>(Row)"]

    A --> B["레코드<br>Record"]
    A --> C["샘플<br>Sample"]
    A --> D["관측치<br>Observation"]
    A --> E["인스턴스<br>Instance"]

    style A fill:#1e40af,color:#fff
```

## 16. 열(Column) 관련 용어

```mermaid
flowchart LR
    A["열<br>(Column)"]

    A --> B["변수<br>Variable"]
    A --> C["특성<br>Feature"]
    A --> D["속성<br>Attribute"]
    A --> E["필드<br>Field"]

    style A fill:#1e40af,color:#fff
```

## 17. 독립변수 vs 종속변수

```mermaid
flowchart TD
    subgraph 독립변수_X
        X1["온도"]
        X2["습도"]
        X3["압력"]
    end

    subgraph 종속변수_y
        Y1["불량여부"]
    end

    X1 --> M["모델"]
    X2 --> M
    X3 --> M
    M --> Y1

    style X1 fill:#dbeafe
    style X2 fill:#dbeafe
    style X3 fill:#dbeafe
    style Y1 fill:#dcfce7
    style M fill:#fef3c7
```

## 18. 변수의 데이터 타입

```mermaid
flowchart TD
    A["변수 타입"]

    A --> B["수치형<br>Numerical"]
    A --> C["범주형<br>Categorical"]

    B --> B1["연속형<br>온도 85.234"]
    B --> B2["이산형<br>불량수 0,1,2"]

    C --> C1["명목형<br>라인 A,B,C"]
    C --> C2["순서형<br>등급 상>중>하"]
    C --> C3["이진형<br>불량 0/1"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 19. 데이터 품질 이슈

```mermaid
flowchart TD
    A["데이터 품질"]

    A --> B["결측치<br>Missing"]
    A --> C["이상치<br>Outlier"]
    A --> D["중복<br>Duplicate"]
    A --> E["노이즈<br>Noise"]
    A --> F["불균형<br>Imbalance"]

    B --> B1["NaN, null"]
    C --> C1["비정상 값"]
    D --> D1["반복 레코드"]
    E --> E1["의미없는 변동"]
    F --> F1["클래스 비율"]

    style A fill:#fecaca
```

## 20. 데이터셋 분할

```mermaid
flowchart LR
    A["전체 데이터<br>100%"]

    A --> B["훈련 세트<br>70%"]
    A --> C["검증 세트<br>15%"]
    A --> D["테스트 세트<br>15%"]

    B --> B1["모델 학습"]
    C --> C1["하이퍼파라미터<br>튜닝"]
    D --> D1["최종 평가<br>(1회만)"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 21. CSV 파일 구조

```mermaid
flowchart TD
    A["CSV 파일"]

    A --> B["Header<br>(첫 행)"]
    A --> C["Delimiter<br>(구분자)"]
    A --> D["Encoding<br>(인코딩)"]

    B --> B1["온도,습도,압력"]
    C --> C1["쉼표(,) 또는 탭(\\t)"]
    D --> D1["UTF-8 또는 CP949"]

    style A fill:#1e40af,color:#fff
```

## 22. JSON 파일 구조

```mermaid
flowchart TD
    A["JSON 파일"]

    A --> B["Key<br>(키)"]
    A --> C["Value<br>(값)"]
    A --> D["Nested<br>(중첩)"]

    B --> B1["'temperature'"]
    C --> C1["85.2"]
    D --> D1["{'sensor': {'temp': 85}}"]

    style A fill:#1e40af,color:#fff
```

## 23. 메타데이터 구조

```mermaid
flowchart TD
    A["메타데이터"]

    A --> B["name<br>데이터 이름"]
    A --> C["version<br>버전"]
    A --> D["rows/columns<br>크기"]
    A --> E["schema<br>컬럼 정의"]
    A --> F["license<br>라이선스"]

    style A fill:#1e40af,color:#fff
```

## 24. 라이선스 비교

```mermaid
flowchart TD
    A["데이터 라이선스"]

    A --> B["CC0<br>퍼블릭도메인"]
    A --> C["CC-BY"]
    A --> D["CC-BY-NC"]
    A --> E["공공누리"]

    B --> B1["자유 이용"]
    C --> C1["출처 표시"]
    D --> D1["비영리만"]
    E --> E1["유형별 상이"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#dbeafe
```

## 25. 데이터 다운로드 흐름

```mermaid
flowchart TD
    A["포털 접속"]
    B["회원 가입/로그인"]
    C["데이터 검색"]
    D["다운로드/API 신청"]
    E["데이터 확보"]
    F["구조 확인"]
    G["분석 시작"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style E fill:#dcfce7
    style G fill:#1e40af,color:#fff
```

## 26. 데이터 구조 확인 체크리스트

```mermaid
flowchart TD
    A["데이터 확인"]

    A --> B["1. shape<br>크기"]
    A --> C["2. columns<br>컬럼 목록"]
    A --> D["3. dtypes<br>데이터 타입"]
    A --> E["4. isnull<br>결측치"]
    A --> F["5. describe<br>기술통계"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
    style C fill:#dcfce7
    style D fill:#dcfce7
    style E fill:#fef3c7
    style F fill:#dcfce7
```

## 27. 용어 매핑

```mermaid
flowchart LR
    subgraph 한국어
        K1["행"]
        K2["열"]
        K3["독립변수"]
        K4["종속변수"]
    end

    subgraph 영어
        E1["Row/Record"]
        E2["Column/Feature"]
        E3["X/Input"]
        E4["y/Target"]
    end

    K1 --> E1
    K2 --> E2
    K3 --> E3
    K4 --> E4

    style K1 fill:#dbeafe
    style K2 fill:#dbeafe
    style K3 fill:#dbeafe
    style K4 fill:#dbeafe
```

## 28. 4차시 학습 흐름

```mermaid
flowchart LR
    A["공공데이터<br>정의"]
    B["포털<br>특성"]
    C["데이터<br>용어"]
    D["실습:<br>다운로드"]
    E["5차시:<br>시각화"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#dcfce7
    style E fill:#1e40af,color:#fff
```
