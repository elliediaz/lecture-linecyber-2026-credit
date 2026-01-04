# [4차시] 공개 데이터셋 확보 및 데이터 생태계 이해 - 다이어그램

## 1. AI 학습 핵심 요소

```mermaid
flowchart LR
    A["데이터 수집"]
    B["전처리"]
    C["모델 학습"]
    D["평가"]

    A --> B --> C --> D

    style A fill:#dbeafe
```

## 2. 공개 데이터 생태계

```mermaid
flowchart TD
    A["공개 데이터"]

    A --> B["공공데이터<br>(정부/공공기관)"]
    A --> C["연구 데이터<br>(학술 목적)"]
    A --> D["경진대회 데이터<br>(AI 대회)"]
    A --> E["AI 학습용 데이터<br>(전문 플랫폼)"]

    style A fill:#dbeafe
```

## 3. 주요 데이터 플랫폼 4가지

```mermaid
flowchart LR
    subgraph 국내["국내 플랫폼"]
        A["공공데이터포털<br>data.go.kr"]
        B["AI 허브<br>aihub.or.kr"]
    end

    subgraph 해외["해외 플랫폼"]
        C["Kaggle<br>kaggle.com"]
        D["UCI Repository<br>archive.ics.uci.edu"]
    end
```

## 4. 공공데이터포털 특징

```mermaid
flowchart TD
    A["공공데이터포털"]

    A --> B["정부 공식 포털"]
    A --> C["43만+ 데이터셋"]
    A --> D["무료 사용"]
    A --> E["API 키 발급"]

    style A fill:#dbeafe
```

## 5. AI 허브 특징

```mermaid
flowchart TD
    A["AI 허브"]

    A --> B["NIA 운영"]
    A --> C["AI 학습 전문"]
    A --> D["이미지/텍스트/음성"]
    A --> E["제조 특화 데이터"]

    style A fill:#dcfce7
```

## 6. Kaggle 특징

```mermaid
flowchart TD
    A["Kaggle"]

    A --> B["세계 최대 커뮤니티"]
    A --> C["20만+ 데이터셋"]
    A --> D["경진대회"]
    A --> E["노트북 공유"]

    style A fill:#fef3c7
```

## 7. UCI Repository 특징

```mermaid
flowchart TD
    A["UCI ML Repository"]

    A --> B["UC Irvine 운영"]
    A --> C["600+ 데이터셋"]
    A --> D["학술 연구용"]
    A --> E["벤치마크 표준"]

    style A fill:#fce7f3
```

## 8. 플랫폼 선택 가이드

```mermaid
flowchart TD
    A["어떤 데이터가 필요?"]

    A -->|"국내 공공"| B["공공데이터포털"]
    A -->|"AI 학습용"| C["AI 허브"]
    A -->|"글로벌/다양"| D["Kaggle"]
    A -->|"학술/벤치마크"| E["UCI"]

    style A fill:#dbeafe
```

## 9. API 키 발급 과정

```mermaid
flowchart TD
    A["회원가입"]
    B["API 메뉴 접속"]
    C["활용 신청"]
    D["API 키 발급"]
    E["코드에서 사용"]

    A --> B --> C --> D --> E

    style D fill:#dcfce7
```

## 10. 공공데이터 API 호출 흐름

```mermaid
flowchart LR
    A["Python 코드"]
    B["requests.get()"]
    C["API 서버"]
    D["JSON 응답"]
    E["데이터 활용"]

    A --> B --> C --> D --> E
```

## 11. Kaggle 데이터 다운로드 방법

```mermaid
flowchart TD
    A["Kaggle 데이터 다운로드"]

    A --> B["방법 1:<br>웹에서 직접"]
    A --> C["방법 2:<br>Kaggle API"]

    B --> D["Download 버튼"]
    C --> E["kaggle datasets download"]

    style A fill:#dbeafe
```

## 12. UCI 데이터 로드 방법

```mermaid
flowchart TD
    A["ucimlrepo 설치"]
    B["fetch_ucirepo(id=...)"]
    C["데이터셋 객체"]
    D["X: 특성"]
    E["y: 타겟"]

    A --> B --> C
    C --> D
    C --> E

    style C fill:#dcfce7
```

## 13. 제조 분야 추천 데이터셋

```mermaid
flowchart TD
    A["제조 데이터셋"]

    A --> B["AI 허브<br>품질 예측"]
    A --> C["공공데이터포털<br>설비 센서"]
    A --> D["Kaggle<br>반도체 공정"]
    A --> E["UCI<br>Steel Plates Faults"]

    style A fill:#dbeafe
```

## 14. 데이터 활용 목적별 분류

```mermaid
flowchart LR
    subgraph 분류["분류 문제"]
        A["불량 분류"]
        B["결함 탐지"]
    end

    subgraph 예측["예측 문제"]
        C["수율 예측"]
        D["품질 예측"]
    end

    subgraph 탐지["이상 탐지"]
        E["설비 이상"]
        F["센서 이상"]
    end
```

## 15. 데이터 구조 확인 5단계

```mermaid
flowchart TD
    A["데이터 확인"]

    A --> B["1. shape<br>크기"]
    A --> C["2. columns<br>컬럼"]
    A --> D["3. dtypes<br>타입"]
    A --> E["4. isnull()<br>결측치"]
    A --> F["5. describe()<br>통계"]

    style A fill:#dbeafe
```

## 16. 데이터 품질 점검

```mermaid
flowchart TD
    A["데이터 품질"]

    A --> B["완전성<br>결측치 없음"]
    A --> C["정확성<br>올바른 값"]
    A --> D["일관성<br>형식 통일"]
    A --> E["적시성<br>최신 데이터"]
```

## 17. 실습 환경 구성

```mermaid
flowchart TD
    A["실습 환경"]

    A --> B["pandas"]
    A --> C["requests"]
    A --> D["ucimlrepo<br>(선택)"]
    A --> E["kaggle<br>(선택)"]

    style A fill:#dbeafe
```

## 18. 데이터셋 파일 형식

```mermaid
flowchart TD
    A["데이터 형식"]

    A --> B["CSV<br>가장 일반적"]
    A --> C["JSON<br>API 응답"]
    A --> D["Excel<br>xlsx"]
    A --> E["Parquet<br>대용량"]

    style B fill:#dcfce7
```

## 19. 4차시 학습 흐름

```mermaid
flowchart LR
    A["데이터 생태계<br>이해"]
    B["플랫폼<br>탐색"]
    C["데이터<br>다운로드"]
    D["구조<br>확인"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 20. 다음 차시 연결

```mermaid
flowchart LR
    A["4차시<br>데이터 확보"]
    B["5차시<br>기술통계/시각화"]
    C["6차시<br>확률분포/검정"]

    A --> B --> C

    style A fill:#dbeafe
    style B fill:#dcfce7
```

