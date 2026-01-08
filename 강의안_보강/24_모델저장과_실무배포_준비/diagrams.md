# [24차시] 모델 저장과 실무 배포 준비 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["22차시<br>모델 해석"]
    B["24차시<br>모델 저장"]
    C["24차시<br>AI API"]

    A --> B --> C

    B --> B1["joblib 저장"]
    B --> B2["Pipeline 구성"]
    B --> B3["배포 체크리스트"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["24차시: 모델 저장과 배포 준비"]

    A --> B["대주제 1<br>joblib 저장/로드"]
    A --> C["대주제 2<br>Pipeline 구성"]
    A --> D["대주제 3<br>배포 체크리스트"]

    B --> B1["dump/load<br>압축, 버전"]
    C --> C1["전처리 + 모델<br>통합 관리"]
    D --> D1["모델, 데이터<br>환경, 테스트, 문서"]

    style A fill:#1e40af,color:#fff
```

## 3. 모델 저장의 필요성

```mermaid
flowchart LR
    A["학습<br>1시간"]
    B["모델 저장"]
    C["로드<br>1초"]
    D["예측<br>0.1초"]

    A --> |"한 번만"| B
    B --> |"필요할 때"| C
    C --> D

    style B fill:#dcfce7
```

## 4. 저장 방법 비교

```mermaid
flowchart TD
    A["모델 저장 방법"]

    A --> B["joblib"]
    B --> B1["sklearn 권장"]
    B --> B2["대용량 효율적"]

    A --> C["pickle"]
    C --> C1["Python 표준"]
    C --> C2["대용량 비효율"]

    A --> D["ONNX"]
    D --> D1["언어 중립"]
    D --> D2["변환 필요"]

    style B fill:#dcfce7
```

## 5. joblib 저장 과정

```mermaid
flowchart LR
    A["학습된 모델"]
    B["joblib.dump()"]
    C["model.pkl"]

    A --> B --> C

    C --> D["joblib.load()"]
    D --> E["복원된 모델"]
    E --> F["predict()"]

    style C fill:#fef3c7
```

## 6. 압축 옵션

```mermaid
flowchart TD
    A["모델 저장"]

    A --> B["압축 없음<br>compress=0"]
    B --> B1["빠름"]
    B --> B2["용량 큼"]

    A --> C["압축<br>compress=3"]
    C --> C1["느림"]
    C --> C2["용량 작음"]

    style C fill:#dcfce7
```

## 7. 버전 관리 중요성

```mermaid
flowchart TD
    A["모델 저장 시"]

    A --> B["모델 파일"]
    A --> C["메타데이터"]

    C --> C1["sklearn 버전"]
    C --> C2["Python 버전"]
    C --> C3["학습 날짜"]
    C --> C4["성능 지표"]

    style C fill:#fef3c7
```

## 8. 전처리 없이 모델만 저장할 때 문제

```mermaid
flowchart TD
    A["학습 시"]
    B["배포 시"]

    A --> A1["scaler.fit_transform()"]
    A1 --> A2["model.fit()"]

    B --> B1["scaler.transform()<br>❌ scaler 없음!"]
    B1 --> B2["model.predict()<br>❌ 스케일 안 맞음"]

    style B1 fill:#fecaca
    style B2 fill:#fecaca
```

## 9. Pipeline 개념

```mermaid
flowchart LR
    subgraph Pipeline
        A["원본 데이터"]
        B["전처리<br>(Scaler)"]
        C["모델<br>(RF)"]
        D["예측"]
    end

    A --> B --> C --> D

    style B fill:#dbeafe
    style C fill:#dbeafe
```

## 10. Pipeline 구성 요소

```mermaid
flowchart TD
    A["Pipeline"]

    A --> B["Step 1: scaler"]
    B --> B1["StandardScaler"]

    A --> C["Step 2: model"]
    C --> C1["RandomForestClassifier"]

    A --> D["이름으로 접근"]
    D --> D1["pipeline['scaler']"]
    D --> D2["pipeline['model']"]

    style A fill:#1e40af,color:#fff
```

## 11. Pipeline fit 과정

```mermaid
flowchart LR
    A["pipeline.fit(X, y)"]

    A --> B["scaler.fit_transform(X)"]
    B --> C["model.fit(X_scaled, y)"]

    style A fill:#dcfce7
```

## 12. Pipeline predict 과정

```mermaid
flowchart LR
    A["pipeline.predict(X_new)"]

    A --> B["scaler.transform(X_new)"]
    B --> C["model.predict(X_scaled)"]
    C --> D["예측 결과"]

    style A fill:#dcfce7
```

## 13. Pipeline 저장의 장점

```mermaid
flowchart TD
    A["Pipeline 저장"]

    A --> B["하나의 파일"]
    B --> B1["전처리 + 모델"]

    A --> C["일관성 보장"]
    C --> C1["학습/예측 동일 전처리"]

    A --> D["관리 편의"]
    D --> D1["버전 관리 단순화"]

    style A fill:#1e40af,color:#fff
```

## 14. 다단계 Pipeline

```mermaid
flowchart LR
    A["원본"]
    B["Imputer<br>결측치 처리"]
    C["Scaler<br>정규화"]
    D["PCA<br>차원 축소"]
    E["Model<br>분류"]
    F["예측"]

    A --> B --> C --> D --> E --> F

    style B fill:#dbeafe
    style C fill:#dbeafe
    style D fill:#dbeafe
    style E fill:#dbeafe
```

## 15. ColumnTransformer

```mermaid
flowchart TD
    A["ColumnTransformer"]

    A --> B["숫자형 열"]
    B --> B1["StandardScaler"]

    A --> C["범주형 열"]
    C --> C1["OneHotEncoder"]

    A --> D["결합된 출력"]

    style A fill:#1e40af,color:#fff
```

## 16. 배포 체크리스트 영역

```mermaid
flowchart TD
    A["배포 체크리스트"]

    A --> B["모델"]
    B --> B1["성능, 버전, 저장"]

    A --> C["데이터"]
    C --> C1["입력 형식, 검증"]

    A --> D["환경"]
    D --> D1["의존성, 리소스"]

    A --> E["테스트"]
    E --> E1["단위, 통합 테스트"]

    A --> F["문서"]
    F --> F1["사용법, 제약사항"]

    style A fill:#1e40af,color:#fff
```

## 17. 입력 데이터 검증 흐름

```mermaid
flowchart TD
    A["입력 데이터"]
    B{"필수 피처<br>있음?"}
    C{"값 범위<br>유효?"}
    D["예측 수행"]
    E["오류 반환"]

    A --> B
    B -->|Yes| C
    B -->|No| E
    C -->|Yes| D
    C -->|No| E

    style D fill:#dcfce7
    style E fill:#fecaca
```

## 18. 배포 워크플로우

```mermaid
flowchart TD
    A["1. 모델 개발 완료"]
    B["2. Pipeline 구성"]
    C["3. 체크리스트 검토"]
    D["4. 테스트 통과"]
    E["5. 문서화 완료"]
    F["6. 배포"]
    G["7. 모니터링"]

    A --> B --> C --> D --> E --> F --> G

    style F fill:#dcfce7
```

## 19. 모델 카드 구조

```mermaid
flowchart TD
    A["모델 카드"]

    A --> B["개요"]
    B --> B1["목적, 버전, 학습일"]

    A --> C["성능"]
    C --> C1["정확도, F1, ROC"]

    A --> D["입력/출력"]
    D --> D1["피처 목록, 형식"]

    A --> E["제약사항"]
    E --> E1["한계, 주의사항"]

    style A fill:#1e40af,color:#fff
```

## 20. 배포 방식 선택

```mermaid
flowchart TD
    A["배포 방식"]

    A --> B["REST API"]
    B --> B1["실시간 예측<br>웹 서비스"]

    A --> C["배치 처리"]
    C --> C1["대량 데이터<br>정기 예측"]

    A --> D["엣지 배포"]
    D --> D1["현장 장비<br>오프라인"]

    A --> E["클라우드"]
    E --> E1["확장성<br>관리 편의"]
```

## 21. 모니터링 흐름

```mermaid
flowchart TD
    A["예측 요청"]
    B["로깅"]
    C["성능 측정"]
    D{"드리프트<br>감지?"}
    E["정상 운영"]
    F["재학습 알림"]

    A --> B --> C --> D
    D -->|No| E
    D -->|Yes| F

    style F fill:#fef3c7
```

## 22. 모델 드리프트

```mermaid
flowchart LR
    A["배포 시점<br>정확도 92%"]
    B["1개월 후<br>정확도 90%"]
    C["3개월 후<br>정확도 85%"]
    D["재학습 필요"]

    A --> B --> C --> D

    style C fill:#fecaca
    style D fill:#fef3c7
```

## 23. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 준비"]
    B["2. Pipeline 구성"]
    C["3. 학습"]
    D["4. 저장"]
    E["5. 로드"]
    F["6. 검증"]
    G["7. 새 데이터 예측"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style G fill:#dcfce7
```

## 24. sklearn 함수 정리

```mermaid
flowchart TD
    A["sklearn"]

    A --> B["Pipeline"]
    B --> B1["fit(), predict()"]
    B --> B2["named_steps"]

    A --> C["make_pipeline"]
    C --> C1["자동 이름 생성"]

    A --> D["ColumnTransformer"]
    D --> D1["열별 전처리"]

    style A fill:#1e40af,color:#fff
```

## 25. joblib 함수 정리

```mermaid
flowchart TD
    A["joblib"]

    A --> B["dump()"]
    B --> B1["객체 저장"]
    B --> B2["compress 옵션"]

    A --> C["load()"]
    C --> C1["객체 로드"]

    style A fill:#1e40af,color:#fff
```

## 26. 핵심 정리

```mermaid
flowchart TD
    A["24차시 핵심"]

    A --> B["joblib"]
    B --> B1["dump/load<br>모델 저장"]

    A --> C["Pipeline"]
    C --> C1["전처리 + 모델<br>통합 저장"]

    A --> D["배포 체크리스트"]
    D --> D1["모델, 데이터<br>환경, 테스트, 문서"]

    style A fill:#1e40af,color:#fff
```

## 27. 다음 차시 연결

```mermaid
flowchart LR
    A["24차시<br>모델 저장"]
    B["24차시<br>AI API"]

    A --> B

    A --> A1["Pipeline"]
    A --> A2["배포 준비"]

    B --> B1["REST API"]
    B --> B2["requests"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

