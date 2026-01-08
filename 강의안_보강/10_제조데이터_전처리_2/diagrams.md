# [10차시] 제조 데이터 전처리 (2) - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["스케일링<br>표준화/정규화"]
    B["인코딩<br>레이블/원핫"]
    C["Pipeline<br>자동화"]
    D["10차시:<br>EDA 종합"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#1e40af,color:#fff
```

## 2. 스케일링 필요성

```mermaid
flowchart TD
    A["변수 간 스케일 차이"]

    A --> B["온도: 80~100"]
    A --> C["생산량: 1000~1500"]
    A --> D["불량률: 0.01~0.05"]

    B --> E["문제점"]
    C --> E
    D --> E

    E --> F["스케일 큰 변수가<br>모델 지배"]
    E --> G["거리 계산 왜곡"]
    E --> H["경사하강법<br>수렴 지연"]

    style A fill:#1e40af,color:#fff
    style E fill:#fecaca
```

## 3. 스케일링 필요 여부

```mermaid
flowchart TD
    A["알고리즘별 스케일링"]

    A --> B["필수"]
    A --> C["불필요"]

    B --> B1["KNN"]
    B --> B2["SVM"]
    B --> B3["신경망"]
    B --> B4["선형회귀"]

    C --> C1["의사결정나무"]
    C --> C2["랜덤포레스트"]
    C --> C3["XGBoost"]

    style B fill:#fecaca
    style C fill:#dcfce7
```

## 4. 표준화 공식

```mermaid
flowchart TD
    A["표준화<br>StandardScaler"]

    A --> B["공식"]
    B --> B1["Z = (X - μ) / σ"]

    A --> C["결과"]
    C --> C1["평균 = 0"]
    C --> C2["표준편차 = 1"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dbeafe
```

## 5. 정규화 공식

```mermaid
flowchart TD
    A["정규화<br>MinMaxScaler"]

    A --> B["공식"]
    B --> B1["X' = (X - min) / (max - min)"]

    A --> C["결과"]
    C --> C1["최솟값 → 0"]
    C --> C2["최댓값 → 1"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dbeafe
```

## 6. RobustScaler 공식

```mermaid
flowchart TD
    A["RobustScaler"]

    A --> B["공식"]
    B --> B1["X' = (X - Q2) / IQR"]

    A --> C["특징"]
    C --> C1["이상치에 강건"]
    C --> C2["중앙값 기준"]

    style A fill:#1e40af,color:#fff
    style C1 fill:#dcfce7
```

## 7. 스케일링 비교

```mermaid
flowchart TD
    A["스케일링 방법 선택"]

    A --> B["StandardScaler"]
    A --> C["MinMaxScaler"]
    A --> D["RobustScaler"]

    B --> B1["일반적인 상황<br>평균0, std1"]
    C --> C1["신경망, 이미지<br>0~1 범위"]
    D --> D1["이상치 多<br>중앙값 기준"]

    style B fill:#dbeafe
    style C fill:#dcfce7
    style D fill:#fef3c7
```

## 8. 스케일링 전후 비교

```mermaid
flowchart LR
    subgraph 원본["원본"]
        A1["온도: 80~100"]
        A2["생산량: 1000~1500"]
    end

    subgraph 표준화["StandardScaler"]
        B1["온도: -2~2"]
        B2["생산량: -2~2"]
    end

    subgraph 정규화["MinMaxScaler"]
        C1["온도: 0~1"]
        C2["생산량: 0~1"]
    end

    원본 --> 표준화
    원본 --> 정규화

    style B1 fill:#dcfce7
    style B2 fill:#dcfce7
    style C1 fill:#fef3c7
    style C2 fill:#fef3c7
```

## 9. fit과 transform

```mermaid
flowchart TD
    A["스케일러 사용법"]

    A --> B["fit(X_train)"]
    A --> C["transform(X)"]
    A --> D["fit_transform(X)"]

    B --> B1["파라미터 학습<br>평균, 표준편차 등"]
    C --> C1["데이터 변환"]
    D --> D1["fit + transform<br>학습 데이터용"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 10. 데이터 누수 방지

```mermaid
flowchart TD
    A["올바른 순서"]

    A --> B["1. scaler.fit(X_train)"]
    B --> C["2. scaler.transform(X_train)"]
    C --> D["3. scaler.transform(X_test)"]

    E["잘못된 순서"]
    E --> F["scaler.fit(전체 데이터)"]
    F --> G["데이터 누수!"]

    style A fill:#dcfce7
    style E fill:#fecaca
    style G fill:#fecaca
```

## 11. 범주형 데이터 유형

```mermaid
flowchart TD
    A["범주형 데이터"]

    A --> B["명목형<br>Nominal"]
    A --> C["순서형<br>Ordinal"]

    B --> B1["순서 없음"]
    B --> B2["라인: A, B, C"]
    B --> B3["색상: 빨강, 파랑"]

    C --> C1["순서 있음"]
    C --> C2["등급: 상, 중, 하"]
    C --> C3["만족도: 1~5"]

    style A fill:#1e40af,color:#fff
```

## 12. 레이블 인코딩

```mermaid
flowchart LR
    A["원본"]
    B["인코딩"]

    A --> B

    subgraph 원본값["원본"]
        A1["상"]
        A2["중"]
        A3["하"]
    end

    subgraph 인코딩값["인코딩"]
        B1["1"]
        B2["2"]
        B3["0"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3

    style B1 fill:#dcfce7
    style B2 fill:#dcfce7
    style B3 fill:#dcfce7
```

## 13. 레이블 인코딩 특징

```mermaid
flowchart TD
    A["LabelEncoder"]

    A --> B["장점"]
    A --> C["단점"]

    B --> B1["간단한 구현"]
    B --> B2["메모리 효율적"]
    B --> B3["트리 모델 적합"]

    C --> C1["숫자 크기에<br>의미 발생"]
    C --> C2["선형 모델<br>부적합"]

    style B fill:#dcfce7
    style C fill:#fecaca
```

## 14. 원-핫 인코딩

```mermaid
flowchart LR
    subgraph 원본["원본"]
        A["라인: A, B, C"]
    end

    subgraph 인코딩["원-핫 인코딩"]
        B1["라인_A"]
        B2["라인_B"]
        B3["라인_C"]
    end

    A --> B1
    A --> B2
    A --> B3

    style B1 fill:#dbeafe
    style B2 fill:#dcfce7
    style B3 fill:#fef3c7
```

## 15. 원-핫 인코딩 결과

```mermaid
flowchart TD
    A["라인 컬럼"]

    A --> B["A"]
    A --> C["B"]
    A --> D["C"]

    B --> B1["[1, 0, 0]"]
    C --> C1["[0, 1, 0]"]
    D --> D1["[0, 0, 1]"]

    style A fill:#1e40af,color:#fff
```

## 16. 인코딩 선택 가이드

```mermaid
flowchart TD
    A["범주형 데이터"]

    A --> B{"순서 있음?"}

    B -->|Yes| C["레이블 인코딩"]
    B -->|No| D{"고유값 수?"}

    D -->|"< 10"| E["원-핫 인코딩"]
    D -->|"> 10"| F["빈도/타겟 인코딩"]

    style C fill:#dcfce7
    style E fill:#dbeafe
    style F fill:#fef3c7
```

## 17. 다중공선성 문제

```mermaid
flowchart TD
    A["원-핫 인코딩 문제"]

    A --> B["라인_A + 라인_B + 라인_C = 1"]
    B --> C["완벽한 선형 관계"]
    C --> D["다중공선성"]

    D --> E["해결: drop_first=True"]
    E --> F["라인_A 제거"]
    F --> G["B=0, C=0 → A"]

    style D fill:#fecaca
    style E fill:#dcfce7
```

## 18. sklearn 전처리 패턴

```mermaid
flowchart LR
    A["객체 생성"]
    B["fit()"]
    C["transform()"]
    D["완료"]

    A --> B --> C --> D

    B --> B1["파라미터 학습"]
    C --> C1["데이터 변환"]

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 19. ColumnTransformer

```mermaid
flowchart TD
    A["ColumnTransformer"]

    A --> B["수치형 컬럼"]
    A --> C["범주형 컬럼"]

    B --> D["StandardScaler"]
    C --> E["OneHotEncoder"]

    D --> F["결합된 출력"]
    E --> F

    style A fill:#1e40af,color:#fff
    style F fill:#dcfce7
```

## 20. ColumnTransformer 구조

```mermaid
flowchart TD
    subgraph 입력["입력 데이터"]
        A1["온도"]
        A2["습도"]
        A3["라인"]
        A4["장비"]
    end

    subgraph CT["ColumnTransformer"]
        B1["StandardScaler"]
        B2["OneHotEncoder"]
    end

    subgraph 출력["출력"]
        C1["온도_scaled"]
        C2["습도_scaled"]
        C3["라인_A"]
        C4["라인_B"]
        C5["..."]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2
    A4 --> B2
    B1 --> C1
    B1 --> C2
    B2 --> C3
    B2 --> C4
    B2 --> C5

    style CT fill:#fef3c7
```

## 21. Pipeline 구조

```mermaid
flowchart TD
    A["Pipeline"]

    A --> B["전처리"]
    B --> C["모델"]

    B --> B1["ColumnTransformer<br>StandardScaler<br>OneHotEncoder"]
    C --> C1["LogisticRegression<br>RandomForest<br>etc."]

    style A fill:#1e40af,color:#fff
```

## 22. Pipeline 사용 흐름

```mermaid
flowchart LR
    A["X_train, y_train"]
    B["pipe.fit()"]
    C["전처리 학습"]
    D["모델 학습"]

    A --> B --> C --> D

    E["X_test"]
    F["pipe.predict()"]
    G["전처리 적용"]
    H["예측"]

    E --> F --> G --> H

    style B fill:#dbeafe
    style F fill:#dcfce7
```

## 23. Pipeline 장점

```mermaid
flowchart TD
    A["Pipeline 장점"]

    A --> B["코드 간결화"]
    A --> C["데이터 누수 방지"]
    A --> D["교차 검증 용이"]
    A --> E["배포 용이"]

    B --> B1["fit/predict<br>한 줄로"]
    C --> C1["자동 분리"]
    D --> D1["GridSearchCV<br>통합"]
    E --> E1["하나의 객체<br>저장/로드"]

    style A fill:#1e40af,color:#fff
```

## 24. Pipeline vs 수동

```mermaid
flowchart TD
    subgraph 수동["수동 방식"]
        A1["scaler.fit(X_train)"]
        A2["X_train_scaled = scaler.transform(X_train)"]
        A3["model.fit(X_train_scaled, y_train)"]
        A4["X_test_scaled = scaler.transform(X_test)"]
        A5["y_pred = model.predict(X_test_scaled)"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph Pipeline["Pipeline"]
        B1["pipe.fit(X_train, y_train)"]
        B2["y_pred = pipe.predict(X_test)"]
        B1 --> B2
    end

    style 수동 fill:#fecaca
    style Pipeline fill:#dcfce7
```

## 25. 전처리 파이프라인 저장

```mermaid
flowchart LR
    A["학습 완료된<br>Pipeline"]
    B["joblib.dump()"]
    C["파일 저장<br>.pkl"]

    A --> B --> C

    D["새 데이터"]
    E["joblib.load()"]
    F["predict()"]

    D --> E --> F

    style A fill:#dbeafe
    style C fill:#dcfce7
```

## 26. 종합 전처리 파이프라인

```mermaid
flowchart TD
    A["원본 데이터"]

    A --> B["수치형 전처리"]
    A --> C["범주형 전처리"]

    B --> D["SimpleImputer<br>(결측치)"]
    D --> E["StandardScaler<br>(스케일링)"]

    C --> F["SimpleImputer<br>(결측치)"]
    F --> G["OneHotEncoder<br>(인코딩)"]

    E --> H["결합"]
    G --> H

    H --> I["모델"]

    style A fill:#1e40af,color:#fff
    style H fill:#fef3c7
    style I fill:#dcfce7
```

## 27. 역변환

```mermaid
flowchart LR
    A["원본 데이터"]
    B["스케일링"]
    C["스케일링된 데이터"]
    D["역변환"]
    E["원본 스케일"]

    A --> B --> C --> D --> E

    B --> B1["scaler.fit_transform()"]
    D --> D1["scaler.inverse_transform()"]

    style C fill:#fef3c7
    style E fill:#dcfce7
```

## 28. 핵심 정리

```mermaid
flowchart TD
    A["10차시 핵심"]

    A --> B["스케일링"]
    A --> C["인코딩"]
    A --> D["Pipeline"]

    B --> B1["StandardScaler<br>MinMaxScaler<br>RobustScaler"]
    C --> C1["LabelEncoder<br>OneHotEncoder"]
    D --> D1["ColumnTransformer<br>+ Model"]

    style A fill:#1e40af,color:#fff
```

## 29. 주의사항 정리

```mermaid
flowchart TD
    A["주의사항"]

    A --> B["fit은 학습<br>데이터만"]
    A --> C["전처리기와<br>모델 함께 저장"]
    A --> D["인코딩 방법<br>데이터 유형별"]

    B --> B1["데이터 누수 방지"]
    C --> C1["일관된 전처리"]
    D --> D1["순서 있음→레이블<br>순서 없음→원핫"]

    style A fill:#fecaca
```

## 30. 다음 차시 연결

```mermaid
flowchart LR
    A["8차시"]
    B["10차시"]
    C["10차시"]

    A --> B --> C

    A --> A1["결측치/이상치"]
    B --> B1["스케일링/인코딩"]
    C --> C1["EDA 종합"]

    style A fill:#e2e8f0
    style B fill:#dbeafe
    style C fill:#dcfce7
```
