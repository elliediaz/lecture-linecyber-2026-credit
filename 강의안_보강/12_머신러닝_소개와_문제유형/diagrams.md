# [12차시] 머신러닝 소개와 문제 유형 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["Part III<br>시작"]
    B["머신러닝<br>개념"]
    C["지도/비지도<br>학습"]
    D["분류/회귀<br>구분"]

    A --> B --> C --> D

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
    style D fill:#fef3c7
```

## 2. Part I-III 여정

```mermaid
flowchart TD
    A["Part I: 기초 (1-4차시)"]
    B["Part II: 데이터 분석 (5-10차시)"]
    C["Part III: 머신러닝 (11-20차시)"]

    A --> B --> C

    A --> A1["Python 기초"]
    A --> A2["데이터 다루기"]

    B --> B1["통계/시각화"]
    B --> B2["전처리/EDA"]

    C --> C1["분류/회귀"]
    C --> C2["딥러닝"]

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#1e40af,color:#fff
```

## 3. 머신러닝이란

```mermaid
flowchart TD
    A["Machine Learning<br>머신러닝"]

    A --> B["정의"]
    B --> B1["데이터에서 스스로<br>패턴을 학습하는 알고리즘"]

    A --> C["핵심 아이디어"]
    C --> C1["사람이 규칙을 정하지 않음"]
    C --> C2["데이터가 규칙을 알려줌"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
```

## 4. 전통 프로그래밍 vs 머신러닝

```mermaid
flowchart LR
    subgraph 전통["전통적 프로그래밍"]
        A1["규칙 + 데이터"] --> A2["결과"]
    end

    subgraph ML["머신러닝"]
        B1["데이터 + 결과"] --> B2["규칙"]
    end

    style 전통 fill:#fecaca
    style ML fill:#dcfce7
```

## 5. 전통 방식 흐름

```mermaid
flowchart TD
    A["개발자"]
    B["규칙 작성"]
    C["프로그램"]
    D["입력 데이터"]
    E["결과"]

    A --> B --> C
    D --> C --> E

    B --> B1["if 온도 > 90: 불량"]
    B --> B2["if 습도 > 70: 불량"]

    style A fill:#fef3c7
    style C fill:#dbeafe
```

## 6. 머신러닝 방식 흐름

```mermaid
flowchart TD
    A["데이터"]
    B["학습"]
    C["모델"]
    D["새 데이터"]
    E["예측"]

    A --> B --> C
    D --> C --> E

    A --> A1["특성 + 정답"]
    C --> C1["학습된 패턴"]
    E --> E1["불량 예측"]

    style B fill:#dcfce7
    style C fill:#1e40af,color:#fff
```

## 7. 머신러닝 학습 과정

```mermaid
flowchart LR
    A["입력<br>(Features)"]
    B["모델"]
    C["출력<br>(Target)"]

    A --> B --> C

    A --> A1["온도, 습도, 속도"]
    B --> B1["패턴 학습"]
    C --> C1["불량 여부"]

    style A fill:#dbeafe
    style B fill:#fef3c7
    style C fill:#dcfce7
```

## 8. 핵심 용어

```mermaid
flowchart TD
    A["ML 핵심 용어"]

    A --> B["특성 (Feature)"]
    A --> C["타겟 (Target)"]
    A --> D["모델 (Model)"]
    A --> E["학습 (Training)"]
    A --> F["예측 (Prediction)"]

    B --> B1["입력 데이터"]
    C --> C1["예측하려는 값"]
    D --> D1["학습된 패턴"]
    E --> E1["패턴 찾기"]
    F --> F1["패턴 적용"]

    style A fill:#1e40af,color:#fff
```

## 9. ML 적합한 경우

```mermaid
flowchart TD
    A["ML 사용 판단"]

    A --> B["적합"]
    A --> C["부적합"]

    B --> B1["규칙이 복잡"]
    B --> B2["데이터 충분"]
    B --> B3["패턴 변화"]
    B --> B4["미묘한 판단"]

    C --> C1["단순 규칙"]
    C --> C2["데이터 부족"]
    C --> C3["100% 정확도 필수"]

    style B fill:#dcfce7
    style C fill:#fecaca
```

## 10. 제조 현장 ML 활용

```mermaid
flowchart TD
    A["제조 현장 ML"]

    A --> B["품질 검사"]
    A --> C["설비 유지보수"]
    A --> D["불량 예측"]
    A --> E["생산 계획"]

    B --> B1["이미지 분류"]
    C --> C1["고장 예측"]
    D --> D1["센서 데이터 분석"]
    E --> E1["수요 예측"]

    style A fill:#1e40af,color:#fff
```

## 11. 머신러닝 종류

```mermaid
flowchart TD
    A["머신러닝"]

    A --> B["지도학습<br>Supervised"]
    A --> C["비지도학습<br>Unsupervised"]
    A --> D["강화학습<br>Reinforcement"]

    B --> B1["정답 있음"]
    C --> C1["정답 없음"]
    D --> D1["보상 기반"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
```

## 12. 지도학습

```mermaid
flowchart TD
    A["지도학습<br>Supervised Learning"]

    A --> B["특징"]
    B --> B1["레이블(정답) 있음"]
    B --> B2["X → y 관계 학습"]
    B --> B3["새 입력 예측"]

    A --> C["과정"]
    C --> C1["데이터 + 정답 제공"]
    C --> C2["패턴 학습"]
    C --> C3["새 데이터 예측"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#dcfce7
```

## 13. 지도학습 과정

```mermaid
sequenceDiagram
    participant D as 데이터
    participant M as 모델
    participant P as 예측

    D->>M: X (특성) + y (정답)
    Note over M: 패턴 학습 (fit)
    M->>P: 학습 완료
    D->>P: 새 데이터 (X_new)
    P-->>D: 예측 결과 (y_pred)
```

## 14. 지도학습 제조 예시

```mermaid
flowchart TD
    A["지도학습 제조 예시"]

    A --> B["불량 분류"]
    A --> C["생산량 예측"]
    A --> D["품질 등급"]
    A --> E["고장 예측"]

    B --> B1["입력: 센서<br>출력: 불량/정상"]
    C --> C1["입력: 설비/인력<br>출력: 생산량"]
    D --> D1["입력: 측정값<br>출력: A/B/C"]
    E --> E1["입력: 진동<br>출력: 고장 여부"]

    style A fill:#1e40af,color:#fff
```

## 15. 비지도학습

```mermaid
flowchart TD
    A["비지도학습<br>Unsupervised Learning"]

    A --> B["특징"]
    B --> B1["레이블 없음"]
    B --> B2["데이터 구조 발견"]
    B --> B3["패턴/그룹 찾기"]

    A --> C["대표 기법"]
    C --> C1["군집화"]
    C --> C2["차원 축소"]
    C --> C3["이상 탐지"]

    style A fill:#1e40af,color:#fff
    style B1 fill:#fef3c7
```

## 16. 비지도학습 과정

```mermaid
flowchart LR
    A["데이터<br>(정답 없음)"]
    B["모델"]
    C["구조 발견"]

    A --> B --> C

    C --> C1["그룹 1"]
    C --> C2["그룹 2"]
    C --> C3["그룹 3"]

    style A fill:#dbeafe
    style C fill:#dcfce7
```

## 17. 지도학습 vs 비지도학습

```mermaid
flowchart LR
    subgraph 지도["지도학습"]
        A1["정답 있음"]
        A2["예측"]
        A3["분류, 회귀"]
    end

    subgraph 비지도["비지도학습"]
        B1["정답 없음"]
        B2["구조 발견"]
        B3["군집화, 차원축소"]
    end

    style 지도 fill:#dcfce7
    style 비지도 fill:#fef3c7
```

## 18. 지도학습의 두 문제

```mermaid
flowchart TD
    A["지도학습"]

    A --> B["분류<br>Classification"]
    A --> C["회귀<br>Regression"]

    B --> B1["범주 예측"]
    B --> B2["정상/불량"]
    B --> B3["A/B/C 등급"]

    C --> C1["숫자 예측"]
    C --> C2["생산량: 1,247개"]
    C --> C3["불량률: 3.5%"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 19. 분류

```mermaid
flowchart TD
    A["분류<br>Classification"]

    A --> B["특징"]
    B --> B1["정해진 범주 중 하나"]
    B --> B2["이산적 값"]
    B --> B3["확률로 해석 가능"]

    A --> C["예시"]
    C --> C1["불량/정상"]
    C --> C2["A/B/C 등급"]
    C --> C3["고장 유형"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
```

## 20. 이진 분류 vs 다중 분류

```mermaid
flowchart LR
    subgraph 이진["이진 분류"]
        A1["2개 클래스"]
        A2["불량 / 정상"]
        A3["합격 / 불합격"]
    end

    subgraph 다중["다중 분류"]
        B1["3개+ 클래스"]
        B2["A / B / C"]
        B3["모터/베어링/전기"]
    end

    style 이진 fill:#dbeafe
    style 다중 fill:#dcfce7
```

## 21. 회귀

```mermaid
flowchart TD
    A["회귀<br>Regression"]

    A --> B["특징"]
    B --> B1["연속적인 숫자"]
    B --> B2["어떤 값이든 가능"]
    B --> B3["오차 최소화"]

    A --> C["예시"]
    C --> C1["생산량: 1,247개"]
    C --> C2["불량률: 3.5%"]
    C --> C3["수명: 87일"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
```

## 22. 분류 vs 회귀 구분

```mermaid
flowchart TD
    A["출력이 뭔가요?"]

    A --> B["범주"]
    A --> C["숫자"]

    B --> B1["분류"]
    B1 --> B2["~인가요?<br>어떤 종류?"]

    C --> C1["회귀"]
    C1 --> C2["얼마나?<br>몇 개?"]

    style A fill:#fef3c7
    style B1 fill:#dbeafe
    style C1 fill:#dcfce7
```

## 23. 문제 유형 판단

```mermaid
flowchart LR
    A["질문"]

    A --> B{"범주형?"}

    B -->|Yes| C["분류"]
    B -->|No| D["회귀"]

    C --> C1["불량인가요?"]
    C --> C2["어떤 등급?"]

    D --> D1["생산량은?"]
    D --> D2["몇 %?"]

    style C fill:#dbeafe
    style D fill:#dcfce7
```

## 24. sklearn 개요

```mermaid
flowchart TD
    A["scikit-learn"]

    A --> B["장점"]
    B --> B1["일관된 API"]
    B --> B2["다양한 알고리즘"]
    B --> B3["전처리/평가 도구"]
    B --> B4["풍부한 문서"]

    A --> C["핵심 패턴"]
    C --> C1["fit: 학습"]
    C --> C2["predict: 예측"]
    C --> C3["score: 평가"]

    style A fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 25. sklearn 기본 흐름

```mermaid
flowchart TD
    A["1. 데이터 준비"]
    B["2. 모델 생성"]
    C["3. 학습 (fit)"]
    D["4. 예측 (predict)"]
    E["5. 평가 (score)"]

    A --> B --> C --> D --> E

    A --> A1["X, y 분리"]
    B --> B1["ModelClass()"]
    C --> C1["model.fit(X, y)"]
    D --> D1["model.predict(X_new)"]
    E --> E1["model.score(X, y)"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 26. 학습/테스트 분리

```mermaid
flowchart TD
    A["전체 데이터"]

    A --> B["학습 데이터<br>80%"]
    A --> C["테스트 데이터<br>20%"]

    B --> B1["패턴 학습"]
    C --> C1["성능 평가"]

    style A fill:#1e40af,color:#fff
    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 27. train_test_split

```mermaid
flowchart LR
    A["X, y"]
    B["train_test_split"]
    C["X_train<br>y_train"]
    D["X_test<br>y_test"]

    A --> B
    B --> C
    B --> D

    C --> C1["학습용 80%"]
    D --> D1["평가용 20%"]

    style B fill:#fef3c7
```

## 28. 분류 모델 흐름

```mermaid
flowchart TD
    A["DecisionTreeClassifier"]

    A --> B["생성"]
    B --> B1["model = DecisionTreeClassifier()"]

    A --> C["학습"]
    C --> C1["model.fit(X_train, y_train)"]

    A --> D["예측"]
    D --> D1["y_pred = model.predict(X_test)"]

    A --> E["평가"]
    E --> E1["accuracy = model.score()"]

    style A fill:#1e40af,color:#fff
    style E fill:#dcfce7
```

## 29. 회귀 모델 흐름

```mermaid
flowchart TD
    A["LinearRegression"]

    A --> B["생성"]
    B --> B1["model = LinearRegression()"]

    A --> C["학습"]
    C --> C1["model.fit(X_train, y_train)"]

    A --> D["예측"]
    D --> D1["y_pred = model.predict(X_test)"]

    A --> E["평가"]
    E --> E1["r2 = model.score()"]

    style A fill:#1e40af,color:#fff
    style E fill:#dcfce7
```

## 30. 모델 일관된 API

```mermaid
flowchart TD
    A["sklearn 모든 모델"]

    A --> B["fit(X, y)"]
    A --> C["predict(X)"]
    A --> D["score(X, y)"]

    B --> B1["학습"]
    C --> C1["예측"]
    D --> D1["평가"]

    E["DecisionTree"] --> A
    F["RandomForest"] --> A
    G["LinearRegression"] --> A
    H["SVM"] --> A

    style A fill:#1e40af,color:#fff
```

## 31. 실습 데이터 구조

```mermaid
flowchart TD
    A["실습 데이터"]

    A --> B["특성 (X)"]
    B --> B1["temperature"]
    B --> B2["humidity"]
    B --> B3["speed"]

    A --> C["타겟 (y)"]
    C --> C1["defect<br>(분류)"]
    C --> C2["production<br>(회귀)"]

    style A fill:#1e40af,color:#fff
    style C1 fill:#dbeafe
    style C2 fill:#dcfce7
```

## 32. 분류/회귀 동시 실습

```mermaid
flowchart LR
    A["같은 특성<br>X"]

    A --> B["분류 모델"]
    A --> C["회귀 모델"]

    B --> B1["defect<br>불량 여부"]
    C --> C1["production<br>생산량"]

    style A fill:#fef3c7
    style B1 fill:#dbeafe
    style C1 fill:#dcfce7
```

## 33. 새 데이터 예측

```mermaid
flowchart TD
    A["새 제품 데이터"]
    B["학습된 모델"]
    C["예측 결과"]

    A --> B --> C

    A --> A1["온도: 87<br>습도: 55<br>속도: 105"]

    C --> C1["분류: 정상/불량"]
    C --> C2["회귀: 1,234개"]

    style B fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 34. 실습 전체 흐름

```mermaid
flowchart TD
    A["데이터 생성"]
    B["X, y 분리"]
    C["train/test 분리"]
    D["모델 학습"]
    E["예측"]
    F["평가"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 35. Part III 커리큘럼

```mermaid
flowchart TD
    A["Part III: 머신러닝"]

    A --> B["12차시: ML 소개"]
    A --> C["12-13차시: 분류 모델"]
    A --> D["14차시: 회귀 모델"]
    A --> E["15-16차시: 평가/튜닝"]
    A --> F["17-18차시: 시계열"]
    A --> G["19-20차시: 딥러닝"]

    style A fill:#1e40af,color:#fff
    style B fill:#dcfce7
```

## 36. 핵심 정리

```mermaid
flowchart TD
    A["12차시 핵심"]

    A --> B["머신러닝"]
    A --> C["학습 유형"]
    A --> D["문제 유형"]

    B --> B1["데이터에서<br>패턴 자동 학습"]

    C --> C1["지도학습: 정답 있음"]
    C --> C2["비지도학습: 정답 없음"]

    D --> D1["분류: 범주 예측"]
    D --> D2["회귀: 숫자 예측"]

    style A fill:#1e40af,color:#fff
```

## 37. sklearn 패턴 정리

```mermaid
flowchart LR
    A["모델 생성<br>ModelClass()"]
    B["학습<br>fit(X, y)"]
    C["예측<br>predict(X)"]
    D["평가<br>score(X, y)"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#fecaca
```

## 38. 다음 차시 연결

```mermaid
flowchart LR
    A["12차시<br>ML 소개"]
    B["12차시<br>의사결정나무"]

    A --> B

    A --> A1["개념 이해"]
    A --> A2["sklearn 기초"]

    B --> B1["첫 분류 모델"]
    B --> B2["트리 구조 해석"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
