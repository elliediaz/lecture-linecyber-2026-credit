# [17차시] 시계열 예측 모델 - 다이어그램

## 1. 시계열 예측 개념

```mermaid
flowchart LR
    subgraph 과거["과거 데이터"]
        A1["1월: 1000"]
        A2["2월: 1050"]
        A3["..."]
        A4["6월: 1200"]
    end

    B["예측 모델"]

    subgraph 미래["미래 예측"]
        C1["7월: ?"]
        C2["8월: ?"]
    end

    과거 --> B --> 미래
```

## 2. 제조 현장 예측 활용

```mermaid
mindmap
  root((시계열<br>예측 활용))
    생산 계획
      생산량 예측
      원자재 발주
    재고 관리
      수요 예측
      안전 재고
    설비 관리
      고장 예측
      예방 정비
    인력 배치
      작업량 예측
      교대 계획
```

## 3. ML 기반 접근법

```mermaid
flowchart TD
    A["시계열 데이터"]
    B["특성 엔지니어링"]
    C["테이블 데이터"]
    D["ML 모델<br>(RandomForest)"]
    E["예측 결과"]

    A --> B --> C --> D --> E
```

## 4. 특성 엔지니어링 종류

```mermaid
flowchart TD
    A["특성 엔지니어링"]

    A --> B["날짜 기반"]
    A --> C["시차 (Lag)"]
    A --> D["롤링 (Rolling)"]

    B --> B1["월, 요일, 주차"]
    C --> C1["어제 값, 7일 전 값"]
    D --> D1["7일 평균, 표준편차"]
```

## 5. 시계열 → 테이블 변환

```mermaid
flowchart LR
    subgraph 원본["원본 시계열"]
        direction TB
        A1["날짜 | 생산량"]
        A2["1/1 | 1000"]
        A3["1/2 | 1050"]
        A4["1/3 | 1020"]
    end

    B["특성 추가"]

    subgraph 테이블["테이블 형태"]
        direction TB
        C1["월|요일|lag_1|ma_7|생산량"]
        C2["1 | 1 | 1000 | ... | 1050"]
        C3["1 | 2 | 1050 | ... | 1020"]
    end

    원본 --> B --> 테이블
```

## 6. 시차 특성 (Lag)

```mermaid
flowchart TD
    A["shift(1): 1일 전 값"]

    subgraph 예시["예시"]
        direction LR
        B1["오늘: 1030"]
        B2["lag_1: 1020"]
        B3["lag_7: 1000"]
    end

    C["과거 값이 예측에 도움!"]

    A --> 예시 --> C
```

## 7. 롤링 특성 (Rolling)

```mermaid
flowchart LR
    subgraph 과정["rolling(7).mean()"]
        A1["1000"]
        A2["1050"]
        A3["1020"]
        A4["1030"]
        A5["1010"]
        A6["1040"]
        A7["1030"]
    end

    B["평균: 1026"]

    과정 --> B
```

## 8. 데이터 누출 방지

```mermaid
flowchart TD
    subgraph 잘못["❌ 잘못된 방법"]
        A1["rolling(7).mean()"]
        A2["오늘 값 포함!"]
    end

    subgraph 올바름["✅ 올바른 방법"]
        B1["shift(1).rolling(7).mean()"]
        B2["어제까지만!"]
    end

    style 올바름 fill:#c8e6c9
```

## 9. shift(1) 효과

```mermaid
flowchart LR
    subgraph 원본["원본"]
        A1["100"]
        A2["105"]
        A3["98"]
    end

    B["shift(1)"]

    subgraph 시프트["shift 후"]
        C1["NaN"]
        C2["100"]
        C3["105"]
    end

    D["어제 데이터만 사용"]

    원본 --> B --> 시프트 --> D
```

## 10. 시간 기준 분할

```mermaid
flowchart LR
    subgraph 전체["전체 데이터 (180일)"]
        direction LR
        A["1월"] --> B["2월"] --> C["3월"] --> D["4월"] --> E["5월"] --> F["6월"]
    end

    subgraph Train["Train"]
        G["1-4월 (과거)"]
    end

    subgraph Test["Test"]
        H["5-6월 (미래)"]
    end

    style Train fill:#bbdefb
    style Test fill:#ffcdd2
```

## 11. 잘못된 분할 vs 올바른 분할

```mermaid
flowchart TD
    subgraph 잘못["❌ 랜덤 분할"]
        A1["미래 데이터가 Train에!"]
        A2["과거 데이터가 Test에!"]
    end

    subgraph 올바름["✅ 시간 기준"]
        B1["과거로 학습"]
        B2["미래로 평가"]
    end

    style 올바름 fill:#c8e6c9
```

## 12. 예측 모델 학습 과정

```mermaid
flowchart TD
    A["1. 특성 정의"]
    B["2. Train/Test 분할"]
    C["3. 모델 학습"]
    D["4. 예측"]
    E["5. 평가"]

    A --> B --> C --> D --> E
```

## 13. 평가 지표 비교

```mermaid
flowchart LR
    subgraph MAE["MAE"]
        A1["평균 절대 오차"]
        A2["단위: 개"]
    end

    subgraph RMSE["RMSE"]
        B1["평균 제곱근 오차"]
        B2["큰 오차에 민감"]
    end

    subgraph MAPE["MAPE"]
        C1["평균 백분율 오차"]
        C2["단위: %"]
    end
```

## 14. MAPE 해석

```mermaid
flowchart TD
    A["MAPE 해석"]

    A --> B["< 10%: 매우 좋음"]
    A --> C["10-20%: 좋음"]
    A --> D["20-50%: 보통"]
    A --> E["> 50%: 개선 필요"]

    style B fill:#c8e6c9
    style C fill:#dcedc8
```

## 15. 특성 중요도

```mermaid
pie title 특성 중요도 예시
    "lag_1 (전일값)" : 45
    "ma_7 (7일평균)" : 25
    "lag_7 (7일전)" : 15
    "요일" : 10
    "월" : 5
```

## 16. 전체 워크플로우

```mermaid
flowchart TD
    A["원본 시계열"]
    B["특성 엔지니어링"]
    C["결측치 제거"]
    D["시간 기준 분할"]
    E["모델 학습"]
    F["예측"]
    G["평가 (MAE, MAPE)"]

    A --> B --> C --> D --> E --> F --> G
```

## 17. 강의 구조

```mermaid
gantt
    title 17차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    시계열 예측이란?          :a2, after a1, 1.5m
    ML 기반 접근법           :a3, after a2, 1.5m
    특성 엔지니어링           :a4, after a3, 2m
    데이터 누출 방지          :a5, after a4, 1.5m
    평가 지표                :a6, after a5, 1.5m

    section 실습편
    실습 소개               :b1, after a6, 2m
    데이터 준비              :b2, after b1, 2m
    날짜/시차 특성           :b3, after b2, 4m
    롤링 특성               :b4, after b3, 2m
    분할 및 학습             :b5, after b4, 4m
    평가 및 시각화           :b6, after b5, 4m

    section 정리
    핵심 요약               :c1, after b6, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 18. 핵심 요약

```mermaid
mindmap
  root((시계열<br>예측 모델))
    특성 엔지니어링
      날짜 특성
      Lag 특성
      Rolling 특성
    데이터 누출
      shift(1) 필수
      미래 정보 제외
    분할
      시간 기준
      과거→미래
    평가
      MAE
      RMSE
      MAPE 10-20%
```

## 19. 예측 결과 시각화

```mermaid
flowchart LR
    subgraph 그래프["시각화"]
        A["실선: 실제값"]
        B["점선: 예측값"]
    end

    C["오차 확인"]
    D["패턴 분석"]

    그래프 --> C
    그래프 --> D
```

## 20. 실무 적용 팁

```mermaid
flowchart TD
    A["실무 적용 팁"]

    A --> B["1. 충분한 데이터 확보"]
    A --> C["2. 외부 변수 추가"]
    A --> D["3. 주기적 재학습"]
    A --> E["4. 예측 오차 모니터링"]

    B --> B1["최소 1년 이상"]
    C --> C1["휴일, 이벤트, 날씨"]
    D --> D1["데이터 패턴 변화 반영"]
    E --> E1["MAPE 추적"]
```
