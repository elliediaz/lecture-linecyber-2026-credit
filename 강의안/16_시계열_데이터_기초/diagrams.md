# [16차시] 시계열 데이터 기초 - 다이어그램

## 1. 시계열 데이터란?

```mermaid
flowchart LR
    subgraph 시계열["시계열 데이터"]
        A1["1월"] --> A2["2월"] --> A3["3월"] --> A4["..."]
    end

    subgraph 특징["특징"]
        B1["시간 순서 중요"]
        B2["순서 바꾸면 의미 없음"]
    end

    시계열 --> 특징
```

## 2. 제조 현장 시계열 예시

```mermaid
mindmap
  root((제조 시계열<br>데이터))
    생산
      일별 생산량
      시간별 출고량
    설비
      온도 로그
      압력 변화
      진동 데이터
    품질
      시간별 불량률
      월별 수율
    에너지
      전력 사용량
      시간대별 소비
```

## 3. 일반 데이터 vs 시계열 데이터

```mermaid
flowchart TD
    subgraph 일반["일반 데이터"]
        A1["순서 바꿔도 OK"]
        A2["각 행이 독립적"]
    end

    subgraph 시계열["시계열 데이터"]
        B1["순서 바꾸면 ❌"]
        B2["시간 의존성 있음"]
        B3["자기상관"]
    end
```

## 4. 시계열 특성

```mermaid
flowchart LR
    A["시계열 특성"]

    A --> B["순서 중요"]
    A --> C["시간 의존성"]
    A --> D["계절성"]

    B --> B1["1월→2월→3월"]
    C --> C1["오늘↔어제 연관"]
    D --> D1["12월 매출↑<br>월요일 생산↓"]
```

## 5. 자기상관 개념

```mermaid
flowchart LR
    subgraph 자기상관["자기상관 (Autocorrelation)"]
        A["어제 생산량: 1000"] --> B["오늘 생산량: 1050"]
        B --> C["내일 생산량: 1030"]
    end

    D["과거 값이 현재에 영향"]

    자기상관 --> D
```

## 6. datetime 모듈

```mermaid
flowchart TD
    A["datetime 모듈"]

    A --> B["datetime.now()"]
    A --> C["datetime(2024, 1, 15)"]
    A --> D["strptime()"]
    A --> E["strftime()"]

    B --> B1["현재 시간"]
    C --> C1["특정 날짜 생성"]
    D --> D1["문자열 → datetime"]
    E --> E1["datetime → 문자열"]
```

## 7. strptime vs strftime

```mermaid
flowchart LR
    subgraph strptime["strptime (parse)"]
        A1["'2024-01-15'"] --> A2["datetime 객체"]
    end

    subgraph strftime["strftime (format)"]
        B1["datetime 객체"] --> B2["'2024년 01월 15일'"]
    end
```

## 8. Pandas 날짜 처리

```mermaid
flowchart TD
    A["pd.to_datetime()"]

    A --> B["다양한 형식 지원"]
    B --> B1["'2024-01-15'"]
    B --> B2["'01/15/2024'"]
    B --> B3["'15-Jan-2024'"]

    A --> C["자동 파싱"]
    A --> D["형식 지정 가능"]
```

## 9. dt 접근자

```mermaid
flowchart TD
    A["df['날짜'].dt"]

    A --> B[".year"]
    A --> C[".month"]
    A --> D[".day"]
    A --> E[".dayofweek"]
    A --> F[".day_name()"]

    B --> B1["연도 추출"]
    C --> C1["월 추출"]
    D --> D1["일 추출"]
    E --> E1["요일 (0=월)"]
    F --> F1["요일명"]
```

## 10. 날짜 인덱스

```mermaid
flowchart TD
    A["df.set_index('날짜')"]

    A --> B["기간 필터링"]
    B --> B1["df['2024-01']"]
    B --> B2["df['2024-01':'2024-06']"]

    A --> C["날짜 연산"]
    A --> D["resample 사용"]
```

## 11. resample 개념

```mermaid
flowchart LR
    subgraph 일별["일별 데이터"]
        A1["1/1: 100"]
        A2["1/2: 105"]
        A3["1/3: 98"]
        A4["..."]
        A5["1/7: 102"]
    end

    B["resample('W').mean()"]

    subgraph 주별["주별 평균"]
        C1["1주차: 101.5"]
    end

    일별 --> B --> 주별
```

## 12. resample 주기

```mermaid
flowchart TD
    A["resample 주기"]

    A --> B["'D' - 일별"]
    A --> C["'W' - 주별"]
    A --> D["'M' - 월별"]
    A --> E["'Q' - 분기별"]
    A --> F["'Y' - 연별"]
    A --> G["'H' - 시간별"]
```

## 13. rolling 이동평균

```mermaid
flowchart LR
    subgraph 원본["원본 데이터"]
        A1["100"]
        A2["105"]
        A3["98"]
        A4["102"]
        A5["108"]
    end

    B["rolling(3).mean()"]

    subgraph 이동평균["3일 이동평균"]
        C1["NaN"]
        C2["NaN"]
        C3["101"]
        C4["101.7"]
        C5["102.7"]
    end

    원본 --> B --> 이동평균
```

## 14. shift 연산

```mermaid
flowchart LR
    subgraph 원본["원본"]
        A1["100"]
        A2["105"]
        A3["98"]
    end

    B["shift(1)"]

    subgraph 시프트["shift(1)"]
        C1["NaN"]
        C2["100"]
        C3["105"]
    end

    D["전일 대비 분석"]

    원본 --> B --> 시프트 --> D
```

## 15. 시계열 분할

```mermaid
flowchart LR
    subgraph 잘못된["❌ 잘못된 방법"]
        A1["무작위 분할"]
        A2["미래→과거 예측"]
    end

    subgraph 올바른["✅ 올바른 방법"]
        B1["시간 기준 분할"]
        B2["과거→미래 예측"]
    end

    style 올바른 fill:#c8e6c9
```

## 16. 시간 기준 분할

```mermaid
flowchart LR
    subgraph 전체["전체 데이터"]
        direction LR
        A["1월"] --> B["2월"] --> C["3월"] --> D["4월"] --> E["5월"] --> F["6월"]
    end

    subgraph Train["Train (과거)"]
        G["1-4월"]
    end

    subgraph Test["Test (미래)"]
        H["5-6월"]
    end

    style Train fill:#bbdefb
    style Test fill:#ffcdd2
```

## 17. 시계열 처리 워크플로우

```mermaid
flowchart TD
    A["1. 데이터 로드"]
    B["2. pd.to_datetime 변환"]
    C["3. 날짜 인덱스 설정"]
    D["4. 날짜 정보 추출"]
    E["5. 리샘플링/이동평균"]
    F["6. 시각화"]

    A --> B --> C --> D --> E --> F
```

## 18. 강의 구조

```mermaid
gantt
    title 16차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    시계열이란?              :a2, after a1, 2m
    시계열 특징              :a3, after a2, 1.5m
    Python datetime          :a4, after a3, 2m
    Pandas 날짜 처리          :a5, after a4, 2.5m

    section 실습편
    실습 소개               :b1, after a5, 2m
    데이터 생성              :b2, after b1, 2m
    날짜 변환               :b3, after b2, 2m
    날짜 정보 추출           :b4, after b3, 2m
    리샘플링                :b5, after b4, 2m
    이동평균                :b6, after b5, 3m
    Shift 연산              :b7, after b6, 2m
    분할 주의사항            :b8, after b7, 2m

    section 정리
    핵심 요약               :c1, after b8, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((시계열<br>데이터 기초))
    시계열
      시간 순서 데이터
      순서 중요
      자기상관
    날짜 처리
      pd.to_datetime
      dt 접근자
      날짜 인덱스
    분석 도구
      resample
      rolling
      shift
    주의사항
      시간 기준 분할
      무작위 분할 금지
```

## 20. 제조 현장 활용

```mermaid
flowchart TD
    A["시계열 분석 활용"]

    A --> B["생산량 예측"]
    A --> C["설비 이상 감지"]
    A --> D["수요 예측"]
    A --> E["품질 추세 분석"]

    B --> B1["일별/주별 예측"]
    C --> C1["센서 데이터 모니터링"]
    D --> D1["계절성 파악"]
    E --> E1["불량률 추이"]
```
