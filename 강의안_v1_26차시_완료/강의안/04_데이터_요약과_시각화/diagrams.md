# [4차시] 데이터 요약과 시각화 - 다이어그램

## 1. 시각화의 중요성 (앤스콤의 4분면)

```mermaid
flowchart TD
    subgraph 숫자만[" 숫자만 보면"]
        A["데이터셋 A, B, C, D"]
        A --> B["평균: 동일"]
        A --> C["분산: 동일"]
        A --> D["상관계수: 동일"]
        B --> E["같은 데이터?"]
        C --> E
        D --> E
    end

    subgraph 그래프로["그래프로 보면"]
        F["완전히 다른 패턴!"]
        G["선형 관계"]
        H["곡선 관계"]
        I["이상치 존재"]
        J["수직 분포"]
    end

    E --> |시각화| F
    F --> G
    F --> H
    F --> I
    F --> J
```

## 2. 기술통계량 분류

```mermaid
mindmap
  root((기술통계량))
    중심 경향
      평균 Mean
        모든 값의 합 / 개수
      중앙값 Median
        정렬 후 가운데 값
      최빈값 Mode
        가장 많이 나타나는 값
    산포도
      표준편차 Std
        평균으로부터의 퍼짐
      분산 Variance
        표준편차의 제곱
      범위 Range
        최대 - 최소
    분포 형태
      왜도 Skewness
        좌우 비대칭
      첨도 Kurtosis
        뾰족한 정도
```

## 3. 평균 vs 중앙값 비교

```mermaid
flowchart LR
    subgraph 데이터["데이터: [100, 100, 100, ..., 1000]"]
        A["9개의 100 + 1개의 1000"]
    end

    데이터 --> B["평균: 190"]
    데이터 --> C["중앙값: 100"]

    B --> D["이상치에 영향 받음"]
    C --> E["이상치에 강건함"]

    subgraph 가이드라인["선택 가이드"]
        F["정규분포 → 평균"]
        G["이상치 존재 → 중앙값"]
        H["제조 데이터 → 둘 다 확인"]
    end
```

## 4. 표준편차와 품질 일관성

```mermaid
flowchart TD
    subgraph 라인A["라인 A"]
        A1["1190, 1200, 1195, 1205, 1210"]
        A2["평균: 1200"]
        A3["표준편차: 7.1"]
        A4["품질 일관됨"]
    end

    subgraph 라인B["라인 B"]
        B1["1000, 1100, 1200, 1300, 1400"]
        B2["평균: 1200"]
        B3["표준편차: 141.4"]
        B4["품질 불안정"]
    end

    A3 --> |작음| C["생산 안정"]
    B3 --> |큼| D["관리 필요"]
```

## 5. 그래프 종류 선택 가이드

```mermaid
flowchart TD
    A[분석 목적] --> B{어떤 정보?}

    B --> |분포 형태| C[히스토그램]
    B --> |이상치와 범위| D[상자그림]
    B --> |두 변수 관계| E[산점도]
    B --> |시간에 따른 변화| F[선 그래프]
    B --> |범주별 비교| G[막대 그래프]

    C --> C1["생산량 분포 확인"]
    D --> D1["라인별 품질 비교"]
    E --> E1["온도 vs 불량률"]
    F --> F1["일별 생산량 추이"]
    G --> G1["라인별 평균 비교"]
```

## 6. Matplotlib 기본 구조

```mermaid
flowchart TD
    A["import matplotlib.pyplot as plt"] --> B["plt.figure()"]
    B --> C["그래프 그리기"]

    C --> D["plt.hist() 히스토그램"]
    C --> E["plt.boxplot() 상자그림"]
    C --> F["plt.scatter() 산점도"]
    C --> G["plt.plot() 선 그래프"]
    C --> H["plt.bar() 막대 그래프"]

    D --> I["레이블 추가"]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J["plt.xlabel(), plt.ylabel()"]
    J --> K["plt.title()"]
    K --> L["plt.legend()"]
    L --> M["plt.show()"]
```

## 7. 히스토그램 해석

```mermaid
flowchart LR
    subgraph 정규분포["정규분포 (종 모양)"]
        A["대칭적 분포"]
        A --> A1["평균 = 중앙값"]
        A --> A2["이상적 품질"]
    end

    subgraph 왼쪽치우침["왼쪽 치우침"]
        B["꼬리가 왼쪽"]
        B --> B1["평균 < 중앙값"]
        B --> B2["낮은 값 이상치"]
    end

    subgraph 오른쪽치우침["오른쪽 치우침"]
        C["꼬리가 오른쪽"]
        C --> C1["평균 > 중앙값"]
        C --> C2["높은 값 이상치"]
    end
```

## 8. 상자그림 구성 요소

```mermaid
flowchart TB
    subgraph 상자그림["상자그림 (Boxplot)"]
        A["o 이상치 (outlier)"]
        A --> B["┬ 최대값 (Q3 + 1.5×IQR 이내)"]
        B --> C["┌──┴──┐"]
        C --> D["│  Q3  │ 75% 지점"]
        D --> E["│──┼──│ 중앙값 50%"]
        E --> F["│  Q1  │ 25% 지점"]
        F --> G["└──┬──┘"]
        G --> H["┴ 최소값 (Q1 - 1.5×IQR 이내)"]
    end

    I["IQR = Q3 - Q1<br>사분위 범위"]
```

## 9. 산점도와 상관관계

```mermaid
flowchart LR
    subgraph 양의상관["양의 상관관계"]
        A["점들이 오른쪽 위로"]
        A --> A1["X 증가 → Y 증가"]
        A --> A2["예: 온도↑ → 불량률↑"]
    end

    subgraph 음의상관["음의 상관관계"]
        B["점들이 오른쪽 아래로"]
        B --> B1["X 증가 → Y 감소"]
        B --> B2["예: 숙련도↑ → 불량률↓"]
    end

    subgraph 무상관["상관관계 없음"]
        C["점들이 무작위 분포"]
        C --> C1["X와 Y 관련 없음"]
        C --> C2["예: 날씨 ↔ 생산량"]
    end
```

## 10. 시각화 Best Practices

```mermaid
mindmap
  root((좋은 그래프))
    필수 요소
      제목
        한눈에 내용 파악
      축 레이블
        단위 포함
      범례
        색상 의미 설명
    디자인
      색상
        3-4개 이내
      폰트
        읽기 쉬운 크기
      레이아웃
        여백 적절히
    저장
      해상도
        보고서용 300dpi
      형식
        PNG 권장
```

## 11. 데이터 시각화 워크플로우

```mermaid
flowchart LR
    A[원시 데이터] --> B[기술통계 확인]
    B --> C{분포 확인}

    C --> D[히스토그램]
    D --> E{이상치 있음?}

    E --> |예| F[상자그림으로 확인]
    E --> |아니오| G[변수 관계 분석]

    F --> G
    G --> H[산점도]
    H --> I{시간 데이터?}

    I --> |예| J[선 그래프]
    I --> |아니오| K[최종 대시보드]

    J --> K
```

## 12. 강의 구조

```mermaid
gantt
    title 4차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)     :a1, 00:00, 2m
    시각화의 중요성      :a2, after a1, 1m
    기술통계량          :a3, after a2, 2m
    표준편차의 의미      :a4, after a3, 2m
    Matplotlib 소개     :a5, after a4, 2m
    그래프 종류 선택     :a6, after a5, 1m

    section 실습편
    실습 소개           :b1, after a6, 2m
    데이터 준비         :b2, after b1, 2m
    히스토그램          :b3, after b2, 3m
    상자그림            :b4, after b3, 3m
    산점도              :b5, after b4, 3m
    여러 그래프         :b6, after b5, 3m
    그래프 저장         :b7, after b6, 2m

    section 정리
    Best Practices      :c1, after b7, 1m
    요약 및 예고        :c2, after c1, 2m
```
