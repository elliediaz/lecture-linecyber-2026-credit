# [3차시] 제조 데이터 다루기 기초 - 다이어그램 (보강판)

## 1. NumPy와 Pandas 역할 비교

```mermaid
flowchart TD
    A["데이터 분석 라이브러리"]

    A --> B["NumPy"]
    A --> C["Pandas"]

    B --> B1["수치 계산"]
    B --> B2["배열 연산"]
    B --> B3["통계 함수"]

    C --> C1["표 데이터"]
    C --> C2["파일 I/O"]
    C --> C3["그룹 집계"]

    style B fill:#dbeafe,stroke:#3b82f6
    style C fill:#dcfce7,stroke:#22c55e
```

## 2. NumPy 배열 구조

```mermaid
flowchart TD
    A["NumPy 배열 (ndarray)"]

    A --> B["1차원 배열"]
    A --> C["2차원 배열"]
    A --> D["N차원 배열"]

    B --> B1["[82, 85, 88, 95]<br>센서 측정값"]
    C --> C1["[[온도, 습도],<br>[온도, 습도]]<br>센서 행렬"]

    style A fill:#1e40af,color:#fff
```

## 3. NumPy vs Python 리스트 속도

```mermaid
flowchart LR
    A["100만 개 데이터<br>평균 계산"]

    A --> B["Python 리스트<br>~500ms"]
    A --> C["NumPy 배열<br>~5ms"]

    B --> D["for 루프 반복"]
    C --> E["벡터화 연산"]

    style C fill:#dcfce7,stroke:#22c55e
    style E fill:#dcfce7,stroke:#22c55e
```

## 4. NumPy 배열 생성 방법

```mermaid
flowchart TD
    A["배열 생성"]

    A --> B["리스트에서<br>np.array([1,2,3])"]
    A --> C["0으로 채우기<br>np.zeros(5)"]
    A --> D["1로 채우기<br>np.ones(5)"]
    A --> E["범위 생성<br>np.arange(0,10,2)"]
    A --> F["균등 간격<br>np.linspace(0,1,5)"]

    style A fill:#dbeafe
```

## 5. NumPy 인덱싱

```mermaid
flowchart LR
    A["arr = [10, 20, 30, 40, 50]"]

    A --> B["arr[0] = 10<br>(첫 번째)"]
    A --> C["arr[-1] = 50<br>(마지막)"]
    A --> D["arr[1:4] = [20,30,40]<br>(슬라이싱)"]

    style A fill:#dbeafe
```

## 6. NumPy 조건 필터링

```mermaid
flowchart TD
    A["temps = [82, 85, 95, 84]"]

    A --> B["mask = temps > 90"]
    B --> C["[False, False, True, False]"]

    C --> D["temps[mask]"]
    D --> E["[95]"]

    style E fill:#dcfce7
```

## 7. NumPy 통계 함수

```mermaid
flowchart TD
    A["NumPy 통계 함수"]

    A --> B["np.mean()<br>평균"]
    A --> C["np.std()<br>표준편차"]
    A --> D["np.max()<br>최대값"]
    A --> E["np.min()<br>최소값"]
    A --> F["np.sum()<br>합계"]
    A --> G["np.median()<br>중앙값"]

    style A fill:#1e40af,color:#fff
```

## 8. Pandas 핵심 객체

```mermaid
flowchart TD
    A["Pandas"]

    A --> B["Series<br>(1차원)"]
    A --> C["DataFrame<br>(2차원)"]

    B --> B1["한 열 데이터<br>df['생산량']"]
    C --> C1["표 전체<br>행/열 구조"]

    style A fill:#1e40af,color:#fff
```

## 9. DataFrame 구조

```mermaid
flowchart TD
    A["DataFrame"]

    A --> B["인덱스<br>(행 라벨)"]
    A --> C["컬럼<br>(열 이름)"]
    A --> D["값<br>(데이터)"]

    subgraph 예시
        E["  | 제품 | 생산량 |"]
        F["0 | A001 | 1200  |"]
        G["1 | A002 | 1150  |"]
    end

    style A fill:#dcfce7
```

## 10. DataFrame 생성 방법

```mermaid
flowchart TD
    A["DataFrame 생성"]

    A --> B["딕셔너리에서<br>pd.DataFrame(dict)"]
    A --> C["CSV에서<br>pd.read_csv()"]
    A --> D["Excel에서<br>pd.read_excel()"]
    A --> E["NumPy에서<br>pd.DataFrame(array)"]

    style B fill:#dcfce7
    style C fill:#dcfce7
```

## 11. DataFrame 탐색 메서드

```mermaid
flowchart TD
    A["데이터 탐색"]

    A --> B["head()<br>처음 5행"]
    A --> C["tail()<br>마지막 5행"]
    A --> D["shape<br>(행, 열) 크기"]
    A --> E["info()<br>데이터 타입"]
    A --> F["describe()<br>기술 통계"]

    style A fill:#dbeafe
```

## 12. 열 선택 방법

```mermaid
flowchart TD
    A["열 선택"]

    A --> B["df['열이름']<br>→ Series"]
    A --> C["df[['열1','열2']]<br>→ DataFrame"]

    B --> B1["1차원 데이터"]
    C --> C1["2차원 데이터"]

    style A fill:#dbeafe
```

## 13. loc vs iloc

```mermaid
flowchart LR
    subgraph loc["loc (라벨 기반)"]
        L1["df.loc[0:2]"]
        L2["→ 0, 1, 2번 행"]
        L3["끝 포함!"]
    end

    subgraph iloc["iloc (정수 기반)"]
        I1["df.iloc[0:2]"]
        I2["→ 0, 1번 행"]
        I3["끝 미포함!"]
    end

    style loc fill:#dbeafe
    style iloc fill:#fef3c7
```

## 14. 조건 필터링 구조

```mermaid
flowchart TD
    A["df[조건]"]

    A --> B["단일 조건<br>df['생산량'] > 1000"]
    A --> C["복합 조건 AND<br>(조건1) & (조건2)"]
    A --> D["복합 조건 OR<br>(조건1) | (조건2)"]

    C --> E["괄호 필수!"]
    D --> E

    style E fill:#fef3c7
```

## 15. 새 열 추가

```mermaid
flowchart TD
    A["새 열 추가"]

    A --> B["계산식<br>df['불량률'] = df['불량'] / df['생산']"]
    A --> C["조건 함수<br>df['등급'] = df['값'].apply(func)"]
    A --> D["상수<br>df['상태'] = '정상'"]

    style A fill:#dcfce7
```

## 16. groupby 집계 흐름

```mermaid
flowchart TD
    A["df.groupby('라인')"]
    B["그룹 분할"]
    C["집계 함수 적용"]
    D["결과 반환"]

    A --> B
    B --> B1["1번 라인 데이터"]
    B --> B2["2번 라인 데이터"]

    B1 --> C
    B2 --> C

    C --> D

    style D fill:#dcfce7
```

## 17. 집계 함수 종류

```mermaid
flowchart TD
    A["집계 함수"]

    A --> B["count()<br>개수"]
    A --> C["sum()<br>합계"]
    A --> D["mean()<br>평균"]
    A --> E["std()<br>표준편차"]
    A --> F["min()/max()<br>최솟값/최댓값"]

    style A fill:#dbeafe
```

## 18. 결측치 처리 흐름

```mermaid
flowchart TD
    A["결측치 확인<br>df.isnull()"]

    A --> B{"처리 방법?"}

    B -->|제거| C["df.dropna()"]
    B -->|채우기| D["df.fillna(값)"]

    D --> D1["0으로<br>fillna(0)"]
    D --> D2["평균으로<br>fillna(df.mean())"]

    style A fill:#dbeafe
```

## 19. CSV 파일 처리

```mermaid
flowchart LR
    A["CSV 파일"]
    B["pd.read_csv()"]
    C["DataFrame"]
    D["분석/처리"]
    E["df.to_csv()"]
    F["CSV 파일"]

    A --> B --> C --> D --> E --> F

    style C fill:#dcfce7
```

## 20. 정렬 방법

```mermaid
flowchart TD
    A["정렬"]

    A --> B["단일 열<br>sort_values('열')"]
    A --> C["다중 열<br>sort_values(['열1','열2'])"]
    A --> D["내림차순<br>ascending=False"]

    style A fill:#dbeafe
```

## 21. 제조 데이터 분석 흐름

```mermaid
flowchart TD
    A["데이터 로드<br>read_csv()"]
    B["탐색<br>head, info"]
    C["파생변수<br>불량률 계산"]
    D["필터링<br>이상 데이터"]
    E["집계<br>groupby"]
    F["보고서<br>출력"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 22. 불량률 계산

```mermaid
flowchart LR
    A["불량수"]
    B["생산량"]
    C["불량률"]
    D["등급"]

    A --> C
    B --> C
    C --> D

    C -->|"불량수 / 생산량"| C
    D -->|"A/B/C 분류"| D

    style C fill:#fef3c7
    style D fill:#dcfce7
```

## 23. 라인별 분석 구조

```mermaid
flowchart TD
    A["전체 데이터"]

    A --> B["1번 라인"]
    A --> C["2번 라인"]
    A --> D["3번 라인"]

    B --> B1["평균 생산량"]
    B --> B2["평균 불량률"]

    C --> C1["평균 생산량"]
    C --> C2["평균 불량률"]

    D --> D1["평균 생산량"]
    D --> D2["평균 불량률"]

    style A fill:#1e40af,color:#fff
```

## 24. 이상 데이터 탐지

```mermaid
flowchart TD
    A["데이터"]

    A --> B{"불량률 > 5%?"}

    B -->|Yes| C["이상 데이터"]
    B -->|No| D["정상 데이터"]

    C --> E["원인 분석"]
    C --> F["조치 필요"]

    style C fill:#fecaca
    style D fill:#dcfce7
```

## 25. 3차시 학습 흐름

```mermaid
flowchart LR
    A["NumPy<br>배열/통계"]
    B["Pandas<br>DataFrame"]
    C["데이터 탐색"]
    D["필터/집계"]
    E["보고서"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 26. 다음 차시 연결

```mermaid
flowchart LR
    A["3차시<br>NumPy/Pandas"]
    B["4차시<br>공개 데이터셋"]
    C["5차시<br>기술통계/시각화"]

    A --> B --> C

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 27. 자주 하는 실수

```mermaid
flowchart TD
    A["자주 하는 실수"]

    A --> B["df['열'] vs df[['열']]<br>Series vs DataFrame"]
    A --> C["and/or 대신<br>& | 사용"]
    A --> D["조건 괄호 누락<br>(조건1) & (조건2)"]
    A --> E["loc vs iloc<br>끝 포함 여부"]

    style A fill:#fecaca
```

## 28. 핵심 메서드 정리

```mermaid
flowchart TD
    subgraph NumPy
        N1["np.array()"]
        N2["mean, max, min"]
        N3["arr[조건]"]
    end

    subgraph Pandas
        P1["pd.DataFrame()"]
        P2["read_csv()"]
        P3["groupby()"]
        P4["head, info, describe"]
    end

    style NumPy fill:#dbeafe
    style Pandas fill:#dcfce7
```
