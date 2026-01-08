# [2차시] Python 시작하기 - 다이어그램 (보강판)

## 1. Python의 위치

```mermaid
flowchart TD
    A["프로그래밍 언어"]

    A --> B["저수준<br>C, C++"]
    A --> C["고수준<br>Python, Java"]

    C --> D["데이터 과학<br>Python"]
    C --> E["엔터프라이즈<br>Java"]

    D --> F["AI/ML"]
    D --> G["데이터 분석"]
    D --> H["자동화"]

    style D fill:#dbeafe,stroke:#3b82f6
    style F fill:#dcfce7,stroke:#22c55e
```

## 2. Python 사용 분야

```mermaid
flowchart TD
    A["Python 활용 분야"]

    A --> B["AI/머신러닝"]
    A --> C["데이터 분석"]
    A --> D["웹 개발"]
    A --> E["자동화"]
    A --> F["IoT/임베디드"]

    B --> B1["TensorFlow<br>PyTorch"]
    C --> C1["Pandas<br>NumPy"]
    D --> D1["Django<br>Flask"]

    style A fill:#1e40af,color:#fff
```

## 3. Python vs 다른 언어

```mermaid
flowchart LR
    subgraph Python
        P1["쉬운 문법"]
        P2["AI 라이브러리 풍부"]
        P3["빠른 개발"]
    end

    subgraph Java
        J1["복잡한 문법"]
        J2["기업용 특화"]
        J3["안정성"]
    end

    subgraph CPP["C++"]
        C1["빠른 속도"]
        C2["어려운 문법"]
        C3["시스템 프로그래밍"]
    end

    style Python fill:#dcfce7
```

## 4. Hello World 비교

```mermaid
flowchart TD
    A["Hello World 출력"]

    A --> B["Java<br>(7줄)"]
    A --> C["C++<br>(5줄)"]
    A --> D["Python<br>(1줄)"]

    B --> B1["public class..."]
    C --> C1["#include..."]
    D --> D1["print('Hello')"]

    style D fill:#dcfce7,stroke:#22c55e
```

## 5. 개발 환경 구성

```mermaid
flowchart TD
    A["Python 개발 환경"]

    A --> B["Anaconda<br>(권장)"]
    A --> C["일반 Python"]

    B --> D["패키지 관리<br>conda"]
    B --> E["Jupyter<br>기본 포함"]
    B --> F["가상환경<br>쉽게 생성"]

    C --> G["패키지 관리<br>pip"]
    C --> H["Jupyter<br>별도 설치"]
    C --> I["가상환경<br>수동 설정"]

    style B fill:#dcfce7,stroke:#22c55e
```

## 6. Anaconda 설치 과정

```mermaid
flowchart TD
    A["anaconda.com<br>접속"]
    B["Windows 64-bit<br>다운로드"]
    C["설치 파일<br>실행"]
    D["설치 옵션<br>선택"]
    E["설치 완료<br>(~10분)"]
    F["conda --version<br>확인"]

    A --> B --> C --> D --> E --> F

    style F fill:#dcfce7,stroke:#22c55e
```

## 7. Jupyter Notebook 구조

```mermaid
flowchart TD
    A["Jupyter Notebook"]

    A --> B["메뉴바"]
    A --> C["코드 셀"]
    A --> D["마크다운 셀"]
    A --> E["출력 영역"]

    C --> C1["Python 코드<br>작성"]
    D --> D1["설명/문서<br>작성"]
    E --> E1["실행 결과<br>표시"]

    style A fill:#dbeafe
```

## 8. Jupyter 단축키

```mermaid
flowchart LR
    subgraph 실행["실행"]
        A["Shift+Enter<br>셀 실행"]
        B["Ctrl+Enter<br>현재 유지"]
    end

    subgraph 편집["셀 편집"]
        C["Esc+A<br>위에 추가"]
        D["Esc+B<br>아래 추가"]
        E["Esc+DD<br>셀 삭제"]
    end

    subgraph 변환["셀 변환"]
        F["Esc+M<br>마크다운"]
        G["Esc+Y<br>코드"]
    end
```

## 9. Python 5가지 자료형

```mermaid
flowchart TD
    A["Python 자료형"]

    A --> B["숫자"]
    A --> C["문자열<br>str"]
    A --> D["리스트<br>list"]
    A --> E["딕셔너리<br>dict"]

    B --> B1["정수 int<br>1000, 30"]
    B --> B2["실수 float<br>85.7, 0.025"]

    D --> D1["순서 O"]
    E --> E1["키-값 쌍"]

    style A fill:#1e40af,color:#fff
```

## 10. 정수와 실수

```mermaid
flowchart LR
    subgraph 정수["정수 (int)"]
        I1["생산량: 1000"]
        I2["불량수: 30"]
        I3["라인번호: 3"]
    end

    subgraph 실수["실수 (float)"]
        F1["온도: 85.7"]
        F2["불량률: 0.025"]
        F3["압력: 101.3"]
    end

    style 정수 fill:#dbeafe
    style 실수 fill:#dcfce7
```

## 11. 문자열 (str)

```mermaid
flowchart TD
    A["문자열 str"]

    A --> B["생성"]
    A --> C["연결"]
    A --> D["포맷팅"]

    B --> B1["'안녕'<br>\"Hello\""]
    C --> C1["+ 연산자"]
    D --> D1["f-string<br>f'{변수}'"]

    style D1 fill:#dcfce7
```

## 12. f-string 포맷팅

```mermaid
flowchart TD
    A["f-string"]

    A --> B["기본<br>f'{변수}'"]
    A --> C["소수점<br>f'{값:.2f}'"]
    A --> D["퍼센트<br>f'{값:.1%}'"]
    A --> E["콤마<br>f'{값:,}'"]
    A --> F["정렬<br>f'{값:>10}'"]

    style A fill:#dbeafe
```

## 13. 리스트 (list)

```mermaid
flowchart TD
    A["리스트 list"]

    A --> B["생성<br>[1, 2, 3]"]
    A --> C["접근<br>list[0]"]
    A --> D["추가<br>.append()"]
    A --> E["수정<br>list[0] = x"]

    C --> C1["인덱스 0부터"]
    C --> C2["음수: -1은 마지막"]

    style A fill:#fef3c7
```

## 14. 리스트 인덱싱

```mermaid
flowchart LR
    A["data = [82, 85, 88, 95, 84]"]

    A --> B["data[0] = 82"]
    A --> C["data[2] = 88"]
    A --> D["data[-1] = 84"]
    A --> E["data[0:3] = [82, 85, 88]"]

    style A fill:#dbeafe
```

## 15. 딕셔너리 (dict)

```mermaid
flowchart TD
    A["딕셔너리 dict"]

    A --> B["생성<br>{'키': 값}"]
    A --> C["접근<br>dict['키']"]
    A --> D["추가<br>dict['새키'] = 값"]
    A --> E["안전 접근<br>.get('키', 기본값)"]

    style A fill:#fce7f3
```

## 16. 딕셔너리 구조

```mermaid
flowchart LR
    A["sensor_data"]

    A --> B["'온도': 85.2"]
    A --> C["'압력': 101.3"]
    A --> D["'습도': 45"]
    A --> E["'상태': '정상'"]

    style A fill:#dbeafe
```

## 17. 조건문 if 구조

```mermaid
flowchart TD
    A["조건 확인"]

    A -->|참| B["if 블록 실행"]
    A -->|거짓| C["elif 조건 확인"]

    C -->|참| D["elif 블록 실행"]
    C -->|거짓| E["else 블록 실행"]

    style B fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#fce7f3
```

## 18. 품질 등급 판정 흐름

```mermaid
flowchart TD
    A["불량률 입력"]

    A --> B{"<= 1%?"}

    B -->|Yes| C["A등급<br>우수"]
    B -->|No| D{"<= 3%?"}

    D -->|Yes| E["B등급<br>양호"]
    D -->|No| F{"<= 5%?"}

    F -->|Yes| G["C등급<br>주의"]
    F -->|No| H["D등급<br>개선필요"]

    style C fill:#dcfce7
    style E fill:#dbeafe
    style G fill:#fef3c7
    style H fill:#fecaca
```

## 19. 반복문 for 구조

```mermaid
flowchart TD
    A["리스트 준비"]
    B["첫 번째 요소 처리"]
    C["다음 요소 있음?"]
    D["다음 요소 처리"]
    E["반복 종료"]

    A --> B --> C
    C -->|Yes| D --> C
    C -->|No| E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 20. enumerate 함수

```mermaid
flowchart LR
    A["temperatures<br>[82, 85, 88]"]

    A --> B["i=0, temp=82"]
    B --> C["i=1, temp=85"]
    C --> D["i=2, temp=88"]

    style A fill:#dbeafe
```

## 21. 제조 데이터 처리 흐름

```mermaid
flowchart TD
    A["센서 데이터<br>수집"]
    B["변수에<br>저장"]
    C["계산<br>불량률 등"]
    D["조건 판정<br>등급 부여"]
    E["결과<br>출력"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 22. 설비 상태 관리 구조

```mermaid
flowchart TD
    A["equipment 딕셔너리"]

    A --> B["EQ-001"]
    A --> C["EQ-002"]
    A --> D["EQ-003"]

    B --> B1["온도: 82"]
    B --> B2["상태: 정상"]

    C --> C1["온도: 91"]
    C --> C2["상태: 주의"]

    D --> D1["온도: 85"]
    D --> D2["상태: 정상"]

    style A fill:#1e40af,color:#fff
```

## 23. 자주 하는 실수

```mermaid
flowchart TD
    A["자주 하는 실수"]

    A --> B["들여쓰기<br>누락"]
    A --> C["콜론(:)<br>누락"]
    A --> D["= vs ==<br>혼동"]
    A --> E["인덱스<br>범위 초과"]

    B --> B1["IndentationError"]
    C --> C1["SyntaxError"]
    D --> D1["할당 vs 비교"]
    E --> E1["IndexError"]

    style A fill:#fecaca
```

## 24. 에러 해결 과정

```mermaid
flowchart TD
    A["에러 발생"]
    B["에러 메시지<br>읽기"]
    C["print로<br>값 확인"]
    D["구글 검색"]
    E["해결책 적용"]
    F["정상 동작"]

    A --> B --> C --> D --> E --> F

    style F fill:#dcfce7
```

## 25. 2차시 학습 흐름

```mermaid
flowchart LR
    A["환경 설정<br>Anaconda"]
    B["자료형<br>5가지"]
    C["제어문<br>if, for"]
    D["실습<br>제조 데이터"]

    A --> B --> C --> D

    style A fill:#dbeafe
    style D fill:#dcfce7
```

## 26. 다음 차시 연결

```mermaid
flowchart LR
    A["2차시<br>Python 기초"]
    B["3차시<br>NumPy/Pandas"]
    C["4차시<br>공개 데이터"]

    A --> B --> C

    style A fill:#dbeafe
    style B fill:#dcfce7
```

## 27. 함수 정의 구조

```mermaid
flowchart TD
    A["def 함수명(매개변수):"]
    B["들여쓰기된<br>코드 블록"]
    C["return 반환값"]

    A --> B --> C

    D["함수 호출<br>함수명(인자)"]
    D --> E["결과 반환"]

    style A fill:#dbeafe
    style E fill:#dcfce7
```

## 28. Python 학습 로드맵

```mermaid
flowchart TD
    A["Python 기초"]

    A --> B["자료형"]
    A --> C["제어문"]
    A --> D["함수"]

    B & C & D --> E["NumPy/Pandas"]

    E --> F["시각화<br>Matplotlib"]
    E --> G["머신러닝<br>Scikit-learn"]

    F & G --> H["딥러닝<br>TensorFlow/PyTorch"]

    style A fill:#dbeafe
    style H fill:#dcfce7
```
