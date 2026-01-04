# [20차시] AI API의 이해와 활용 - 다이어그램

## 1. API 개념

```mermaid
flowchart LR
    subgraph Client["클라이언트"]
        A["Python 코드"]
    end

    subgraph Server["서버"]
        B["AI 서비스"]
    end

    A -->|"요청 (Request)"| B
    B -->|"응답 (Response)"| A
```

## 2. API 비유: 음식점

```mermaid
flowchart LR
    subgraph Customer["손님"]
        A["주문"]
    end

    subgraph Restaurant["음식점"]
        B["메뉴판<br>(API 문서)"]
        C["주방<br>(서버)"]
    end

    A --> B --> C
    C -->|"음식<br>(결과)"| A
```

## 3. 웹 API 구조

```mermaid
flowchart TD
    A["Python 코드"]
    B["requests 라이브러리"]
    C["HTTP 요청"]
    D["인터넷"]
    E["AI 서버"]
    F["응답 (JSON)"]

    A --> B --> C --> D --> E
    E --> F --> A
```

## 4. HTTP 메서드

```mermaid
flowchart LR
    subgraph GET["GET"]
        A1["데이터 조회"]
        A2["URL 파라미터"]
    end

    subgraph POST["POST"]
        B1["데이터 전송/처리"]
        B2["요청 본문 (Body)"]
    end
```

## 5. 요청과 응답

```mermaid
sequenceDiagram
    participant C as Python 코드
    participant S as AI 서버

    C->>S: POST /predict
    Note over C,S: {"temp": 85, "humidity": 50}
    S->>C: 200 OK
    Note over C,S: {"probability": 0.15}
```

## 6. HTTP 상태 코드

```mermaid
flowchart TD
    A["상태 코드"]

    A --> B["2xx 성공"]
    A --> C["4xx 클라이언트 오류"]
    A --> D["5xx 서버 오류"]

    B --> B1["200 OK"]
    C --> C1["400 Bad Request"]
    C --> C2["401 Unauthorized"]
    C --> C3["404 Not Found"]
    D --> D1["500 Server Error"]
```

## 7. JSON 형식

```mermaid
flowchart LR
    subgraph 요청["요청 JSON"]
        A["{<br>  'temp': 85,<br>  'humidity': 50<br>}"]
    end

    subgraph 응답["응답 JSON"]
        B["{<br>  'probability': 0.15,<br>  'status': 'normal'<br>}"]
    end

    요청 --> 응답
```

## 8. API 키 인증

```mermaid
flowchart LR
    A["Python 코드"]
    B["API 키"]
    C["요청 헤더"]
    D["서버"]

    A --> B --> C --> D
    D -->|"인증 확인"| E["응답"]
```

## 9. API 키 보안

```mermaid
flowchart TD
    subgraph 잘못["❌ 잘못된 방법"]
        A1["API_KEY = 'abc123'<br>코드에 직접 작성"]
    end

    subgraph 올바름["✅ 올바른 방법"]
        B1["환경 변수"]
        B2[".env 파일"]
    end

    style 올바름 fill:#c8e6c9
```

## 10. 환경 변수 사용

```mermaid
flowchart LR
    A["터미널"]
    B["export MY_KEY=abc"]
    C["Python"]
    D["os.environ.get()"]
    E["API 키 사용"]

    A --> B
    C --> D --> E
```

## 11. AI API 종류

```mermaid
mindmap
  root((AI API))
    이미지
      객체 인식
      OCR
      얼굴 인식
    텍스트
      번역
      감정 분석
      요약
    음성
      STT
      TTS
    생성 AI
      ChatGPT
      Claude
```

## 12. 직접 개발 vs API 사용

```mermaid
flowchart TD
    subgraph 직접["직접 개발"]
        A1["데이터 수집"]
        A2["모델 학습"]
        A3["인프라 구축"]
        A4["유지보수"]
    end

    subgraph API["API 사용"]
        B1["API 키 발급"]
        B2["코드 몇 줄"]
        B3["즉시 사용"]
    end

    style API fill:#c8e6c9
```

## 13. requests 흐름

```mermaid
flowchart TD
    A["import requests"]
    B["requests.get() 또는 .post()"]
    C["response 객체"]
    D["status_code 확인"]
    E["response.json()"]

    A --> B --> C --> D --> E
```

## 14. 오류 처리

```mermaid
flowchart TD
    A["try"]
    B["API 호출"]
    C{성공?}
    D["결과 처리"]
    E["except"]
    F["오류 처리"]

    A --> B --> C
    C -->|"예"| D
    C -->|"아니오"| E --> F
```

## 15. 타임아웃

```mermaid
flowchart LR
    A["요청"]
    B{5초 내 응답?}
    C["성공"]
    D["TimeoutError"]

    A --> B
    B -->|"예"| C
    B -->|"아니오"| D
```

## 16. API 클라이언트 클래스

```mermaid
classDiagram
    class AIClient {
        -base_url
        -api_key
        -headers
        +__init__(base_url, api_key)
        +predict(data)
        +get_status()
    }
```

## 17. 제조 분야 AI API 활용

```mermaid
flowchart TD
    A["제조 현장"]

    A --> B["비전 검사 API"]
    A --> C["예측 유지보수 API"]
    A --> D["문서 OCR API"]

    B --> B1["외관 불량 검출"]
    C --> C1["설비 고장 예측"]
    D --> D1["작업 지시서 인식"]
```

## 18. 강의 구조

```mermaid
gantt
    title 20차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    API란?                  :a2, after a1, 2m
    AI API 장점              :a3, after a2, 1.5m
    requests 라이브러리        :a4, after a3, 2m
    응답 처리                :a5, after a4, 1.5m
    API 키 관리              :a6, after a5, 1m

    section 실습편
    실습 소개               :b1, after a6, 2m
    GET/POST 요청           :b2, after b1, 4m
    응답 처리               :b3, after b2, 2m
    오류 처리               :b4, after b3, 3m
    헤더/인증               :b5, after b4, 2m
    환경 변수               :b6, after b5, 2m
    클라이언트 클래스         :b7, after b6, 3m

    section 정리
    핵심 요약               :c1, after b7, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((AI API<br>활용))
    API
      요청/응답
      HTTP 메서드
      JSON 형식
    requests
      get()
      post()
      headers
    보안
      API 키
      환경 변수
    오류 처리
      try-except
      timeout
      status_code
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>기본 API"]
    B["다음<br>LLM API"]
    C["이후<br>서비스화"]

    A --> B --> C
```
