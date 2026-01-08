# [25차시] AI API의 이해와 활용 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["23차시<br>모델 저장"]
    B["25차시<br>AI API"]
    C["25차시<br>LLM API"]

    A --> B --> C

    B --> B1["REST API 개념"]
    B --> B2["requests 라이브러리"]
    B --> B3["JSON 처리"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["25차시: AI API의 이해와 활용"]

    A --> B["대주제 1<br>REST API 개념"]
    A --> C["대주제 2<br>requests 라이브러리"]
    A --> D["대주제 3<br>JSON 처리"]

    B --> B1["HTTP 메서드<br>상태 코드"]
    C --> C1["get/post<br>응답 처리"]
    D --> D1["직렬화<br>역직렬화"]

    style A fill:#1e40af,color:#fff
```

## 3. API 기본 개념

```mermaid
flowchart LR
    A["클라이언트<br>(내 프로그램)"]
    B["API"]
    C["서버<br>(외부 서비스)"]

    A -->|요청| B
    B -->|응답| A
    B <--> C

    style B fill:#fef3c7
```

## 4. REST API 구조

```mermaid
flowchart TD
    A["REST API"]

    A --> B["URL"]
    B --> B1["자원 식별<br>/api/products/123"]

    A --> C["HTTP 메서드"]
    C --> C1["동작 정의<br>GET, POST, PUT, DELETE"]

    A --> D["헤더"]
    D --> D1["메타정보<br>인증, 타입"]

    A --> E["바디"]
    E --> E1["데이터<br>JSON 형식"]

    style A fill:#1e40af,color:#fff
```

## 5. HTTP 메서드

```mermaid
flowchart TD
    A["HTTP 메서드"]

    A --> B["GET"]
    B --> B1["데이터 조회"]

    A --> C["POST"]
    C --> C1["데이터 생성"]

    A --> D["PUT"]
    D --> D1["데이터 수정"]

    A --> E["DELETE"]
    E --> E1["데이터 삭제"]

    style C fill:#dcfce7
```

## 6. 요청-응답 흐름

```mermaid
sequenceDiagram
    participant C as 클라이언트
    participant S as 서버

    C->>S: POST /predict
    Note over C,S: {"temperature": 200}
    S->>S: 모델 예측
    S->>C: 200 OK
    Note over C,S: {"prediction": "normal"}
```

## 7. HTTP 상태 코드

```mermaid
flowchart TD
    A["HTTP 상태 코드"]

    A --> B["2xx 성공"]
    B --> B1["200 OK"]
    B --> B2["201 Created"]

    A --> C["4xx 클라이언트 에러"]
    C --> C1["400 Bad Request"]
    C --> C2["401 Unauthorized"]
    C --> C3["404 Not Found"]

    A --> D["5xx 서버 에러"]
    D --> D1["500 Internal Error"]

    style B fill:#dcfce7
    style C fill:#fef3c7
    style D fill:#fecaca
```

## 8. API 인증 방식

```mermaid
flowchart TD
    A["API 인증"]

    A --> B["API Key"]
    B --> B1["Header:<br>X-API-Key: xxx"]

    A --> C["Bearer Token"]
    C --> C1["Header:<br>Authorization: Bearer xxx"]

    A --> D["Basic Auth"]
    D --> D1["Header:<br>Authorization: Basic xxx"]

    style A fill:#1e40af,color:#fff
```

## 9. requests 라이브러리

```mermaid
flowchart TD
    A["requests"]

    A --> B["GET 요청"]
    B --> B1["requests.get(url)"]

    A --> C["POST 요청"]
    C --> C1["requests.post(url, json=data)"]

    A --> D["응답 처리"]
    D --> D1["response.status_code"]
    D --> D2["response.json()"]

    style A fill:#1e40af,color:#fff
```

## 10. GET 요청 흐름

```mermaid
flowchart LR
    A["requests.get()"]
    B["URL + 파라미터"]
    C["서버"]
    D["response"]

    A --> B --> C
    C --> D --> A

    style A fill:#dbeafe
```

## 11. POST 요청 흐름

```mermaid
flowchart LR
    A["requests.post()"]
    B["URL + JSON 데이터"]
    C["서버"]
    D["response"]

    A --> B --> C
    C --> D --> A

    style A fill:#dcfce7
```

## 12. 응답 객체 속성

```mermaid
flowchart TD
    A["response"]

    A --> B["status_code"]
    B --> B1["200, 404, 500..."]

    A --> C["text"]
    C --> C1["응답 문자열"]

    A --> D["json()"]
    D --> D1["딕셔너리 변환"]

    A --> E["ok"]
    E --> E1["True/False"]

    style A fill:#1e40af,color:#fff
```

## 13. 에러 처리 패턴

```mermaid
flowchart TD
    A["API 호출"]
    B{"성공?"}
    C["데이터 처리"]
    D["에러 처리"]

    A --> B
    B -->|Yes| C
    B -->|No| D

    D --> D1["로깅"]
    D --> D2["재시도"]
    D --> D3["알림"]

    style C fill:#dcfce7
    style D fill:#fecaca
```

## 14. JSON 구조

```mermaid
flowchart TD
    A["JSON"]

    A --> B["객체 { }"]
    B --> B1["키-값 쌍"]

    A --> C["배열 [ ]"]
    C --> C1["순서있는 목록"]

    A --> D["값 타입"]
    D --> D1["string, number"]
    D --> D2["true, false, null"]

    style A fill:#1e40af,color:#fff
```

## 15. Python-JSON 변환

```mermaid
flowchart LR
    A["Python dict"]
    B["json.dumps()"]
    C["JSON 문자열"]
    D["json.loads()"]

    A --> B --> C
    C --> D --> A

    style B fill:#dbeafe
    style D fill:#dcfce7
```

## 16. JSON 파일 처리

```mermaid
flowchart TD
    A["JSON 파일"]

    A --> B["저장"]
    B --> B1["json.dump(data, f)"]

    A --> C["로드"]
    C --> C1["json.load(f)"]

    style A fill:#fef3c7
```

## 17. requests + JSON 자동 처리

```mermaid
flowchart LR
    A["Python dict"]
    B["requests.post(<br>json=data)"]
    C["자동 변환"]
    D["JSON 전송"]

    A --> B --> C --> D

    E["JSON 응답"]
    F["response.json()"]
    G["자동 변환"]
    H["Python dict"]

    E --> F --> G --> H

    style C fill:#dcfce7
    style G fill:#dcfce7
```

## 18. 중첩 JSON 구조

```mermaid
flowchart TD
    A["request"]

    A --> B["input"]
    B --> C["sensors"]
    C --> C1["temperature: 200"]
    C --> C2["pressure: 50"]

    B --> D["metadata"]
    D --> D1["machine_id: M001"]
    D --> D2["timestamp: ..."]

    style A fill:#1e40af,color:#fff
```

## 19. API 호출 완전 흐름

```mermaid
flowchart TD
    A["1. 데이터 준비"]
    B["2. JSON 변환"]
    C["3. HTTP 요청"]
    D["4. 응답 수신"]
    E["5. 상태 확인"]
    F{"성공?"}
    G["6. JSON 파싱"]
    H["7. 데이터 사용"]
    I["에러 처리"]

    A --> B --> C --> D --> E --> F
    F -->|Yes| G --> H
    F -->|No| I

    style H fill:#dcfce7
    style I fill:#fecaca
```

## 20. 세션 사용

```mermaid
flowchart TD
    A["Session 생성"]
    B["공통 헤더 설정"]
    C["요청 1"]
    D["요청 2"]
    E["요청 N"]
    F["Session 종료"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dbeafe
```

## 21. 타임아웃 처리

```mermaid
flowchart TD
    A["요청 전송"]
    B{"5초 내<br>응답?"}
    C["정상 처리"]
    D["Timeout 예외"]
    E["에러 처리"]

    A --> B
    B -->|Yes| C
    B -->|No| D --> E

    style C fill:#dcfce7
    style D fill:#fecaca
```

## 22. 제조업 API 활용

```mermaid
flowchart TD
    A["제조 시스템"]

    A --> B["품질 예측 API"]
    B --> B1["ML 모델 호출"]

    A --> C["설비 모니터링 API"]
    C --> C1["센서 데이터 조회"]

    A --> D["이상 분석 API"]
    D --> D1["LLM 원인 분석"]

    style A fill:#1e40af,color:#fff
```

## 23. 실습 흐름

```mermaid
flowchart TD
    A["1. 공개 API 호출"]
    B["2. GET 요청 테스트"]
    C["3. POST 요청 테스트"]
    D["4. JSON 변환 실습"]
    E["5. 에러 처리 실습"]
    F["6. 예측 시뮬레이션"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 24. 핵심 함수 정리

```mermaid
flowchart TD
    A["핵심 함수"]

    A --> B["requests"]
    B --> B1["get(url, params)"]
    B --> B2["post(url, json)"]

    A --> C["json"]
    C --> C1["dumps() - 직렬화"]
    C --> C2["loads() - 역직렬화"]

    A --> D["response"]
    D --> D1["status_code"]
    D --> D2["json()"]

    style A fill:#1e40af,color:#fff
```

## 25. 핵심 정리

```mermaid
flowchart TD
    A["25차시 핵심"]

    A --> B["REST API"]
    B --> B1["HTTP 기반<br>GET/POST"]

    A --> C["requests"]
    C --> C1["API 호출<br>응답 처리"]

    A --> D["JSON"]
    D --> D1["데이터 교환<br>형식"]

    style A fill:#1e40af,color:#fff
```

## 26. 실무 팁

```mermaid
flowchart TD
    A["실무 팁"]

    A --> B["API 키"]
    B --> B1["환경 변수로 관리"]

    A --> C["타임아웃"]
    C --> C1["항상 설정"]

    A --> D["에러 처리"]
    D --> D1["try-except 필수"]

    A --> E["로깅"]
    E --> E1["요청/응답 기록"]

    style A fill:#fef3c7
```

## 27. 다음 차시 연결

```mermaid
flowchart LR
    A["25차시<br>AI API"]
    B["25차시<br>LLM API"]

    A --> B

    A --> A1["REST API 기초"]
    A --> A2["requests 사용"]

    B --> B1["OpenAI/Claude"]
    B --> B2["프롬프트 작성"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

