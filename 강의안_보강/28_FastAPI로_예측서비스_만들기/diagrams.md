# [28차시] FastAPI로 예측 서비스 만들기 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["26차시<br>Streamlit"]
    B["28차시<br>FastAPI"]
    C["과정 완료"]

    A --> B --> C

    B --> B1["FastAPI 기본"]
    B --> B2["Pydantic 검증"]
    B --> B3["예측 API"]

    style B fill:#1e40af,color:#fff
    style C fill:#dcfce7
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["28차시: FastAPI로 예측 서비스 만들기"]

    A --> B["대주제 1<br>FastAPI 소개"]
    A --> C["대주제 2<br>Pydantic 검증"]
    A --> D["대주제 3<br>예측 API"]

    B --> B1["GET/POST<br>경로/쿼리"]
    C --> C1["BaseModel<br>Field 검증"]
    D --> D1["엔드포인트<br>배포"]

    style A fill:#1e40af,color:#fff
```

## 3. Streamlit vs FastAPI

```mermaid
flowchart TD
    A["비교"]

    A --> B["Streamlit"]
    B --> B1["웹 UI"]
    B --> B2["사람이 사용"]
    B --> B3["브라우저 접속"]

    A --> C["FastAPI"]
    C --> C1["REST API"]
    C --> C2["프로그램이 호출"]
    C --> C3["HTTP 요청"]

    style B fill:#dbeafe
    style C fill:#dcfce7
```

## 4. FastAPI 요청 흐름

```mermaid
flowchart LR
    A["클라이언트"]
    B["HTTP 요청"]
    C["FastAPI 서버"]
    D["응답 (JSON)"]

    A --> B --> C --> D --> A

    style C fill:#1e40af,color:#fff
```

## 5. FastAPI 실행 흐름

```mermaid
flowchart LR
    A["main.py 작성"]
    B["uvicorn main:app"]
    C["localhost:8000"]
    D["/docs (Swagger)"]

    A --> B --> C --> D

    style D fill:#dcfce7
```

## 6. HTTP 메서드

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

## 7. 데코레이터 구조

```mermaid
flowchart TD
    A["@app.get('/path')"]
    B["def function()"]
    C["return {'key': 'value'}"]

    A --> B --> C

    style A fill:#fef3c7
```

## 8. 경로 매개변수

```mermaid
flowchart LR
    A["/items/{item_id}"]
    B["item_id: int"]
    C["자동 타입 검증"]

    A --> B --> C

    D["/items/123"]
    E["item_id = 123"]

    D --> E

    style C fill:#dcfce7
```

## 9. 쿼리 매개변수

```mermaid
flowchart LR
    A["/search?q=temp&limit=10"]
    B["q: str"]
    C["limit: int = 10"]

    A --> B
    A --> C

    style A fill:#fef3c7
```

## 10. Pydantic BaseModel

```mermaid
flowchart TD
    A["BaseModel"]

    A --> B["필드 정의"]
    B --> B1["name: str"]
    B --> B2["value: float"]

    A --> C["자동 검증"]
    C --> C1["타입 체크"]
    C --> C2["변환"]

    style A fill:#1e40af,color:#fff
```

## 11. Pydantic Field 옵션

```mermaid
flowchart TD
    A["Field 옵션"]

    A --> B["ge/le"]
    B --> B1[">=, <="]

    A --> C["gt/lt"]
    C --> C1[">, <"]

    A --> D["min_length"]
    D --> D1["문자열 최소 길이"]

    A --> E["description"]
    E --> E1["필드 설명"]

    style A fill:#1e40af,color:#fff
```

## 12. 요청/응답 모델

```mermaid
flowchart TD
    A["API 모델"]

    A --> B["Request 모델"]
    B --> B1["입력 데이터 정의"]

    A --> C["Response 모델"]
    C --> C1["출력 데이터 정의"]

    style A fill:#1e40af,color:#fff
```

## 13. POST 요청 흐름

```mermaid
flowchart TD
    A["POST /predict"]
    B["Request Body (JSON)"]
    C["PredictionRequest 검증"]
    D["예측 로직"]
    E["PredictionResponse"]
    F["JSON 응답"]

    A --> B --> C --> D --> E --> F

    style C fill:#fef3c7
    style D fill:#dbeafe
```

## 14. 데이터 검증 흐름

```mermaid
flowchart TD
    A["JSON 입력"]
    B{"Pydantic<br>검증"}
    C["검증 통과"]
    D["422 에러"]
    E["비즈니스 로직"]

    A --> B
    B -->|Valid| C --> E
    B -->|Invalid| D

    style C fill:#dcfce7
    style D fill:#fecaca
```

## 15. 프로젝트 구조

```mermaid
flowchart TD
    A["prediction_api/"]

    A --> B["main.py"]
    B --> B1["FastAPI 앱"]

    A --> C["models.py"]
    C --> C1["Pydantic 모델"]

    A --> D["predictor.py"]
    D --> D1["예측 로직"]

    A --> E["model.pkl"]
    E --> E1["ML 모델"]

    style A fill:#1e40af,color:#fff
```

## 16. 예측 API 구조

```mermaid
flowchart TD
    A["main.py"]

    A --> B["FastAPI 초기화"]
    A --> C["모델 로드"]
    A --> D["@app.post('/predict')"]
    A --> E["@app.get('/health')"]

    D --> D1["예측 수행"]
    D --> D2["결과 반환"]

    style A fill:#1e40af,color:#fff
```

## 17. 에러 처리

```mermaid
flowchart TD
    A["요청 처리"]
    B{"에러 발생?"}
    C["정상 응답"]
    D["HTTPException"]
    E["에러 응답"]

    A --> B
    B -->|No| C
    B -->|Yes| D --> E

    style C fill:#dcfce7
    style E fill:#fecaca
```

## 18. 배치 예측

```mermaid
flowchart LR
    A["BatchRequest"]
    B["items: List"]
    C["각 항목 예측"]
    D["BatchResponse"]
    E["results: List"]

    A --> B --> C --> D --> E

    style C fill:#dbeafe
```

## 19. API 테스트

```mermaid
flowchart TD
    A["API 테스트"]

    A --> B["Swagger UI"]
    B --> B1["/docs"]

    A --> C["curl"]
    C --> C1["curl -X POST"]

    A --> D["Python"]
    D --> D1["requests.post()"]

    style A fill:#1e40af,color:#fff
```

## 20. Docker 배포

```mermaid
flowchart TD
    A["Dockerfile"]
    B["docker build"]
    C["Docker Image"]
    D["docker run"]
    E["컨테이너 실행"]

    A --> B --> C --> D --> E

    style E fill:#dcfce7
```

## 21. 배포 옵션

```mermaid
flowchart TD
    A["배포 방식"]

    A --> B["Docker"]
    B --> B1["컨테이너화"]

    A --> C["클라우드"]
    C --> C1["AWS/GCP/Azure"]

    A --> D["서버리스"]
    D --> D1["Lambda/Cloud Run"]

    style A fill:#1e40af,color:#fff
```

## 22. 실습 흐름

```mermaid
flowchart TD
    A["1. 환경 설정"]
    B["2. models.py 작성"]
    C["3. main.py 작성"]
    D["4. 서버 실행"]
    E["5. API 테스트"]
    F["6. 배포"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 23. 핵심 코드 구조

```mermaid
flowchart TD
    A["FastAPI 핵심"]

    A --> B["FastAPI()"]
    B --> B1["앱 인스턴스"]

    A --> C["@app.post()"]
    C --> C1["엔드포인트"]

    A --> D["BaseModel"]
    D --> D1["데이터 검증"]

    style A fill:#1e40af,color:#fff
```

## 24. 전체 과정 요약

```mermaid
flowchart LR
    A["23차시<br>모델 저장"]
    B["24차시<br>AI API"]
    C["25차시<br>LLM API"]
    D["26차시<br>Streamlit"]
    E["28차시<br>FastAPI"]

    A --> B --> C --> D --> E

    style E fill:#dcfce7
```

## 25. ML 서비스 아키텍처

```mermaid
flowchart TD
    A["ML 모델"]
    B["FastAPI 서버"]
    C["REST API"]

    D["웹 앱<br>(Streamlit)"]
    E["다른 시스템"]
    F["모바일 앱"]

    A --> B --> C

    C --> D
    C --> E
    C --> F

    style B fill:#1e40af,color:#fff
```

## 26. 핵심 정리

```mermaid
flowchart TD
    A["28차시 핵심"]

    A --> B["FastAPI"]
    B --> B1["빠른 Python<br>웹 프레임워크"]

    A --> C["Pydantic"]
    C --> C1["자동 데이터<br>검증"]

    A --> D["POST API"]
    D --> D1["예측 서비스<br>엔드포인트"]

    style A fill:#1e40af,color:#fff
```

## 27. 과정 완료

```mermaid
flowchart TD
    A["제조업 AI 과정 완료"]

    A --> B["데이터 분석"]
    A --> C["ML 모델 개발"]
    A --> D["모델 해석"]
    A --> E["배포와 서비스"]

    E --> E1["저장 (joblib)"]
    E --> E2["UI (Streamlit)"]
    E --> E3["API (FastAPI)"]

    style A fill:#1e40af,color:#fff
    style E fill:#dcfce7
```

