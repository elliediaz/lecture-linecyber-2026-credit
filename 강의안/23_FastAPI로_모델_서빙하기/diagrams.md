# [23차시] FastAPI로 예측 서비스 만들기 - 다이어그램

## 1. FastAPI란?

```mermaid
flowchart LR
    A["Python 코드<br>(main.py)"]
    B["FastAPI"]
    C["uvicorn 서버"]
    D["REST API"]

    A --> B --> C --> D
```

## 2. Streamlit vs FastAPI

```mermaid
flowchart TD
    subgraph Streamlit["Streamlit"]
        A1["웹 UI"]
        A2["대시보드"]
        A3["사람이 사용"]
    end

    subgraph FastAPI["FastAPI"]
        B1["REST API"]
        B2["JSON 응답"]
        B3["프로그램이 호출"]
    end

    style Streamlit fill:#dbeafe
    style FastAPI fill:#d1fae5
```

## 3. REST API 흐름

```mermaid
sequenceDiagram
    participant C as 클라이언트
    participant F as FastAPI
    participant M as ML 모델

    C->>F: POST /predict
    Note over C,F: {"temp": 90, "humidity": 55}
    F->>M: model.predict()
    M->>F: prediction
    F->>C: 200 OK
    Note over C,F: {"prediction": 1, "label": "불량"}
```

## 4. HTTP 메서드

```mermaid
flowchart LR
    subgraph Methods["HTTP 메서드"]
        A["GET<br>조회"]
        B["POST<br>생성/처리"]
        C["PUT<br>수정"]
        D["DELETE<br>삭제"]
    end

    B --> E["ML 예측에 주로 사용"]
```

## 5. FastAPI 앱 구조

```mermaid
flowchart TD
    A["from fastapi import FastAPI"]
    B["app = FastAPI()"]
    C["@app.get('/')"]
    D["@app.post('/predict')"]
    E["uvicorn main:app"]

    A --> B --> C & D --> E
```

## 6. 경로 매개변수

```mermaid
flowchart LR
    A["GET /items/42"]
    B["@app.get('/items/{item_id}')"]
    C["item_id = 42"]
    D["return {'item_id': 42}"]

    A --> B --> C --> D
```

## 7. 쿼리 매개변수

```mermaid
flowchart LR
    A["GET /items?skip=0&limit=5"]
    B["@app.get('/items')"]
    C["skip=0, limit=5"]
    D["return {'skip': 0, 'limit': 5}"]

    A --> B --> C --> D
```

## 8. Pydantic 모델

```mermaid
flowchart TD
    A["class PredictionInput(BaseModel)"]

    A --> B["temperature: float"]
    A --> C["humidity: float"]
    A --> D["speed: float"]

    E["자동 검증"]
    B & C & D --> E
```

## 9. 요청/응답 구조

```mermaid
flowchart LR
    subgraph Request["요청 (Request)"]
        A1["{<br>'temperature': 90,<br>'humidity': 55,<br>'speed': 100<br>}"]
    end

    subgraph Response["응답 (Response)"]
        B1["{<br>'prediction': 1,<br>'probability': 0.5,<br>'label': '불량'<br>}"]
    end

    Request --> Response
```

## 10. POST /predict 흐름

```mermaid
flowchart TD
    A["POST /predict"]
    B["JSON 파싱"]
    C["Pydantic 검증"]
    D{유효한 데이터?}
    E["predict_quality()"]
    F["응답 반환"]
    G["HTTPException"]

    A --> B --> C --> D
    D -->|예| E --> F
    D -->|아니오| G
```

## 11. 에러 처리

```mermaid
flowchart TD
    A["입력 데이터"]
    B{검증}
    C["정상 처리"]
    D["HTTPException"]
    E["400 Bad Request"]
    F["500 Server Error"]

    A --> B
    B -->|통과| C
    B -->|실패| D --> E
```

## 12. 자동 문서화

```mermaid
flowchart LR
    A["FastAPI 코드"]
    B["/docs<br>(Swagger UI)"]
    C["/redoc<br>(ReDoc)"]

    A --> B & C

    B --> D["API 테스트 가능"]
    C --> E["문서 열람"]
```

## 13. ML 모델 연동

```mermaid
flowchart TD
    A["서버 시작"]
    B["@app.on_event('startup')"]
    C["model = joblib.load()"]
    D["요청 수신"]
    E["model.predict()"]
    F["응답 반환"]

    A --> B --> C
    D --> E --> F
```

## 14. 배치 예측

```mermaid
flowchart TD
    A["BatchInput"]
    B["items: List[PredictionInput]"]
    C["for item in items"]
    D["predict_quality()"]
    E["results 리스트"]
    F["응답"]

    A --> B --> C --> D --> E --> F
```

## 15. Docker 배포

```mermaid
flowchart LR
    A["Dockerfile"]
    B["docker build"]
    C["이미지"]
    D["docker run"]
    E["컨테이너"]
    F["API 서비스"]

    A --> B --> C --> D --> E --> F
```

## 16. API 클라이언트

```mermaid
flowchart TD
    subgraph Clients["클라이언트"]
        A1["cURL"]
        A2["Python requests"]
        A3["Streamlit"]
        A4["모바일 앱"]
    end

    B["FastAPI 서버"]

    Clients --> B
```

## 17. 전체 아키텍처

```mermaid
flowchart LR
    subgraph Frontend["프론트엔드"]
        A["Streamlit<br>웹 UI"]
    end

    subgraph Backend["백엔드"]
        B["FastAPI<br>REST API"]
        C["ML 모델"]
    end

    subgraph Data["데이터"]
        D["센서 데이터"]
    end

    A <-->|HTTP| B
    B --> C
    D --> A & B
```

## 18. 강의 구조

```mermaid
gantt
    title 23차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    FastAPI란?              :a2, after a1, 1.5m
    Streamlit vs FastAPI    :a3, after a2, 1m
    기본 사용법              :a4, after a3, 2m
    HTTP 메서드             :a5, after a4, 1m
    Pydantic               :a6, after a5, 1.5m
    이론 정리               :a7, after a6, 1m

    section 실습편
    실습 소개               :b1, after a7, 1.5m
    기본 API 구조           :b2, after b1, 2m
    입력 스키마 정의         :b3, after b2, 2m
    예측 함수               :b4, after b3, 2m
    예측 엔드포인트          :b5, after b4, 2m
    API 테스트              :b6, after b5, 2m
    에러 처리               :b7, after b6, 1.5m
    ML 모델 연동            :b8, after b7, 2m
    배치 예측               :b9, after b8, 1.5m
    모델 정보 API           :b10, after b9, 1.5m

    section 정리
    핵심 요약               :c1, after b10, 1.5m
    다음 차시 예고           :c2, after c1, 1m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((FastAPI<br>예측 API))
    기본
      uvicorn
      @app.get
      @app.post
    Pydantic
      BaseModel
      Field
      자동 검증
    문서
      /docs
      Swagger UI
    에러
      HTTPException
      status_code
    배포
      Docker
      uvicorn
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>FastAPI<br>(API 서빙)"]
    B["다음<br>모델 해석<br>(특성 중요도)"]
    C["이후<br>모델 저장<br>배포"]

    A --> B --> C
```
