# [23차시] FastAPI로 모델 서빙하기 - 강사 스크립트

## 강의 정보
- **차시**: 23차시 (25분)
- **유형**: 실습 중심
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 지난 시간 복습 [1.5분]

> 안녕하세요, 23차시를 시작하겠습니다.
>
> 지난 시간에 Streamlit으로 웹 애플리케이션을 만들어봤습니다. 슬라이더와 버튼으로 대화형 UI를 구성했죠.
>
> 오늘은 **FastAPI**로 백엔드 API를 만들어봅니다. ML 모델을 다른 프로그램이 호출할 수 있도록 서비스하는 방법이에요!

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, FastAPI의 기본 사용법을 익힙니다.
> 둘째, REST API 엔드포인트를 만듭니다.
> 셋째, ML 모델 예측 API를 구축합니다.
>
> Streamlit은 사람이 직접 사용하는 UI, FastAPI는 프로그램이 호출하는 API입니다!

---

## 전개 (19분)

### 섹션 1: FastAPI 소개 (4분)

#### FastAPI란? [2분]

> **FastAPI**는 Python으로 API를 만드는 현대적인 웹 프레임워크입니다.
>
> ```bash
> pip install fastapi uvicorn
> ```
>
> 세 가지 큰 장점이 있어요.
>
> 첫째, **빠릅니다**. Node.js나 Go 수준의 성능이에요.
> 둘째, **쉽습니다**. 직관적인 코드로 빠르게 개발할 수 있어요.
> 셋째, **자동 문서화**됩니다. Swagger UI가 자동으로 생성돼요!

#### 첫 번째 API [2분]

> *(코드 시연)*
>
> ```python
> from fastapi import FastAPI
>
> app = FastAPI()
>
> @app.get("/")
> def read_root():
>     return {"message": "Hello, FastAPI!"}
> ```
>
> 이게 전부입니다! @app.get("/")는 GET 요청을 처리하는 엔드포인트예요.
>
> ```bash
> uvicorn main:app --reload
> ```
>
> 실행하고 http://localhost:8000 에 접속하면 JSON 응답이 보여요!

---

### 섹션 2: 경로와 쿼리 매개변수 (5분)

#### 경로 매개변수 (Path Parameters) [2분]

> *(코드 시연)*
>
> ```python
> @app.get("/items/{item_id}")
> def read_item(item_id: int):
>     return {"item_id": item_id}
> ```
>
> URL에 {item_id}가 들어가면 경로 매개변수예요.
>
> /items/42 로 요청하면 {"item_id": 42} 가 반환됩니다.
>
> int 타입 힌트를 주면 자동으로 타입 검증도 해줘요!

#### 쿼리 매개변수 (Query Parameters) [2분]

> ```python
> @app.get("/items")
> def read_items(skip: int = 0, limit: int = 10):
>     return {"skip": skip, "limit": limit}
> ```
>
> /items?skip=0&limit=5 처럼 물음표 뒤에 오는 게 쿼리 매개변수예요.
>
> 기본값을 지정하면 선택적 매개변수가 됩니다.

#### 자동 문서화 [1분]

> http://localhost:8000/docs 에 접속해보세요!
>
> Swagger UI가 자동으로 생성되어 있습니다. 모든 엔드포인트를 테스트할 수 있어요.
>
> 이게 FastAPI의 가장 큰 장점 중 하나입니다!

---

### 섹션 3: POST 요청과 Pydantic (4분)

#### Request Body [2분]

> *(코드 시연)*
>
> ```python
> from pydantic import BaseModel
>
> class Item(BaseModel):
>     name: str
>     price: float
>     quantity: int = 1
>
> @app.post("/items")
> def create_item(item: Item):
>     return {"name": item.name, "total": item.price * item.quantity}
> ```
>
> POST 요청은 데이터를 서버에 보낼 때 사용해요.
>
> **Pydantic**의 BaseModel을 사용하면 요청 데이터를 자동으로 검증해줍니다!
>
> name이 문자열이 아니면 자동으로 에러를 반환해요.

#### 데이터 검증 [2분]

> Pydantic은 타입 검증을 자동으로 해줘요.
>
> 잘못된 데이터가 오면 422 Unprocessable Entity 에러가 발생합니다.
>
> 개발자가 직접 검증 코드를 작성하지 않아도 돼서 편리해요!

---

### 섹션 4: ML 모델 예측 API (6분)

#### 예측 API 만들기 [3분]

> *(코드 시연)*
>
> ```python
> from fastapi import FastAPI
> from pydantic import BaseModel
> import joblib
> import numpy as np
>
> app = FastAPI(title="품질 예측 API")
>
> # 모델 로드 (앱 시작 시 한 번만)
> model = joblib.load("model.pkl")
>
> class PredictionInput(BaseModel):
>     temperature: float
>     humidity: float
>     speed: float
>
> @app.post("/predict")
> def predict(data: PredictionInput):
>     features = np.array([[data.temperature, data.humidity, data.speed]])
>     prediction = model.predict(features)[0]
>     return {
>         "prediction": int(prediction),
>         "label": "불량" if prediction == 1 else "정상"
>     }
> ```
>
> 이게 ML 모델 서빙의 핵심이에요!
>
> 모델은 앱 시작 시 한 번만 로드하고, 요청이 올 때마다 예측을 수행합니다.

#### API 호출하기 [2분]

> *(코드 시연)*
>
> ```python
> import requests
>
> response = requests.post(
>     "http://localhost:8000/predict",
>     json={"temperature": 90, "humidity": 55, "speed": 100}
> )
> print(response.json())
> # {"prediction": 1, "label": "불량"}
> ```
>
> 다른 Python 프로그램에서 requests로 API를 호출할 수 있어요.
>
> Streamlit 앱에서 이 API를 호출하면? 프론트엔드와 백엔드가 분리된 구조가 됩니다!

#### Streamlit vs FastAPI [1분]

> 언제 뭘 쓰나요?
>
> **Streamlit**: 대시보드, 데모, 사람이 직접 사용
> **FastAPI**: 백엔드 API, 다른 프로그램이 호출
>
> 실무에서는 둘을 조합해서 씁니다. FastAPI로 모델 서빙하고, Streamlit이나 React가 호출하는 구조예요.

---

## 정리 (3분)

### 핵심 내용 요약 [1.5분]

> 오늘 배운 핵심 내용:
>
> 1. **FastAPI**: pip install fastapi uvicorn
> 2. **엔드포인트**: @app.get("/"), @app.post("/predict")
> 3. **Pydantic**: BaseModel로 데이터 검증
> 4. **모델 서빙**: joblib.load() → model.predict()
> 5. **자동 문서화**: /docs에서 Swagger UI
>
> ML 모델을 API로 배포하는 기본 구조를 배웠습니다!

### 다음 차시 예고 [1분]

> 다음 24차시에서는 **모델 해석과 특성 중요도**를 배웁니다.
>
> 모델이 왜 그렇게 예측했는지 설명하는 방법이에요.
>
> feature_importances_를 사용해서 어떤 변수가 중요한지 분석합니다!

### 마무리 인사 [0.5분]

> ML 모델을 API로 서비스하는 방법을 배웠습니다.
>
> 다음 시간에 뵙겠습니다. 수고하셨습니다!

---

## 강의 노트

### 예상 질문

1. "Flask와 뭐가 다른가요?"
   → FastAPI가 더 빠르고, 타입 힌트 기반 자동 검증, 자동 문서화 제공

2. "실제 배포는 어떻게?"
   → Docker + uvicorn, 클라우드(AWS, GCP, Azure) 사용

3. "여러 모델 서빙하려면?"
   → 각 모델별 엔드포인트 생성 (/predict/model1, /predict/model2)

### 시간 조절 팁
- 시간 부족: Docker 배포 부분 생략
- 시간 여유: 에러 처리(HTTPException) 추가 설명

### 실습 팁
- uvicorn --reload 옵션: 코드 수정 시 자동 재시작
- /docs 페이지에서 직접 API 테스트 가능
