# [23차시] FastAPI로 예측 서비스 만들기 - 강사 스크립트

## 강의 정보
- **차시**: 23차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 23차시를 시작하겠습니다.
>
> 지난 시간에 Streamlit으로 품질 예측 웹앱을 만들었습니다. 슬라이더와 버튼으로 사용자가 직접 예측을 실행하는 대시보드를 구축했죠.
>
> 오늘은 **FastAPI**로 백엔드 API를 만들어봅니다. ML 모델을 다른 프로그램이 호출할 수 있도록 서비스하는 방법이에요.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, FastAPI의 기본 사용법을 익힙니다.
> 둘째, REST API 엔드포인트를 만듭니다.
> 셋째, ML 모델 예측 API를 구축합니다.

---

### 핵심 내용 (8분)

#### FastAPI란? [1.5min]

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

#### Streamlit vs FastAPI [1min]

> Streamlit과 FastAPI는 목적이 달라요.
>
> **Streamlit**은 웹 UI를 만들어요. 사람이 직접 사용하는 대시보드죠.
> **FastAPI**는 REST API를 만들어요. 다른 프로그램이 호출하는 인터페이스입니다.
>
> 실무에서는 둘을 조합해요. FastAPI로 백엔드를 만들고, Streamlit이나 다른 프론트엔드가 호출하는 구조입니다.

#### 기본 사용법 [2min]

> 첫 번째 API를 만들어볼게요.
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
> `@app.get("/")`는 GET 요청을 처리하는 데코레이터예요. 딕셔너리를 반환하면 자동으로 JSON으로 변환돼요.
>
> `uvicorn main:app --reload`로 실행하면 http://localhost:8000 에서 확인할 수 있어요.

#### HTTP 메서드와 Pydantic [1.5min]

> REST API에서 주로 사용하는 메서드는 네 가지예요.
>
> **GET**은 데이터 조회, **POST**는 데이터 생성이나 처리, **PUT**은 수정, **DELETE**는 삭제예요.
> ML 예측 API는 주로 **POST**를 사용해요. 입력 데이터를 보내서 예측을 받으니까요.
>
> **Pydantic**은 데이터 검증 라이브러리예요.
>
> ```python
> from pydantic import BaseModel
>
> class SensorData(BaseModel):
>     temperature: float
>     humidity: float
> ```
>
> 이렇게 모델을 정의하면 요청 데이터를 자동으로 검증해줍니다. 타입이 맞지 않으면 에러를 반환해요.

#### 자동 문서화 [1min]

> FastAPI의 가장 큰 장점 중 하나가 **자동 문서화**예요.
>
> `http://localhost:8000/docs`에 접속하면 Swagger UI가 나와요.
> 모든 엔드포인트 목록, 요청/응답 스키마를 볼 수 있고, 직접 테스트도 가능해요.
>
> 따로 문서를 작성하지 않아도 되니까 정말 편리합니다.

#### 이론 정리 [1min]

> 정리하면, FastAPI는 빠르고 쉬운 Python 웹 프레임워크예요.
> uvicorn으로 서버를 실행하고, Pydantic으로 데이터를 검증하고, /docs에서 자동 문서를 볼 수 있습니다.

---

## 실습편 (15-20분)

### 실습 소개 [1.5min]

> 이제 실습 시간입니다. 품질 예측 API를 만들어봅니다.
>
> **실습 목표**입니다.
> 1. FastAPI 앱을 생성합니다.
> 2. 예측 엔드포인트를 구현합니다.
> 3. API를 테스트합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from fastapi import FastAPI
> from pydantic import BaseModel
> ```

### 실습 1: 기본 API 구조 [2min]

> 첫 번째 실습입니다. FastAPI 앱을 생성합니다.
>
> ```python
> from fastapi import FastAPI
>
> app = FastAPI(
>     title="품질 예측 API",
>     description="제조 품질을 예측하는 API",
>     version="1.0.0"
> )
>
> @app.get("/")
> def root():
>     return {"message": "품질 예측 API 서버"}
>
> @app.get("/health")
> def health():
>     return {"status": "ok"}
> ```
>
> title과 description은 Swagger UI에 표시돼요. health 엔드포인트는 서버 상태 확인용입니다.

### 실습 2: 입력 스키마 정의 [2min]

> 두 번째 실습입니다. Pydantic으로 입력 데이터 스키마를 정의합니다.
>
> ```python
> from pydantic import BaseModel, Field
>
> class PredictionInput(BaseModel):
>     temperature: float = Field(..., ge=0, le=200, description="온도")
>     humidity: float = Field(..., ge=0, le=100, description="습도")
>     speed: float = Field(..., ge=0, le=200, description="속도")
> ```
>
> `Field`를 사용하면 값의 범위를 제한할 수 있어요. ge는 이상, le는 이하입니다.
> 범위를 벗어나면 자동으로 에러를 반환해요.

### 실습 3: 예측 함수 [2min]

> 세 번째 실습입니다. 예측 함수를 만듭니다.
>
> ```python
> def predict_quality(temp, humidity, speed):
>     score = 0
>     if temp > 90: score += 30
>     if humidity > 60: score += 20
>     if speed > 110: score += 15
>
>     probability = min(score / 100, 1.0)
>     prediction = 1 if probability > 0.3 else 0
>     return prediction, probability
> ```
>
> 간단한 규칙 기반 예측이에요. 실제로는 여기에 ML 모델을 넣으면 됩니다.

### 실습 4: 예측 엔드포인트 [2min]

> 네 번째 실습입니다. POST /predict 엔드포인트를 구현합니다.
>
> ```python
> class PredictionOutput(BaseModel):
>     prediction: int
>     probability: float
>     label: str
>
> @app.post("/predict", response_model=PredictionOutput)
> def predict(data: PredictionInput):
>     prediction, probability = predict_quality(
>         data.temperature, data.humidity, data.speed
>     )
>     return {
>         "prediction": prediction,
>         "probability": probability,
>         "label": "불량" if prediction == 1 else "정상"
>     }
> ```
>
> `response_model`로 출력 스키마도 정의할 수 있어요. 문서화에 도움이 됩니다.

### 실습 5: API 테스트 [2min]

> 다섯 번째 실습입니다. API를 테스트합니다.
>
> /docs 페이지에서 직접 테스트하거나, Python으로 호출할 수 있어요.
>
> ```python
> import requests
>
> response = requests.post(
>     "http://localhost:8000/predict",
>     json={"temperature": 95, "humidity": 65, "speed": 100}
> )
> print(response.json())
> ```
>
> 지난 시간에 배운 requests 라이브러리를 사용하면 됩니다.

### 실습 6: 에러 처리 [1.5min]

> 여섯 번째 실습입니다. 에러 처리를 추가합니다.
>
> ```python
> from fastapi import HTTPException
>
> @app.post("/predict")
> def predict(data: PredictionInput):
>     if data.temperature < 0:
>         raise HTTPException(
>             status_code=400,
>             detail="온도는 0 이상이어야 합니다"
>         )
>     ...
> ```
>
> `HTTPException`을 발생시키면 해당 상태 코드와 메시지를 반환해요.

### 실습 7: ML 모델 연동 [2min]

> 일곱 번째 실습입니다. 실제 ML 모델을 연동합니다.
>
> ```python
> import joblib
>
> model = None
>
> @app.on_event("startup")
> def load_model():
>     global model
>     model = joblib.load("quality_model.pkl")
> ```
>
> `@app.on_event("startup")`은 서버 시작 시 한 번 실행돼요.
> 모델을 미리 로드해두면 요청마다 로드하지 않아서 빠릅니다.

### 실습 8: 배치 예측 [1.5min]

> 여덟 번째 실습입니다. 여러 데이터를 한번에 예측합니다.
>
> ```python
> from typing import List
>
> class BatchInput(BaseModel):
>     items: List[PredictionInput]
>
> @app.post("/predict/batch")
> def predict_batch(data: BatchInput):
>     results = []
>     for item in data.items:
>         pred, prob = predict_quality(...)
>         results.append({"prediction": pred, "probability": prob})
>     return {"results": results}
> ```
>
> 배치 예측은 여러 데이터를 한번에 처리할 때 유용해요.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **FastAPI**는 Python 웹 프레임워크로, uvicorn으로 실행해요.
>
> **Pydantic**으로 입력/출력 데이터를 검증해요. BaseModel을 상속해서 스키마를 정의합니다.
>
> **엔드포인트**는 @app.get, @app.post 데코레이터로 만들어요. ML 예측은 주로 POST를 사용합니다.
>
> **/docs**에서 자동 생성된 Swagger UI로 API를 테스트할 수 있어요.

#### 다음 차시 예고 [1min]

> 다음 24차시에서는 **모델 해석과 변수별 영향력 분석**을 배웁니다.
>
> 모델이 왜 그렇게 예측했는지 설명하는 방법이에요. feature_importances_와 SHAP을 사용합니다.

#### 마무리 [0.5min]

> ML 모델을 REST API로 배포하는 법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- FastAPI, uvicorn 설치

### 주의사항
- uvicorn --reload 옵션: 개발 시 코드 수정하면 자동 재시작
- 모델 로드는 startup 이벤트에서 한 번만
- 프로덕션에서는 gunicorn + uvicorn 조합 사용

### 예상 질문
1. "Flask와 뭐가 다른가요?"
   → FastAPI가 더 빠르고, 타입 힌트 기반 자동 검증, 자동 문서화 제공

2. "실제 배포는 어떻게?"
   → Docker + uvicorn, 클라우드(AWS, GCP, Azure) 사용

3. "Streamlit 앱에서 FastAPI 호출하려면?"
   → requests.post()로 API 호출, JSON 응답 처리

4. "비동기 처리가 필요한가요?"
   → 단순 ML 예측은 동기로 충분. I/O 바운드 작업 많으면 async 고려
