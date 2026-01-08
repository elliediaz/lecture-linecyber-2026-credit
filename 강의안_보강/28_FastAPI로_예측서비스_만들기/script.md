# [28차시] FastAPI로 예측 서비스 만들기 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 28차시 (최종) |
| 주제 | FastAPI로 예측 서비스 만들기 |
| 시간 | 30분 (이론 15분 + 실습 13분 + 정리 2분) |
| 학습 목표 | FastAPI 기본, Pydantic 검증, POST 엔드포인트 |

---

## 학습 목표

1. FastAPI 기본 구조를 이해한다
2. Pydantic으로 데이터를 검증한다
3. POST 엔드포인트로 예측 API를 만든다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | FastAPI 소개와 기본 |
| 대주제 2 | 5분 | Pydantic 데이터 검증 |
| 대주제 3 | 5분 | 예측 API 구현 |
| 실습 | 11분 | API 개발 실습 |
| 정리 | 2분 | 전체 과정 요약 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 Streamlit으로 웹앱을 만들었습니다. 사람이 브라우저에서 사용하는 UI였죠."

> "오늘은 다른 프로그램이 호출할 수 있는 API를 만듭니다. FastAPI라는 프레임워크를 사용합니다."

> "Streamlit은 사람을 위한 것, FastAPI는 프로그램을 위한 것이라고 생각하면 됩니다."

---

### 대주제 1: FastAPI 소개와 기본 (5분)

#### 슬라이드 4-6: FastAPI란

> "FastAPI는 Python 웹 프레임워크입니다. 이름처럼 빠른 성능이 특징이에요. Node.js나 Go 수준입니다."

> "가장 좋은 점은 자동으로 API 문서가 생성된다는 겁니다. 코드만 작성하면 Swagger UI 문서가 만들어져요."

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
```

---

#### 슬라이드 7-9: 실행과 문서

> "설치는 pip install fastapi uvicorn입니다. uvicorn은 서버를 실행하는 프로그램이에요."

> "실행은 uvicorn main:app --reload입니다. main은 파일명, app은 FastAPI 인스턴스 이름이에요."

> "서버가 실행되면 http://localhost:8000/docs에서 자동 생성된 문서를 볼 수 있습니다. 이게 정말 편해요."

---

#### 슬라이드 10-12: 경로와 쿼리 매개변수

> "경로 매개변수는 URL에 포함됩니다. /items/123 이런 식으로요."

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

> "타입 힌트를 쓰면 자동으로 검증됩니다. item_id: int라고 하면 문자열이 들어오면 에러가 나요."

> "쿼리 매개변수는 ?key=value 형식입니다. 기본값을 지정할 수도 있어요."

---

### 대주제 2: Pydantic 데이터 검증 (5분)

#### 슬라이드 13-15: Pydantic이란

> "Pydantic은 데이터 검증 라이브러리입니다. FastAPI와 함께 쓰면 강력해요."

```python
from pydantic import BaseModel

class SensorData(BaseModel):
    temperature: float
    pressure: float
    speed: float
```

> "BaseModel을 상속받아서 클래스를 만들면, 자동으로 타입 검증이 됩니다."

> "temperature에 문자열이 들어오면 에러가 나고, 숫자면 float로 자동 변환됩니다."

---

#### 슬라이드 16-18: 필드 검증

> "Field를 쓰면 더 상세한 검증이 가능해요."

```python
from pydantic import Field

class SensorData(BaseModel):
    temperature: float = Field(..., ge=100, le=300)
```

> "ge는 greater or equal, 100 이상이어야 합니다. le는 less or equal, 300 이하여야 해요."

> "이런 검증 로직을 직접 작성할 필요가 없어요. Pydantic이 자동으로 해줍니다."

---

### 대주제 3: 예측 API 구현 (5분)

#### 슬라이드 19-21: 프로젝트 구조

> "실제 예측 API 프로젝트 구조를 봅시다."

```
prediction_api/
├── main.py           # FastAPI 앱
├── models.py         # Pydantic 모델
├── predictor.py      # 예측 로직
└── quality_pipeline.pkl  # ML 모델
```

> "역할별로 파일을 분리하면 관리하기 좋습니다."

---

#### 슬라이드 22-24: 예측 엔드포인트

> "POST /predict 엔드포인트를 만듭니다."

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = predictor.predict(request.dict())
    return PredictionResponse(
        prediction=result['prediction'],
        probability=result['probability'],
        ...
    )
```

> "PredictionRequest로 입력을 받고, PredictionResponse로 응답합니다."

> "response_model을 지정하면 응답 형식이 문서에 명시됩니다."

---

#### 슬라이드 25-27: 테스트와 배포

> "API 테스트는 curl이나 requests로 합니다."

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"temperature": 200, "pressure": 50, "speed": 100}
)
print(response.json())
```

> "배포는 Docker를 많이 씁니다. Dockerfile 만들고 빌드하면 어디서든 실행할 수 있어요."

---

### 실습편 (11분)

#### 슬라이드 28-30: 환경 설정

```python
# 필요 패키지 설치
pip install fastapi uvicorn pydantic pandas scikit-learn joblib

# 파일 생성
# main.py
```

---

#### 슬라이드 31-33: 기본 앱 작성

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="품질 예측 API")

class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=100, le=300)
    pressure: float = Field(..., ge=20, le=100)
    speed: float = Field(..., ge=50, le=200)

class PredictionResponse(BaseModel):
    prediction: str
    risk_score: int
```

---

#### 슬라이드 34-36: 예측 엔드포인트 추가

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 간단한 규칙 기반 예측
    risk_score = 0
    if request.temperature > 250:
        risk_score += 30
    if request.pressure > 70:
        risk_score += 20

    prediction = "defect" if risk_score > 40 else "normal"

    return PredictionResponse(
        prediction=prediction,
        risk_score=risk_score
    )
```

---

#### 슬라이드 37-39: 서버 실행 및 테스트

```bash
# 터미널에서 실행
uvicorn main:app --reload

# 브라우저에서 문서 확인
# http://localhost:8000/docs

# curl로 테스트
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 200, "pressure": 50, "speed": 100}'
```

---

### 정리 (2분)

#### 슬라이드 40-41: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**FastAPI**는 빠르고 문서가 자동 생성되는 Python 웹 프레임워크입니다."

> "**Pydantic**은 데이터 검증 라이브러리로, BaseModel과 Field로 타입과 범위를 검증합니다."

> "**POST 엔드포인트**로 예측 API를 만들고, 다른 프로그램이 호출할 수 있게 했습니다."

---

#### 슬라이드 42-43: 전체 과정 정리

> "이것으로 전체 과정이 끝났습니다. 23차시부터 28차시까지 정리하면:"

> "모델 저장(joblib, Pipeline) → AI API 이해(REST, JSON) → LLM API(프롬프트) → Streamlit(웹 UI) → FastAPI(API 서버)"

> "이제 여러분은 ML 모델을 개발하고, 저장하고, 웹과 API로 배포할 수 있습니다."

> "전체 과정 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: FastAPI와 Flask 차이가 뭔가요?

> "FastAPI는 더 빠르고 타입 힌트 기반 자동 검증이 있어요. Flask는 더 간단하고 역사가 깁니다. 새 프로젝트에는 FastAPI를 추천합니다."

### Q2: 실제 서비스에서 보안은 어떻게 하나요?

> "API 키나 JWT 토큰으로 인증합니다. FastAPI는 OAuth2를 쉽게 구현할 수 있는 기능이 있어요."

### Q3: 동시 요청이 많으면 어떡하나요?

> "FastAPI는 비동기를 지원해서 동시 처리가 좋습니다. 더 많은 트래픽은 Kubernetes로 스케일링하거나 로드 밸런서를 씁니다."

### Q4: ML 모델이 크면 로드 시간이 오래 걸리는데요?

> "앱 시작 시 한 번만 로드하세요. 전역 변수나 의존성 주입으로 모델을 관리하면 됩니다."

---

## 참고 자료

### 공식 문서
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [Uvicorn 문서](https://www.uvicorn.org/)

### 관련 차시
- 26차시: Streamlit으로 웹앱 만들기
- 24차시: AI API의 이해와 활용

---

## 체크리스트

수업 전:
- [ ] FastAPI, uvicorn 설치 확인
- [ ] 예제 모델 파일 준비
- [ ] 포트 8000 사용 가능 확인

수업 중:
- [ ] Swagger 문서 시연
- [ ] Pydantic 검증 시연
- [ ] 전체 과정 연결 설명

수업 후:
- [ ] 완성된 API 코드 배포
- [ ] Docker 배포 가이드 공유
- [ ] 전체 과정 수료 안내

