---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 23차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# FastAPI로 예측 서비스 만들기

## 23차시 | Part IV. AI 서비스화와 활용

**ML 모델을 REST API로 배포하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **FastAPI**의 기본 사용법을 익힌다
2. **REST API 엔드포인트**를 만든다
3. **ML 모델 예측 API**를 구축한다

---

# FastAPI란?

## 현대적인 Python 웹 프레임워크

```bash
pip install fastapi uvicorn
```

### 특징
- **빠름**: Node.js, Go 수준의 성능
- **쉬움**: 직관적인 코드
- **자동 문서화**: Swagger UI 자동 생성
- **타입 힌트**: Python 타입 기반 검증

---

# Streamlit vs FastAPI

## 역할 비교

| 구분 | Streamlit | FastAPI |
|------|-----------|---------|
| 용도 | 대시보드, UI | 백엔드 API |
| 사용자 | 사람이 직접 사용 | 다른 프로그램이 호출 |
| 출력 | 웹 페이지 | JSON 데이터 |
| 특징 | 시각적 상호작용 | 프로그래밍 인터페이스 |

> **조합**: FastAPI(백엔드) + Streamlit(프론트엔드)

---

# Hello, FastAPI!

## 첫 번째 API

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

```bash
uvicorn main:app --reload
# http://localhost:8000 접속
```

---

# 자동 API 문서

## Swagger UI

```
http://localhost:8000/docs
```

- 모든 엔드포인트 목록
- 요청/응답 스키마
- **직접 테스트** 가능!

```
http://localhost:8000/redoc
```

- 대안 문서 UI (더 깔끔한 스타일)

---

# HTTP 메서드

## REST API 기본

| 메서드 | 용도 | 예시 |
|--------|------|------|
| GET | 데이터 조회 | 모델 정보 확인 |
| POST | 데이터 생성/처리 | 예측 요청 |
| PUT | 데이터 수정 | 설정 업데이트 |
| DELETE | 데이터 삭제 | 캐시 삭제 |

> ML 예측 API는 주로 **POST** 사용

---

# 경로 매개변수

## Path Parameters

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# GET /items/42 → {"item_id": 42}
```

```python
@app.get("/lines/{line_id}/sensors/{sensor_id}")
def read_sensor(line_id: str, sensor_id: int):
    return {"line": line_id, "sensor": sensor_id}

# GET /lines/A/sensors/1
```

---

# 쿼리 매개변수

## Query Parameters

```python
@app.get("/items")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# GET /items?skip=0&limit=5
```

```python
from typing import Optional

@app.get("/search")
def search(q: Optional[str] = None):
    if q:
        return {"query": q}
    return {"message": "검색어를 입력하세요"}
```

---

# POST 요청

## Pydantic 모델

```python
from pydantic import BaseModel

class SensorData(BaseModel):
    temperature: float
    humidity: float
    speed: float

@app.post("/data")
def create_data(data: SensorData):
    return {
        "received": True,
        "temperature": data.temperature
    }
```

> **Pydantic**으로 데이터 자동 검증!

---

# 이론 정리

## FastAPI 핵심

| 개념 | 설명 |
|------|------|
| FastAPI | Python 웹 프레임워크 |
| uvicorn | ASGI 서버 |
| @app.get | GET 엔드포인트 |
| @app.post | POST 엔드포인트 |
| Pydantic | 데이터 검증 모델 |
| /docs | 자동 문서 |

---

# - 실습편 -

## 23차시

**품질 예측 API 만들기**

---

# 실습 개요

## ML 모델 예측 API

### 목표
- FastAPI로 API 서버 만들기
- 예측 엔드포인트 구현
- 요청/응답 테스트

### 실습 환경
```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
```

---

# 실습 1: 기본 API 구조

## FastAPI 앱 생성

```python
from fastapi import FastAPI

app = FastAPI(
    title="품질 예측 API",
    description="제조 품질을 예측하는 API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "품질 예측 API 서버"}

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

# 실습 2: 입력 스키마 정의

## Pydantic 모델

```python
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    temperature: float = Field(..., ge=0, le=200, description="온도 (°C)")
    humidity: float = Field(..., ge=0, le=100, description="습도 (%)")
    speed: float = Field(..., ge=0, le=200, description="속도 (RPM)")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 85.0,
                "humidity": 50.0,
                "speed": 100.0
            }
        }
```

---

# 실습 3: 예측 함수

## 간단한 규칙 기반 예측

```python
def predict_quality(temp, humidity, speed):
    """품질 예측 (실제로는 ML 모델 사용)"""
    score = 0
    if temp > 90:
        score += 30
    if humidity > 60:
        score += 20
    if speed > 110:
        score += 15

    probability = min(score / 100, 1.0)
    prediction = 1 if probability > 0.3 else 0
    return prediction, probability
```

---

# 실습 4: 예측 엔드포인트

## POST /predict

```python
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    label: str

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    prediction, probability = predict_quality(
        data.temperature,
        data.humidity,
        data.speed
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "label": "불량" if prediction == 1 else "정상"
    }
```

---

# 실습 5: API 테스트

## cURL / Python

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 95, "humidity": 65, "speed": 100}'
```

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"temperature": 95, "humidity": 65, "speed": 100}
)
print(response.json())
# {"prediction": 1, "probability": 0.5, "label": "불량"}
```

---

# 실습 6: 에러 처리

## HTTPException

```python
from fastapi import HTTPException

@app.post("/predict")
def predict(data: PredictionInput):
    # 입력 검증
    if data.temperature < 0:
        raise HTTPException(
            status_code=400,
            detail="온도는 0 이상이어야 합니다"
        )

    # 예측 수행
    prediction, probability = predict_quality(...)
    return {...}
```

---

# 실습 7: ML 모델 연동

## joblib 모델 로드

```python
import joblib
import numpy as np

# 서버 시작 시 모델 로드
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("quality_model.pkl")

@app.post("/predict/ml")
def predict_ml(data: PredictionInput):
    features = np.array([[
        data.temperature, data.humidity, data.speed
    ]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
```

---

# 실습 8: 배치 예측

## 여러 데이터 한번에

```python
from typing import List

class BatchInput(BaseModel):
    items: List[PredictionInput]

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    results = []
    for item in data.items:
        pred, prob = predict_quality(
            item.temperature, item.humidity, item.speed
        )
        results.append({
            "prediction": pred,
            "probability": prob
        })
    return {"results": results}
```

---

# 실습 9: 모델 정보 API

## GET 엔드포인트

```python
@app.get("/model/info")
def model_info():
    return {
        "name": "Quality Prediction Model",
        "version": "1.0.0",
        "features": ["temperature", "humidity", "speed"],
        "target": "defect (0=정상, 1=불량)"
    }

@app.get("/model/features")
def model_features():
    return {
        "temperature": {"min": 70, "max": 100, "unit": "°C"},
        "humidity": {"min": 30, "max": 70, "unit": "%"},
        "speed": {"min": 80, "max": 120, "unit": "RPM"}
    }
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] FastAPI 앱 생성
- [ ] Pydantic 입력/출력 모델 정의
- [ ] POST /predict 엔드포인트 구현
- [ ] HTTPException 에러 처리
- [ ] /docs에서 API 테스트
- [ ] requests로 API 호출

---

# 배포 방법

## Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn joblib scikit-learn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t quality-api .
docker run -p 8000:8000 quality-api
```

---

# 다음 차시 예고

## 24차시: 모델 해석과 변수별 영향력 분석

### 학습 내용
- Feature Importance
- SHAP 값
- 모델 설명 기법

> 모델이 **왜 그렇게 예측했는지** 설명하기!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **FastAPI**: 빠른 Python 웹 프레임워크
2. **Pydantic**: 데이터 검증 모델
3. **@app.post**: POST 엔드포인트
4. **HTTPException**: 에러 처리
5. **/docs**: 자동 API 문서
6. **uvicorn**: ASGI 서버

---

# 감사합니다

## 23차시: FastAPI로 예측 서비스 만들기

**ML 모델을 REST API로 배포하는 법을 배웠습니다!**
