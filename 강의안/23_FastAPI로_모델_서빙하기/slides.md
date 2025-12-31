---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 23차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# FastAPI로 모델 서빙하기

## 23차시 | AI 기초체력훈련 (Pre AI-Campus)

**ML 모델을 API로 배포하기**

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
- **빠름**: 높은 성능 (Node.js, Go 수준)
- **쉬움**: 직관적인 코드
- **자동 문서화**: Swagger UI 자동 생성
- **타입 힌트**: Python 타입 기반 검증

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

- 대안 문서 UI

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
@app.get("/users/{user_id}/items/{item_id}")
def read_user_item(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}
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

## Request Body

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1

@app.post("/items")
def create_item(item: Item):
    return {
        "name": item.name,
        "total": item.price * item.quantity
    }
```

> **Pydantic**으로 데이터 검증!

---

# ML 모델 서빙

## 예측 API 만들기

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="품질 예측 API")

# 모델 로드
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    speed: float

@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[data.temperature, data.humidity, data.speed]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction), "label": "불량" if prediction == 1 else "정상"}
```

---

# 요청 예시

## cURL / Python

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 90, "humidity": 55, "speed": 100}'
```

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"temperature": 90, "humidity": 55, "speed": 100}
)
print(response.json())
# {"prediction": 1, "label": "불량"}
```

---

# 에러 처리

## HTTPException

```python
from fastapi import HTTPException

@app.post("/predict")
def predict(data: PredictionInput):
    if data.temperature < 0 or data.temperature > 200:
        raise HTTPException(
            status_code=400,
            detail="온도는 0~200 범위여야 합니다"
        )

    # 예측 수행
    ...
```

---

# 비동기 처리

## async/await

```python
@app.get("/async-example")
async def async_example():
    # 비동기 작업 가능
    return {"message": "비동기 응답"}
```

### 언제 async?
- I/O 바운드 작업 (DB, 외부 API)
- 동시성이 중요할 때

> 단순 ML 예측은 동기로 충분

---

# 배포

## Docker + Gunicorn

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn joblib scikit-learn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

---

# Streamlit vs FastAPI

## 언제 무엇을?

| | Streamlit | FastAPI |
|--|-----------|---------|
| 용도 | 대시보드, 데모 | 백엔드 API |
| 사용자 | 직접 사용 | 다른 프로그램이 호출 |
| 출력 | 웹 UI | JSON 응답 |
| 배포 | Streamlit Cloud | Docker, 클라우드 |

> **조합**: FastAPI(백엔드) + Streamlit(프론트)

---

# 다음 차시 예고

## 24차시: 모델 해석과 특성 중요도

- Feature Importance
- 모델 해석 기법
- 실무에서의 모델 설명

> 모델이 **왜 그렇게 예측했는지** 설명하기

---

# 감사합니다

## AI 기초체력훈련 23차시

**FastAPI로 모델 서빙하기**

ML 모델을 API로 배포하는 법을 배웠습니다!
