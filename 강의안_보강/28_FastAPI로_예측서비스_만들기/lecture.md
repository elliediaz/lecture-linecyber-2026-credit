# 28차시: FastAPI로 예측 서비스 만들기

## 학습 목표

1. **FastAPI** 기본 구조를 이해함
2. **Pydantic**으로 데이터를 검증함
3. **POST 엔드포인트**로 예측 API를 만듦

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | FastAPI 소개와 기본 |
| 대주제 2 | 10분 | Pydantic 데이터 검증 |
| 대주제 3 | 8분 | 예측 API 구현 |
| 정리 | 2분 | 핵심 요약 |

---

## 대주제 1: FastAPI 소개와 기본

### 1.1 Streamlit vs FastAPI

| 항목 | Streamlit | FastAPI |
|-----|-----------|---------|
| **용도** | 사용자 UI | API 서버 |
| **호출 방식** | 브라우저 | HTTP 요청 |
| **대상** | 사람 | 프로그램 |
| **출력** | 웹 페이지 | JSON |

- **Streamlit**: 사람이 쓰는 웹앱
- **FastAPI**: 프로그램이 호출하는 API

### 1.2 FastAPI란?

**Python 웹 프레임워크**
- 빠른 성능 (Node.js, Go 수준)
- 자동 API 문서 생성
- 타입 힌트 기반 검증
- 비동기 지원

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
```

### 1.3 FastAPI 설치

```bash
pip install fastapi uvicorn
```

- **fastapi**: 프레임워크
- **uvicorn**: ASGI 서버 (앱 실행용)

### 1.4 첫 번째 API

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

실행:
```bash
uvicorn main:app --reload
```

### 1.5 서버 실행

```bash
uvicorn main:app --reload

INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

- **main**: 파일명 (main.py)
- **app**: FastAPI 인스턴스 이름
- **--reload**: 코드 변경 시 자동 재시작

### 1.6 자동 API 문서

FastAPI는 자동으로 문서를 생성함

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

코드만 작성하면 문서가 자동 생성

### 1.7 HTTP 메서드 데코레이터

```python
@app.get("/items")      # 조회
def get_items():
    return {"items": [...]}

@app.post("/items")     # 생성
def create_item():
    return {"created": True}

@app.put("/items/{id}") # 수정
def update_item(id: int):
    return {"updated": id}

@app.delete("/items/{id}") # 삭제
def delete_item(id: int):
    return {"deleted": id}
```

### 1.8 경로 매개변수

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# /items/123 -> {"item_id": 123}
# /items/abc -> 에러 (int 아님)
```

타입 힌트로 자동 검증

### 1.9 쿼리 매개변수

```python
@app.get("/search")
def search_items(q: str, limit: int = 10):
    return {"query": q, "limit": limit}

# /search?q=temperature&limit=5
# -> {"query": "temperature", "limit": 5}

# /search?q=temperature
# -> {"query": "temperature", "limit": 10}  # 기본값
```

### 1.10 경로 + 쿼리 조합

```python
@app.get("/lines/{line_id}/products")
def get_line_products(
    line_id: int,
    status: str = "all",
    limit: int = 100
):
    return {
        "line_id": line_id,
        "status": status,
        "limit": limit
    }

# /lines/3/products?status=defect&limit=50
```

---

## 대주제 2: Pydantic 데이터 검증

### 2.1 Pydantic이란?

**데이터 검증 라이브러리**
- 타입 검증 자동화
- JSON <-> Python 변환
- FastAPI와 완벽 통합

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1
```

### 2.2 BaseModel 정의

```python
from pydantic import BaseModel
from typing import Optional

class SensorData(BaseModel):
    temperature: float
    pressure: float
    speed: float
    humidity: Optional[float] = 50.0
    vibration: float = 5.0
```

- **필수 필드**: 타입만 지정
- **선택 필드**: Optional 또는 기본값

### 2.3 자동 검증 예시

```python
# 유효한 데이터
data = SensorData(
    temperature=200,
    pressure=50,
    speed=100
)
# OK! humidity=50.0, vibration=5.0 (기본값)

# 잘못된 데이터
data = SensorData(
    temperature="hot",  # 문자열!
    pressure=50,
    speed=100
)
# 에러: temperature must be float
```

### 2.4 POST 요청에서 사용

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SensorData(BaseModel):
    temperature: float
    pressure: float
    speed: float

@app.post("/predict")
def predict(data: SensorData):
    return {
        "received": data.dict(),
        "prediction": "normal"
    }
```

### 2.5 요청/응답 모델 분리

```python
# 요청 모델
class PredictionRequest(BaseModel):
    temperature: float
    pressure: float
    speed: float
    humidity: float = 50.0
    vibration: float = 5.0

# 응답 모델
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_score: int
```

### 2.6 응답 모델 지정

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # 예측 로직
    return PredictionResponse(
        prediction="normal",
        probability=0.85,
        risk_score=15
    )
```

응답 형식이 문서에 명시됨

### 2.7 필드 검증

```python
from pydantic import BaseModel, Field

class SensorData(BaseModel):
    temperature: float = Field(
        ...,  # 필수
        ge=100,  # >= 100
        le=300,  # <= 300
        description="온도 (도)"
    )
    pressure: float = Field(
        ...,
        ge=20,
        le=100,
        description="압력 (kPa)"
    )
```

### 2.8 Field 옵션

| 옵션 | 의미 |
|-----|------|
| `ge` | >= (greater or equal) |
| `le` | <= (less or equal) |
| `gt` | > (greater than) |
| `lt` | < (less than) |
| `min_length` | 문자열 최소 길이 |
| `max_length` | 문자열 최대 길이 |
| `regex` | 정규식 패턴 |

### 2.9 복잡한 검증

```python
from pydantic import BaseModel, validator

class SensorData(BaseModel):
    temperature: float
    pressure: float

    @validator('temperature')
    def check_temperature(cls, v):
        if v < 100 or v > 300:
            raise ValueError('온도는 100-300 범위여야 합니다')
        return v

    @validator('pressure')
    def check_pressure(cls, v, values):
        temp = values.get('temperature', 200)
        if temp > 250 and v > 70:
            raise ValueError('고온에서 압력이 너무 높습니다')
        return v
```

---

## 대주제 3: 예측 API 구현

### 3.1 프로젝트 구조

```
prediction_api/
+-- main.py           # FastAPI 앱
+-- models.py         # Pydantic 모델
+-- predictor.py      # 예측 로직
+-- quality_pipeline.pkl  # ML 모델
+-- requirements.txt  # 의존성
```

### 3.2 models.py - 데이터 모델

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=100, le=300)
    pressure: float = Field(..., ge=20, le=100)
    speed: float = Field(..., ge=50, le=200)
    humidity: float = Field(50.0, ge=20, le=80)
    vibration: float = Field(5.0, ge=0, le=15)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_score: int
    timestamp: datetime
    anomalies: List[str] = []
```

### 3.3 predictor.py - 예측 로직

```python
import joblib
import pandas as pd

class QualityPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.features = ['temperature', 'pressure', 'speed',
                        'humidity', 'vibration']

    def predict(self, data: dict) -> dict:
        df = pd.DataFrame([data])
        df = df[self.features]

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]

        return {
            'prediction': 'defect' if prediction == 1 else 'normal',
            'probability': float(max(probability)),
            'defect_prob': float(probability[1])
        }
```

### 3.4 main.py - FastAPI 앱

```python
from fastapi import FastAPI, HTTPException
from models import PredictionRequest, PredictionResponse
from predictor import QualityPredictor
from datetime import datetime

app = FastAPI(
    title="품질 예측 API",
    description="제조 공정 품질 예측 서비스",
    version="1.0.0"
)

# 모델 로드
predictor = QualityPredictor('quality_pipeline.pkl')
```

### 3.5 예측 엔드포인트

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # 예측 수행
        result = predictor.predict(request.dict())

        # 이상 항목 확인
        anomalies = []
        if request.temperature > 250:
            anomalies.append("온도 초과")
        if request.vibration > 10:
            anomalies.append("진동 초과")

        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            risk_score=int(result['defect_prob'] * 100),
            timestamp=datetime.now(),
            anomalies=anomalies
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.6 헬스 체크 엔드포인트

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    return {
        "service": "Quality Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

### 3.7 배치 예측 엔드포인트

```python
from typing import List

class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    defect_count: int

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    results = []
    for item in request.items:
        result = predict(item)
        results.append(result)

    defect_count = sum(1 for r in results if r.prediction == 'defect')

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        defect_count=defect_count
    )
```

### 3.8 에러 처리

```python
from fastapi import HTTPException
from pydantic import ValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors()
        }
    )

# 직접 에러 발생
@app.post("/predict")
def predict(request: PredictionRequest):
    if request.temperature > 300:
        raise HTTPException(
            status_code=400,
            detail="온도가 허용 범위를 초과했습니다"
        )
```

### 3.9 API 테스트 - curl

```bash
# GET 요청
curl http://localhost:8000/health

# POST 요청
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 200,
    "pressure": 50,
    "speed": 100,
    "humidity": 55,
    "vibration": 5
  }'
```

### 3.10 API 테스트 - Python

```python
import requests

url = "http://localhost:8000/predict"

data = {
    "temperature": 200,
    "pressure": 50,
    "speed": 100,
    "humidity": 55,
    "vibration": 5
}

response = requests.post(url, json=data)
print(response.json())
# {"prediction": "normal", "probability": 0.85, ...}
```

### 3.11 배포 옵션

| 방식 | 특징 |
|-----|------|
| **Docker** | 컨테이너화, 이식성 |
| **클라우드** | AWS, GCP, Azure |
| **서버리스** | AWS Lambda, GCP Cloud Run |
| **온프레미스** | 직접 서버 운영 |

### 3.12 Docker 배포

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t quality-api .
docker run -p 8000:8000 quality-api
```

### 3.13 requirements.txt

```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## 실습: 품질 예측 API 전체 코드

### Pydantic 모델 정의

```python
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class SensorData(BaseModel):
    """센서 데이터 입력 모델"""
    temperature: float = Field(
        ..., ge=100, le=300,
        description="온도 (도)", example=200
    )
    pressure: float = Field(
        ..., ge=20, le=100,
        description="압력 (kPa)", example=50
    )
    speed: float = Field(
        ..., ge=50, le=200,
        description="속도 (rpm)", example=100
    )
    humidity: float = Field(
        default=50.0, ge=20, le=80,
        description="습도 (%)", example=55
    )
    vibration: float = Field(
        default=5.0, ge=0, le=15,
        description="진동 (mm/s)", example=5.0
    )

class PredictionResponse(BaseModel):
    """예측 결과 응답 모델"""
    prediction: str
    probability: float
    risk_score: int
    timestamp: datetime
    anomalies: List[str] = []
    recommendations: List[str] = []

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    version: str
    timestamp: datetime
    model_loaded: bool
```

### 예측기 클래스

```python
class QualityPredictor:
    """품질 예측기"""

    def __init__(self):
        self.model_loaded = True
        self.thresholds = {
            'temperature': (150, 250),
            'pressure': (30, 70),
            'speed': (80, 150),
            'humidity': (40, 60),
            'vibration': (0, 10)
        }

    def predict(self, data: dict) -> dict:
        """품질 예측 수행"""
        risk_score = 0
        anomalies = []
        recommendations = []

        # 온도 체크
        temp = data.get('temperature', 200)
        if temp > self.thresholds['temperature'][1]:
            risk_score += 30
            anomalies.append(f"온도 초과: {temp}도")
            recommendations.append("냉각 시스템 점검 필요")

        # 압력 체크
        pressure = data.get('pressure', 50)
        if pressure > self.thresholds['pressure'][1]:
            risk_score += 25
            anomalies.append(f"압력 초과: {pressure}kPa")
            recommendations.append("압력 밸브 확인 필요")

        # 진동 체크
        vibration = data.get('vibration', 5)
        if vibration > self.thresholds['vibration'][1]:
            risk_score += 20
            anomalies.append(f"진동 초과: {vibration}mm/s")
            recommendations.append("베어링 상태 점검 필요")

        # 최종 예측
        risk_score = min(100, risk_score)
        probability = risk_score / 100

        if risk_score >= 50:
            prediction = 'defect'
        elif risk_score >= 30:
            prediction = 'warning'
        else:
            prediction = 'normal'
            recommendations = ["정상 운전 유지"]

        return {
            'prediction': prediction,
            'probability': probability,
            'risk_score': risk_score,
            'anomalies': anomalies,
            'recommendations': recommendations
        }

predictor = QualityPredictor()
```

### FastAPI 앱 정의

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(
    title="품질 예측 API",
    description="제조 공정 품질 예측 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "품질 예측 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        model_loaded=predictor.model_loaded
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: SensorData):
    try:
        result = predictor.predict(data.dict())

        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            risk_score=result['risk_score'],
            timestamp=datetime.now(),
            anomalies=result['anomalies'],
            recommendations=result['recommendations']
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}"
        )
```

### 실행 코드

```python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

## 핵심 정리

### 1. FastAPI 기본
- `@app.get()`, `@app.post()` 데코레이터
- 경로/쿼리 매개변수
- 자동 문서 생성 (/docs)

### 2. Pydantic 검증
- BaseModel 정의
- Field 옵션 (ge, le, gt, lt)
- @validator 커스텀 검증

### 3. 예측 API
- POST 엔드포인트
- 에러 처리 (HTTPException)
- 배치 예측

### 핵심 코드

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=100, le=300)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    result = model.predict(data.dict())
    return PredictionResponse(...)
```

---

## 과정 정리

| 차시 | 주제 | 핵심 |
|-----|------|------|
| 24 | 모델 저장 | joblib, Pipeline |
| 25 | AI API | REST, requests, JSON |
| 26 | LLM API | 프롬프트 엔지니어링 |
| 27 | Streamlit | 웹 UI |
| **28** | **FastAPI** | **REST API 서버** |

---

## 체크리스트

- [ ] FastAPI 설치 및 실행
- [ ] GET/POST 엔드포인트 구현
- [ ] Pydantic 모델 정의
- [ ] 필드 검증 추가
- [ ] 예측 API 구현
- [ ] 에러 처리 추가
- [ ] API 테스트
