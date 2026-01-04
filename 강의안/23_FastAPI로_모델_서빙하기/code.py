"""
[23차시] FastAPI로 예측 서비스 만들기 - 실습 코드
제조 AI 과정 | Part IV. AI 서비스화와 활용

학습목표:
1. FastAPI의 기본 사용법을 익힌다
2. REST API 엔드포인트를 만든다
3. ML 모델 예측 API를 구축한다

실행: uvicorn code:app --reload
문서: http://localhost:8000/docs
"""

# ============================================================
# 1. FastAPI 설치 확인
# ============================================================
print("=" * 50)
print("[23차시] FastAPI로 예측 서비스 만들기")
print("제조 AI 과정 | Part IV. AI 서비스화와 활용")
print("=" * 50)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI가 설치되지 않았습니다.")
    print("설치: pip install fastapi uvicorn")

import numpy as np

# ============================================================
# 2. FastAPI 앱 생성
# ============================================================
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="품질 예측 API",
        description="제조 공정 품질을 예측하는 ML 모델 API",
        version="1.0.0"
    )

    # ============================================================
    # 3. 기본 엔드포인트
    # ============================================================

    @app.get("/")
    def read_root():
        """루트 엔드포인트 - API 정보 반환"""
        return {
            "message": "품질 예측 API에 오신 것을 환영합니다!",
            "docs": "/docs에서 API 문서를 확인하세요"
        }

    @app.get("/health")
    def health_check():
        """헬스 체크 엔드포인트"""
        return {"status": "healthy", "model_loaded": True}

    # ============================================================
    # 4. 경로 매개변수 예시
    # ============================================================

    @app.get("/items/{item_id}")
    def read_item(item_id: int, detail: bool = False):
        """
        경로 매개변수와 쿼리 매개변수 예시

        - item_id: 경로 매개변수 (필수)
        - detail: 쿼리 매개변수 (선택)
        """
        result = {"item_id": item_id}
        if detail:
            result["description"] = f"Item {item_id}의 상세 정보"
        return result

    # ============================================================
    # 5. Pydantic 모델 정의
    # ============================================================

    class PredictionInput(BaseModel):
        """예측 요청 데이터 모델"""
        temperature: float = Field(..., ge=0, le=200, description="온도 (°C)")
        humidity: float = Field(..., ge=0, le=100, description="습도 (%)")
        speed: float = Field(..., ge=0, le=200, description="속도 (rpm)")

        class Config:
            schema_extra = {
                "example": {
                    "temperature": 85.0,
                    "humidity": 50.0,
                    "speed": 100.0
                }
            }

    class PredictionOutput(BaseModel):
        """예측 응답 데이터 모델"""
        prediction: int
        label: str
        confidence: float
        input_data: dict

    class BatchPredictionInput(BaseModel):
        """배치 예측 요청 데이터 모델"""
        data: list[PredictionInput]

    # ============================================================
    # 6. 간단한 예측 함수 (실제로는 model.predict 사용)
    # ============================================================

    def mock_predict(temperature: float, humidity: float, speed: float):
        """
        가상의 예측 함수
        실제로는 joblib.load('model.pkl')로 로드한 모델 사용
        """
        # 간단한 규칙 기반 예측
        defect_prob = 0.1 + 0.02 * (temperature - 85) + 0.01 * (humidity - 50)
        defect_prob = max(0, min(1, defect_prob))

        prediction = 1 if defect_prob > 0.3 else 0
        confidence = defect_prob if prediction == 1 else (1 - defect_prob)

        return prediction, confidence

    # ============================================================
    # 7. 예측 엔드포인트
    # ============================================================

    @app.post("/predict", response_model=PredictionOutput)
    def predict(data: PredictionInput):
        """
        단일 예측 수행

        - 온도, 습도, 속도를 입력받아 품질 예측
        - 결과: 0(정상) 또는 1(불량)
        """
        prediction, confidence = mock_predict(
            data.temperature,
            data.humidity,
            data.speed
        )

        return PredictionOutput(
            prediction=prediction,
            label="불량" if prediction == 1 else "정상",
            confidence=round(confidence, 3),
            input_data=data.dict()
        )

    @app.post("/predict/batch")
    def predict_batch(batch_data: BatchPredictionInput):
        """
        배치 예측 수행

        - 여러 데이터를 한 번에 예측
        """
        results = []
        for item in batch_data.data:
            prediction, confidence = mock_predict(
                item.temperature,
                item.humidity,
                item.speed
            )
            results.append({
                "prediction": prediction,
                "label": "불량" if prediction == 1 else "정상",
                "confidence": round(confidence, 3)
            })

        return {
            "count": len(results),
            "results": results
        }

    # ============================================================
    # 8. 에러 처리 예시
    # ============================================================

    @app.post("/predict/validated")
    def predict_with_validation(data: PredictionInput):
        """
        검증이 포함된 예측 엔드포인트
        """
        # 추가 비즈니스 로직 검증
        if data.temperature > 150:
            raise HTTPException(
                status_code=400,
                detail="온도가 너무 높습니다. 150°C 이하로 입력하세요."
            )

        if data.speed < 50:
            raise HTTPException(
                status_code=400,
                detail="속도가 너무 낮습니다. 50rpm 이상으로 입력하세요."
            )

        prediction, confidence = mock_predict(
            data.temperature,
            data.humidity,
            data.speed
        )

        return {
            "prediction": prediction,
            "label": "불량" if prediction == 1 else "정상",
            "confidence": round(confidence, 3)
        }

# ============================================================
# 9. API 호출 예시 (클라이언트 코드)
# ============================================================
print("\n" + "=" * 50)
print("API 호출 예시 코드")
print("=" * 50)

print("""
# FastAPI 서버 실행 (터미널에서):
uvicorn code:app --reload

# Python에서 API 호출:
import requests

# 단일 예측
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "temperature": 90,
        "humidity": 55,
        "speed": 100
    }
)
print(response.json())
# 출력: {"prediction": 1, "label": "불량", "confidence": 0.35, ...}

# 배치 예측
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "data": [
            {"temperature": 85, "humidity": 50, "speed": 100},
            {"temperature": 92, "humidity": 60, "speed": 95},
            {"temperature": 80, "humidity": 45, "speed": 105}
        ]
    }
)
print(response.json())

# cURL로 호출:
# curl -X POST "http://localhost:8000/predict" \\
#   -H "Content-Type: application/json" \\
#   -d '{"temperature": 90, "humidity": 55, "speed": 100}'
""")

# ============================================================
# 10. 실제 모델 서빙 코드 예시
# ============================================================
print("\n" + "=" * 50)
print("실제 모델 서빙 코드 예시")
print("=" * 50)

print("""
# 실제 ML 모델 서빙 시 코드 구조:

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 앱 시작 시 모델 로드 (한 번만)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")  # 전처리기도 로드

class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    speed: float

@app.post("/predict")
def predict(data: PredictionInput):
    # 입력 데이터 준비
    features = np.array([[
        data.temperature,
        data.humidity,
        data.speed
    ]])

    # 전처리 적용
    features_scaled = scaler.transform(features)

    # 예측 수행
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "불량" if prediction == 1 else "정상",
        "confidence": float(max(probability))
    }
""")

# ============================================================
# 11. Docker 배포 예시
# ============================================================
print("\n" + "=" * 50)
print("Docker 배포 예시")
print("=" * 50)

print("""
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# requirements.txt
fastapi
uvicorn
joblib
numpy
scikit-learn

# 빌드 및 실행
docker build -t ml-api .
docker run -p 8000:8000 ml-api
""")

# ============================================================
# 12. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│              FastAPI 모델 서빙 핵심 정리               │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 설치                                               │
│     pip install fastapi uvicorn                       │
│                                                        │
│  ▶ 기본 구조                                           │
│     app = FastAPI()                                   │
│     @app.get("/") / @app.post("/predict")            │
│                                                        │
│  ▶ Pydantic 모델                                       │
│     class Input(BaseModel):                           │
│         feature: float                                │
│                                                        │
│  ▶ 실행                                               │
│     uvicorn main:app --reload                        │
│                                                        │
│  ▶ 문서                                               │
│     http://localhost:8000/docs (Swagger UI)          │
│                                                        │
│  ▶ Streamlit vs FastAPI                              │
│     Streamlit = UI/대시보드 (사람이 사용)             │
│     FastAPI = 백엔드 API (프로그램이 호출)            │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 모델 해석과 변수별 영향력 분석
""")

# ============================================================
# 13. 테스트 실행 (FastAPI 없이도 확인 가능)
# ============================================================
if not FASTAPI_AVAILABLE:
    print("\n" + "=" * 50)
    print("테스트 실행 (FastAPI 없이)")
    print("=" * 50)

    # mock_predict 함수 정의 (FastAPI 없을 때)
    def mock_predict(temperature, humidity, speed):
        defect_prob = 0.1 + 0.02 * (temperature - 85) + 0.01 * (humidity - 50)
        defect_prob = max(0, min(1, defect_prob))
        prediction = 1 if defect_prob > 0.3 else 0
        confidence = defect_prob if prediction == 1 else (1 - defect_prob)
        return prediction, confidence

    test_cases = [
        (85, 50, 100),   # 정상 조건
        (92, 60, 100),   # 높은 온도, 습도
        (80, 45, 100),   # 낮은 온도, 습도
    ]

    print("\n예측 테스트:")
    for temp, hum, speed in test_cases:
        pred, conf = mock_predict(temp, hum, speed)
        label = "불량" if pred == 1 else "정상"
        print(f"  온도={temp}, 습도={hum}, 속도={speed}")
        print(f"  → 예측: {label} (신뢰도: {conf:.1%})")
        print()
