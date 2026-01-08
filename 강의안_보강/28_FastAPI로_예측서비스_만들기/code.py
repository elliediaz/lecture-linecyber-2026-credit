"""
[28차시] FastAPI로 예측 서비스 만들기 - 실습 코드

학습 목표:
1. FastAPI 기본 구조를 이해한다
2. Pydantic으로 데이터를 검증한다
3. POST 엔드포인트로 예측 API를 만든다

실습 환경: Python 3.8+, fastapi, uvicorn, pydantic

실행 방법:
    uvicorn code:app --reload

또는 main.py로 저장하여:
    uvicorn main:app --reload
"""

from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    print("FastAPI 및 Pydantic 로드 완료")
except ImportError:
    print("필요 패키지 설치: pip install fastapi uvicorn pydantic")
    exit(1)

# ============================================================
# 1. FastAPI 앱 초기화
# ============================================================

app = FastAPI(
    title="품질 예측 API",
    description="""
    제조 공정 품질 예측 서비스

    ## 기능
    * 단일 센서 데이터 품질 예측
    * 배치 데이터 예측
    * 헬스 체크

    ## 사용법
    `/docs`에서 Swagger UI를 통해 API를 테스트할 수 있습니다.
    """,
    version="1.0.0",
    contact={
        "name": "제조 AI 팀",
        "email": "ai@manufacturing.com"
    }
)

# CORS 설정 (다른 도메인에서 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. Pydantic 모델 정의
# ============================================================

class SensorData(BaseModel):
    """센서 데이터 입력 모델"""
    temperature: float = Field(
        ...,
        ge=100,
        le=300,
        description="온도 (도)",
        example=200
    )
    pressure: float = Field(
        ...,
        ge=20,
        le=100,
        description="압력 (kPa)",
        example=50
    )
    speed: float = Field(
        ...,
        ge=50,
        le=200,
        description="속도 (rpm)",
        example=100
    )
    humidity: float = Field(
        default=50.0,
        ge=20,
        le=80,
        description="습도 (%)",
        example=55
    )
    vibration: float = Field(
        default=5.0,
        ge=0,
        le=15,
        description="진동 (mm/s)",
        example=5.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 200,
                "pressure": 50,
                "speed": 100,
                "humidity": 55,
                "vibration": 5.0
            }
        }

    @validator('temperature')
    def validate_temperature(cls, v, values):
        """온도 추가 검증"""
        return v

    @validator('pressure')
    def validate_pressure(cls, v, values):
        """고온에서의 압력 검증"""
        temp = values.get('temperature', 200)
        if temp > 250 and v > 80:
            raise ValueError('고온(>250)에서 압력은 80 이하여야 합니다')
        return v


class PredictionResponse(BaseModel):
    """예측 결과 응답 모델"""
    prediction: str = Field(..., description="예측 결과 (normal/defect)")
    probability: float = Field(..., ge=0, le=1, description="예측 확률")
    risk_score: int = Field(..., ge=0, le=100, description="위험 점수")
    timestamp: datetime = Field(default_factory=datetime.now, description="예측 시각")
    anomalies: List[str] = Field(default=[], description="감지된 이상 항목")
    recommendations: List[str] = Field(default=[], description="권장 조치")


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청 모델"""
    items: List[SensorData] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답 모델"""
    results: List[PredictionResponse]
    total: int
    normal_count: int
    defect_count: int
    average_risk_score: float


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    version: str
    timestamp: datetime
    model_loaded: bool


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: str
    detail: str
    timestamp: datetime


# ============================================================
# 3. 예측 로직 (시뮬레이션)
# ============================================================

class QualityPredictor:
    """품질 예측기 (시뮬레이션)"""

    def __init__(self):
        self.model_loaded = True
        self.feature_names = ['temperature', 'pressure', 'speed', 'humidity', 'vibration']
        self.thresholds = {
            'temperature': (150, 250),
            'pressure': (30, 70),
            'speed': (80, 150),
            'humidity': (40, 60),
            'vibration': (0, 10)
        }

    def predict(self, data: dict) -> dict:
        """
        품질 예측 수행

        Parameters:
        -----------
        data : dict
            센서 데이터

        Returns:
        --------
        dict : 예측 결과
        """
        # 위험 점수 계산
        risk_score = 0
        anomalies = []
        recommendations = []

        # 온도 체크
        temp = data.get('temperature', 200)
        if temp > self.thresholds['temperature'][1]:
            risk_score += 30
            anomalies.append(f"온도 초과: {temp}도 > {self.thresholds['temperature'][1]}도")
            recommendations.append("냉각 시스템 점검 필요")
        elif temp < self.thresholds['temperature'][0]:
            risk_score += 15
            anomalies.append(f"온도 미달: {temp}도 < {self.thresholds['temperature'][0]}도")
            recommendations.append("예열 시간 확인 필요")

        # 압력 체크
        pressure = data.get('pressure', 50)
        if pressure > self.thresholds['pressure'][1]:
            risk_score += 25
            anomalies.append(f"압력 초과: {pressure}kPa > {self.thresholds['pressure'][1]}kPa")
            recommendations.append("압력 밸브 확인 필요")
        elif pressure < self.thresholds['pressure'][0]:
            risk_score += 10
            anomalies.append(f"압력 미달: {pressure}kPa < {self.thresholds['pressure'][0]}kPa")

        # 속도 체크
        speed = data.get('speed', 100)
        if speed > self.thresholds['speed'][1]:
            risk_score += 15
            anomalies.append(f"속도 초과: {speed}rpm > {self.thresholds['speed'][1]}rpm")
        elif speed < self.thresholds['speed'][0]:
            risk_score += 10
            anomalies.append(f"속도 미달: {speed}rpm < {self.thresholds['speed'][0]}rpm")

        # 진동 체크
        vibration = data.get('vibration', 5)
        if vibration > self.thresholds['vibration'][1]:
            risk_score += 20
            anomalies.append(f"진동 초과: {vibration}mm/s > {self.thresholds['vibration'][1]}mm/s")
            recommendations.append("베어링 상태 점검 필요")

        # 습도 체크
        humidity = data.get('humidity', 50)
        if humidity < self.thresholds['humidity'][0] or humidity > self.thresholds['humidity'][1]:
            risk_score += 10
            if humidity < self.thresholds['humidity'][0]:
                anomalies.append(f"습도 미달: {humidity}%")
            else:
                anomalies.append(f"습도 초과: {humidity}%")

        # 최종 예측
        risk_score = min(100, risk_score)  # 최대 100
        probability = risk_score / 100

        if risk_score >= 50:
            prediction = 'defect'
            if not recommendations:
                recommendations.append("즉시 점검 필요")
        elif risk_score >= 30:
            prediction = 'warning'
            if not recommendations:
                recommendations.append("모니터링 강화 권장")
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


# 전역 예측기 인스턴스
predictor = QualityPredictor()

# ============================================================
# 4. API 엔드포인트
# ============================================================

@app.get("/", tags=["Root"])
def root():
    """
    API 루트 엔드포인트

    서비스 정보를 반환합니다.
    """
    return {
        "service": "품질 예측 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    헬스 체크 엔드포인트

    서비스 상태를 확인합니다.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        model_loaded=predictor.model_loaded
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: SensorData):
    """
    품질 예측 엔드포인트

    센서 데이터를 입력받아 품질을 예측합니다.

    - **temperature**: 온도 (100-300도)
    - **pressure**: 압력 (20-100kPa)
    - **speed**: 속도 (50-200rpm)
    - **humidity**: 습도 (20-80%, 기본값 50)
    - **vibration**: 진동 (0-15mm/s, 기본값 5)
    """
    try:
        # 예측 수행
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


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """
    배치 예측 엔드포인트

    여러 센서 데이터를 한 번에 예측합니다.
    최대 100개까지 처리 가능합니다.
    """
    try:
        results = []
        total_risk = 0

        for item in request.items:
            result = predictor.predict(item.dict())
            results.append(PredictionResponse(
                prediction=result['prediction'],
                probability=result['probability'],
                risk_score=result['risk_score'],
                timestamp=datetime.now(),
                anomalies=result['anomalies'],
                recommendations=result['recommendations']
            ))
            total_risk += result['risk_score']

        normal_count = sum(1 for r in results if r.prediction == 'normal')
        defect_count = sum(1 for r in results if r.prediction == 'defect')

        return BatchPredictionResponse(
            results=results,
            total=len(results),
            normal_count=normal_count,
            defect_count=defect_count,
            average_risk_score=total_risk / len(results) if results else 0
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"배치 예측 중 오류 발생: {str(e)}"
        )


@app.get("/thresholds", tags=["Configuration"])
def get_thresholds():
    """
    임계값 조회 엔드포인트

    센서별 정상 범위 임계값을 반환합니다.
    """
    return {
        "thresholds": predictor.thresholds,
        "description": {
            "temperature": "온도 (도)",
            "pressure": "압력 (kPa)",
            "speed": "속도 (rpm)",
            "humidity": "습도 (%)",
            "vibration": "진동 (mm/s)"
        }
    }


@app.get("/history", tags=["History"])
def get_prediction_history(
    limit: int = Query(default=10, ge=1, le=100, description="조회할 개수"),
    prediction_type: Optional[str] = Query(default=None, description="필터링할 예측 타입")
):
    """
    예측 히스토리 조회 (시뮬레이션)

    실제 서비스에서는 데이터베이스에서 조회합니다.
    """
    # 시뮬레이션 데이터
    sample_history = [
        {
            "id": i,
            "timestamp": datetime.now().isoformat(),
            "prediction": "normal" if i % 3 != 0 else "defect",
            "risk_score": np.random.randint(10, 80)
        }
        for i in range(1, limit + 1)
    ]

    if prediction_type:
        sample_history = [h for h in sample_history if h['prediction'] == prediction_type]

    return {
        "history": sample_history,
        "total": len(sample_history)
    }


# ============================================================
# 5. 에러 핸들러
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================
# 6. 시작 이벤트
# ============================================================

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    print("=" * 50)
    print("[28차시] FastAPI 품질 예측 서비스")
    print("=" * 50)
    print(f"서버 시작: {datetime.now()}")
    print(f"문서: http://localhost:8000/docs")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행"""
    print("서버 종료")


# ============================================================
# 7. 메인 실행 (개발용)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("""
    ============================================================
    [28차시] FastAPI로 예측 서비스 만들기

    실행 방법:
    1. 이 파일을 main.py로 저장
    2. 터미널에서: uvicorn main:app --reload
    3. 브라우저에서: http://localhost:8000/docs

    또는 직접 실행:
    python main.py
    ============================================================
    """)

    uvicorn.run(
        "code:app",  # 파일명:앱변수명
        host="0.0.0.0",
        port=8000,
        reload=True
    )
