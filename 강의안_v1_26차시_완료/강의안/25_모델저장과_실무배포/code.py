"""
[25차시] 모델 저장과 실무 배포 준비 - 실습 코드
제조 AI 과정 | Part IV. AI 서비스화와 활용

학습목표:
1. joblib으로 모델을 저장하고 불러온다
2. 모델 버전 관리 방법을 이해한다
3. 실무 배포 체크리스트를 활용한다

실행: python code.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# 1. 데이터 준비 및 모델 학습
# ============================================================
print("=" * 60)
print("[25차시] 모델 저장과 실무 배포 준비")
print("제조 AI 과정 | Part IV. AI 서비스화와 활용")
print("=" * 60)

print("\n▶ 1. 데이터 준비 및 모델 학습")
print("-" * 50)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 제조 데이터 생성
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n_samples),
    'humidity': np.random.normal(50, 8, n_samples),
    'speed': np.random.normal(100, 10, n_samples),
    'pressure': np.random.normal(1.0, 0.1, n_samples)
})

defect_prob = (
    0.1 +
    0.3 * ((data['temperature'] - 85) / 10) +
    0.2 * ((data['humidity'] - 50) / 16)
)
data['defect'] = (defect_prob > 0.3).astype(int)

feature_names = ['temperature', 'humidity', 'speed', 'pressure']
X = data[feature_names]
y = data['defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"모델 학습 완료!")
print(f"테스트 정확도: {accuracy:.3f}")

# ============================================================
# 2. joblib으로 모델 저장
# ============================================================
print("\n" + "=" * 50)
print("2. joblib으로 모델 저장")
print("=" * 50)

import joblib

print("""
▶ joblib이란?
   - scikit-learn 공식 권장 저장 방법
   - NumPy 배열에 최적화
   - pickle보다 대용량 모델에 효율적
""")

# 모델 저장
joblib.dump(model, 'model.pkl')
print("▶ 모델 저장: model.pkl")

# 전처리기 저장
joblib.dump(scaler, 'scaler.pkl')
print("▶ 전처리기 저장: scaler.pkl")

# 파일 크기 확인
import os
model_size = os.path.getsize('model.pkl') / 1024  # KB
scaler_size = os.path.getsize('scaler.pkl') / 1024  # KB
print(f"\n▶ 파일 크기:")
print(f"   model.pkl: {model_size:.1f} KB")
print(f"   scaler.pkl: {scaler_size:.1f} KB")

# ============================================================
# 3. 모델 불러오기 및 예측
# ============================================================
print("\n" + "=" * 50)
print("3. 모델 불러오기 및 예측")
print("=" * 50)

# 모델 불러오기
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

print("▶ 모델 로드 완료!")

# 새 데이터로 예측
new_data = np.array([
    [90, 55, 100, 1.0],   # 높은 온도, 습도
    [82, 48, 100, 1.0],   # 정상 범위
    [95, 65, 100, 1.0],   # 매우 높음
])

new_data_scaled = loaded_scaler.transform(new_data)
predictions = loaded_model.predict(new_data_scaled)

print("\n▶ 예측 결과:")
for i, (data, pred) in enumerate(zip(new_data, predictions)):
    result = "불량" if pred == 1 else "정상"
    print(f"   데이터 {i+1}: 온도={data[0]}, 습도={data[1]} → {result}")

# ============================================================
# 4. 파이프라인으로 한번에 저장
# ============================================================
print("\n" + "=" * 50)
print("4. 파이프라인으로 한번에 저장")
print("=" * 50)

from sklearn.pipeline import Pipeline

# 파이프라인 생성
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 파이프라인 학습 (원본 데이터 사용)
pipeline.fit(X_train, y_train)

# 파이프라인 저장
joblib.dump(pipeline, 'pipeline.pkl')
print("▶ 파이프라인 저장: pipeline.pkl")

# 파이프라인 불러오기 및 예측
loaded_pipeline = joblib.load('pipeline.pkl')
pipeline_predictions = loaded_pipeline.predict(new_data)

print("\n▶ 파이프라인 예측 결과:")
for i, (data, pred) in enumerate(zip(new_data, pipeline_predictions)):
    result = "불량" if pred == 1 else "정상"
    print(f"   데이터 {i+1}: {result}")

print("""
✅ 파이프라인의 장점:
   - 전처리 + 모델을 하나로 관리
   - 파일 1개만 저장/로드
   - 실수 방지 (스케일러 누락 등)
""")

# ============================================================
# 5. 메타데이터와 함께 저장
# ============================================================
print("\n" + "=" * 50)
print("5. 메타데이터와 함께 저장")
print("=" * 50)

# 모델 패키지 생성
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'version': '1.0.0',
    'trained_date': datetime.now().isoformat(),
    'accuracy': accuracy,
    'n_samples': len(X_train),
    'description': '제조 품질 예측 모델'
}

# 저장
joblib.dump(model_package, 'model_package.pkl')
print("▶ 모델 패키지 저장: model_package.pkl")

# 불러와서 정보 확인
loaded_package = joblib.load('model_package.pkl')
print("\n▶ 모델 정보:")
print(f"   버전: {loaded_package['version']}")
print(f"   학습 일시: {loaded_package['trained_date']}")
print(f"   정확도: {loaded_package['accuracy']:.3f}")
print(f"   학습 데이터 수: {loaded_package['n_samples']}")
print(f"   설명: {loaded_package['description']}")
print(f"   특성: {loaded_package['feature_names']}")

# 패키지에서 모델 추출해서 예측
pkg_model = loaded_package['model']
pkg_scaler = loaded_package['scaler']
pkg_predictions = pkg_model.predict(pkg_scaler.transform(new_data))
print(f"\n▶ 패키지 모델 예측: {pkg_predictions}")

# ============================================================
# 6. pickle vs joblib 비교
# ============================================================
print("\n" + "=" * 50)
print("6. pickle vs joblib 비교")
print("=" * 50)

import pickle
import time

# pickle 저장
start = time.time()
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)
pickle_save_time = time.time() - start

# joblib 저장
start = time.time()
joblib.dump(model, 'model_joblib.pkl')
joblib_save_time = time.time() - start

print(f"""
▶ 저장 시간 비교:
   pickle: {pickle_save_time:.4f}초
   joblib: {joblib_save_time:.4f}초

▶ 권장사항:
   - ML 모델: joblib 사용
   - 일반 Python 객체: pickle 사용
   - 대용량 NumPy 배열: joblib 압도적 우위
""")

# ============================================================
# 7. 버전 관리 패턴
# ============================================================
print("\n" + "=" * 50)
print("7. 버전 관리 패턴")
print("=" * 50)

print("""
▶ 파일명 규칙 예시:
   quality_model_v1.0_20260101.pkl
   quality_model_v1.1_20260115.pkl
   quality_model_v2.0_20260201.pkl

▶ 폴더 구조:
   models/
   ├── production/
   │   └── current_model.pkl       # 현재 사용 중
   ├── staging/
   │   └── candidate_model.pkl     # 테스트 중
   └── archive/
       ├── model_v1.0.pkl          # 이전 버전
       └── model_v1.1.pkl
""")

# 버전이 포함된 파일명으로 저장
version = "1.0"
date_str = datetime.now().strftime("%Y%m%d")
versioned_filename = f"quality_model_v{version}_{date_str}.pkl"
joblib.dump(model, versioned_filename)
print(f"▶ 버전 포함 저장: {versioned_filename}")

# ============================================================
# 8. 실무 배포 체크리스트
# ============================================================
print("\n" + "=" * 50)
print("8. 실무 배포 체크리스트")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────┐
│              배포 전 체크리스트                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ▶ 모델 검증                                        │
│     □ 테스트 데이터 성능 확인 (정확도, F1 등)       │
│     □ 다양한 입력 케이스 테스트                     │
│     □ 에지 케이스 처리 (NULL, 이상치 등)            │
│                                                      │
│  ▶ 파일 확인                                        │
│     □ 모델 파일 존재 (model.pkl)                    │
│     □ 전처리기 파일 존재 (scaler.pkl)               │
│     □ 메타데이터 기록 (버전, 날짜, 성능)            │
│                                                      │
│  ▶ 환경 확인                                        │
│     □ Python 버전 일치                              │
│     □ scikit-learn 버전 일치                        │
│     □ requirements.txt 작성                         │
│                                                      │
│  ▶ 코드 확인                                        │
│     □ 모델 로드 코드 테스트                         │
│     □ 에러 처리 로직 구현                           │
│     □ 로깅 설정                                     │
│                                                      │
└─────────────────────────────────────────────────────┘
""")

# ============================================================
# 9. requirements.txt 생성
# ============================================================
print("\n" + "=" * 50)
print("9. requirements.txt 예시")
print("=" * 50)

requirements_content = """# AI 기초체력훈련 - 모델 배포용 패키지
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
fastapi==0.100.0
uvicorn==0.23.0
pydantic==2.0.0
"""

print("▶ requirements.txt 내용:")
print(requirements_content)

# 파일로 저장
with open('requirements_example.txt', 'w') as f:
    f.write(requirements_content)
print("▶ 파일 저장: requirements_example.txt")

print("""
▶ 명령어:
   pip freeze > requirements.txt   # 현재 환경 저장
   pip install -r requirements.txt # 환경 복원
""")

# ============================================================
# 10. API 서버 코드 예시
# ============================================================
print("\n" + "=" * 50)
print("10. FastAPI 서버 코드 예시")
print("=" * 50)

api_code = '''
# main.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="품질 예측 API", version="1.0")

# 앱 시작 시 모델 로드
model_package = joblib.load("model_package.pkl")
model = model_package["model"]
scaler = model_package["scaler"]

@app.get("/health")
def health():
    return {"status": "healthy", "version": model_package["version"]}

@app.post("/predict")
def predict(temperature: float, humidity: float, speed: float, pressure: float):
    features = np.array([[temperature, humidity, speed, pressure]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return {
        "prediction": int(prediction),
        "label": "불량" if prediction == 1 else "정상"
    }

# 실행: uvicorn main:app --reload
'''

print(api_code)

# ============================================================
# 11. 정리 및 파일 정리
# ============================================================
print("\n" + "=" * 50)
print("11. 생성된 파일 정리")
print("=" * 50)

created_files = [
    'model.pkl',
    'scaler.pkl',
    'pipeline.pkl',
    'model_package.pkl',
    'model_pickle.pkl',
    'model_joblib.pkl',
    versioned_filename,
    'requirements_example.txt'
]

print("▶ 이번 실습에서 생성된 파일:")
for f in created_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"   {f}: {size:.1f} KB")

# ============================================================
# 12. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("12. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│           모델 저장과 배포 핵심 정리                  │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 저장/불러오기                                       │
│     joblib.dump(model, 'model.pkl')                   │
│     model = joblib.load('model.pkl')                  │
│                                                        │
│  ▶ 전처리기도 함께 저장                               │
│     joblib.dump(scaler, 'scaler.pkl')                 │
│     또는 Pipeline 사용                                 │
│                                                        │
│  ▶ 메타데이터 관리                                    │
│     model_info = {'model': model, 'version': '1.0'}   │
│     joblib.dump(model_info, 'package.pkl')            │
│                                                        │
│  ▶ 버전 관리                                          │
│     파일명에 버전, 날짜 포함                           │
│     requirements.txt로 환경 고정                       │
│                                                        │
│  ▶ 배포 체크리스트                                    │
│     성능 검증, 파일 확인, 환경 일치 확인              │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: AI 프로젝트 종합 실습

   26차시에서 전체 ML 워크플로우를 종합 실습합니다.
   데이터 수집 → 분석 → 모델 학습 → 배포까지!

   수고하셨습니다!
""")
