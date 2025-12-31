---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 25차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 모델 저장과 실무 배포 준비

## 25차시 | AI 기초체력훈련 (Pre AI-Campus)

**학습한 모델을 저장하고 재사용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **joblib**으로 모델을 저장하고 불러온다
2. **모델 버전 관리** 방법을 이해한다
3. **실무 배포 체크리스트**를 활용한다

---

# 왜 모델을 저장해야 하는가?

## 학습에는 시간이 걸린다!

```
데이터 준비 → 모델 학습 (10분~수시간) → 예측
```

### 저장이 필요한 이유
- 매번 학습하면 **시간 낭비**
- 다른 환경에서 **재사용** 필요
- API 서버에서 **로드만** 하면 됨

> 한 번 학습, 여러 번 사용!

---

# joblib 소개

## 가장 많이 쓰는 저장 방법

```python
import joblib

# 저장
joblib.dump(model, 'model.pkl')

# 불러오기
loaded_model = joblib.load('model.pkl')

# 예측
predictions = loaded_model.predict(X_new)
```

### 왜 joblib?
- NumPy 배열에 최적화
- pickle보다 **대용량 모델에 효율적**
- scikit-learn 공식 권장

---

# 저장 예시

## 전체 워크플로우

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 2. 성능 확인
print(f"정확도: {model.score(X_test, y_test):.3f}")

# 3. 모델 저장
joblib.dump(model, 'quality_model.pkl')
print("모델 저장 완료!")
```

---

# 불러오기 예시

## 저장된 모델 사용

```python
import joblib
import numpy as np

# 모델 불러오기
model = joblib.load('quality_model.pkl')

# 새 데이터 예측
new_data = np.array([[85, 50, 100, 1.0]])
prediction = model.predict(new_data)

print(f"예측 결과: {'불량' if prediction[0] == 1 else '정상'}")
```

### 장점
- 학습 시간 0초!
- 어디서든 동일한 예측

---

# 전처리기도 함께 저장

## scaler도 저장해야!

```python
from sklearn.preprocessing import StandardScaler
import joblib

# 학습 시
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 저장 (둘 다!)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')
```

```python
# 불러올 때
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

X_new_scaled = scaler.transform(X_new)  # 같은 스케일링!
prediction = model.predict(X_new_scaled)
```

---

# 파이프라인 저장

## 더 깔끔한 방법

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 파이프라인으로 묶기
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# 한 파일로 저장!
joblib.dump(pipeline, 'pipeline.pkl')
```

> 전처리 + 모델을 하나로 관리

---

# pickle vs joblib

## 비교

| | pickle | joblib |
|--|--------|--------|
| 내장 | Python 내장 | 설치 필요 |
| 대용량 | 느림 | **빠름** |
| NumPy | 비효율 | **최적화** |
| 사용 | 일반 객체 | ML 모델 권장 |

```python
# pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# joblib (더 간단)
joblib.dump(model, 'model.pkl')
```

---

# 모델 버전 관리

## 파일명 규칙

```
models/
├── quality_model_v1.0_20260101.pkl
├── quality_model_v1.1_20260115.pkl
├── quality_model_v2.0_20260201.pkl
└── scaler_v2.0_20260201.pkl
```

### 포함할 정보
- 모델명
- 버전 번호
- 날짜
- (선택) 성능 지표

---

# 메타데이터 저장

## 모델 정보 함께 관리

```python
import joblib
from datetime import datetime

# 모델과 메타데이터
model_info = {
    'model': model,
    'scaler': scaler,
    'feature_names': ['temp', 'humidity', 'speed', 'pressure'],
    'version': '2.0',
    'trained_date': datetime.now().isoformat(),
    'accuracy': 0.92,
    'description': '품질 예측 모델 v2'
}

joblib.dump(model_info, 'model_package.pkl')
```

```python
# 불러오기
info = joblib.load('model_package.pkl')
model = info['model']
print(f"버전: {info['version']}, 정확도: {info['accuracy']}")
```

---

# 실무 배포 체크리스트

## 배포 전 확인사항

### 1. 모델 검증
- [ ] 테스트 데이터 성능 확인
- [ ] 다양한 입력에 대한 예측 테스트
- [ ] 에지 케이스 처리 확인

### 2. 파일 확인
- [ ] 모델 파일 존재
- [ ] 전처리기 파일 존재
- [ ] 버전 정보 기록

### 3. 환경 확인
- [ ] Python 버전 일치
- [ ] scikit-learn 버전 일치
- [ ] requirements.txt 작성

---

# requirements.txt

## 패키지 버전 고정

```text
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
fastapi==0.100.0
uvicorn==0.23.0
```

```bash
# 현재 환경 저장
pip freeze > requirements.txt

# 환경 복원
pip install -r requirements.txt
```

> 버전 불일치 → 모델 로드 실패 가능!

---

# 배포 구조 예시

## 프로젝트 폴더

```
ml_project/
├── models/
│   ├── model_v2.0.pkl
│   └── scaler_v2.0.pkl
├── app/
│   └── main.py           # FastAPI 앱
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# FastAPI에서 모델 로드

## 실무 패턴

```python
from fastapi import FastAPI
import joblib

app = FastAPI()

# 앱 시작 시 한 번만 로드
model = joblib.load('models/model_v2.0.pkl')
scaler = joblib.load('models/scaler_v2.0.pkl')

@app.post("/predict")
def predict(data: PredictionInput):
    features = [[data.temp, data.humidity, data.speed, data.pressure]]
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return {"result": "불량" if prediction == 1 else "정상"}
```

---

# 과정 총정리

## 25차시 AI 기초체력훈련

| Part | 내용 |
|------|------|
| **I** | 윤리, Python, NumPy, Pandas |
| **II** | 통계, EDA, 전처리, 시각화 |
| **III** | ML 모델, 평가, 튜닝, 딥러닝 |
| **IV** | API, 웹앱, 해석, **배포** |

> 데이터 → 분석 → 모델 → 배포까지 전 과정 완료!

---

# 감사합니다

## AI 기초체력훈련 25차시

**모델 저장과 실무 배포 준비**

25차시 과정을 모두 마쳤습니다!

**수고하셨습니다!**
