---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 25차시'
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

# 모델 저장과 실무 배포 준비

## 25차시 | Part IV. AI 서비스화와 활용

**학습한 모델을 저장하고 재사용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **joblib**으로 모델을 저장하고 불러온다
2. **모델 버전 관리** 방법을 이해한다
3. **실무 배포 체크리스트**를 활용한다

---

# 왜 모델을 저장해야 하나?

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

# 저장 (한 줄!)
joblib.dump(model, 'model.pkl')

# 불러오기 (한 줄!)
loaded_model = joblib.load('model.pkl')

# 예측
predictions = loaded_model.predict(X_new)
```

### 왜 joblib?
- NumPy 배열에 최적화
- pickle보다 **대용량 모델에 효율적**
- scikit-learn 공식 권장

---

# 저장 워크플로우

## 전체 흐름

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

print(f"예측: {'불량' if prediction[0] == 1 else '정상'}")
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

> 예측할 때 같은 스케일러를 써야 올바른 결과!

---

# 파이프라인으로 한번에

## 더 깔끔한 방법

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# 하나의 파일로 저장!
joblib.dump(pipeline, 'pipeline.pkl')
```

> 전처리 + 모델을 하나로 관리

---

# 이론 정리

## 모델 저장 핵심

| 개념 | 코드 |
|------|------|
| 저장 | joblib.dump(model, 'file.pkl') |
| 불러오기 | joblib.load('file.pkl') |
| 전처리기 | scaler도 함께 저장 |
| 파이프라인 | Pipeline으로 묶어서 관리 |

---

# - 실습편 -

## 25차시

**모델 저장과 배포 실습**

---

# 실습 개요

## 품질 예측 모델 저장

### 목표
- joblib으로 모델 저장/불러오기
- 전처리기 함께 관리
- 메타데이터 저장
- 배포 체크리스트 확인

### 실습 환경
```python
import joblib
from sklearn.pipeline import Pipeline
```

---

# 실습 1: 모델 학습

## 랜덤포레스트 모델

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"정확도: {model.score(X_test_scaled, y_test):.3f}")
```

---

# 실습 2: joblib 저장

## 모델과 전처리기 저장

```python
import joblib

# 모델 저장
joblib.dump(model, 'model.pkl')
print("모델 저장 완료!")

# 전처리기 저장
joblib.dump(scaler, 'scaler.pkl')
print("전처리기 저장 완료!")
```

```python
import os
print(f"model.pkl: {os.path.getsize('model.pkl')/1024:.1f} KB")
```

---

# 실습 3: 모델 불러오기

## 저장된 모델로 예측

```python
# 불러오기
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# 새 데이터 예측
new_data = np.array([[90, 55, 100, 1.0]])
new_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_scaled)

print(f"예측: {'불량' if prediction[0] == 1 else '정상'}")
```

---

# 실습 4: 파이프라인

## 하나로 묶어서 관리

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.pkl')

# 불러와서 바로 예측
loaded_pipeline = joblib.load('pipeline.pkl')
pred = loaded_pipeline.predict(new_data)  # 스케일링 자동!
```

---

# 실습 5: 메타데이터 저장

## 버전 정보 함께 관리

```python
from datetime import datetime

model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': ['temp', 'humidity', 'speed', 'pressure'],
    'version': '1.0.0',
    'trained_date': datetime.now().isoformat(),
    'accuracy': 0.92
}

joblib.dump(model_package, 'model_package.pkl')
```

```python
info = joblib.load('model_package.pkl')
print(f"버전: {info['version']}, 정확도: {info['accuracy']}")
```

---

# 실습 6: 버전 관리

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

# 실습 7: 배포 체크리스트

## 배포 전 확인사항

### 1. 모델 검증
- [ ] 테스트 데이터 성능 확인
- [ ] 다양한 입력에 대한 예측 테스트
- [ ] 에지 케이스 처리 확인

### 2. 파일 확인
- [ ] 모델 파일 존재
- [ ] 전처리기 파일 존재
- [ ] 버전 정보 기록

---

# 실습 8: requirements.txt

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

# 실습 정리

## 핵심 체크포인트

- [ ] joblib.dump()로 모델 저장
- [ ] joblib.load()로 모델 불러오기
- [ ] 전처리기도 함께 저장
- [ ] Pipeline으로 한번에 관리
- [ ] 메타데이터 포함 저장
- [ ] requirements.txt 작성

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

# 다음 차시 예고

## 26차시: AI 프로젝트 종합 실습

### 학습 내용
- 전체 ML 워크플로우 종합 실습
- 데이터 수집 → 모델 배포 프로젝트
- 과정 총정리 및 후속 학습 안내

> 25차시 동안 배운 모든 내용을 **종합 실습**!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **joblib.dump()**: 모델 저장
2. **joblib.load()**: 모델 불러오기
3. **전처리기 저장**: scaler도 함께
4. **Pipeline**: 하나로 묶어서 관리
5. **메타데이터**: 버전, 날짜, 성능 기록
6. **requirements.txt**: 환경 고정

---

# 감사합니다

## 25차시: 모델 저장과 실무 배포 준비

**학습한 모델을 저장하고 재사용하는 방법을 배웠습니다!**
