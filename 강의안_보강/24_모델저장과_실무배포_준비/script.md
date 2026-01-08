# [24차시] 모델 저장과 실무 배포 준비 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 24차시 |
| 주제 | 모델 저장과 실무 배포 준비 |
| 시간 | 30분 (이론 15분 + 실습 13분 + 정리 2분) |
| 학습 목표 | joblib 저장, Pipeline 구성, 배포 체크리스트 |

---

## 학습 목표

1. joblib으로 모델을 저장하고 로드한다
2. Pipeline으로 전처리와 모델을 통합한다
3. 실무 배포 체크리스트를 작성한다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | joblib 저장/로드 |
| 대주제 2 | 6분 | Pipeline 구성 |
| 대주제 3 | 4분 | 배포 체크리스트 |
| 실습 | 11분 | Pipeline 저장 실습 |
| 정리 | 2분 | 요약 및 다음 차시 예고 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 모델 해석을 배웠습니다. Feature Importance와 Permutation Importance로 어떤 변수가 중요한지 파악했죠."

> "모델 해석까지 끝났으면 이제 뭘 해야 할까요? 현장에서 사용할 수 있게 배포 준비를 해야 합니다."

> "오늘은 모델을 저장하고, Pipeline으로 묶고, 배포 전 체크리스트를 살펴봅니다."

---

### 대주제 1: joblib 저장/로드 (5분)

#### 슬라이드 4-6: 왜 저장하는가

> "모델 학습에 시간이 오래 걸리는데, 매번 학습할 수는 없죠. 한 번 학습한 모델을 파일로 저장해두면 필요할 때 바로 로드해서 쓸 수 있습니다."

> "저장 방법 중에서 sklearn 모델에는 joblib을 권장합니다. 대용량 numpy 배열을 효율적으로 처리하거든요."

---

#### 슬라이드 7-9: 저장과 로드

```python
import joblib

# 저장
joblib.dump(model, 'quality_model.pkl')

# 로드
loaded_model = joblib.load('quality_model.pkl')
prediction = loaded_model.predict(X_new)
```

> "dump로 저장하고, load로 불러옵니다. 확장자는 .pkl이나 .joblib을 씁니다."

> "로드한 모델은 바로 predict를 호출할 수 있어요. 학습 과정 없이 바로 예측합니다."

---

#### 슬라이드 10-12: 압축과 주의사항

> "모델 파일이 크면 압축 옵션을 쓸 수 있습니다. compress=3 정도면 적당합니다."

```python
joblib.dump(model, 'model.pkl.gz', compress=3)
```

> "주의할 점은 버전 호환성입니다. sklearn 버전이 다르면 로드가 안 될 수 있어요. 저장할 때 버전 정보도 같이 기록해두세요."

> "그리고 보안 주의. 출처를 모르는 pkl 파일은 로드하지 마세요. 악성 코드가 실행될 수 있습니다."

---

### 대주제 2: Pipeline 구성 (6분)

#### 슬라이드 13-15: 왜 Pipeline인가

> "모델만 저장하면 되는 게 아닙니다. 전처리도 저장해야 해요."

> "예를 들어 StandardScaler로 학습했으면, 예측할 때도 같은 scaler를 써야 합니다. 평균과 분산이 맞아야 하거든요."

```python
# 이렇게 따로 관리하면 번거로움
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 저장할 때 둘 다 저장해야 함
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')
```

---

#### 슬라이드 16-18: Pipeline 구성

> "Pipeline을 쓰면 전처리와 모델을 하나로 묶을 수 있습니다."

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# 학습
pipeline.fit(X_train, y_train)

# 예측 (자동으로 스케일링 후 예측)
pipeline.predict(X_new)
```

> "fit을 호출하면 scaler도 fit하고, model도 fit합니다. 순서대로요."

> "predict를 호출하면 scaler.transform 후 model.predict가 자동 실행됩니다."

---

#### 슬라이드 19-21: Pipeline 저장

> "Pipeline의 최대 장점은 저장이 간편하다는 겁니다."

```python
# Pipeline 통째로 저장
joblib.dump(pipeline, 'quality_pipeline.pkl')

# 로드 후 바로 예측
loaded_pipeline = joblib.load('quality_pipeline.pkl')
prediction = loaded_pipeline.predict(X_new)
```

> "하나의 파일로 전처리와 모델을 모두 관리할 수 있어요."

> "데이터 누수 방지에도 좋습니다. fit은 학습 데이터에만 적용되니까요."

---

#### 슬라이드 22-24: 고급 Pipeline

> "여러 단계를 추가할 수도 있습니다."

```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler()),                    # 정규화
    ('model', RandomForestClassifier())              # 모델
])
```

> "ColumnTransformer를 쓰면 열마다 다른 전처리를 할 수 있어요. 숫자형은 스케일링, 범주형은 인코딩 이런 식으로요."

---

### 대주제 3: 배포 체크리스트 (4분)

#### 슬라이드 25-27: 체크리스트 개요

> "모델 배포는 단순히 파일 저장만이 아닙니다. 실무에서 잘 동작하려면 확인할 게 많아요."

> "크게 다섯 영역으로 나눕니다: 모델, 데이터, 환경, 테스트, 문서."

---

#### 슬라이드 28-30: 모델/데이터 체크

> "모델 영역에서는 최종 성능을 기록하고, 버전을 관리합니다. v1.0.0 이런 식으로요."

> "데이터 영역에서는 입력 형식을 정의합니다. 어떤 피처가 필요하고, 각 값의 범위는 어떤지."

```python
def validate_input(data):
    if not (100 <= data['temperature'] <= 300):
        raise ValueError("온도 범위 초과")
```

> "입력 검증 로직을 미리 만들어두면 배포 후 오류를 줄일 수 있습니다."

---

#### 슬라이드 31-33: 환경/테스트 체크

> "환경 체크에서 가장 중요한 건 requirements.txt입니다. 의존성 패키지 목록을 만들어두세요."

```
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
```

> "테스트는 모델 로드, 예측, 오류 처리를 확인합니다. 단위 테스트를 작성해두면 안전합니다."

---

#### 슬라이드 34-36: 문서화

> "마지막으로 문서화입니다. 모델 카드라고 하는데, 모델의 목적, 성능, 입력/출력, 제약사항을 정리합니다."

> "문서가 없으면 다른 사람이 사용하기 어렵고, 나중에 본인도 헷갈립니다."

---

### 실습편 (11분)

#### 슬라이드 37-39: 데이터 준비

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 제조 데이터 생성
np.random.seed(42)
n_samples = 1000

data = {
    'temperature': np.random.normal(200, 30, n_samples),
    'pressure': np.random.normal(50, 15, n_samples),
    'speed': np.random.normal(100, 20, n_samples),
    'humidity': np.random.normal(60, 10, n_samples),
    'vibration': np.random.normal(5, 2, n_samples)
}

df = pd.DataFrame(data)
```

---

#### 슬라이드 40-42: Pipeline 구성 및 학습

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 학습
pipeline.fit(X_train, y_train)
print(f"정확도: {pipeline.score(X_test, y_test):.2%}")
```

---

#### 슬라이드 43-45: 저장 및 로드

```python
import joblib

# 저장
joblib.dump(pipeline, 'quality_pipeline.pkl')

# 로드
loaded_pipeline = joblib.load('quality_pipeline.pkl')

# 검증
print(f"로드 후 정확도: {loaded_pipeline.score(X_test, y_test):.2%}")
```

---

#### 슬라이드 46-48: 새 데이터 예측

```python
# 새 데이터 (전처리 안 된 원본)
new_data = [[210, 55, 105, 58, 6]]  # temp, press, speed, humid, vib

# 예측 (자동으로 스케일링 후 예측)
prediction = loaded_pipeline.predict(new_data)
proba = loaded_pipeline.predict_proba(new_data)

print(f"예측 결과: {'불량' if prediction[0] == 1 else '정상'}")
print(f"불량 확률: {proba[0][1]:.1%}")
```

---

### 정리 (2분)

#### 슬라이드 49-50: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**joblib**으로 모델을 저장하고 로드합니다. dump와 load 두 함수만 기억하세요."

> "**Pipeline**으로 전처리와 모델을 통합합니다. 한 번에 저장하고, 한 번에 예측합니다."

> "**배포 체크리스트**는 모델, 데이터, 환경, 테스트, 문서 다섯 영역을 확인합니다."

---

#### 슬라이드 51-52: 다음 차시 예고

> "다음 시간에는 AI API를 배웁니다. REST API가 뭔지, requests 라이브러리로 API를 어떻게 호출하는지 알아봅니다."

> "오늘 수업 마무리합니다. 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: pickle과 joblib 차이가 뭔가요?

> "둘 다 Python 객체를 저장하는 방법입니다. joblib은 numpy 배열을 더 효율적으로 처리해서 sklearn 모델에 권장됩니다."

### Q2: Pipeline 없이 scaler와 model을 따로 저장해도 되나요?

> "됩니다. 하지만 관리가 번거롭고 실수하기 쉬워요. Pipeline을 쓰면 하나로 묶어서 관리하니 훨씬 편합니다."

### Q3: 모델 버전 관리는 어떻게 하나요?

> "파일 이름에 버전을 넣거나, DVC 같은 도구를 쓸 수 있어요. 간단하게는 model_v1.pkl, model_v2.pkl 이런 식으로요."

### Q4: 배포 후 모델 성능이 떨어지면 어떻게 하나요?

> "모델 드리프트라고 합니다. 정기적으로 성능을 모니터링하고, 일정 수준 이하로 떨어지면 재학습하세요."

---

## 참고 자료

### 공식 문서
- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [joblib 문서](https://joblib.readthedocs.io/)

### 관련 차시
- 22차시: 모델 해석과 변수별 영향력 분석
- 24차시: AI API의 이해와 활용

---

## 체크리스트

수업 전:
- [ ] joblib 설치 확인
- [ ] 예제 데이터 준비
- [ ] Pipeline 코드 테스트

수업 중:
- [ ] 저장/로드 시연
- [ ] Pipeline 장점 강조
- [ ] 체크리스트 중요성 설명

수업 후:
- [ ] 실습 코드 배포
- [ ] 배포 체크리스트 템플릿 공유

