---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [24차시] 모델 저장과 실무 배포 준비

## 학습한 모델을 현장에서 사용하기

---

# 학습 목표

1. **joblib**으로 모델을 저장하고 로드한다
2. **Pipeline**으로 전처리와 모델을 통합한다
3. 실무 **배포 체크리스트**를 작성한다

---

# 지난 시간 복습

- **Feature Importance**: 트리 모델 내장 중요도
- **Permutation Importance**: 성능 하락 기반 중요도
- **모델 해석**: 신뢰, 디버깅, 개선, 규제 대응

**오늘**: 해석이 끝난 모델을 저장하고 배포 준비

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | joblib으로 모델 저장/로드 |
| 대주제 2 | 12분 | Pipeline 구성과 활용 |
| 대주제 3 | 6분 | 배포 체크리스트 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## joblib으로 모델 저장과 로드

---

# 왜 모델을 저장해야 하는가?

**학습의 문제점**:
```
학습: 1시간 (데이터 로드 → 학습 → 튜닝)
예측: 1초
```

- 매번 학습하면 시간 낭비
- 학습된 모델을 파일로 저장
- 필요할 때 로드해서 바로 사용

---

# 모델 저장 방법 비교

| 방법 | 장점 | 단점 |
|-----|------|------|
| **joblib** | 대용량 배열 효율적, sklearn 권장 | Python 전용 |
| **pickle** | Python 표준 | 대용량 비효율 |
| **ONNX** | 언어 중립, 배포 최적화 | 변환 필요 |

**sklearn 모델 → joblib 권장**

---

# joblib 설치 확인

```python
# scikit-learn 설치 시 함께 설치됨
import joblib

# 버전 확인
print(f"joblib 버전: {joblib.__version__}")
```

별도 설치 필요 시:
```bash
pip install joblib
```

---

# 모델 저장하기

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 2. 모델 저장
joblib.dump(model, 'quality_model.pkl')
print("모델 저장 완료!")
```

**`.pkl` 또는 `.joblib` 확장자 사용**

---

# 모델 로드하기

```python
import joblib

# 모델 로드
loaded_model = joblib.load('quality_model.pkl')

# 바로 예측 사용
prediction = loaded_model.predict(X_new)
print(f"예측 결과: {prediction}")
```

**학습 없이 바로 예측 가능**

---

# 저장 시 압축 옵션

```python
# 압축 없이 저장 (빠름, 용량 큼)
joblib.dump(model, 'model.pkl')

# 압축해서 저장 (느림, 용량 작음)
joblib.dump(model, 'model.pkl.gz', compress=3)

# 압축 레벨: 0(없음) ~ 9(최대)
joblib.dump(model, 'model.pkl.gz', compress=('gzip', 6))
```

**대용량 모델은 압축 권장**

---

# 파일 크기 비교

```python
import os

# 저장
joblib.dump(model, 'model.pkl')
joblib.dump(model, 'model_compressed.pkl.gz', compress=3)

# 크기 확인
size1 = os.path.getsize('model.pkl')
size2 = os.path.getsize('model_compressed.pkl.gz')

print(f"압축 전: {size1/1024:.1f} KB")
print(f"압축 후: {size2/1024:.1f} KB")
print(f"압축률: {(1-size2/size1)*100:.1f}%")
```

---

# 저장 주의사항

**1. 버전 호환성**:
```python
# 저장 시 버전 기록
import sklearn
metadata = {
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9',
    'created_at': '2026-01-08'
}
joblib.dump({'model': model, 'metadata': metadata}, 'model_bundle.pkl')
```

---

# 저장 주의사항 (계속)

**2. 보안 주의**:
```python
# 신뢰할 수 없는 파일 로드 금지!
# pickle/joblib은 임의 코드 실행 가능

# 안전한 방법: 출처 확인된 파일만 로드
model = joblib.load('trusted_model.pkl')
```

**3. 경로 관리**:
```python
import os
MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, 'model_v1.pkl'))
```

---

# 실습: 품질 예측 모델 저장/로드

```python
# 1. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"학습 정확도: {model.score(X_train, y_train):.2%}")

# 2. 저장
joblib.dump(model, 'quality_model.pkl')

# 3. 로드
loaded = joblib.load('quality_model.pkl')

# 4. 검증
print(f"로드 후 정확도: {loaded.score(X_test, y_test):.2%}")
```

---

<!-- _class: lead -->
# 대주제 2
## Pipeline 구성과 활용

---

# 왜 Pipeline이 필요한가?

**문제 상황**:
```python
# 학습 시
scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 배포 시
X_new_scaled = scaler.transform(X_new)  # scaler도 저장해야 함!
prediction = model.predict(X_new_scaled)
```

**전처리 객체를 따로 관리해야 하는 번거로움**

---

# Pipeline이란?

**정의**: 전처리 + 모델을 하나로 묶은 객체

```
[원본 데이터] → [전처리] → [모델] → [예측]
               └─────── Pipeline ──────┘
```

**장점**:
- 한 번에 저장/로드
- 데이터 누수 방지
- 코드 간결화

---

# Pipeline 기본 구조

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # 1단계: 스케일링
    ('model', RandomForestClassifier()) # 2단계: 모델
])

# 학습 (자동으로 순서대로 처리)
pipeline.fit(X_train, y_train)
```

---

# Pipeline 예측

```python
# 예측 (전처리 + 모델 한 번에)
prediction = pipeline.predict(X_new)

# 내부적으로:
# 1. scaler.transform(X_new)
# 2. model.predict(X_scaled)
```

**단일 객체로 전체 워크플로우 관리**

---

# Pipeline 단계 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),       # 정규화
    ('pca', PCA(n_components=5)),       # 차원 축소
    ('model', RandomForestClassifier()) # 분류
])

# 3단계 파이프라인
```

---

# Pipeline 저장/로드

```python
import joblib

# 파이프라인 저장 (전처리 + 모델 통째로)
joblib.dump(pipeline, 'quality_pipeline.pkl')

# 파이프라인 로드
loaded_pipeline = joblib.load('quality_pipeline.pkl')

# 바로 예측 (전처리 자동 적용)
prediction = loaded_pipeline.predict(X_new)
```

**전처리 객체 따로 관리 불필요!**

---

# make_pipeline 간편 함수

```python
from sklearn.pipeline import make_pipeline

# 이름 자동 생성
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=5),
    RandomForestClassifier()
)

# 자동 이름: standardscaler, pca, randomforestclassifier
print(pipeline.named_steps)
```

---

# Pipeline에서 하이퍼파라미터 접근

```python
# 이름__파라미터 형식
pipeline.set_params(model__n_estimators=200)

# GridSearchCV와 함께 사용
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

# ColumnTransformer: 열별 전처리

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 열 종류별 다른 전처리
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['temperature', 'pressure', 'speed']),
    ('cat', OneHotEncoder(), ['machine_type', 'shift'])
])

# Pipeline에 통합
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier())
])
```

---

# 실무 Pipeline 예제

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 숫자형 전처리
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler())                     # 정규화
])

# 전체 Pipeline
full_pipeline = Pipeline([
    ('preprocessor', num_pipeline),
    ('model', RandomForestClassifier(n_estimators=100))
])
```

---

# Pipeline 시각화

```python
from sklearn import set_config

# 다이어그램 표시 설정
set_config(display='diagram')

# Jupyter에서 Pipeline 시각화
pipeline  # 실행하면 시각적 다이어그램 표시
```

```
Pipeline
├── StandardScaler
└── RandomForestClassifier
```

---

# Pipeline 장점 정리

| 장점 | 설명 |
|-----|------|
| **일관성** | 학습/예측 시 동일한 전처리 보장 |
| **데이터 누수 방지** | fit은 학습 데이터에만 적용 |
| **저장 간편** | 하나의 객체로 저장/로드 |
| **GridSearchCV 통합** | 전처리 파라미터도 튜닝 가능 |
| **코드 가독성** | 워크플로우가 명확 |

---

# 실습: 품질 예측 Pipeline 구축

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 학습
pipeline.fit(X_train, y_train)
print(f"정확도: {pipeline.score(X_test, y_test):.2%}")

# 저장
joblib.dump(pipeline, 'quality_pipeline.pkl')
```

---

<!-- _class: lead -->
# 대주제 3
## 배포 체크리스트

---

# 모델 배포란?

**정의**: 학습된 모델을 실제 환경에서 사용 가능하게 만드는 과정

```
[개발 환경]              [운영 환경]
  학습 완료    →  배포  →   실시간 예측
  모델 저장              API 서비스
```

---

# 배포 전 체크리스트 개요

| 영역 | 항목 |
|-----|------|
| **모델** | 성능, 저장, 버전 |
| **데이터** | 입력 형식, 전처리 |
| **환경** | 의존성, 리소스 |
| **테스트** | 단위/통합 테스트 |
| **문서** | 사용법, 제약사항 |

---

# 1. 모델 체크리스트

```markdown
[ ] 최종 성능 지표 기록 (정확도, F1 등)
[ ] 학습/검증/테스트 결과 문서화
[ ] 모델 파일 저장 (pipeline.pkl)
[ ] 모델 버전 관리 (v1.0.0)
[ ] Feature Importance 분석 완료
```

---

# 2. 데이터 체크리스트

```markdown
[ ] 입력 데이터 형식 정의 (JSON 스키마)
[ ] 필수 피처 목록 확인
[ ] 결측치 처리 방법 문서화
[ ] 입력값 범위 검증 로직
[ ] 예시 입력/출력 준비
```

---

# 입력 데이터 검증 예시

```python
def validate_input(data):
    """입력 데이터 검증"""
    required_features = ['temperature', 'pressure', 'speed',
                        'humidity', 'vibration']

    # 필수 피처 확인
    for feature in required_features:
        if feature not in data:
            raise ValueError(f"Missing feature: {feature}")

    # 값 범위 확인
    if not (100 <= data['temperature'] <= 300):
        raise ValueError("Temperature out of range")

    return True
```

---

# 3. 환경 체크리스트

```markdown
[ ] requirements.txt 생성
[ ] Python 버전 명시
[ ] sklearn 버전 호환성 확인
[ ] 메모리 요구사항 측정
[ ] 예측 소요 시간 측정
```

---

# requirements.txt 생성

```python
# 현재 환경의 패키지 목록
pip freeze > requirements.txt

# 또는 필요한 것만 선택
# requirements.txt 내용:
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
```

---

# 4. 테스트 체크리스트

```markdown
[ ] 단위 테스트 (모델 로드, 예측)
[ ] 입력 검증 테스트
[ ] 경계값 테스트
[ ] 성능 테스트 (응답 시간)
[ ] 오류 처리 테스트
```

---

# 단위 테스트 예시

```python
import unittest
import joblib

class TestQualityModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load('quality_pipeline.pkl')

    def test_model_loads(self):
        """모델 로드 확인"""
        self.assertIsNotNone(self.model)

    def test_prediction_shape(self):
        """예측 결과 형태 확인"""
        X_sample = [[200, 50, 100, 60, 5]]
        pred = self.model.predict(X_sample)
        self.assertEqual(len(pred), 1)
```

---

# 5. 문서 체크리스트

```markdown
[ ] 모델 사용법 문서
[ ] API 명세서
[ ] 입력/출력 예시
[ ] 제약사항 및 한계
[ ] 모니터링 가이드
```

---

# 모델 카드 (Model Card) 작성

```markdown
## 품질 예측 모델 v1.0

### 개요
- 목적: 제조 공정 품질 불량 예측
- 모델: RandomForest Classifier
- 학습일: 2026-01-08

### 성능
- 정확도: 92.5%
- F1 Score: 0.89

### 입력
- temperature (float): 150-250
- pressure (float): 30-70
- speed (float): 80-120
```

---

# 배포 방식 선택

| 방식 | 적합한 상황 |
|-----|------------|
| **REST API** | 실시간 예측, 웹 서비스 |
| **배치 처리** | 대량 데이터, 정기 예측 |
| **엣지 배포** | 현장 장비, 오프라인 |
| **클라우드** | 확장성, 관리 편의 |

**26-27차시에서 상세히 다룸**

---

# 배포 후 모니터링

```python
# 예측 로깅 예시
import logging
from datetime import datetime

logging.basicConfig(filename='predictions.log')

def predict_with_logging(model, X, request_id):
    """로깅 포함 예측"""
    start = datetime.now()
    prediction = model.predict(X)
    elapsed = (datetime.now() - start).total_seconds()

    logging.info(f"[{request_id}] Input: {X.tolist()}, "
                 f"Prediction: {prediction.tolist()}, "
                 f"Time: {elapsed:.3f}s")

    return prediction
```

---

# 모델 드리프트 감지

**드리프트**: 시간이 지남에 따라 모델 성능이 저하되는 현상

```python
# 월별 성능 모니터링
def check_drift(model, X_new, y_actual):
    """드리프트 감지"""
    current_acc = model.score(X_new, y_actual)
    baseline_acc = 0.92  # 배포 시점 성능

    if current_acc < baseline_acc - 0.05:
        print("⚠️ 성능 저하 감지! 재학습 고려")
        return True
    return False
```

---

# 전체 배포 워크플로우

```
1. 모델 개발 완료
   ↓
2. Pipeline 구성
   ↓
3. 체크리스트 검토
   ↓
4. 테스트 통과
   ↓
5. 문서화 완료
   ↓
6. 배포 및 모니터링
```

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **joblib 모델 저장/로드**
   - `joblib.dump()`, `joblib.load()`
   - 압축 옵션, 버전 관리

2. **Pipeline 구성**
   - 전처리 + 모델 통합
   - 한 번에 저장, 데이터 누수 방지

3. **배포 체크리스트**
   - 모델, 데이터, 환경, 테스트, 문서

---

# 핵심 코드

```python
# 모델 저장/로드
import joblib
joblib.dump(model, 'model.pkl')
loaded = joblib.load('model.pkl')

# Pipeline 구성
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.pkl')
```

---

# 체크리스트

- [ ] joblib으로 모델 저장
- [ ] 저장된 모델 로드 테스트
- [ ] Pipeline 구성
- [ ] Pipeline 저장/로드
- [ ] 배포 체크리스트 작성
- [ ] 입력 검증 로직 구현

---

# 다음 차시 예고

## [24차시] AI API의 이해와 활용

- REST API 개념과 구조
- requests 라이브러리 사용법
- JSON 데이터 처리
- 외부 API 호출 실습

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습: Pipeline으로 품질 예측 모델 저장하기

