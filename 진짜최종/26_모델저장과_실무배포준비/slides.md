---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [26차시] 모델 저장과 실무 배포 준비

## 학습한 모델을 현장에서 사용하기

---

# 학습 목표

1. **joblib**으로 모델을 저장하고 로드합니다
2. **Pipeline**으로 전처리와 모델을 통합합니다
3. **버전 관리**와 메타데이터를 체계적으로 관리합니다
4. 실무 **배포 체크리스트**를 작성합니다

---

# 지난 시간 복습

- **딥러닝 해석**: 은닉층 활성화 패턴 분석
- **학습 추적**: 손실 함수, 가중치 변화 모니터링
- **콜백 활용**: EarlyStopping, ModelCheckpoint

**오늘**: 해석과 추적이 완료된 모델을 저장하고 배포 준비

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | joblib으로 모델 저장/로드 |
| 대주제 2 | 12분 | Pipeline 구성과 활용 |
| 대주제 3 | 6분 | 버전 관리와 배포 체크리스트 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## joblib으로 모델 저장과 로드

---

# 왜 모델을 저장해야 하는가?

**학습의 문제점**:
```
학습: 1시간 (데이터 로드 -> 학습 -> 튜닝)
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

**sklearn 모델은 joblib 권장**

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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. 모델 저장
joblib.dump(model, 'quality_model.pkl')
print("모델 저장 완료")
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

# 저장 주의사항: 버전 호환성

```python
# 저장 시 버전 기록
import sklearn
metadata = {
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9',
    'created_at': '2026-01-15'
}
joblib.dump({'model': model, 'metadata': metadata}, 'model_bundle.pkl')
```

**버전 불일치 시 로드 실패 가능**

---

# 저장 주의사항: 보안

```python
# 신뢰할 수 없는 파일 로드 금지
# pickle/joblib은 임의 코드 실행 가능

# 안전한 방법: 출처 확인된 파일만 로드
model = joblib.load('trusted_model.pkl')
```

**경로 관리**:
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
X_new_scaled = scaler.transform(X_new)  # scaler도 저장해야 함
prediction = model.predict(X_new_scaled)
```

**전처리 객체를 따로 관리해야 하는 번거로움**

---

# Pipeline이란?

**정의**: 전처리 + 모델을 하나로 묶은 객체

```
[원본 데이터] -> [전처리] -> [모델] -> [예측]
                +-------- Pipeline --------+
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
    ('scaler', StandardScaler()),       # 1단계: 스케일링
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

# Pipeline 다단계 구성

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler()),                    # 정규화
    ('model', RandomForestClassifier())              # 분류
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

**전처리 객체 따로 관리 불필요**

---

# make_pipeline 간편 함수

```python
from sklearn.pipeline import make_pipeline

# 이름 자동 생성
pipeline = make_pipeline(
    StandardScaler(),
    SimpleImputer(strategy='median'),
    RandomForestClassifier()
)

# 자동 이름: standardscaler, simpleimputer, randomforestclassifier
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
|-- StandardScaler
+-- RandomForestClassifier
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
## 버전 관리와 배포 체크리스트

---

# 모델 버전 관리의 중요성

**버전 관리가 필요한 이유**:
- 모델 성능 변화 추적
- 문제 발생 시 롤백 가능
- 실험 재현성 보장
- 팀 협업 시 혼란 방지

---

# 시맨틱 버전 관리

**버전 형식: MAJOR.MINOR.PATCH**

| 버전 | 변경 시점 |
|------|----------|
| MAJOR (1.x.x) | 입출력 형식 변경, 호환성 깨짐 |
| MINOR (x.1.x) | 새 기능 추가, 하위 호환 유지 |
| PATCH (x.x.1) | 버그 수정, 성능 개선 |

```
v1.0.0 -> v1.0.1 (버그 수정)
v1.0.1 -> v1.1.0 (새 피처 추가)
v1.1.0 -> v2.0.0 (입력 형식 변경)
```

---

# 메타데이터 기록 항목

```python
import sklearn
from datetime import datetime

metadata = {
    'model_name': 'quality_predictor',
    'version': 'v1.0.0',
    'created_at': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9',
    'features': feature_cols,
    'target': 'quality',
    'train_accuracy': 0.95,
    'test_accuracy': 0.92,
    'dataset_hash': 'abc123...'
}
```

---

# 모델 번들 저장

```python
import joblib
import json

# 모델 + 메타데이터 번들
model_bundle = {
    'pipeline': pipeline,
    'metadata': metadata,
    'feature_names': feature_cols
}

# 번들 저장
joblib.dump(model_bundle, 'model_v1.0.0.pkl')

# 메타데이터 JSON 별도 저장 (가독성)
with open('model_v1.0.0_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

# 모델 디렉토리 구조

```
models/
|-- v1.0.0/
|   |-- model.pkl
|   |-- metadata.json
|   +-- requirements.txt
|-- v1.1.0/
|   |-- model.pkl
|   |-- metadata.json
|   +-- requirements.txt
+-- latest -> v1.1.0/  (심볼릭 링크)
```

---

# 배포 전 체크리스트 개요

| 영역 | 항목 |
|-----|------|
| **모델** | 성능, 저장, 버전 |
| **데이터** | 입력 형식, 전처리, 검증 |
| **환경** | 의존성, 리소스 |
| **테스트** | 단위/통합/부하 테스트 |
| **문서** | 사용법, 제약사항, 모니터링 |

---

# 1. 모델 체크리스트

```markdown
[ ] 최종 성능 지표 기록 (정확도, F1, AUC 등)
[ ] 학습/검증/테스트 결과 문서화
[ ] 모델 파일 저장 (pipeline.pkl)
[ ] 모델 버전 태깅 (v1.0.0)
[ ] Feature Importance 분석 완료
[ ] 과적합 여부 확인 (학습/테스트 성능 차이)
[ ] 기준선 모델 대비 성능 검증
```

---

# 2. 데이터 체크리스트

```markdown
[ ] 입력 데이터 스키마 정의 (JSON Schema)
[ ] 필수 피처 목록 및 데이터 타입 명시
[ ] 결측치 처리 방법 문서화
[ ] 이상치 처리 정책 수립
[ ] 입력값 범위 검증 로직 구현
[ ] 예시 입력/출력 준비
[ ] 데이터 전처리 파이프라인 포함 확인
```

---

# 입력 데이터 검증 예시

```python
def validate_input(data, required_features, value_ranges):
    """입력 데이터 검증"""
    errors = []

    # 필수 피처 확인
    for feature in required_features:
        if feature not in data:
            errors.append(f"필수 피처 누락: {feature}")

    # 값 범위 확인
    for feature, (min_val, max_val) in value_ranges.items():
        if feature in data:
            if not (min_val <= data[feature] <= max_val):
                errors.append(f"{feature}: 범위 초과 ({min_val}-{max_val})")

    if errors:
        raise ValueError(f"검증 실패: {errors}")
    return True
```

---

# 3. 환경 체크리스트

```markdown
[ ] requirements.txt 생성 (버전 고정)
[ ] Python 버전 명시
[ ] sklearn 버전 호환성 확인
[ ] 필요 메모리 용량 측정
[ ] 단일 예측 소요 시간 측정
[ ] 배치 예측 처리량 측정
[ ] GPU 필요 여부 확인
```

---

# requirements.txt 생성

```python
# 현재 환경의 정확한 버전 기록
# requirements.txt 내용:
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
```

```bash
# 전체 환경 내보내기
pip freeze > requirements_full.txt

# 핵심 패키지만 선별 기록 권장
```

---

# 4. 테스트 체크리스트

```markdown
[ ] 모델 로드 테스트 (파일 존재, 로드 성공)
[ ] 단일 예측 테스트 (정상 입력)
[ ] 배치 예측 테스트 (여러 샘플)
[ ] 입력 검증 테스트 (잘못된 입력 거부)
[ ] 경계값 테스트 (최소/최대값)
[ ] 결측치 입력 테스트
[ ] 성능 테스트 (응답 시간 기준 충족)
[ ] 부하 테스트 (동시 요청 처리)
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

# 테스트 실행

```python
    def test_prediction_values(self):
        """예측값 범위 확인"""
        X_sample = [[200, 50, 100, 60, 5]]
        pred = self.model.predict(X_sample)
        # 이진 분류인 경우
        self.assertIn(pred[0], [0, 1])

    def test_probability_sum(self):
        """확률 합계 확인"""
        X_sample = [[200, 50, 100, 60, 5]]
        proba = self.model.predict_proba(X_sample)
        self.assertAlmostEqual(sum(proba[0]), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
```

---

# 5. 문서 체크리스트

```markdown
[ ] 모델 카드 (Model Card) 작성
[ ] API 명세서 (입력/출력 형식)
[ ] 입력/출력 예시 코드
[ ] 제약사항 및 한계 명시
[ ] 에러 코드 및 처리 방법
[ ] 성능 모니터링 가이드
[ ] 재학습 기준 및 절차
```

---

# 모델 카드 (Model Card) 작성

```markdown
## 품질 예측 모델 v1.0.0

### 개요
- 목적: 제조 공정 품질 불량 예측
- 모델: RandomForest Classifier
- 학습일: 2026-01-15

### 성능
- 정확도: 92.5%
- F1 Score: 0.89
- AUC: 0.94

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

**27-28차시에서 Streamlit, FastAPI 활용법 학습**

---

# 배포 후 모니터링

```python
import logging
from datetime import datetime

logging.basicConfig(filename='predictions.log', level=logging.INFO)

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
def check_drift(model, X_new, y_actual, baseline_acc=0.92):
    """드리프트 감지"""
    current_acc = model.score(X_new, y_actual)

    drift_threshold = 0.05
    if current_acc < baseline_acc - drift_threshold:
        print(f"성능 저하 감지: {baseline_acc:.2%} -> {current_acc:.2%}")
        print("재학습을 고려하십시오")
        return True
    return False
```

---

# 전체 배포 워크플로우

```
1. 모델 개발 완료
   |
2. Pipeline 구성
   |
3. 버전 태깅 및 메타데이터 기록
   |
4. 체크리스트 검토
   |
5. 테스트 통과
   |
6. 문서화 완료
   |
7. 배포 및 모니터링
```

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **joblib 모델 저장/로드**
   - `joblib.dump()`, `joblib.load()`
   - 압축 옵션, 버전 호환성 주의

2. **Pipeline 구성**
   - 전처리 + 모델 통합
   - 한 번에 저장, 데이터 누수 방지

3. **버전 관리와 배포 체크리스트**
   - 시맨틱 버전 관리, 메타데이터 기록
   - 모델, 데이터, 환경, 테스트, 문서 점검

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
- [ ] 버전 태깅 및 메타데이터 기록
- [ ] 배포 체크리스트 작성
- [ ] 입력 검증 로직 구현

---

# 다음 차시 예고

## [27차시] Streamlit으로 웹앱 만들기

- Streamlit 기본 구조와 설치
- 사용자 입력 컴포넌트
- 데이터 시각화 연동
- 저장한 모델을 웹앱으로 서비스

---

<!-- _class: lead -->
# 수고하셨습니다

## 실습: Pipeline으로 품질 예측 모델 저장하기

