# [24차시] 모델 저장과 실무 배포 준비

## 학습 목표

| 번호 | 목표 |
|:----:|------|
| 1 | joblib으로 모델을 저장하고 로드함 |
| 2 | Pipeline으로 전처리와 모델을 통합함 |
| 3 | 실무 배포 체크리스트를 작성함 |

---

## 실습 데이터: Breast Cancer 데이터셋

sklearn에서 제공하는 유방암 진단 데이터셋을 활용함

### 데이터 특성

| 항목 | 내용 |
|------|------|
| 샘플 수 | 569개 |
| 특성 수 | 30개 |
| 클래스 | 악성(malignant), 양성(benign) |
| 출처 | UCI ML Repository |

실제 의료 데이터를 기반으로 구성되어 분류 모델 학습에 적합함

---

## Part 1: joblib으로 모델 저장과 로드

### 1.1 왜 모델을 저장해야 하는가?

학습에는 시간이 오래 걸리지만, 예측은 빠름

```
학습: 1시간 (데이터 로드 -> 학습 -> 튜닝)
예측: 1초
```

모델 저장의 이점:
- 매번 학습하는 시간 낭비를 방지함
- 학습된 모델을 파일로 저장함
- 필요할 때 로드해서 바로 사용 가능함

---

### 1.2 모델 저장 방법 비교

| 방법 | 장점 | 단점 |
|-----|------|------|
| **joblib** | 대용량 배열 효율적, sklearn 권장 | Python 전용 |
| **pickle** | Python 표준 | 대용량 비효율 |
| **ONNX** | 언어 중립, 배포 최적화 | 변환 필요 |

sklearn 모델은 joblib 사용을 권장함

---

### 1.3 실습 환경 설정

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# sklearn imports
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[24차시] 모델 저장과 실무 배포 준비")
print("=" * 60)
```

---

### 1.4 데이터 로드

```python
# sklearn의 유방암 진단 데이터셋 로드
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print("[Breast Cancer 데이터셋 개요]")
print(f"  샘플 수: {len(X)}")
print(f"  특성 수: {X.shape[1]}")
print(f"  클래스: {list(cancer.target_names)} (0=악성, 1=양성)")
print(f"  클래스 분포: 악성={np.sum(y==0)}, 양성={np.sum(y==1)}")
```

---

```python
print("\n[주요 특성 (처음 10개)]")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  {i+1:2d}. {name}")

print("\n[데이터 미리보기]")
print(X.head())

# 특성 목록
feature_cols = list(cancer.feature_names)
```

#### 데이터 해설

| 특성 종류 | 예시 |
|----------|------|
| mean | mean radius, mean texture |
| se (표준오차) | radius error, texture error |
| worst (최악값) | worst radius, worst texture |

총 10개 기본 특성 x 3개 통계량 = 30개 특성

---

### 1.5 데이터 분할

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")
print(f"양성 비율 (학습): {y_train.mean():.1%}")
print(f"양성 비율 (테스트): {y_test.mean():.1%}")
```

stratify=y 옵션으로 클래스 비율을 유지하여 분할함

---

### 1.6 기본 모델 학습 (Pipeline 없이)

```python
# 전처리와 모델을 따로 관리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"학습 정확도: {train_acc:.2%}")
print(f"테스트 정확도: {test_acc:.2%}")
```

문제점: scaler와 model을 따로 저장하고 관리해야 함

---

### 1.7 모델 저장하기 (joblib.dump)

```python
# 저장 디렉토리 생성
MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 저장
model_path = os.path.join(MODEL_DIR, 'cancer_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'cancer_scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"모델 저장: {model_path}")
print(f"스케일러 저장: {scaler_path}")

# 파일 크기 확인
model_size = os.path.getsize(model_path)
scaler_size = os.path.getsize(scaler_path)
print(f"모델 크기: {model_size/1024:.1f} KB")
print(f"스케일러 크기: {scaler_size/1024:.1f} KB")
```

---

### 1.8 압축 저장

```python
# 압축 저장
compressed_path = os.path.join(MODEL_DIR, 'cancer_model_compressed.pkl.gz')
joblib.dump(model, compressed_path, compress=3)

compressed_size = os.path.getsize(compressed_path)
print(f"압축 전: {model_size/1024:.1f} KB")
print(f"압축 후: {compressed_size/1024:.1f} KB")
print(f"압축률: {(1 - compressed_size/model_size)*100:.1f}%")
```

#### 압축 옵션

| 레벨 | 특징 |
|------|------|
| 0 | 압축 없음 (빠름, 용량 큼) |
| 3 | 일반적인 선택 |
| 9 | 최대 압축 (느림, 용량 작음) |

대용량 모델은 압축 저장을 권장함

---

### 1.9 모델 로드하기 (joblib.load)

```python
# 모델 로드
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# 로드 후 예측 검증
X_test_scaled_loaded = loaded_scaler.transform(X_test)
loaded_acc = loaded_model.score(X_test_scaled_loaded, y_test)

print(f"로드 후 정확도: {loaded_acc:.2%}")
print(f"원본과 동일: {loaded_acc == test_acc}")
```

학습 없이 바로 예측 가능함

---

### 1.10 저장 주의사항

#### 버전 호환성

```python
import sklearn

metadata = {
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9',
    'created_at': '2026-01-09'
}
joblib.dump({'model': model, 'metadata': metadata}, 'model_bundle.pkl')
```

저장 시 버전을 함께 기록해야 함

#### 보안 주의

```python
# 신뢰할 수 없는 파일 로드 금지!
# pickle/joblib은 임의 코드 실행이 가능함

# 안전한 방법: 출처가 확인된 파일만 로드
model = joblib.load('trusted_model.pkl')
```

---

## Part 2: Pipeline 구성과 활용

### 2.1 왜 Pipeline이 필요한가?

문제 상황:
```python
# 학습 시
scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 배포 시
X_new_scaled = scaler.transform(X_new)  # scaler도 저장해야 함!
prediction = model.predict(X_new_scaled)
```

전처리 객체를 따로 관리해야 하는 번거로움이 발생함

---

### 2.2 Pipeline이란?

전처리 + 모델을 하나로 묶은 객체임

```
[원본 데이터] -> [전처리] -> [모델] -> [예측]
               +-------- Pipeline --------+
```

Pipeline 장점:
- 한 번에 저장/로드 가능
- 데이터 누수 방지
- 코드 간결화

---

### 2.3 Pipeline 기본 구조

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # 1단계: 스케일링
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # 2단계: 모델
])

# 학습 (자동으로 순서대로 처리)
pipeline.fit(X_train, y_train)

# 예측 (자동으로 스케일링 후 예측)
train_acc_pipe = pipeline.score(X_train, y_train)
test_acc_pipe = pipeline.score(X_test, y_test)

print(f"학습 정확도: {train_acc_pipe:.2%}")
print(f"테스트 정확도: {test_acc_pipe:.2%}")
```

---

### 2.4 Pipeline 단계 접근

```python
print(f"단계 이름: {list(pipeline.named_steps.keys())}")
print(f"스케일러: {pipeline.named_steps['scaler']}")
print(f"모델: {pipeline.named_steps['model']}")
```

named_steps 속성으로 각 단계에 접근 가능함

---

### 2.5 Pipeline 저장/로드

```python
# 저장
pipeline_path = os.path.join(MODEL_DIR, 'cancer_pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"Pipeline 저장: {pipeline_path}")

pipeline_size = os.path.getsize(pipeline_path)
print(f"파일 크기: {pipeline_size/1024:.1f} KB")

# 로드
loaded_pipeline = joblib.load(pipeline_path)

# 검증
loaded_pipe_acc = loaded_pipeline.score(X_test, y_test)
print(f"로드 후 정확도: {loaded_pipe_acc:.2%}")
```

전처리 객체를 따로 관리할 필요가 없음

---

### 2.6 새 데이터 예측

```python
# 새 데이터 예측
print("새 데이터 예측:")
new_data = X_test.iloc[:3].copy()

predictions = loaded_pipeline.predict(new_data)
probabilities = loaded_pipeline.predict_proba(new_data)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    diagnosis = cancer.target_names[pred]
    print(f"  샘플 {i+1}: {diagnosis} (양성 확률: {prob[1]:.1%})")
```

predict()만 호출하면 전처리가 자동으로 적용됨

---

### 2.7 make_pipeline 간편 함수

```python
from sklearn.pipeline import make_pipeline

# make_pipeline은 이름을 자동 생성
simple_pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

simple_pipeline.fit(X_train, y_train)
print(f"자동 생성된 단계 이름: {list(simple_pipeline.named_steps.keys())}")
print(f"정확도: {simple_pipeline.score(X_test, y_test):.2%}")
```

자동 이름: standardscaler, randomforestclassifier

---

### 2.8 다단계 Pipeline

```python
from sklearn.impute import SimpleImputer

# 결측치 처리 + 정규화 + 모델
multi_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler()),                    # 정규화
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

multi_pipeline.fit(X_train, y_train)
print(f"다단계 Pipeline 정확도: {multi_pipeline.score(X_test, y_test):.2%}")
print(f"단계: {list(multi_pipeline.named_steps.keys())}")
```

---

### 2.9 메타데이터와 함께 저장

```python
import sklearn

# 메타데이터 생성
metadata = {
    'model_name': 'Breast Cancer Diagnosis Model',
    'version': 'v1.0.0',
    'created_at': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'features': feature_cols,
    'target_names': list(cancer.target_names),
    'train_accuracy': float(train_acc_pipe),
    'test_accuracy': float(test_acc_pipe),
    'dataset': 'sklearn.datasets.load_breast_cancer'
}

# 모델 번들 저장
model_bundle = {
    'pipeline': pipeline,
    'metadata': metadata
}

bundle_path = os.path.join(MODEL_DIR, 'cancer_model_bundle.pkl')
joblib.dump(model_bundle, bundle_path)
print(f"모델 번들 저장: {bundle_path}")
```

---

```python
# 메타데이터 JSON 저장 (별도)
metadata_path = os.path.join(MODEL_DIR, 'cancer_model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"메타데이터 저장: {metadata_path}")

print("\n메타데이터 내용 (주요 항목):")
for key in ['model_name', 'version', 'test_accuracy', 'dataset']:
    print(f"  {key}: {metadata[key]}")
```

메타데이터를 함께 저장하면 모델 이력 관리가 용이함

---

### 2.10 Pipeline 장점 정리

| 장점 | 설명 |
|-----|------|
| **일관성** | 학습/예측 시 동일한 전처리 보장 |
| **데이터 누수 방지** | fit은 학습 데이터에만 적용 |
| **저장 간편** | 하나의 객체로 저장/로드 |
| **GridSearchCV 통합** | 전처리 파라미터도 튜닝 가능 |
| **코드 가독성** | 워크플로우가 명확함 |

---

## Part 3: 배포 체크리스트

### 3.1 모델 배포란?

학습된 모델을 실제 환경에서 사용 가능하게 만드는 과정임

```
[개발 환경]              [운영 환경]
  학습 완료    ->  배포  ->   실시간 예측
  모델 저장              API 서비스
```

---

### 3.2 배포 전 체크리스트 개요

| 영역 | 항목 |
|-----|------|
| **모델** | 성능, 저장, 버전 |
| **데이터** | 입력 형식, 전처리 |
| **환경** | 의존성, 리소스 |
| **테스트** | 단위/통합 테스트 |
| **문서** | 사용법, 제약사항 |

---

### 3.3 입력 검증 함수 구현

```python
def validate_input(data, required_features):
    """입력 데이터 검증"""
    # DataFrame 또는 dict 처리
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)

    # 필수 피처 확인
    missing = set(required_features) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # 값 범위 확인
    warnings = []
    for col in data.columns:
        if (data[col] < 0).any():
            warnings.append(f"{col}: 음수 값 존재")

    return True, warnings
```

---

```python
# 검증 테스트
print("검증 테스트:")

# 정상 입력
valid_data = X_test.iloc[0].to_dict()
is_valid, warnings = validate_input(valid_data, feature_cols)
print(f"  정상 입력: 검증 통과={is_valid}, 경고 수={len(warnings)}")
```

입력 검증으로 잘못된 데이터로 인한 오류를 방지함

---

### 3.4 예측 함수 (검증 포함)

```python
def predict_diagnosis(data, model_path='./models/cancer_pipeline.pkl'):
    """
    유방암 진단 예측 함수

    Parameters:
    -----------
    data : dict or DataFrame
        입력 데이터
    model_path : str
        모델 파일 경로

    Returns:
    --------
    dict : 예측 결과
    """
    # 1. 데이터 변환
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)

    # 2. 모델 로드
    model = joblib.load(model_path)

    # 3. 예측
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    # 4. 결과 반환
    results = []
    for i, (pred, prob) in enumerate(zip(prediction, probability)):
        results.append({
            'prediction': cancer.target_names[pred],
            'malignant_probability': float(prob[0]),
            'benign_probability': float(prob[1]),
            'confidence': float(max(prob))
        })

    return results if len(results) > 1 else results[0]
```

---

```python
# 예측 테스트
print("예측 테스트:")
test_input = X_test.iloc[0].to_dict()
result = predict_diagnosis(test_input)
print(f"  예측: {result['prediction']}")
print(f"  양성 확률: {result['benign_probability']:.1%}")
print(f"  신뢰도: {result['confidence']:.1%}")
```

---

### 3.5 배포 체크리스트 생성

```python
checklist = """
## 유방암 진단 모델 배포 체크리스트

### 1. 모델 관련
- [x] 최종 성능 지표 기록 (정확도: {test_acc:.2%})
- [x] 학습/검증/테스트 결과 문서화
- [x] 모델 파일 저장 (cancer_pipeline.pkl)
- [x] 모델 버전 관리 (v1.0.0)
- [x] 메타데이터 저장

### 2. 데이터 관련
- [x] 입력 데이터 형식 정의
- [x] 필수 피처 수: {n_features}개
- [x] 데이터셋: sklearn.datasets.load_breast_cancer
- [x] 입력값 검증 구현

### 3. 환경 관련
- [x] Python 버전: 3.9
- [x] sklearn 버전: {sklearn_ver}
- [x] requirements.txt 생성
- [ ] 메모리 요구사항 측정
- [ ] 예측 소요 시간 측정

### 4. 테스트 관련
- [x] 모델 로드 테스트
- [x] 예측 기능 테스트
- [x] 입력 검증 테스트
- [ ] 경계값 테스트
- [ ] 부하 테스트

### 5. 문서 관련
- [x] 모델 카드 작성
- [ ] API 명세서
- [x] 입력/출력 예시
- [ ] 모니터링 가이드
"""
```

---

### 3.6 requirements.txt 생성

```python
requirements = """# Breast Cancer Diagnosis Model Dependencies
# Generated: {date}
# Dataset: sklearn.datasets.load_breast_cancer

scikit-learn>={sklearn_ver}
numpy>=1.21.0
pandas>=1.3.0
joblib>=1.1.0
matplotlib>=3.4.0
""".format(
    date=datetime.now().strftime('%Y-%m-%d'),
    sklearn_ver=sklearn.__version__
)

requirements_path = os.path.join(MODEL_DIR, 'requirements.txt')
with open(requirements_path, 'w') as f:
    f.write(requirements)

print(f"requirements.txt 저장: {requirements_path}")
print("내용:")
print(requirements)
```

---

### 3.7 예측 성능 측정

```python
import time

# 단일 예측 시간 측정
n_trials = 100
single_times = []

for _ in range(n_trials):
    single_input = X_test.iloc[[0]]
    start = time.time()
    _ = loaded_pipeline.predict(single_input)
    single_times.append(time.time() - start)

print(f"단일 예측 성능 (n={n_trials}):")
print(f"  평균: {np.mean(single_times)*1000:.2f} ms")
print(f"  최소: {np.min(single_times)*1000:.2f} ms")
print(f"  최대: {np.max(single_times)*1000:.2f} ms")

# 배치 예측 시간 측정
batch_sizes = [10, 50, 100]
print("\n배치 예측 성능:")

for batch_size in batch_sizes:
    batch_input = X_test.iloc[:batch_size]
    start = time.time()
    _ = loaded_pipeline.predict(batch_input)
    elapsed = time.time() - start
    print(f"  배치 {batch_size}: {elapsed*1000:.2f} ms ({elapsed*1000/batch_size:.2f} ms/sample)")
```

---

### 3.8 배포 방식 선택

| 방식 | 적합한 상황 |
|-----|------------|
| **REST API** | 실시간 예측, 웹 서비스 |
| **배치 처리** | 대량 데이터, 정기 예측 |
| **엣지 배포** | 현장 장비, 오프라인 |
| **클라우드** | 확장성, 관리 편의 |

27-28차시에서 FastAPI, Streamlit을 활용한 배포 방법을 상세히 다룸

---

### 3.9 모델 드리프트 감지

드리프트: 시간이 지남에 따라 모델 성능이 저하되는 현상임

```python
def check_drift(model, X_new, y_actual):
    """드리프트 감지"""
    current_acc = model.score(X_new, y_actual)
    baseline_acc = 0.96  # 배포 시점 성능

    if current_acc < baseline_acc - 0.05:
        print("성능 저하 감지! 재학습 고려")
        return True
    return False
```

주기적으로 성능을 모니터링하여 드리프트를 감지해야 함

---

### 3.10 특성 중요도 시각화

```python
# Pipeline에서 모델 추출
rf_model = pipeline.named_steps['model']
importances = rf_model.feature_importances_

# 정렬
fi_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("[Feature Importance Top 10]")
for _, row in fi_df.head(10).iterrows():
    bar = '*' * int(row['Importance'] * 50)
    print(f"  {row['Feature'][:25]:25s}: {row['Importance']:.3f} {bar}")
```

---

```python
# 시각화
fig, ax = plt.subplots(figsize=(12, 10))
top15 = fi_df.head(15)
ax.barh(top15['Feature'], top15['Importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Breast Cancer Diagnosis Model')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
```

특성 중요도를 분석하여 모델의 예측 근거를 이해함

---

## 24차시 핵심 정리

### joblib 저장/로드

| 함수 | 설명 |
|------|------|
| joblib.dump(obj, path) | 모델 저장 |
| joblib.load(path) | 모델 로드 |
| compress=3 | 압축 옵션 |

### Pipeline 구성

```python
# Pipeline 구성 및 저장
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.pkl')

# 로드 후 예측
loaded = joblib.load('pipeline.pkl')
prediction = loaded.predict(X_new)
```

### 배포 체크리스트

| 영역 | 핵심 항목 |
|------|----------|
| 모델 | 성능 기록, 버전 관리, 메타데이터 |
| 데이터 | 입력 형식 정의, 검증 로직 |
| 환경 | requirements.txt, 리소스 측정 |
| 테스트 | 단위/통합/부하 테스트 |
| 문서 | 사용법, 제약사항, 모니터링 |

---

## 저장된 파일 목록

```
models/
├── cancer_model.pkl              # 모델만
├── cancer_scaler.pkl             # 스케일러만
├── cancer_model_compressed.pkl.gz # 압축 모델
├── cancer_pipeline.pkl           # Pipeline (권장)
├── cancer_model_bundle.pkl       # 메타데이터 포함
├── cancer_model_metadata.json    # 메타데이터 JSON
├── deployment_checklist.md       # 배포 체크리스트
├── requirements.txt              # 의존성 목록
└── feature_importance.png        # 특성 중요도 시각화
```

---

## 다음 차시 예고

### 25차시: AI API의 이해와 활용

학습 내용:
- REST API 개념과 구조
- requests 라이브러리 사용법
- JSON 데이터 처리
- 외부 API 호출 실습

저장한 모델을 API로 서비스하는 방법을 학습함
