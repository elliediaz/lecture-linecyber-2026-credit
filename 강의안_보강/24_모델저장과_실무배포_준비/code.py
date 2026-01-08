"""
[24차시] 모델 저장과 실무 배포 준비 - 실습 코드

학습 목표:
1. joblib으로 모델을 저장하고 로드한다
2. Pipeline으로 전처리와 모델을 통합한다
3. 실무 배포 체크리스트를 작성한다

실습 환경: Python 3.8+, scikit-learn, joblib

데이터셋:
- sklearn.datasets.load_breast_cancer (유방암 진단 분류)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# sklearn imports
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# joblib for saving/loading
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[24차시] 모델 저장과 실무 배포 준비")
print("=" * 60)

# ============================================================
# 1. 실제 데이터셋 로드 (Breast Cancer)
# ============================================================
print("\n" + "=" * 60)
print("1. 실제 데이터셋 로드 (Breast Cancer)")
print("=" * 60)

# sklearn의 유방암 진단 데이터셋 로드
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print("\n[Breast Cancer 데이터셋 개요]")
print(f"  샘플 수: {len(X)}")
print(f"  특성 수: {X.shape[1]}")
print(f"  클래스: {list(cancer.target_names)} (0=악성, 1=양성)")
print(f"  클래스 분포: 악성={np.sum(y==0)}, 양성={np.sum(y==1)}")

print("\n[주요 특성 (처음 10개)]")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  {i+1:2d}. {name}")

print("\n[데이터 미리보기]")
print(X.head())

# 특성 이름 간소화 (feature_cols)
feature_cols = list(cancer.feature_names)

# ============================================================
# 2. 데이터 분할
# ============================================================
print("\n" + "=" * 60)
print("2. 데이터 분할")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")
print(f"양성 비율 (학습): {y_train.mean():.1%}")
print(f"양성 비율 (테스트): {y_test.mean():.1%}")

# ============================================================
# 3. 기본 모델 학습 (Pipeline 없이)
# ============================================================
print("\n" + "=" * 60)
print("3. 기본 모델 학습 (Pipeline 없이)")
print("=" * 60)

# 전처리와 모델을 따로 관리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"\n학습 정확도: {train_acc:.2%}")
print(f"테스트 정확도: {test_acc:.2%}")

print("\n[분류 리포트]")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

print("\n문제점: scaler와 model을 따로 저장/관리해야 함")

# ============================================================
# 4. joblib으로 모델 저장/로드
# ============================================================
print("\n" + "=" * 60)
print("4. joblib으로 모델 저장/로드")
print("=" * 60)

# 저장 디렉토리 생성
MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# 4.1 모델 저장
print("\n4.1 모델 저장:")
model_path = os.path.join(MODEL_DIR, 'cancer_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'cancer_scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"  모델 저장: {model_path}")
print(f"  스케일러 저장: {scaler_path}")

# 파일 크기 확인
model_size = os.path.getsize(model_path)
scaler_size = os.path.getsize(scaler_path)
print(f"  모델 크기: {model_size/1024:.1f} KB")
print(f"  스케일러 크기: {scaler_size/1024:.1f} KB")

# 4.2 압축 저장
print("\n4.2 압축 저장:")
compressed_path = os.path.join(MODEL_DIR, 'cancer_model_compressed.pkl.gz')
joblib.dump(model, compressed_path, compress=3)
compressed_size = os.path.getsize(compressed_path)
print(f"  압축 전: {model_size/1024:.1f} KB")
print(f"  압축 후: {compressed_size/1024:.1f} KB")
print(f"  압축률: {(1 - compressed_size/model_size)*100:.1f}%")

# 4.3 모델 로드
print("\n4.3 모델 로드:")
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# 로드 후 예측 검증
X_test_scaled_loaded = loaded_scaler.transform(X_test)
loaded_acc = loaded_model.score(X_test_scaled_loaded, y_test)
print(f"  로드 후 정확도: {loaded_acc:.2%}")
print(f"  원본과 동일: {loaded_acc == test_acc}")

# ============================================================
# 5. Pipeline 구성
# ============================================================
print("\n" + "=" * 60)
print("5. Pipeline 구성")
print("=" * 60)

# 5.1 기본 Pipeline
print("\n5.1 기본 Pipeline:")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 학습
pipeline.fit(X_train, y_train)

# 예측 (자동으로 스케일링 후 예측)
train_acc_pipe = pipeline.score(X_train, y_train)
test_acc_pipe = pipeline.score(X_test, y_test)

print(f"  학습 정확도: {train_acc_pipe:.2%}")
print(f"  테스트 정확도: {test_acc_pipe:.2%}")

# 5.2 Pipeline 단계 접근
print("\n5.2 Pipeline 단계 접근:")
print(f"  단계 이름: {list(pipeline.named_steps.keys())}")
print(f"  스케일러: {pipeline.named_steps['scaler']}")
print(f"  모델: {pipeline.named_steps['model']}")

# ============================================================
# 6. Pipeline 저장/로드
# ============================================================
print("\n" + "=" * 60)
print("6. Pipeline 저장/로드")
print("=" * 60)

# 저장
pipeline_path = os.path.join(MODEL_DIR, 'cancer_pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"\nPipeline 저장: {pipeline_path}")

pipeline_size = os.path.getsize(pipeline_path)
print(f"파일 크기: {pipeline_size/1024:.1f} KB")

# 로드
loaded_pipeline = joblib.load(pipeline_path)

# 검증
loaded_pipe_acc = loaded_pipeline.score(X_test, y_test)
print(f"\n로드 후 정확도: {loaded_pipe_acc:.2%}")

# 새 데이터 예측
print("\n새 데이터 예측:")
# 실제 테스트 데이터에서 샘플 추출
new_data = X_test.iloc[:3].copy()

predictions = loaded_pipeline.predict(new_data)
probabilities = loaded_pipeline.predict_proba(new_data)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    diagnosis = cancer.target_names[pred]
    print(f"  샘플 {i+1}: {diagnosis} (양성 확률: {prob[1]:.1%})")

# ============================================================
# 7. make_pipeline 간편 함수
# ============================================================
print("\n" + "=" * 60)
print("7. make_pipeline 간편 함수")
print("=" * 60)

# make_pipeline은 이름을 자동 생성
simple_pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

simple_pipeline.fit(X_train, y_train)
print(f"\n자동 생성된 단계 이름: {list(simple_pipeline.named_steps.keys())}")
print(f"정확도: {simple_pipeline.score(X_test, y_test):.2%}")

# ============================================================
# 8. 다단계 Pipeline
# ============================================================
print("\n" + "=" * 60)
print("8. 다단계 Pipeline")
print("=" * 60)

# 결측치 처리 + 정규화 + 모델
multi_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

multi_pipeline.fit(X_train, y_train)
print(f"\n다단계 Pipeline 정확도: {multi_pipeline.score(X_test, y_test):.2%}")
print(f"단계: {list(multi_pipeline.named_steps.keys())}")

# ============================================================
# 9. 메타데이터와 함께 저장
# ============================================================
print("\n" + "=" * 60)
print("9. 메타데이터와 함께 저장")
print("=" * 60)

import sklearn

# 메타데이터 생성
metadata = {
    'model_name': 'Breast Cancer Diagnosis Model',
    'version': 'v1.0.0',
    'created_at': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9',
    'features': feature_cols,
    'target': 'diagnosis',
    'target_names': list(cancer.target_names),
    'train_accuracy': float(train_acc_pipe),
    'test_accuracy': float(test_acc_pipe),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'dataset': 'sklearn.datasets.load_breast_cancer'
}

# 모델 번들 저장
model_bundle = {
    'pipeline': pipeline,
    'metadata': metadata
}

bundle_path = os.path.join(MODEL_DIR, 'cancer_model_bundle.pkl')
joblib.dump(model_bundle, bundle_path)
print(f"\n모델 번들 저장: {bundle_path}")

# 메타데이터 JSON 저장 (별도)
metadata_path = os.path.join(MODEL_DIR, 'cancer_model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"메타데이터 저장: {metadata_path}")

print("\n메타데이터 내용 (주요 항목):")
for key in ['model_name', 'version', 'test_accuracy', 'dataset']:
    print(f"  {key}: {metadata[key]}")

# ============================================================
# 10. 입력 검증 함수
# ============================================================
print("\n" + "=" * 60)
print("10. 입력 검증 함수")
print("=" * 60)

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

    # 값 범위 확인 (유방암 데이터 특성에 맞게)
    warnings = []

    # 음수값 체크
    for col in data.columns:
        if (data[col] < 0).any():
            warnings.append(f"{col}: 음수 값 존재")

    # 이상치 체크 (학습 데이터 범위 기준)
    for col in data.columns:
        if col in X_train.columns:
            q1 = X_train[col].quantile(0.01)
            q99 = X_train[col].quantile(0.99)
            out_of_range = (data[col] < q1) | (data[col] > q99)
            if out_of_range.any():
                warnings.append(f"{col}: {out_of_range.sum()}개 값이 일반 범위 외")

    return True, warnings

# 테스트
print("\n검증 테스트:")

# 정상 입력
valid_data = X_test.iloc[0].to_dict()
is_valid, warnings = validate_input(valid_data, feature_cols)
print(f"  정상 입력: 검증 통과={is_valid}, 경고 수={len(warnings)}")

# ============================================================
# 11. 예측 함수 (검증 포함)
# ============================================================
print("\n" + "=" * 60)
print("11. 예측 함수 (검증 포함)")
print("=" * 60)

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

# 예측 테스트
print("\n예측 테스트:")
test_input = X_test.iloc[0].to_dict()
result = predict_diagnosis(test_input)
print(f"  예측: {result['prediction']}")
print(f"  양성 확률: {result['benign_probability']:.1%}")
print(f"  신뢰도: {result['confidence']:.1%}")

# ============================================================
# 12. 배포 체크리스트 생성
# ============================================================
print("\n" + "=" * 60)
print("12. 배포 체크리스트 생성")
print("=" * 60)

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
""".format(
    test_acc=test_acc_pipe,
    n_features=len(feature_cols),
    sklearn_ver=sklearn.__version__
)

print(checklist)

# 체크리스트 파일 저장
checklist_path = os.path.join(MODEL_DIR, 'deployment_checklist.md')
with open(checklist_path, 'w', encoding='utf-8') as f:
    f.write(checklist)
print(f"\n체크리스트 저장: {checklist_path}")

# ============================================================
# 13. requirements.txt 생성
# ============================================================
print("\n" + "=" * 60)
print("13. requirements.txt 생성")
print("=" * 60)

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
print("\n내용:")
print(requirements)

# ============================================================
# 14. 예측 성능 측정
# ============================================================
print("\n" + "=" * 60)
print("14. 예측 성능 측정")
print("=" * 60)

import time

# 단일 예측 시간 측정
n_trials = 100
single_times = []

for _ in range(n_trials):
    single_input = X_test.iloc[[0]]
    start = time.time()
    _ = loaded_pipeline.predict(single_input)
    single_times.append(time.time() - start)

print(f"\n단일 예측 성능 (n={n_trials}):")
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

# ============================================================
# 15. 특성 중요도 시각화
# ============================================================
print("\n" + "=" * 60)
print("15. 특성 중요도 시각화")
print("=" * 60)

# Pipeline에서 모델 추출
rf_model = pipeline.named_steps['model']
importances = rf_model.feature_importances_

# 정렬
fi_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n[Feature Importance Top 10]")
for _, row in fi_df.head(10).iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature'][:25]:25s}: {row['Importance']:.3f} {bar}")

# 시각화
fig, ax = plt.subplots(figsize=(12, 10))
top15 = fi_df.head(15)
ax.barh(top15['Feature'], top15['Importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Breast Cancer Diagnosis Model')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  → 'models/feature_importance.png' 저장됨")

# ============================================================
# 16. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("16. 핵심 정리")
print("=" * 60)

print("""
[24차시 핵심 정리]

1. joblib 저장/로드
   - joblib.dump(model, 'model.pkl')
   - joblib.load('model.pkl')
   - compress 옵션으로 압축 가능

2. Pipeline 구성
   - Pipeline([('name', transformer), ...])
   - 전처리 + 모델 통합
   - 하나의 파일로 저장/로드

3. 배포 체크리스트
   - 모델: 성능, 버전, 저장
   - 데이터: 형식, 검증, 범위
   - 환경: 의존성, 리소스
   - 테스트: 단위, 통합, 부하
   - 문서: 사용법, 제약사항

4. 사용한 데이터셋
   - sklearn.datasets.load_breast_cancer
   - 569샘플, 30특성
   - 이진 분류 (악성/양성)
   - 실제 의료 데이터 기반

5. 핵심 코드
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
""")

# ============================================================
# 17. 저장된 파일 목록
# ============================================================
print("\n" + "=" * 60)
print("17. 저장된 파일 목록")
print("=" * 60)

print(f"\n저장 디렉토리: {os.path.abspath(MODEL_DIR)}")
print("\n파일 목록:")
for filename in sorted(os.listdir(MODEL_DIR)):
    filepath = os.path.join(MODEL_DIR, filename)
    size = os.path.getsize(filepath)
    print(f"  {filename}: {size/1024:.1f} KB")

print("\n다음 차시 예고: 25차시 - AI API의 이해와 활용")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
