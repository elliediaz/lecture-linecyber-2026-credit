"""
[26차시] AI 프로젝트 종합 실습 - 실습 코드
제조 AI 과정 | Part IV. AI 서비스화와 활용

학습목표:
1. 전체 ML 워크플로우를 종합 실습한다
2. 26차시 전체 과정의 핵심 내용을 복습한다
3. AI 분야 후속 학습 방향을 파악한다

실행: python code.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# 전체 ML 워크플로우 종합 실습
# ============================================================
print("=" * 60)
print("[26차시] AI 프로젝트 종합 실습")
print("제조 AI 과정 | Part IV. AI 서비스화와 활용")
print("=" * 60)

# ============================================================
# 1단계: 문제 정의
# ============================================================
print("\n" + "=" * 50)
print("1단계: 문제 정의")
print("=" * 50)

print("""
▶ 프로젝트: 제조 공정 품질 예측 시스템

   목표: 센서 데이터를 기반으로 제품 불량 여부 예측
   입력: 온도, 습도, 속도, 압력
   출력: 정상(0) / 불량(1)
   활용: 사전 불량 감지로 불량률 감소

   → 명확한 목표 설정이 프로젝트 성공의 첫걸음!
""")

# ============================================================
# 2단계: 데이터 수집 및 확인
# ============================================================
print("\n" + "=" * 50)
print("2단계: 데이터 수집 및 확인")
print("=" * 50)

# 제조 데이터 생성 (실제로는 CSV 파일에서 로드)
np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n_samples),
    'humidity': np.random.normal(50, 8, n_samples),
    'speed': np.random.normal(100, 10, n_samples),
    'pressure': np.random.normal(1.0, 0.1, n_samples)
})

# 불량 확률 계산 (온도, 습도 영향)
defect_prob = (
    0.1 +
    0.3 * ((df['temperature'] - 85) / 10) +
    0.2 * ((df['humidity'] - 50) / 16)
)
df['defect'] = (defect_prob > 0.3).astype(int)

print(f"▶ 데이터 크기: {df.shape}")
print(f"\n▶ 기술통계:")
print(df.describe().round(2))
print(f"\n▶ 상위 5행:")
print(df.head())

# ============================================================
# 3단계: EDA 및 전처리
# ============================================================
print("\n" + "=" * 50)
print("3단계: EDA 및 전처리")
print("=" * 50)

# 결측치 확인
print("▶ 결측치 확인:")
print(df.isnull().sum())

# 타겟 분포
print("\n▶ 타겟 분포 (defect):")
print(df['defect'].value_counts())
print(f"   불량률: {df['defect'].mean():.1%}")

# 상관관계
print("\n▶ 상관관계 (defect 기준):")
correlations = df.corr()['defect'].drop('defect').sort_values(ascending=False)
for col, corr in correlations.items():
    print(f"   {col}: {corr:.3f}")

# ============================================================
# 4단계: 모델 학습
# ============================================================
print("\n" + "=" * 50)
print("4단계: 모델 학습")
print("=" * 50)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 특성과 타겟 분리
feature_names = ['temperature', 'humidity', 'speed', 'pressure']
X = df[feature_names]
y = df['defect']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"▶ 학습 데이터: {X_train.shape[0]}개")
print(f"▶ 테스트 데이터: {X_test.shape[0]}개")

# 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("▶ StandardScaler 적용 완료")

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("▶ RandomForestClassifier 학습 완료!")

# ============================================================
# 5단계: 모델 평가
# ============================================================
print("\n" + "=" * 50)
print("5단계: 모델 평가")
print("=" * 50)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 예측
y_pred = model.predict(X_test_scaled)

# 정확도
accuracy = accuracy_score(y_test, y_pred)
print(f"▶ 정확도: {accuracy:.3f}")

# 혼동행렬
print("\n▶ 혼동행렬:")
cm = confusion_matrix(y_test, y_pred)
print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
print(f"   FN={cm[1,0]}, TP={cm[1,1]}")

# 분류 보고서
print("\n▶ 분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))

# ============================================================
# 6단계: 모델 해석
# ============================================================
print("\n" + "=" * 50)
print("6단계: 모델 해석 (Feature Importance)")
print("=" * 50)

# 특성 중요도
importance = model.feature_importances_

print("▶ 특성 중요도:")
for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
    bar = "█" * int(imp * 30)
    print(f"   {name:12s}: {imp:.3f} {bar}")

print("\n▶ 해석: 온도와 습도가 품질에 가장 큰 영향을 미침")

# ============================================================
# 7단계: 모델 저장
# ============================================================
print("\n" + "=" * 50)
print("7단계: 모델 저장 (joblib)")
print("=" * 50)

import joblib
import os

# 모델 패키지 생성
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'version': '1.0.0',
    'trained_date': datetime.now().isoformat(),
    'accuracy': accuracy,
    'n_samples': len(X_train),
    'description': '제조 품질 예측 모델 (종합 실습)'
}

# 저장
joblib.dump(model_package, 'quality_model_package.pkl')
print("▶ 모델 패키지 저장: quality_model_package.pkl")

# 파일 크기 확인
file_size = os.path.getsize('quality_model_package.pkl') / 1024
print(f"▶ 파일 크기: {file_size:.1f} KB")

# 저장된 정보 확인
print("\n▶ 저장된 모델 정보:")
print(f"   버전: {model_package['version']}")
print(f"   정확도: {model_package['accuracy']:.3f}")
print(f"   학습 일시: {model_package['trained_date'][:19]}")

# ============================================================
# 8단계: API 서비스 (FastAPI 코드 예시)
# ============================================================
print("\n" + "=" * 50)
print("8단계: API 서비스 (FastAPI)")
print("=" * 50)

api_code = '''
# main.py - FastAPI 품질 예측 서비스
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="품질 예측 API", version="1.0")

# 앱 시작 시 모델 로드
pkg = joblib.load("quality_model_package.pkl")
model = pkg["model"]
scaler = pkg["scaler"]

@app.get("/health")
def health():
    return {"status": "healthy", "version": pkg["version"]}

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

# API 테스트 시뮬레이션
print("▶ API 테스트 시뮬레이션:")
test_cases = [
    [90, 55, 100, 1.0],   # 높은 온도, 습도
    [82, 48, 100, 1.0],   # 정상 범위
    [95, 65, 100, 1.0],   # 매우 높음
]

loaded_pkg = joblib.load('quality_model_package.pkl')
loaded_model = loaded_pkg['model']
loaded_scaler = loaded_pkg['scaler']

for i, case in enumerate(test_cases):
    scaled = loaded_scaler.transform([case])
    pred = loaded_model.predict(scaled)[0]
    label = "불량" if pred == 1 else "정상"
    print(f"   케이스 {i+1}: 온도={case[0]}, 습도={case[1]} → {label}")

# ============================================================
# ML 워크플로우 8단계 완료!
# ============================================================
print("\n" + "=" * 50)
print("ML 워크플로우 8단계 완료!")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────┐
│           ML 프로젝트 8단계 체크리스트               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [✓] 1. 문제 정의: 품질 예측 시스템                 │
│  [✓] 2. 데이터 수집: manufacturing_data             │
│  [✓] 3. EDA/전처리: 결측치, 분포, 스케일링         │
│  [✓] 4. 모델 학습: RandomForestClassifier           │
│  [✓] 5. 모델 평가: 정확도, F1 Score                │
│  [✓] 6. 모델 해석: Feature Importance               │
│  [✓] 7. 모델 저장: joblib + 메타데이터              │
│  [✓] 8. API 서비스: FastAPI                         │
│                                                      │
└─────────────────────────────────────────────────────┘
""")

# ============================================================
# 과정 총정리
# ============================================================
print("\n" + "=" * 50)
print("26차시 과정 총정리")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────┐
│           Part I: AI 윤리와 환경 구축 (1-3차시)     │
├─────────────────────────────────────────────────────┤
│  • AI 활용 윤리와 데이터 보호                        │
│  • Python 시작하기                                   │
│  • 제조 데이터 다루기 기초                           │
│                                                      │
│  핵심 역량: pandas, DataFrame, 기본 데이터 처리      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│       Part II: 기초 수리와 데이터 분석 (4-9차시)    │
├─────────────────────────────────────────────────────┤
│  • 데이터 요약과 시각화                              │
│  • 확률분포와 품질 검정                              │
│  • 상관분석과 예측의 기초                            │
│  • 제조 데이터 전처리 (1), (2)                       │
│  • 제조 데이터 탐색 분석 종합                        │
│                                                      │
│  핵심 역량: 통계, 시각화, 전처리, EDA               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│       Part III: 문제 중심 모델링 실습 (10-19차시)   │
├─────────────────────────────────────────────────────┤
│  • 머신러닝 소개와 문제 유형                         │
│  • 분류 모델: 의사결정나무, 랜덤포레스트             │
│  • 예측 모델: 선형회귀, 다항회귀                     │
│  • 모델 평가와 반복 검증                             │
│  • 모델 설정값 최적화                                │
│  • 시계열 데이터 기초, 예측 모델                     │
│  • 딥러닝 입문: 신경망 기초, MLP 품질 예측           │
│                                                      │
│  핵심 역량: 분류, 회귀, 평가, 시계열, 딥러닝        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│       Part IV: AI 서비스화와 활용 (20-26차시)       │
├─────────────────────────────────────────────────────┤
│  • AI API의 이해와 활용                              │
│  • LLM API와 프롬프트 작성법                         │
│  • Streamlit으로 웹앱 만들기                         │
│  • FastAPI로 예측 서비스 만들기                      │
│  • 모델 해석과 변수별 영향력 분석                    │
│  • 모델 저장과 실무 배포 준비                        │
│  • AI 프로젝트 종합 실습                             │
│                                                      │
│  핵심 역량: API, LLM, 웹앱, 배포, 서비스화          │
└─────────────────────────────────────────────────────┘
""")

# ============================================================
# 핵심 라이브러리 정리
# ============================================================
print("\n" + "=" * 50)
print("핵심 라이브러리 정리")
print("=" * 50)

print("""
┌──────────────┬──────────────────────────────────────┐
│   영역       │   라이브러리                          │
├──────────────┼──────────────────────────────────────┤
│ 데이터       │ pandas, numpy                         │
│ 시각화       │ matplotlib, seaborn                   │
│ ML          │ scikit-learn                          │
│ DL          │ keras, tensorflow                     │
│ 서비스       │ streamlit, fastapi                    │
│ 저장         │ joblib                                │
└──────────────┴──────────────────────────────────────┘
""")

# ============================================================
# 후속 학습 안내
# ============================================================
print("\n" + "=" * 50)
print("후속 학습 안내")
print("=" * 50)

print("""
▶ 심화 학습 추천:
   • 딥러닝: CNN, RNN, Transformer
   • 자연어처리: 텍스트 분류, 챗봇
   • 컴퓨터 비전: 이미지 분류, 객체 탐지

▶ 실무 역량:
   • MLOps: 모델 운영, 모니터링
   • 클라우드: AWS, GCP, Azure ML
   • 데이터 엔지니어링: 파이프라인 구축

▶ 자격증 추천:
   • 빅데이터분석기사
   • ADsP (데이터분석 준전문가)

▶ 학습 자료:
   • Coursera, 네이버 부스트캠프
   • Kaggle 경진대회
   • 핸즈온 머신러닝, 밑바닥부터 시작하는 딥러닝
""")

# ============================================================
# 마무리
# ============================================================
print("\n" + "=" * 50)
print("26차시 과정을 마치며")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────┐
│                                                      │
│     Python 기초부터 AI 서비스 배포까지               │
│     전체 ML 파이프라인을 경험했습니다!               │
│                                                      │
│     데이터 수집 → 분석 → 모델링 → 배포               │
│                                                      │
│     이 과정이 AI 분야로 나아가는                     │
│     좋은 첫걸음이 되길 바랍니다.                     │
│                                                      │
│               수고 많으셨습니다!                      │
│                                                      │
└─────────────────────────────────────────────────────┘
""")

# 생성된 파일 정리
if os.path.exists('quality_model_package.pkl'):
    print(f"\n▶ 생성된 파일: quality_model_package.pkl ({file_size:.1f} KB)")
