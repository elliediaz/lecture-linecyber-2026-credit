---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 26차시'
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

# AI 프로젝트 종합 실습

## 26차시 | Part IV. AI 서비스화와 활용

**전체 ML 워크플로우 종합 실습 및 과정 총정리**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **전체 ML 워크플로우**를 종합 실습한다
2. **26차시 전체 과정**의 핵심 내용을 복습한다
3. **AI 분야 후속 학습** 방향을 파악한다

---

# 오늘의 진행 순서

## 종합 실습 및 총정리

| 순서 | 내용 | 시간 |
|------|------|------|
| 1 | ML 워크플로우 종합 실습 | 15분 |
| 2 | 과정 총정리 | 8분 |
| 3 | 후속 학습 안내 및 Q&A | 7분 |

---

# Part I: 전체 ML 워크플로우

## 데이터에서 서비스까지

```
┌─────────────────────────────────────────────────────┐
│               ML 프로젝트 흐름 (8단계)               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. 문제 정의 → 2. 데이터 수집                      │
│       ↓                                              │
│  3. EDA/전처리 → 4. 모델 학습                       │
│       ↓                                              │
│  5. 평가/튜닝 → 6. 모델 해석                        │
│       ↓                                              │
│  7. 모델 저장 → 8. API/웹 서비스                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

# - 실습편 -

## 26차시

**품질 예측 프로젝트 종합 실습**

---

# 실습 1: 문제 정의

## 프로젝트 목표

```python
"""
프로젝트: 제조 공정 품질 예측 시스템

목표: 센서 데이터를 기반으로 제품 불량 여부 예측
입력: 온도, 습도, 속도, 압력
출력: 정상(0) / 불량(1)

활용: 사전 불량 감지로 불량률 감소
"""
```

> 명확한 목표 설정이 프로젝트 성공의 첫걸음!

---

# 실습 2: 데이터 수집 및 확인

## 데이터 로드

```python
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('manufacturing_data.csv')

# 기본 정보 확인
print(df.shape)        # (500, 5)
print(df.info())       # 컬럼 타입
print(df.describe())   # 기술통계
print(df.head())       # 상위 5행
```

---

# 실습 3: EDA 및 전처리

## 탐색적 데이터 분석

```python
import matplotlib.pyplot as plt

# 결측치 확인
print(df.isnull().sum())

# 타겟 분포
print(df['defect'].value_counts())

# 상관관계
print(df.corr()['defect'].sort_values())

# 시각화
plt.hist(df['temperature'], bins=20)
plt.title('온도 분포')
plt.show()
```

---

# 실습 4: 모델 학습

## 랜덤포레스트 분류

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 데이터 분할
X = df.drop('defect', axis=1)
y = df['defect']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 전처리 + 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)
```

---

# 실습 5: 모델 평가

## 성능 지표 확인

```python
from sklearn.metrics import accuracy_score, classification_report

# 예측
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# 평가
print(f"정확도: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

> 정확도, 정밀도, 재현율, F1 Score 확인!

---

# 실습 6: 모델 해석

## 특성 중요도 분석

```python
# Feature Importance
importance = model.feature_importances_
for name, imp in zip(X.columns, importance):
    print(f"{name}: {imp:.3f}")

# 시각화
plt.barh(X.columns, importance)
plt.xlabel('중요도')
plt.title('Feature Importance')
plt.show()
```

> 온도가 품질에 가장 큰 영향!

---

# 실습 7: 모델 저장

## joblib으로 저장

```python
import joblib
from datetime import datetime

# 메타데이터 포함 저장
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': list(X.columns),
    'version': '1.0.0',
    'trained_date': datetime.now().isoformat(),
    'accuracy': accuracy_score(y_test, y_pred)
}

joblib.dump(model_package, 'quality_model_package.pkl')
print("모델 저장 완료!")
```

---

# 실습 8: API 서비스

## FastAPI 구현

```python
from fastapi import FastAPI
import joblib

app = FastAPI(title="품질 예측 API")

# 모델 로드
pkg = joblib.load("quality_model_package.pkl")
model = pkg['model']
scaler = pkg['scaler']

@app.post("/predict")
def predict(temp: float, humidity: float, speed: float, pressure: float):
    features = [[temp, humidity, speed, pressure]]
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    return {"prediction": "불량" if pred == 1 else "정상"}
```

---

# 실습 정리

## ML 워크플로우 8단계 완료

- [x] 1. 문제 정의: 품질 예측 시스템
- [x] 2. 데이터 수집: manufacturing_data.csv
- [x] 3. EDA/전처리: 결측치, 분포, 스케일링
- [x] 4. 모델 학습: RandomForestClassifier
- [x] 5. 모델 평가: 정확도, F1 Score
- [x] 6. 모델 해석: Feature Importance
- [x] 7. 모델 저장: joblib + 메타데이터
- [x] 8. API 서비스: FastAPI

---

# 과정 총정리

## Part I: AI 윤리와 환경 구축 (1-3차시)

### 학습 내용
- AI 활용 윤리와 데이터 보호
- Python 시작하기
- 제조 데이터 다루기 기초

### 핵심 역량
```python
import pandas as pd
df = pd.DataFrame({'col': [1, 2, 3]})
df.head()
```

---

# 과정 총정리

## Part II: 기초 수리와 데이터 분석 (4-9차시)

### 학습 내용
- 데이터 요약과 시각화
- 확률분포와 품질 검정
- 상관분석과 예측의 기초
- 데이터 전처리 실무
- 탐색적 데이터 분석 종합

### 핵심 역량
```python
df.describe()
df.groupby('category').mean()
plt.hist(df['value'])
```

---

# 과정 총정리

## Part III: 문제 중심 모델링 실습 (10-19차시)

### 학습 내용
- 머신러닝 소개와 문제 유형
- 분류 모델, 예측 모델
- 모델 평가와 반복 검증
- 시계열 데이터와 예측
- 딥러닝 입문 및 실습

### 핵심 역량
```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

# 과정 총정리

## Part IV: AI 서비스화와 활용 (20-26차시)

### 학습 내용
- AI API의 이해와 활용
- LLM API와 프롬프트 작성법
- Streamlit 웹앱
- FastAPI 예측 서비스
- 모델 해석과 저장
- **종합 프로젝트**

### 핵심 역량
```python
joblib.dump(model, 'model.pkl')
app = FastAPI()
```

---

# 핵심 라이브러리 정리

## 자주 사용한 도구들

| 영역 | 라이브러리 | 용도 |
|------|-----------|------|
| 데이터 | pandas, numpy | 데이터 처리 |
| 시각화 | matplotlib, seaborn | 차트, 그래프 |
| ML | scikit-learn | 모델 학습/평가 |
| DL | keras, tensorflow | 딥러닝 |
| 서비스 | streamlit, fastapi | 웹앱, API |
| 저장 | joblib | 모델 저장 |

---

# 후속 학습 로드맵

## 다음 단계 추천

### 심화 학습
- **딥러닝**: CNN, RNN, Transformer
- **자연어처리**: 텍스트 분류, 챗봇
- **컴퓨터 비전**: 이미지 분류, 객체 탐지

### 실무 역량
- **MLOps**: 모델 운영, 모니터링
- **클라우드**: AWS, GCP, Azure ML
- **데이터 엔지니어링**: 파이프라인 구축

---

# 추천 학습 자료

## 온라인 강의
- Coursera: Machine Learning (Andrew Ng)
- 네이버 부스트캠프
- K-디지털 트레이닝

## 도서
- 핸즈온 머신러닝
- 밑바닥부터 시작하는 딥러닝
- 파이썬 머신러닝 완벽 가이드

## 커뮤니티
- Kaggle 경진대회
- GitHub 오픈소스 프로젝트

---

# 수료 안내

## 수료 조건

| 항목 | 기준 |
|------|------|
| 출석률 | 80% 이상 |
| 과제 제출 | 모든 과제 제출 |
| 최종 프로젝트 | 발표 완료 |

## 수료 혜택
- 수료증 발급
- 후속 교육 과정 우선 신청권
- 수료생 네트워크 참여

---

# 정리 및 Q&A

## 26차시 과정을 마치며

1. **전체 ML 워크플로우**: 문제 정의 → 서비스 배포
2. **Part I-IV 핵심**: Python부터 FastAPI까지
3. **후속 학습**: 딥러닝, MLOps, 클라우드

> 데이터 수집부터 서비스 배포까지
> 전체 AI 프로젝트 파이프라인을 경험했습니다!

---

# 감사합니다

## 26차시: AI 프로젝트 종합 실습

**제조데이터를 활용한 AI 이해와 예측 모델 구축**

26차시 전 과정을 모두 마쳤습니다!

**수고 많으셨습니다!**
