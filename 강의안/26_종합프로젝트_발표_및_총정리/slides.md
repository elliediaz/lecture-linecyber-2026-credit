---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 26차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 종합 프로젝트 발표 및 과정 총정리

## 26차시 | AI 기초체력훈련 (Pre AI-Campus)

**최종 발표와 전체 과정 마무리**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **프로젝트 결과물**을 발표하고 피드백을 주고받는다
2. **25차시 전체 과정**의 핵심 내용을 복습한다
3. **AI 분야 후속 학습** 방향을 파악한다

---

# 오늘의 진행 순서

## 발표 및 총정리

| 순서 | 내용 | 시간 |
|------|------|------|
| 1 | 프로젝트 발표 | 12분 |
| 2 | 피드백 및 토론 | 5분 |
| 3 | 과정 총정리 | 5분 |
| 4 | 수료 안내 및 Q&A | 3분 |

---

# 프로젝트 발표 안내

## 발표 구성 (팀/개인당 3-4분)

### 필수 포함 내용
1. **프로젝트 개요**: 해결하고자 한 문제
2. **데이터 및 모델**: 사용한 데이터, 선택한 알고리즘
3. **결과**: 모델 성능, 주요 인사이트
4. **시연**: 웹 서비스 또는 API 데모

### 발표 팁
- 핵심만 간결하게
- 시각 자료 활용
- 어려웠던 점과 해결 과정 공유

---

# 프로젝트 평가 기준

## 평가 항목

| 항목 | 배점 | 세부 기준 |
|------|------|----------|
| **문제 정의** | 20% | 명확한 목표 설정 |
| **데이터 처리** | 20% | 전처리, EDA 적절성 |
| **모델링** | 25% | 알고리즘 선택, 성능 |
| **서비스화** | 25% | API/웹앱 구현 |
| **발표** | 10% | 전달력, 시간 준수 |

---

# 과정 총정리

## Part I: AI 기초와 Python (1~6차시)

### 학습 내용
- AI 활용 윤리 및 저작권
- Python 환경 구축과 기초 문법
- NumPy와 Pandas 기초

### 핵심 역량
```python
import numpy as np
import pandas as pd

# 데이터프레임 생성과 조작
df = pd.DataFrame({'col': [1, 2, 3]})
```

---

# 과정 총정리

## Part II: 데이터 분석 기초 (7~12차시)

### 학습 내용
- 기술통계와 확률분포
- 탐색적 데이터 분석 (EDA)
- 데이터 전처리와 시각화

### 핵심 역량
```python
# EDA와 시각화
df.describe()
df.groupby('category').mean()
plt.hist(df['value'])
```

---

# 과정 총정리

## Part III: 머신러닝 (13~20차시)

### 학습 내용
- 회귀, 분류, 군집화 모델
- 모델 평가 지표 (정확도, F1, RMSE)
- 하이퍼파라미터 튜닝
- 딥러닝 기초

### 핵심 역량
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

# 과정 총정리

## Part IV: 서비스화와 배포 (21~26차시)

### 학습 내용
- Streamlit 웹 대시보드
- FastAPI 모델 서빙
- 모델 해석 (Feature Importance)
- 모델 저장과 배포
- **종합 프로젝트**

### 핵심 역량
```python
import joblib
from fastapi import FastAPI

model = joblib.load('model.pkl')
app = FastAPI()
```

---

# 전체 ML 워크플로우

## 데이터에서 서비스까지

```
┌─────────────────────────────────────────────────────┐
│                   ML 프로젝트 흐름                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. 문제 정의 → 2. 데이터 수집                      │
│       ↓                                              │
│  3. EDA/전처리 → 4. 모델 학습                       │
│       ↓                                              │
│  5. 평가/튜닝 → 6. 모델 저장                        │
│       ↓                                              │
│  7. API 개발 → 8. 웹 서비스 배포                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

# 핵심 라이브러리 정리

## 자주 사용한 도구들

| 영역 | 라이브러리 | 용도 |
|------|-----------|------|
| 데이터 | pandas, numpy | 데이터 처리 |
| 시각화 | matplotlib, seaborn | 차트, 그래프 |
| ML | scikit-learn | 모델 학습/평가 |
| DL | tensorflow, keras | 딥러닝 |
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

# 후속 학습 로드맵

## 추천 자격증

| 자격증 | 주관 | 난이도 |
|--------|------|--------|
| **SQLD** | 한국데이터산업진흥원 | 초급 |
| **빅데이터분석기사** | 한국데이터산업진흥원 | 중급 |
| **ADsP** | 한국데이터산업진흥원 | 초급 |
| **TensorFlow 인증** | Google | 중급 |
| **AWS ML Specialty** | Amazon | 고급 |

---

# 추천 학습 자료

## 온라인 강의
- Coursera: Machine Learning (Andrew Ng)
- 네이버 부스트캠프
- 카카오 AI 캠프

## 도서
- 핸즈온 머신러닝
- 밑바닥부터 시작하는 딥러닝
- 파이썬 머신러닝 완벽 가이드

## 커뮤니티
- Kaggle 경진대회
- GitHub 오픈소스 프로젝트
- AI 관련 스터디 그룹

---

# 수료 안내

## 수료 조건

| 항목 | 기준 |
|------|------|
| 출석률 | 80% 이상 |
| 과제 제출 | 모든 과제 제출 |
| 최종 프로젝트 | 발표 완료 |

## 수료 혜택
- AI 기초체력훈련 수료증 발급
- 후속 교육 과정 우선 신청권
- 수료생 네트워크 참여

---

# Q&A

## 질문과 답변

궁금한 점이 있으시면 질문해주세요!

### 자주 묻는 질문
1. **수료증은 언제 받나요?**
   → 과정 종료 후 1주 이내 발급

2. **추가 학습 자료는 어디서?**
   → 학습 플랫폼에서 계속 열람 가능

3. **후속 과정은?**
   → AI 심화 과정 별도 안내 예정

---

# 감사합니다

## AI 기초체력훈련 26차시

**종합 프로젝트 발표 및 과정 총정리**

26차시 전 과정을 모두 마쳤습니다!

**수고 많으셨습니다!**

---

# 부록: 핵심 코드 요약

## 데이터 로드부터 배포까지

```python
# 1. 데이터 로드
import pandas as pd
df = pd.read_csv('data.csv')

# 2. 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 모델 학습
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. 저장
import joblib
joblib.dump(model, 'model.pkl')
```

