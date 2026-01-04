# [26차시] AI 프로젝트 종합 실습 - 강사 스크립트

## 강의 정보
- **차시**: 26차시 (25-30분) - 마지막 차시
- **유형**: 종합 실습 + 총정리
- **구성**: 종합 실습 15분 + 총정리 15분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 종합 실습편 (15분)

### 도입 (2분)

#### 인사 및 오늘 안내 [2분]

> 안녕하세요, 드디어 마지막 26차시입니다.
>
> 지난 25차시 동안 Python 기초부터 모델 배포까지 함께 해오셨습니다.
>
> 오늘은 **전체 ML 워크플로우를 종합 실습**하고, **과정 총정리**로 마무리하겠습니다.
>
> 문제 정의부터 API 서비스까지 8단계를 빠르게 실습해봅니다!

---

### 종합 실습 (13분)

#### 실습 1: 문제 정의 [1min]

> 첫 번째 단계, 문제 정의입니다.
>
> 프로젝트: 제조 공정 품질 예측 시스템
> 목표: 센서 데이터로 불량 여부 예측
> 입력: 온도, 습도, 속도, 압력
> 출력: 정상(0) / 불량(1)
>
> 명확한 목표 설정이 프로젝트 성공의 첫걸음입니다!

#### 실습 2: 데이터 확인 [1.5min]

> 두 번째 단계, 데이터 수집 및 확인입니다.
>
> ```python
> import pandas as pd
> df = pd.read_csv('manufacturing_data.csv')
>
> print(df.shape)      # 데이터 크기
> print(df.describe()) # 기술통계
> print(df.head())     # 상위 5행
> ```
>
> 데이터를 로드하고 기본 정보를 확인합니다.

#### 실습 3: EDA 및 전처리 [1.5min]

> 세 번째 단계, 탐색적 데이터 분석입니다.
>
> ```python
> print(df.isnull().sum())           # 결측치
> print(df['defect'].value_counts()) # 타겟 분포
> print(df.corr()['defect'])         # 상관관계
> ```
>
> 결측치 처리, 이상치 확인, 분포 파악을 합니다.

#### 실습 4: 모델 학습 [2min]

> 네 번째 단계, 모델 학습입니다.
>
> ```python
> from sklearn.model_selection import train_test_split
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.preprocessing import StandardScaler
>
> X_train, X_test, y_train, y_test = train_test_split(...)
>
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)
>
> model = RandomForestClassifier(n_estimators=100)
> model.fit(X_train_scaled, y_train)
> ```
>
> 전처리하고 랜덤포레스트로 학습합니다.

#### 실습 5: 모델 평가 [1.5min]

> 다섯 번째 단계, 모델 평가입니다.
>
> ```python
> from sklearn.metrics import accuracy_score, classification_report
>
> y_pred = model.predict(X_test_scaled)
> print(f"정확도: {accuracy_score(y_test, y_pred):.3f}")
> print(classification_report(y_test, y_pred))
> ```
>
> 정확도, 정밀도, 재현율, F1 Score를 확인합니다.

#### 실습 6: 모델 해석 [1.5min]

> 여섯 번째 단계, 모델 해석입니다.
>
> ```python
> importance = model.feature_importances_
> for name, imp in zip(X.columns, importance):
>     print(f"{name}: {imp:.3f}")
> ```
>
> 온도가 품질에 가장 큰 영향을 미친다는 것을 알 수 있습니다.

#### 실습 7: 모델 저장 [1.5min]

> 일곱 번째 단계, 모델 저장입니다.
>
> ```python
> import joblib
>
> model_package = {
>     'model': model,
>     'scaler': scaler,
>     'version': '1.0.0'
> }
> joblib.dump(model_package, 'quality_model_package.pkl')
> ```
>
> 메타데이터와 함께 저장합니다.

#### 실습 8: API 서비스 [2min]

> 마지막 여덟 번째 단계, API 서비스입니다.
>
> ```python
> from fastapi import FastAPI
>
> app = FastAPI(title="품질 예측 API")
>
> @app.post("/predict")
> def predict(temp: float, humidity: float, ...):
>     features = [[temp, humidity, speed, pressure]]
>     pred = model.predict(scaler.transform(features))[0]
>     return {"prediction": "불량" if pred == 1 else "정상"}
> ```
>
> 8단계 전체 워크플로우를 완성했습니다!

---

## 총정리편 (15분)

### 과정 총정리 (8분)

#### Part I-II 요약 [2min]

> 26차시 동안 배운 내용을 빠르게 정리하겠습니다.
>
> **Part I (1-3차시): AI 윤리와 환경 구축**
> - AI 윤리, Python 기초, Pandas
> - 데이터를 다루는 기본기를 익혔습니다
>
> **Part II (4-9차시): 기초 수리와 데이터 분석**
> - 통계, 시각화, 전처리, EDA
> - 데이터를 탐색하고 정제하는 방법을 배웠습니다

#### Part III-IV 요약 [2min]

> **Part III (10-19차시): 문제 중심 모델링**
> - 분류, 회귀, 평가, 시계열, 딥러닝
> - 모델을 학습하고 개선하는 전 과정을 경험했습니다
>
> **Part IV (20-26차시): AI 서비스화**
> - API, LLM, Streamlit, FastAPI, 해석, 배포
> - 모델을 실제 서비스로 만드는 방법을 배웠습니다
>
> 데이터 수집부터 서비스 배포까지, 전체 ML 파이프라인을 경험하셨습니다!

#### 핵심 라이브러리 정리 [2min]

> 자주 사용한 도구들을 정리합니다.
>
> - **데이터**: pandas, numpy
> - **시각화**: matplotlib, seaborn
> - **ML**: scikit-learn
> - **DL**: keras, tensorflow
> - **서비스**: streamlit, fastapi
> - **저장**: joblib
>
> 이 도구들을 활용해 AI 프로젝트를 수행할 수 있습니다!

#### 후속 학습 안내 [2min]

> 앞으로의 학습 방향을 안내드립니다.
>
> **심화 학습 추천**:
> - 딥러닝: CNN, RNN, Transformer
> - 자연어처리, 컴퓨터 비전
>
> **자격증 추천**:
> - 빅데이터분석기사, ADsP
>
> **학습 자료**:
> - Coursera, Kaggle, 도서
>
> 오늘 배운 기초 위에 계속 쌓아가시면 됩니다!

---

### 마무리 (7분)

#### 수료 안내 [2min]

> 수료 조건을 안내드립니다.
>
> - 출석률 80% 이상
> - 과제 제출 완료
> - 최종 프로젝트 발표 완료
>
> 수료증은 과정 종료 후 발급됩니다.
> 학습 플랫폼은 수료 후에도 일정 기간 열람 가능합니다.

#### Q&A [4min]

> 질문 있으시면 편하게 해주세요.
>
> *(질문 응답)*

#### 마무리 인사 [1min]

> 26차시 과정을 모두 마쳤습니다.
>
> Python도 처음, 머신러닝도 처음이셨던 분들이
> 이제 직접 모델을 만들고 서비스로 배포할 수 있게 되셨습니다.
>
> 이 과정이 AI 분야로 나아가는 좋은 첫걸음이 되길 바랍니다.
>
> 수고 많으셨습니다. 감사합니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- 종합 실습용 데이터 (manufacturing_data.csv)

### 주의사항
- 종합 실습은 시간 내 완료 가능하도록 핵심만 진행
- 총정리는 빠른 리뷰 형태로 진행
- 마무리 시간 충분히 확보

### 예상 질문
1. "수료증에 성적이 기재되나요?"
   → 수료/미수료만 표기, 별도 성적 기재 없음

2. "강의 자료는 계속 볼 수 있나요?"
   → 플랫폼 정책에 따라 일정 기간 열람 가능

3. "후속 과정은 언제 시작하나요?"
   → 별도 공지 예정, 수료생 우선 안내

### 시간 조절 팁
- **시간 부족**: 후속 학습 안내 간략히
- **시간 여유**: 추가 Q&A, 개별 질문 응답
