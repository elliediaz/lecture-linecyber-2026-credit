# [25차시] 모델 저장과 실무 배포 준비 - 강사 스크립트

## 강의 정보
- **차시**: 25차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 25차시를 시작하겠습니다.
>
> 지난 시간에 모델 해석과 특성 중요도 분석을 배웠습니다. 모델이 왜 그렇게 예측하는지 설명할 수 있게 됐죠.
>
> 오늘은 **모델 저장과 배포 준비**를 배웁니다. 학습한 모델을 파일로 저장하고, 나중에 불러와서 사용하는 방법이에요!

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, joblib으로 모델을 저장하고 불러옵니다.
> 둘째, 모델 버전 관리 방법을 이해합니다.
> 셋째, 실무 배포 체크리스트를 활용합니다.

---

### 핵심 내용 (8분)

#### 왜 모델을 저장해야 하나? [2min]

> 모델 학습에는 시간이 걸려요. 간단한 모델도 몇 분, 복잡한 모델은 몇 시간이 걸리죠.
>
> 매번 예측할 때마다 학습하면 시간 낭비예요.
>
> 그래서 한 번 학습한 모델을 **저장**해두고, 나중에 **불러와서** 사용해요.
>
> 실제로 이렇게 사용해요:
>
> 1. 개발 환경에서 모델 학습 (Jupyter 노트북)
> 2. 모델 파일로 저장 (model.pkl)
> 3. API 서버에서 파일 로드
> 4. 요청이 오면 바로 예측!

#### joblib 사용법 [2min]

> **joblib**은 scikit-learn에서 공식 권장하는 저장 방법이에요.
>
> ```python
> import joblib
>
> # 저장 (한 줄!)
> joblib.dump(model, 'model.pkl')
>
> # 불러오기 (한 줄!)
> loaded_model = joblib.load('model.pkl')
> ```
>
> NumPy 배열에 최적화되어 있어서 ML 모델에 적합해요.
> pickle보다 대용량 모델에서 훨씬 빠릅니다.

#### 전처리기도 함께 저장 [2min]

> 주의할 점! 전처리기도 같이 저장해야 해요.
>
> ```python
> # 학습할 때
> scaler = StandardScaler()
> X_scaled = scaler.fit_transform(X_train)
> model.fit(X_scaled, y_train)
>
> # 저장 (둘 다!)
> joblib.dump(scaler, 'scaler.pkl')
> joblib.dump(model, 'model.pkl')
> ```
>
> 예측할 때 같은 스케일러를 써야 올바른 결과가 나와요!
>
> 더 깔끔한 방법은 파이프라인으로 묶는 거예요.
>
> ```python
> pipeline = Pipeline([
>     ('scaler', StandardScaler()),
>     ('model', RandomForestClassifier())
> ])
> joblib.dump(pipeline, 'pipeline.pkl')
> ```

#### 버전 관리와 배포 준비 [2min]

> 실무에서는 모델이 여러 번 업데이트돼요.
>
> 파일명에 규칙을 정해두면 관리가 편해요:
>
> ```
> quality_model_v1.0_20260101.pkl
> quality_model_v2.0_20260201.pkl
> ```
>
> 배포 전에는 반드시 확인해야 할 것들이 있어요:
>
> 1. **모델 검증**: 테스트 데이터 성능 OK?
> 2. **파일 확인**: model.pkl, scaler.pkl 있음?
> 3. **환경 확인**: Python 버전, 라이브러리 버전 일치?
>
> requirements.txt로 환경을 고정해두면 버전 문제를 예방할 수 있어요.

---

## 실습편 (15-20분)

### 실습 소개 [1.5min]

> 이제 실습 시간입니다. 품질 예측 모델을 저장하고 불러와봅니다.
>
> **실습 목표**입니다.
> 1. joblib으로 모델 저장/불러오기
> 2. 전처리기 함께 관리
> 3. 메타데이터 저장
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import joblib
> from sklearn.pipeline import Pipeline
> ```

### 실습 1: 모델 학습 [2min]

> 첫 번째 실습입니다. 랜덤포레스트 모델을 학습합니다.
>
> ```python
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.preprocessing import StandardScaler
>
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)
>
> model = RandomForestClassifier(n_estimators=100, random_state=42)
> model.fit(X_train_scaled, y_train)
> ```
>
> 전처리기와 모델을 각각 학습했어요.

### 실습 2: joblib 저장 [2min]

> 두 번째 실습입니다. joblib으로 저장합니다.
>
> ```python
> import joblib
>
> joblib.dump(model, 'model.pkl')
> joblib.dump(scaler, 'scaler.pkl')
> ```
>
> 정말 간단하죠? 이게 전부예요!
>
> 파일 크기도 확인해볼까요?

### 실습 3: 모델 불러오기 [2min]

> 세 번째 실습입니다. 저장된 모델을 불러와서 예측합니다.
>
> ```python
> loaded_model = joblib.load('model.pkl')
> loaded_scaler = joblib.load('scaler.pkl')
>
> new_data = np.array([[90, 55, 100, 1.0]])
> new_scaled = loaded_scaler.transform(new_data)
> prediction = loaded_model.predict(new_scaled)
> ```
>
> 학습 시간 0초로 바로 예측할 수 있어요!

### 실습 4: 파이프라인 [2min]

> 네 번째 실습입니다. 파이프라인으로 한번에 관리합니다.
>
> ```python
> from sklearn.pipeline import Pipeline
>
> pipeline = Pipeline([
>     ('scaler', StandardScaler()),
>     ('model', RandomForestClassifier(n_estimators=100))
> ])
>
> pipeline.fit(X_train, y_train)
> joblib.dump(pipeline, 'pipeline.pkl')
> ```
>
> 파이프라인을 쓰면 전처리와 모델을 하나로 묶을 수 있어요.
> 파일 하나만 관리하면 되니까 실수를 방지할 수 있죠.

### 실습 5: 메타데이터 저장 [2min]

> 다섯 번째 실습입니다. 버전 정보를 함께 저장합니다.
>
> ```python
> from datetime import datetime
>
> model_package = {
>     'model': model,
>     'scaler': scaler,
>     'version': '1.0.0',
>     'accuracy': 0.92,
>     'trained_date': datetime.now().isoformat()
> }
>
> joblib.dump(model_package, 'model_package.pkl')
> ```
>
> 불러올 때 버전과 성능을 바로 확인할 수 있어요!

### 실습 6: 배포 체크리스트 [2min]

> 여섯 번째 실습입니다. 배포 전 체크리스트를 확인합니다.
>
> 1. 모델 검증: 테스트 데이터 성능 확인
> 2. 파일 확인: model.pkl, scaler.pkl 있음?
> 3. 환경 확인: requirements.txt 작성
>
> 특히 scikit-learn 버전이 다르면 모델 로드가 실패할 수 있어요!
>
> ```bash
> pip freeze > requirements.txt
> ```

### 실습 7: 프로젝트 구조 [1.5min]

> 일곱 번째 실습입니다. 실무 프로젝트 구조를 확인합니다.
>
> ```
> ml_project/
> ├── models/
> │   └── model_v2.0.pkl
> ├── app/
> │   └── main.py
> ├── requirements.txt
> └── README.md
> ```
>
> models 폴더에 모델, app 폴더에 API 코드를 분리해요.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **joblib.dump()**: 모델 저장
> **joblib.load()**: 모델 불러오기
> **전처리기도 저장**: scaler 잊지 말기
> **파이프라인**: 하나로 묶어서 관리
> **requirements.txt**: 환경 고정
>
> 한 번 학습, 여러 번 사용!

#### 다음 차시 예고 [1min]

> 다음 26차시는 마지막 수업으로 **AI 프로젝트 종합 실습**을 진행합니다.
>
> 25차시 동안 배운 모든 내용을 종합해서 프로젝트를 완성하고,
> 과정 총정리와 후속 학습 안내를 드릴게요!

#### 마무리 [0.5min]

> 모델 저장과 배포 준비 방법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- joblib, scikit-learn 설치

### 주의사항
- .pkl 파일이 Git에 올라가지 않도록 .gitignore 설정
- 모델 파일은 용량이 크므로 별도 저장소(S3 등) 권장
- scikit-learn 버전 일치 중요

### 예상 질문
1. "pickle과 joblib 중 뭘 써야 하나요?"
   → ML 모델은 joblib 권장. NumPy 배열에 최적화

2. "모델 파일 용량이 너무 크면?"
   → joblib의 compress 옵션 사용 가능

3. "버전이 달라서 로드 안 되면?"
   → 같은 버전 환경에서 다시 학습 필요

4. "파이프라인과 개별 저장 중 뭐가 좋나요?"
   → 파이프라인이 실수 방지에 유리, 하지만 개별 저장이 유연성 높음
