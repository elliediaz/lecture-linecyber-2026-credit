# [25차시] 모델 저장과 실무 배포 준비 - 다이어그램

## 1. 왜 모델을 저장하나?

```mermaid
flowchart LR
    A["데이터 준비"]
    B["모델 학습<br>(10분~수시간)"]
    C["예측"]

    A --> B --> C

    D["매번 학습?<br>시간 낭비!"]

    B -.-> D

    style D fill:#fee2e2
```

## 2. 모델 저장 워크플로우

```mermaid
flowchart TD
    A["모델 학습<br>(개발 환경)"]
    B["모델 저장<br>(model.pkl)"]
    C["모델 로드<br>(서버 환경)"]
    D["예측 수행"]

    A --> B --> C --> D

    E["학습: 1회"]
    F["예측: 무한회"]

    B --> E
    D --> F

    style E fill:#dbeafe
    style F fill:#dcfce7
```

## 3. joblib 사용법

```mermaid
flowchart LR
    subgraph 저장["저장"]
        A["joblib.dump(model, 'model.pkl')"]
    end

    subgraph 불러오기["불러오기"]
        B["model = joblib.load('model.pkl')"]
    end

    저장 --> 불러오기
```

## 4. 저장 코드 흐름

```mermaid
flowchart TD
    A["from sklearn.ensemble import RandomForestClassifier"]
    B["import joblib"]
    C["model.fit(X_train, y_train)"]
    D["joblib.dump(model, 'model.pkl')"]
    E["저장 완료!"]

    A --> B --> C --> D --> E
```

## 5. 전처리기도 함께 저장

```mermaid
flowchart TD
    subgraph 학습["학습 시"]
        A["scaler.fit_transform()"]
        B["model.fit()"]
    end

    subgraph 저장["저장"]
        C["joblib.dump(scaler, 'scaler.pkl')"]
        D["joblib.dump(model, 'model.pkl')"]
    end

    학습 --> 저장
```

## 6. 불러와서 예측

```mermaid
flowchart TD
    A["scaler = joblib.load('scaler.pkl')"]
    B["model = joblib.load('model.pkl')"]
    C["new_scaled = scaler.transform(new_data)"]
    D["prediction = model.predict(new_scaled)"]

    A --> C
    B --> D
    C --> D
```

## 7. 파이프라인 장점

```mermaid
flowchart LR
    subgraph 개별["개별 저장"]
        A1["scaler.pkl"]
        A2["model.pkl"]
    end

    subgraph 파이프라인["파이프라인"]
        B1["pipeline.pkl"]
    end

    C["하나로 관리<br>실수 방지"]

    파이프라인 --> C

    style C fill:#dcfce7
```

## 8. 파이프라인 구조

```mermaid
flowchart TD
    A["Pipeline"]

    A --> B["('scaler', StandardScaler())"]
    A --> C["('model', RandomForestClassifier())"]

    D["pipeline.fit(X_train, y_train)"]
    E["joblib.dump(pipeline, 'pipeline.pkl')"]

    B & C --> D --> E
```

## 9. pickle vs joblib

```mermaid
flowchart TD
    subgraph pickle["pickle"]
        A1["Python 내장"]
        A2["일반 객체"]
        A3["대용량: 느림"]
    end

    subgraph joblib["joblib"]
        B1["설치 필요"]
        B2["ML 모델 최적화"]
        B3["대용량: 빠름"]
    end

    C["ML 모델은<br>joblib 권장!"]

    joblib --> C

    style C fill:#dbeafe
```

## 10. 메타데이터 저장

```mermaid
flowchart TD
    A["model_package = {"]
    B["'model': model,"]
    C["'scaler': scaler,"]
    D["'version': '1.0.0',"]
    E["'accuracy': 0.92,"]
    F["'trained_date': '...'"]
    G["}"]

    H["joblib.dump(model_package, 'package.pkl')"]

    A --> B --> C --> D --> E --> F --> G --> H
```

## 11. 버전 관리 파일명

```mermaid
flowchart TD
    A["models/"]

    A --> B["quality_model_v1.0_20260101.pkl"]
    A --> C["quality_model_v1.1_20260115.pkl"]
    A --> D["quality_model_v2.0_20260201.pkl"]
    A --> E["scaler_v2.0_20260201.pkl"]
```

## 12. 배포 전 체크리스트

```mermaid
flowchart TD
    A["배포 전 확인"]

    A --> B["1. 모델 검증<br>테스트 데이터 성능"]
    A --> C["2. 파일 확인<br>model.pkl, scaler.pkl"]
    A --> D["3. 환경 확인<br>Python, sklearn 버전"]

    E["배포 준비 완료!"]

    B & C & D --> E

    style E fill:#dcfce7
```

## 13. requirements.txt

```mermaid
flowchart LR
    A["pip freeze"]
    B[">"]
    C["requirements.txt"]

    A --> B --> C

    D["scikit-learn==1.3.0<br>numpy==1.24.0<br>pandas==2.0.0"]

    C --> D
```

## 14. 프로젝트 폴더 구조

```mermaid
flowchart TD
    A["ml_project/"]

    A --> B["models/"]
    A --> C["app/"]
    A --> D["requirements.txt"]
    A --> E["Dockerfile"]

    B --> B1["model_v2.0.pkl"]
    B --> B2["scaler_v2.0.pkl"]

    C --> C1["main.py"]
```

## 15. FastAPI 연동

```mermaid
flowchart TD
    A["앱 시작"]
    B["model = joblib.load()"]
    C["scaler = joblib.load()"]
    D["@app.post('/predict')"]
    E["scaler.transform()"]
    F["model.predict()"]
    G["응답 반환"]

    A --> B & C
    D --> E --> F --> G
```

## 16. 배포 아키텍처

```mermaid
flowchart LR
    subgraph 개발["개발 환경"]
        A["Jupyter Notebook"]
        B["모델 학습"]
        C["model.pkl 저장"]
    end

    subgraph 서버["서버 환경"]
        D["model.pkl 로드"]
        E["FastAPI"]
        F["예측 API"]
    end

    개발 --> 서버
```

## 17. 버전 불일치 문제

```mermaid
flowchart TD
    A["학습 환경<br>sklearn 1.3.0"]
    B["배포 환경<br>sklearn 1.2.0"]
    C["로드 실패!"]

    A --> B --> C

    D["requirements.txt로<br>버전 고정!"]

    C -.-> D

    style C fill:#fee2e2
    style D fill:#dcfce7
```

## 18. 강의 구조

```mermaid
gantt
    title 25차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    왜 저장해야 하나?         :a2, after a1, 2m
    joblib 사용법            :a3, after a2, 2m
    전처리기도 저장          :a4, after a3, 2m
    버전 관리/배포 준비       :a5, after a4, 2m

    section 실습편
    실습 소개               :b1, after a5, 1.5m
    모델 학습               :b2, after b1, 2m
    joblib 저장             :b3, after b2, 2m
    모델 불러오기            :b4, after b3, 2m
    파이프라인              :b5, after b4, 2m
    메타데이터 저장          :b6, after b5, 2m
    배포 체크리스트          :b7, after b6, 2m
    프로젝트 구조           :b8, after b7, 1.5m

    section 정리
    핵심 요약               :c1, after b8, 1.5m
    다음 차시 예고           :c2, after c1, 1m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((모델 저장<br>배포 준비))
    joblib
      dump 저장
      load 불러오기
      pkl 파일
    전처리기
      scaler도 저장
      Pipeline
    버전 관리
      파일명 규칙
      메타데이터
    배포 준비
      requirements.txt
      체크리스트
      폴더 구조
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>모델 저장<br>(배포 준비)"]
    B["다음<br>종합 실습<br>(프로젝트)"]
    C["이후<br>실무 적용<br>(현장)"]

    A --> B --> C
```
