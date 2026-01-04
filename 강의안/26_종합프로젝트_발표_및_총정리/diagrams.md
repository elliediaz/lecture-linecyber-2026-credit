# [26차시] AI 프로젝트 종합 실습 - 다이어그램

## 1. ML 프로젝트 전체 워크플로우

```mermaid
flowchart TD
    A["1. 문제 정의"]
    B["2. 데이터 수집"]
    C["3. EDA/전처리"]
    D["4. 모델 학습"]
    E["5. 모델 평가"]
    F["6. 모델 해석"]
    G["7. 모델 저장"]
    H["8. API 서비스"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#dbeafe
    style H fill:#dcfce7
```

## 2. 종합 실습 목표

```mermaid
flowchart LR
    A["센서 데이터<br>(온도, 습도, 속도, 압력)"]
    B["ML 모델<br>(RandomForest)"]
    C["예측 결과<br>(정상/불량)"]

    A --> B --> C

    style B fill:#dbeafe
```

## 3. 문제 정의 단계

```mermaid
flowchart TD
    A["문제 정의"]

    A --> B["목표: 불량 예측"]
    A --> C["입력: 센서 4종"]
    A --> D["출력: 정상/불량"]
    A --> E["활용: 사전 감지"]

    style A fill:#dbeafe
```

## 4. 데이터 확인 체크리스트

```mermaid
flowchart TD
    A["데이터 확인"]

    A --> B["df.shape<br>데이터 크기"]
    A --> C["df.info()<br>컬럼 타입"]
    A --> D["df.describe()<br>기술통계"]
    A --> E["df.head()<br>샘플 확인"]
```

## 5. EDA 및 전처리

```mermaid
flowchart TD
    A["탐색적 분석"]

    A --> B["결측치 확인<br>isnull().sum()"]
    A --> C["타겟 분포<br>value_counts()"]
    A --> D["상관관계<br>corr()"]

    E["전처리 완료"]

    B & C & D --> E

    style E fill:#dcfce7
```

## 6. 모델 학습 파이프라인

```mermaid
flowchart TD
    A["원본 데이터"]
    B["train_test_split"]
    C["StandardScaler"]
    D["RandomForestClassifier"]
    E["학습 완료"]

    A --> B --> C --> D --> E

    style E fill:#dcfce7
```

## 7. 모델 평가 지표

```mermaid
flowchart LR
    A["평가 지표"]

    A --> B["정확도<br>Accuracy"]
    A --> C["정밀도<br>Precision"]
    A --> D["재현율<br>Recall"]
    A --> E["F1 Score"]

    style A fill:#dbeafe
```

## 8. 특성 중요도 분석

```mermaid
flowchart TD
    A["Feature Importance"]

    A --> B["온도: 0.35"]
    A --> C["습도: 0.28"]
    A --> D["속도: 0.22"]
    A --> E["압력: 0.15"]

    F["온도가 가장 중요!"]

    B --> F

    style F fill:#dcfce7
```

## 9. 모델 저장 구조

```mermaid
flowchart TD
    A["model_package = {"]
    B["'model': model,"]
    C["'scaler': scaler,"]
    D["'version': '1.0.0',"]
    E["'accuracy': 0.92"]
    F["}"]

    G["joblib.dump()"]
    H["package.pkl"]

    A --> B --> C --> D --> E --> F --> G --> H
```

## 10. FastAPI 서비스

```mermaid
flowchart TD
    A["클라이언트 요청"]
    B["FastAPI"]
    C["scaler.transform()"]
    D["model.predict()"]
    E["응답: 정상/불량"]

    A --> B --> C --> D --> E
```

## 11. 8단계 워크플로우 완성

```mermaid
flowchart LR
    subgraph 분석["데이터 분석"]
        A["문제 정의"]
        B["데이터 수집"]
        C["EDA/전처리"]
    end

    subgraph 모델링["모델링"]
        D["모델 학습"]
        E["평가"]
        F["해석"]
    end

    subgraph 배포["배포"]
        G["저장"]
        H["API"]
    end

    분석 --> 모델링 --> 배포
```

## 12. Part I 요약 (1-3차시)

```mermaid
mindmap
  root((Part I<br>환경 구축))
    1차시
      AI 윤리
      데이터 보호
    2차시
      Python 기초
      개발 환경
    3차시
      pandas
      데이터 다루기
```

## 13. Part II 요약 (4-9차시)

```mermaid
mindmap
  root((Part II<br>데이터 분석))
    통계
      기술통계
      확률분포
      가설검정
    전처리
      결측치
      이상치
      스케일링
    시각화
      matplotlib
      seaborn
    EDA
      탐색 분석
      상관관계
```

## 14. Part III 요약 (10-19차시)

```mermaid
mindmap
  root((Part III<br>모델링))
    분류
      의사결정나무
      랜덤포레스트
    회귀
      선형회귀
      다항회귀
    평가
      교차검증
      하이퍼파라미터
    시계열
      날짜 처리
      예측 모델
    딥러닝
      신경망
      MLP
```

## 15. Part IV 요약 (20-26차시)

```mermaid
mindmap
  root((Part IV<br>서비스화))
    API
      requests
      REST API
    LLM
      OpenAI API
      프롬프트
    웹앱
      Streamlit
    서비스
      FastAPI
      uvicorn
    해석
      Feature Importance
      SHAP
    배포
      joblib
      버전 관리
    종합
      프로젝트
      총정리
```

## 16. 핵심 라이브러리 정리

```mermaid
flowchart TD
    subgraph 데이터["데이터 처리"]
        A["pandas"]
        B["numpy"]
    end

    subgraph 시각화["시각화"]
        C["matplotlib"]
        D["seaborn"]
    end

    subgraph ML["머신러닝"]
        E["scikit-learn"]
    end

    subgraph DL["딥러닝"]
        F["keras"]
        G["tensorflow"]
    end

    subgraph 서비스["서비스"]
        H["streamlit"]
        I["fastapi"]
    end

    subgraph 저장["저장"]
        J["joblib"]
    end
```

## 17. 후속 학습 로드맵

```mermaid
flowchart LR
    A["이번 과정<br>(기초)"]
    B["심화 학습<br>(딥러닝)"]
    C["실무 적용<br>(현장)"]
    D["전문가<br>(MLOps)"]

    A --> B --> C --> D
```

## 18. 심화 학습 방향

```mermaid
flowchart TD
    A["후속 학습"]

    A --> B["딥러닝<br>CNN, RNN, Transformer"]
    A --> C["자연어처리<br>텍스트, 챗봇"]
    A --> D["컴퓨터 비전<br>이미지, 객체 탐지"]
    A --> E["MLOps<br>배포, 모니터링"]
```

## 19. 수료 조건

```mermaid
flowchart TD
    A["수료 조건"]

    A --> B["출석률<br>80% 이상"]
    A --> C["과제 제출<br>모든 과제"]
    A --> D["최종 프로젝트<br>발표 완료"]

    E["수료증 발급!"]

    B & C & D --> E

    style E fill:#dcfce7
```

## 20. 과정 전체 여정

```mermaid
flowchart LR
    A["시작<br>Python 기초"]
    B["성장<br>데이터 분석"]
    C["심화<br>모델링"]
    D["완성<br>서비스화"]
    E["졸업<br>AI 엔지니어"]

    A --> B --> C --> D --> E

    style A fill:#dbeafe
    style E fill:#dcfce7
```

