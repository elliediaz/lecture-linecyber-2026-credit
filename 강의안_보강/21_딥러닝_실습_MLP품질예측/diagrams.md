# [21차시] 딥러닝 실습: MLP로 품질 예측 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["19차시<br>신경망 기초"]
    B["21차시<br>MLP 실습"]
    C["21차시<br>딥러닝 심화"]

    A --> B --> C

    B --> B1["Keras 기초"]
    B --> B2["MLP 구현"]
    B --> B3["학습 개선"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["21차시: MLP 실습"]

    A --> B["대주제 1<br>Keras 기초"]
    A --> C["대주제 2<br>MLP 구현"]
    A --> D["대주제 3<br>학습 개선"]

    B --> B1["Sequential, Dense<br>compile"]
    C --> C1["fit, predict<br>evaluate"]
    D --> D1["EarlyStopping<br>Dropout"]

    style A fill:#1e40af,color:#fff
```

## 3. NumPy vs Keras 비교

```mermaid
flowchart TD
    A["신경망 구현"]

    A --> B["NumPy"]
    B --> B1["순전파 직접 작성"]
    B --> B2["역전파 직접 계산"]
    B --> B3["50-100줄 코드"]

    A --> C["Keras"]
    C --> C1["model.predict()"]
    C --> C2["자동 미분"]
    C --> C3["5-10줄 코드"]

    style C fill:#dcfce7
```

## 4. Sequential 모델 구조

```mermaid
flowchart TD
    A["Sequential 모델"]

    A --> B["Dense(64, relu)"]
    B --> C["Dropout(0.3)"]
    C --> D["Dense(32, relu)"]
    D --> E["Dropout(0.2)"]
    E --> F["Dense(1, sigmoid)"]

    style A fill:#1e40af,color:#fff
```

## 5. Dense 층 파라미터

```mermaid
flowchart TD
    A["Dense 층"]

    A --> B["units"]
    B --> B1["노드(뉴런) 개수"]

    A --> C["activation"]
    C --> C1["활성화 함수"]

    A --> D["input_shape"]
    D --> D1["입력 형태<br>(첫 번째 층만)"]

    style A fill:#1e40af,color:#fff
```

## 6. 활성화 함수 선택

```mermaid
flowchart TD
    A["활성화 함수 선택"]

    A --> B["은닉층"]
    B --> B1["✅ relu"]

    A --> C["출력층 - 이진분류"]
    C --> C1["✅ sigmoid"]

    A --> D["출력층 - 다중분류"]
    D --> D1["✅ softmax"]

    A --> E["출력층 - 회귀"]
    E --> E1["None (linear)"]

    style A fill:#1e40af,color:#fff
```

## 7. 모델 컴파일 설정

```mermaid
flowchart TD
    A["model.compile()"]

    A --> B["optimizer"]
    B --> B1["adam (권장)"]
    B --> B2["sgd, rmsprop"]

    A --> C["loss"]
    C --> C1["binary_crossentropy"]
    C --> C2["categorical_crossentropy"]
    C --> C3["mse"]

    A --> D["metrics"]
    D --> D1["accuracy"]
    D --> D2["AUC"]

    style A fill:#1e40af,color:#fff
```

## 8. 손실 함수 선택

```mermaid
flowchart TD
    A["손실 함수"]

    A --> B["이진 분류"]
    B --> B1["binary_crossentropy"]

    A --> C["다중 분류"]
    C --> C1["categorical_crossentropy"]
    C --> C2["sparse_categorical<br>_crossentropy"]

    A --> D["회귀"]
    D --> D1["mse, mae"]

    style A fill:#1e40af,color:#fff
```

## 9. 옵티마이저 비교

```mermaid
flowchart TD
    A["옵티마이저"]

    A --> B["SGD"]
    B --> B1["기본 경사하강법"]

    A --> C["Adam"]
    C --> C1["✅ 가장 많이 사용"]
    C --> C2["적응적 학습률"]

    A --> D["RMSprop"]
    D --> D1["RNN에 적합"]

    style C fill:#dcfce7
```

## 10. fit 파라미터

```mermaid
flowchart TD
    A["model.fit()"]

    A --> B["epochs"]
    B --> B1["전체 데이터 반복 횟수"]

    A --> C["batch_size"]
    C --> C1["한 번에 학습할 샘플 수"]

    A --> D["validation_split"]
    D --> D1["검증 데이터 비율"]

    A --> E["callbacks"]
    E --> E1["EarlyStopping 등"]

    style A fill:#1e40af,color:#fff
```

## 11. 배치 크기 영향

```mermaid
flowchart TD
    A["배치 크기"]

    A --> B["작음 (16-32)"]
    B --> B1["잦은 업데이트"]
    B --> B2["노이즈 많음"]
    B --> B3["일반화 좋음"]

    A --> C["큼 (128-256)"]
    C --> C1["안정적 업데이트"]
    C --> C2["메모리 많이 필요"]
    C --> C3["빠른 학습"]

    style A fill:#1e40af,color:#fff
```

## 12. Keras 워크플로우

```mermaid
flowchart TD
    A["1. 데이터 준비"]
    B["2. 모델 생성<br>Sequential"]
    C["3. 컴파일<br>compile"]
    D["4. 학습<br>fit"]
    E["5. 평가<br>evaluate"]
    F["6. 예측<br>predict"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 13. 데이터 전처리

```mermaid
flowchart TD
    A["데이터 전처리"]

    A --> B["정규화"]
    B --> B1["StandardScaler"]
    B --> B2["평균 0, 표준편차 1"]

    A --> C["분할"]
    C --> C1["train_test_split"]
    C --> C2["80:20"]

    A --> D["중요"]
    D --> D1["신경망은<br>정규화 필수!"]

    style D fill:#fecaca
```

## 14. Dropout 동작

```mermaid
flowchart TD
    A["Dropout"]

    A --> B["학습 시"]
    B --> B1["무작위 노드 비활성화"]
    B --> B2["30% 끄기 (0.3)"]

    A --> C["예측 시"]
    C --> C1["모든 노드 활성화"]
    C --> C2["출력 × (1-rate)"]

    A --> D["효과"]
    D --> D1["과적합 방지"]
    D --> D2["일반화 향상"]

    style A fill:#1e40af,color:#fff
```

## 15. 학습 곡선 패턴

```mermaid
flowchart TD
    A["학습 곡선 해석"]

    A --> B["정상 학습"]
    B --> B1["Train↓, Val↓"]
    B --> B2["계속 학습"]

    A --> C["과적합"]
    C --> C1["Train↓, Val↑"]
    C --> C2["조기 종료 필요"]

    A --> D["학습 정체"]
    D --> D1["Train→, Val→"]
    D --> D2["학습률 조정"]

    style B fill:#dcfce7
    style C fill:#fecaca
```

## 16. EarlyStopping 동작

```mermaid
flowchart TD
    A["EarlyStopping"]

    A --> B["monitor"]
    B --> B1["val_loss 모니터링"]

    A --> C["patience"]
    C --> C1["N 에포크 개선 없으면 중단"]

    A --> D["restore_best_weights"]
    D --> D1["최적 가중치 복원"]

    style A fill:#1e40af,color:#fff
```

## 17. EarlyStopping 예시

```mermaid
flowchart TD
    A["Epoch 45: val_loss=0.289<br>(최저)"]
    B["Epoch 46: val_loss=0.291"]
    C["Epoch 47: val_loss=0.295"]
    D["..."]
    E["Epoch 55: val_loss=0.312<br>(patience=10 도달)"]
    F["학습 중단<br>Epoch 45 가중치 복원"]

    A --> B --> C --> D --> E --> F

    style A fill:#dcfce7
    style F fill:#1e40af,color:#fff
```

## 18. 콜백 종류

```mermaid
flowchart TD
    A["Keras 콜백"]

    A --> B["EarlyStopping"]
    B --> B1["조기 종료"]

    A --> C["ModelCheckpoint"]
    C --> C1["모델 저장"]

    A --> D["ReduceLROnPlateau"]
    D --> D1["학습률 감소"]

    A --> E["TensorBoard"]
    E --> E1["시각화"]

    style A fill:#1e40af,color:#fff
```

## 19. ModelCheckpoint

```mermaid
flowchart TD
    A["ModelCheckpoint"]

    A --> B["filepath"]
    B --> B1["저장 경로"]

    A --> C["monitor"]
    C --> C1["val_loss"]

    A --> D["save_best_only"]
    D --> D1["최적 모델만 저장"]

    style A fill:#1e40af,color:#fff
```

## 20. 모델 저장/로드

```mermaid
flowchart TD
    A["모델 저장"]

    A --> B["model.save()"]
    B --> B1["'model.keras'"]

    A --> C["load_model()"]
    C --> C1["저장된 모델 불러오기"]

    A --> D["활용"]
    D --> D1["학습 없이<br>바로 예측"]

    style A fill:#1e40af,color:#fff
```

## 21. BatchNormalization

```mermaid
flowchart TD
    A["BatchNormalization"]

    A --> B["역할"]
    B --> B1["각 층 출력 정규화"]

    A --> C["장점"]
    C --> C1["학습 안정화"]
    C --> C2["빠른 수렴"]
    C --> C3["높은 학습률 가능"]

    A --> D["위치"]
    D --> D1["Dense 뒤, Dropout 앞"]

    style A fill:#1e40af,color:#fff
```

## 22. L2 정규화

```mermaid
flowchart TD
    A["L2 정규화"]

    A --> B["kernel_regularizer"]
    B --> B1["l2(0.01)"]

    A --> C["효과"]
    C --> C1["가중치 크기 제한"]
    C --> C2["과적합 방지"]

    A --> D["손실 함수"]
    D --> D1["Loss + λ×Σw²"]

    style A fill:#1e40af,color:#fff
```

## 23. 과적합 해결책

```mermaid
flowchart TD
    A["과적합 해결"]

    A --> B["Dropout"]
    B --> B1["무작위 노드 비활성화"]

    A --> C["EarlyStopping"]
    C --> C1["적절한 시점 종료"]

    A --> D["L2 정규화"]
    D --> D1["가중치 크기 제한"]

    A --> E["데이터 증강"]
    E --> E1["데이터 늘리기"]

    style A fill:#1e40af,color:#fff
```

## 24. 학습률 조정

```mermaid
flowchart TD
    A["학습률"]

    A --> B["기본값"]
    B --> B1["Adam: 0.001"]

    A --> C["조정"]
    C --> C1["더 작게: 0.0001"]
    C --> C2["더 크게: 0.01"]

    A --> D["ReduceLROnPlateau"]
    D --> D1["자동 감소"]

    style A fill:#1e40af,color:#fff
```

## 25. ReduceLROnPlateau

```mermaid
flowchart TD
    A["ReduceLROnPlateau"]

    A --> B["monitor"]
    B --> B1["val_loss"]

    A --> C["factor"]
    C --> C1["0.5 (절반으로)"]

    A --> D["patience"]
    D --> D1["개선 없을 때 대기"]

    A --> E["min_lr"]
    E --> E1["최소 학습률"]

    style A fill:#1e40af,color:#fff
```

## 26. 하이퍼파라미터 튜닝

```mermaid
flowchart TD
    A["하이퍼파라미터"]

    A --> B["구조"]
    B --> B1["은닉층 수: 1-3"]
    B --> B2["노드 수: 32-256"]

    A --> C["정규화"]
    C --> C1["Dropout: 0.1-0.5"]

    A --> D["학습"]
    D --> D1["학습률: 0.0001-0.01"]
    D --> D2["배치: 16-128"]

    style A fill:#1e40af,color:#fff
```

## 27. 평가 지표

```mermaid
flowchart TD
    A["평가 지표"]

    A --> B["accuracy"]
    B --> B1["정확도"]

    A --> C["precision/recall"]
    C --> C1["정밀도/재현율"]

    A --> D["AUC"]
    D --> D1["ROC 곡선 아래 면적"]

    A --> E["F1-score"]
    E --> E1["정밀도×재현율 조화평균"]

    style A fill:#1e40af,color:#fff
```

## 28. 예측과 평가

```mermaid
flowchart TD
    A["모델 사용"]

    A --> B["predict"]
    B --> B1["확률 예측"]
    B --> B2["(y_prob > 0.5)"]

    A --> C["evaluate"]
    C --> C1["손실 + 지표 반환"]

    A --> D["classification_report"]
    D --> D1["상세 분류 보고서"]

    style A fill:#1e40af,color:#fff
```

## 29. ML vs DL 비교

```mermaid
flowchart TD
    A["모델 선택"]

    A --> B["RandomForest"]
    B --> B1["정형 데이터"]
    B --> B2["해석 필요"]
    B --> B3["데이터 적음"]

    A --> C["MLP (DL)"]
    C --> C1["복잡한 패턴"]
    C --> C2["대용량 데이터"]
    C --> C3["GPU 활용"]

    style A fill:#1e40af,color:#fff
```

## 30. 딥러닝 적합 상황

```mermaid
flowchart TD
    A["딥러닝이 유리"]

    A --> B["이미지/자연어"]
    A --> C["복잡한 비선형 패턴"]
    A --> D["대용량 데이터"]

    E["머신러닝이 유리"]
    E --> F["정형 데이터"]
    E --> G["해석 중요"]
    E --> H["데이터 적음"]

    style A fill:#1e40af,color:#fff
    style E fill:#059669,color:#fff
```

## 31. 실습 흐름

```mermaid
flowchart TD
    A["1. 데이터 생성"]
    B["2. 전처리<br>(정규화, 분할)"]
    C["3. 모델 생성<br>(Sequential)"]
    D["4. 컴파일"]
    E["5. 학습 (fit)"]
    F["6. 평가 (evaluate)"]
    G["7. 시각화"]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#dbeafe
    style G fill:#dcfce7
```

## 32. 완전한 코드 구조

```mermaid
flowchart TD
    A["임포트"]
    B["데이터 준비"]
    C["Sequential 모델"]
    D["Dense + Dropout"]
    E["compile"]
    F["callbacks 설정"]
    G["fit"]
    H["evaluate"]
    I["save"]

    A --> B --> C --> D --> E --> F --> G --> H --> I

    style A fill:#dbeafe
    style I fill:#dcfce7
```

## 33. 파라미터 수 계산

```mermaid
flowchart TD
    A["파라미터 수"]

    A --> B["입력(4) → Dense(64)"]
    B --> B1["4×64 + 64 = 320"]

    A --> C["Dense(64) → Dense(32)"]
    C --> C1["64×32 + 32 = 2,080"]

    A --> D["Dense(32) → Dense(1)"]
    D --> D1["32×1 + 1 = 33"]

    A --> E["총계: 2,433"]

    style A fill:#1e40af,color:#fff
```

## 34. 핵심 정리

```mermaid
flowchart TD
    A["21차시 핵심"]

    A --> B["Keras 기초"]
    B --> B1["Sequential<br>Dense, Dropout"]

    A --> C["학습 과정"]
    C --> C1["compile → fit<br>→ evaluate"]

    A --> D["개선 기법"]
    D --> D1["EarlyStopping<br>BatchNormalization"]

    style A fill:#1e40af,color:#fff
```

## 35. 다음 차시 연결

```mermaid
flowchart LR
    A["21차시<br>MLP 실습"]
    B["21차시<br>딥러닝 심화"]

    A --> B

    A --> A1["Dense 층"]
    B --> B1["CNN (이미지)"]
    B --> B2["RNN (시계열)"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```
