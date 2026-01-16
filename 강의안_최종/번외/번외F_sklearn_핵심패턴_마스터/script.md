# 번외F: sklearn 핵심 패턴 마스터 - 강사 스크립트

## 개요
- **총 시간**: 30분
- **구성**: 이론 10분 + 실습 20분
- **난이도**: 초급~중급
- **데이터셋**: Iris (sklearn), Titanic (seaborn)

---

## Part 1: sklearn 설계 철학 (4분)

### 슬라이드 1-6: 도입 (2분)

> "오늘은 sklearn을 '사용'하는 것을 넘어 '이해'하는 시간입니다."

> "예제를 복사 붙여넣기 하는 것과, 왜 그렇게 하는지 아는 것은 다릅니다."

> "sklearn의 가장 큰 장점은 **일관성**입니다. 모든 객체가 같은 패턴을 따릅니다."

### 슬라이드 7-9: fit/predict/transform (2분)

> "sklearn의 모든 객체는 fit() 메서드를 가집니다. 이것이 '학습'입니다."

> "모델은 predict()로 예측하고, 전처리기는 transform()으로 변환합니다."

> "중요한 건 둘 다 fit() 이후에만 사용할 수 있다는 것입니다."

---

## Part 2: fit과 transform의 구분 (6분)

### 슬라이드 10-13: fit()의 역할 (3분)

> "fit()이 하는 일을 구체적으로 봅시다."

> "LinearRegression은 계수를 학습하고, StandardScaler는 평균과 표준편차를 계산합니다."

> "OneHotEncoder는 어떤 범주가 있는지 목록을 학습합니다."

### 슬라이드 14-17: 왜 테스트에 fit하면 안 되는가? (3분)

> "가장 중요한 규칙입니다. **fit()은 학습 데이터에서만!**"

> "테스트 데이터에 fit()을 하면 어떻게 될까요?"

> "학습 데이터와 테스트 데이터의 기준이 달라집니다."

> "예를 들어 스케일링에서, 학습 데이터 평균이 10인데 테스트 평균이 15라면?"

> "각각 다른 기준으로 정규화되어 모델이 제대로 예측할 수 없습니다."

---

## Part 3: fit_transform과 올바른 사용법 (5분)

### 슬라이드 18-20: fit_transform() (2분)

> "fit_transform()은 fit()과 transform()을 한 번에 하는 편의 메서드입니다."

> "**학습 데이터에서만 사용**하세요. 테스트에는 transform()만 씁니다."

```python
X_train_scaled = scaler.fit_transform(X_train)  # OK
X_test_scaled = scaler.transform(X_test)        # OK
```

### 슬라이드 21-23: 전체 흐름 (3분)

> "전체 흐름을 정리해봅시다."

> "1. 스케일러를 학습 데이터로 fit_transform"
> "2. 테스트 데이터는 transform만"
> "3. 모델을 스케일된 학습 데이터로 fit"
> "4. 스케일된 테스트 데이터로 predict"

---

## Part 4: Pipeline (8분)

### 슬라이드 24-27: Pipeline 소개 (3분)

> "지금까지 본 것처럼 전처리와 모델을 따로 관리하면 복잡합니다."

> "Pipeline은 이 모든 단계를 **하나의 객체**로 묶어줍니다."

> "장점이 많습니다. 코드가 간결해지고, 순서가 보장되고, 무엇보다 fit/transform이 자동 관리됩니다."

### 슬라이드 28-30: Pipeline 사용법 (3분)

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)  # 내부에서 자동 처리
pipeline.predict(X_test)         # 변환 + 예측 자동
```

> "fit()을 호출하면 내부적으로 스케일러는 fit_transform, 모델은 fit을 합니다."

> "predict()를 호출하면 스케일러는 transform, 모델은 predict를 합니다."

### 슬라이드 31-33: make_pipeline (2분)

> "make_pipeline을 쓰면 이름을 자동으로 지어줍니다."

> "named_steps로 각 단계에 접근할 수 있습니다."

---

## Part 5: ColumnTransformer (7분)

### 슬라이드 34-37: ColumnTransformer 소개 (3분)

> "실제 데이터는 수치형과 범주형이 섞여 있습니다."

> "수치형은 스케일링, 범주형은 인코딩이 필요합니다."

> "ColumnTransformer는 **컬럼별로 다른 전처리**를 적용합니다."

### 슬라이드 38-42: ColumnTransformer 실습 (4분)

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'fare']),
    ('cat', OneHotEncoder(), ['sex', 'embarked'])
])
```

> "이름, 변환기, 컬럼 리스트 순서입니다."

> "이제 Pipeline과 결합해서 완전한 워크플로우를 만들 수 있습니다."

---

## 실습 안내 (10분)

> "이제 직접 해봅시다. code.py를 열고 실습을 진행합니다."

### 실습 체크포인트

1. 기본 패턴 확인: fit/transform 구분 (3분)
2. Pipeline 구성 및 사용 (3분)
3. ColumnTransformer + Pipeline 조합 (3분)
4. 모델 저장 및 로드 (1분)

---

## 마무리

> "sklearn의 핵심은 **일관성**입니다."

> "fit()은 학습, transform()은 변환, predict()는 예측."

> "이 세 가지만 기억하면 어떤 sklearn 객체도 사용할 수 있습니다."

> "Pipeline을 쓰면 실수 없이 올바른 순서로 처리됩니다."

---

## 예상 Q&A

**Q1**: "fit_transform()은 왜 있나요? fit()과 transform()을 따로 하면 되잖아요."
**A1**: "편의를 위한 것입니다. 학습 데이터에서는 두 단계를 항상 연속으로 하니까요. 단, 테스트에서는 절대 쓰면 안 됩니다."

**Q2**: "Pipeline 없이 해도 되나요?"
**A2**: "됩니다. 하지만 실수할 가능성이 높아집니다. 특히 모델 저장/로드 시 전처리기를 빠뜨리는 경우가 많습니다."

**Q3**: "ColumnTransformer에서 처리 안 하는 컬럼은요?"
**A3**: "remainder='passthrough' 옵션으로 그대로 유지하거나, remainder='drop'으로 제거합니다. 기본은 drop입니다."

**Q4**: "GridSearchCV에서 Pipeline 파라미터는 어떻게 지정하나요?"
**A4**: "'단계이름__파라미터' 형식입니다. 예: 'classifier__n_estimators'"

---

## 핵심 메시지

```
fit()은 학습에서만, transform()은 어디서든!
Pipeline으로 묶으면 자동 관리!
```
