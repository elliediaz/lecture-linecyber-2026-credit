# [14차시] 모델 평가와 반복 검증 - 강사 스크립트

## 강의 정보
- **차시**: 14차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 14차시를 시작하겠습니다.
>
> 지금까지 분류 모델과 회귀 모델을 배웠습니다. 모델을 만들고 score()로 성능을 확인했죠.
>
> 그런데 이 평가 방법이 **충분히 신뢰할 수 있을까요?** 오늘은 더 정확한 평가 방법을 배웁니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 교차검증의 개념과 필요성을 이해합니다.
> 둘째, 과대적합과 과소적합을 진단합니다.
> 셋째, 혼동행렬, 정밀도, 재현율을 해석합니다.

---

### 핵심 내용 (8분)

#### 기존 방법의 문제 [1.5분]

> 지금까지는 train_test_split으로 한 번 나눠서 평가했죠.
>
> 문제가 있어요. 운 좋게 쉬운 테스트 데이터가 뽑힐 수 있고, 운 나쁘게 어려운 데이터가 뽑힐 수도 있어요.
>
> 85%라는 점수가 나왔는데, 진짜 85%인지 아니면 운이 좋았던 건지 알기 어렵습니다.

#### 교차검증 개념 [2min]

> **교차검증(Cross Validation)**은 데이터를 여러 번 나눠서 평가하고 평균을 계산하는 방법입니다.
>
> **K-Fold 교차검증**이 가장 많이 사용돼요. 데이터를 K개로 나누고, 각각을 한 번씩 테스트 데이터로 사용합니다.
>
> 5-Fold라면: 1회차에서 첫 번째가 테스트, 2회차에서 두 번째가 테스트... 이렇게 5번 평가하고 평균을 냅니다.
>
> 결과: "평균 84% (±1.4%)" 이렇게 나오면 더 신뢰할 수 있죠.
>
> ```python
> from sklearn.model_selection import cross_val_score
> scores = cross_val_score(model, X, y, cv=5)
> print(f"평균: {scores.mean():.3f}")
> ```

#### 과대적합과 과소적합 [2min]

> 모델이 복잡하면 학습 데이터를 잘 맞추지만 새 데이터에서는 성능이 떨어질 수 있어요. 이것을 **과대적합(Overfitting)**이라고 합니다.
>
> 반대로 모델이 너무 단순하면 학습 데이터조차 잘 못 맞춰요. 이것은 **과소적합(Underfitting)**입니다.
>
> 진단 방법은 간단해요. 학습 정확도와 테스트 정확도를 비교합니다.
>
> - **과소적합**: 둘 다 낮음 (학습 70%, 테스트 68%)
> - **과대적합**: 학습만 높음 (학습 98%, 테스트 75%)
> - **적절한 모델**: 비슷하게 높음 (학습 86%, 테스트 84%)

#### 혼동행렬 [1.5min]

> 정확도만으로는 부족할 때가 있어요. 예를 들어 불량률이 5%인 데이터에서 "모든 것을 정상"이라고 예측해도 정확도 95%가 나옵니다.
>
> **혼동행렬(Confusion Matrix)**은 더 자세한 정보를 줍니다.
>
> - TN: 정상을 정상으로 예측 (맞음)
> - FP: 정상을 불량으로 예측 (틀림)
> - FN: 불량을 정상으로 예측 (틀림)
> - TP: 불량을 불량으로 예측 (맞음)

#### 정밀도와 재현율 [1min]

> **정밀도(Precision)**: 불량이라고 예측한 것 중 진짜 불량 비율입니다.
>
> **재현율(Recall)**: 실제 불량 중 잡아낸 비율이에요.
>
> 제조 현장에서는 보통 **재현율**이 중요합니다. 불량품을 놓치면 고객에게 가니까요.
>
> 암 진단도 마찬가지예요. 환자를 놓치면 안 되니까 재현율이 중요합니다.

---

## 실습편 (15-20분)

### 실습 소개 [2분]

> 이제 실습 시간입니다. 교차검증과 분류 평가를 직접 해봅니다.
>
> **실습 목표**입니다.
> 1. cross_val_score로 교차검증합니다.
> 2. 학습/테스트 점수로 과대적합을 진단합니다.
> 3. 혼동행렬과 분류 리포트를 분석합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from sklearn.model_selection import cross_val_score
> from sklearn.metrics import confusion_matrix, classification_report
> ```

### 실습 1: 데이터 준비 [2min]

> 첫 번째 실습입니다. 제조 데이터를 생성합니다.
>
> 불량 예측 문제를 다시 다룹니다. 온도, 습도, 속도로 불량 여부를 예측해요.

### 실습 2: 교차검증 [2min]

> 두 번째 실습입니다. cross_val_score로 교차검증합니다.
>
> ```python
> from sklearn.model_selection import cross_val_score
>
> model = RandomForestClassifier(n_estimators=100, random_state=42)
> scores = cross_val_score(model, X, y, cv=5)
>
> print(f"각 Fold 점수: {scores}")
> print(f"평균: {scores.mean():.3f}")
> print(f"표준편차: {scores.std():.3f}")
> ```
>
> 표준편차가 작으면 모델이 안정적이라는 뜻이에요.

### 실습 3: 과대적합 진단 [2min]

> 세 번째 실습입니다. 학습과 테스트 점수를 비교합니다.
>
> ```python
> model.fit(X_train, y_train)
> train_score = model.score(X_train, y_train)
> test_score = model.score(X_test, y_test)
>
> print(f"학습: {train_score:.1%}")
> print(f"테스트: {test_score:.1%}")
> ```
>
> 차이가 10%p 이상이면 과대적합을 의심하세요.

### 실습 4: 혼동행렬 [2min]

> 네 번째 실습입니다. 혼동행렬을 만들고 시각화합니다.
>
> ```python
> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>
> y_pred = model.predict(X_test)
> cm = confusion_matrix(y_test, y_pred)
>
> disp = ConfusionMatrixDisplay(cm, display_labels=['정상', '불량'])
> disp.plot(cmap='Blues')
> ```
>
> TN, FP, FN, TP가 각각 몇 개인지 한눈에 볼 수 있어요.

### 실습 5: 정밀도, 재현율, F1 [2min]

> 다섯 번째 실습입니다. 상세 평가 지표를 계산합니다.
>
> ```python
> from sklearn.metrics import precision_score, recall_score, f1_score
>
> precision = precision_score(y_test, y_pred)
> recall = recall_score(y_test, y_pred)
> f1 = f1_score(y_test, y_pred)
>
> print(f"정밀도: {precision:.3f}")
> print(f"재현율: {recall:.3f}")
> print(f"F1 Score: {f1:.3f}")
> ```
>
> 제조 현장에서는 재현율이 높은지 확인하세요.

### 실습 6: 분류 리포트 [2min]

> 여섯 번째 실습입니다. classification_report로 종합 평가합니다.
>
> ```python
> from sklearn.metrics import classification_report
>
> print(classification_report(y_test, y_pred,
>                             target_names=['정상', '불량']))
> ```
>
> precision, recall, f1-score를 한 번에 볼 수 있어서 편리해요.

### 실습 7: 모델 비교 [2min]

> 마지막 실습입니다. 여러 모델을 교차검증으로 비교합니다.
>
> ```python
> models = {
>     '의사결정나무': DecisionTreeClassifier(max_depth=5),
>     '랜덤포레스트': RandomForestClassifier(n_estimators=100)
> }
>
> for name, model in models.items():
>     scores = cross_val_score(model, X, y, cv=5)
>     print(f"{name}: {scores.mean():.3f} (±{scores.std():.3f})")
> ```
>
> 여러 모델을 공정하게 비교할 수 있어요.

---

### 정리 (3분)

#### 핵심 요약 [1.5분]

> 오늘 배운 내용을 정리하겠습니다.
>
> **교차검증**은 여러 번 평가해서 신뢰성을 확보합니다. cross_val_score로 간단하게 할 수 있어요.
>
> **과대적합**은 학습 점수만 높고 테스트 점수가 낮을 때입니다. 모델을 단순화해서 해결해요.
>
> **혼동행렬**로 분류 결과를 상세하게 볼 수 있습니다.
>
> **정밀도와 재현율**은 상황에 따라 중요도가 달라요. 제조 현장에서는 보통 재현율이 중요합니다.

#### 다음 차시 예고 [1min]

> 다음 15차시에서는 **모델 설정값 최적화**를 배웁니다.
>
> max_depth, n_estimators 같은 값을 어떻게 설정하면 좋을까요? GridSearchCV로 최적의 조합을 자동으로 찾아봅니다.

#### 마무리 [0.5min]

> 모델을 제대로 평가하는 방법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)

### 주의사항
- 교차검증의 필요성 충분히 설명
- 과대적합 진단 방법 강조
- 정밀도/재현율 차이 명확히 설명

### 예상 질문
1. "cv는 몇으로 하면 되나요?"
   → 보통 5 또는 10. 데이터가 적으면 5, 많으면 10

2. "정밀도와 재현율 중 뭐가 더 중요한가요?"
   → 도메인에 따라 다름. 제조 불량은 보통 재현율 중요

3. "교차검증은 항상 해야 하나요?"
   → 모델 선택/비교할 때 필수. 최종 모델은 전체 데이터로 학습

4. "혼동행렬에서 FN이 뭔가요?"
   → False Negative. 실제 불량을 정상으로 예측 (놓친 것)
