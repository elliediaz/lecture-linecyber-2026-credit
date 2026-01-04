# [24차시] 모델 해석과 변수별 영향력 분석 - 강사 스크립트

## 강의 정보
- **차시**: 24차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 24차시를 시작하겠습니다.
>
> 지난 시간에 FastAPI로 ML 모델을 REST API로 서비스하는 방법을 배웠습니다.
>
> 오늘은 **모델 해석**을 배웁니다. 모델이 왜 그렇게 예측했는지 설명하는 방법이에요!

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 모델 해석의 필요성을 이해합니다.
> 둘째, 특성 중요도(Feature Importance)를 분석합니다.
> 셋째, Permutation Importance를 활용합니다.

---

### 핵심 내용 (8분)

#### 왜 모델 해석이 필요한가? [2분]

> 머신러닝 모델, 특히 복잡한 모델은 **블랙박스**처럼 작동해요.
>
> 입력을 넣으면 결과가 나오는데, 그 과정을 알 수 없죠.
>
> "왜 이 제품이 불량인가요?"
> "모델이 그렇다고 했으니까요..." 이렇게 답하면 안 되겠죠?
>
> 모델 해석이 필요한 이유는 세 가지입니다.
>
> 첫째, **신뢰**. 현장 담당자가 모델을 믿으려면 이유를 알아야 해요.
> 둘째, **디버깅**. 모델이 이상한 패턴을 학습하진 않았는지 확인해야 해요.
> 셋째, **규제**. 금융이나 의료 분야에서는 설명 의무가 있어요.

#### 특성 중요도 (Feature Importance) [2min]

> **특성 중요도**는 각 변수가 예측에 얼마나 기여하는지 보여줘요.
>
> ```python
> from sklearn.ensemble import RandomForestClassifier
>
> model = RandomForestClassifier()
> model.fit(X_train, y_train)
>
> print(model.feature_importances_)
> # [0.35, 0.30, 0.25, 0.10]
> ```
>
> 랜덤포레스트, 의사결정트리 같은 트리 기반 모델은 학습하면서 자동으로 계산해줘요.
>
> 숫자는 비율이고, 합치면 1이 됩니다.

#### 제조 데이터 해석 예시 [1.5min]

> 품질 예측 모델에서 특성 중요도를 분석했더니:
>
> - 온도: 35%
> - 습도: 30%
> - 속도: 25%
> - 압력: 10%
>
> **온도**가 품질에 가장 큰 영향을 미친다는 뜻이에요.
>
> 불량률을 줄이려면 온도 관리를 최우선으로 해야 하는 거죠!

#### Permutation Importance [1.5min]

> Feature Importance는 학습 중에 계산되는데, 편향이 있을 수 있어요.
>
> **Permutation Importance**는 더 신뢰할 수 있는 방법이에요.
>
> 원리는 간단해요:
>
> 1. 특정 특성의 값을 무작위로 섞어요
> 2. 예측 성능이 얼마나 떨어지는지 봐요
> 3. 많이 떨어지면? 그 특성이 중요한 거예요!
>
> 실무에서는 먼저 Feature Importance로 빠르게 확인하고,
> 중요한 분석이나 보고서에는 Permutation Importance를 사용해요.

#### 이론 정리 [1min]

> 정리하면, 모델 해석은 예측 이유를 설명하는 것이에요.
> Feature Importance는 model.feature_importances_로 바로 확인하고,
> Permutation Importance는 sklearn.inspection 모듈을 사용합니다.

---

## 실습편 (15-20분)

### 실습 소개 [1.5min]

> 이제 실습 시간입니다. 품질 예측 모델을 해석해봅니다.
>
> **실습 목표**입니다.
> 1. 랜덤포레스트 모델을 학습합니다.
> 2. Feature Importance를 분석합니다.
> 3. Permutation Importance와 비교합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.inspection import permutation_importance
> ```

### 실습 1: 데이터 준비 [2min]

> 첫 번째 실습입니다. 제조 공정 데이터를 준비합니다.
>
> ```python
> import numpy as np
> import pandas as pd
>
> np.random.seed(42)
> n_samples = 500
>
> data = pd.DataFrame({
>     'temperature': np.random.normal(85, 5, n_samples),
>     'humidity': np.random.normal(50, 8, n_samples),
>     'speed': np.random.normal(100, 10, n_samples),
>     'pressure': np.random.normal(1.0, 0.1, n_samples)
> })
> ```
>
> 온도, 습도, 속도, 압력 네 가지 변수가 있어요.

### 실습 2: 모델 학습 [2min]

> 두 번째 실습입니다. 랜덤포레스트 모델을 학습합니다.
>
> ```python
> from sklearn.ensemble import RandomForestClassifier
>
> model = RandomForestClassifier(n_estimators=100, random_state=42)
> model.fit(X_train, y_train)
>
> print(f"정확도: {model.score(X_test, y_test):.3f}")
> ```
>
> n_estimators=100은 100개의 트리를 사용한다는 뜻이에요.

### 실습 3: Feature Importance [2min]

> 세 번째 실습입니다. 특성 중요도를 확인합니다.
>
> ```python
> importance = model.feature_importances_
>
> for name, imp in zip(feature_names, importance):
>     print(f"{name}: {imp:.3f} ({imp*100:.1f}%)")
> ```
>
> 학습된 모델에서 `.feature_importances_` 속성으로 바로 접근할 수 있어요.
>
> 합계가 1인지 확인해보세요. 항상 1이 됩니다!

### 실습 4: 중요도 시각화 [2min]

> 네 번째 실습입니다. 막대 그래프로 시각화합니다.
>
> ```python
> import matplotlib.pyplot as plt
>
> sorted_idx = np.argsort(importance)
> plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
> plt.xlabel('중요도')
> plt.title('Feature Importance')
> plt.show()
> ```
>
> 정렬해서 보면 어떤 변수가 중요한지 한눈에 알 수 있어요.

### 실습 5: Permutation Importance [2min]

> 다섯 번째 실습입니다. Permutation Importance를 계산합니다.
>
> ```python
> from sklearn.inspection import permutation_importance
>
> result = permutation_importance(
>     model, X_test, y_test,
>     n_repeats=10,
>     random_state=42
> )
>
> print(result.importances_mean)
> ```
>
> n_repeats=10이면 10번 반복해서 평균을 계산해요.
> 여러 번 하면 결과가 더 안정적입니다.

### 실습 6: 두 방법 비교 [2min]

> 여섯 번째 실습입니다. 두 방법을 나란히 비교합니다.
>
> ```python
> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
>
> axes[0].barh(feature_names, importance, color='steelblue')
> axes[0].set_title('Feature Importance')
>
> axes[1].barh(feature_names, result.importances_mean, color='coral')
> axes[1].set_title('Permutation Importance')
>
> plt.show()
> ```
>
> 두 방법의 순위가 비슷한지 확인해보세요.

### 실습 7: 결과 해석 [2min]

> 일곱 번째 실습입니다. 비즈니스 인사이트를 도출합니다.
>
> 분석 결과를 보고서 형태로 정리해요.
>
> "모델 분석 결과, 온도가 품질에 35%로 가장 큰 영향을 미칩니다.
> 85도 초과 시 불량률이 급증하므로, 온도 모니터링 강화를 권장합니다."
>
> 숫자만 나열하지 말고, 비즈니스 의사결정에 도움이 되게 해석해야 해요.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **모델 해석**은 왜 그렇게 예측했는지 설명하는 것이에요.
>
> **Feature Importance**는 model.feature_importances_로 확인해요.
> 트리 기반 모델에서 자동으로 계산됩니다.
>
> **Permutation Importance**는 sklearn.inspection을 사용해요.
> 더 신뢰할 수 있는 방법입니다.
>
> 주의사항으로, 상관관계와 인과관계를 구분해야 해요.
> 도메인 지식과 함께 해석하는 것이 중요합니다.

#### 다음 차시 예고 [1min]

> 다음 25차시에서는 **모델 저장과 실무 배포 준비**를 배웁니다.
>
> joblib으로 모델을 저장하고 불러오는 방법,
> 실무에서 배포할 때 체크리스트를 다룹니다!

#### 마무리 [0.5min]

> 모델 해석 방법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- scikit-learn 설치

### 주의사항
- feature_importances_는 학습 후 바로 접근 가능
- Permutation Importance는 시간이 걸리므로 n_repeats를 낮게 시작
- 시각화 시 정렬하여 보여주기

### 예상 질문
1. "SHAP은 안 배우나요?"
   → SHAP은 고급 과정에서 다룸. 이번 차시는 기본적인 방법 학습

2. "딥러닝 모델도 해석 가능?"
   → 가능하지만 더 어려움. Grad-CAM, Attention 등 사용

3. "중요도가 비슷하면?"
   → 상관관계 분석 필요. 도메인 지식으로 우선순위 결정

4. "음수 값이 나오면?"
   → Permutation에서 음수는 해당 특성이 노이즈일 수 있음
