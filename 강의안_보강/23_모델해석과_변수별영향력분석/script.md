# [23차시] 모델 해석과 변수별 영향력 분석 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 23차시 |
| 주제 | 모델 해석과 변수별 영향력 분석 |
| 시간 | 30분 (이론 15분 + 실습 13분 + 정리 2분) |
| 학습 목표 | 모델 해석, Feature/Permutation Importance |

---

## 학습 목표

1. 모델 해석의 필요성과 종류를 이해한다
2. Feature Importance를 계산하고 해석한다
3. Permutation Importance를 활용한다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | 모델 해석의 필요성 |
| 대주제 2 | 5분 | Feature Importance |
| 대주제 3 | 5분 | Permutation Importance |
| 실습 | 11분 | 품질 예측 모델 해석 |
| 정리 | 2분 | 요약 및 다음 차시 예고 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 CNN, RNN 같은 고급 딥러닝 아키텍처를 살펴봤습니다."

> "오늘은 다시 모델 해석으로 돌아옵니다. 모델이 왜 그런 예측을 했는지 이해하는 방법을 배웁니다."

> "아무리 좋은 모델이라도 '왜?'에 답할 수 없으면 실무에서 쓰기 어렵거든요."

---

### 대주제 1: 모델 해석의 필요성 (5분)

#### 슬라이드 4-6: 블랙박스 문제

> "품질 예측 모델이 불량이라고 예측했습니다. 현장에서 이걸 어떻게 받아들일까요?"

> "'왜 불량인데?' '뭘 바꾸면 되는데?' 이런 질문이 나옵니다. 모델이 이유를 설명하지 못하면 아무도 안 써요."

> "이게 블랙박스 문제입니다. 입력이 들어가고 출력이 나오는데 중간이 안 보여요."

---

#### 슬라이드 7-9: 왜 중요한가

> "모델 해석이 중요한 이유를 네 가지로 정리하면:"

> "첫째, **신뢰**입니다. 의사결정자가 근거를 알아야 결정할 수 있어요."

> "둘째, **디버깅**입니다. 이상한 패턴을 학습했는지 확인할 수 있어요."

> "셋째, **개선**입니다. 중요한 변수를 알면 데이터 수집 방향을 정할 수 있어요."

> "넷째, **규제**입니다. EU AI Act처럼 설명 의무가 법으로 정해지는 추세입니다."

---

### 대주제 2: Feature Importance (5분)

#### 슬라이드 10-12: 개념

> "Feature Importance는 모델이 예측할 때 각 변수가 얼마나 기여했는지를 수치로 보여줍니다."

> "트리 모델에서는 각 변수로 분할할 때 불순도가 얼마나 줄었는지를 합산합니다."

> "많이 사용되고, 불순도를 많이 줄인 변수가 중요한 거예요."

---

#### 슬라이드 13-15: 추출과 시각화

```python
# Feature Importance 추출
importances = model.feature_importances_

# 시각화
plt.barh(feature_names, importances)
plt.title('Feature Importance')
```

> "feature_importances_ 속성으로 바로 꺼낼 수 있어요. 합하면 1이 됩니다."

> "막대 그래프로 그리면 어떤 변수가 중요한지 한눈에 보입니다."

---

#### 슬라이드 16-18: 해석과 활용

> "온도가 35%로 가장 중요하면, 온도 모니터링을 강화해야 합니다."

> "습도가 10%로 낮으면, 습도 센서에 투자하는 건 비효율적일 수 있어요."

> "주의할 점은 상관된 변수가 있으면 중요도가 분산된다는 겁니다. 온도와 습도가 상관이 높으면 둘 다 낮게 나올 수 있어요."

---

### 대주제 3: Permutation Importance (5분)

#### 슬라이드 19-21: 개념

> "Permutation Importance는 변수 값을 무작위로 섞어서 성능이 얼마나 떨어지는지 봅니다."

> "중요한 변수를 섞으면 성능이 크게 떨어지고, 중요하지 않은 변수를 섞어도 별 차이가 없어요."

> "이 방법은 모든 모델에 적용할 수 있다는 게 장점입니다. 신경망에도 쓸 수 있어요."

---

#### 슬라이드 22-24: sklearn 구현

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

importances = result.importances_mean
std = result.importances_std
```

> "n_repeats는 반복 횟수입니다. 여러 번 섞어서 평균을 내면 안정적인 결과가 나와요."

> "표준편차도 같이 나오니까 신뢰 구간을 그릴 수 있습니다."

---

#### 슬라이드 25-27: 비교와 활용

> "Feature Importance와 Permutation Importance를 같이 보면 좋습니다."

> "두 방법의 순위가 비슷하면 결과를 신뢰할 수 있어요."

> "순위가 다르면 상관 변수 문제일 수 있으니 더 자세히 살펴봐야 합니다."

---

### 실습편 (11분)

#### 슬라이드 28-30: 데이터 준비

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 제조 데이터 로드
X = df[['temperature', 'pressure', 'speed', 'humidity', 'vibration']]
y = df['defect']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

#### 슬라이드 31-33: 모델 학습

```python
# RandomForest 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 성능 확인
print(f"정확도: {model.score(X_test, y_test):.2%}")
```

---

#### 슬라이드 34-36: Feature Importance

```python
# 추출
fi = model.feature_importances_

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(X.columns, fi)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
```

---

#### 슬라이드 37-39: Permutation Importance

```python
from sklearn.inspection import permutation_importance

# 계산
result = permutation_importance(model, X_test, y_test, n_repeats=10)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(X.columns, result.importances_mean)
plt.errorbar(result.importances_mean, X.columns,
             xerr=result.importances_std, fmt='o')
plt.xlabel('Importance')
plt.title('Permutation Importance')
```

---

#### 슬라이드 40-42: 결과 비교

```python
# 비교 테이블
comparison = pd.DataFrame({
    'Feature': X.columns,
    'Feature Importance': fi,
    'Permutation Importance': result.importances_mean
})
comparison = comparison.sort_values('Permutation Importance', ascending=False)
print(comparison)
```

> "두 방법의 순위가 비슷한지 확인합니다. 비슷하면 신뢰도가 높아요."

---

### 정리 (2분)

#### 슬라이드 43-44: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**모델 해석**은 신뢰, 디버깅, 개선, 규제 대응에 필수입니다."

> "**Feature Importance**는 트리 모델에서 빠르게 추출할 수 있지만 상관 변수 문제가 있습니다."

> "**Permutation Importance**는 모든 모델에 적용 가능하고, 테스트 데이터 기반이라 신뢰도가 높습니다."

---

#### 슬라이드 45-46: 다음 차시 예고

> "다음 시간에는 모델 저장과 실무 배포를 배웁니다. joblib으로 저장하고, Pipeline을 구성하는 방법을 다룹니다."

> "오늘 수업 마무리합니다. 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: 어떤 방법이 더 좋나요?

> "둘 다 쓰는 게 좋습니다. Feature Importance는 빠르게 탐색하고, Permutation Importance로 검증하세요."

### Q2: 신경망도 해석할 수 있나요?

> "Permutation Importance는 신경망에도 적용 가능합니다. 더 정밀한 해석이 필요하면 SHAP를 사용하세요."

### Q3: 중요도가 0인 변수는 제거해야 하나요?

> "바로 제거하지 마세요. 다른 변수와 조합했을 때 중요할 수 있어요. 제거 후 성능을 비교해보세요."

### Q4: SHAP는 뭔가요?

> "SHAP는 게임 이론 기반 해석 방법입니다. 개별 예측에 대해 각 변수가 얼마나 기여했는지 정밀하게 알려줘요."

---

## 참고 자료

### 공식 문서
- [sklearn Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [SHAP 라이브러리](https://shap.readthedocs.io/)

### 관련 차시
- 21차시: 딥러닝 심화
- 23차시: 모델 저장과 실무 배포

---

## 체크리스트

수업 전:
- [ ] 예제 데이터 준비
- [ ] Feature Importance 코드 테스트
- [ ] Permutation Importance 실행 시간 확인

수업 중:
- [ ] 블랙박스 문제 강조
- [ ] 비즈니스 인사이트 연결
- [ ] 두 방법 비교 설명

수업 후:
- [ ] 실습 코드 배포
- [ ] 모델 저장 예고
