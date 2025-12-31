---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 24차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 모델 해석과 특성 중요도 분석

## 24차시 | AI 기초체력훈련 (Pre AI-Campus)

**모델이 왜 그렇게 예측했는지 설명하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **모델 해석**의 필요성을 이해한다
2. **특성 중요도(Feature Importance)**를 분석한다
3. **Permutation Importance**를 활용한다

---

# 왜 모델 해석이 필요한가?

## 블랙박스 vs 해석 가능한 모델

```
        입력 → [  ?  ] → 예측
              블랙박스
```

### 해석이 필요한 이유
- **신뢰**: "왜 불량이라고 판단했나요?"
- **디버깅**: 모델이 이상한 패턴을 학습하진 않았나?
- **규제**: 금융, 의료 분야는 설명 의무

> 모델 성능 + **설명 가능성** = 실무 적용

---

# 특성 중요도 (Feature Importance)

## 어떤 변수가 예측에 기여하는가?

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 특성 중요도 확인
print(model.feature_importances_)
# [0.32, 0.28, 0.25, 0.15]
```

### 의미
- 각 특성이 예측에 얼마나 기여하는지 **비율**
- 합하면 1 (또는 100%)
- **트리 기반 모델**에서 제공 (RF, XGBoost, Decision Tree)

---

# 특성 중요도 시각화

## 막대 그래프

```python
import matplotlib.pyplot as plt
import pandas as pd

# 중요도 정렬
importance_df = pd.DataFrame({
    '특성': feature_names,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=True)

# 시각화
plt.barh(importance_df['특성'], importance_df['중요도'])
plt.xlabel('중요도')
plt.title('특성 중요도')
plt.show()
```

---

# 제조 데이터 예시

## 품질 예측 모델의 특성 중요도

```
온도      ████████████████████  35%
습도      ████████████████      30%
속도      ██████████████        25%
압력      ██████                10%
```

### 해석
- **온도**가 품질에 가장 큰 영향
- 불량률 감소 → 온도 관리 우선!
- 압력은 상대적으로 영향 적음

---

# Permutation Importance

## 더 신뢰할 수 있는 중요도 측정

```python
from sklearn.inspection import permutation_importance

# Permutation Importance 계산
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

print(result.importances_mean)
```

### 원리
1. 특정 특성의 값을 **무작위로 섞음**
2. 예측 성능이 얼마나 떨어지는지 측정
3. 많이 떨어지면 → 중요한 특성!

---

# 두 방법 비교

## Feature Importance vs Permutation

| | Feature Importance | Permutation Importance |
|--|-------------------|----------------------|
| 계산 | 학습 중 계산 | 학습 후 계산 |
| 기반 | 불순도 감소 | 성능 변화 |
| 장점 | 빠름 | 더 신뢰성 |
| 단점 | 편향 가능 | 느림 |

> **추천**: 먼저 Feature Importance로 빠르게 확인,
> 중요한 분석은 Permutation Importance 사용

---

# 시각화 코드 전체

## 특성 중요도 분석

```python
import matplotlib.pyplot as plt
import numpy as np

# Feature Importance
feat_imp = model.feature_importances_
sorted_idx = np.argsort(feat_imp)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Feature Importance
axes[0].barh(range(len(sorted_idx)), feat_imp[sorted_idx])
axes[0].set_yticks(range(len(sorted_idx)))
axes[0].set_yticklabels(np.array(feature_names)[sorted_idx])
axes[0].set_title('Feature Importance')

# 2. Permutation Importance
axes[1].boxplot(result.importances[sorted_idx].T, vert=False)
axes[1].set_title('Permutation Importance')

plt.tight_layout()
plt.show()
```

---

# 주의사항

## 해석 시 유의점

### 1. 상관된 특성
```
온도 ↔ 습도 (상관관계 높음)
→ 중요도가 분산될 수 있음
```

### 2. 스케일 영향
- 원래 특성 중요도는 스케일에 민감하지 않음
- 단, 해석 시 단위 고려 필요

### 3. 인과관계 ≠ 상관관계
- 중요도가 높다 ≠ 원인이다
- 비즈니스 도메인 지식 필요

---

# 실무 활용

## 모델 해석 결과 보고

```
┌─────────────────────────────────────────┐
│     품질 예측 모델 분석 보고서           │
├─────────────────────────────────────────┤
│ 모델 성능: 정확도 92%, F1 0.89          │
│                                          │
│ 주요 영향 요인:                          │
│   1. 온도 (35%) - 85°C 초과 시 불량↑    │
│   2. 습도 (30%) - 60% 초과 시 불량↑     │
│   3. 속도 (25%) - 영향 중간             │
│                                          │
│ 권장사항:                                │
│   - 온도 모니터링 강화                   │
│   - 습도 제어 시스템 도입 검토           │
└─────────────────────────────────────────┘
```

---

# 다른 해석 방법

## 고급 기법 소개

| 방법 | 설명 | 난이도 |
|------|------|--------|
| SHAP | 각 예측에 대한 기여도 | ★★★ |
| LIME | 개별 예측 해석 | ★★☆ |
| PDP | 부분 의존성 플롯 | ★★☆ |

> 이번 차시: Feature Importance, Permutation
> 고급 과정: SHAP, LIME

---

# 다음 차시 예고

## 25차시: 모델 저장과 실무 배포 준비

- joblib로 모델 저장/불러오기
- 모델 버전 관리
- 실무 배포 체크리스트

> 학습한 모델을 **저장하고 재사용**하는 방법!

---

# 감사합니다

## AI 기초체력훈련 24차시

**모델 해석과 특성 중요도 분석**

모델이 무엇을 배웠는지 이해할 수 있게 되었습니다!
