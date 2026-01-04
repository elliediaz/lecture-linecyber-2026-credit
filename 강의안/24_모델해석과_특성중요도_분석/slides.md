---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 24차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# 모델 해석과 변수별 영향력 분석

## 24차시 | Part IV. AI 서비스화와 활용

**모델이 왜 그렇게 예측했는지 설명하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **모델 해석**의 필요성을 이해한다
2. **특성 중요도(Feature Importance)**를 분석한다
3. **Permutation Importance**를 활용한다

---

# 왜 모델 해석이 필요한가?

## 블랙박스 문제

```
      입력 → [  ???  ] → 예측
            블랙박스
```

### 해석이 필요한 세 가지 이유

| 이유 | 설명 |
|------|------|
| 신뢰 | "왜 불량이라고 판단했나요?" |
| 디버깅 | 모델이 이상한 패턴을 학습하진 않았나? |
| 규제 | 금융, 의료 분야는 설명 의무 |

> 모델 성능 + **설명 가능성** = 실무 적용

---

# 특성 중요도란?

## Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 특성 중요도 확인 (한 줄!)
print(model.feature_importances_)
# [0.35, 0.30, 0.25, 0.10]
```

### 의미
- 각 특성이 예측에 얼마나 기여하는지 **비율**
- 합하면 1 (= 100%)
- **트리 기반 모델**에서 자동 제공

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

# Permutation Importance

## 더 신뢰할 수 있는 방법

```python
from sklearn.inspection import permutation_importance

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

| 구분 | Feature Importance | Permutation Importance |
|------|-------------------|----------------------|
| 계산 시점 | 학습 중 | 학습 후 |
| 기반 | 불순도 감소 | 성능 변화 |
| 장점 | 빠름 | 더 신뢰성 |
| 단점 | 편향 가능 | 느림 |

> **추천**: Feature Importance로 빠르게 확인 →
> 중요 분석은 Permutation Importance 사용

---

# 해석 시 주의사항

## 세 가지 유의점

### 1. 상관된 특성
```
온도 ↔ 습도 (상관관계 높음)
→ 중요도가 분산될 수 있음
```

### 2. 스케일 영향
- 특성 중요도는 스케일에 민감하지 않음
- 단, 해석 시 단위 고려 필요

### 3. 인과관계 ≠ 상관관계
- 중요도가 높다 ≠ 원인이다
- 비즈니스 도메인 지식 필요

---

# 이론 정리

## 모델 해석 핵심

| 개념 | 설명 |
|------|------|
| 모델 해석 | 예측 이유를 설명 |
| Feature Importance | model.feature_importances_ |
| Permutation | sklearn.inspection |
| 주의사항 | 상관관계, 인과관계 구분 |

---

# - 실습편 -

## 24차시

**특성 중요도 분석 실습**

---

# 실습 개요

## 품질 예측 모델 해석

### 목표
- 랜덤포레스트 모델 학습
- Feature Importance 분석
- Permutation Importance 비교
- 결과 시각화 및 해석

### 실습 환경
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
```

---

# 실습 1: 데이터 준비

## 제조 공정 데이터

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n_samples),
    'humidity': np.random.normal(50, 8, n_samples),
    'speed': np.random.normal(100, 10, n_samples),
    'pressure': np.random.normal(1.0, 0.1, n_samples)
})

# 불량 여부 생성 (온도, 습도가 주요 요인)
defect_prob = 0.1 + 0.3 * (data['temperature'] - 85) / 10
data['defect'] = (defect_prob > 0.3).astype(int)
```

---

# 실습 2: 모델 학습

## 랜덤포레스트 분류기

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

feature_names = ['temperature', 'humidity', 'speed', 'pressure']
X = data[feature_names]
y = data['defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"정확도: {model.score(X_test, y_test):.3f}")
```

---

# 실습 3: Feature Importance

## 특성 중요도 확인

```python
# 특성 중요도 추출
importance = model.feature_importances_

# 출력
for name, imp in zip(feature_names, importance):
    print(f"{name}: {imp:.3f} ({imp*100:.1f}%)")

# 출력 예시:
# temperature: 0.350 (35.0%)
# humidity: 0.300 (30.0%)
# speed: 0.250 (25.0%)
# pressure: 0.100 (10.0%)
```

> 합계는 항상 1 (= 100%)

---

# 실습 4: 중요도 시각화

## 막대 그래프

```python
import matplotlib.pyplot as plt

# 정렬
sorted_idx = np.argsort(importance)

# 시각화
plt.figure(figsize=(8, 5))
plt.barh(
    np.array(feature_names)[sorted_idx],
    importance[sorted_idx],
    color='steelblue'
)
plt.xlabel('중요도')
plt.title('Feature Importance')
plt.show()
```

---

# 실습 5: Permutation Importance

## 더 신뢰할 수 있는 분석

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

# 결과 출력
for name, imp, std in zip(
    feature_names,
    result.importances_mean,
    result.importances_std
):
    print(f"{name}: {imp:.3f} (±{std:.3f})")
```

---

# 실습 6: 두 방법 비교

## 나란히 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature Importance
axes[0].barh(feature_names, importance, color='steelblue')
axes[0].set_title('Feature Importance')

# Permutation Importance
axes[1].barh(
    feature_names,
    result.importances_mean,
    xerr=result.importances_std,
    color='coral'
)
axes[1].set_title('Permutation Importance')

plt.tight_layout()
plt.show()
```

---

# 실습 7: 결과 해석

## 비즈니스 인사이트 도출

```
┌─────────────────────────────────────────┐
│     품질 예측 모델 분석 결과             │
├─────────────────────────────────────────┤
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

# 실습 정리

## 핵심 체크포인트

- [ ] 랜덤포레스트 모델 학습
- [ ] model.feature_importances_ 확인
- [ ] 막대 그래프 시각화
- [ ] permutation_importance() 실행
- [ ] 두 방법 비교 분석
- [ ] 비즈니스 인사이트 도출

---

# 고급 기법 소개

## SHAP, LIME

| 방법 | 설명 | 특징 |
|------|------|------|
| SHAP | 각 예측에 대한 기여도 | 개별 예측 해석 |
| LIME | 로컬 해석 모델 | 어떤 모델에도 적용 |
| PDP | 부분 의존성 플롯 | 변수 효과 시각화 |

> 이번 차시: Feature Importance, Permutation
> 고급 과정: SHAP, LIME

---

# 다음 차시 예고

## 25차시: 모델 저장과 실무 배포 준비

### 학습 내용
- joblib로 모델 저장/불러오기
- 모델 버전 관리
- 실무 배포 체크리스트

> 학습한 모델을 **저장하고 재사용**하는 방법!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **모델 해석**: 왜 그렇게 예측했는지 설명
2. **Feature Importance**: model.feature_importances_
3. **Permutation Importance**: sklearn.inspection
4. **시각화**: 막대 그래프로 비교
5. **주의사항**: 상관관계 ≠ 인과관계
6. **실무 활용**: 비즈니스 인사이트 도출

---

# 감사합니다

## 24차시: 모델 해석과 변수별 영향력 분석

**모델이 무엇을 배웠는지 설명할 수 있게 되었습니다!**
