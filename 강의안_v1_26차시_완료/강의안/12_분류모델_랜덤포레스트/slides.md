---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 12차시'
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

# 분류 모델 (2): 랜덤포레스트

## 12차시 | Part III. 문제 중심 모델링 실습

**여러 트리가 모여 숲을 이룹니다**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **앙상블 학습**의 개념을 설명한다
2. **랜덤포레스트**의 원리를 이해한다
3. **RandomForestClassifier**로 분류 모델을 구축한다

---

# 지난 시간 복습

## 의사결정나무의 한계

### 장점 ✅
- 해석이 쉬움
- 전처리 간단

### 단점 ❌
- **불안정함**: 데이터가 조금만 바뀌어도 트리가 크게 변함
- **과대적합**: 트리가 깊어지면 외워버림
- **성능 한계**: 단일 트리의 한계

> 해결책? **여러 트리를 합치자!**

---

# 앙상블 학습이란?

## Ensemble Learning

> 여러 모델을 **합쳐서** 더 좋은 성능을 내는 방법

### 비유: 집단 지성

```
                개별 전문가
              /     |     \
           의견1   의견2   의견3
              \     |     /
               ─────┼─────
                    ▼
              다수결 / 평균
                    ▼
               최종 결정
```

> 혼자보다 **여럿이 함께** 결정하면 더 정확!

---

# 랜덤포레스트란?

## Random Forest

> 여러 개의 의사결정나무를 만들어 **투표**로 결정

```
   트리1    트리2    트리3    ...    트리100
     │        │        │              │
   정상      불량      정상    ...    정상
     │        │        │              │
     └────────┴────────┴──────────────┘
                      │
                   투표!
                      │
               정상 (72표) → 최종: 정상
               불량 (28표)
```

---

# "랜덤"의 의미

## 두 가지 랜덤

### 1. 데이터 랜덤 (Bootstrap)
```
전체 데이터 [1,2,3,4,5,6,7,8,9,10]
    ↓ 랜덤 샘플링 (복원 추출)
트리1: [1,1,3,5,5,6,7,8,9,9]
트리2: [2,2,3,3,4,6,7,8,10,10]
```

### 2. 특성 랜덤 (Feature Sampling)
```
전체 특성: [온도, 습도, 속도]
    ↓ 랜덤 선택
트리1: [온도, 습도]
트리2: [습도, 속도]
```

---

# 왜 랜덤이 좋을까?

## 다양성이 힘!

```
┌──────────────────────────────────────────────┐
│ 같은 데이터, 같은 특성 → 같은 트리 100개     │
│ → 100개가 다 틀리면 다 틀림 (다양성 없음)    │
├──────────────────────────────────────────────┤
│ 다른 데이터, 다른 특성 → 다양한 트리 100개   │
│ → 서로 다른 실수 → 서로 보완 (다양성 있음)   │
└──────────────────────────────────────────────┘
```

> 약한 모델들이 모여 **강한 모델**이 된다!

---

# sklearn으로 구현하기

## RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터 준비
X = df[['온도', '습도', '속도']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 랜덤포레스트 모델
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

# 주요 하이퍼파라미터

## 조절할 수 있는 것들

```python
model = RandomForestClassifier(
    n_estimators=100,     # 트리 개수 (기본 100)
    max_depth=10,         # 각 트리의 최대 깊이
    min_samples_split=2,  # 분할 최소 샘플
    min_samples_leaf=1,   # 리프 최소 샘플
    max_features='sqrt',  # 특성 선택 개수
    random_state=42
)
```

### 핵심 파라미터
| 파라미터 | 설명 | 권장 |
|---------|------|------|
| n_estimators | 트리 개수 | 100~500 |
| max_depth | 트리 깊이 | 데이터에 따라 |

---

# 의사결정나무 vs 랜덤포레스트

## 성능 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 의사결정나무
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"의사결정나무: {dt.score(X_test, y_test):.1%}")

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"랜덤포레스트: {rf.score(X_test, y_test):.1%}")
```

> 대부분의 경우 **랜덤포레스트가 더 높은 정확도**!

---

# 이론 정리

## 랜덤포레스트 핵심

| 항목 | 설명 |
|------|------|
| 원리 | 여러 트리의 투표 |
| 핵심 | 다양성 (데이터/특성 랜덤) |
| 장점 | 높은 성능, 과대적합 방지 |
| 단점 | 해석 어려움 |
| 핵심 파라미터 | n_estimators, max_depth |

---

# - 실습편 -

## 12차시

**랜덤포레스트 실습**

---

# 실습 개요

## 랜덤포레스트로 불량 분류

### 목표
- 랜덤포레스트 모델 학습
- 의사결정나무와 성능 비교
- 특성 중요도 분석

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

---

# 실습 1: 데이터 준비

## 제조 데이터

```python
np.random.seed(42)
n = 500

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

defect_prob = 0.05 + 0.03*(df['온도']-80)/5 + 0.02*(df['습도']-40)/10
df['불량여부'] = (np.random.random(n) < defect_prob).astype(int)
```

---

# 실습 2: 랜덤포레스트 학습

## RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

X = df[['온도', '습도', '속도']]
y = df['불량여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"정확도: {model.score(X_test, y_test):.1%}")
```

---

# 실습 3: 성능 비교

## 의사결정나무 vs 랜덤포레스트

```python
from sklearn.tree import DecisionTreeClassifier

# 의사결정나무
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

print(f"의사결정나무: {dt_acc:.1%}")
print(f"랜덤포레스트: {rf_acc:.1%}")
```

---

# 실습 4: 특성 중요도

## 어떤 변수가 중요할까?

```python
importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print(importance)

# 시각화
importance.plot(kind='barh', x='특성', y='중요도')
plt.title('특성 중요도')
plt.show()
```

---

# 실습 5: n_estimators 실험

## 트리 개수에 따른 성능

```python
for n_trees in [10, 50, 100, 200, 500]:
    model_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model_temp.fit(X_train, y_train)
    acc = model_temp.score(X_test, y_test)
    print(f"트리 {n_trees}개: {acc:.1%}")
```

> 어느 정도 이상은 성능 향상이 미미 (보통 100개면 충분)

---

# 실습 6: OOB 점수

## 교차검증 없이 성능 추정

```python
model_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
model_oob.fit(X_train, y_train)

print(f"OOB 점수: {model_oob.oob_score_:.1%}")
print(f"테스트 점수: {model_oob.score(X_test, y_test):.1%}")
```

> OOB = Bootstrap에서 제외된 데이터로 평가

---

# 실습 7: 예측 확률

## predict_proba

```python
new_data = [[90, 55, 100]]
pred = model.predict(new_data)
proba = model.predict_proba(new_data)

print(f"예측: {'불량' if pred[0]==1 else '정상'}")
print(f"정상 확률: {proba[0][0]:.1%}")
print(f"불량 확률: {proba[0][1]:.1%}")
```

---

# 랜덤포레스트 장단점

## 정리

### 장점 ✅
- **높은 성능**: 대부분의 문제에서 좋은 결과
- **안정적**: 과대적합 위험 적음
- **특성 중요도**: 중요한 변수 파악 가능

### 단점 ❌
- **해석 어려움**: 100개 트리를 다 볼 수 없음
- **학습 시간**: 단일 트리보다 오래 걸림

---

# 언제 무엇을 쓸까?

## 의사결정나무 vs 랜덤포레스트

| 상황 | 추천 모델 |
|------|----------|
| 모델 설명이 필요할 때 | 의사결정나무 |
| 최대 성능이 필요할 때 | 랜덤포레스트 |
| 과대적합이 걱정될 때 | 랜덤포레스트 |
| 실무 대부분의 경우 | 랜덤포레스트 |

---

# 실습 정리

## 핵심 체크포인트

- [ ] RandomForestClassifier 생성 및 학습
- [ ] 의사결정나무와 성능 비교
- [ ] 특성 중요도 분석
- [ ] n_estimators 실험
- [ ] OOB 점수 확인

---

# 다음 차시 예고

## 13차시: 예측 모델 - 선형회귀와 다항회귀

### 학습 내용
- 선형회귀 복습
- 다항회귀 (Polynomial Regression)
- 회귀 모델 평가 지표

> 분류가 끝났으니 이제 **숫자 예측**!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **앙상블**: 여러 모델을 합쳐서 더 좋은 성능
2. **랜덤포레스트**: 다양한 트리들의 투표
3. **n_estimators**: 100개 정도면 충분
4. **특성 중요도**: 중요한 변수 파악 가능

---

# 감사합니다

## 12차시: 분류 모델 (2) - 랜덤포레스트

**앙상블의 힘을 경험했습니다!**
