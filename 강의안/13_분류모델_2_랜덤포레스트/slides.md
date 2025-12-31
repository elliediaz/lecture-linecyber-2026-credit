---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 13차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# 분류 모델 (2): 랜덤포레스트

## 13차시 | AI 기초체력훈련 (Pre AI-Campus)

**여러 트리가 모여 숲을 이룹니다**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **앙상블 학습**의 개념을 설명한다
2. **랜덤포레스트**의 원리를 이해한다
3. **RandomForestClassifier**로 분류 모델을 구축한다

---

# 지난 시간 복습

## 의사결정트리의 한계

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

> 여러 개의 의사결정트리를 만들어 **투표**로 결정

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
트리3: [1,3,4,5,5,6,7,9,9,10]
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
model = RandomForestClassifier(
    n_estimators=100,     # 트리 개수
    random_state=42
)
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

# 의사결정트리 vs 랜덤포레스트

## 성능 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 의사결정트리
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"의사결정트리: {dt.score(X_test, y_test):.1%}")

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"랜덤포레스트: {rf.score(X_test, y_test):.1%}")
```

> 대부분의 경우 **랜덤포레스트가 더 높은 정확도**!

---

# 특성 중요도

## Feature Importance

```python
# 특성 중요도 확인
importance = model.feature_importances_

for name, imp in zip(X.columns, importance):
    print(f"{name}: {imp:.1%}")

# 출력 예시:
# 온도: 45.2%
# 습도: 32.1%
# 속도: 22.7%
```

> 어떤 특성이 예측에 중요한지 알 수 있어요!

---

# 특성 중요도 시각화

## 막대 그래프로 확인

```python
import matplotlib.pyplot as plt

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

---

# n_estimators 효과

## 트리 개수에 따른 성능

```
트리 개수    테스트 정확도
   10          82%
   50          85%
  100          86%
  200          86%
  500          86%
```

> 어느 정도 이상은 **성능 향상이 미미**
> 보통 100~200개면 충분!

---

# OOB (Out-of-Bag) 점수

## 별도 검증 없이 성능 추정

```python
model = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,       # OOB 점수 활성화
    random_state=42
)
model.fit(X_train, y_train)

print(f"OOB 점수: {model.oob_score_:.1%}")
```

### OOB란?
- Bootstrap 샘플링에서 제외된 데이터
- 약 37%의 데이터가 각 트리 학습에서 제외됨
- 이 데이터로 성능을 추정 (교차검증과 비슷)

---

# 랜덤포레스트 장단점

## 정리

### 장점 ✅
- **높은 성능**: 대부분의 문제에서 좋은 결과
- **안정적**: 과대적합 위험 적음
- **특성 중요도**: 중요한 변수 파악 가능
- **병렬 처리**: n_jobs=-1로 빠른 학습

### 단점 ❌
- **해석 어려움**: 100개 트리를 다 볼 수 없음
- **메모리**: 트리가 많으면 메모리 사용 증가
- **학습 시간**: 단일 트리보다 오래 걸림

---

# 실습 정리

## 전체 코드

```python
from sklearn.ensemble import RandomForestClassifier

# 1. 모델 생성
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# 2. 학습
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가
accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.1%}")
```

---

# 언제 무엇을 쓸까?

## 의사결정트리 vs 랜덤포레스트

| 상황 | 추천 모델 |
|------|----------|
| 모델 설명이 필요할 때 | 의사결정트리 |
| 최대 성능이 필요할 때 | 랜덤포레스트 |
| 빠른 학습이 필요할 때 | 의사결정트리 |
| 과대적합이 걱정될 때 | 랜덤포레스트 |
| 입문자 학습용 | 의사결정트리 |
| 실무 대부분의 경우 | 랜덤포레스트 |

---

# 다음 차시 예고

## 14차시: 회귀 모델

- 선형회귀 복습
- 다항회귀 (Polynomial Regression)
- 회귀 모델 평가 지표

> 분류가 끝났으니 이제 **숫자 예측**!

---

# 감사합니다

## AI 기초체력훈련 13차시

**분류 모델 (2): 랜덤포레스트**

앙상블의 힘을 경험했습니다!
