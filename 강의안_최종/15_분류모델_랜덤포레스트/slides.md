---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 14차시'
footer: '공공데이터를 활용한 AI 예측 모델 구축'
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
  table { font-size: 0.9em; }
  .highlight { background-color: #fef3c7; padding: 10px; border-radius: 8px; }
  .important { background-color: #fee2e2; padding: 10px; border-radius: 8px; }
  .tip { background-color: #d1fae5; padding: 10px; border-radius: 8px; }
---

# 분류 모델 (2) - 랜덤포레스트

## 14차시 | Part III. 문제 중심 모델링 실습

**여러 나무를 모아 숲을 만듭니다!**

---

# 지난 시간 복습

## 12차시에서 배운 것

- **의사결정나무**: 질문 기반 분류
- **지니 불순도**: 데이터 섞임 정도 측정
- **max_depth**: 과대적합 방지
- **특성 중요도**: 변수별 기여도

### 의사결정나무의 한계
- 과대적합 쉬움
- **불안정** (데이터 변화에 민감)

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **앙상블 학습의 개념**을 설명한다 |
| 2 | **랜덤포레스트의 원리**를 이해한다 |
| 3 | **RandomForestClassifier**를 사용한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  앙상블     │ →  │  랜덤포레   │ → │   sklearn   │
│  학습 개념  │    │  스트 원리  │    │    실습     │
└─────────────┘    └─────────────┘    └─────────────┘
   집단 지성        배깅, 부트스트랩    모델 학습/평가
   왜 여러 개?      특성 랜덤 선택      파라미터 튜닝
```

---

<!-- _class: lead -->

# Part 1

## 앙상블 학습의 개념

---

# 앙상블이란?

## Ensemble = 집단, 합주단

> 여러 모델의 예측을 **결합**하여
> 단일 모델보다 **더 나은 성능**을 얻는 방법

### 일상 비유
- 한 명의 전문가 vs **여러 전문가의 협의**
- 한 번의 시험 vs **여러 번 시험의 평균**

---

# 집단 지성의 힘

## 왜 여러 모델이 더 좋을까?

### 젤리빈 실험
- 병에 담긴 젤리빈 개수 맞추기
- 개인 추측: 큰 오차
- **집단 평균**: 실제 값에 가까움!

### 핵심 원리
- 각 모델의 **오류가 서로 다름**
- 평균내면 **오류가 상쇄됨**

---

# 앙상블 학습의 조건

## 다양성이 중요!

<div class="highlight">

### 좋은 앙상블의 조건
1. 각 모델이 **어느 정도 정확**해야 함
2. 각 모델이 **서로 다른 오류**를 내야 함

### 나쁜 예
- 똑같은 모델 100개 → 효과 없음
- 모든 모델이 같은 곳에서 틀림 → 효과 없음

</div>

---

# 앙상블 방법 종류

## 대표적인 3가지

```
              앙상블 방법
                  │
     ┌────────────┼────────────┐
     │            │            │
  배깅        부스팅        스태킹
 (Bagging)   (Boosting)    (Stacking)
     │            │            │
 병렬 학습    순차 학습     계층 학습
랜덤포레스트  XGBoost        모델 결합
```

---

# 배깅 vs 부스팅

## 핵심 차이

| 구분 | 배깅 (Bagging) | 부스팅 (Boosting) |
|------|---------------|------------------|
| **학습 방식** | 병렬 (독립) | 순차 (의존) |
| **데이터** | 랜덤 샘플링 | 가중치 조정 |
| **목표** | 분산 감소 | 편향 감소 |
| **대표 모델** | 랜덤포레스트 | XGBoost, LightGBM |
| **과대적합** | 강함 | 주의 필요 |

---

# 배깅 (Bagging)

## Bootstrap Aggregating

> 데이터를 **부트스트랩 샘플링**하여
> 여러 모델을 **독립적으로** 학습

```
      원본 데이터
          │
    ┌─────┼─────┐
    ↓     ↓     ↓
  샘플1  샘플2  샘플3   ← 부트스트랩
    │     │     │
  모델1  모델2  모델3   ← 독립 학습
    │     │     │
    └─────┼─────┘
          ↓
       투표/평균        ← 결합
```

---

# 부트스트랩 샘플링

## Bootstrap Sampling

> 원본 데이터에서 **복원 추출**로 새 데이터셋 생성

### 예시 (원본: [A, B, C, D, E])
```
샘플1: [A, A, C, D, E]  ← A가 2번
샘플2: [B, C, C, E, E]  ← C, E가 2번
샘플3: [A, B, D, D, D]  ← D가 3번
```

### 특징
- 일부 데이터는 **여러 번** 선택됨
- 일부 데이터는 **선택 안 됨** (약 37%)

---

<!-- _class: lead -->

# Part 2

## 랜덤포레스트의 원리

---

# 랜덤포레스트란?

## Random Forest

> **의사결정나무 여러 개**를 만들고
> 결과를 **투표**로 결합하는 앙상블 방법

### 핵심 아이디어
- 나무 하나는 불안정
- 나무 **여러 개 모아서 숲**을 만들면 안정적

---

# 랜덤포레스트의 두 가지 랜덤

## 1. 데이터 랜덤 (배깅)

```
원본 데이터 → 부트스트랩 샘플링 → 각 트리마다 다른 데이터
```

## 2. 특성 랜덤

```
전체 특성 중 → 일부만 랜덤 선택 → 각 분할에서 다른 특성 고려
```

**두 가지 랜덤으로 트리들이 다양해짐!**

---

# 특성 랜덤 선택

## max_features

> 각 분할마다 **일부 특성만** 고려

### 예시 (특성 4개: 온도, 습도, 속도, 압력)
```
분할 1: [온도, 습도] 중 선택
분할 2: [속도, 압력] 중 선택
분할 3: [온도, 압력] 중 선택
```

### 효과
- 트리들이 **더 다양**해짐
- 특정 특성에 **덜 의존**

---

# 랜덤포레스트 학습 과정

## 단계별 설명

```
[원본 데이터]
      │
      ├──→ 부트스트랩 1 → 특성 랜덤 → 트리 1
      │
      ├──→ 부트스트랩 2 → 특성 랜덤 → 트리 2
      │
      ├──→ 부트스트랩 3 → 특성 랜덤 → 트리 3
      │
      └──→ ... (n_estimators개)
```

---

# 랜덤포레스트 예측

## 투표 (Voting)

### 분류: 다수결 투표
```
트리 1: 정상  ─┐
트리 2: 불량   │  → 다수결 → 정상 (2:1)
트리 3: 정상  ─┘
```

### 회귀: 평균
```
트리 1: 1200  ─┐
트리 2: 1150   │  → 평균 → 1183
트리 3: 1200  ─┘
```

---

# OOB (Out-of-Bag) 점수

## 별도 테스트 없이 성능 추정

> 각 트리 학습에 **사용되지 않은** 데이터로 평가

### 부트스트랩 특성
- 약 **37%**의 데이터가 각 트리 학습에서 제외됨
- 이 데이터로 해당 트리 평가 → OOB 점수

```python
model = RandomForestClassifier(oob_score=True)
model.fit(X, y)
print(f"OOB 점수: {model.oob_score_:.3f}")
```

---

# 의사결정나무 vs 랜덤포레스트

## 비교

| 구분 | 의사결정나무 | 랜덤포레스트 |
|------|-------------|-------------|
| **모델 수** | 1개 | 다수 (100+) |
| **안정성** | 불안정 | 안정적 |
| **과대적합** | 쉬움 | 저항력 강함 |
| **해석** | 용이 | 어려움 |
| **속도** | 빠름 | 느림 |
| **성능** | 보통 | 우수 |

---

# 랜덤포레스트 장단점

## 정리

<div class="tip">

### 장점
- 과대적합 저항력 **강함**
- 높은 예측 **정확도**
- 특성 중요도 **신뢰도 높음**
- **별도 튜닝 없이도** 좋은 성능

### 단점
- 학습/예측 **시간 오래 걸림**
- 메모리 **많이 사용**
- 개별 트리 **해석 어려움**

</div>

---

<!-- _class: lead -->

# Part 3

## sklearn으로 랜덤포레스트 실습

---

# RandomForestClassifier

## sklearn 클래스

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # 트리 개수
    max_depth=None,        # 트리 최대 깊이
    max_features='sqrt',   # 분할시 고려할 특성 수
    min_samples_split=2,   # 분할 최소 샘플
    random_state=42,       # 재현성
    n_jobs=-1              # 병렬 처리 (전체 CPU)
)
```

---

# 주요 파라미터

## 성능에 영향을 주는 파라미터

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `n_estimators` | 트리 개수 | 100~500 |
| `max_depth` | 트리 최대 깊이 | None 또는 5~20 |
| `max_features` | 분할시 특성 수 | 'sqrt' (기본) |
| `min_samples_leaf` | 리프 최소 샘플 | 1~5 |
| `n_jobs` | CPU 코어 수 | -1 (전체) |

---

# n_estimators의 영향

## 트리 개수에 따른 성능

```
트리 개수:   10      50     100     200     500
성능:       낮음 → 상승 → 포화 → 거의 동일 → 동일

           ↑         최적점
           │    ──────●──────────────
           │   /
    성능   │  /
           │ /
           └─────────────────────────→ 트리 개수
```

<div class="highlight">

**100~200개**면 대부분 충분. 더 늘려도 성능 향상 미미!

</div>

---

# 실습 데이터 준비

## 제조 불량 분류

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
    'pressure': np.random.normal(1.0, 0.1, n),
})
defect_prob = 0.1 + 0.03*(df['temperature']-80) + 0.01*(df['humidity']-45)
df['defect'] = (np.random.random(n) < defect_prob).astype(int)

X = df[['temperature', 'humidity', 'speed', 'pressure']]
y = df['defect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

# 모델 학습

## fit 메서드

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 생성
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 학습
model.fit(X_train, y_train)

print("학습 완료!")
print(f"트리 개수: {len(model.estimators_)}")
```

---

# 예측 및 평가

## predict, score

```python
from sklearn.metrics import classification_report

# 예측
y_pred = model.predict(X_test)

# 정확도
print(f"학습 정확도: {model.score(X_train, y_train):.1%}")
print(f"테스트 정확도: {model.score(X_test, y_test):.1%}")

# 분류 보고서
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
```

---

# OOB 점수 활용

## 별도 검증 세트 없이 평가

```python
# OOB 점수 활성화
model_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # OOB 점수 계산
    random_state=42,
    n_jobs=-1
)

model_oob.fit(X_train, y_train)

print(f"OOB 점수: {model_oob.oob_score_:.3f}")
print(f"테스트 정확도: {model_oob.score(X_test, y_test):.3f}")
```

---

# 의사결정나무 vs 랜덤포레스트 비교

## 코드로 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 의사결정나무
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
print(f"의사결정나무 - 학습: {dt.score(X_train, y_train):.1%}, "
      f"테스트: {dt.score(X_test, y_test):.1%}")

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"랜덤포레스트 - 학습: {rf.score(X_train, y_train):.1%}, "
      f"테스트: {rf.score(X_test, y_test):.1%}")
```

---

# 특성 중요도

## feature_importances_

```python
import matplotlib.pyplot as plt

# 특성 중요도
importances = model.feature_importances_
feature_names = ['temperature', 'humidity', 'speed', 'pressure']

# 정렬
sorted_idx = np.argsort(importances)[::-1]

print("특성 중요도:")
for idx in sorted_idx:
    print(f"  {feature_names[idx]}: {importances[idx]:.3f}")

# 시각화
plt.barh(range(len(importances)), importances[sorted_idx])
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel('중요도')
plt.title('랜덤포레스트 특성 중요도')
plt.show()
```

---

# n_estimators 실험

## 트리 개수별 성능

```python
estimators_range = [10, 25, 50, 100, 200, 300]
scores = []

for n_est in estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    scores.append(score)
    print(f"트리 {n_est:3}개: 정확도 {score:.3f}")

# 시각화
plt.plot(estimators_range, scores, 'o-')
plt.xlabel('트리 개수')
plt.ylabel('테스트 정확도')
plt.title('트리 개수별 성능')
plt.grid(True)
plt.show()
```

---

# 개별 트리 확인

## estimators_ 속성

```python
# 개별 트리 접근
tree_0 = model.estimators_[0]  # 첫 번째 트리
tree_1 = model.estimators_[1]  # 두 번째 트리

print(f"첫 번째 트리 깊이: {tree_0.get_depth()}")
print(f"두 번째 트리 깊이: {tree_1.get_depth()}")

# 개별 트리 예측
pred_0 = tree_0.predict(X_test[:5])
pred_1 = tree_1.predict(X_test[:5])
pred_rf = model.predict(X_test[:5])

print("\n처음 5개 샘플 예측:")
print(f"트리 1: {list(pred_0)}")
print(f"트리 2: {list(pred_1)}")
print(f"앙상블: {list(pred_rf)}")
```

---

<!-- _class: lead -->

# 실습편

## 제조 불량 분류 프로젝트

---

# 실습 목표

## 랜덤포레스트 완전 활용

1. 데이터 준비 및 분할
2. 랜덤포레스트 학습
3. 의사결정나무와 비교
4. 최적 파라미터 탐색
5. 특성 중요도 분석

---

# 실습 1: 데이터 준비

## 불량 분류 데이터

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 2000  # 더 많은 데이터

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
    'pressure': np.random.normal(1.0, 0.1, n),
    'vibration': np.random.normal(5, 1, n),  # 진동 추가
})

# 복잡한 불량 패턴
defect_prob = 0.1 + 0.02*(df['temperature']-80) + 0.01*(df['humidity']-45)
defect_prob += 0.02 * df['vibration']
df['defect'] = (np.random.random(n) < defect_prob).astype(int)

print(f"데이터 크기: {df.shape}")
print(f"불량률: {df['defect'].mean():.1%}")
```

---

# 실습 2: 의사결정나무 vs 랜덤포레스트

## 성능 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = df.drop('defect', axis=1)
y = df['defect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 의사결정나무
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
print(f"의사결정나무 테스트 정확도: {dt.score(X_test, y_test):.1%}")

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"랜덤포레스트 테스트 정확도: {rf.score(X_test, y_test):.1%}")
```

---

# 실습 3: 안정성 비교

## 여러 번 실험

```python
dt_scores = []
rf_scores = []

for i in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    dt = DecisionTreeClassifier(max_depth=10, random_state=i)
    dt.fit(X_tr, y_tr)
    dt_scores.append(dt.score(X_te, y_te))

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=i)
    rf.fit(X_tr, y_tr)
    rf_scores.append(rf.score(X_te, y_te))

print(f"의사결정나무 - 평균: {np.mean(dt_scores):.1%}, 표준편차: {np.std(dt_scores):.3f}")
print(f"랜덤포레스트 - 평균: {np.mean(rf_scores):.1%}, 표준편차: {np.std(rf_scores):.3f}")
```

<div class="tip">

랜덤포레스트의 **표준편차가 더 작음** → 더 안정적!

</div>

---

# 실습 4: 최적 n_estimators 찾기

## 트리 개수 실험

```python
estimators_range = [10, 25, 50, 75, 100, 150, 200]
train_scores = []
test_scores = []

for n_est in estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    print(f"n_estimators={n_est:3}: "
          f"학습 {train_scores[-1]:.3f}, 테스트 {test_scores[-1]:.3f}")

# 최적값
best_idx = np.argmax(test_scores)
print(f"\n최적 트리 개수: {estimators_range[best_idx]}")
```

---

# 실습 5: 특성 중요도 분석

## 두 모델 비교

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 의사결정나무
ax1 = axes[0]
sorted_idx_dt = np.argsort(dt.feature_importances_)[::-1]
ax1.barh(range(5), dt.feature_importances_[sorted_idx_dt])
ax1.set_yticks(range(5))
ax1.set_yticklabels([X.columns[i] for i in sorted_idx_dt])
ax1.set_title('의사결정나무 특성 중요도')

# 랜덤포레스트
ax2 = axes[1]
sorted_idx_rf = np.argsort(rf.feature_importances_)[::-1]
ax2.barh(range(5), rf.feature_importances_[sorted_idx_rf])
ax2.set_yticks(range(5))
ax2.set_yticklabels([X.columns[i] for i in sorted_idx_rf])
ax2.set_title('랜덤포레스트 특성 중요도')

plt.tight_layout()
plt.show()
```

---

# 실습 6: 새 데이터 예측

## 실무 적용

```python
# 새 제품 데이터
new_products = pd.DataFrame({
    'temperature': [87, 92, 78],
    'humidity': [55, 68, 42],
    'speed': [100, 95, 105],
    'pressure': [1.0, 0.95, 1.05],
    'vibration': [5.5, 6.2, 4.8]
})

# 예측
predictions = rf.predict(new_products)
probabilities = rf.predict_proba(new_products)

print("예측 결과:")
for i in range(len(new_products)):
    status = "불량" if predictions[i] == 1 else "정상"
    prob = probabilities[i][1] * 100
    print(f"  제품 {i+1}: {status} (불량 확률: {prob:.1f}%)")
```

---

# 핵심 정리

## 14차시 요약

| 개념 | 설명 |
|------|------|
| **앙상블** | 여러 모델 결합으로 성능 향상 |
| **배깅** | 병렬 학습 + 투표/평균 |
| **랜덤포레스트** | 배깅 + 특성 랜덤 선택 |
| **부트스트랩** | 복원 추출로 다양한 데이터셋 |
| **OOB 점수** | 사용 안 된 데이터로 평가 |
| **n_estimators** | 트리 개수 (100~200 권장) |

---

# 실무 가이드

## 랜덤포레스트 사용 시

<div class="highlight">

### 권장 설정
- `n_estimators`: 100 이상 (시간 여유 있으면 200+)
- `max_depth`: None 또는 15~20
- `max_features`: 'sqrt' (기본값)
- `n_jobs`: -1 (전체 CPU 활용)

### 언제 사용?
- 빠른 프로토타이핑 (튜닝 없이도 좋은 성능)
- 특성 중요도 분석
- 안정적인 예측 필요할 때

</div>

---

# 다음 차시 예고

## 14차시: 예측 모델 - 선형/다항 회귀

### 학습 내용
- 회귀 문제의 이해
- LinearRegression
- 다항 회귀와 PolynomialFeatures

<div class="tip">

분류에서 **회귀**로! 숫자를 예측합니다.

</div>

---

# 감사합니다

## 14차시: 분류 모델 - 랜덤포레스트

**숲을 만들어 더 강력해졌습니다!**

Q&A
