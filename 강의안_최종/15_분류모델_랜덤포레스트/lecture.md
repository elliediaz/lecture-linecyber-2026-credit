# [14차시] 분류 모델 (2) - 랜덤포레스트

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **앙상블 학습의 개념**을 집단 지성 원리로 설명함
2. **랜덤포레스트의 원리**를 배깅과 특성 랜덤 선택으로 이해함
3. **RandomForestClassifier**를 사용하여 안정적인 분류 모델을 구축함

---

## 실습 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| **Breast Cancer** | sklearn.datasets | 랜덤포레스트 분류 실습 (유방암 진단) |

Breast Cancer 데이터셋은 종양의 특성을 기반으로 악성(malignant)/양성(benign)을 분류하는 이진 분류 문제임.

---

## 강의 구성

| 파트 | 주제 | 핵심 내용 |
|:----:|------|----------|
| 1 | 앙상블 학습의 개념 | 집단 지성, 배깅 vs 부스팅 |
| 2 | 랜덤포레스트의 원리 | 두 가지 랜덤, 투표, OOB |
| 3 | sklearn으로 랜덤포레스트 실습 | RandomForestClassifier, 파라미터 |

---

## 파트 1: 앙상블 학습의 개념

### 개념 설명

#### 앙상블(Ensemble)이란?

여러 모델의 예측을 **결합**하여 단일 모델보다 **더 나은 성능**을 얻는 방법임.

일상 비유:
- 한 명의 전문가 vs **여러 전문가의 협의**
- 한 번의 시험 vs **여러 번 시험의 평균**

#### 집단 지성의 힘

젤리빈 실험:
- 병에 담긴 젤리빈 개수 맞추기
- 개인 추측: 큰 오차
- **집단 평균**: 실제 값에 가까움

핵심 원리:
- 각 모델의 **오류가 서로 다름**
- 평균내면 **오류가 상쇄됨**

#### 좋은 앙상블의 조건

| 조건 | 좋은 예 | 나쁜 예 |
|------|---------|---------|
| 각 모델이 어느 정도 정확해야 함 | 정확도 70% 이상 | 정확도 50% (랜덤 추측) |
| 각 모델이 서로 다른 오류를 내야 함 | 다양한 오류 패턴 | 똑같은 모델 100개 |

#### 앙상블 방법 종류

```
              앙상블 방법
                  |
     +------------+------------+
     |            |            |
  배깅        부스팅        스태킹
 (Bagging)   (Boosting)    (Stacking)
     |            |            |
 병렬 학습    순차 학습     계층 학습
랜덤포레스트  XGBoost        모델 결합
```

#### 배깅 vs 부스팅

| 구분 | 배깅 (Bagging) | 부스팅 (Boosting) |
|------|---------------|------------------|
| **학습 방식** | 병렬 (독립) | 순차 (의존) |
| **데이터** | 랜덤 샘플링 | 가중치 조정 |
| **목표** | 분산 감소 | 편향 감소 |
| **대표 모델** | 랜덤포레스트 | XGBoost, LightGBM |
| **과대적합** | 강함 | 주의 필요 |

#### 배깅(Bagging) - Bootstrap Aggregating

데이터를 부트스트랩 샘플링하여 여러 모델을 독립적으로 학습하는 방식임.

```
      원본 데이터
          |
    +-----+-----+
    v     v     v
  샘플1  샘플2  샘플3   <-- 부트스트랩
    |     |     |
  모델1  모델2  모델3   <-- 독립 학습
    |     |     |
    +-----+-----+
          v
       투표/평균        <-- 결합
```

#### 부트스트랩 샘플링(Bootstrap Sampling)

원본 데이터에서 **복원 추출**로 새 데이터셋을 생성함.

예시 (원본: [A, B, C, D, E]):
- 샘플1: [A, A, C, D, E] - A가 2번
- 샘플2: [B, C, C, E, E] - C, E가 2번
- 샘플3: [A, B, D, D, D] - D가 3번

특징:
- 일부 데이터는 **여러 번** 선택됨
- 일부 데이터는 **선택 안 됨** (약 37%)

### 실습 코드

#### 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
import time

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Breast Cancer 데이터셋 로딩
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
print("Breast Cancer 데이터셋 로딩 완료")

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"특성 개수: {len(cancer.feature_names)}")
print(f"클래스: {list(cancer.target_names)}")  # malignant(악성), benign(양성)

print(f"\n클래스별 샘플 수:")
for i, name in enumerate(cancer.target_names):
    count = (df['target'] == i).sum()
    print(f"  {name}: {count}개 ({count/len(df):.1%})")
```

### 결과 해설

- Breast Cancer 데이터셋은 569개 샘플, 30개 특성으로 구성됨
- 악성(malignant) 212개, 양성(benign) 357개로 다소 불균형한 분포를 보임
- 30개 특성은 종양의 반경, 질감, 둘레, 면적 등 다양한 측정값을 포함함

---

## 파트 2: 랜덤포레스트의 원리

### 개념 설명

#### 랜덤포레스트(Random Forest)란?

**의사결정나무 여러 개**를 만들고 결과를 **투표**로 결합하는 앙상블 방법임.

핵심 아이디어:
- 나무 하나는 불안정함
- 나무 **여러 개 모아서 숲**을 만들면 안정적임

#### 랜덤포레스트의 두 가지 랜덤

| 랜덤 유형 | 설명 |
|----------|------|
| **1. 데이터 랜덤 (배깅)** | 원본 데이터 --> 부트스트랩 샘플링 --> 각 트리마다 다른 데이터 |
| **2. 특성 랜덤** | 전체 특성 중 --> 일부만 랜덤 선택 --> 각 분할에서 다른 특성 고려 |

두 가지 랜덤으로 트리들이 다양해짐.

#### 특성 랜덤 선택 (max_features)

각 분할마다 일부 특성만 고려함.

예시 (특성 4개: 온도, 습도, 속도, 압력):
- 분할 1: [온도, 습도] 중 선택
- 분할 2: [속도, 압력] 중 선택
- 분할 3: [온도, 압력] 중 선택

효과:
- 트리들이 **더 다양**해짐
- 특정 특성에 **덜 의존**

#### 랜덤포레스트 학습 과정

```
[원본 데이터]
      |
      +---> 부트스트랩 1 --> 특성 랜덤 --> 트리 1
      |
      +---> 부트스트랩 2 --> 특성 랜덤 --> 트리 2
      |
      +---> 부트스트랩 3 --> 특성 랜덤 --> 트리 3
      |
      +---> ... (n_estimators개)
```

#### 랜덤포레스트 예측 - 투표(Voting)

| 문제 유형 | 결합 방식 |
|----------|----------|
| **분류** | 다수결 투표 |
| **회귀** | 평균 |

분류 예시:
```
트리 1: 정상  --+
트리 2: 불량   |--> 다수결 --> 정상 (2:1)
트리 3: 정상  --+
```

#### OOB (Out-of-Bag) 점수

각 트리 학습에 **사용되지 않은** 데이터로 평가하는 방식임.

부트스트랩 특성:
- 약 **37%**의 데이터가 각 트리 학습에서 제외됨
- 이 데이터로 해당 트리 평가 --> OOB 점수

별도 테스트 세트 없이도 성능 추정이 가능함.

#### 의사결정나무 vs 랜덤포레스트

| 구분 | 의사결정나무 | 랜덤포레스트 |
|------|-------------|-------------|
| **모델 수** | 1개 | 다수 (100+) |
| **안정성** | 불안정 | 안정적 |
| **과대적합** | 쉬움 | 저항력 강함 |
| **해석** | 용이 | 어려움 |
| **속도** | 빠름 | 느림 |
| **성능** | 보통 | 우수 |

#### 랜덤포레스트 장단점

| 장점 | 단점 |
|------|------|
| 과대적합 저항력 **강함** | 학습/예측 **시간 오래 걸림** |
| 높은 예측 **정확도** | 메모리 **많이 사용** |
| 특성 중요도 **신뢰도 높음** | 개별 트리 **해석 어려움** |
| **별도 튜닝 없이도** 좋은 성능 | |

### 실습 코드

#### 데이터 분할

```python
# 주요 특성 선택 (30개 중 10개 사용)
feature_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                   'mean smoothness', 'mean compactness', 'mean concavity',
                   'mean concave points', 'mean symmetry', 'mean fractal dimension']
X = df[feature_columns]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"학습 데이터 악성 비율: {(y_train == 0).mean():.1%}")
print(f"테스트 데이터 악성 비율: {(y_test == 0).mean():.1%}")
```

### 결과 해설

- 10개의 주요 특성을 선택하여 학습에 사용함
- 학습 455개, 테스트 114개로 분할됨
- stratify 옵션으로 악성/양성 비율이 유지됨

---

## 파트 3: sklearn으로 랜덤포레스트 실습

### 개념 설명

#### RandomForestClassifier 주요 파라미터

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `n_estimators` | 트리 개수 | 100~500 |
| `max_depth` | 트리 최대 깊이 | None 또는 5~20 |
| `max_features` | 분할시 특성 수 | 'sqrt' (기본) |
| `min_samples_leaf` | 리프 최소 샘플 | 1~5 |
| `n_jobs` | CPU 코어 수 | -1 (전체) |
| `oob_score` | OOB 점수 계산 | True |

#### n_estimators의 영향

```
트리 개수:   10      50     100     200     500
성능:       낮음 --> 상승 --> 포화 --> 거의 동일 --> 동일

           ^         최적점
           |    ------*--------------
           |   /
    성능   |  /
           | /
           +-------------------------> 트리 개수
```

**100~200개**면 대부분 충분함. 더 늘려도 성능 향상은 미미함.

### 실습 코드

#### 의사결정나무 vs 랜덤포레스트 비교

```python
# 의사결정나무
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
start_time = time.time()
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - start_time

dt_train_score = dt_model.score(X_train, y_train)
dt_test_score = dt_model.score(X_test, y_test)

print(f"[의사결정나무]")
print(f"  학습 시간: {dt_train_time:.3f}초")
print(f"  학습 정확도: {dt_train_score:.1%}")
print(f"  테스트 정확도: {dt_test_score:.1%}")

# 랜덤포레스트
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

rf_train_score = rf_model.score(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)

print(f"\n[랜덤포레스트 (n_estimators=100)]")
print(f"  학습 시간: {rf_train_time:.3f}초")
print(f"  학습 정확도: {rf_train_score:.1%}")
print(f"  테스트 정확도: {rf_test_score:.1%}")
print(f"  트리 개수: {len(rf_model.estimators_)}")

print(f"\n[비교 결과]")
print(f"  테스트 정확도 차이: {rf_test_score - dt_test_score:+.1%}")
```

#### OOB 점수 활용

```python
# OOB 점수 활성화
rf_oob = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    oob_score=True,  # OOB 점수 계산
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)

print(f"[OOB 점수 활용]")
print(f"  OOB 점수: {rf_oob.oob_score_:.3f}")
print(f"  테스트 점수: {rf_oob.score(X_test, y_test):.3f}")
print(f"  차이: {abs(rf_oob.oob_score_ - rf_oob.score(X_test, y_test)):.3f}")
print("  --> OOB 점수가 테스트 점수와 유사함")
```

#### n_estimators 실험

```python
estimators_range = [10, 25, 50, 75, 100, 150, 200, 300]
train_scores = []
test_scores = []
train_times = []

print(f"{'트리 개수':>8} {'학습시간':>10} {'학습정확도':>10} {'테스트정확도':>12}")
print("-" * 45)

for n_est in estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)
    train_times.append(train_time)

    print(f"{n_est:>8} {train_time:>10.3f}s {train_score:>10.1%} {test_score:>12.1%}")

best_idx = np.argmax(test_scores)
print(f"\n최적 트리 개수: {estimators_range[best_idx]}")
print(f"최고 테스트 정확도: {test_scores[best_idx]:.1%}")
```

#### 안정성 비교 실험

```python
dt_scores_list = []
rf_scores_list = []

for i in range(10):
    # 데이터 랜덤 분할
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=i)

    # 의사결정나무
    dt = DecisionTreeClassifier(max_depth=10, random_state=i)
    dt.fit(X_tr, y_tr)
    dt_scores_list.append(dt.score(X_te, y_te))

    # 랜덤포레스트
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=i, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_scores_list.append(rf.score(X_te, y_te))

print(f"[10회 실험 결과]")
print(f"\n의사결정나무:")
print(f"  평균: {np.mean(dt_scores_list):.1%}")
print(f"  표준편차: {np.std(dt_scores_list):.3f}")

print(f"\n랜덤포레스트:")
print(f"  평균: {np.mean(rf_scores_list):.1%}")
print(f"  표준편차: {np.std(rf_scores_list):.3f}")

print(f"\n[안정성 비교]")
if np.std(rf_scores_list) < np.std(dt_scores_list):
    ratio = np.std(dt_scores_list)/np.std(rf_scores_list)
    print(f"  랜덤포레스트 표준편차가 {ratio:.1f}배 작음")
    print(f"  --> 랜덤포레스트가 더 안정적")
```

#### 특성 중요도 비교

```python
dt_importances = dt_model.feature_importances_
rf_importances = rf_model.feature_importances_

print(f"[특성 중요도]")
print(f"{'특성':>25} {'의사결정나무':>12} {'랜덤포레스트':>14}")
print("-" * 55)
for i, col in enumerate(feature_columns):
    print(f"{col:>25} {dt_importances[i]:>12.3f} {rf_importances[i]:>14.3f}")
```

#### 개별 트리 확인

```python
print(f"[랜덤포레스트 개별 트리]")
print(f"트리 개수: {len(rf_model.estimators_)}")

# 처음 5개 트리 정보
print("\n처음 5개 트리 정보:")
for i in range(5):
    tree = rf_model.estimators_[i]
    print(f"  트리 {i+1}: 깊이={tree.get_depth()}, 리프 수={tree.get_n_leaves()}")

# 개별 트리 예측 비교
print("\n처음 3개 샘플에 대한 개별 트리 예측:")
sample_data = X_test.iloc[:3]
for i in range(3):
    tree_preds = [rf_model.estimators_[j].predict([sample_data.iloc[i]])[0]
                  for j in range(5)]
    ensemble_pred = rf_model.predict([sample_data.iloc[i]])[0]
    actual = y_test.iloc[i]
    print(f"\n  샘플 {i+1}:")
    print(f"    트리 1-5 예측: {tree_preds}")
    print(f"    앙상블 예측: {ensemble_pred} (실제: {actual})")
```

#### 새 데이터 예측

```python
# 새로운 환자 데이터 (가상)
new_patients = pd.DataFrame({
    'mean radius': [14.0, 18.0, 12.0, 20.0, 11.0],
    'mean texture': [18.0, 22.0, 16.0, 25.0, 15.0],
    'mean perimeter': [90.0, 120.0, 78.0, 135.0, 72.0],
    'mean area': [600.0, 1000.0, 450.0, 1200.0, 380.0],
    'mean smoothness': [0.10, 0.12, 0.09, 0.13, 0.08],
    'mean compactness': [0.10, 0.15, 0.08, 0.18, 0.07],
    'mean concavity': [0.08, 0.12, 0.05, 0.15, 0.04],
    'mean concave points': [0.04, 0.08, 0.03, 0.10, 0.02],
    'mean symmetry': [0.18, 0.20, 0.17, 0.22, 0.16],
    'mean fractal dimension': [0.06, 0.07, 0.06, 0.08, 0.05]
})

print("[새 환자 데이터 (주요 특성)]")
print(new_patients[['mean radius', 'mean area', 'mean concavity']].to_string())

# 랜덤포레스트 예측
rf_predictions = rf_model.predict(new_patients)
rf_probabilities = rf_model.predict_proba(new_patients)

print("\n[진단 예측 결과]")
print(f"{'환자':>4} {'RF진단':>12} {'RF확신도':>10}")
print("-" * 30)
for i in range(len(new_patients)):
    rf_diag = cancer.target_names[rf_predictions[i]]
    rf_conf = max(rf_probabilities[i]) * 100
    print(f"{i+1:>4} {rf_diag:>12} {rf_conf:>9.1f}%")
```

### 결과 해설

- 랜덤포레스트가 의사결정나무보다 일관되게 높은 테스트 정확도를 기록함
- 10회 반복 실험에서 랜덤포레스트의 표준편차가 더 작아 안정적임
- 특성 중요도에서 랜덤포레스트가 더 균형 잡힌 분포를 보여줌

---

## 핵심 정리

### 앙상블 학습 핵심 개념

| 개념 | 설명 |
|------|------|
| **앙상블** | 여러 모델 결합으로 성능 향상 |
| **배깅** | 병렬 학습 + 투표/평균 |
| **랜덤포레스트** | 배깅 + 특성 랜덤 선택 |
| **부트스트랩** | 복원 추출로 다양한 데이터셋 |
| **OOB 점수** | 사용 안 된 데이터로 평가 |
| **n_estimators** | 트리 개수 (100~200 권장) |

### 주요 파라미터 가이드

| 파라미터 | 권장값 | 효과 |
|----------|--------|------|
| n_estimators | 100 이상 | 성능 향상 |
| max_depth | None 또는 15~20 | 과대적합 방지 |
| max_features | 'sqrt' (기본) | 다양성 확보 |
| n_jobs | -1 | 전체 CPU 활용 |
| oob_score | True | 성능 추정 |

### sklearn 사용법

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 생성 및 학습
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 평가
accuracy = model.score(X_test, y_test)
oob_score = model.oob_score_

# 특성 중요도
importances = model.feature_importances_

# 개별 트리 접근
trees = model.estimators_
```

### 실무 권장사항

| 상황 | 권장 |
|------|------|
| 빠른 프로토타이핑 | 랜덤포레스트 (튜닝 없이도 좋은 성능) |
| 특성 중요도 분석 | 랜덤포레스트 (신뢰도 높음) |
| 안정적인 예측 필요 | 랜덤포레스트 |
| 해석이 중요 | 의사결정나무 (단일 트리) |
