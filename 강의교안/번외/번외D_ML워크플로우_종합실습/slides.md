---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# 번외D: ML 워크플로우 종합 실습
## 데이터에서 예측까지 한 번에!

**AI 기초체력훈련 과정**
제조 AI 예측 모델 개발

---

## 학습 목표

1. 머신러닝 전체 워크플로우를 **처음부터 끝까지** 경험함
2. 문제 정의부터 모델 평가까지 **단계별 흐름** 이해
3. 여러 모델을 **비교**하고 최적 모델 선택
4. 실제 데이터셋(Titanic)으로 **생존 예측** 수행

---

## 왜 전체 워크플로우가 중요한가?

| 문제 | 원인 |
|------|------|
| 각 단계는 알지만 연결이 안 됨 | 개별 학습만 진행 |
| 실제 프로젝트에서 막막함 | 전체 흐름 경험 부족 |
| 어디서 시작해야 할지 모름 | 체계적 프로세스 미숙지 |

**해결책**: 한 번에 전체 흐름을 경험!

---

## ML 워크플로우 6단계

```
┌─────────────────────────────────────────────────────────────┐
│  1. 문제 정의  →  2. 데이터 탐색  →  3. 전처리           │
│       ↓                                    ↓               │
│  6. 결과 해석  ←  5. 모델 평가   ←  4. 모델 학습        │
└─────────────────────────────────────────────────────────────┘
```

---

## 오늘 실습할 문제: 타이타닉 생존 예측

| 항목 | 내용 |
|------|------|
| **문제 유형** | 이진 분류 (생존 0/1) |
| **데이터셋** | Titanic (seaborn 내장) |
| **특성** | 나이, 성별, 객실 등급, 요금 등 |
| **목표** | 승객의 생존 여부 예측 |

---

# Part 1: 문제 정의
## 분류 vs 회귀 판단하기

---

## 문제 유형 판단 기준

| 구분 | 분류 (Classification) | 회귀 (Regression) |
|------|----------------------|-------------------|
| **목표** | 범주 예측 | 연속값 예측 |
| **예시** | 생존/사망, 불량/양품 | 가격, 온도, 수량 |
| **출력** | 0, 1, 2, ... | 23.5, 100.2, ... |
| **평가** | 정확도, F1 | MSE, R² |

---

## 타이타닉 문제 분석

**질문**: "이 승객은 생존했는가?"

| 분석 항목 | 내용 |
|----------|------|
| 예측 대상 | 생존 여부 (Survived) |
| 가능한 값 | 0 (사망), 1 (생존) |
| **결론** | **이진 분류 문제** |

---

## 제조 현장에서의 적용

| 제조 문제 | 유형 | 설명 |
|----------|------|------|
| 불량품 탐지 | 분류 | 양품/불량 |
| 설비 고장 예측 | 분류 | 정상/고장 |
| 생산량 예측 | 회귀 | 수량 (연속값) |
| 품질 점수 예측 | 회귀 | 점수 (연속값) |

---

# Part 2: 데이터 탐색 (EDA)
## 데이터 이해하기

---

## 데이터 로드

```python
import pandas as pd
import seaborn as sns

# Titanic 데이터셋 로드
df = sns.load_dataset('titanic')
print(f"데이터 크기: {df.shape}")
# 출력: 데이터 크기: (891, 15)
```

---

## 데이터 구조 확인

```python
df.info()
```

| 컬럼 | 타입 | 결측치 | 설명 |
|------|------|--------|------|
| survived | int | 0 | 생존 여부 (타겟) |
| pclass | int | 0 | 객실 등급 (1, 2, 3) |
| sex | object | 0 | 성별 |
| age | float | 177 | 나이 |
| fare | float | 0 | 요금 |

---

## 기초 통계량

```python
df.describe()
```

| 항목 | age | fare | survived |
|------|-----|------|----------|
| count | 714 | 891 | 891 |
| mean | 29.7 | 32.2 | 0.38 |
| std | 14.5 | 49.7 | 0.49 |
| min | 0.42 | 0 | 0 |
| max | 80 | 512 | 1 |

**발견**: 생존율 약 38%, 나이 결측치 존재

---

## 타겟 변수 분포

```python
df['survived'].value_counts()
```

| 생존 | 수 | 비율 |
|------|-----|------|
| 0 (사망) | 549 | 62% |
| 1 (생존) | 342 | 38% |

약간의 불균형이 있지만 심각하지 않음

---

## 주요 특성과 생존의 관계

```python
# 성별에 따른 생존율
df.groupby('sex')['survived'].mean()
```

| 성별 | 생존율 |
|------|--------|
| female | 0.74 |
| male | 0.19 |

**발견**: 여성의 생존율이 훨씬 높음!

---

## 객실 등급과 생존의 관계

```python
df.groupby('pclass')['survived'].mean()
```

| 등급 | 생존율 |
|------|--------|
| 1등석 | 0.63 |
| 2등석 | 0.47 |
| 3등석 | 0.24 |

**발견**: 등급이 높을수록 생존율 높음

---

# Part 3: 데이터 전처리
## 모델이 이해할 수 있는 형태로

---

## 전처리 단계

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  결측치 처리 │ → │ 범주형 인코딩│ → │  특성 선택   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 결측치 확인

```python
df.isnull().sum()
```

| 컬럼 | 결측치 수 |
|------|-----------|
| age | 177 |
| embarked | 2 |
| deck | 688 |
| embark_town | 2 |

---

## 결측치 처리 전략

| 컬럼 | 전략 | 이유 |
|------|------|------|
| age | 중앙값 대체 | 연속형, 이상치 영향 줄임 |
| embarked | 최빈값 대체 | 범주형, 2개만 결측 |
| deck | 컬럼 제거 | 결측치 77% 이상 |

---

## 결측치 처리 코드

```python
# 나이: 중앙값으로 대체
df['age'].fillna(df['age'].median(), inplace=True)

# embarked: 최빈값으로 대체
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# deck: 너무 많은 결측치 → 제거
df.drop(columns=['deck'], inplace=True)
```

---

## 범주형 변수 인코딩

| 변환 전 | 변환 후 |
|---------|---------|
| sex: male/female | sex_male: 0/1 |
| embarked: S/C/Q | embarked_C, embarked_Q, embarked_S |

```python
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
```

---

## 최종 특성 선택

```python
# 사용할 특성
features = ['pclass', 'age', 'sibsp', 'parch', 'fare',
            'sex_male', 'embarked_Q', 'embarked_S']

X = df[features]
y = df['survived']

print(f"특성 수: {X.shape[1]}")
# 출력: 특성 수: 8
```

---

## 데이터 분할

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% 테스트용
    random_state=42,    # 재현성
    stratify=y          # 클래스 비율 유지
)

print(f"학습 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")
```

---

# Part 4: 모델 학습
## 여러 모델 비교하기

---

## 비교할 3가지 모델

| 모델 | 특징 | 장점 |
|------|------|------|
| LogisticRegression | 선형 분류 | 빠름, 해석 쉬움 |
| DecisionTree | 규칙 기반 | 직관적, 시각화 가능 |
| RandomForest | 앙상블 | 높은 성능, 과적합 방지 |

---

## 모델 학습 코드

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 3가지 모델 정의
models = {
    'Logistic': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=100)
}
```

---

## 모델 학습 및 예측

```python
results = {}

for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 저장
    accuracy = model.score(X_test, y_test)
    results[name] = accuracy

    print(f"{name}: {accuracy:.3f}")
```

---

## 모델 비교 결과

| 모델 | 정확도 |
|------|--------|
| Logistic | 0.799 |
| DecisionTree | 0.782 |
| **RandomForest** | **0.821** |

**RandomForest가 가장 높은 성능!**

---

# Part 5: 모델 평가
## 정확도 외의 지표들

---

## 왜 정확도만으로는 부족한가?

**불균형 데이터 예시**:
- 불량률 1%인 제품
- "모두 양품"이라고 예측 → 정확도 99%
- 하지만 불량품을 하나도 못 찾음!

---

## 혼동 행렬 (Confusion Matrix)

```python
from sklearn.metrics import confusion_matrix

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
```

|  | 예측: 사망 | 예측: 생존 |
|--|----------|----------|
| **실제: 사망** | TN (105) | FP (5) |
| **실제: 생존** | FN (27) | TP (42) |

---

## 주요 평가 지표

| 지표 | 공식 | 의미 |
|------|------|------|
| 정확도 | (TP+TN) / 전체 | 전체 중 맞춘 비율 |
| 정밀도 | TP / (TP+FP) | 생존 예측 중 실제 생존 |
| 재현율 | TP / (TP+FN) | 실제 생존 중 찾아낸 비율 |
| F1 | 2×정밀도×재현율 / (정밀도+재현율) | 정밀도와 재현율의 조화평균 |

---

## Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

```
              precision  recall  f1-score  support
           0      0.80    0.95      0.87      110
           1      0.89    0.61      0.72       69
    accuracy                        0.82      179
   macro avg      0.85    0.78      0.80      179
```

---

## 제조 현장에서의 해석

| 상황 | 중요 지표 |
|------|----------|
| 불량품 탐지 | **재현율** (불량 놓치면 안 됨) |
| 고가 설비 점검 | **정밀도** (잘못된 경보 비용) |
| 균형 잡힌 판단 | **F1 Score** |

---

# Part 6: 결과 해석
## 비즈니스 인사이트 도출

---

## 특성 중요도 분석

```python
import matplotlib.pyplot as plt

# RandomForest의 특성 중요도
importance = best_model.feature_importances_
features_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)
```

---

## 특성 중요도 결과

| 순위 | 특성 | 중요도 |
|------|------|--------|
| 1 | sex_male | 0.28 |
| 2 | fare | 0.26 |
| 3 | age | 0.24 |
| 4 | pclass | 0.12 |
| 5 | sibsp | 0.05 |

**인사이트**: 성별, 요금, 나이가 생존에 가장 큰 영향

---

## 비즈니스 인사이트

1. **성별**: 여성 우선 구조 정책 ("여성과 아이 먼저")
2. **요금/등급**: 상위 등급 승객이 더 좋은 위치의 객실 배정
3. **나이**: 어린이 우선 구조

---

## 제조업 적용 예시

| 타이타닉 | 제조업 품질 예측 |
|----------|------------------|
| 성별 → 생존 | 설비 종류 → 불량 |
| 요금 → 생존 | 온도 → 불량 |
| 나이 → 생존 | 작업 시간 → 불량 |

---

## 전체 워크플로우 요약

```python
# 1. 데이터 로드
df = sns.load_dataset('titanic')

# 2. 전처리
df['age'].fillna(df['age'].median(), inplace=True)
df = pd.get_dummies(df, columns=['sex', 'embarked'])

# 3. 분할
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 4. 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. 평가
print(classification_report(y_test, model.predict(X_test)))
```

---

## 체크리스트

| 단계 | 체크 항목 |
|------|----------|
| 문제 정의 | 분류/회귀 판단 |
| 데이터 탐색 | 크기, 결측치, 분포 확인 |
| 전처리 | 결측치 처리, 인코딩 |
| 데이터 분할 | train/test 분리 |
| 모델 학습 | 여러 모델 비교 |
| 평가 | 정확도 + 추가 지표 |
| 해석 | 특성 중요도, 인사이트 |

---

## 다음 단계

- **번외 E**: 특성 공학으로 더 좋은 특성 만들기
- **번외 F**: sklearn 패턴 마스터 (Pipeline)
- **12차시~**: 개별 모델 심화 학습

---

## 핵심 정리

1. **ML 워크플로우**: 문제 정의 → EDA → 전처리 → 학습 → 평가 → 해석
2. **문제 유형**: 분류 vs 회귀 구분이 첫 단계
3. **모델 비교**: 여러 모델을 비교하고 최적 선택
4. **평가 지표**: 정확도 외에 정밀도, 재현율, F1도 확인
5. **인사이트**: 특성 중요도로 비즈니스 해석

---

# 실습 시간

Titanic 데이터셋으로 전체 워크플로우를
직접 실행해봅시다!

---

# Q&A

질문이 있으신가요?
