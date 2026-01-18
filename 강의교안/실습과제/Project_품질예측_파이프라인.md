# 종합 프로젝트: 제조 품질 예측 파이프라인

**관련 차시**: 전체 과정 통합 (1~27차시)
**난이도**: ★★★★☆ (고급)
**예상 소요 시간**: 2~3시간

---

## 프로젝트 개요

이 프로젝트는 27차시 전체 과정에서 배운 내용을 통합하여 **제조 품질 예측 파이프라인**을 구축하는 미니 프로젝트입니다.

**데이터 탐색(EDA) → 전처리 → 모델 학습 → 평가 → 예측**까지의 전 과정을 직접 수행합니다.

---

## 학습 목표

1. 데이터 분석 파이프라인의 전체 흐름 이해하기
2. 복수의 모델을 비교하고 최적 모델을 선택하기
3. 분석 결과를 해석하고 보고서로 정리하기
4. 실무 적용 관점에서 모델의 한계와 개선점 제안하기

---

## 프로젝트 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Step 1: 데이터 탐색 (EDA)                │
│    - 데이터 로드 및 기초 통계량 확인                        │
│    - 분포 시각화 (히스토그램, 상자그림)                     │
│    - 상관관계 분석 (상관행렬, 히트맵)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Step 2: 데이터 전처리                      │
│    - 결측치 확인 및 처리                                    │
│    - 이상치 확인                                            │
│    - 특성 스케일링 (StandardScaler)                         │
│    - 학습/테스트 데이터 분할                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Step 3: 모델 학습                         │
│    - 기준 모델: 로지스틱 회귀                               │
│    - 비교 모델 1: 의사결정나무                              │
│    - 비교 모델 2: 랜덤포레스트                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Step 4: 모델 평가                         │
│    - 혼동행렬, 정밀도, 재현율, F1 스코어                    │
│    - 교차검증 (5-Fold CV)                                   │
│    - 모델 성능 비교표 작성                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Step 5: 결과 해석 및 보고서 작성               │
│    - 최적 모델 선택 및 근거                                 │
│    - 실무 적용 제안                                         │
│    - 한계점 및 향후 개선 방향                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 데이터 탐색 (EDA)

### 1.1 데이터 로드 및 기본 탐색

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# 기본 정보 확인
print("=== 데이터 크기 ===")
print(f"행 수: {df.shape[0]}, 열 수: {df.shape[1]}")

print("\n=== 컬럼 정보 ===")
print(df.info())

print("\n=== 기술통계량 ===")
print(df.describe())
```

**작성할 내용 (1.1)**:
- 데이터의 행/열 수는?
- 수치형 컬럼과 범주형 컬럼은 각각 몇 개인가요?
- 결측치가 있는 컬럼이 있나요?

---

### 1.2 타겟 변수 분포 확인

```python
# 불량 비율 확인
print("=== 타겟 변수 분포 ===")
print(df['Machine failure'].value_counts())
print(f"\n불량 비율: {df['Machine failure'].mean()*100:.2f}%")

# 시각화
plt.figure(figsize=(8, 5))
df['Machine failure'].value_counts().plot(kind='bar', color=['green', 'red'],
                                          edgecolor='black')
plt.xlabel('Machine Failure (0: Normal, 1: Failure)')
plt.ylabel('Count')
plt.title('Target Variable Distribution')
plt.xticks(rotation=0)
plt.show()
```

**작성할 내용 (1.2)**:
- 정상(0)과 불량(1)의 비율은?
- 이 데이터는 균형 데이터인가요, 불균형 데이터인가요?
- 불균형이 모델 학습에 미치는 영향은?

---

### 1.3 특성 분포 시각화

```python
# 주요 특성 컬럼
feature_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# 히스토그램
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

axes[5].axis('off')  # 빈 서브플롯 숨기기
plt.tight_layout()
plt.show()
```

**작성할 내용 (1.3)**:
- 각 특성의 분포 형태를 간단히 설명하세요 (정규분포, 치우침 등)
- 이상치가 의심되는 특성이 있나요?

---

### 1.4 상관관계 분석

```python
# 상관행렬 계산
corr_matrix = df[feature_cols + ['Machine failure']].corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()
```

**작성할 내용 (1.4)**:
- Machine failure와 가장 상관관계가 높은 특성은?
- 특성들 간에 강한 상관관계(|r| > 0.7)가 있나요?
- 다중공선성 우려가 있는 특성 쌍은?

---

## Step 2: 데이터 전처리

### 2.1 특성 선택 및 데이터 분할

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 특성과 타겟 분리
X = df[feature_cols]
y = df['Machine failure']

# 학습/테스트 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")
print(f"\n학습 데이터 불량 비율: {y_train.mean()*100:.2f}%")
print(f"테스트 데이터 불량 비율: {y_test.mean()*100:.2f}%")
```

**작성할 내용 (2.1)**:
- `stratify=y`를 사용하는 이유는?
- 분할 후 학습/테스트 데이터의 불량 비율이 유사한지 확인하세요

---

### 2.2 특성 스케일링

```python
# StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일링 결과 확인
print("=== 스케일링 전 (학습 데이터 평균) ===")
print(X_train.mean().round(2).values)

print("\n=== 스케일링 후 (학습 데이터 평균) ===")
print(X_train_scaled.mean(axis=0).round(4))
```

**작성할 내용 (2.2)**:
- 스케일링 전후의 평균값 변화를 기록하세요
- 어떤 모델에서 스케일링이 필수인가요?

---

## Step 3: 모델 학습

### 3.1 기준 모델: 로지스틱 회귀

```python
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

print("로지스틱 회귀 학습 완료!")
print(f"학습 정확도: {model_lr.score(X_train_scaled, y_train):.4f}")
print(f"테스트 정확도: {model_lr.score(X_test_scaled, y_test):.4f}")
```

---

### 3.2 비교 모델 1: 의사결정나무

```python
from sklearn.tree import DecisionTreeClassifier

# 의사결정나무 모델
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)  # 스케일링 불필요

print("\n의사결정나무 학습 완료!")
print(f"학습 정확도: {model_dt.score(X_train, y_train):.4f}")
print(f"테스트 정확도: {model_dt.score(X_test, y_test):.4f}")
```

---

### 3.3 비교 모델 2: 랜덤포레스트

```python
from sklearn.ensemble import RandomForestClassifier

# 랜덤포레스트 모델
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)  # 스케일링 불필요

print("\n랜덤포레스트 학습 완료!")
print(f"학습 정확도: {model_rf.score(X_train, y_train):.4f}")
print(f"테스트 정확도: {model_rf.score(X_test, y_test):.4f}")
```

**작성할 내용 (3.1~3.3)**:
- 각 모델의 학습/테스트 정확도를 표로 정리하세요
- 학습 정확도와 테스트 정확도의 차이가 큰 모델은? (과적합 의심)

---

## Step 4: 모델 평가

### 4.1 평가 함수 정의

```python
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test, model_name, use_scaled=False):
    """모델 평가 함수"""
    X_eval = X_test_scaled if use_scaled else X_test

    # 예측
    y_pred = model.predict(X_eval)

    # 평가 지표 계산
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = model.score(X_eval, y_test)

    print(f"=== {model_name} ===")
    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 혼동행렬 시각화
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Normal', 'Failure'])
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

    return {'model': model_name, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1': f1}
```

---

### 4.2 각 모델 평가

```python
# 각 모델 평가
results = []
results.append(evaluate_model(model_lr, X_test, y_test, 'Logistic Regression', use_scaled=True))
results.append(evaluate_model(model_dt, X_test, y_test, 'Decision Tree'))
results.append(evaluate_model(model_rf, X_test, y_test, 'Random Forest'))

# 결과 비교표
results_df = pd.DataFrame(results)
print("\n=== 모델 성능 비교표 ===")
print(results_df.to_string(index=False))
```

**작성할 내용 (4.2)**:
- 모델 성능 비교표를 기록하세요
- 정확도가 가장 높은 모델은?
- 재현율이 가장 높은 모델은?

---

### 4.3 교차검증

```python
# 5-Fold 교차검증
print("\n=== 5-Fold 교차검증 ===")

for model, name, use_scaled in [(model_lr, 'Logistic Regression', True),
                                 (model_dt, 'Decision Tree', False),
                                 (model_rf, 'Random Forest', False)]:
    X_cv = X_train_scaled if use_scaled else X_train
    cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='f1')
    print(f"{name}: F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**작성할 내용 (4.3)**:
- 교차검증 F1 스코어가 가장 높은 모델은?
- 표준편차가 작은 모델은 어떤 의미가 있나요?

---

## Step 5: 결과 해석 및 보고서 작성

### 5.1 최종 모델 선택

아래 질문에 답하며 최종 모델을 선택하세요.

**작성할 내용**:

1. **최종 선택 모델**: _______________

2. **선택 이유** (3가지 이상):
   - 이유 1:
   - 이유 2:
   - 이유 3:

3. **제조 현장 관점에서의 해석**:
   - 불량 탐지율(재현율)이 충분한가?
   - 오탐(FP)으로 인한 비용은 허용 가능한가?
   - 모델의 해석 가능성은 중요한가?

---

### 5.2 특성 중요도 분석 (트리 기반 모델)

```python
# 랜덤포레스트 특성 중요도
importance = model_rf.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("=== 특성 중요도 ===")
print(importance_df.to_string(index=False))

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()
```

**작성할 내용 (5.2)**:
- 가장 중요한 특성 Top 3는?
- 이 결과가 제조 공정에서 납득이 되나요?

---

### 5.3 실무 적용 제안

**작성할 내용**:

1. **모델 배포 시 고려사항**:
   - 실시간 예측이 필요한가?
   - 모델 업데이트 주기는?
   - 모니터링 방법은?

2. **한계점**:
   - 데이터의 한계:
   - 모델의 한계:
   - 평가의 한계:

3. **향후 개선 방향**:
   - 추가로 수집하면 좋을 데이터:
   - 시도해볼 만한 다른 모델/기법:
   - 성능 개선 전략:

---

## 평가 기준

| 항목 | 배점 | 세부 기준 |
|------|:----:|----------|
| Step 1: EDA | 20점 | 데이터 탐색 및 시각화 완성도 |
| Step 2: 전처리 | 15점 | 적절한 전처리 적용 |
| Step 3: 모델 학습 | 20점 | 3개 이상 모델 학습 |
| Step 4: 평가 | 25점 | 다양한 평가 지표 활용, 교차검증 |
| Step 5: 보고서 | 20점 | 결과 해석 및 실무 제안의 타당성 |
| **합계** | **100점** | |

---

## 제출물

1. **Jupyter Notebook** (.ipynb): 전체 코드 및 실행 결과
2. **보고서**: 각 Step의 "작성할 내용"에 대한 답변 (노트북 내 마크다운 또는 별도 문서)
3. **시각화 이미지**: 주요 그래프 캡처 (히트맵, 혼동행렬, 특성 중요도 등)

---

## 변형 옵션

### 옵션 A: 다른 데이터셋 사용
- Kaggle의 제조/품질 관련 데이터셋으로 교체
- 예: [Steel Plates Faults Dataset](https://www.kaggle.com/datasets/uciml/faulty-steel-plates)

### 옵션 B: 다른 모델 추가
- XGBoost, LightGBM 등 부스팅 모델 추가
- MLP (딥러닝) 모델 추가

### 옵션 C: 하이퍼파라미터 튜닝
- GridSearchCV 또는 RandomizedSearchCV로 최적 파라미터 탐색

---

## 참고 자료

- 전체 강의 슬라이드 및 실습 코드
- [scikit-learn 문서](https://scikit-learn.org/stable/)
- [Pandas 문서](https://pandas.pydata.org/docs/)
- [Seaborn 문서](https://seaborn.pydata.org/)
