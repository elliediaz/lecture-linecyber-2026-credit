# 번외D: ML 워크플로우 종합 실습

## 학습 목표

| 목표 | 설명 |
|------|------|
| 전체 흐름 경험 | 문제 정의부터 결과 해석까지 한 번에 |
| 단계별 이해 | 6단계 워크플로우 체득 |
| 모델 비교 | 여러 모델을 비교하고 최적 선택 |
| 실습 | Titanic 데이터셋으로 생존 예측 |

---

## ML 워크플로우 6단계

```
1. 문제 정의  →  2. 데이터 탐색  →  3. 전처리
       ↓                                 ↓
6. 결과 해석  ←  5. 모델 평가   ←  4. 모델 학습
```

각 단계가 개별적으로 배운 내용이지만, 전체를 연결하는 경험이 필요함

---

# Part 1: 문제 정의

## 분류 vs 회귀 판단

| 구분 | 분류 | 회귀 |
|------|------|------|
| 목표 | 범주 예측 | 연속값 예측 |
| 예시 | 생존/사망, 불량/양품 | 가격, 온도, 수량 |
| 출력 | 0, 1, 2, ... | 23.5, 100.2, ... |

---

## 타이타닉 문제 분석

- **질문**: "이 승객은 생존했는가?"
- **예측 대상**: survived (0=사망, 1=생존)
- **결론**: 이진 분류 문제

---

# Part 2: 데이터 탐색 (EDA)

## 라이브러리 임포트 및 데이터 로드

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

```python
# Titanic 데이터셋 로드
df = sns.load_dataset('titanic')
print(f"데이터 크기: {df.shape}")
```

**출력**: `데이터 크기: (891, 15)` - 891명의 승객, 15개 컬럼

---

## 데이터 구조 확인

```python
df.info()
```

| 컬럼 | 타입 | 결측치 | 설명 |
|------|------|--------|------|
| survived | int | 0 | 생존 여부 (타겟) |
| pclass | int | 0 | 객실 등급 |
| sex | object | 0 | 성별 |
| age | float | 177 | 나이 |
| fare | float | 0 | 요금 |

age 컬럼에 177개 결측치 존재

---

## 기초 통계량

```python
df.describe()
```

| 항목 | age | fare | survived |
|------|-----|------|----------|
| mean | 29.7 | 32.2 | 0.38 |
| std | 14.5 | 49.7 | 0.49 |
| min | 0.42 | 0 | 0 |
| max | 80 | 512 | 1 |

생존율 약 38%

---

## 결측치 확인

```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '비율(%)': missing_pct
})
print(missing_df[missing_df['결측치 수'] > 0])
```

| 컬럼 | 결측치 수 | 비율(%) |
|------|-----------|---------|
| age | 177 | 19.9 |
| deck | 688 | 77.2 |
| embarked | 2 | 0.2 |

---

## 타겟 변수 분포

```python
print(df['survived'].value_counts())
print(f"\n생존율: {df['survived'].mean():.1%}")
```

| 생존 | 수 | 비율 |
|------|-----|------|
| 0 (사망) | 549 | 62% |
| 1 (생존) | 342 | 38% |

---

## 성별에 따른 생존율

```python
df.groupby('sex')['survived'].mean()
```

| 성별 | 생존율 |
|------|--------|
| female | 0.74 |
| male | 0.19 |

여성의 생존율이 훨씬 높음

---

## 객실 등급에 따른 생존율

```python
df.groupby('pclass')['survived'].mean()
```

| 등급 | 생존율 |
|------|--------|
| 1등석 | 0.63 |
| 2등석 | 0.47 |
| 3등석 | 0.24 |

등급이 높을수록 생존율 높음

---

# Part 3: 데이터 전처리

## 전처리 단계

```
결측치 처리  →  범주형 인코딩  →  특성 선택  →  데이터 분할
```

---

## 결측치 처리

```python
df_processed = df.copy()

# age: 중앙값으로 대체 (이상치에 강건)
age_median = df_processed['age'].median()
df_processed['age'].fillna(age_median, inplace=True)
print(f"age 결측치 → 중앙값({age_median})으로 대체")

# embarked: 최빈값으로 대체
embarked_mode = df_processed['embarked'].mode()[0]
df_processed['embarked'].fillna(embarked_mode, inplace=True)

# deck: 결측치 너무 많음 → 제거
df_processed.drop(columns=['deck'], inplace=True)
```

| 컬럼 | 처리 방법 | 이유 |
|------|----------|------|
| age | 중앙값 대체 | 이상치 영향 줄임 |
| embarked | 최빈값 대체 | 2개만 결측 |
| deck | 컬럼 제거 | 77% 결측 |

---

## 범주형 변수 인코딩

```python
# One-Hot Encoding
df_processed = pd.get_dummies(
    df_processed,
    columns=['sex', 'embarked'],
    drop_first=True  # 다중공선성 방지
)
```

| 변환 전 | 변환 후 |
|---------|---------|
| sex: male/female | sex_male: 0/1 |
| embarked: S/C/Q | embarked_Q, embarked_S |

drop_first=True: male=0이면 female임을 알 수 있으므로 하나만 유지

---

## 특성 선택

```python
features = [
    'pclass',      # 객실 등급
    'age',         # 나이
    'sibsp',       # 형제/배우자 수
    'parch',       # 부모/자녀 수
    'fare',        # 요금
    'sex_male',    # 성별 (1=남성)
    'embarked_Q',  # 승선항 Q
    'embarked_S'   # 승선항 S
]

X = df_processed[features]
y = df_processed['survived']

print(f"특성 수: {len(features)}")
```

총 8개 특성 선택

---

## 데이터 분할

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% 테스트용
    random_state=42,    # 재현성
    stratify=y          # 클래스 비율 유지
)

print(f"학습 데이터: {len(X_train)}건")
print(f"테스트 데이터: {len(X_test)}건")
```

| 구분 | 크기 |
|------|------|
| 학습 데이터 | 712건 (80%) |
| 테스트 데이터 | 179건 (20%) |

stratify=y: 원본의 생존:사망 비율을 학습/테스트에도 유지

---

# Part 4: 모델 학습

## 3가지 모델 정의

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
```

| 모델 | 특징 |
|------|------|
| Logistic Regression | 선형 분류, 빠름 |
| Decision Tree | 규칙 기반, 직관적 |
| Random Forest | 앙상블, 높은 성능 |

---

## 모델 학습 및 비교

```python
results = {}

for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred
    }

    print(f"{name}: {accuracy:.3f}")
```

---

## 모델 비교 결과

| 모델 | 정확도 |
|------|--------|
| Logistic Regression | 0.799 |
| Decision Tree | 0.782 |
| **Random Forest** | **0.821** |

Random Forest가 가장 높은 성능

---

## 최고 성능 모델 선택

```python
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
y_pred_best = results[best_name]['y_pred']

print(f"최고 성능 모델: {best_name}")
print(f"정확도: {results[best_name]['accuracy']:.3f}")
```

---

# Part 5: 모델 평가

## 혼동 행렬

```python
cm = confusion_matrix(y_test, y_pred_best)
print(f"              예측:사망  예측:생존")
print(f"실제:사망     {cm[0,0]:^8}  {cm[0,1]:^8}")
print(f"실제:생존     {cm[1,0]:^8}  {cm[1,1]:^8}")
```

|  | 예측: 사망 | 예측: 생존 |
|--|----------|----------|
| **실제: 사망** | TN (105) | FP (5) |
| **실제: 생존** | FN (27) | TP (42) |

---

## 평가 지표

```python
print(classification_report(y_test, y_pred_best,
                           target_names=['사망(0)', '생존(1)']))
```

| 지표 | 공식 | 의미 |
|------|------|------|
| 정확도 | (TP+TN) / 전체 | 전체 중 맞춘 비율 |
| 정밀도 | TP / (TP+FP) | 생존 예측 중 실제 생존 |
| 재현율 | TP / (TP+FN) | 실제 생존 중 찾아낸 비율 |
| F1 | 조화평균 | 정밀도와 재현율 균형 |

---

## 지표 해석

```python
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"정밀도: {precision:.3f}")
print(f"재현율: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

| 지표 | 값 | 해석 |
|------|-----|------|
| 정밀도 | 0.89 | 생존 예측의 89%가 실제 생존 |
| 재현율 | 0.61 | 실제 생존자의 61%를 찾음 |
| F1 | 0.72 | 정밀도와 재현율의 균형 |

---

# Part 6: 결과 해석

## 특성 중요도 분석

```python
importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

print(importance_df)
```

| 순위 | 특성 | 중요도 |
|------|------|--------|
| 1 | sex_male | 0.28 |
| 2 | fare | 0.26 |
| 3 | age | 0.24 |
| 4 | pclass | 0.12 |
| 5 | sibsp | 0.05 |

---

## 비즈니스 인사이트

| 발견 | 해석 |
|------|------|
| 성별이 가장 중요 | "여성과 아이 먼저" 구조 정책 |
| 요금/등급 중요 | 상위 등급 승객이 좋은 위치 배정 |
| 나이 영향 | 어린이 우선 구조 |

---

## 제조업 적용 예시

| 타이타닉 | 제조업 품질 예측 |
|----------|------------------|
| 성별 → 생존 | 설비 종류 → 불량 |
| 요금 → 생존 | 온도 → 불량 |
| 나이 → 생존 | 작업 시간 → 불량 |

---

# 전체 워크플로우 요약

## 한눈에 보는 코드

```python
# 1. 데이터 로드
df = sns.load_dataset('titanic')

# 2. 전처리
df['age'].fillna(df['age'].median(), inplace=True)
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# 3. 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# 4. 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. 평가
print(classification_report(y_test, model.predict(X_test)))

# 6. 해석
print(model.feature_importances_)
```

---

## 체크리스트

| 단계 | 체크 항목 |
|------|----------|
| 문제 정의 | 분류/회귀 판단 완료 |
| 데이터 탐색 | 크기, 결측치, 분포 확인 |
| 전처리 | 결측치 처리, 인코딩 완료 |
| 데이터 분할 | train/test 분리 (stratify) |
| 모델 학습 | 여러 모델 비교 완료 |
| 평가 | 정확도 + 추가 지표 확인 |
| 해석 | 특성 중요도, 인사이트 도출 |

---

## 핵심 정리

1. **ML 워크플로우**: 문제 정의 → EDA → 전처리 → 학습 → 평가 → 해석
2. **문제 유형**: 분류 vs 회귀 구분이 첫 단계
3. **모델 비교**: 여러 모델을 비교하고 최적 선택
4. **평가 지표**: 정확도 외에 정밀도, 재현율, F1도 확인
5. **인사이트**: 특성 중요도로 비즈니스 해석

---

## 다음 학습

- **번외 E**: 특성 공학으로 더 좋은 특성 만들기
- **번외 F**: sklearn 패턴 마스터 (Pipeline)
- **12차시~**: 개별 모델 심화 학습
