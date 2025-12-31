---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 9차시'
footer: '© 2026 AI 기초체력훈련'
---

# 데이터 전처리 실무 (2)

## 9차시 | AI 기초체력훈련 (Pre AI-Campus)

**스케일링과 인코딩**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **스케일링**(정규화, 표준화)의 필요성을 이해한다
2. **범주형 데이터**를 인코딩하는 방법을 적용한다
3. sklearn의 **전처리 도구**를 활용한다

---

# 왜 스케일링이 필요한가?

## 변수 간 스케일 차이

```
온도: 80 ~ 100 (범위 20)
생산량: 1000 ~ 1500 (범위 500)
습도: 0 ~ 100 (범위 100)
```

### 문제점
- 스케일이 큰 변수가 모델에 과도한 영향
- 거리 기반 알고리즘(KNN 등)에서 왜곡
- 경사하강법 수렴 속도 저하

---

# 표준화 (Standardization)

## Z-score 변환

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 결과: 평균 0, 표준편차 1
```

### 특징
- 평균 = 0, 표준편차 = 1
- 이상치에 민감
- 대부분의 ML 알고리즘에 적합

---

# 정규화 (Normalization)

## Min-Max 스케일링

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 결과: 0 ~ 1 범위
```

### 특징
- 범위: [0, 1]
- 원래 분포 형태 유지
- 신경망에서 주로 사용

---

# 스케일링 비교

## 상황별 선택

| 방법 | 범위 | 사용 시점 |
|------|------|----------|
| StandardScaler | 평균0, 표준편차1 | 일반적인 ML 모델 |
| MinMaxScaler | [0, 1] | 신경망, 이미지 |
| RobustScaler | 중앙값 기준 | 이상치가 많을 때 |

---

# 범주형 데이터

## 숫자가 아닌 데이터

```python
라인: ['A', 'B', 'C']
등급: ['상', '중', '하']
불량유형: ['스크래치', '찍힘', '변색']
```

### 문제점
- 대부분의 ML 모델은 숫자만 입력 가능
- 범주를 숫자로 변환해야 함

---

# 레이블 인코딩

## 순서가 있는 범주

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['등급_encoded'] = le.fit_transform(df['등급'])

# 하 → 0, 상 → 1, 중 → 2 (알파벳순)
```

### 주의
- 순서 관계가 의미 없어도 숫자로 표현됨
- 트리 기반 모델에서는 괜찮음

---

# 원-핫 인코딩

## 순서가 없는 범주

```python
# pandas
df_encoded = pd.get_dummies(df, columns=['라인'])

# sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['라인']])
```

### 결과
```
라인_A  라인_B  라인_C
  1      0      0
  0      1      0
  0      0      1
```

---

# 인코딩 선택 가이드

## 상황별 방법

| 데이터 유형 | 권장 방법 |
|------------|----------|
| 순서 있는 범주 (등급) | 레이블/순서 인코딩 |
| 순서 없는 범주 (라인) | 원-핫 인코딩 |
| 고유값 많은 범주 | 타겟 인코딩, 빈도 인코딩 |

---

# sklearn Pipeline

## 전처리 자동화

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

---

# ColumnTransformer

## 열별 다른 전처리

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['온도', '습도']),
        ('cat', OneHotEncoder(), ['라인'])
    ]
)

X_processed = preprocessor.fit_transform(X)
```

---

# 학습 정리

## 오늘 배운 내용

### 1. 스케일링
- StandardScaler: 평균0, 표준편차1
- MinMaxScaler: 0~1 범위

### 2. 인코딩
- LabelEncoder: 순서 있는 범주
- OneHotEncoder: 순서 없는 범주

### 3. Pipeline
- 전처리 자동화
- ColumnTransformer로 열별 처리

---

# 다음 차시 예고

## 10차시: 탐색적 데이터분석(EDA) 종합

- EDA 전체 흐름
- 데이터 이해부터 인사이트 도출까지
- 제조 데이터 종합 분석 실습

---

# 감사합니다

## AI 기초체력훈련 9차시

**데이터 전처리 실무 (2)**

다음 시간에 만나요!
