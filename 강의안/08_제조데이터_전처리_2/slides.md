---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 8차시'
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

# 제조 데이터 전처리 (2)

## 8차시 | Part II. 기초 수리와 데이터 분석

**스케일링과 인코딩**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **스케일링**(정규화, 표준화)의 필요성을 이해한다
2. **범주형 데이터**를 인코딩하는 방법을 적용한다
3. sklearn의 **전처리 도구**를 활용한다

---

# 왜 스케일링이 필요한가?

## 변수 간 스케일 차이 문제

```
온도: 80 ~ 100 (범위 20)
생산량: 1000 ~ 1500 (범위 500)
습도: 40 ~ 80 (범위 40)
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
- **평균 = 0, 표준편차 = 1**
- 대부분의 ML 알고리즘에 적합
- 이상치에 다소 민감

---

# 정규화 (Min-Max Scaling)

## 0~1 범위로 변환

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 결과: 0 ~ 1 범위
```

### 특징
- **범위: [0, 1]**
- 원래 분포 형태 유지
- 신경망에서 주로 사용

---

# 스케일링 비교

## 상황별 선택

| 방법 | 결과 | 사용 시점 |
|------|------|----------|
| StandardScaler | 평균0, 표준편차1 | 일반적인 ML 모델 |
| MinMaxScaler | [0, 1] | 신경망, 이미지 |
| RobustScaler | 중앙값 기준 | 이상치가 많을 때 |

> 대부분의 경우 **StandardScaler**를 먼저 시도

---

# 범주형 데이터

## 숫자가 아닌 데이터

```python
라인: ['A', 'B', 'C']
등급: ['상', '중', '하']
불량유형: ['스크래치', '찍힘', '변색']
```

### 문제점
- 대부분의 ML 모델은 **숫자만 입력 가능**
- 범주를 숫자로 변환해야 함

---

# 레이블 인코딩

## 순서가 있는 범주에 사용

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['등급_숫자'] = le.fit_transform(df['등급'])

# 상 → 0, 중 → 1, 하 → 2 (알파벳순)
```

### 주의점
- 숫자 크기에 의미가 생김 (0 < 1 < 2)
- 트리 기반 모델에서는 괜찮음
- 선형 모델에서는 원-핫 인코딩 권장

---

# 원-핫 인코딩

## 순서가 없는 범주에 사용

```python
# pandas 방법 (간편)
df_encoded = pd.get_dummies(df, columns=['라인'])

# sklearn 방법
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['라인']])
```

### 결과
```
라인_A  라인_B  라인_C
  1      0      0
  0      1      0
```

---

# 인코딩 선택 가이드

## 데이터 유형별

| 데이터 유형 | 권장 방법 | 예시 |
|------------|----------|------|
| 순서 있는 범주 | 레이블 인코딩 | 등급(상/중/하) |
| 순서 없는 범주 | 원-핫 인코딩 | 라인(A/B/C) |
| 고유값 많음 | 타겟/빈도 인코딩 | 제품코드 |

---

# 이론 정리

## 핵심 포인트

### 스케일링
- **StandardScaler**: 평균 0, 표준편차 1
- **MinMaxScaler**: 0~1 범위

### 인코딩
- **LabelEncoder**: 순서 있는 범주
- **OneHotEncoder / get_dummies**: 순서 없는 범주

### 주의사항
- 학습 데이터로 fit, 테스트 데이터는 transform만

---

# - 실습편 -

## 8차시

**스케일링과 인코딩 실습**

---

# 실습 개요

## 전처리 도구 활용

### 실습 목표
- StandardScaler, MinMaxScaler 적용
- LabelEncoder, OneHotEncoder 적용
- 전처리 전후 데이터 비교

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 데이터 준비

## 샘플 데이터 생성

```python
np.random.seed(42)
n = 100

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '생산량': np.random.normal(1200, 50, n),
    '습도': np.random.normal(60, 10, n),
    '라인': np.random.choice(['A', 'B', 'C'], n),
    '등급': np.random.choice(['상', '중', '하'], n)
})

print(df.head())
print(df.describe())
```

---

# 실습 2: 스케일 차이 확인

## 변수별 범위 비교

```python
print("=== 변수별 범위 ===")
for col in ['온도', '생산량', '습도']:
    print(f"{col}: {df[col].min():.1f} ~ {df[col].max():.1f}")
    print(f"       범위: {df[col].max() - df[col].min():.1f}")
```

> 생산량의 범위가 다른 변수보다 훨씬 큼

---

# 실습 3: 표준화 적용

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# 수치형 열 선택
numeric_cols = ['온도', '생산량', '습도']
X = df[numeric_cols].values

# 표준화
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# DataFrame으로 변환
df_std = pd.DataFrame(X_std, columns=numeric_cols)

print("=== 표준화 후 ===")
print(df_std.describe().round(2))
```

---

# 실습 4: 정규화 적용

## MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# 정규화
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# DataFrame으로 변환
df_mm = pd.DataFrame(X_mm, columns=numeric_cols)

print("=== 정규화 후 ===")
print(df_mm.describe().round(2))
```

---

# 실습 5: 스케일링 비교 시각화

## 전/후 분포

```python
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, col in enumerate(numeric_cols):
    # 원본
    axes[i, 0].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i, 0].set_title(f'{col} (원본)')

    # 표준화
    axes[i, 1].hist(df_std[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i, 1].set_title(f'{col} (표준화)')

    # 정규화
    axes[i, 2].hist(df_mm[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i, 2].set_title(f'{col} (정규화)')

plt.tight_layout()
plt.show()
```

---

# 실습 6: 레이블 인코딩

## 등급 변환

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['등급_숫자'] = le.fit_transform(df['등급'])

print("=== 레이블 인코딩 결과 ===")
print(df[['등급', '등급_숫자']].drop_duplicates().sort_values('등급_숫자'))

# 클래스 확인
print(f"\n클래스: {le.classes_}")
```

---

# 실습 7: 원-핫 인코딩 (pandas)

## get_dummies 활용

```python
# pandas 방법 (간편)
df_onehot = pd.get_dummies(df, columns=['라인'], prefix='라인')

print("=== 원-핫 인코딩 결과 ===")
print(df_onehot[['라인_A', '라인_B', '라인_C']].head(10))
print(f"\n열 목록: {df_onehot.columns.tolist()}")
```

---

# 실습 8: 원-핫 인코딩 (sklearn)

## OneHotEncoder 활용

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
line_encoded = encoder.fit_transform(df[['라인']])

# 열 이름 생성
feature_names = encoder.get_feature_names_out(['라인'])
df_line_enc = pd.DataFrame(line_encoded, columns=feature_names)

print("=== sklearn OneHotEncoder 결과 ===")
print(df_line_enc.head())
print(f"\n카테고리: {encoder.categories_}")
```

---

# 실습 9: 종합 전처리

## 수치형 + 범주형 통합

```python
# 최종 데이터프레임 구성
df_final = pd.DataFrame()

# 수치형: 표준화
df_final[['온도_std', '생산량_std', '습도_std']] = df_std

# 범주형: 원-핫 인코딩
df_final = pd.concat([df_final, df_onehot[['라인_A', '라인_B', '라인_C']]], axis=1)

# 등급: 레이블 인코딩
df_final['등급'] = df['등급_숫자']

print("=== 최종 전처리 데이터 ===")
print(df_final.head())
print(f"\n열 수: {len(df_final.columns)}")
```

---

# 실습 10: 역변환

## 원래 값으로 복원

```python
# 표준화 역변환
X_original = scaler_std.inverse_transform(X_std)
df_restored = pd.DataFrame(X_original, columns=numeric_cols)

print("=== 역변환 결과 ===")
print("원본:")
print(df[numeric_cols].head(3).round(2))
print("\n복원:")
print(df_restored.head(3).round(2))
```

> 모델 예측 후 원래 스케일로 복원할 때 사용

---

# 실습 정리

## 핵심 체크포인트

### 스케일링
- [ ] StandardScaler().fit_transform(X)
- [ ] MinMaxScaler().fit_transform(X)
- [ ] 역변환: scaler.inverse_transform()

### 인코딩
- [ ] LabelEncoder().fit_transform() (순서 있음)
- [ ] pd.get_dummies() 또는 OneHotEncoder() (순서 없음)

---

# 다음 차시 예고

## 9차시: 제조 데이터 탐색 분석 종합

### 학습 내용
- EDA 전체 워크플로우
- 데이터 이해부터 인사이트 도출까지
- 제조 데이터 종합 분석 실습

### 준비물
- 1~8차시 내용 복습
- 실습 데이터 준비

---

# 정리 및 Q&A

## 오늘의 핵심

1. **스케일링**: StandardScaler(표준화), MinMaxScaler(정규화)
2. **인코딩**: LabelEncoder(순서O), OneHotEncoder(순서X)
3. **주의**: fit은 학습 데이터만, 테스트는 transform만

### 자주 하는 실수
- 테스트 데이터에도 fit 적용 (데이터 누수)
- 원-핫 인코딩 후 다중공선성 문제

---

# 감사합니다

## 8차시: 제조 데이터 전처리 (2)

**다음 시간에 EDA 종합을 배워봅시다!**
