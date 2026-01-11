# [10차시] 제조 데이터 전처리 (2): 스케일링, 인코딩, 파이프라인

## 학습 목표

| 번호 | 목표 |
|:----:|------|
| 1 | 스케일링(정규화, 표준화)의 필요성을 이해함 |
| 2 | 범주형 데이터 인코딩 방법(LabelEncoder, OneHotEncoder)을 적용함 |
| 3 | sklearn 전처리 도구(ColumnTransformer, Pipeline)를 활용함 |

---

## 실습 데이터: Diamonds 데이터셋

본 실습에서는 seaborn의 Diamonds 데이터셋을 활용함

### 데이터 특성

| 변수 | 설명 | 유형 |
|------|------|------|
| carat | 다이아몬드 무게 (캐럿) | 수치형 |
| cut | 컷팅 품질 | 순서형 범주 |
| color | 색상 등급 | 순서형 범주 |
| clarity | 투명도 | 순서형 범주 |
| depth | 깊이 비율 (%) | 수치형 |
| table | 테이블 너비 (%) | 수치형 |
| price | 가격 (USD) | 수치형 (타겟) |
| x, y, z | 크기 (mm) | 수치형 |

### 범주형 변수 순서

```
cut:     Fair < Good < Very Good < Premium < Ideal
color:   J < I < H < G < F < E < D (색상이 투명할수록 좋음)
clarity: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF
```

---

## Part 1: 스케일링(정규화, 표준화)의 필요성 이해

### 1.1 스케일 차이가 발생하는 이유

제조 데이터에서 변수마다 단위와 범위가 다름

```
온도:     80 ~ 100      (범위 20)
생산량:   1000 ~ 1500   (범위 500)
불량률:   0.01 ~ 0.05   (범위 0.04)
```

스케일 차이의 문제점:
- 스케일이 큰 변수가 모델에 과도한 영향을 미침
- 거리 기반 알고리즘(KNN, SVM)에서 결과 왜곡이 발생함
- 경사하강법의 수렴 속도가 저하됨

---

### 1.2 실습 데이터 로드

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
```

---

```python
# Diamonds 데이터셋 로드
df = sns.load_dataset('diamonds')

# 분석을 위해 5000개 샘플링
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

print(f"데이터 형태: {df.shape}")
print(df.head(10))
```

#### 실행 결과 해설

- 53,940개 중 5,000개를 샘플링하여 사용함
- 수치형 7개 + 범주형 3개 = 총 10개 컬럼으로 구성됨
- price가 타겟 변수가 됨

---

### 1.3 스케일 차이 확인

```python
numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

print("=== 변수별 통계 ===")
print(df[numeric_cols].describe().round(2))

print("\n=== 변수별 범위 ===")
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    print(f"  {col:8}: {col_min:>10.2f} ~ {col_max:>10.2f} (범위: {col_range:>10.2f})")
```

#### 결과 해설

| 변수 | 범위 | 특징 |
|------|------|------|
| price | 300 ~ 18,000+ | 범위가 매우 넓음 |
| carat | 0.2 ~ 5.0 | 범위가 좁음 |
| depth, table | 50 ~ 70 | 범위가 좁음 |

price 변수가 다른 변수 대비 10~100배 큰 범위를 가지므로, 스케일링 없이 모델에 입력하면 price가 과도한 영향을 미침

---

### 1.4 스케일 차이 시각화

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

main_cols = ['carat', 'depth', 'table', 'price', 'x', 'y']
for i, col in enumerate(main_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[i].legend()

plt.tight_layout()
plt.show()
```

#### 시각화 해설

- price: 왼쪽으로 치우친 분포 (right-skewed)
- carat: price와 유사하게 치우침
- depth, table: 정규분포에 가까움
- x, y: 비슷한 분포 (다이아몬드 크기)

---

### 1.5 스케일링 방법 비교

| 방법 | 공식 | 결과 | 적용 시점 |
|------|------|------|----------|
| **StandardScaler** | (X - mean) / std | 평균=0, std=1 | 일반적인 ML |
| **MinMaxScaler** | (X - min) / (max - min) | [0, 1] | 신경망, 이미지 |
| **RobustScaler** | (X - Q2) / IQR | 중앙값 기준 | 이상치가 많을 때 |

---

### 1.6 표준화 (StandardScaler) 적용

```python
scale_cols = ['carat', 'depth', 'table', 'price']
X_numeric = df[scale_cols].values

# StandardScaler 적용
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X_numeric)

# DataFrame으로 변환
df_std = pd.DataFrame(X_std, columns=[f'{col}_std' for col in scale_cols])

print("=== 표준화 후 통계 ===")
print(df_std.describe().round(4))

print("\n=== 학습된 파라미터 ===")
for col, mean, std in zip(scale_cols, scaler_std.mean_, scaler_std.scale_):
    print(f"  {col}: mean={mean:.2f}, std={std:.2f}")
```

#### 결과 해설

- 모든 변수의 평균이 0에 가까워짐
- 모든 변수의 표준편차가 1에 가까워짐
- scaler_std.mean_과 scaler_std.scale_에 학습된 파라미터가 저장됨

---

### 1.7 정규화 (MinMaxScaler) 적용

```python
# MinMaxScaler 적용
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X_numeric)

# DataFrame으로 변환
df_mm = pd.DataFrame(X_mm, columns=[f'{col}_mm' for col in scale_cols])

print("=== 정규화 후 통계 ===")
print(df_mm.describe().round(4))

print("\n=== 학습된 파라미터 ===")
for col, min_val, max_val in zip(scale_cols, scaler_mm.data_min_, scaler_mm.data_max_):
    print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}")
```

#### 결과 해설

- 모든 변수의 최솟값이 0, 최댓값이 1이 됨
- 값의 범위가 [0, 1]로 통일됨
- 이상치가 있으면 대부분의 값이 한쪽으로 몰릴 수 있음

---

### 1.8 RobustScaler 적용 (이상치에 강건)

```python
# RobustScaler 적용
scaler_rb = RobustScaler()
X_rb = scaler_rb.fit_transform(X_numeric)

# DataFrame으로 변환
df_rb = pd.DataFrame(X_rb, columns=[f'{col}_rb' for col in scale_cols])

print("=== RobustScaler 후 통계 ===")
print(df_rb.describe().round(4))
```

#### RobustScaler 특징

```
공식: X_scaled = (X - Q2) / (Q3 - Q1)

- 중앙값(Q2)과 IQR 사용
- 이상치의 영향을 최소화함
- price처럼 치우친 분포에 적합함
```

---

### 1.9 스케일링 비교 시각화

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 원본
axes[0].hist(df['price'], bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[0].set_title('price - Original')
axes[0].set_xlabel('Price (USD)')

# StandardScaler
axes[1].hist(df_std['price_std'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('price - StandardScaler')
axes[1].set_xlabel('Standardized Price')

# MinMaxScaler
axes[2].hist(df_mm['price_mm'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[2].set_title('price - MinMaxScaler')
axes[2].set_xlabel('Normalized Price')

# RobustScaler
axes[3].hist(df_rb['price_rb'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[3].set_title('price - RobustScaler')
axes[3].set_xlabel('Robust Scaled Price')

plt.tight_layout()
plt.show()
```

#### 시각화 해설

- 분포의 모양은 동일하게 유지됨
- 축의 범위만 변경됨
- RobustScaler는 중앙값 기준으로 변환되어 이상치 영향이 적음

---

### 1.10 역변환 (Inverse Transform)

```python
# 표준화 역변환
X_restored = scaler_std.inverse_transform(X_std)

print("=== 원본 vs 역변환 비교 (처음 3행) ===")
print("\n원본:")
print(df[scale_cols].head(3).round(2))
print("\n역변환:")
print(pd.DataFrame(X_restored[:3], columns=scale_cols).round(2))
```

#### 역변환 활용

- 예측 결과를 원래 단위로 표시할 때 사용함
- 스케일러 객체에 학습된 파라미터가 있어야 역변환 가능함
- 모델 배포 시 전처리기와 모델을 함께 저장해야 함

---

### 1.11 스케일링 주의사항

```
핵심 원칙: fit은 학습 데이터만, transform은 모든 데이터에
```

```python
# 잘못된 방법 (데이터 누수 발생)
scaler.fit(전체_데이터)  # 테스트 정보가 학습에 포함됨

# 올바른 방법
scaler.fit(X_train)      # 학습 데이터만으로 fit
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 같은 기준으로 변환
```

---

## Part 2: 범주형 데이터 인코딩 방법

### 2.1 범주형 데이터 확인

```python
categorical_cols = ['cut', 'color', 'clarity']

print("=== 범주형 컬럼 고유값 ===")
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"\n{col}:")
    print(f"  고유값 ({len(unique_values)}개): {list(unique_values)}")
```

#### 범주형 변수 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **명목형 (Nominal)** | 순서 없음 | 라인(A/B/C), 색상 |
| **순서형 (Ordinal)** | 순서 있음 | 등급(상/중/하), 만족도 |

Diamonds 데이터셋의 cut, color, clarity는 모두 순서형 범주임

---

### 2.2 레이블 인코딩 (LabelEncoder)

```python
from sklearn.preprocessing import LabelEncoder

# cut 인코딩
le = LabelEncoder()
df['cut_encoded'] = le.fit_transform(df['cut'])

print("=== 레이블 인코딩 결과 (cut) ===")
print(df[['cut', 'cut_encoded']].drop_duplicates().sort_values('cut_encoded'))

print(f"\n=== 클래스 순서 ===")
print(f"  {le.classes_}")

# 역변환
original = le.inverse_transform([0, 1, 2, 3, 4])
print(f"\n=== 역변환 예시 ===")
print(f"  [0, 1, 2, 3, 4] -> {list(original)}")
```

#### 레이블 인코딩 주의사항

```
LabelEncoder는 알파벳 순서로 인코딩됨!

실제 순서: Fair < Good < Very Good < Premium < Ideal
인코딩 결과: Fair(0) < Good(1) < Ideal(2) < Premium(3) < Very Good(4)

=> 실제 순서와 다름!
```

---

### 2.3 순서형 인코딩 (OrdinalEncoder) - 올바른 순서

```python
from sklearn.preprocessing import OrdinalEncoder

# cut의 올바른 순서 정의
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

# OrdinalEncoder로 순서 지정
ordinal_encoder = OrdinalEncoder(categories=[cut_order])
df['cut_ordinal'] = ordinal_encoder.fit_transform(df[['cut']])

print("=== 순서형 인코딩 결과 (cut) ===")
print(df[['cut', 'cut_ordinal']].drop_duplicates().sort_values('cut_ordinal'))
```

#### OrdinalEncoder vs LabelEncoder

| 항목 | LabelEncoder | OrdinalEncoder |
|------|-------------|----------------|
| 순서 지정 | 불가능 (알파벳순) | 가능 |
| 입력 형식 | 1차원 배열 | 2차원 배열 |
| 사용 시점 | 타겟 변수 인코딩 | 특성 변수 인코딩 |

---

### 2.4 원-핫 인코딩 (pandas get_dummies)

```python
# color 원-핫 인코딩
df_onehot = pd.get_dummies(df, columns=['color'], prefix='color')

print("=== 원-핫 인코딩 결과 (color) ===")
color_cols = [col for col in df_onehot.columns if col.startswith('color_')]
print(df_onehot[color_cols].head(10))

print(f"\n=== 새로운 컬럼 ({len(color_cols)}개) ===")
print(f"  {color_cols}")
```

#### 원-핫 인코딩 원리

```
원본: [D, E, F, G, H, I, J]

변환 후:
    color_D  color_E  color_F  color_G  color_H  color_I  color_J
0      1        0        0        0        0        0        0
1      0        1        0        0        0        0        0
2      0        0        1        0        0        0        0
...

- 해당 범주면 1, 아니면 0
- 순서/크기 관계가 없어짐
```

---

### 2.5 drop_first 옵션 (다중공선성 방지)

```python
# 다중공선성 방지를 위해 첫 번째 컬럼 제거
df_onehot_drop = pd.get_dummies(df, columns=['color'], prefix='color', drop_first=True)

print("=== drop_first=True 결과 ===")
color_cols_drop = [col for col in df_onehot_drop.columns if col.startswith('color_')]
print(df_onehot_drop[color_cols_drop].head(10))
print(f"\n=> 첫 번째 범주(color_D)가 제거됨: {len(color_cols_drop)}개 컬럼")
```

#### 다중공선성 문제

```
color_D + color_E + ... + color_J = 1 (항상)

문제점:
- 변수들 간 완벽한 선형 관계가 존재함
- 선형 모델에서 계수 추정이 불안정해짐

해결책:
- drop_first=True로 하나의 컬럼 제거
- 모든 값이 0이면 제거된 범주임을 알 수 있음
```

---

### 2.6 원-핫 인코딩 (sklearn OneHotEncoder)

```python
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder 적용
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
color_encoded = encoder.fit_transform(df[['color']])

# 컬럼명 생성
feature_names = encoder.get_feature_names_out(['color'])
df_color_enc = pd.DataFrame(color_encoded, columns=feature_names)

print("=== OneHotEncoder 결과 ===")
print(df_color_enc.head(10))

print(f"\n=== 카테고리 ===")
print(f"  {encoder.categories_}")
```

#### get_dummies vs OneHotEncoder

| 항목 | pd.get_dummies | OneHotEncoder |
|------|---------------|---------------|
| 사용 편의성 | 간편함 | 코드 복잡 |
| 새 범주 처리 | 어려움 | handle_unknown='ignore' |
| Pipeline 통합 | 어려움 | 쉬움 |
| 적합한 상황 | EDA, 빠른 실험 | 프로덕션 |

---

### 2.7 빈도 인코딩 (Frequency Encoding)

```python
# clarity별 빈도 계산
freq = df['clarity'].value_counts(normalize=True)
df['clarity_freq'] = df['clarity'].map(freq)

print("=== 빈도 인코딩 결과 (clarity) ===")
print(df[['clarity', 'clarity_freq']].drop_duplicates().sort_values('clarity_freq', ascending=False))
```

#### 빈도 인코딩 활용

- 고유값이 많은 범주(제품코드 1000개+)에 적합함
- 원-핫 인코딩 시 컬럼 수가 폭발하는 것을 방지함
- 희귀한 범주일수록 낮은 값을 가짐

---

### 2.8 인코딩 선택 가이드

| 데이터 유형 | 예시 | 권장 방법 |
|------------|------|----------|
| 순서 있는 범주 | 등급(상/중/하) | OrdinalEncoder |
| 순서 없는 범주 | 라인(A/B/C) | OneHotEncoder |
| 고유값 많음 | 제품코드 1000개 | 빈도 인코딩 |
| 이진 범주 | 성별(남/여) | LabelEncoder or OneHotEncoder |

---

## Part 3: sklearn 전처리 도구 활용

### 3.1 ColumnTransformer 기본 사용

컬럼별로 다른 전처리를 적용하는 도구임

```python
from sklearn.compose import ColumnTransformer

# 컬럼별 전처리 정의
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# 적용
X_processed = preprocessor.fit_transform(df)

print(f"=== 전처리 전/후 shape ===")
print(f"  전: {df[numeric_features + categorical_features].shape}")
print(f"  후: {X_processed.shape}")
```

#### ColumnTransformer 구조

```
입력 데이터
[carat | depth | table | x | y | z | cut | color]
                    |
           ColumnTransformer
    [StandardScaler]   [OneHotEncoder]
       (수치형)           (범주형)
                    |
출력 데이터
[carat_std | depth_std | ... | cut_Fair | cut_Good | ... | color_D | ...]
```

---

### 3.2 수치형 전처리 파이프라인

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 결측치 + 스케일링 파이프라인
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler())                    # 스케일링
])

print("=== 수치형 전처리 파이프라인 ===")
print(numeric_transformer)
```

---

### 3.3 범주형 전처리 파이프라인

```python
# 결측치 + 인코딩 파이프라인
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측치: 최빈값
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

print("=== 범주형 전처리 파이프라인 ===")
print(categorical_transformer)
```

---

### 3.4 종합 전처리 파이프라인

```python
# ColumnTransformer로 통합
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("=== 종합 전처리기 ===")
print(full_preprocessor)
```

#### 전처리 파이프라인 구조

```
원본 데이터
    |
ColumnTransformer
[수치형 파이프라인]  [범주형 파이프라인]
  - Imputer           - Imputer
  - Scaler            - Encoder
    |
전처리된 데이터
```

---

### 3.5 Pipeline: 전처리 + 모델 연결

```python
# 데이터 분할 (가격 예측)
X = df[numeric_features + categorical_features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"=== 데이터 분할 ===")
print(f"  학습 데이터: {X_train.shape}")
print(f"  테스트 데이터: {X_test.shape}")

# 전체 파이프라인 (전처리 + 모델)
full_pipeline = Pipeline([
    ('preprocessor', full_preprocessor),
    ('regressor', LinearRegression())
])

# 학습
full_pipeline.fit(X_train, y_train)

# 평가
train_score = full_pipeline.score(X_train, y_train)
test_score = full_pipeline.score(X_test, y_test)

print(f"\n=== 모델 성능 (R^2 Score) ===")
print(f"  학습 데이터: {train_score:.4f}")
print(f"  테스트 데이터: {test_score:.4f}")
```

#### Pipeline 장점

1. **코드 간결화**: fit/predict 한 번에 처리함
2. **데이터 누수 방지**: 학습/테스트 분리가 자동으로 됨
3. **교차 검증 용이**: GridSearchCV와 쉽게 연동됨

---

### 3.6 새 데이터 예측

```python
# 새로운 다이아몬드 데이터
new_data = pd.DataFrame({
    'carat': [0.5, 1.0, 2.0],
    'depth': [61.5, 62.0, 60.5],
    'table': [57.0, 58.0, 56.0],
    'x': [5.0, 6.5, 8.0],
    'y': [5.0, 6.5, 8.0],
    'z': [3.1, 4.0, 4.8],
    'cut': ['Ideal', 'Very Good', 'Premium'],
    'color': ['E', 'G', 'F']
})

print("=== 새로운 다이아몬드 데이터 ===")
print(new_data)

# 예측 (전처리 자동 적용)
predictions = full_pipeline.predict(new_data)

print("\n=== 가격 예측 결과 ===")
for i, (_, row) in enumerate(new_data.iterrows()):
    print(f"  {row['carat']}ct {row['cut']} {row['color']} -> ${predictions[i]:,.0f}")
```

#### 예측 과정

Pipeline이 자동으로 수행하는 작업:
1. 수치형 변수 → SimpleImputer → StandardScaler
2. 범주형 변수 → SimpleImputer → OneHotEncoder
3. 전처리된 데이터 → LinearRegression.predict

---

### 3.7 파이프라인 저장 및 로드

```python
import joblib

# 저장
joblib.dump(full_pipeline, 'diamond_price_pipeline.pkl')
print("=== 파이프라인 저장 완료 ===")
print("  파일: diamond_price_pipeline.pkl")

# 로드
loaded_pipeline = joblib.load('diamond_price_pipeline.pkl')
print("=== 파이프라인 로드 완료 ===")

# 로드된 파이프라인으로 예측
loaded_predictions = loaded_pipeline.predict(new_data)
print(f"=== 로드된 파이프라인 예측 결과 ===")
print(f"  {loaded_predictions.round(0)}")
```

#### 저장 시 주의사항

```
반드시 전처리기와 모델을 함께 저장해야 함!

잘못된 방법:
  - 모델만 저장
  - 전처리된 데이터로만 학습

올바른 방법:
  - Pipeline 전체를 joblib으로 저장
  - 새 데이터에 동일한 전처리가 자동 적용됨
```

---

### 3.8 커스텀 전처리 파이프라인 함수

```python
def create_preprocessing_pipeline(numeric_features, categorical_features,
                                   scaler='standard', imputer_strategy='median'):
    """
    범용 전처리 파이프라인 생성 함수

    Parameters:
    -----------
    numeric_features : list - 수치형 컬럼 목록
    categorical_features : list - 범주형 컬럼 목록
    scaler : str - 스케일러 종류 ('standard', 'minmax', 'robust')
    imputer_strategy : str - 결측치 전략

    Returns:
    --------
    preprocessor : ColumnTransformer
    """
    # 스케일러 선택
    if scaler == 'standard':
        scaler_obj = StandardScaler()
    elif scaler == 'minmax':
        scaler_obj = MinMaxScaler()
    elif scaler == 'robust':
        scaler_obj = RobustScaler()
    else:
        scaler_obj = StandardScaler()

    # 수치형 전처리
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('scaler', scaler_obj)
    ])

    # 범주형 전처리
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 통합
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
```

---

```python
# 사용 예시: RobustScaler 적용 (가격 이상치 때문)
custom_preprocessor = create_preprocessing_pipeline(
    numeric_features=['carat', 'depth', 'table', 'x', 'y', 'z'],
    categorical_features=['cut', 'color', 'clarity'],
    scaler='robust',
    imputer_strategy='median'
)

# 모델 연결
custom_pipeline = Pipeline([
    ('preprocessor', custom_preprocessor),
    ('regressor', LinearRegression())
])

# 학습 및 평가
custom_pipeline.fit(X_train, y_train)
custom_score = custom_pipeline.score(X_test, y_test)

print(f"=== RobustScaler 파이프라인 성능 ===")
print(f"  R^2 Score: {custom_score:.4f}")
```

---

## 인코딩 방법 비교 요약

### cut 컬럼 인코딩 비교

| 방법 | 결과 | 특징 |
|------|------|------|
| **LabelEncoder** | Fair:0, Good:1, Ideal:2, Premium:3, Very Good:4 | 알파벳 순서, 실제 순서와 다름 |
| **OrdinalEncoder** | Fair:0, Good:1, Very Good:2, Premium:3, Ideal:4 | 순서 지정 가능 |
| **OneHotEncoder** | cut_Fair, cut_Good, cut_Ideal, cut_Premium, cut_Very_Good | 각각 0 또는 1 |

### 선택 가이드

```
순서 있는 범주 + 트리 모델: OrdinalEncoder
순서 없는 범주 + 선형 모델: OneHotEncoder
고차원 범주 (100개+): 빈도 인코딩 or 타겟 인코딩
```

---

## 10차시 핵심 정리

### 스케일링

| 방법 | 공식 | 결과 | 적용 시점 |
|------|------|------|----------|
| StandardScaler | (X - mean) / std | 평균=0, std=1 | 일반적 |
| MinMaxScaler | (X - min) / (max - min) | [0, 1] | 신경망 |
| RobustScaler | (X - Q2) / IQR | 중앙값 기준 | 이상치 있을 때 |

### 인코딩

| 방법 | 적용 대상 | 주의사항 |
|------|----------|----------|
| LabelEncoder | 타겟 변수 | 알파벳 순서 |
| OrdinalEncoder | 순서 있는 특성 | 순서 지정 필수 |
| OneHotEncoder | 순서 없는 특성 | drop_first로 다중공선성 방지 |

### Pipeline

| 도구 | 기능 |
|------|------|
| ColumnTransformer | 컬럼별 다른 전처리 적용 |
| Pipeline | 전처리 + 모델 연결, 자동화 |
| joblib | 전처리기 + 모델 함께 저장 |

---

## 다음 차시 예고

### 11차시: 제조 데이터 탐색 분석 종합 (EDA)

학습 내용:
- EDA 전체 워크플로우 이해
- 데이터 이해부터 인사이트 도출까지
- 제조 데이터 종합 분석 프로젝트

준비물:
- 1-10차시 내용 복습
- Python 환경 점검
