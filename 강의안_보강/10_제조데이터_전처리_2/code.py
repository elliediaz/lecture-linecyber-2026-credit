"""
[10차시] 제조 데이터 전처리 (2): 스케일링, 인코딩, 파이프라인
========================================

학습 목표:
1. 스케일링(정규화, 표준화)의 필요성 이해
2. 범주형 데이터 인코딩 방법 적용
3. sklearn 전처리 도구(Pipeline) 활용

실습 환경:
- Python 3.8+
- pandas, numpy, sklearn, matplotlib, seaborn

데이터:
- Diamonds 데이터셋 (다이아몬드 품질 및 가격 데이터)
- 출처: seaborn built-in dataset
- 범주형 변수 풍부 (cut, color, clarity) - 인코딩 학습에 최적
- 수치형 변수 (carat, depth, price 등) - 스케일링 학습에 적합
"""

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

print("=" * 60)
print("[10차시] 제조 데이터 전처리 (2): 스케일링, 인코딩, 파이프라인")
print("=" * 60)


# ============================================================
# Part 1: 스케일링(정규화, 표준화)의 필요성 이해
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 스케일링(정규화, 표준화)의 필요성 이해")
print("=" * 60)


# --------------------------------------
# 1.1 실습 데이터 로드 - Diamonds 데이터셋
# --------------------------------------
print("\n[1.1] 실습 데이터 로드 - Diamonds 데이터셋")
print("-" * 40)

# Diamonds 데이터셋 로드
try:
    df = sns.load_dataset('diamonds')
    print("Diamonds 데이터셋 로드 성공!")
except Exception as e:
    print(f"seaborn 로드 실패: {e}")
    print("온라인에서 데이터를 다운로드합니다...")
    try:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv'
        df = pd.read_csv(url)
        print("온라인 다운로드 성공!")
    except Exception as e2:
        print(f"온라인 로드도 실패: {e2}")
        print("대체 데이터를 생성합니다...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'carat': np.random.uniform(0.2, 3.0, n),
            'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], n),
            'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n),
            'clarity': np.random.choice(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], n),
            'depth': np.random.normal(61, 2, n),
            'table': np.random.normal(57, 2, n),
            'price': np.random.randint(300, 20000, n),
            'x': np.random.uniform(3.5, 9.0, n),
            'y': np.random.uniform(3.5, 9.0, n),
            'z': np.random.uniform(2.0, 5.5, n)
        })

# 분석을 위해 샘플링 (전체 데이터가 너무 크므로)
df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

print(f"\n데이터 형태: {df.shape}")
print(f"\n처음 10행:\n{df.head(10)}")

# 데이터 설명
print("\n=== 변수 설명 ===")
print("carat: 다이아몬드 무게 (캐럿)")
print("cut: 컷팅 품질 (Fair < Good < Very Good < Premium < Ideal)")
print("color: 색상 등급 (J(worst) ~ D(best))")
print("clarity: 투명도 (I1(worst) ~ IF(best))")
print("depth: 깊이 비율 (%)")
print("table: 테이블 너비 (%)")
print("price: 가격 (USD)")
print("x, y, z: 크기 (mm)")


# --------------------------------------
# 1.2 스케일 차이 확인
# --------------------------------------
print("\n[1.2] 스케일 차이 확인")
print("-" * 40)

numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

print("=== 변수별 통계 ===")
print(df[numeric_cols].describe().round(2))

print("\n=== 변수별 범위 ===")
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    print(f"  {col:8}: {col_min:>10.2f} ~ {col_max:>10.2f} (범위: {col_range:>10.2f})")

print("\n=== 스케일 차이 해석 ===")
print("- price: 326 ~ 18,823 (매우 넓은 범위)")
print("- carat: 0.2 ~ 5.0 (좁은 범위)")
print("- depth/table: 50~70 정도 (좁은 범위)")
print("=> 머신러닝 모델에 이대로 입력하면 price가 과도한 영향을 미침!")


# --------------------------------------
# 1.3 스케일 차이 시각화
# --------------------------------------
print("\n[1.3] 스케일 차이 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# 주요 수치형 변수 분포
main_cols = ['carat', 'depth', 'table', 'price', 'x', 'y']
for i, col in enumerate(main_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[i].legend()

plt.tight_layout()
plt.savefig('scale_difference.png', dpi=100, bbox_inches='tight')
plt.close()
print("스케일 차이 시각화 저장: scale_difference.png")


# --------------------------------------
# 1.4 표준화 (StandardScaler)
# --------------------------------------
print("\n[1.4] 표준화 (StandardScaler)")
print("-" * 40)

# 스케일링할 컬럼 선택
scale_cols = ['carat', 'depth', 'table', 'price']

# 수치형 데이터 추출
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


# --------------------------------------
# 1.5 정규화 (MinMaxScaler)
# --------------------------------------
print("\n[1.5] 정규화 (MinMaxScaler)")
print("-" * 40)

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


# --------------------------------------
# 1.6 RobustScaler
# --------------------------------------
print("\n[1.6] RobustScaler (이상치에 강건)")
print("-" * 40)

# RobustScaler 적용
scaler_rb = RobustScaler()
X_rb = scaler_rb.fit_transform(X_numeric)

# DataFrame으로 변환
df_rb = pd.DataFrame(X_rb, columns=[f'{col}_rb' for col in scale_cols])

print("=== RobustScaler 후 통계 ===")
print(df_rb.describe().round(4))

print("\n=== 이상치가 있을 때 스케일러 비교 ===")
print("- StandardScaler: 평균/표준편차 사용 -> 이상치에 민감")
print("- MinMaxScaler: 최솟값/최댓값 사용 -> 이상치에 매우 민감")
print("- RobustScaler: 중앙값/IQR 사용 -> 이상치에 강건")
print("=> price처럼 이상치가 있으면 RobustScaler 권장")


# --------------------------------------
# 1.7 스케일링 비교 시각화
# --------------------------------------
print("\n[1.7] 스케일링 비교 시각화")
print("-" * 40)

# price만 비교
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
plt.savefig('scaling_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("스케일링 비교 시각화 저장: scaling_comparison.png")


# --------------------------------------
# 1.8 역변환 (Inverse Transform)
# --------------------------------------
print("\n[1.8] 역변환 (Inverse Transform)")
print("-" * 40)

# 표준화 역변환
X_restored = scaler_std.inverse_transform(X_std)

print("=== 원본 vs 역변환 비교 (처음 3행) ===")
print("\n원본:")
print(df[scale_cols].head(3).round(2))
print("\n역변환:")
print(pd.DataFrame(X_restored[:3], columns=scale_cols).round(2))

print("\n=> 역변환으로 원래 값 복원 가능 (모델 배포 시 중요!)")


# ============================================================
# Part 2: 범주형 데이터 인코딩 방법
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 범주형 데이터 인코딩 방법")
print("=" * 60)


# --------------------------------------
# 2.1 범주형 데이터 확인
# --------------------------------------
print("\n[2.1] 범주형 데이터 확인")
print("-" * 40)

categorical_cols = ['cut', 'color', 'clarity']

print("=== 범주형 컬럼 고유값 ===")
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"\n{col}:")
    print(f"  고유값 ({len(unique_values)}개): {list(unique_values)}")

# 순서가 있는 범주 (Ordinal)
print("\n=== 순서가 있는 범주 (Ordinal) ===")
print("cut: Fair < Good < Very Good < Premium < Ideal")
print("color: J < I < H < G < F < E < D (색상이 투명할수록 좋음)")
print("clarity: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF")


# --------------------------------------
# 2.2 레이블 인코딩 (LabelEncoder)
# --------------------------------------
print("\n[2.2] 레이블 인코딩 (LabelEncoder)")
print("-" * 40)

# cut 인코딩 (순서가 있는 범주)
le = LabelEncoder()
df['cut_encoded'] = le.fit_transform(df['cut'])

print("=== 레이블 인코딩 결과 (cut) ===")
print(df[['cut', 'cut_encoded']].drop_duplicates().sort_values('cut_encoded'))

print(f"\n=== 클래스 순서 ===")
print(f"  {le.classes_}")

# 역변환
print(f"\n=== 역변환 예시 ===")
original = le.inverse_transform([0, 1, 2, 3, 4])
print(f"  [0, 1, 2, 3, 4] -> {list(original)}")

print("\n주의: LabelEncoder는 알파벳 순서로 인코딩!")
print("=> 실제 순서(Fair < Ideal)와 다를 수 있음")


# --------------------------------------
# 2.3 순서형 인코딩 (OrdinalEncoder) - 올바른 순서
# --------------------------------------
print("\n[2.3] 순서형 인코딩 (OrdinalEncoder)")
print("-" * 40)

# cut의 올바른 순서 정의
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # J가 가장 나쁨

# OrdinalEncoder로 순서 지정
ordinal_encoder = OrdinalEncoder(categories=[cut_order])
df['cut_ordinal'] = ordinal_encoder.fit_transform(df[['cut']])

print("=== 순서형 인코딩 결과 (cut) ===")
print(df[['cut', 'cut_ordinal']].drop_duplicates().sort_values('cut_ordinal'))
print("\n=> 이제 순서가 올바름: Fair(0) < ... < Ideal(4)")


# --------------------------------------
# 2.4 원-핫 인코딩 (pandas get_dummies)
# --------------------------------------
print("\n[2.4] 원-핫 인코딩 (pandas get_dummies)")
print("-" * 40)

# color 원-핫 인코딩 (순서가 있지만 원-핫으로도 가능)
df_onehot = pd.get_dummies(df, columns=['color'], prefix='color')

print("=== 원-핫 인코딩 결과 (color) ===")
color_cols = [col for col in df_onehot.columns if col.startswith('color_')]
print(df_onehot[color_cols].head(10))

print(f"\n=== 새로운 컬럼 ({len(color_cols)}개) ===")
print(f"  {color_cols}")


# --------------------------------------
# 2.5 원-핫 인코딩 with drop_first
# --------------------------------------
print("\n[2.5] 원-핫 인코딩 (drop_first=True)")
print("-" * 40)

# 다중공선성 방지를 위해 첫 번째 컬럼 제거
df_onehot_drop = pd.get_dummies(df, columns=['color'], prefix='color', drop_first=True)

print("=== drop_first=True 결과 ===")
color_cols_drop = [col for col in df_onehot_drop.columns if col.startswith('color_')]
print(df_onehot_drop[color_cols_drop].head(10))
print(f"\n=> 첫 번째 범주(color_D)가 제거됨: {len(color_cols_drop)}개 컬럼")
print("   (모든 color_X가 0이면 color_D)")


# --------------------------------------
# 2.6 원-핫 인코딩 (sklearn OneHotEncoder)
# --------------------------------------
print("\n[2.6] 원-핫 인코딩 (sklearn OneHotEncoder)")
print("-" * 40)

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


# --------------------------------------
# 2.7 빈도 인코딩 (Frequency Encoding)
# --------------------------------------
print("\n[2.7] 빈도 인코딩 (Frequency Encoding)")
print("-" * 40)

# clarity별 빈도 계산
freq = df['clarity'].value_counts(normalize=True)
df['clarity_freq'] = df['clarity'].map(freq)

print("=== 빈도 인코딩 결과 (clarity) ===")
print(df[['clarity', 'clarity_freq']].drop_duplicates().sort_values('clarity_freq', ascending=False))

print("\n=> 희귀한 범주일수록 낮은 값")


# ============================================================
# Part 3: sklearn 전처리 도구 활용
# ============================================================
print("\n" + "=" * 60)
print("Part 3: sklearn 전처리 도구 활용")
print("=" * 60)


# --------------------------------------
# 3.1 ColumnTransformer 기본 사용
# --------------------------------------
print("\n[3.1] ColumnTransformer 기본 사용")
print("-" * 40)

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
print(f"  (수치형 {len(numeric_features)}개 + 범주형 원-핫 인코딩)")

# 컬럼명 확인
try:
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"\n=== 출력 컬럼명 ({len(feature_names_out)}개) ===")
    print(f"  {list(feature_names_out[:10])}...")
except:
    print("\n(sklearn 버전에 따라 컬럼명 출력 방식이 다름)")


# --------------------------------------
# 3.2 수치형 전처리 파이프라인
# --------------------------------------
print("\n[3.2] 수치형 전처리 파이프라인")
print("-" * 40)

# 결측치 + 스케일링 파이프라인
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
    ('scaler', StandardScaler())                    # 스케일링
])

print("=== 수치형 전처리 파이프라인 ===")
print(numeric_transformer)


# --------------------------------------
# 3.3 범주형 전처리 파이프라인
# --------------------------------------
print("\n[3.3] 범주형 전처리 파이프라인")
print("-" * 40)

# 결측치 + 인코딩 파이프라인
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측치 처리
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 인코딩
])

print("=== 범주형 전처리 파이프라인 ===")
print(categorical_transformer)


# --------------------------------------
# 3.4 종합 전처리 파이프라인
# --------------------------------------
print("\n[3.4] 종합 전처리 파이프라인")
print("-" * 40)

# ColumnTransformer로 통합
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("=== 종합 전처리기 ===")
print(full_preprocessor)


# --------------------------------------
# 3.5 Pipeline: 전처리 + 모델
# --------------------------------------
print("\n[3.5] Pipeline: 전처리 + 모델")
print("-" * 40)

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


# --------------------------------------
# 3.6 새 데이터 예측
# --------------------------------------
print("\n[3.6] 새 데이터 예측")
print("-" * 40)

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

# 예측 (전처리 자동 적용!)
predictions = full_pipeline.predict(new_data)

print("\n=== 가격 예측 결과 ===")
for i, (_, row) in enumerate(new_data.iterrows()):
    print(f"  {row['carat']}ct {row['cut']} {row['color']} -> ${predictions[i]:,.0f}")


# --------------------------------------
# 3.7 파이프라인 저장/로드
# --------------------------------------
print("\n[3.7] 파이프라인 저장/로드")
print("-" * 40)

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


# ============================================================
# 종합 실습: 전처리 파이프라인 구성
# ============================================================
print("\n" + "=" * 60)
print("종합 실습: 전처리 파이프라인 구성")
print("=" * 60)


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


# 사용 예시
print("\n[종합 실습] 커스텀 전처리 파이프라인")
print("-" * 40)

# 전처리기 생성 (RobustScaler 사용 - 가격 이상치 때문)
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


# ============================================================
# 인코딩 방법 비교 요약
# ============================================================
print("\n" + "=" * 60)
print("인코딩 방법 비교 요약")
print("=" * 60)

print("""
=== cut 컬럼 인코딩 비교 ===

원본 데이터: ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']

1. LabelEncoder (순서 X, 알파벳순)
   Fair:0, Good:1, Ideal:2, Premium:3, Very Good:4
   -> 문제: 실제 순서와 다름!

2. OrdinalEncoder (순서 O)
   Fair:0, Good:1, Very Good:2, Premium:3, Ideal:4
   -> 순서 지정 가능

3. OneHotEncoder (원-핫)
   cut_Fair, cut_Good, cut_Ideal, cut_Premium, cut_Very_Good
   -> 각각 0 또는 1

=== 선택 가이드 ===
- 순서 있는 범주 + 트리 모델: OrdinalEncoder
- 순서 없는 범주 + 선형 모델: OneHotEncoder
- 고차원 범주 (100개+): 빈도 인코딩 or 타겟 인코딩
""")


# ============================================================
# 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("10차시 핵심 정리")
print("=" * 60)

print("""
[스케일링]
------------------------------------------------------
StandardScaler: Z = (X - mean) / std
  - 평균 0, 표준편차 1
  - 일반적인 상황에서 사용

MinMaxScaler: X' = (X - min) / (max - min)
  - 0~1 범위
  - 신경망, 이미지

RobustScaler: X' = (X - Q2) / IQR
  - 이상치에 강건
  - price처럼 치우친 분포

[인코딩]
------------------------------------------------------
LabelEncoder:
  - 알파벳 순서로 숫자 부여
  - 주의: 실제 순서와 다를 수 있음

OrdinalEncoder:
  - 순서 지정 가능
  - 순서가 있는 범주에 적합 (cut, clarity)

OneHotEncoder / get_dummies:
  - 순서가 없는 범주
  - 이진 벡터로 변환
  - drop_first로 다중공선성 방지

[Pipeline]
------------------------------------------------------
ColumnTransformer:
  - 컬럼별 다른 전처리 적용
  - 수치형 + 범주형 통합

Pipeline:
  - 전처리 + 모델 연결
  - fit/predict 한번에
  - 데이터 누수 방지

[Diamonds 데이터 인사이트]
------------------------------------------------------
- carat이 가격에 가장 큰 영향
- cut/color/clarity 순서 인코딩 시 OrdinalEncoder 권장
- price 분포가 치우침 -> RobustScaler 권장
""")

print("\n다음 차시 예고: 제조 데이터 탐색 분석 종합 (EDA)")
print("=" * 60)
