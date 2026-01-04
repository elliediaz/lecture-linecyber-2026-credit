"""
[9차시] 데이터 전처리 실무 (2) - 실습 코드

스케일링과 인코딩을 실습합니다.

학습목표:
- 스케일링(정규화, 표준화)의 필요성 이해
- 범주형 데이터 인코딩 방법 적용
- sklearn의 전처리 도구 활용
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("9차시: 데이터 전처리 실무 (2)")
print("=" * 60)
print()


# =============================================================
# 1. 샘플 데이터 생성
# =============================================================

print("=" * 60)
print("1. 샘플 데이터 생성")
print("=" * 60)

np.random.seed(42)
n = 200

df = pd.DataFrame({
    '라인': np.random.choice(['A', 'B', 'C'], n),
    '등급': np.random.choice(['상', '중', '하'], n, p=[0.2, 0.5, 0.3]),
    '불량유형': np.random.choice(['스크래치', '찍힘', '변색', '기타'], n),
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '생산량': np.random.normal(1200, 100, n),
    '불량수': np.random.poisson(25, n)
})

print("데이터 샘플:")
print(df.head())
print(f"\n데이터 타입:")
print(df.dtypes)
print()


# =============================================================
# 2. 스케일링 필요성 확인
# =============================================================

print("=" * 60)
print("2. 스케일링 필요성")
print("=" * 60)

numeric_cols = ['온도', '습도', '생산량', '불량수']

print("[수치형 열 통계]")
print(df[numeric_cols].describe().round(2))
print()

print("[스케일 차이]")
for col in numeric_cols:
    print(f"{col}: 범위 [{df[col].min():.1f}, {df[col].max():.1f}], "
          f"범위폭 {df[col].max() - df[col].min():.1f}")
print()
print("→ 생산량의 스케일이 온도/습도보다 훨씬 큼")
print("→ 스케일링 없이 모델에 넣으면 생산량이 과도한 영향")
print()


# =============================================================
# 3. StandardScaler (표준화)
# =============================================================

print("=" * 60)
print("3. StandardScaler (표준화)")
print("=" * 60)

# 표준화
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(df[numeric_cols])

df_standard = pd.DataFrame(X_standard, columns=[f'{col}_표준화' for col in numeric_cols])

print("[표준화 후 통계]")
print(f"평균: {X_standard.mean(axis=0).round(6)}")
print(f"표준편차: {X_standard.std(axis=0).round(6)}")
print()

print("[표준화 전후 비교]")
print(df_standard.describe().round(2))
print()


# =============================================================
# 4. MinMaxScaler (정규화)
# =============================================================

print("=" * 60)
print("4. MinMaxScaler (정규화)")
print("=" * 60)

# 정규화
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(df[numeric_cols])

df_minmax = pd.DataFrame(X_minmax, columns=[f'{col}_정규화' for col in numeric_cols])

print("[정규화 후 통계]")
print(f"최소값: {X_minmax.min(axis=0)}")
print(f"최대값: {X_minmax.max(axis=0)}")
print()

print("[정규화 전후 비교]")
print(df_minmax.describe().round(2))
print()


# =============================================================
# 5. 스케일링 비교 시각화
# =============================================================

print("=" * 60)
print("5. 스케일링 비교 시각화")
print("=" * 60)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for idx, col in enumerate(numeric_cols):
    # 원본
    axes[0, idx].hist(df[col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'{col}\n(원본)')
    axes[0, idx].axvline(df[col].mean(), color='red', linestyle='--')

    # 표준화
    axes[1, idx].hist(X_standard[:, idx], bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, idx].set_title(f'{col}\n(표준화)')
    axes[1, idx].axvline(0, color='red', linestyle='--')

    # 정규화
    axes[2, idx].hist(X_minmax[:, idx], bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[2, idx].set_title(f'{col}\n(정규화)')
    axes[2, idx].axvline(0.5, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('01_scaling_comparison.png', dpi=150)
plt.show()

print("'01_scaling_comparison.png' 저장 완료")
print()


# =============================================================
# 6. RobustScaler (이상치에 강건)
# =============================================================

print("=" * 60)
print("6. RobustScaler (이상치에 강건)")
print("=" * 60)

# 이상치 있는 데이터 생성
df_outlier = df.copy()
df_outlier.loc[0:5, '온도'] = [60, 65, 110, 115, 120, 55]

# 각 스케일러 비교
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 원본
axes[0].hist(df_outlier['온도'], bins=20, color='steelblue', edgecolor='black')
axes[0].set_title('원본 (이상치 포함)')

for idx, (name, scaler) in enumerate(scalers.items()):
    scaled = scaler.fit_transform(df_outlier[['온도']])
    axes[idx+1].hist(scaled, bins=20, edgecolor='black', alpha=0.7)
    axes[idx+1].set_title(name)

plt.tight_layout()
plt.savefig('02_robust_scaler.png', dpi=150)
plt.show()

print("RobustScaler: 중앙값과 IQR 기준으로 스케일링")
print("→ 이상치의 영향을 덜 받음")
print()
print("'02_robust_scaler.png' 저장 완료")
print()


# =============================================================
# 7. LabelEncoder (레이블 인코딩)
# =============================================================

print("=" * 60)
print("7. LabelEncoder (레이블 인코딩)")
print("=" * 60)

# 등급 인코딩 (순서 있는 범주)
le = LabelEncoder()
df['등급_encoded'] = le.fit_transform(df['등급'])

print("[LabelEncoder 결과]")
print("클래스:", le.classes_)
print()
print("변환 예시:")
print(df[['등급', '등급_encoded']].drop_duplicates().sort_values('등급_encoded'))
print()

print("주의: 알파벳 순서로 인코딩됨 ('상'=0, '중'=2, '하'=1)")
print("순서가 중요하면 OrdinalEncoder 사용")
print()


# =============================================================
# 8. OrdinalEncoder (순서 인코딩)
# =============================================================

print("=" * 60)
print("8. OrdinalEncoder (순서 인코딩)")
print("=" * 60)

# 명시적 순서 지정
oe = OrdinalEncoder(categories=[['하', '중', '상']])
df['등급_ordinal'] = oe.fit_transform(df[['등급']])

print("[OrdinalEncoder 결과 (순서 지정)]")
print("변환 예시:")
print(df[['등급', '등급_ordinal']].drop_duplicates().sort_values('등급_ordinal'))
print()
print("→ 하=0, 중=1, 상=2 (의미 있는 순서)")
print()


# =============================================================
# 9. OneHotEncoder (원-핫 인코딩)
# =============================================================

print("=" * 60)
print("9. OneHotEncoder (원-핫 인코딩)")
print("=" * 60)

# pandas get_dummies
df_onehot_pd = pd.get_dummies(df[['라인']], prefix='라인')
print("[pandas get_dummies 결과]")
print(df_onehot_pd.head())
print()

# sklearn OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
onehot_array = ohe.fit_transform(df[['라인']])
feature_names = ohe.get_feature_names_out(['라인'])
df_onehot_sk = pd.DataFrame(onehot_array, columns=feature_names)

print("[sklearn OneHotEncoder 결과]")
print(df_onehot_sk.head())
print()


# =============================================================
# 10. 인코딩 시각화
# =============================================================

print("=" * 60)
print("10. 인코딩 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 원본 범주 분포
df['라인'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('원본: 라인별 분포')
axes[0].set_xlabel('라인')
axes[0].set_ylabel('빈도')

# 레이블 인코딩
df['등급'].value_counts().plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('등급별 분포 (레이블 인코딩 대상)')
axes[1].set_xlabel('등급')

# 원-핫 인코딩 결과
df_onehot_pd.sum().plot(kind='bar', ax=axes[2], color='green', edgecolor='black')
axes[2].set_title('원-핫 인코딩 결과')
axes[2].set_xlabel('라인')

plt.tight_layout()
plt.savefig('03_encoding_visualization.png', dpi=150)
plt.show()

print("'03_encoding_visualization.png' 저장 완료")
print()


# =============================================================
# 11. ColumnTransformer (열별 다른 전처리)
# =============================================================

print("=" * 60)
print("11. ColumnTransformer")
print("=" * 60)

# 전처리 정의
numeric_features = ['온도', '습도', '생산량']
categorical_features = ['라인', '불량유형']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

# 전처리 적용
X = df[numeric_features + categorical_features]
X_processed = preprocessor.fit_transform(X)

# 열 이름 가져오기
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

df_processed = pd.DataFrame(X_processed, columns=feature_names)

print("[ColumnTransformer 결과]")
print(f"원본 열: {list(X.columns)}")
print(f"변환 후 열: {feature_names}")
print()
print(df_processed.head())
print()


# =============================================================
# 12. Pipeline (전처리 + 모델)
# =============================================================

print("=" * 60)
print("12. Pipeline (전처리 + 모델)")
print("=" * 60)

# 데이터 준비
X = df[numeric_features + categorical_features]
y = df['불량수']

# 파이프라인 생성
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 학습
pipe.fit(X, y)

# 예측
y_pred = pipe.predict(X)

# 평가
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("[Pipeline 결과]")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print()

# 새 데이터로 예측
new_data = pd.DataFrame({
    '온도': [85, 90],
    '습도': [50, 55],
    '생산량': [1200, 1300],
    '라인': ['A', 'B'],
    '불량유형': ['스크래치', '찍힘']
})

predictions = pipe.predict(new_data)
print("[새 데이터 예측]")
for i, pred in enumerate(predictions):
    print(f"샘플 {i+1}: 예측 불량수 = {pred:.1f}")
print()


# =============================================================
# 13. fit과 transform 구분
# =============================================================

print("=" * 60)
print("13. fit과 transform 구분")
print("=" * 60)

from sklearn.model_selection import train_test_split

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_cols], df['불량수'], test_size=0.2, random_state=42
)

# 올바른 방법
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 학습 데이터: fit + transform
X_test_scaled = scaler.transform(X_test)        # 테스트 데이터: transform만!

print("[올바른 스케일링 방법]")
print("학습 데이터: fit_transform() - 통계량 계산 + 변환")
print("테스트 데이터: transform() - 학습 데이터 통계량으로 변환만")
print()

print("학습 데이터 스케일링 후 평균:", X_train_scaled.mean(axis=0).round(6))
print("테스트 데이터 스케일링 후 평균:", X_test_scaled.mean(axis=0).round(6))
print()
print("→ 테스트 데이터 평균이 0이 아닌 것이 정상!")
print("→ 학습 데이터의 통계량을 사용했기 때문")
print()


# =============================================================
# 14. 인코딩 선택 가이드
# =============================================================

print("=" * 60)
print("14. 인코딩 선택 가이드")
print("=" * 60)

print("""
[범주형 데이터 인코딩 선택 가이드]

┌────────────────────┬──────────────────────┬────────────────────┐
│ 데이터 유형        │ 권장 방법            │ 예시               │
├────────────────────┼──────────────────────┼────────────────────┤
│ 순서 있는 범주     │ OrdinalEncoder       │ 등급(상/중/하)     │
│ (순서 의미 있음)   │                      │ 학력(고졸/대졸)    │
├────────────────────┼──────────────────────┼────────────────────┤
│ 순서 없는 범주     │ OneHotEncoder        │ 색상(빨/파/노)     │
│ (카디널리티 낮음)  │ get_dummies          │ 라인(A/B/C)        │
├────────────────────┼──────────────────────┼────────────────────┤
│ 순서 없는 범주     │ Target Encoding      │ 우편번호           │
│ (카디널리티 높음)  │ Frequency Encoding   │ 제품코드           │
├────────────────────┼──────────────────────┼────────────────────┤
│ 트리 기반 모델     │ LabelEncoder         │ 모든 범주형        │
│                    │ (숫자 순서 무관)     │                    │
└────────────────────┴──────────────────────┴────────────────────┘
""")

print("""
[스케일링 선택 가이드]

┌────────────────────┬──────────────────────┬────────────────────┐
│ 상황               │ 권장 방법            │ 특징               │
├────────────────────┼──────────────────────┼────────────────────┤
│ 일반적인 ML 모델   │ StandardScaler       │ 평균=0, 표준편차=1 │
├────────────────────┼──────────────────────┼────────────────────┤
│ 신경망, 이미지     │ MinMaxScaler         │ [0, 1] 범위        │
├────────────────────┼──────────────────────┼────────────────────┤
│ 이상치가 많은 경우 │ RobustScaler         │ 중앙값, IQR 기준   │
├────────────────────┼──────────────────────┼────────────────────┤
│ 트리 기반 모델     │ 스케일링 불필요      │ 분기점만 중요      │
└────────────────────┴──────────────────────┴────────────────────┘
""")

print("=" * 60)
print("9차시 실습 완료!")
print("=" * 60)
