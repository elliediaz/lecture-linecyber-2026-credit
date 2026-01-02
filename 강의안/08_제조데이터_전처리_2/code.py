# [8차시] 제조 데이터 전처리 (2) - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 실습 1: 샘플 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 샘플 데이터 생성")
print("=" * 50)

np.random.seed(42)
n = 100

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '생산량': np.random.normal(1200, 50, n),
    '습도': np.random.normal(60, 10, n),
    '라인': np.random.choice(['A', 'B', 'C'], n),
    '등급': np.random.choice(['상', '중', '하'], n)
})

print("데이터 샘플:")
print(df.head())
print(f"\n데이터 크기: {df.shape}")
print("\n기술통계:")
print(df.describe())

# ============================================================
# 실습 2: 스케일 차이 확인
# ============================================================
print("\n" + "=" * 50)
print("실습 2: 스케일 차이 확인")
print("=" * 50)

print("=== 변수별 범위 ===")
for col in ['온도', '생산량', '습도']:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    print(f"{col}: {col_min:.1f} ~ {col_max:.1f} (범위: {col_range:.1f})")

# 스케일 차이 시각화
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, col in enumerate(['온도', '생산량', '습도']):
    axes[i].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} 분포')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('빈도')

plt.tight_layout()
plt.suptitle('원본 데이터 분포', y=1.02)
plt.show()

# ============================================================
# 실습 3: 표준화 (StandardScaler)
# ============================================================
print("\n" + "=" * 50)
print("실습 3: 표준화 (StandardScaler)")
print("=" * 50)

from sklearn.preprocessing import StandardScaler

# 수치형 열 선택
numeric_cols = ['온도', '생산량', '습도']
X = df[numeric_cols].values

# 표준화
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# DataFrame으로 변환
df_std = pd.DataFrame(X_std, columns=numeric_cols)

print("=== 표준화 후 기술통계 ===")
print(df_std.describe().round(4))

print("\n=== 변환 전후 비교 (첫 5행) ===")
print("원본:")
print(df[numeric_cols].head().round(2))
print("\n표준화 후:")
print(df_std.head().round(4))

# ============================================================
# 실습 4: 정규화 (MinMaxScaler)
# ============================================================
print("\n" + "=" * 50)
print("실습 4: 정규화 (MinMaxScaler)")
print("=" * 50)

from sklearn.preprocessing import MinMaxScaler

# 정규화
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# DataFrame으로 변환
df_mm = pd.DataFrame(X_mm, columns=numeric_cols)

print("=== 정규화 후 기술통계 ===")
print(df_mm.describe().round(4))

print("\n=== 변환 전후 비교 (첫 5행) ===")
print("원본:")
print(df[numeric_cols].head().round(2))
print("\n정규화 후:")
print(df_mm.head().round(4))

# ============================================================
# 실습 5: 스케일링 비교 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 5: 스케일링 비교 시각화")
print("=" * 50)

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, col in enumerate(numeric_cols):
    # 원본
    axes[i, 0].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i, 0].set_title(f'{col} (원본)')
    axes[i, 0].set_ylabel('빈도')

    # 표준화
    axes[i, 1].hist(df_std[col], bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[i, 1].set_title(f'{col} (표준화)')

    # 정규화
    axes[i, 2].hist(df_mm[col], bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[i, 2].set_title(f'{col} (정규화)')

plt.tight_layout()
plt.show()

# ============================================================
# 실습 6: 레이블 인코딩
# ============================================================
print("\n" + "=" * 50)
print("실습 6: 레이블 인코딩")
print("=" * 50)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['등급_숫자'] = le.fit_transform(df['등급'])

print("=== 레이블 인코딩 결과 ===")
print(df[['등급', '등급_숫자']].drop_duplicates().sort_values('등급_숫자'))

print(f"\n클래스 목록: {le.classes_}")
print(f"클래스 → 숫자 매핑:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

# 역변환 예시
print("\n=== 역변환 예시 ===")
sample_encoded = [0, 1, 2]
sample_decoded = le.inverse_transform(sample_encoded)
print(f"인코딩된 값: {sample_encoded}")
print(f"역변환된 값: {list(sample_decoded)}")

# ============================================================
# 실습 7: 원-핫 인코딩 (pandas)
# ============================================================
print("\n" + "=" * 50)
print("실습 7: 원-핫 인코딩 (pandas get_dummies)")
print("=" * 50)

# pandas 방법
df_onehot = pd.get_dummies(df, columns=['라인'], prefix='라인')

print("=== 원-핫 인코딩 결과 ===")
print(df_onehot[['라인_A', '라인_B', '라인_C']].head(10))

print(f"\n열 목록: {df_onehot.columns.tolist()}")
print(f"원본 열 수: {len(df.columns)}")
print(f"인코딩 후 열 수: {len(df_onehot.columns)}")

# ============================================================
# 실습 8: 원-핫 인코딩 (sklearn)
# ============================================================
print("\n" + "=" * 50)
print("실습 8: 원-핫 인코딩 (sklearn OneHotEncoder)")
print("=" * 50)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
line_encoded = encoder.fit_transform(df[['라인']])

# 열 이름 생성
feature_names = encoder.get_feature_names_out(['라인'])
df_line_enc = pd.DataFrame(line_encoded, columns=feature_names)

print("=== sklearn OneHotEncoder 결과 ===")
print(df_line_enc.head())
print(f"\n카테고리: {encoder.categories_}")
print(f"특성 이름: {feature_names}")

# ============================================================
# 실습 9: 종합 전처리 데이터 구성
# ============================================================
print("\n" + "=" * 50)
print("실습 9: 종합 전처리 데이터 구성")
print("=" * 50)

# 최종 데이터프레임 구성
df_final = pd.DataFrame()

# 수치형: 표준화
df_final[['온도_std', '생산량_std', '습도_std']] = df_std

# 범주형: 원-핫 인코딩 (라인)
df_final = pd.concat([df_final, df_onehot[['라인_A', '라인_B', '라인_C']]], axis=1)

# 범주형: 레이블 인코딩 (등급 - 순서 있음)
df_final['등급'] = df['등급_숫자']

print("=== 최종 전처리 데이터 ===")
print(df_final.head(10))
print(f"\n열 목록: {df_final.columns.tolist()}")
print(f"열 수: {len(df_final.columns)}")
print(f"행 수: {len(df_final)}")

# 데이터 타입 확인
print("\n=== 데이터 타입 ===")
print(df_final.dtypes)

# ============================================================
# 실습 10: 역변환
# ============================================================
print("\n" + "=" * 50)
print("실습 10: 역변환")
print("=" * 50)

# 표준화 역변환
X_original = scaler_std.inverse_transform(X_std)
df_restored = pd.DataFrame(X_original, columns=numeric_cols)

print("=== 역변환 결과 비교 ===")
print("원본 데이터 (첫 5행):")
print(df[numeric_cols].head().round(2))
print("\n복원된 데이터 (첫 5행):")
print(df_restored.head().round(2))

# 차이 확인
diff = np.abs(df[numeric_cols].values - df_restored.values).max()
print(f"\n최대 오차: {diff:.10f}")
print("→ 역변환 후 원본과 거의 동일함")

# ============================================================
# 실습 11: fit과 transform 분리 예시
# ============================================================
print("\n" + "=" * 50)
print("실습 11: fit과 transform 분리 (학습/테스트 시나리오)")
print("=" * 50)

from sklearn.model_selection import train_test_split

# 데이터 분리 (80% 학습, 20% 테스트)
X_train, X_test = train_test_split(df[numeric_cols], test_size=0.2, random_state=42)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# 올바른 방법: 학습 데이터로 fit, 테스트 데이터는 transform만
scaler = StandardScaler()

# 학습 데이터: fit_transform
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터: transform만! (fit 하면 안 됨)
X_test_scaled = scaler.transform(X_test)

print("\n=== 학습 데이터 통계 (변환 후) ===")
print(f"평균: {X_train_scaled.mean(axis=0).round(4)}")
print(f"표준편차: {X_train_scaled.std(axis=0).round(4)}")

print("\n=== 테스트 데이터 통계 (변환 후) ===")
print(f"평균: {X_test_scaled.mean(axis=0).round(4)}")
print(f"표준편차: {X_test_scaled.std(axis=0).round(4)}")

print("\n⚠️ 주의: 테스트 데이터의 평균이 정확히 0이 아닌 것은 정상입니다.")
print("   학습 데이터의 통계량으로 변환했기 때문입니다.")

# ============================================================
# 실습 12: 전처리 요약
# ============================================================
print("\n" + "=" * 50)
print("실습 12: 전처리 요약")
print("=" * 50)

print("=" * 50)
print("         8차시 전처리 요약 리포트")
print("=" * 50)

print(f"\n[원본 데이터]")
print(f"행 수: {len(df)}")
print(f"열 수: {len(df.columns)}")
print(f"수치형: {numeric_cols}")
print(f"범주형: ['라인', '등급']")

print(f"\n[스케일링]")
print(f"- 표준화 (StandardScaler): 평균 0, 표준편차 1")
print(f"- 정규화 (MinMaxScaler): 범위 [0, 1]")

print(f"\n[인코딩]")
print(f"- 등급 (순서 있음): 레이블 인코딩 → 상:0, 중:1, 하:2")
print(f"- 라인 (순서 없음): 원-핫 인코딩 → 라인_A, 라인_B, 라인_C")

print(f"\n[최종 데이터]")
print(f"행 수: {len(df_final)}")
print(f"열 수: {len(df_final.columns)}")
print(f"열 목록: {df_final.columns.tolist()}")

print("\n" + "=" * 50)
print("⚠️ 핵심 주의사항")
print("=" * 50)
print("1. fit은 학습 데이터에만 적용")
print("2. 테스트 데이터는 transform만 적용")
print("3. 원-핫 인코딩 시 열 수 증가 주의")
print("4. 순서 있는 범주 → 레이블, 없는 범주 → 원-핫")
print("=" * 50)
