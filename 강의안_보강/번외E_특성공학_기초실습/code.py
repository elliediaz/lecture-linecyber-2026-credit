"""
번외E: 특성 공학 기초 실습
===========================
좋은 특성이 좋은 모델을 만든다!

데이터셋: Tips (seaborn 내장)
목표: tip 금액 예측 (회귀)

내용:
1. 수치형 변환 (로그, 비율, 구간화)
2. 범주형 인코딩 (빈도, 타겟)
3. 날짜/시간 특성 추출
4. 특성 선택 (상관관계)
5. 성능 비교
"""

# ============================================================
# 라이브러리 임포트
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("번외E: 특성 공학 기초 실습")
print("=" * 60)

# ============================================================
# 데이터 로드 및 확인
# ============================================================
print("\n" + "=" * 60)
print("데이터 로드 및 확인")
print("=" * 60)

# Tips 데이터셋 로드
df = sns.load_dataset('tips')

print(f"\n[데이터 크기]")
print(f"행: {df.shape[0]}, 열: {df.shape[1]}")

print(f"\n[컬럼 정보]")
print(df.dtypes)

print(f"\n[데이터 미리보기]")
print(df.head())

print(f"\n[기초 통계량]")
print(df.describe())

# ============================================================
# Part 1: 수치형 특성 변환
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 수치형 특성 변환")
print("=" * 60)

# 작업용 복사본
df_eng = df.copy()

# 1-1. 로그 변환
print("\n[1-1. 로그 변환]")
print(f"변환 전 왜도(skewness): {df_eng['total_bill'].skew():.3f}")

# log1p = log(x + 1) → 0 처리 안전
df_eng['log_total_bill'] = np.log1p(df_eng['total_bill'])

print(f"변환 후 왜도(skewness): {df_eng['log_total_bill'].skew():.3f}")
print("→ 왜도가 0에 가까울수록 정규분포에 가까움")

# 1-2. 비율 특성 생성
print("\n[1-2. 비율 특성 생성]")

# 팁 비율
df_eng['tip_ratio'] = df_eng['tip'] / df_eng['total_bill']
print(f"팁 비율 평균: {df_eng['tip_ratio'].mean():.1%}")
print(f"팁 비율 범위: {df_eng['tip_ratio'].min():.1%} ~ {df_eng['tip_ratio'].max():.1%}")

# 1인당 금액
df_eng['per_person'] = df_eng['total_bill'] / df_eng['size']
print(f"\n1인당 금액 평균: ${df_eng['per_person'].mean():.2f}")

# 1-3. 구간화 (Binning)
print("\n[1-3. 구간화 (Binning)]")

# 등간격 구간화
df_eng['bill_cut'] = pd.cut(
    df_eng['total_bill'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)
print("등간격(cut) 분포:")
print(df_eng['bill_cut'].value_counts())

# 등빈도 구간화
df_eng['bill_qcut'] = pd.qcut(
    df_eng['total_bill'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
print("\n등빈도(qcut) 분포:")
print(df_eng['bill_qcut'].value_counts())

# ============================================================
# Part 2: 범주형 특성 인코딩
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 범주형 특성 인코딩")
print("=" * 60)

# 2-1. 빈도 인코딩 (Frequency Encoding)
print("\n[2-1. 빈도 인코딩]")

# 요일별 빈도 계산
day_freq = df_eng['day'].value_counts(normalize=True)
print("요일별 빈도:")
print(day_freq)

# 빈도로 인코딩
df_eng['day_freq'] = df_eng['day'].map(day_freq)
print("\n인코딩 결과 (샘플):")
print(df_eng[['day', 'day_freq']].head(10))

# 2-2. 타겟 인코딩 (Target Encoding)
print("\n[2-2. 타겟 인코딩]")

# 요일별 팁 평균
day_target = df_eng.groupby('day')['tip'].mean()
print("요일별 평균 팁:")
print(day_target)

# 타겟 인코딩
df_eng['day_target'] = df_eng['day'].map(day_target)
print("\n인코딩 결과 (샘플):")
print(df_eng[['day', 'day_target']].head(10))

print("\n[주의] 타겟 인코딩은 데이터 누출 위험!")
print("→ 반드시 학습 데이터로만 평균 계산")

# 2-3. 기본 인코딩 (참고)
print("\n[2-3. One-Hot Encoding (참고)]")
df_onehot = pd.get_dummies(df_eng[['day', 'time']], drop_first=True)
print(df_onehot.head())

# ============================================================
# Part 3: 날짜/시간 특성
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 날짜/시간 특성")
print("=" * 60)

# Tips 데이터에는 datetime이 없으므로 기존 컬럼 활용
print("\n[시간 관련 특성 생성]")

# 저녁 식사 여부
df_eng['is_dinner'] = (df_eng['time'] == 'Dinner').astype(int)
print(f"저녁 식사 비율: {df_eng['is_dinner'].mean():.1%}")

# 주말 여부
df_eng['is_weekend'] = df_eng['day'].isin(['Sat', 'Sun']).astype(int)
print(f"주말 비율: {df_eng['is_weekend'].mean():.1%}")

print("\n생성된 특성 (샘플):")
print(df_eng[['day', 'time', 'is_dinner', 'is_weekend']].head(10))

# ============================================================
# Part 4: 특성 선택
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 특성 선택")
print("=" * 60)

# 4-1. 상관관계 분석
print("\n[4-1. 타겟과의 상관관계]")

# 수치형 특성 선택
numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
print(f"수치형 특성: {numeric_cols}")

# 상관관계 계산
correlations = df_eng[numeric_cols].corr()['tip'].abs()
correlations = correlations.sort_values(ascending=False)

print("\n타겟(tip)과의 상관계수:")
for col, corr in correlations.items():
    bar = "█" * int(corr * 30)
    print(f"{col:18} {corr:.3f} {bar}")

# 4-2. 특성 선택
print("\n[4-2. 상관관계 기반 특성 선택]")

threshold = 0.3
selected_features = correlations[correlations > threshold].index.tolist()
selected_features.remove('tip')  # 타겟 제외

print(f"임계값: {threshold}")
print(f"선택된 특성: {selected_features}")

# 4-3. 다중공선성 확인
print("\n[4-3. 특성 간 상관관계 (다중공선성 확인)]")
feature_corr = df_eng[selected_features].corr()
print(feature_corr.round(2))

print("\n→ total_bill과 log_total_bill 상관관계 높음 → 하나만 사용 권장")

# ============================================================
# Part 5: 성능 비교
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 성능 비교 (특성 공학 전 vs 후)")
print("=" * 60)

# 타겟
y = df_eng['tip']

# 5-1. 기본 특성
print("\n[5-1. 기본 특성으로 학습]")
X_basic = df_eng[['total_bill', 'size']]
print(f"특성: {list(X_basic.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

model_basic = LinearRegression()
model_basic.fit(X_train, y_train)
y_pred_basic = model_basic.predict(X_test)

rmse_basic = np.sqrt(mean_squared_error(y_test, y_pred_basic))
r2_basic = r2_score(y_test, y_pred_basic)
print(f"RMSE: {rmse_basic:.3f}")
print(f"R²: {r2_basic:.3f}")

# 5-2. 공학 후 특성
print("\n[5-2. 공학 후 특성으로 학습]")
X_engineered = df_eng[['log_total_bill', 'per_person', 'size',
                       'is_dinner', 'is_weekend', 'day_freq']]
print(f"특성: {list(X_engineered.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

model_eng = LinearRegression()
model_eng.fit(X_train, y_train)
y_pred_eng = model_eng.predict(X_test)

rmse_eng = np.sqrt(mean_squared_error(y_test, y_pred_eng))
r2_eng = r2_score(y_test, y_pred_eng)
print(f"RMSE: {rmse_eng:.3f}")
print(f"R²: {r2_eng:.3f}")

# 5-3. 비교
print("\n[5-3. 성능 비교]")
print(f"{'특성':15} {'RMSE':>10} {'R²':>10}")
print("-" * 40)
print(f"{'기본':15} {rmse_basic:>10.3f} {r2_basic:>10.3f}")
print(f"{'공학 후':15} {rmse_eng:>10.3f} {r2_eng:>10.3f}")

improvement = (rmse_basic - rmse_eng) / rmse_basic * 100
print(f"\nRMSE 개선율: {improvement:.1f}%")

# ============================================================
# 시각화
# ============================================================
print("\n" + "=" * 60)
print("시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 로그 변환 효과
ax1 = axes[0, 0]
ax1.hist(df_eng['total_bill'], bins=30, alpha=0.5, label='Original', color='blue')
ax1.hist(df_eng['log_total_bill'] * 10, bins=30, alpha=0.5, label='Log (scaled)', color='orange')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Log Transformation Effect')
ax1.legend()

# 2. 상관관계 히트맵
ax2 = axes[0, 1]
corr_subset = df_eng[['tip', 'total_bill', 'log_total_bill', 'per_person', 'size']].corr()
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, center=0)
ax2.set_title('Correlation Heatmap')

# 3. 요일별 팁 분포
ax3 = axes[1, 0]
df_eng.boxplot(column='tip', by='day', ax=ax3)
ax3.set_xlabel('Day')
ax3.set_ylabel('Tip')
ax3.set_title('Tip Distribution by Day')
plt.suptitle('')  # Remove automatic title

# 4. 성능 비교
ax4 = axes[1, 1]
models = ['Basic', 'Engineered']
rmse_values = [rmse_basic, rmse_eng]
r2_values = [r2_basic, r2_eng]

x = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x - width/2, rmse_values, width, label='RMSE', color='#e74c3c')
bars2 = ax4.bar(x + width/2, r2_values, width, label='R²', color='#3498db')

ax4.set_ylabel('Score')
ax4.set_title('Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()

# 값 표시
for bar, val in zip(bars1, rmse_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom')
for bar, val in zip(bars2, r2_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('feature_engineering_results.png', dpi=150, bbox_inches='tight')
print("시각화 저장 완료: feature_engineering_results.png")
plt.show()

# ============================================================
# 생성된 특성 요약
# ============================================================
print("\n" + "=" * 60)
print("생성된 특성 요약")
print("=" * 60)

print("""
[수치형 변환]
- log_total_bill: 총 금액의 로그 변환
- tip_ratio: 팁 / 총 금액
- per_person: 1인당 금액

[구간화]
- bill_cut: 총 금액 등간격 구간화
- bill_qcut: 총 금액 등빈도 구간화

[범주형 인코딩]
- day_freq: 요일 빈도 인코딩
- day_target: 요일 타겟 인코딩

[시간 특성]
- is_dinner: 저녁 식사 여부
- is_weekend: 주말 여부
""")

# ============================================================
# 특성 공학 체크리스트
# ============================================================
print("\n" + "=" * 60)
print("특성 공학 체크리스트")
print("=" * 60)

print("""
[수치형 특성]
□ 분포 확인 (왜도) → 로그 변환 필요?
□ 스케일링 필요? (StandardScaler, MinMaxScaler)
□ 구간화가 의미있을까?

[비율/조합 특성]
□ 의미있는 비율 계산 가능?
□ 특성 간 상호작용 특성?

[범주형 특성]
□ 범주 개수 확인 → One-Hot vs 빈도/타겟 인코딩
□ 순서 있는 범주? → Label Encoding

[날짜/시간]
□ 연/월/일/요일/시간 분해
□ 주말/공휴일 여부
□ 주기적 특성? → 삼각함수 변환

[특성 선택]
□ 타겟과의 상관관계 확인
□ 특성 간 다중공선성 확인
□ 분산 낮은 특성 제거?
""")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
