"""
[18차시] 시계열 예측 모델 - 실습 코드
학습목표: 시계열 특성 엔지니어링, ML 기반 시계열 예측, 예측 성능 평가
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 시계열 데이터 생성
# ============================================================
print("=" * 50)
print("1. 시계열 데이터 생성")
print("=" * 50)

np.random.seed(42)

# 6개월 데이터 (180일)
n_days = 180
dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

# 생산량: 추세 + 주간 패턴 + 노이즈
trend = np.linspace(1000, 1150, n_days)
weekly_pattern = 50 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
noise = np.random.normal(0, 30, n_days)
production = trend + weekly_pattern + noise

df = pd.DataFrame({
    '날짜': dates,
    '생산량': production
})

print(df.head())
print(f"\n데이터 기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
print(f"데이터 수: {len(df)}일")

# ============================================================
# 2. 특성 엔지니어링
# ============================================================
print("\n" + "=" * 50)
print("2. 특성 엔지니어링")
print("=" * 50)

# 날짜 기반 특성
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek
df['주차'] = df['날짜'].dt.isocalendar().week
df['주말'] = (df['날짜'].dt.dayofweek >= 5).astype(int)

print("▶ 날짜 기반 특성 추가")

# 시차 특성 (Lag Features)
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
df['lag_14'] = df['생산량'].shift(14) # 14일 전

print("▶ 시차(lag) 특성 추가")

# 롤링 특성 (Rolling Features) - shift(1) 먼저!
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
df['ma_14'] = df['생산량'].shift(1).rolling(14).mean()
df['std_7'] = df['생산량'].shift(1).rolling(7).std()

print("▶ 롤링 특성 추가 (shift 후 rolling!)")

# 변화량 특성
df['diff_1'] = df['생산량'].diff(1)  # 전일 대비 변화
df['diff_7'] = df['생산량'].diff(7)  # 전주 대비 변화

print("▶ 변화량 특성 추가")

print("\n특성 목록:")
print(df.columns.tolist())

# ============================================================
# 3. 데이터 준비
# ============================================================
print("\n" + "=" * 50)
print("3. 데이터 준비")
print("=" * 50)

# 결측치 제거 (lag/rolling으로 생긴 NaN)
df_clean = df.dropna().copy()
print(f"▶ 결측치 제거 후: {len(df_clean)}일")

# 특성 선택
features = ['월', '요일', '주말', 'lag_1', 'lag_7', 'ma_7', 'std_7']
target = '생산량'

print(f"▶ 사용할 특성: {features}")

# ============================================================
# 4. 시간 기준 Train/Test 분할
# ============================================================
print("\n" + "=" * 50)
print("4. 시간 기준 분할")
print("=" * 50)

split_date = '2024-05-01'

train = df_clean[df_clean['날짜'] < split_date]
test = df_clean[df_clean['날짜'] >= split_date]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print(f"▶ 분할 기준: {split_date}")
print(f"   학습: {len(train)}일 ({train['날짜'].min().date()} ~ {train['날짜'].max().date()})")
print(f"   테스트: {len(test)}일 ({test['날짜'].min().date()} ~ {test['날짜'].max().date()})")

# ============================================================
# 5. 모델 학습
# ============================================================
print("\n" + "=" * 50)
print("5. 모델 학습")
print("=" * 50)

# RandomForest 모델
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Linear Regression 모델 (비교용)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("▶ RandomForest, LinearRegression 학습 완료")

# ============================================================
# 6. 평가 지표
# ============================================================
print("\n" + "=" * 50)
print("6. 평가 지표")
print("=" * 50)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{model_name}:")
    print(f"   MAE: {mae:.2f}개")
    print(f"   RMSE: {rmse:.2f}개")
    print(f"   MAPE: {mape:.2f}%")
    return mape

rf_mape = evaluate_model(y_test.values, rf_pred, "RandomForest")
lr_mape = evaluate_model(y_test.values, lr_pred, "LinearRegression")

print("\n▶ MAPE 해석:")
print("   < 10%: 매우 좋음")
print("   10~20%: 좋음")
print("   20~50%: 보통")
print("   > 50%: 개선 필요")

# ============================================================
# 7. 예측 결과 시각화
# ============================================================
print("\n" + "=" * 50)
print("7. 예측 결과 시각화")
print("=" * 50)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# RandomForest 예측
axes[0].plot(test['날짜'], y_test, label='Actual', linewidth=2)
axes[0].plot(test['날짜'], rf_pred, label='Predicted', linewidth=2, linestyle='--')
axes[0].set_title(f'RandomForest Prediction (MAPE: {rf_mape:.1f}%)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Production')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LinearRegression 예측
axes[1].plot(test['날짜'], y_test, label='Actual', linewidth=2)
axes[1].plot(test['날짜'], lr_pred, label='Predicted', linewidth=2, linestyle='--')
axes[1].set_title(f'LinearRegression Prediction (MAPE: {lr_mape:.1f}%)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Production')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ timeseries_prediction.png 저장됨")

# ============================================================
# 8. 특성 중요도
# ============================================================
print("\n" + "=" * 50)
print("8. 특성 중요도 (RandomForest)")
print("=" * 50)

importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.to_string(index=False))

print("\n★ lag_1 (전일 생산량)이 가장 중요합니다!")
print("  → '어제 생산량'이 '오늘 생산량' 예측에 핵심")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importance)))
ax.barh(importance['feature'], importance['importance'], color=colors)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance for Time Series Prediction')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('ts_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n▶ ts_feature_importance.png 저장됨")

# ============================================================
# 9. 잔차 분석
# ============================================================
print("\n" + "=" * 50)
print("9. 잔차 분석")
print("=" * 50)

residuals = y_test.values - rf_pred

print(f"잔차 평균: {residuals.mean():.2f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.2f}")

# 잔차 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 잔차 시계열
axes[0].plot(test['날짜'], residuals)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title('Residuals Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual')

# 잔차 히스토그램
axes[1].hist(residuals, bins=20, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_title('Residual Distribution')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ residual_analysis.png 저장됨")

# ============================================================
# 10. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("10. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│               시계열 예측 모델 핵심                     │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 특성 엔지니어링                                     │
│     - 날짜: df['월'] = df['날짜'].dt.month             │
│     - Lag: df['lag_1'] = df['생산량'].shift(1)        │
│     - Rolling: df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
│                                                        │
│  ⚠️ 데이터 누출 방지                                   │
│     - Rolling 전에 shift(1) 필수!                      │
│     - 시간 기준으로 Train/Test 분할                    │
│                                                        │
│  ▶ 모델                                                │
│     model = RandomForestRegressor()                   │
│     model.fit(X_train, y_train)                       │
│     predictions = model.predict(X_test)               │
│                                                        │
│  ▶ 평가                                                │
│     - MAPE < 10%: 매우 좋음                            │
│     - lag_1이 가장 중요한 특성                         │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 딥러닝 입문 - 신경망 기초
""")
