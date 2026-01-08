"""
[19차시] 시계열 예측 모델 - 실습 코드
=============================================

학습 목표:
1. 시계열 특성 엔지니어링을 수행한다
2. ML 모델로 시계열을 예측한다
3. 시계열 예측을 평가한다

실습 내용:
- 특성 엔지니어링 (날짜, Lag, Rolling)
- RandomForest로 시계열 예측
- 평가 지표 (MAE, RMSE, MAPE)
- 특성 중요도 분석
- 시각화

데이터셋: AirPassengers (1949-1960 월별 항공 승객 수)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 시계열 데이터 로드 (AirPassengers)
# ============================================================

print("=" * 60)
print("Part 1: 시계열 데이터 로드 (AirPassengers)")
print("=" * 60)

# AirPassengers 데이터셋 로드
try:
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    print("\n[데이터 로드 성공]")
except Exception as e:
    print(f"\n[데이터 로드 실패: {e}]")
    print("오프라인 모드로 샘플 데이터를 생성합니다.")
    # 오프라인 대비 샘플 데이터 (추세 + 계절성 시뮬레이션)
    np.random.seed(42)
    n = 144
    dates = pd.date_range(start='1949-01', periods=n, freq='MS')
    trend = np.linspace(100, 400, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 20, n)
    passengers = trend + seasonal + noise
    df = pd.DataFrame({'Month': dates.strftime('%Y-%m'), 'Passengers': passengers.astype(int)})

# 열 이름 표준화
df.columns = ['Month', 'Passengers']

# 날짜 파싱 및 인덱스 설정
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"기간: {df.index.min()} ~ {df.index.max()}")
print(f"\n기술통계:")
print(df['Passengers'].describe())


# ============================================================
# Part 2: 특성 엔지니어링 - 날짜 특성
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 특성 엔지니어링 - 날짜 특성")
print("=" * 60)

features = pd.DataFrame(index=df.index)

# 기본 날짜 특성
features['year'] = df.index.year
features['month'] = df.index.month
features['quarter'] = df.index.quarter

# 주기적 인코딩 (사인/코사인) - 월의 순환 특성 반영
features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

# 분기 인코딩
features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)

# 시간 순서 (추세 반영용)
features['time_idx'] = np.arange(len(df))

print("\n[날짜 특성]")
print(features.head(12))


# ============================================================
# Part 3: 특성 엔지니어링 - Lag 특성
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 특성 엔지니어링 - Lag 특성")
print("=" * 60)

# Lag 특성 (반드시 shift로 미래 누출 방지!)
lag_months = [1, 2, 3, 6, 12]

for lag in lag_months:
    features[f'lag_{lag}'] = df['Passengers'].shift(lag)

print("\n[Lag 특성]")
print(f"생성된 Lag: {lag_months}")
print(features[[f'lag_{lag}' for lag in lag_months]].tail(10))


# ============================================================
# Part 4: 특성 엔지니어링 - Rolling 특성
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 특성 엔지니어링 - Rolling 특성")
print("=" * 60)

# Rolling 특성 (shift 먼저!)
windows = [3, 6, 12]

for window in windows:
    # 이동평균
    features[f'ma_{window}'] = df['Passengers'].shift(1).rolling(window).mean()
    # 이동 표준편차
    features[f'std_{window}'] = df['Passengers'].shift(1).rolling(window).std()

# 추가 Rolling 특성
features['max_12'] = df['Passengers'].shift(1).rolling(12).max()
features['min_12'] = df['Passengers'].shift(1).rolling(12).min()
features['range_12'] = features['max_12'] - features['min_12']

print("\n[Rolling 특성]")
print(f"생성된 윈도우: {windows}")
print(features[['ma_3', 'ma_12', 'std_12', 'max_12', 'min_12']].tail(10))


# ============================================================
# Part 5: 특성 엔지니어링 - 변화 특성
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 특성 엔지니어링 - 변화 특성")
print("=" * 60)

# 차분 (변화량) - shift 필수
features['diff_1'] = df['Passengers'].shift(1) - df['Passengers'].shift(2)
features['diff_12'] = df['Passengers'].shift(1) - df['Passengers'].shift(13)  # 전년 동월 대비

# 변화율
features['pct_1'] = df['Passengers'].shift(1).pct_change()

# 전년 동월 대비 비율
features['yoy_ratio'] = df['Passengers'].shift(1) / df['Passengers'].shift(13)

print("\n[변화 특성]")
print(features[['diff_1', 'diff_12', 'pct_1', 'yoy_ratio']].tail(10))


# ============================================================
# Part 6: 데이터 준비 및 분할
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 데이터 준비 및 분할")
print("=" * 60)

# 타겟
target = df['Passengers']

# NaN 제거
features_clean = features.dropna()
target_clean = target.loc[features_clean.index]

print(f"\n[NaN 제거]")
print(f"원본: {len(features)}개")
print(f"정제 후: {len(features_clean)}개")
print(f"제거된 행: {len(features) - len(features_clean)}개")

# 시간 기준 분할 (마지막 2년을 테스트로)
split_date = '1958-12-01'

X_train = features_clean[:split_date]
y_train = target_clean[:split_date]
X_test = features_clean[split_date:]
y_test = target_clean[split_date:]

print(f"\n[시간 기준 분할]")
print(f"분할 기준: {split_date}")
print(f"학습: {len(X_train)}개 ({X_train.index.min()} ~ {X_train.index.max()})")
print(f"테스트: {len(X_test)}개 ({X_test.index.min()} ~ {X_test.index.max()})")


# ============================================================
# Part 7: 모델 학습 및 예측
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 모델 학습 및 예측")
print("=" * 60)

# RandomForest 모델
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 학습
model_rf.fit(X_train, y_train)

# 예측
y_pred_rf = model_rf.predict(X_test)

print("\n[RandomForest 학습 완료]")
print(f"학습 R²: {model_rf.score(X_train, y_train):.4f}")
print(f"테스트 R²: {model_rf.score(X_test, y_test):.4f}")


# ============================================================
# Part 8: 평가 지표 계산
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 평가 지표 계산")
print("=" * 60)


def calculate_mape(y_true, y_pred):
    """MAPE 계산 (0 제외)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# 평가 지표
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
mape = calculate_mape(y_test.values, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("\n[평가 지표]")
print(f"MAE:  {mae:.2f} (평균 절대 오차)")
print(f"RMSE: {rmse:.2f} (평균 제곱근 오차)")
print(f"MAPE: {mape:.2f}% (평균 절대 비율 오차)")
print(f"R²:   {r2:.4f} (결정계수)")

# 해석
print("\n[지표 해석]")
print(f"-> 평균적으로 {mae:.0f}명 정도 오차")
print(f"-> 상대적으로 {mape:.1f}% 오차")
if mape < 5:
    print("-> 매우 좋은 예측 성능!")
elif mape < 10:
    print("-> 좋은 예측 성능")
elif mape < 20:
    print("-> 보통 수준의 예측 성능")
else:
    print("-> 개선 필요")


# ============================================================
# Part 9: 특성 중요도 분석
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 특성 중요도 분석")
print("=" * 60)

# 특성 중요도
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n[특성 중요도 Top 15]")
print(importance_df.head(15).to_string(index=False))


# ============================================================
# Part 10: 모델 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 모델 비교")
print("=" * 60)

# 선형회귀 비교
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mape_lr = calculate_mape(y_test.values, y_pred_lr)

print("\n[모델 비교]")
print(f"{'모델':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-" * 54)
print(f"{'LinearRegression':<20} {mae_lr:>10.2f} {rmse_lr:>10.2f} {mape_lr:>9.2f}%")
print(f"{'RandomForest':<20} {mae:>10.2f} {rmse:>10.2f} {mape:>9.2f}%")

if mape_lr > 0:
    improvement = (mape_lr - mape) / mape_lr * 100
    print(f"\n-> RandomForest가 MAPE 기준 {improvement:.1f}% 개선")


# ============================================================
# Part 11: TimeSeriesSplit 교차검증
# ============================================================

print("\n" + "=" * 60)
print("Part 11: TimeSeriesSplit 교차검증")
print("=" * 60)

# 전체 데이터로 교차검증
X_full = features_clean
y_full = target_clean

tscv = TimeSeriesSplit(n_splits=5)

print("\n[TimeSeriesSplit 교차검증]")
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), 1):
    X_tr = X_full.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_full.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    model_cv = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model_cv.fit(X_tr, y_tr)
    score = model_cv.score(X_val, y_val)
    cv_scores.append(score)

    print(f"Fold {fold}: R² = {score:.4f} "
          f"(학습: {len(train_idx)}, 검증: {len(val_idx)})")

print(f"\n평균 R²: {np.mean(cv_scores):.4f} (+/-{np.std(cv_scores):.4f})")


# ============================================================
# Part 12: 잔차 분석
# ============================================================

print("\n" + "=" * 60)
print("Part 12: 잔차 분석")
print("=" * 60)

residuals = y_test.values - y_pred_rf

print("\n[잔차 통계]")
print(f"평균: {np.mean(residuals):.2f} (0에 가까워야 함)")
print(f"표준편차: {np.std(residuals):.2f}")
print(f"최소: {np.min(residuals):.2f}")
print(f"최대: {np.max(residuals):.2f}")


# ============================================================
# Part 13: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 13: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 실제 vs 예측
ax1 = axes[0, 0]
ax1.plot(y_test.index, y_test.values, label='실제', alpha=0.7, linewidth=2)
ax1.plot(y_test.index, y_pred_rf, label='예측', alpha=0.7, linewidth=2)
ax1.set_title(f'실제 vs 예측 (MAPE: {mape:.1f}%)')
ax1.set_xlabel('연도')
ax1.set_ylabel('승객 수')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 특성 중요도 Top 10
ax2 = axes[0, 1]
top10 = importance_df.head(10)
ax2.barh(range(10), top10['importance'].values, color='steelblue')
ax2.set_yticks(range(10))
ax2.set_yticklabels(top10['feature'].values)
ax2.set_xlabel('중요도')
ax2.set_title('특성 중요도 Top 10')
ax2.invert_yaxis()

# 3. 잔차 분포
ax3 = axes[0, 2]
ax3.hist(residuals, bins=15, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('잔차 (실제 - 예측)')
ax3.set_ylabel('빈도')
ax3.set_title('잔차 분포')

# 4. 시간별 잔차
ax4 = axes[1, 0]
ax4.plot(y_test.index, residuals, alpha=0.7, linewidth=2)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('연도')
ax4.set_ylabel('잔차')
ax4.set_title('시간별 잔차')
ax4.grid(True, alpha=0.3)

# 5. 실제 vs 예측 산점도
ax5 = axes[1, 1]
ax5.scatter(y_test.values, y_pred_rf, alpha=0.6)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2)
ax5.set_xlabel('실제 승객 수')
ax5.set_ylabel('예측 승객 수')
ax5.set_title(f'실제 vs 예측 (R² = {r2:.4f})')
ax5.grid(True, alpha=0.3)

# 6. 모델 비교
ax6 = axes[1, 2]
models = ['Linear\nRegression', 'Random\nForest']
mapes = [mape_lr, mape]
colors = ['lightcoral', 'steelblue']
bars = ax6.bar(models, mapes, color=colors)
ax6.set_ylabel('MAPE (%)')
ax6.set_title('모델별 MAPE 비교')
for bar, val in zip(bars, mapes):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('19_timeseries_forecasting.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 19_timeseries_forecasting.png")


# ============================================================
# Part 14: 예측 구간
# ============================================================

print("\n" + "=" * 60)
print("Part 14: 예측 구간")
print("=" * 60)

# 잔차 표준편차 기반 예측 구간
std_residual = np.std(residuals)
confidence = 1.96  # 95% 신뢰구간

lower = y_pred_rf - confidence * std_residual
upper = y_pred_rf + confidence * std_residual

# 실제값이 구간 내에 있는 비율
in_interval = np.mean((y_test.values >= lower) & (y_test.values <= upper))

print(f"\n[95% 예측 구간]")
print(f"잔차 표준편차: {std_residual:.2f}")
print(f"실제값이 구간 내 비율: {in_interval:.1%}")

# 예측 구간 시각화
plt.figure(figsize=(14, 5))
plt.fill_between(y_test.index, lower, upper, alpha=0.3, label='95% 예측 구간')
plt.plot(y_test.index, y_test.values, 'b-', alpha=0.7, label='실제', linewidth=2)
plt.plot(y_test.index, y_pred_rf, 'r-', alpha=0.7, label='예측', linewidth=2)
plt.xlabel('연도')
plt.ylabel('승객 수')
plt.title(f'예측 구간 (실제값 포함률: {in_interval:.1%})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('19_prediction_interval.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n예측 구간 시각화 저장: 19_prediction_interval.png")


# ============================================================
# Part 15: 전체 기간 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 15: 전체 기간 시각화")
print("=" * 60)

# 전체 기간 예측
y_train_pred = model_rf.predict(X_train)
y_test_pred = y_pred_rf

plt.figure(figsize=(14, 5))
plt.plot(y_train.index, y_train.values, 'b-', alpha=0.5, label='학습 실제', linewidth=1.5)
plt.plot(y_train.index, y_train_pred, 'g--', alpha=0.7, label='학습 예측', linewidth=1.5)
plt.plot(y_test.index, y_test.values, 'b-', alpha=0.8, label='테스트 실제', linewidth=2)
plt.plot(y_test.index, y_test_pred, 'r--', alpha=0.8, label='테스트 예측', linewidth=2)
plt.axvline(x=pd.Timestamp(split_date), color='gray', linestyle=':', label='분할 기준', linewidth=2)
plt.xlabel('연도')
plt.ylabel('승객 수')
plt.title('AirPassengers 시계열 예측 (1949-1960)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('19_full_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n전체 기간 시각화 저장: 19_full_forecast.png")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 19차시 시계열 예측 모델")
print("=" * 60)

print("""
1. 특성 엔지니어링
   - 날짜 특성: month, year, quarter + 주기적 인코딩(sin/cos)
   - Lag 특성: shift(1), shift(12) 등
   - Rolling 특성: shift(1).rolling(n).mean()
   - 중요: shift(1) 필수로 미래 누출 방지!

2. ML 모델 적용
   from sklearn.ensemble import RandomForestRegressor

   model = RandomForestRegressor(n_estimators=100)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

3. 시간 기준 분할
   # 시간 순서 유지 (랜덤 분할 금지!)
   split_date = '1958-12-01'
   X_train = features[:split_date]
   X_test = features[split_date:]

4. 평가 지표
   from sklearn.metrics import mean_absolute_error, mean_squared_error

   MAE: 평균 절대 오차 (해석 쉬움)
   RMSE: 평균 제곱근 오차 (큰 오차 패널티)
   MAPE: 평균 절대 비율 오차 (% 단위)

5. TimeSeriesSplit 교차검증
   from sklearn.model_selection import TimeSeriesSplit
   tscv = TimeSeriesSplit(n_splits=5)

6. 특성 중요도
   importance = model.feature_importances_
   -> 보통 lag_1, lag_12 등이 높게 나옴

7. 사용한 데이터셋
   - AirPassengers: 1949-1960 월별 항공 승객 수
   - 추세(trend)와 계절성(seasonality)이 있는 클래식 시계열
""")

print("\n다음 차시 예고: 20차시 - 딥러닝 입문: 신경망 기초")
