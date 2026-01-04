# [17차시] 시계열 예측 모델 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("17차시: 시계열 예측 모델")
print("과거 데이터로 미래를 예측합니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 시계열 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 시계열 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_days = 180

# 날짜 범위 생성
dates = pd.date_range('2024-01-01', periods=n_days)

# 생산량 데이터 (추세 + 계절성 + 노이즈)
trend = np.linspace(1000, 1100, n_days)  # 상승 추세
seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # 주간 패턴
noise = np.random.randn(n_days) * 20

production = trend + seasonal + noise

# DataFrame 생성
df = pd.DataFrame({
    '날짜': dates,
    '생산량': production.astype(int)
})
df = df.set_index('날짜')

print(f"데이터 크기: {df.shape}")
print(f"기간: {df.index.min().date()} ~ {df.index.max().date()}")
print("\n처음 5행:")
print(df.head())
print()


# ============================================================
# 실습 2: 날짜 기반 특성
# ============================================================
print("=" * 50)
print("실습 2: 날짜 기반 특성")
print("=" * 50)

# 날짜 특성 추출
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek  # 0=월요일
df['주차'] = df.index.isocalendar().week

# 특수 날짜 플래그
df['월초'] = (df.index.day <= 5).astype(int)
df['월말'] = (df.index.day >= 25).astype(int)
df['주말'] = (df.index.dayofweek >= 5).astype(int)

print("날짜 특성 생성 완료:")
print(df[['생산량', '월', '요일', '주말']].head(10))
print()


# ============================================================
# 실습 3: 시차 특성 (Lag Features)
# ============================================================
print("=" * 50)
print("실습 3: 시차 특성 (Lag Features)")
print("=" * 50)

# 시차 특성 생성 (과거 값)
df['lag_1'] = df['생산량'].shift(1)   # 1일 전
df['lag_7'] = df['생산량'].shift(7)   # 7일 전
df['lag_14'] = df['생산량'].shift(14) # 14일 전

print("시차 특성 생성:")
print(df[['생산량', 'lag_1', 'lag_7', 'lag_14']].head(15))
print()


# ============================================================
# 실습 4: 롤링 특성 (Rolling Features)
# ============================================================
print("=" * 50)
print("실습 4: 롤링 특성 (Rolling Features)")
print("=" * 50)

# ⚠️ 중요: shift(1)을 먼저 적용해서 데이터 누출 방지!
# 이동평균 (어제까지의 평균)
df['ma_7'] = df['생산량'].shift(1).rolling(7).mean()
df['ma_14'] = df['생산량'].shift(1).rolling(14).mean()

# 이동 표준편차 (변동성)
df['std_7'] = df['생산량'].shift(1).rolling(7).std()

# 이동 최대/최소
df['max_7'] = df['생산량'].shift(1).rolling(7).max()
df['min_7'] = df['생산량'].shift(1).rolling(7).min()

print("롤링 특성 생성 (shift(1) 적용!):")
print(df[['생산량', 'ma_7', 'std_7']].tail(10))
print()


# ============================================================
# 실습 5: 결측치 처리
# ============================================================
print("=" * 50)
print("실습 5: 결측치 처리")
print("=" * 50)

print(f"원본 데이터 크기: {len(df)}개")
print(f"결측치 개수:")
print(df.isnull().sum())

# 결측치 제거
df = df.dropna()
print(f"\n결측치 제거 후: {len(df)}개")
print(f"제거된 행: {180 - len(df)}개")
print()


# ============================================================
# 실습 6: 시간 기준 분할
# ============================================================
print("=" * 50)
print("실습 6: 시간 기준 분할")
print("=" * 50)

# ✅ 올바른 방법: 시간 기준 분할
split_date = '2024-05-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

print(f"분할 기준일: {split_date}")
print(f"Train: {len(train)}개 ({train.index.min().date()} ~ {train.index.max().date()})")
print(f"Test: {len(test)}개 ({test.index.min().date()} ~ {test.index.max().date()})")
print(f"Train 비율: {len(train) / len(df) * 100:.1f}%")

print("\n❌ 잘못된 방법: train_test_split (랜덤 분할)")
print("   → 미래 데이터가 Train에 포함되어 데이터 누출!")
print()


# ============================================================
# 실습 7: 모델 학습
# ============================================================
print("=" * 50)
print("실습 7: 모델 학습 (RandomForestRegressor)")
print("=" * 50)

# 특성 정의
features = ['월', '요일', 'lag_1', 'lag_7', 'ma_7', 'std_7', '주말']

# 학습/테스트 데이터
X_train = train[features]
y_train = train['생산량']
X_test = test[features]
y_test = test['생산량']

print(f"특성: {features}")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
print("\n모델 학습 완료!")
print()


# ============================================================
# 실습 8: 성능 평가
# ============================================================
print("=" * 50)
print("실습 8: 성능 평가 (MAE, RMSE, MAPE)")
print("=" * 50)

# MAE: Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)

# RMSE: Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# MAPE: Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"MAE:  {mae:.1f}개 (평균 절대 오차)")
print(f"RMSE: {rmse:.1f}개 (평균 제곱근 오차)")
print(f"MAPE: {mape:.1f}% (평균 백분율 오차)")

print("\nMAPE 해석:")
print("-" * 30)
if mape < 10:
    print(f"  {mape:.1f}% → 매우 좋음!")
elif mape < 20:
    print(f"  {mape:.1f}% → 좋음")
elif mape < 50:
    print(f"  {mape:.1f}% → 보통")
else:
    print(f"  {mape:.1f}% → 개선 필요")
print()


# ============================================================
# 실습 9: 결과 시각화
# ============================================================
print("=" * 50)
print("실습 9: 결과 시각화")
print("=" * 50)

# 실제 vs 예측 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 전체 기간 + 예측 결과
axes[0].plot(train.index, train['생산량'], label='Train', alpha=0.7)
axes[0].plot(test.index, y_test, label='Test (실제)', linewidth=2)
axes[0].plot(test.index, predictions, label='예측', linestyle='--', linewidth=2)
axes[0].axvline(x=pd.Timestamp(split_date), color='red', linestyle=':', label='분할 시점')
axes[0].set_title('생산량 예측 결과')
axes[0].set_xlabel('날짜')
axes[0].set_ylabel('생산량')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 예측 오차
errors = y_test - predictions
axes[1].bar(test.index, errors, alpha=0.7, color='coral')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_title('예측 오차 (실제 - 예측)')
axes[1].set_xlabel('날짜')
axes[1].set_ylabel('오차')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('시계열_예측_결과.png', dpi=100)
plt.close()
print("예측 결과 그래프 저장: 시계열_예측_결과.png")
print()


# ============================================================
# 실습 10: 특성 중요도
# ============================================================
print("=" * 50)
print("실습 10: 특성 중요도")
print("=" * 50)

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("특성 중요도:")
print("-" * 40)
for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:<10} {row['importance']:.3f} {bar}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance['feature'], importance['importance'], color='steelblue')
ax.set_xlabel('중요도')
ax.set_title('특성 중요도 (Feature Importance)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('특성_중요도.png', dpi=100)
plt.close()
print("\n특성 중요도 그래프 저장: 특성_중요도.png")
print()


# ============================================================
# 실습 11: 예측 결과 상세 분석
# ============================================================
print("=" * 50)
print("실습 11: 예측 결과 상세 분석")
print("=" * 50)

# 예측 결과 DataFrame
results = pd.DataFrame({
    '실제': y_test,
    '예측': predictions.astype(int),
    '오차': (y_test - predictions).astype(int),
    '오차율(%)': np.abs((y_test - predictions) / y_test * 100).round(1)
})

print("예측 결과 샘플 (처음 10개):")
print(results.head(10))

print(f"\n오차 통계:")
print(f"  평균 오차: {results['오차'].mean():.1f}개")
print(f"  오차 표준편차: {results['오차'].std():.1f}개")
print(f"  최대 과대예측: {results['오차'].min():.0f}개")
print(f"  최대 과소예측: {results['오차'].max():.0f}개")
print()


# ============================================================
# 실습 12: 요일별 예측 성능
# ============================================================
print("=" * 50)
print("실습 12: 요일별 예측 성능")
print("=" * 50)

# 요일 정보 추가
results['요일'] = test['요일']
weekday_names = ['월', '화', '수', '목', '금', '토', '일']

# 요일별 MAPE
weekday_mape = results.groupby('요일')['오차율(%)'].mean()

print("요일별 평균 오차율 (MAPE):")
print("-" * 30)
for dow, mape_val in weekday_mape.items():
    print(f"  {weekday_names[dow]}요일: {mape_val:.1f}%")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              시계열 예측 모델 핵심 정리                 │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 특성 엔지니어링                                     │
│     시계열 → 테이블 형태로 변환                        │
│                                                        │
│     1. 날짜 특성: df.index.month, dayofweek            │
│     2. 시차 특성: df['생산량'].shift(1)                │
│     3. 롤링 특성: shift(1).rolling(7).mean()           │
│                                                        │
│  ▶ 데이터 누출 방지 (매우 중요!)                       │
│     롤링 특성 → shift(1) 먼저!                         │
│     ❌ rolling(7).mean()                               │
│     ✅ shift(1).rolling(7).mean()                      │
│                                                        │
│  ▶ 시간 기준 분할                                      │
│     train = df[df.index < split_date]                  │
│     test = df[df.index >= split_date]                  │
│     ❌ train_test_split (랜덤 분할 금지!)              │
│                                                        │
│  ▶ 평가 지표                                           │
│     MAE: {mae:.1f}개                                    │
│     RMSE: {rmse:.1f}개                                  │
│     MAPE: {mape:.1f}% (10-20% 이내가 좋음)             │
│                                                        │
│  ★ lag_1 (전일값)이 가장 중요한 특성!                  │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 딥러닝 입문 - 신경망 기초
""")

print("=" * 60)
print("17차시 실습 완료!")
print("과거 데이터로 미래를 예측하는 법을 배웠습니다!")
print("=" * 60)
