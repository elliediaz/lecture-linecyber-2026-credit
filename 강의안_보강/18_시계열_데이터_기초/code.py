"""
[18차시] 시계열 데이터 기초 - 실습 코드
=============================================

학습 목표:
1. 시계열 데이터의 특성을 이해한다
2. Python datetime 처리를 수행한다
3. 시계열 전처리 기법을 활용한다

실습 내용:
- datetime 모듈 사용
- Pandas 날짜 처리 (to_datetime, dt 접근자)
- resample (주기 변환)
- rolling (이동 통계)
- shift (Lag 특성)
- 시간 기준 분할

데이터셋: AirPassengers (1949-1960 월별 항공 승객 수)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: Python datetime 기초
# ============================================================

print("=" * 60)
print("Part 1: Python datetime 기초")
print("=" * 60)

# 현재 시각
now = datetime.now()
print(f"\n현재 시각: {now}")

# 특정 날짜 생성
dt = datetime(2025, 1, 5, 14, 30, 0)
print(f"특정 날짜: {dt}")

# 문자열 → datetime (strptime)
date_string = "2025-01-05"
parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
print(f"파싱된 날짜: {parsed_date}")

# datetime → 문자열 (strftime)
formatted = dt.strftime("%Y년 %m월 %d일 %H시 %M분")
print(f"포맷된 문자열: {formatted}")

# timedelta로 날짜 계산
future = now + timedelta(days=7)
past = now - timedelta(hours=3)
print(f"\n7일 후: {future}")
print(f"3시간 전: {past}")

# 날짜 차이 계산
dt1 = datetime(2025, 1, 1)
dt2 = datetime(2025, 1, 31)
diff = dt2 - dt1
print(f"\n날짜 차이: {diff.days}일")


# ============================================================
# Part 2: 실제 시계열 데이터 로드 (AirPassengers)
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 실제 시계열 데이터 로드 (AirPassengers)")
print("=" * 60)

# AirPassengers 데이터셋 로드
try:
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    print("\n[데이터 로드 성공]")
except Exception as e:
    print(f"\n[데이터 로드 실패: {e}]")
    print("오프라인 모드로 샘플 데이터를 생성합니다.")
    # 오프라인 대비 샘플 데이터
    np.random.seed(42)
    dates = pd.date_range(start='1949-01', periods=144, freq='MS')
    passengers = 100 + np.cumsum(np.random.normal(2, 10, 144))
    passengers = np.maximum(passengers, 50)  # 최소값 보장
    df = pd.DataFrame({'Month': dates.strftime('%Y-%m'), 'Passengers': passengers.astype(int)})

# 열 이름 확인 및 정리
print(f"\n원본 열 이름: {df.columns.tolist()}")

# 열 이름 표준화
df.columns = ['Month', 'Passengers']

# 날짜 파싱
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"기간: {df.index.min()} ~ {df.index.max()}")
print(f"\n데이터 타입:")
print(df.dtypes)
print(f"\n첫 5행:")
print(df.head())
print(f"\n기술 통계:")
print(df.describe())


# ============================================================
# Part 3: Pandas 날짜 처리
# ============================================================

print("\n" + "=" * 60)
print("Part 3: Pandas 날짜 처리")
print("=" * 60)

# pd.to_datetime 확인
print(f"\n인덱스 타입: {type(df.index)}")

# 다양한 형식 파싱
test_dates = [
    "2025-01-05",
    "01/05/2025",
    "Jan 5, 2025",
    "2025/01/05 14:30:00"
]

print("\n[다양한 날짜 형식 파싱]")
for d in test_dates:
    parsed = pd.to_datetime(d)
    print(f"  '{d}' → {parsed}")


# ============================================================
# Part 4: dt 접근자로 정보 추출
# ============================================================

print("\n" + "=" * 60)
print("Part 4: dt 접근자로 정보 추출")
print("=" * 60)

# 날짜 특성 추출 (인덱스에서)
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofweek'] = df.index.dayofweek  # 0=월요일
df['month_name'] = df.index.month_name()

print("\n[추출된 날짜 특성]")
print(df[['Passengers', 'year', 'month', 'quarter', 'month_name']].head(12))

# 월별 평균 (계절성 확인)
print("\n[월별 평균 승객 수 - 계절성 확인]")
monthly_avg = df.groupby('month')['Passengers'].mean()
for month, passengers in monthly_avg.items():
    months_kr = ['1월', '2월', '3월', '4월', '5월', '6월',
                 '7월', '8월', '9월', '10월', '11월', '12월']
    print(f"  {months_kr[month-1]}: {passengers:.1f}명")


# ============================================================
# Part 5: DatetimeIndex 슬라이싱
# ============================================================

print("\n" + "=" * 60)
print("Part 5: DatetimeIndex 슬라이싱")
print("=" * 60)

# 특정 연도
print("\n[1955년 데이터]")
print(f"데이터 수: {len(df['1955'])}")
print(df['1955'][['Passengers', 'month_name']])

# 기간 선택
print("\n[1949-01 ~ 1950-12 데이터]")
subset = df['1949':'1950']
print(f"데이터 수: {len(subset)}")

# 연도-월 선택
print("\n[1960년 상반기 데이터]")
first_half = df['1960-01':'1960-06']
print(first_half[['Passengers']])


# ============================================================
# Part 6: resample - 주기 변환
# ============================================================

print("\n" + "=" * 60)
print("Part 6: resample - 주기 변환")
print("=" * 60)

# 월별 → 분기별 평균
quarterly_mean = df[['Passengers']].resample('Q').mean()
print("\n[분기별 평균 (월별 → 분기별)]")
print(f"원본 shape: {df.shape}")
print(f"분기별 shape: {quarterly_mean.shape}")
print(quarterly_mean.head(8))

# 월별 → 연도별 통계
print("\n[연도별 다양한 집계]")
yearly_agg = df['Passengers'].resample('YE').agg(['mean', 'max', 'min', 'std'])
print(yearly_agg)

# 월별 → 반기별
print("\n[반기별 집계]")
semiannual = df[['Passengers']].resample('6ME').mean()
print(f"반기별 shape: {semiannual.shape}")
print(semiannual.head(8))


# ============================================================
# Part 7: rolling - 이동 통계
# ============================================================

print("\n" + "=" * 60)
print("Part 7: rolling - 이동 통계")
print("=" * 60)

# 3개월 이동평균
df['ma_3'] = df['Passengers'].rolling(window=3).mean()

# 12개월(1년) 이동평균 - 계절성 제거
df['ma_12'] = df['Passengers'].rolling(window=12).mean()

# 이동 표준편차
df['std_12'] = df['Passengers'].rolling(window=12).std()

# 이동 최댓값/최솟값
df['max_12'] = df['Passengers'].rolling(window=12).max()
df['min_12'] = df['Passengers'].rolling(window=12).min()

print("\n[이동 통계 결과]")
print(df[['Passengers', 'ma_3', 'ma_12', 'std_12']].tail(12))

# NaN 개수 확인
print(f"\nNaN 개수 (ma_3): {df['ma_3'].isna().sum()}")
print(f"NaN 개수 (ma_12): {df['ma_12'].isna().sum()}")


# ============================================================
# Part 8: shift - Lag 특성
# ============================================================

print("\n" + "=" * 60)
print("Part 8: shift - Lag 특성")
print("=" * 60)

# Lag 특성 생성
df['lag_1'] = df['Passengers'].shift(1)      # 1개월 전
df['lag_3'] = df['Passengers'].shift(3)      # 3개월 전
df['lag_12'] = df['Passengers'].shift(12)    # 12개월 전 (작년 같은 달)

print("\n[Lag 특성 결과]")
print(df[['Passengers', 'lag_1', 'lag_3', 'lag_12']].head(15))

# NaN 개수
print(f"\nNaN 개수 (lag_1): {df['lag_1'].isna().sum()}")
print(f"NaN 개수 (lag_3): {df['lag_3'].isna().sum()}")
print(f"NaN 개수 (lag_12): {df['lag_12'].isna().sum()}")


# ============================================================
# Part 9: 데이터 누출 방지 - 올바른 Rolling
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 데이터 누출 방지")
print("=" * 60)

# 잘못된 방법: 현재 값 포함
wrong_ma = df['Passengers'].rolling(3).mean()

# 올바른 방법: shift 먼저
correct_ma = df['Passengers'].shift(1).rolling(3).mean()

df['wrong_ma_3'] = wrong_ma
df['correct_ma_3'] = correct_ma

print("\n[잘못된 vs 올바른 Rolling]")
print(df[['Passengers', 'wrong_ma_3', 'correct_ma_3']].iloc[3:9])

print("\n주의:")
print("  - wrong_ma_3: 현재 승객수가 평균에 포함됨 (미래 누출!)")
print("  - correct_ma_3: 이전 달까지의 데이터만 사용 (안전)")


# ============================================================
# Part 10: diff와 pct_change
# ============================================================

print("\n" + "=" * 60)
print("Part 10: diff와 pct_change")
print("=" * 60)

# 1개월 차분
df['diff_1'] = df['Passengers'].diff(1)

# 12개월 차분 (전년 동월 대비 변화)
df['diff_12'] = df['Passengers'].diff(12)

# 변화율
df['pct_change'] = df['Passengers'].pct_change()

# 전년 동월 대비 변화율
df['yoy_change'] = df['Passengers'].pct_change(12)

print("\n[차분 및 변화율]")
print(df[['Passengers', 'diff_1', 'pct_change', 'yoy_change']].tail(12))

# 변화율 통계
print(f"\n월간 변화율 통계:")
print(f"  평균: {df['pct_change'].mean():.4f} ({df['pct_change'].mean()*100:.2f}%)")
print(f"  표준편차: {df['pct_change'].std():.4f}")

print(f"\n전년 동월 대비 변화율 통계:")
print(f"  평균: {df['yoy_change'].mean():.4f} ({df['yoy_change'].mean()*100:.2f}%)")


# ============================================================
# Part 11: 특성 엔지니어링 종합
# ============================================================

print("\n" + "=" * 60)
print("Part 11: 특성 엔지니어링 종합")
print("=" * 60)

# 예측을 위한 특성 세트 생성
features = pd.DataFrame(index=df.index)

# 날짜 특성
features['month'] = df.index.month
features['quarter'] = df.index.quarter
features['year'] = df.index.year

# 주기적 인코딩 (월의 순환 특성 반영)
features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

# Lag 특성 (shift 필수!)
features['lag_1'] = df['Passengers'].shift(1)
features['lag_2'] = df['Passengers'].shift(2)
features['lag_3'] = df['Passengers'].shift(3)
features['lag_12'] = df['Passengers'].shift(12)

# Rolling 특성 (shift 먼저!)
features['ma_3'] = df['Passengers'].shift(1).rolling(3).mean()
features['ma_6'] = df['Passengers'].shift(1).rolling(6).mean()
features['ma_12'] = df['Passengers'].shift(1).rolling(12).mean()
features['std_3'] = df['Passengers'].shift(1).rolling(3).std()

# 타겟
features['target'] = df['Passengers']

print("\n[특성 세트]")
print(features.head(15))
print(f"\n특성 수: {len(features.columns) - 1}")  # target 제외


# ============================================================
# Part 12: 시간 기준 분할
# ============================================================

print("\n" + "=" * 60)
print("Part 12: 시간 기준 분할")
print("=" * 60)

# NaN 제거
features_clean = features.dropna()
print(f"\n원본 데이터: {len(features)}개")
print(f"NaN 제거 후: {len(features_clean)}개")

# 시간 기준 분할 (마지막 2년을 테스트로)
split_date = '1958-12-01'

train = features_clean[:split_date]
test = features_clean[split_date:]

print(f"\n[시간 기준 분할]")
print(f"분할 기준: {split_date}")
print(f"학습 데이터: {len(train)}개 ({train.index.min()} ~ {train.index.max()})")
print(f"테스트 데이터: {len(test)}개 ({test.index.min()} ~ {test.index.max()})")

# 학습/테스트 분리
X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop('target', axis=1)
y_test = test['target']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ============================================================
# Part 13: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 13: 시각화")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# 1. 원본 시계열
ax1 = axes[0, 0]
df['Passengers'].plot(ax=ax1, alpha=0.7)
ax1.set_title('항공 승객 수 시계열 (1949-1960)')
ax1.set_xlabel('연도')
ax1.set_ylabel('승객 수 (천 명)')
ax1.grid(True, alpha=0.3)

# 2. 원본 vs 이동평균
ax2 = axes[0, 1]
df['Passengers'].plot(ax=ax2, alpha=0.3, label='원본')
df['ma_12'].plot(ax=ax2, label='12개월 이동평균', linewidth=2)
ax2.set_title('원본 vs 12개월 이동평균')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 연도별 평균
ax3 = axes[1, 0]
yearly_mean = df.groupby('year')['Passengers'].mean()
ax3.bar(yearly_mean.index, yearly_mean.values, color='steelblue')
ax3.set_title('연도별 평균 승객 수')
ax3.set_xlabel('연도')
ax3.set_ylabel('평균 승객 수')
ax3.grid(True, alpha=0.3, axis='y')

# 4. 월별 평균 (계절성)
ax4 = axes[1, 1]
monthly_avg = df.groupby('month')['Passengers'].mean()
months_kr = ['1월', '2월', '3월', '4월', '5월', '6월',
             '7월', '8월', '9월', '10월', '11월', '12월']
ax4.bar(range(1, 13), monthly_avg.values, color='coral')
ax4.set_title('월별 평균 승객 수 (계절성)')
ax4.set_xlabel('월')
ax4.set_ylabel('평균 승객 수')
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(months_kr, rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

# 5. 전년 동월 대비 변화율
ax5 = axes[2, 0]
df['yoy_change'].dropna().plot(ax=ax5, alpha=0.7)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax5.set_title('전년 동월 대비 변화율 (YoY)')
ax5.set_xlabel('연도')
ax5.set_ylabel('변화율')
ax5.grid(True, alpha=0.3)

# 6. 학습/테스트 분할 시각화
ax6 = axes[2, 1]
train['target'].plot(ax=ax6, label='학습', alpha=0.7)
test['target'].plot(ax=ax6, label='테스트', alpha=0.7)
ax6.axvline(x=pd.Timestamp(split_date), color='red', linestyle='--', label='분할 기준')
ax6.set_title('학습/테스트 분할')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('18_timeseries_basics.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 18_timeseries_basics.png")


# ============================================================
# Part 14: TimeSeriesSplit (sklearn)
# ============================================================

print("\n" + "=" * 60)
print("Part 14: TimeSeriesSplit (sklearn)")
print("=" * 60)

from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit 교차검증
tscv = TimeSeriesSplit(n_splits=5)

print("\n[TimeSeriesSplit 교차검증]")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    train_start = X_train.index[train_idx[0]]
    train_end = X_train.index[train_idx[-1]]
    val_start = X_train.index[val_idx[0]]
    val_end = X_train.index[val_idx[-1]]

    print(f"Fold {fold}:")
    print(f"  학습: {train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')} ({len(train_idx)}개)")
    print(f"  검증: {val_start.strftime('%Y-%m')} ~ {val_end.strftime('%Y-%m')} ({len(val_idx)}개)")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 18차시 시계열 데이터 기초")
print("=" * 60)

print("""
1. 시계열 데이터 특성
   - 시간 순서가 중요
   - 시간 의존성, 자기상관, 계절성
   - 시간 기준 분할 필수 (랜덤 분할 금지!)

2. datetime 처리
   # 문자열 → datetime
   df['date'] = pd.to_datetime(df['date'])

   # dt 접근자로 정보 추출
   df['month'] = df['date'].dt.month
   df['year'] = df['date'].dt.year
   df['quarter'] = df['date'].dt.quarter

3. 전처리 기법
   # resample: 주기 변환
   quarterly = df.resample('Q').mean()

   # rolling: 이동 통계
   df['ma_12'] = df['value'].rolling(12).mean()

   # shift: Lag 특성
   df['lag_1'] = df['value'].shift(1)

4. 데이터 누출 방지 (매우 중요!)
   # 잘못됨
   df['ma_3'] = df['value'].rolling(3).mean()

   # 올바름 (shift 먼저!)
   df['ma_3'] = df['value'].shift(1).rolling(3).mean()

5. 시간 기준 분할
   split_date = '1958-12-01'
   train = df[:split_date]
   test = df[split_date:]

6. sklearn TimeSeriesSplit
   from sklearn.model_selection import TimeSeriesSplit
   tscv = TimeSeriesSplit(n_splits=5)

7. 사용한 데이터셋
   - AirPassengers: 1949-1960 월별 항공 승객 수
   - 추세(trend)와 계절성(seasonality)이 명확한 클래식 시계열 데이터
""")

print("\n다음 차시 예고: 19차시 - 시계열 예측 모델")
