"""
[17차시] 시계열 데이터 기초 - 실습 코드
학습목표: 시계열 데이터 특성 이해, datetime 처리, 시계열 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. datetime 모듈 기초
# ============================================================
print("=" * 50)
print("1. datetime 모듈 기초")
print("=" * 50)

# 현재 시간
now = datetime.now()
print(f"현재 시간: {now}")

# 특정 날짜 생성
date = datetime(2024, 1, 15)
print(f"특정 날짜: {date}")

# 문자열 → datetime (strptime)
date_str = '2024-01-15'
date_parsed = datetime.strptime(date_str, '%Y-%m-%d')
print(f"문자열 → datetime: {date_parsed}")

# datetime → 문자열 (strftime)
formatted = now.strftime('%Y년 %m월 %d일 %H시 %M분')
print(f"datetime → 문자열: {formatted}")

# ============================================================
# 2. 날짜 연산 (timedelta)
# ============================================================
print("\n" + "=" * 50)
print("2. 날짜 연산")
print("=" * 50)

today = datetime(2024, 6, 15)

# 날짜 더하기/빼기
next_week = today + timedelta(days=7)
last_month = today - timedelta(days=30)

print(f"오늘: {today.strftime('%Y-%m-%d')}")
print(f"7일 후: {next_week.strftime('%Y-%m-%d')}")
print(f"30일 전: {last_month.strftime('%Y-%m-%d')}")

# 두 날짜 사이 간격
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 12, 31)
diff = date2 - date1
print(f"\n{date1.strftime('%Y-%m-%d')} ~ {date2.strftime('%Y-%m-%d')}: {diff.days}일")

# ============================================================
# 3. 제조 시계열 데이터 생성
# ============================================================
print("\n" + "=" * 50)
print("3. 시계열 데이터 생성")
print("=" * 50)

np.random.seed(42)

# 180일간의 생산 데이터
n_days = 180
dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

# 생산량 (추세 + 계절성 + 노이즈)
trend = np.linspace(1000, 1100, n_days)  # 상승 추세
seasonality = 50 * np.sin(np.arange(n_days) * 2 * np.pi / 7)  # 주간 패턴
noise = np.random.normal(0, 30, n_days)
production = trend + seasonality + noise

# 불량수
defects = np.random.poisson(12, n_days)

df = pd.DataFrame({
    '날짜': dates,
    '생산량': production,
    '불량수': defects
})

print(df.head(10))
print(f"\n데이터 기간: {df['날짜'].min()} ~ {df['날짜'].max()}")

# ============================================================
# 4. Pandas 날짜 처리
# ============================================================
print("\n" + "=" * 50)
print("4. Pandas 날짜 처리")
print("=" * 50)

# 날짜 정보 추출 (dt 접근자)
df['연도'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0=월, 6=일
df['주차'] = df['날짜'].dt.isocalendar().week

print("▶ 날짜 정보 추출:")
print(df[['날짜', '연도', '월', '일', '요일', '주차']].head())

# 요일별 평균 생산량
print("\n▶ 요일별 평균 생산량:")
weekday_avg = df.groupby('요일')['생산량'].mean()
weekday_names = ['월', '화', '수', '목', '금', '토', '일']
for i, avg in enumerate(weekday_avg):
    print(f"   {weekday_names[i]}요일: {avg:.1f}")

# ============================================================
# 5. 날짜 인덱스 활용
# ============================================================
print("\n" + "=" * 50)
print("5. 날짜 인덱스 활용")
print("=" * 50)

# 날짜를 인덱스로 설정
df_indexed = df.set_index('날짜')

# 날짜로 필터링
print("▶ 2024년 3월 데이터:")
march_data = df_indexed['2024-03']
print(f"   데이터 수: {len(march_data)}")
print(f"   평균 생산량: {march_data['생산량'].mean():.1f}")

# 기간 필터링
print("\n▶ 2024년 1분기 (1~3월):")
q1_data = df_indexed['2024-01':'2024-03']
print(f"   데이터 수: {len(q1_data)}")
print(f"   평균 생산량: {q1_data['생산량'].mean():.1f}")

# ============================================================
# 6. 시계열 시각화
# ============================================================
print("\n" + "=" * 50)
print("6. 시계열 시각화")
print("=" * 50)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 생산량 추이
axes[0].plot(df['날짜'], df['생산량'], linewidth=1)
axes[0].set_title('Daily Production')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Production')
axes[0].grid(True, alpha=0.3)

# 불량수 추이
axes[1].plot(df['날짜'], df['불량수'], linewidth=1, color='red')
axes[1].set_title('Daily Defects')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Defects')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ timeseries_basic.png 저장됨")

# ============================================================
# 7. 리샘플링 (Resampling)
# ============================================================
print("\n" + "=" * 50)
print("7. 리샘플링")
print("=" * 50)

# 월별 집계
monthly = df_indexed.resample('M').agg({
    '생산량': 'mean',
    '불량수': 'sum'
})
print("▶ 월별 집계:")
print(monthly)

# 주별 집계
weekly = df_indexed.resample('W').mean()
print("\n▶ 주별 평균 (처음 5개):")
print(weekly.head())

# ============================================================
# 8. 이동평균 (Rolling Mean)
# ============================================================
print("\n" + "=" * 50)
print("8. 이동평균")
print("=" * 50)

# 이동평균 계산
df['MA_7'] = df['생산량'].rolling(window=7).mean()    # 7일 이동평균
df['MA_30'] = df['생산량'].rolling(window=30).mean()  # 30일 이동평균

print("▶ 이동평균 예시:")
print(df[['날짜', '생산량', 'MA_7', 'MA_30']].tail(10))

# 이동평균 시각화
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['날짜'], df['생산량'], alpha=0.5, label='Daily', linewidth=1)
ax.plot(df['날짜'], df['MA_7'], label='7-day MA', linewidth=2)
ax.plot(df['날짜'], df['MA_30'], label='30-day MA', linewidth=2)
ax.set_title('Production with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Production')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('moving_average.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ moving_average.png 저장됨")

# ============================================================
# 9. Shift 연산 (시차 변수)
# ============================================================
print("\n" + "=" * 50)
print("9. Shift 연산")
print("=" * 50)

# 시차 변수 생성
df['전일_생산량'] = df['생산량'].shift(1)   # 1일 전
df['일주일전_생산량'] = df['생산량'].shift(7)  # 7일 전

# 변화량과 변화율
df['일별_변화'] = df['생산량'] - df['생산량'].shift(1)
df['일별_변화율'] = df['생산량'].pct_change() * 100

print("▶ 시차 변수 예시:")
print(df[['날짜', '생산량', '전일_생산량', '일별_변화', '일별_변화율']].iloc[1:6])

# ============================================================
# 10. 시계열 Train/Test 분할
# ============================================================
print("\n" + "=" * 50)
print("10. 시계열 Train/Test 분할")
print("=" * 50)

# 시간 기준 분할 (무작위 분할 X)
split_date = '2024-05-01'
train = df[df['날짜'] < split_date]
test = df[df['날짜'] >= split_date]

print(f"▶ 분할 기준일: {split_date}")
print(f"   학습 데이터: {len(train)}일 ({train['날짜'].min()} ~ {train['날짜'].max()})")
print(f"   테스트 데이터: {len(test)}일 ({test['날짜'].min()} ~ {test['날짜'].max()})")

print("""
⚠️ 시계열 데이터는 반드시 시간 기준으로 분할!
   - 과거 → 학습
   - 미래 → 테스트
   - 무작위 분할하면 미래 정보가 학습에 들어감 (정보 누출)
""")

# ============================================================
# 11. 핵심 요약
# ============================================================
print("=" * 50)
print("11. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                시계열 데이터 기초 정리                   │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 날짜 처리                                           │
│     df['날짜'] = pd.to_datetime(df['날짜'])            │
│     df['연도'] = df['날짜'].dt.year                    │
│     df['월'] = df['날짜'].dt.month                     │
│     df['요일'] = df['날짜'].dt.dayofweek               │
│                                                        │
│  ▶ 날짜 인덱스                                         │
│     df = df.set_index('날짜')                         │
│     df['2024-01']  # 2024년 1월                       │
│                                                        │
│  ▶ 리샘플링                                            │
│     df.resample('M').mean()  # 월별 평균               │
│     df.resample('W').sum()   # 주별 합계               │
│                                                        │
│  ▶ 이동평균                                            │
│     df['생산량'].rolling(7).mean()  # 7일 이동평균     │
│                                                        │
│  ▶ 시차 변수                                           │
│     df['전일'] = df['생산량'].shift(1)                 │
│     df['변화'] = df['생산량'].pct_change()             │
│                                                        │
│  ★ Train/Test 분할: 시간 기준으로! (무작위 X)           │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 시계열 예측 모델
""")
