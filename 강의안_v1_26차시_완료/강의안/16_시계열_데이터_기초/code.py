# [16차시] 시계열 데이터 기초 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("16차시: 시계열 데이터 기초")
print("시간에 따라 변하는 데이터를 다뤄봅니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: Python datetime 기초
# ============================================================
print("=" * 50)
print("실습 1: Python datetime 기초")
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
print(f"문자열 파싱: {date_parsed}")

# datetime → 문자열 (strftime)
formatted = now.strftime('%Y년 %m월 %d일')
print(f"포맷 변환: {formatted}")
print()


# ============================================================
# 실습 2: timedelta 날짜 연산
# ============================================================
print("=" * 50)
print("실습 2: timedelta 날짜 연산")
print("=" * 50)

today = datetime.now()
print(f"오늘: {today.strftime('%Y-%m-%d')}")

# 7일 후
next_week = today + timedelta(days=7)
print(f"7일 후: {next_week.strftime('%Y-%m-%d')}")

# 30일 전
last_month = today - timedelta(days=30)
print(f"30일 전: {last_month.strftime('%Y-%m-%d')}")

# 두 날짜 사이 간격
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 12, 31)
diff = date2 - date1
print(f"2024년 일수: {diff.days}일")
print()


# ============================================================
# 실습 3: 시계열 데이터 생성
# ============================================================
print("=" * 50)
print("실습 3: 시계열 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_days = 180

# 날짜 범위 생성
dates = pd.date_range('2024-01-01', periods=n_days)
print(f"날짜 범위: {dates[0]} ~ {dates[-1]}")

# 생산량 데이터 (추세 + 노이즈)
trend = np.linspace(1000, 1100, n_days)  # 상승 추세
noise = np.random.randn(n_days) * 30
production = trend + noise

# 온도 데이터 (계절성)
temperature = 85 + 5 * np.sin(2 * np.pi * np.arange(n_days) / 30) + np.random.randn(n_days) * 2

# DataFrame 생성
df = pd.DataFrame({
    '날짜': dates,
    '생산량': production.astype(int),
    '온도': temperature.round(1)
})

print(f"데이터 크기: {df.shape}")
print("\n처음 5행:")
print(df.head())
print()


# ============================================================
# 실습 4: pd.to_datetime 변환
# ============================================================
print("=" * 50)
print("실습 4: pd.to_datetime 변환")
print("=" * 50)

# 이미 datetime이지만, 문자열인 경우 변환 예시
print(f"변환 전 dtype: {df['날짜'].dtype}")

# 다양한 형식 변환 예시
print("\npd.to_datetime 다양한 형식:")
print(f"  '2024-01-15' → {pd.to_datetime('2024-01-15')}")
print(f"  '01/15/2024' → {pd.to_datetime('01/15/2024')}")
print(f"  '15-Jan-2024' → {pd.to_datetime('15-Jan-2024')}")

# 날짜를 인덱스로 설정
df = df.set_index('날짜')
print(f"\n인덱스로 설정 후:")
print(df.head())
print()


# ============================================================
# 실습 5: dt 접근자로 날짜 정보 추출
# ============================================================
print("=" * 50)
print("실습 5: dt 접근자로 날짜 정보 추출")
print("=" * 50)

# 인덱스에서 정보 추출
df['연도'] = df.index.year
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek  # 0=월요일, 6=일요일
df['요일명'] = df.index.day_name()
df['주차'] = df.index.isocalendar().week

print("날짜 정보 추출 결과:")
print(df[['생산량', '연도', '월', '일', '요일', '요일명']].head(10))
print()


# ============================================================
# 실습 6: 날짜 필터링
# ============================================================
print("=" * 50)
print("실습 6: 날짜 필터링")
print("=" * 50)

# 특정 월 선택
jan_data = df['2024-01']
print(f"1월 데이터: {len(jan_data)}개")

# 기간 선택
first_quarter = df['2024-01-01':'2024-03-31']
print(f"1분기 데이터: {len(first_quarter)}개")

# 특정 조건
high_production = df[df['생산량'] > 1080]
print(f"생산량 1080 초과: {len(high_production)}개")
print()


# ============================================================
# 실습 7: 리샘플링 (resample)
# ============================================================
print("=" * 50)
print("실습 7: 리샘플링 (resample)")
print("=" * 50)

# 일별 → 주별 평균
weekly_avg = df['생산량'].resample('W').mean()
print("주별 평균 생산량:")
print(weekly_avg.head())

# 일별 → 월별 합계
monthly_sum = df['생산량'].resample('M').sum()
print("\n월별 총 생산량:")
print(monthly_sum)

# 일별 → 월별 통계
monthly_stats = df['생산량'].resample('M').agg(['mean', 'std', 'min', 'max'])
print("\n월별 생산량 통계:")
print(monthly_stats)
print()


# ============================================================
# 실습 8: 이동평균 (rolling)
# ============================================================
print("=" * 50)
print("실습 8: 이동평균 (rolling)")
print("=" * 50)

# 7일 이동평균
df['이동평균_7일'] = df['생산량'].rolling(window=7).mean()

# 30일 이동평균
df['이동평균_30일'] = df['생산량'].rolling(window=30).mean()

print("이동평균 계산 결과:")
print(df[['생산량', '이동평균_7일', '이동평균_30일']].tail(10))

# 시각화
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['생산량'], alpha=0.5, label='일별 생산량')
ax.plot(df.index, df['이동평균_7일'], label='7일 이동평균', linewidth=2)
ax.plot(df.index, df['이동평균_30일'], label='30일 이동평균', linewidth=2)
ax.set_title('생산량 추이 및 이동평균')
ax.set_xlabel('날짜')
ax.set_ylabel('생산량')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('이동평균_시각화.png', dpi=100)
plt.close()
print("\n이동평균 그래프 저장: 이동평균_시각화.png")
print()


# ============================================================
# 실습 9: Shift 연산
# ============================================================
print("=" * 50)
print("실습 9: Shift 연산")
print("=" * 50)

# 1일 전 값
df['전일_생산량'] = df['생산량'].shift(1)

# 일별 변화량
df['변화량'] = df['생산량'] - df['생산량'].shift(1)

# 일별 변화율
df['변화율'] = df['생산량'].pct_change()

print("Shift 연산 결과:")
print(df[['생산량', '전일_생산량', '변화량', '변화율']].head(10))

# 변화량 통계
print(f"\n일별 변화량 통계:")
print(f"  평균: {df['변화량'].mean():.2f}")
print(f"  표준편차: {df['변화량'].std():.2f}")
print(f"  최대 증가: {df['변화량'].max():.0f}")
print(f"  최대 감소: {df['변화량'].min():.0f}")
print()


# ============================================================
# 실습 10: 요일별 분석
# ============================================================
print("=" * 50)
print("실습 10: 요일별 분석")
print("=" * 50)

# 요일별 평균 생산량
weekday_avg = df.groupby('요일')['생산량'].mean()
print("요일별 평균 생산량:")
weekday_names = ['월', '화', '수', '목', '금', '토', '일']
for i, avg in enumerate(weekday_avg):
    print(f"  {weekday_names[i]}요일: {avg:.1f}")

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 생산량 추이
axes[0].plot(df.index, df['생산량'])
axes[0].set_title('일별 생산량 추이')
axes[0].set_xlabel('날짜')
axes[0].set_ylabel('생산량')
axes[0].grid(True, alpha=0.3)

# 요일별 평균
axes[1].bar(range(7), weekday_avg)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(weekday_names)
axes[1].set_title('요일별 평균 생산량')
axes[1].set_xlabel('요일')
axes[1].set_ylabel('평균 생산량')

plt.tight_layout()
plt.savefig('시계열_분석.png', dpi=100)
plt.close()
print("\n분석 그래프 저장: 시계열_분석.png")
print()


# ============================================================
# 실습 11: 시계열 분할 (중요!)
# ============================================================
print("=" * 50)
print("실습 11: 시계열 분할 (중요!)")
print("=" * 50)

print("❌ 잘못된 방법: train_test_split (무작위 분할)")
print("   → 미래 데이터로 과거를 예측하는 꼴!")
print()

print("✅ 올바른 방법: 시간 기준 분할")

# 시간 기준 분할
split_date = '2024-05-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

print(f"\n분할 기준일: {split_date}")
print(f"Train 데이터: {len(train)}개 ({train.index.min().date()} ~ {train.index.max().date()})")
print(f"Test 데이터: {len(test)}개 ({test.index.min().date()} ~ {test.index.max().date()})")
print(f"Train 비율: {len(train) / len(df) * 100:.1f}%")
print()


# ============================================================
# 실습 12: 월별 시계열 시각화
# ============================================================
print("=" * 50)
print("실습 12: 월별 시계열 시각화")
print("=" * 50)

# 월별 집계
monthly_data = df['생산량'].resample('M').agg(['mean', 'std', 'sum'])
monthly_data.columns = ['평균', '표준편차', '합계']

print("월별 생산량 통계:")
print(monthly_data)

# 월별 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 월별 평균
monthly_data['평균'].plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('월별 평균 생산량')
axes[0].set_xlabel('월')
axes[0].set_ylabel('평균 생산량')
axes[0].tick_params(axis='x', rotation=45)

# 월별 합계
monthly_data['합계'].plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('월별 총 생산량')
axes[1].set_xlabel('월')
axes[1].set_ylabel('총 생산량')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('월별_분석.png', dpi=100)
plt.close()
print("\n월별 분석 그래프 저장: 월별_분석.png")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              시계열 데이터 기초 핵심 정리               │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 시계열 데이터                                       │
│     시간 순서로 관측된 데이터                           │
│     순서가 중요! (바꾸면 의미 없음)                     │
│                                                        │
│  ▶ 날짜 변환                                           │
│     df['날짜'] = pd.to_datetime(df['날짜'])             │
│     df = df.set_index('날짜')                          │
│                                                        │
│  ▶ 날짜 정보 추출 (dt 접근자)                          │
│     df.index.year, month, day                          │
│     df.index.dayofweek (0=월요일)                      │
│                                                        │
│  ▶ 리샘플링 (resample)                                 │
│     df['생산량'].resample('W').mean()  # 주별 평균      │
│     df['생산량'].resample('M').sum()   # 월별 합계      │
│                                                        │
│  ▶ 이동평균 (rolling)                                  │
│     df['생산량'].rolling(window=7).mean()              │
│     → 노이즈 제거, 추세 파악                           │
│                                                        │
│  ▶ 시차 변수 (shift)                                   │
│     df['생산량'].shift(1)  # 전일 값                   │
│     df['생산량'].pct_change()  # 변화율                │
│                                                        │
│  ★ 시계열 분할: 반드시 시간 기준!                       │
│     train = df[df.index < split_date]                  │
│     test = df[df.index >= split_date]                  │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 시계열 예측 모델
""")

print("=" * 60)
print("16차시 실습 완료!")
print("시간에 따른 데이터를 다루는 법을 배웠습니다!")
print("=" * 60)
