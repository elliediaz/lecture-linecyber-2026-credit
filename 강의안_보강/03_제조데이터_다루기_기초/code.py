"""
[3차시] 제조 데이터 다루기 기초 - 실습 코드 (보강판)

학습목표:
1. NumPy 배열의 개념과 기본 연산을 이해한다
2. Pandas DataFrame으로 표 형태 데이터를 다룬다
3. CSV 파일을 불러오고 제조 데이터 탐색을 수행한다
4. 실무 분석: 불량률 계산, 라인별 분석을 수행한다

데이터셋:
- seaborn의 tips 데이터셋 (레스토랑 팁 데이터)
- seaborn의 penguins 데이터셋 (펭귄 측정 데이터)
"""

import numpy as np
import pandas as pd

# seaborn에서 실제 공개 데이터셋 로드
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn not installed. Run 'pip install seaborn' for real datasets.")
    SEABORN_AVAILABLE = False

print("=" * 60)
print("[3차시] 제조 데이터 다루기 기초 - 실습 코드")
print("=" * 60)


# ============================================================
# 실습 1: NumPy 배열 생성
# ============================================================

print("\n[실습 1] NumPy 배열 생성")
print("-" * 40)

# 1-1. 리스트에서 배열 만들기
print("\n1-1. 리스트에서 배열 생성")
temperatures = np.array([82, 85, 88, 95, 84, 91, 86])
print(f"온도 데이터: {temperatures}")
print(f"데이터 타입: {temperatures.dtype}")
print(f"배열 크기: {temperatures.shape}")
print(f"요소 개수: {temperatures.size}")


# 1-2. 2차원 배열 (행렬)
print("\n1-2. 2차원 배열 생성")
sensor_matrix = np.array([
    [82, 45, 1.2, 0.3],   # 온도, 습도, 압력, 진동
    [85, 48, 1.1, 0.4],
    [88, 52, 1.3, 0.2],
    [95, 60, 1.0, 0.8]
])
print(f"센서 행렬:\n{sensor_matrix}")
print(f"형상 (shape): {sensor_matrix.shape}")  # (4, 3) = 4행 3열


# 1-3. 특수 배열 생성
print("\n1-3. 특수 배열 생성")
print(f"zeros(5): {np.zeros(5)}")
print(f"ones(5): {np.ones(5)}")
print(f"arange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"linspace(0, 1, 5): {np.linspace(0, 1, 5)}")


# ============================================================
# 실습 2: NumPy 인덱싱과 슬라이싱
# ============================================================

print("\n[실습 2] NumPy 인덱싱과 슬라이싱")
print("-" * 40)

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"원본 배열: {arr}")

# 2-1. 인덱싱
print("\n2-1. 인덱싱")
print(f"arr[0] (첫 번째): {arr[0]}")
print(f"arr[-1] (마지막): {arr[-1]}")
print(f"arr[3] (4번째): {arr[3]}")

# 2-2. 슬라이싱
print("\n2-2. 슬라이싱")
print(f"arr[1:4]: {arr[1:4]}")     # 1~3번 인덱스
print(f"arr[:3]: {arr[:3]}")       # 처음~2번
print(f"arr[5:]: {arr[5:]}")       # 5번~끝
print(f"arr[::2]: {arr[::2]}")     # 짝수 인덱스

# 2-3. 조건 인덱싱 (불리언 마스크)
print("\n2-3. 조건 인덱싱")
mask = arr > 40
print(f"arr > 40: {mask}")
print(f"arr[arr > 40]: {arr[mask]}")

# 2-4. 2차원 배열 접근
print("\n2-4. 2차원 배열 접근")
data = np.array([
    [82, 45, 1.2],
    [85, 48, 1.1],
    [88, 52, 1.3]
])
print(f"원본:\n{data}")
print(f"data[0, 0] (0행 0열): {data[0, 0]}")
print(f"data[1, :] (1행 전체): {data[1, :]}")
print(f"data[:, 0] (0열 전체): {data[:, 0]}")


# ============================================================
# 실습 3: NumPy 벡터화 연산
# ============================================================

print("\n[실습 3] NumPy 벡터화 연산")
print("-" * 40)

temps = np.array([82, 85, 88, 95, 84])
print(f"원본 온도: {temps}")

# 3-1. 스칼라 연산
print("\n3-1. 스칼라 연산")
print(f"temps + 10: {temps + 10}")
print(f"temps * 1.1: {temps * 1.1}")
print(f"temps - 80 (편차): {temps - 80}")

# 3-2. 단위 변환
print("\n3-2. 화씨 변환")
fahrenheit = temps * 9/5 + 32
print(f"화씨 온도: {np.round(fahrenheit, 1)}")

# 3-3. 배열 간 연산
print("\n3-3. 배열 간 연산")
production = np.array([1200, 1150, 1300, 1180, 1250])
defects = np.array([24, 35, 26, 42, 25])
defect_rates = defects / production
print(f"생산량: {production}")
print(f"불량수: {defects}")
print(f"불량률: {np.round(defect_rates, 4)}")

# 3-4. 비교 연산
print("\n3-4. 비교 연산")
print(f"temps > 90: {temps > 90}")
print(f"90도 초과 값: {temps[temps > 90]}")
print(f"90도 초과 개수: {np.sum(temps > 90)}")


# ============================================================
# 실습 4: NumPy 통계 함수
# ============================================================

print("\n[실습 4] NumPy 통계 함수")
print("-" * 40)

production = np.array([1200, 1150, 1300, 1180, 1250, 1320, 1100])
print(f"생산량 데이터: {production}")

# 4-1. 기본 통계
print("\n4-1. 기본 통계")
print(f"합계 (sum): {np.sum(production)}")
print(f"평균 (mean): {np.mean(production):.2f}")
print(f"중앙값 (median): {np.median(production)}")
print(f"표준편차 (std): {np.std(production):.2f}")
print(f"분산 (var): {np.var(production):.2f}")
print(f"최대값 (max): {np.max(production)}")
print(f"최소값 (min): {np.min(production)}")
print(f"범위 (ptp): {np.ptp(production)}")

# 4-2. 위치 찾기
print("\n4-2. 위치 찾기")
print(f"최대값 위치: {np.argmax(production)} (값: {production[np.argmax(production)]})")
print(f"최소값 위치: {np.argmin(production)} (값: {production[np.argmin(production)]})")

# 4-3. 2차원 배열 축별 통계
print("\n4-3. 2차원 배열 축별 통계")
data_2d = np.array([
    [100, 200, 150],  # 라인1
    [120, 180, 160],  # 라인2
    [110, 190, 140]   # 라인3
])
print(f"데이터:\n{data_2d}")
print(f"axis=0 (열별) 평균: {np.mean(data_2d, axis=0)}")
print(f"axis=1 (행별) 평균: {np.mean(data_2d, axis=1)}")

# 4-4. 조건 기반 통계
print("\n4-4. 조건 기반 통계")
temps = np.array([82, 85, 88, 95, 84, 91, 86, 93])
normal_temps = temps[temps <= 90]
print(f"정상 온도 (90도 이하): {normal_temps}")
print(f"정상 온도 평균: {np.mean(normal_temps):.1f}")


# ============================================================
# 실습 5: NumPy 조건 처리
# ============================================================

print("\n[실습 5] NumPy 조건 처리")
print("-" * 40)

temps = np.array([82, 85, 88, 95, 84, 91, 86])
print(f"온도 데이터: {temps}")

# 5-1. np.where (조건에 따른 값 변경)
print("\n5-1. np.where - 조건에 따른 값 변경")
status = np.where(temps > 90, "경고", "정상")
print(f"상태: {status}")

# 온도 조정 (90도 초과 시 5도 감소)
adjusted = np.where(temps > 90, temps - 5, temps)
print(f"조정 후: {adjusted}")

# 5-2. np.select (다중 조건)
print("\n5-2. np.select - 다중 조건")
defect_rates = np.array([0.01, 0.025, 0.035, 0.06, 0.015])
conditions = [
    defect_rates <= 0.02,
    defect_rates <= 0.04,
    defect_rates > 0.04
]
grades = ["A", "B", "C"]
result = np.select(conditions, grades)
print(f"불량률: {defect_rates}")
print(f"등급: {result}")


# ============================================================
# 실습 6: Pandas DataFrame 생성
# ============================================================

print("\n[실습 6] Pandas DataFrame 생성")
print("-" * 40)

# 6-1. 딕셔너리에서 생성
print("\n6-1. 딕셔너리에서 DataFrame 생성")
data = {
    "제품코드": ["A001", "A002", "A003", "B001", "B002"],
    "생산량": [1200, 1150, 1300, 980, 1050],
    "불량수": [24, 35, 26, 15, 22],
    "라인": [1, 2, 1, 3, 3]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print(f"\n크기: {df.shape}")
print(f"열 이름: {list(df.columns)}")

# 6-2. 날짜 범위 포함
print("\n6-2. 날짜 포함 DataFrame")
weekly_data = {
    "날짜": pd.date_range("2024-12-09", periods=7),
    "라인": [1, 2, 1, 2, 1, 2, 1],
    "생산량": [1200, 1150, 1300, 1180, 1250, 1220, 1280],
    "불량수": [24, 35, 26, 42, 25, 30, 22],
    "가동시간": [8.5, 8.0, 9.0, 7.5, 8.5, 8.0, 9.0]
}
df_week = pd.DataFrame(weekly_data)
print(df_week)


# ============================================================
# 실습 7: 실제 공개 데이터셋으로 DataFrame 탐색
# ============================================================

print("\n[실습 7] 실제 공개 데이터셋으로 DataFrame 탐색")
print("-" * 40)

# seaborn의 tips 데이터셋 로드 (실제 공개 데이터)
# tips 데이터셋: 레스토랑에서 수집된 팁 관련 데이터
if SEABORN_AVAILABLE:
    tips = sns.load_dataset('tips')
    print("\n[Tips 데이터셋 - 레스토랑 팁 데이터]")
    print("이 데이터셋은 실제 레스토랑에서 수집된 244건의 식사 기록입니다.")
    print("변수: total_bill(총액), tip(팁), sex(성별), smoker(흡연여부), day(요일), time(시간), size(인원)")
else:
    # seaborn이 없을 경우 대체 데이터 생성
    tips = pd.DataFrame({
        'total_bill': [16.99, 10.34, 21.01, 23.68, 24.59, 25.29, 8.77, 26.88, 15.04, 14.78],
        'tip': [1.01, 1.66, 3.50, 3.31, 3.61, 4.71, 2.00, 3.12, 1.96, 3.23],
        'sex': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male'],
        'smoker': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'],
        'day': ['Sun', 'Sun', 'Sun', 'Sun', 'Sun', 'Sun', 'Sun', 'Sun', 'Sun', 'Sun'],
        'time': ['Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner'],
        'size': [2, 3, 3, 2, 4, 4, 2, 4, 2, 2]
    })
    print("\n[Tips 샘플 데이터 (seaborn 미설치 - 축소 버전)]")

print("\n7-1. head() - 처음 5행")
print(tips.head())

print("\n7-2. tail() - 마지막 5행")
print(tips.tail())

print("\n7-3. info() - 데이터 정보")
tips.info()

print("\n7-4. describe() - 기본 통계")
print(tips.describe())

print("\n7-5. 데이터 타입")
print(tips.dtypes)


# ============================================================
# 실습 8: 열/행 선택
# ============================================================

print("\n[실습 8] 열/행 선택")
print("-" * 40)

# 8-1. 열 선택
print("8-1. 열 선택")
print("tips['total_bill'] (Series):")
print(tips["total_bill"].head())
print(f"타입: {type(tips['total_bill'])}")

print("\ntips[['total_bill', 'tip']] (DataFrame):")
print(tips[["total_bill", "tip"]].head())
print(f"타입: {type(tips[['total_bill', 'tip']])}")

# 8-2. loc - 라벨 기반 선택
print("\n8-2. loc - 라벨 기반 선택")
print("tips.loc[0] (0번 행):")
print(tips.loc[0])

print("\ntips.loc[0:2] (0~2번 행, 끝 포함!):")
print(tips.loc[0:2])

print("\ntips.loc[0, 'total_bill'] (0번 행의 'total_bill' 값):", tips.loc[0, "total_bill"])

# 8-3. iloc - 정수 인덱스 기반 선택
print("\n8-3. iloc - 정수 인덱스 기반 선택")
print("tips.iloc[0:2] (0~1번 행, 끝 미포함!):")
print(tips.iloc[0:2])

print("\ntips.iloc[0, 1] (0번 행, 1번 열 값):", tips.iloc[0, 1])


# ============================================================
# 실습 9: 조건 필터링
# ============================================================

print("\n[실습 9] 조건 필터링")
print("-" * 40)

print("원본 데이터 (처음 5행):")
print(tips.head())

# 9-1. 단일 조건
print("\n9-1. total_bill > 20")
high_bill = tips[tips["total_bill"] > 20]
print(f"20달러 초과 건수: {len(high_bill)}건")
print(high_bill.head())

# 9-2. 복합 조건 (AND)
print("\n9-2. total_bill > 20 AND tip > 5")
filtered = tips[(tips["total_bill"] > 20) & (tips["tip"] > 5)]
print(f"필터링 결과: {len(filtered)}건")
print(filtered.head())

# 9-3. 복합 조건 (OR)
print("\n9-3. day == 'Sat' OR day == 'Sun'")
weekend = tips[(tips["day"] == "Sat") | (tips["day"] == "Sun")]
print(f"주말 데이터: {len(weekend)}건")

# 9-4. isin 활용
print("\n9-4. isin - day가 ['Sat', 'Sun']에 포함")
weekend_v2 = tips[tips["day"].isin(["Sat", "Sun"])]
print(f"주말 데이터: {len(weekend_v2)}건")

# 9-5. 문자열 조건
print("\n9-5. time이 'Dinner'로 시작")
dinner = tips[tips["time"] == "Dinner"]
print(f"저녁 식사 건수: {len(dinner)}건")


# ============================================================
# 실습 10: 새 열 추가
# ============================================================

print("\n[실습 10] 새 열 추가")
print("-" * 40)

# tips 데이터의 복사본 생성
tips_analysis = tips.copy()

# 10-1. 팁 비율 계산
tips_analysis["tip_rate"] = tips_analysis["tip"] / tips_analysis["total_bill"]
print("10-1. 팁 비율(tip_rate) 열 추가:")
print(tips_analysis[["total_bill", "tip", "tip_rate"]].head())

# 10-2. 퍼센트로 표시
tips_analysis["tip_pct"] = (tips_analysis["tip_rate"] * 100).round(2)
print("\n10-2. 팁 비율 퍼센트(tip_pct) 열 추가:")
print(tips_analysis[["total_bill", "tip", "tip_rate", "tip_pct"]].head())

# 10-3. 1인당 금액 계산
tips_analysis["per_person"] = (tips_analysis["total_bill"] / tips_analysis["size"]).round(2)
print("\n10-3. 1인당 금액(per_person) 열 추가:")
print(tips_analysis[["total_bill", "size", "per_person"]].head())

# 10-4. 등급 분류 (apply 함수)
def classify_tip(rate):
    """팁 비율에 따른 등급 분류"""
    if rate >= 0.20:
        return "A"
    elif rate >= 0.15:
        return "B"
    else:
        return "C"

tips_analysis["grade"] = tips_analysis["tip_rate"].apply(classify_tip)
print("\n10-4. 팁 등급(grade) 열 추가:")
print(tips_analysis[["tip_rate", "tip_pct", "grade"]].head(10))


# ============================================================
# 실습 11: 그룹별 집계 (groupby)
# ============================================================

print("\n[실습 11] 그룹별 집계 (groupby)")
print("-" * 40)

# 11-1. 요일별 평균
print("11-1. 요일별 평균 total_bill")
day_avg = tips.groupby("day")["total_bill"].mean()
print(day_avg)

# 11-2. 요일별 여러 통계
print("\n11-2. 요일별 상세 통계")
day_stats = tips.groupby("day").agg({
    "total_bill": ["count", "mean", "sum"],
    "tip": ["mean", "sum"],
}).round(2)
print(day_stats)

# 11-3. 성별 집계
print("\n11-3. 성별 평균")
sex_stats = tips.groupby("sex")[["total_bill", "tip"]].mean().round(2)
print(sex_stats)

# 11-4. 최고/최저 팁 요일
print("\n11-4. 최고/최저 팁 요일")
day_tip = tips.groupby("day")["tip"].mean()
best_day = day_tip.idxmax()
worst_day = day_tip.idxmin()
print(f"최고 팁 요일: {best_day} (평균 팁: ${day_tip[best_day]:.2f})")
print(f"최저 팁 요일: {worst_day} (평균 팁: ${day_tip[worst_day]:.2f})")


# ============================================================
# 실습 12: 결측치 처리
# ============================================================

print("\n[실습 12] 결측치 처리")
print("-" * 40)

# 결측치가 있는 샘플 데이터
df_missing = pd.DataFrame({
    "온도": [85, 88, np.nan, 92, 87],
    "습도": [45, np.nan, 52, np.nan, 48],
    "압력": [101, 102, 100, 103, 101]
})
print("결측치 있는 데이터:")
print(df_missing)

# 12-1. 결측치 확인
print("\n12-1. 결측치 개수:")
print(df_missing.isnull().sum())

# 12-2. 결측치 있는 행
print("\n12-2. 결측치 있는 행:")
print(df_missing[df_missing.isnull().any(axis=1)])

# 12-3. 결측치 제거
print("\n12-3. dropna() - 결측치 있는 행 제거:")
print(df_missing.dropna())

# 12-4. 결측치 채우기
print("\n12-4. fillna(0) - 0으로 채우기:")
print(df_missing.fillna(0))

# 12-5. 평균으로 채우기
print("\n12-5. 열 평균으로 채우기:")
df_filled = df_missing.fillna(df_missing.mean())
print(df_filled)


# ============================================================
# 실습 13: 정렬
# ============================================================

print("\n[실습 13] 정렬")
print("-" * 40)

print("원본 데이터 (처음 5행):")
print(tips[["day", "time", "total_bill", "tip"]].head())

# 13-1. 단일 열 오름차순
print("\n13-1. total_bill 오름차순:")
tips_sorted = tips.sort_values("total_bill")
print(tips_sorted[["total_bill", "tip"]].head())

# 13-2. 단일 열 내림차순
print("\n13-2. total_bill 내림차순:")
tips_sorted = tips.sort_values("total_bill", ascending=False)
print(tips_sorted[["total_bill", "tip"]].head())

# 13-3. 다중 열 정렬
print("\n13-3. day 오름차순 -> total_bill 내림차순:")
tips_sorted = tips.sort_values(["day", "total_bill"], ascending=[True, False])
print(tips_sorted[["day", "total_bill", "tip"]].head(10))

# 13-4. 상위/하위 N개
print("\n13-4. total_bill 상위 5개:")
print(tips.nlargest(5, "total_bill")[["total_bill", "tip"]])

print("\n13-5. tip 하위 5개 (가장 낮은):")
print(tips.nsmallest(5, "tip")[["total_bill", "tip"]])


# ============================================================
# 실습 14: CSV 파일 저장 및 읽기
# ============================================================

print("\n[실습 14] CSV 파일 저장 및 읽기")
print("-" * 40)

# CSV 저장
output_path = "tips_analysis.csv"
tips_analysis.to_csv(output_path, index=False, encoding="utf-8")
print(f"'{output_path}' 파일 저장 완료")

# CSV 읽기
df_loaded = pd.read_csv(output_path)
print(f"\n'{output_path}' 파일 로드:")
print(df_loaded.head())

# 파일 정리
import os
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"\n'{output_path}' 파일 삭제 완료")


# ============================================================
# 실습 15: 종합 분석 - Tips 데이터셋 보고서
# ============================================================

print("\n[실습 15] 종합 분석 - Tips 데이터셋 보고서")
print("-" * 40)

# tips_analysis 데이터 사용
analysis_df = tips_analysis.copy()

# 보고서 출력
print("\n" + "=" * 60)
print("             레스토랑 팁 분석 보고서")
print("=" * 60)
print(f"분석 건수: {len(analysis_df)}건")
print("-" * 60)

print("\n[전체 현황]")
print(f"  총 매출액: ${analysis_df['total_bill'].sum():,.2f}")
print(f"  총 팁 수입: ${analysis_df['tip'].sum():,.2f}")
print(f"  평균 팁 비율: {analysis_df['tip_rate'].mean()*100:.2f}%")
print(f"  평균 테이블 인원: {analysis_df['size'].mean():.1f}명")
print(f"  평균 1인당 금액: ${analysis_df['per_person'].mean():.2f}")

print("\n[요일별 성과]")
day_summary = analysis_df.groupby("day").agg({
    "total_bill": "sum",
    "tip_rate": "mean",
    "size": "mean"
}).round(3)
for day in analysis_df["day"].unique():
    day_data = day_summary.loc[day]
    print(f"  {day}: 매출 ${day_data['total_bill']:,.2f}, "
          f"팁비율 {day_data['tip_rate']*100:.2f}%, "
          f"평균인원 {day_data['size']:.1f}명")

print("\n[등급별 분포]")
grade_counts = analysis_df["grade"].value_counts().sort_index()
for grade, count in grade_counts.items():
    pct = count / len(analysis_df) * 100
    print(f"  {grade}등급: {count}건 ({pct:.1f}%)")

print("\n[고액 결제 (상위 5%)]")
threshold = analysis_df["total_bill"].quantile(0.95)
high_spenders = analysis_df[analysis_df["total_bill"] >= threshold]
print(f"  기준: ${threshold:.2f} 이상")
print(f"  건수: {len(high_spenders)}건")
print(f"  평균 팁 비율: {high_spenders['tip_rate'].mean()*100:.2f}%")

print("\n[최고/최저 팁 요일]")
day_tip = analysis_df.groupby("day")["tip_rate"].mean()
best = day_tip.idxmax()
worst = day_tip.idxmin()
print(f"  최고: {best} (팁비율 {day_tip[best]*100:.2f}%)")
print(f"  최저: {worst} (팁비율 {day_tip[worst]*100:.2f}%)")

print("=" * 60)


# ============================================================
# 핵심 요약
# ============================================================

print("\n" + "=" * 60)
print("[3차시 핵심 요약]")
print("=" * 60)

summary = """
1. NumPy 기본
   - np.array(): 배열 생성
   - 벡터화 연산: arr + 10, arr * 2
   - 통계: mean, std, max, min, sum
   - 조건 필터링: arr[arr > 값]

2. Pandas DataFrame
   - pd.DataFrame(dict): 테이블 생성
   - pd.read_csv(): CSV 파일 읽기
   - df.to_csv(): CSV 파일 저장

3. 실제 공개 데이터셋 활용
   - seaborn: sns.load_dataset('tips'), sns.load_dataset('penguins')
   - sklearn: load_iris(), load_wine(), load_diabetes()

4. 데이터 탐색
   - head(), tail(): 앞/뒤 데이터
   - info(): 데이터 타입, 결측치
   - describe(): 기술 통계

5. 열/행 선택
   - df['열']: Series 반환
   - df[['열1', '열2']]: DataFrame 반환
   - df.loc[]: 라벨 기반 (끝 포함)
   - df.iloc[]: 정수 기반 (끝 미포함)

6. 조건 필터링
   - df[조건]: 조건에 맞는 행
   - (조건1) & (조건2): AND 조건
   - (조건1) | (조건2): OR 조건

7. 그룹 집계
   - df.groupby('열'): 그룹 생성
   - .agg({'열': ['함수1', '함수2']}): 다중 집계
"""

print(summary)

print("=" * 60)
print("다음 차시: 공개 데이터셋 확보 및 데이터 생태계 이해")
print("=" * 60)
