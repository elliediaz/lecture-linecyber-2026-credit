"""
[3차시] 데이터 다루기 기초 - 실습 코드

NumPy와 Pandas의 기본 사용법을 실습합니다.

학습목표:
- NumPy 배열 생성과 연산
- Pandas DataFrame 기본 조작
- CSV 파일 읽기/쓰기 및 데이터 탐색
"""

import numpy as np
import pandas as pd

# =============================================================
# 1. NumPy 배열 생성
# =============================================================

print("=" * 60)
print("1. NumPy 배열 생성")
print("=" * 60)

# 리스트에서 배열 만들기
temperatures = np.array([82, 85, 88, 95, 84, 91, 86])
print(f"1차원 배열: {temperatures}")
print(f"데이터 타입: {temperatures.dtype}")
print(f"배열 크기: {temperatures.shape}")

# 2차원 배열 (행렬)
sensor_data = np.array([
    [82, 45, 1.2],   # 온도, 습도, 압력
    [85, 48, 1.1],
    [88, 52, 1.3],
    [95, 60, 1.0]
])
print(f"\n2차원 배열:\n{sensor_data}")
print(f"배열 크기: {sensor_data.shape}")  # (4, 3) = 4행 3열

# 특수 배열 생성
print("\n[특수 배열 생성]")
print(f"zeros(3): {np.zeros(3)}")
print(f"ones(3): {np.ones(3)}")
print(f"arange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"linspace(0, 1, 5): {np.linspace(0, 1, 5)}")
print()


# =============================================================
# 2. NumPy 인덱싱과 슬라이싱
# =============================================================

print("=" * 60)
print("2. NumPy 인덱싱과 슬라이싱")
print("=" * 60)

arr = np.array([10, 20, 30, 40, 50])
print(f"원본 배열: {arr}")

# 인덱싱
print(f"\n[인덱싱]")
print(f"arr[0]: {arr[0]}")     # 첫 번째
print(f"arr[-1]: {arr[-1]}")   # 마지막
print(f"arr[2]: {arr[2]}")     # 세 번째

# 슬라이싱
print(f"\n[슬라이싱]")
print(f"arr[1:4]: {arr[1:4]}")   # 1~3번 인덱스
print(f"arr[:3]: {arr[:3]}")     # 처음~2번
print(f"arr[2:]: {arr[2:]}")     # 2번~끝

# 조건 인덱싱 (불리언 마스크)
print(f"\n[조건 인덱싱]")
mask = arr > 25
print(f"arr > 25: {mask}")
print(f"arr[arr > 25]: {arr[mask]}")

# 2차원 배열 접근
print(f"\n[2차원 배열 접근]")
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"원본:\n{data}")
print(f"data[0, 2]: {data[0, 2]}")     # 0행 2열
print(f"data[1, :]: {data[1, :]}")     # 1행 전체
print(f"data[:, 0]: {data[:, 0]}")     # 0열 전체
print()


# =============================================================
# 3. NumPy 연산
# =============================================================

print("=" * 60)
print("3. NumPy 연산")
print("=" * 60)

temps = np.array([82, 85, 88, 95, 84])
print(f"원본 배열: {temps}")

# 스칼라 연산 (벡터화)
print(f"\n[스칼라 연산]")
print(f"temps + 10: {temps + 10}")
print(f"temps * 2: {temps * 2}")
print(f"temps - 80: {temps - 80}")

# 화씨 → 섭씨 변환
temps_celsius = (temps - 32) * 5/9
print(f"섭씨 변환: {np.round(temps_celsius, 1)}")

# 배열 간 연산
print(f"\n[배열 간 연산]")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"a: {a}, b: {b}")
print(f"a + b: {a + b}")
print(f"a * b: {a * b}")

# 비교 연산
print(f"\n[비교 연산]")
print(f"temps > 90: {temps > 90}")
print(f"90도 초과 값: {temps[temps > 90]}")
print()


# =============================================================
# 4. NumPy 통계 함수
# =============================================================

print("=" * 60)
print("4. NumPy 통계 함수")
print("=" * 60)

production = np.array([1200, 1150, 1300, 1180, 1250, 1320, 1100])
print(f"생산량 데이터: {production}")

print(f"\n[기본 통계]")
print(f"합계 (sum): {np.sum(production)}")
print(f"평균 (mean): {np.mean(production):.2f}")
print(f"표준편차 (std): {np.std(production):.2f}")
print(f"분산 (var): {np.var(production):.2f}")
print(f"최대값 (max): {np.max(production)}")
print(f"최소값 (min): {np.min(production)}")
print(f"중앙값 (median): {np.median(production)}")

# 2차원 배열 축별 통계
print(f"\n[2차원 배열 축별 통계]")
data_2d = np.array([
    [100, 200, 150],  # 라인1
    [120, 180, 160],  # 라인2
    [110, 190, 140]   # 라인3
])
print(f"데이터:\n{data_2d}")
print(f"axis=0 (열별) 평균: {np.mean(data_2d, axis=0)}")
print(f"axis=1 (행별) 평균: {np.mean(data_2d, axis=1)}")
print()


# =============================================================
# 5. Pandas DataFrame 생성
# =============================================================

print("=" * 60)
print("5. Pandas DataFrame 생성")
print("=" * 60)

# 딕셔너리에서 생성
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
print()


# =============================================================
# 6. DataFrame 기본 탐색
# =============================================================

print("=" * 60)
print("6. DataFrame 기본 탐색")
print("=" * 60)

# 더 큰 샘플 데이터 생성
np.random.seed(42)
n = 100

df_large = pd.DataFrame({
    "날짜": pd.date_range("2024-01-01", periods=n),
    "제품코드": np.random.choice(["A001", "A002", "B001", "B002", "C001"], n),
    "라인": np.random.choice([1, 2, 3], n),
    "생산량": np.random.randint(900, 1400, n),
    "불량수": np.random.randint(5, 50, n)
})

print("[head() - 처음 5행]")
print(df_large.head())

print("\n[tail() - 마지막 5행]")
print(df_large.tail())

print("\n[info() - 데이터 정보]")
print(df_large.info())

print("\n[describe() - 기본 통계]")
print(df_large.describe())

print("\n[데이터 타입]")
print(df_large.dtypes)
print()


# =============================================================
# 7. 열/행 선택
# =============================================================

print("=" * 60)
print("7. 열/행 선택")
print("=" * 60)

# 열 선택
print("[열 선택]")
print("df['생산량'] (Series):")
print(df["생산량"])

print("\ndf[['제품코드', '생산량']] (DataFrame):")
print(df[["제품코드", "생산량"]])

# 행 선택 - loc (라벨 기반)
print("\n[loc - 라벨 기반 선택]")
print("df.loc[0] (0번 행):")
print(df.loc[0])

print("\ndf.loc[0:2] (0~2번 행, 끝 포함!):")
print(df.loc[0:2])

# 행 선택 - iloc (정수 인덱스 기반)
print("\n[iloc - 정수 인덱스 기반 선택]")
print("df.iloc[0:2] (0~1번 행, 끝 미포함!):")
print(df.iloc[0:2])

print("\ndf.iloc[0, 1] (0행 1열 값):", df.iloc[0, 1])
print()


# =============================================================
# 8. 조건 필터링
# =============================================================

print("=" * 60)
print("8. 조건 필터링")
print("=" * 60)

print("원본 데이터:")
print(df)

# 단일 조건
print("\n[생산량 > 1100]")
high_prod = df[df["생산량"] > 1100]
print(high_prod)

# 복합 조건
print("\n[생산량 > 1100 AND 라인 == 1]")
filtered = df[(df["생산량"] > 1100) & (df["라인"] == 1)]
print(filtered)

# isin 활용
print("\n[라인이 1 또는 2인 경우]")
line_12 = df[df["라인"].isin([1, 2])]
print(line_12)

# 문자열 조건
print("\n[제품코드가 'A'로 시작하는 경우]")
a_products = df[df["제품코드"].str.startswith("A")]
print(a_products)
print()


# =============================================================
# 9. 새 열 추가와 수정
# =============================================================

print("=" * 60)
print("9. 새 열 추가와 수정")
print("=" * 60)

# 불량률 계산
df["불량률"] = df["불량수"] / df["생산량"]
print("불량률 열 추가 후:")
print(df)

# 등급 분류
def classify_grade(rate):
    if rate < 0.02:
        return "A"
    elif rate < 0.03:
        return "B"
    else:
        return "C"

df["등급"] = df["불량률"].apply(classify_grade)
print("\n등급 열 추가 후:")
print(df)

# 여러 열 조합
df["양품수"] = df["생산량"] - df["불량수"]
print("\n양품수 열 추가 후:")
print(df[["제품코드", "생산량", "불량수", "양품수"]])
print()


# =============================================================
# 10. 그룹별 집계 (groupby)
# =============================================================

print("=" * 60)
print("10. 그룹별 집계 (groupby)")
print("=" * 60)

# 라인별 평균
print("[라인별 평균 생산량]")
line_avg = df.groupby("라인")["생산량"].mean()
print(line_avg)

# 라인별 여러 통계
print("\n[라인별 상세 통계]")
line_stats = df.groupby("라인").agg({
    "생산량": ["count", "mean", "sum"],
    "불량수": ["mean", "sum"],
    "불량률": "mean"
})
print(line_stats)

# 등급별 집계
print("\n[등급별 평균]")
grade_stats = df.groupby("등급")[["생산량", "불량률"]].mean()
print(grade_stats)
print()


# =============================================================
# 11. 결측치 처리
# =============================================================

print("=" * 60)
print("11. 결측치 처리")
print("=" * 60)

# 결측치가 있는 샘플 데이터
df_missing = pd.DataFrame({
    "A": [1, 2, np.nan, 4],
    "B": [5, np.nan, np.nan, 8],
    "C": [9, 10, 11, 12]
})
print("결측치 있는 데이터:")
print(df_missing)

# 결측치 확인
print("\n[결측치 개수]")
print(df_missing.isnull().sum())

# 결측치 있는 행 확인
print("\n[결측치 있는 행]")
print(df_missing[df_missing.isnull().any(axis=1)])

# 결측치 제거
print("\n[dropna() - 결측치 있는 행 제거]")
print(df_missing.dropna())

# 결측치 채우기
print("\n[fillna(0) - 0으로 채우기]")
print(df_missing.fillna(0))

# 평균으로 채우기
print("\n[열 평균으로 채우기]")
df_filled = df_missing.fillna(df_missing.mean())
print(df_filled)
print()


# =============================================================
# 12. 정렬
# =============================================================

print("=" * 60)
print("12. 정렬")
print("=" * 60)

print("원본 데이터:")
print(df[["제품코드", "생산량", "불량률"]])

# 단일 열 정렬
print("\n[생산량 오름차순]")
df_sorted = df.sort_values("생산량")
print(df_sorted[["제품코드", "생산량"]])

print("\n[생산량 내림차순]")
df_sorted = df.sort_values("생산량", ascending=False)
print(df_sorted[["제품코드", "생산량"]])

# 다중 열 정렬
print("\n[라인 오름차순, 생산량 내림차순]")
df_sorted = df.sort_values(["라인", "생산량"], ascending=[True, False])
print(df_sorted[["제품코드", "라인", "생산량"]])
print()


# =============================================================
# 13. CSV 파일 저장 및 읽기 (시뮬레이션)
# =============================================================

print("=" * 60)
print("13. CSV 파일 저장 및 읽기")
print("=" * 60)

# CSV 저장
output_path = "sample_output.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"'{output_path}' 파일 저장 완료")

# CSV 읽기
df_loaded = pd.read_csv(output_path)
print(f"\n'{output_path}' 파일 로드 완료:")
print(df_loaded.head())

# 파일 정리 (실습 후 삭제)
import os
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"\n'{output_path}' 파일 삭제 완료")
print()


# =============================================================
# 14. 종합 실습: 제조 데이터 분석
# =============================================================

print("=" * 60)
print("14. 종합 실습: 제조 데이터 분석")
print("=" * 60)

# 샘플 제조 데이터 생성
np.random.seed(42)
n = 30

production_df = pd.DataFrame({
    "날짜": pd.date_range("2024-01-01", periods=n),
    "라인": np.random.choice([1, 2, 3], n),
    "생산량": np.random.randint(1000, 1400, n),
    "불량수": np.random.randint(10, 60, n),
    "가동시간": np.random.uniform(7, 10, n).round(1)
})

# 파생 변수 추가
production_df["불량률"] = production_df["불량수"] / production_df["생산량"]
production_df["시간당생산량"] = (production_df["생산량"] / production_df["가동시간"]).round(1)

print("=== 제조 데이터 샘플 ===")
print(production_df.head(10))

print("\n=== 기본 통계 ===")
print(production_df.describe())

print("\n=== 라인별 분석 ===")
line_analysis = production_df.groupby("라인").agg({
    "생산량": ["mean", "sum"],
    "불량률": "mean",
    "시간당생산량": "mean"
}).round(2)
print(line_analysis)

print("\n=== 이상 데이터 탐지 (불량률 5% 초과) ===")
abnormal = production_df[production_df["불량률"] > 0.05]
print(f"이상 데이터 수: {len(abnormal)}건")
print(abnormal[["날짜", "라인", "생산량", "불량수", "불량률"]])

print("\n=== 요약 ===")
print(f"총 생산량: {production_df['생산량'].sum():,}개")
print(f"평균 불량률: {production_df['불량률'].mean():.2%}")
print(f"최고 효율 라인: {production_df.groupby('라인')['시간당생산량'].mean().idxmax()}번")

print()
print("=" * 60)
print("3차시 실습 완료!")
print("=" * 60)
