"""
[2차시] Python 환경 구축과 기초 - 실습 코드

이 파일은 Jupyter Notebook에서 셀 단위로 실행하거나,
전체를 한 번에 실행할 수 있습니다.

학습목표:
- Python 기본 자료형 (int, float, str, list, dict) 이해
- 조건문 (if/elif/else) 활용
- 반복문 (for, while) 활용
"""

# =============================================================
# 1. 기본 자료형 - 정수와 실수
# =============================================================

print("=" * 50)
print("1. 정수(int)와 실수(float)")
print("=" * 50)

# 정수 (int): 소수점 없는 숫자
production_count = 1000
line_number = 3

# 실수 (float): 소수점 있는 숫자
defect_rate = 0.025
temperature = 85.7

print(f"생산량: {production_count}개")
print(f"라인 번호: {line_number}번")
print(f"불량률: {defect_rate} ({defect_rate * 100}%)")
print(f"온도: {temperature}°C")

# 사칙연산
print("\n[사칙연산 예시]")
print(f"100 + 50 = {100 + 50}")      # 덧셈
print(f"100 - 30 = {100 - 30}")      # 뺄셈
print(f"100 * 3 = {100 * 3}")        # 곱셈
print(f"100 / 3 = {100 / 3:.2f}")    # 나눗셈 (결과는 항상 float)
print(f"100 // 3 = {100 // 3}")      # 정수 나눗셈
print(f"100 % 3 = {100 % 3}")        # 나머지
print(f"2 ** 10 = {2 ** 10}")        # 거듭제곱
print()


# =============================================================
# 2. 기본 자료형 - 문자열
# =============================================================

print("=" * 50)
print("2. 문자열(str)")
print("=" * 50)

# 문자열 생성
product_name = "반도체 센서 모듈"
product_code = 'SM-001'

print(f"제품명: {product_name}")
print(f"제품코드: {product_code}")

# 문자열 연결
line = "라인"
number = "01"
full_name = line + number
print(f"라인명: {full_name}")

# f-string 활용 (Python 3.6+)
line_num = 3
rate = 0.025
message = f"라인 {line_num}번 불량률: {rate:.2%}"
print(message)

# 유용한 문자열 메서드
sample_text = "  Hello, Python!  "
print(f"\n[문자열 메서드]")
print(f"원본: '{sample_text}'")
print(f"strip(): '{sample_text.strip()}'")        # 양쪽 공백 제거
print(f"upper(): '{sample_text.upper()}'")        # 대문자 변환
print(f"lower(): '{sample_text.lower()}'")        # 소문자 변환
print(f"replace(): '{sample_text.replace('Python', 'AI')}'")
print()


# =============================================================
# 3. 기본 자료형 - 리스트
# =============================================================

print("=" * 50)
print("3. 리스트(list)")
print("=" * 50)

# 리스트 생성
daily_production = [1200, 1150, 1300, 1180, 1250]
print(f"5일간 생산량: {daily_production}")

# 인덱싱 (0부터 시작!)
print(f"\n[인덱싱]")
print(f"첫 번째 (인덱스 0): {daily_production[0]}")
print(f"두 번째 (인덱스 1): {daily_production[1]}")
print(f"마지막 (인덱스 -1): {daily_production[-1]}")

# 슬라이싱
print(f"\n[슬라이싱]")
print(f"처음 3개 [0:3]: {daily_production[0:3]}")
print(f"2번째부터 끝까지 [1:]: {daily_production[1:]}")
print(f"처음부터 3번째까지 [:3]: {daily_production[:3]}")

# 리스트 수정
print(f"\n[리스트 수정]")
daily_production.append(1280)  # 끝에 추가
print(f"append(1280) 후: {daily_production}")

daily_production[0] = 1210  # 값 변경
print(f"인덱스 0 값 변경 후: {daily_production}")

# 리스트 내장 함수
print(f"\n[리스트 통계]")
print(f"합계 sum(): {sum(daily_production)}")
print(f"길이 len(): {len(daily_production)}")
print(f"최대값 max(): {max(daily_production)}")
print(f"최소값 min(): {min(daily_production)}")
print()


# =============================================================
# 4. 기본 자료형 - 딕셔너리
# =============================================================

print("=" * 50)
print("4. 딕셔너리(dict)")
print("=" * 50)

# 딕셔너리 생성
sensor_data = {
    "온도": 85.2,
    "습도": 45,
    "압력": 1.2,
    "상태": "정상"
}
print(f"센서 데이터: {sensor_data}")

# 값 접근
print(f"\n[값 접근]")
print(f"온도: {sensor_data['온도']}°C")
print(f"상태: {sensor_data['상태']}")

# get() 메서드 (키가 없을 때 안전)
print(f"진동 (없는 키): {sensor_data.get('진동', '측정안함')}")

# 딕셔너리 수정
print(f"\n[딕셔너리 수정]")
sensor_data["진동"] = 0.02  # 새 키 추가
sensor_data["온도"] = 86.0  # 값 수정
print(f"수정 후: {sensor_data}")

# 키와 값 순회
print(f"\n[딕셔너리 순회]")
for key, value in sensor_data.items():
    print(f"  {key}: {value}")
print()


# =============================================================
# 5. 조건문 (if/elif/else)
# =============================================================

print("=" * 50)
print("5. 조건문 (if/elif/else)")
print("=" * 50)

# 기본 조건문
defect_rate = 0.03

print(f"불량률: {defect_rate * 100}%")
print("판정 결과:")

if defect_rate > 0.05:
    print("  → 경고: 불량률이 높습니다! 라인 점검 필요")
elif defect_rate > 0.02:
    print("  → 주의: 불량률 모니터링이 필요합니다")
else:
    print("  → 정상: 품질 기준을 충족합니다")

# 비교 연산자 예시
print(f"\n[비교 연산자]")
a, b = 10, 20
print(f"a = {a}, b = {b}")
print(f"a == b: {a == b}")   # 같다
print(f"a != b: {a != b}")   # 다르다
print(f"a > b: {a > b}")     # 크다
print(f"a < b: {a < b}")     # 작다
print(f"a >= 10: {a >= 10}") # 크거나 같다
print(f"a <= 10: {a <= 10}") # 작거나 같다

# 논리 연산자 예시
print(f"\n[논리 연산자]")
temp = 85
humidity = 55

print(f"온도: {temp}°C, 습도: {humidity}%")

if temp > 80 and humidity > 50:
    print("  → 온도와 습도 모두 높음 (and)")

if temp > 90 or humidity > 70:
    print("  → 온도 또는 습도가 위험 수준 (or)")
else:
    print("  → 온도/습도 위험 수준 아님")

if not (temp > 90):
    print("  → 온도는 90도 미만 (not)")
print()


# =============================================================
# 6. 반복문 (for)
# =============================================================

print("=" * 50)
print("6. 반복문 (for)")
print("=" * 50)

# 리스트 순회
print("[리스트 순회]")
production = [1200, 1150, 1300]
for daily in production:
    print(f"  생산량: {daily}개")

# range() 활용
print(f"\n[range() 활용]")
for i in range(5):
    print(f"  라인 {i + 1} 점검 중...")

# enumerate() 활용 - 인덱스와 값 동시 접근
print(f"\n[enumerate() 활용]")
products = ["센서A", "센서B", "모듈C"]
for idx, name in enumerate(products):
    print(f"  {idx}: {name}")

# 조건과 함께 사용
print(f"\n[조건과 함께 사용 - 이상 온도 탐지]")
temperatures = [82, 85, 88, 95, 84, 91, 86]
threshold = 90

for i, temp in enumerate(temperatures):
    if temp > threshold:
        print(f"  [경고] {i + 1}번째 측정: {temp}°C (기준 초과!)")
    else:
        print(f"  [정상] {i + 1}번째 측정: {temp}°C")
print()


# =============================================================
# 7. 반복문 (while)
# =============================================================

print("=" * 50)
print("7. 반복문 (while)")
print("=" * 50)

# 기본 while 반복문
print("[카운트다운]")
count = 5
while count > 0:
    print(f"  {count}...")
    count -= 1
print("  발사!")

# 조건 기반 반복
print(f"\n[품질 검사 시뮬레이션]")
import random
random.seed(42)  # 재현성을 위해 시드 고정

defect_count = 0
check_count = 0
max_defects = 3

while defect_count < max_defects:
    check_count += 1
    # 10% 확률로 불량 발생
    if random.random() < 0.1:
        defect_count += 1
        print(f"  검사 #{check_count}: 불량 발견! (누적: {defect_count}개)")

print(f"  → 총 {check_count}개 검사 후 {max_defects}개 불량 발견")
print()


# =============================================================
# 8. 실습 예제: 품질 등급 판정 시스템
# =============================================================

print("=" * 50)
print("8. 실습 예제: 품질 등급 판정 시스템")
print("=" * 50)


def determine_quality_grade(defect_rate: float) -> tuple:
    """
    불량률에 따른 품질 등급 판정

    Args:
        defect_rate: 불량률 (0.0 ~ 1.0)

    Returns:
        (등급, 설명) 튜플
    """
    if defect_rate <= 0.01:
        return ("A", "우수 - 출하 가능")
    elif defect_rate <= 0.03:
        return ("B", "양호 - 출하 가능")
    elif defect_rate <= 0.05:
        return ("C", "주의 - 재검사 필요")
    else:
        return ("D", "불량 - 출하 불가")


# 여러 라인의 품질 판정
print("[5개 라인 품질 판정]")
line_data = {
    "라인1": 0.008,
    "라인2": 0.025,
    "라인3": 0.045,
    "라인4": 0.012,
    "라인5": 0.062
}

for line_name, rate in line_data.items():
    grade, description = determine_quality_grade(rate)
    print(f"  {line_name}: 불량률 {rate:.1%} → 등급 {grade} ({description})")
print()


# =============================================================
# 9. 실습 예제: 일일 생산량 분석
# =============================================================

print("=" * 50)
print("9. 실습 예제: 일일 생산량 분석")
print("=" * 50)

# 10일간 생산량 데이터
production_data = [1200, 1150, 1300, 1180, 1250, 1320, 1100, 1280, 1190, 1260]
days = ["월", "화", "수", "목", "금", "토", "일", "월", "화", "수"]

# 기본 통계 계산
total = sum(production_data)
average = total / len(production_data)
max_value = max(production_data)
min_value = min(production_data)

print(f"[기본 통계]")
print(f"  총 생산량: {total:,}개")
print(f"  일평균: {average:,.1f}개")
print(f"  최대: {max_value:,}개")
print(f"  최소: {min_value:,}개")

# 목표 대비 분석
target = 1200
print(f"\n[목표({target}개) 대비 분석]")
above_target = 0
below_target = 0

for i, (day, prod) in enumerate(zip(days, production_data)):
    if prod >= target:
        status = "달성"
        above_target += 1
    else:
        status = "미달"
        below_target += 1
    diff = prod - target
    print(f"  {i + 1}일차({day}): {prod:,}개 [{status}] ({diff:+d})")

print(f"\n[요약]")
print(f"  목표 달성: {above_target}일")
print(f"  목표 미달: {below_target}일")
print(f"  달성률: {above_target / len(production_data) * 100:.1f}%")
print()


# =============================================================
# 10. 타입 변환 (Type Casting)
# =============================================================

print("=" * 50)
print("10. 타입 변환 예시")
print("=" * 50)

# 문자열 → 숫자
user_input = "1000"
count = int(user_input)
print(f"문자열 '{user_input}' → 정수 {count} (type: {type(count).__name__})")

rate_str = "0.025"
rate = float(rate_str)
print(f"문자열 '{rate_str}' → 실수 {rate} (type: {type(rate).__name__})")

# 숫자 → 문자열
number = 1200
text = str(number)
print(f"정수 {number} → 문자열 '{text}' (type: {type(text).__name__})")

# range → list
numbers = list(range(5))
print(f"range(5) → 리스트 {numbers}")

print()
print("=" * 50)
print("2차시 실습 완료!")
print("=" * 50)
