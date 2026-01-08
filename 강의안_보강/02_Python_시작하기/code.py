"""
[2차시] Python 시작하기 - 실습 코드 (보강판)

학습목표:
1. Anaconda와 Jupyter Notebook을 설치하고 실행한다
2. Python 기본 데이터 종류(숫자, 문자, 목록 등)를 이해한다
3. 조건문과 반복문으로 간단한 프로그램을 작성한다
4. 제조 현장 데이터를 Python으로 다루는 기초를 익힌다
"""

# ============================================================
# 실습 환경 확인
# ============================================================

print("=" * 60)
print("[2차시] Python 시작하기 - 실습 코드")
print("=" * 60)


# ============================================================
# 실습 1: 변수와 기본 자료형
# ============================================================

print("\n[실습 1] 변수와 기본 자료형")
print("-" * 40)

# 1-1. 정수 (int)
print("\n1-1. 정수 (int)")
daily_production = 1200      # 일 생산량
defect_count = 30            # 불량 개수
line_number = 3              # 라인 번호

print(f"일 생산량: {daily_production}개")
print(f"불량 개수: {defect_count}개")
print(f"라인 번호: {line_number}번")

# 정수 연산
total_week = daily_production * 5
remaining = 1000 - 750
print(f"5일 생산량: {total_week}개")
print(f"남은 목표: {remaining}개")


# 1-2. 실수 (float)
print("\n1-2. 실수 (float)")
temperature = 85.7           # 온도
defect_rate = 0.025          # 불량률 2.5%
pressure = 101.325           # 압력

print(f"온도: {temperature}°C")
print(f"불량률: {defect_rate} ({defect_rate * 100}%)")
print(f"압력: {pressure} kPa")

# 실수 계산
good_count = daily_production * (1 - defect_rate)
print(f"양품 수: {good_count:.0f}개")


# 1-3. 문자열 (str)
print("\n1-3. 문자열 (str)")
product_name = "센서 모듈 A"
line_id = "LINE-01"
status = "정상"

print(f"제품명: {product_name}")
print(f"라인 ID: {line_id}")
print(f"상태: {status}")

# 문자열 연결
full_id = line_id + "-" + status
print(f"전체 ID: {full_id}")

# 문자열 메서드
print(f"대문자: {product_name.upper()}")
print(f"소문자: {'HELLO'.lower()}")
print(f"교체: {'A-B-C'.replace('-', '_')}")


# 1-4. 리스트 (list)
print("\n1-4. 리스트 (list)")
weekly_production = [1200, 1150, 1300, 1180, 1250]
print(f"주간 생산량: {weekly_production}")

# 인덱스 접근
print(f"첫째 날: {weekly_production[0]}개")
print(f"마지막 날: {weekly_production[-1]}개")
print(f"처음 3일: {weekly_production[0:3]}")

# 리스트 메서드
weekly_production.append(1280)
print(f"추가 후: {weekly_production}")

# 기본 통계
print(f"합계: {sum(weekly_production)}개")
print(f"평균: {sum(weekly_production)/len(weekly_production):.0f}개")
print(f"최대: {max(weekly_production)}개")
print(f"최소: {min(weekly_production)}개")


# 1-5. 딕셔너리 (dict)
print("\n1-5. 딕셔너리 (dict)")
sensor_data = {
    "온도": 85.2,
    "압력": 101.3,
    "습도": 45,
    "상태": "정상"
}
print(f"센서 데이터: {sensor_data}")

# 값 접근
print(f"온도: {sensor_data['온도']}°C")
print(f"상태: {sensor_data['상태']}")

# 값 추가/수정
sensor_data["진동"] = 0.42
sensor_data["온도"] = 86.0
print(f"수정 후: {sensor_data}")

# 안전한 접근
speed = sensor_data.get("속도", 0)
print(f"속도 (없으면 0): {speed}")


# ============================================================
# 실습 2: f-string 포맷팅
# ============================================================

print("\n[실습 2] f-string 포맷팅")
print("-" * 40)

value = 1234.5678
rate = 0.0253
big_num = 1234567

# 소수점 자릿수
print(f"소수점 2자리: {value:.2f}")
print(f"소수점 없음: {value:.0f}")

# 퍼센트 표시
print(f"퍼센트 1자리: {rate:.1%}")
print(f"퍼센트 2자리: {rate:.2%}")

# 천 단위 콤마
print(f"콤마 표시: {big_num:,}")

# 정렬
name = "온도"
print(f"왼쪽 정렬: [{name:<10}]")
print(f"오른쪽 정렬: [{name:>10}]")
print(f"가운데 정렬: [{name:^10}]")


# ============================================================
# 실습 3: 조건문 (if)
# ============================================================

print("\n[실습 3] 조건문 (if)")
print("-" * 40)

# 3-1. 기본 조건문
print("\n3-1. 기본 조건문")
temperature = 92

if temperature > 90:
    print(f"온도 {temperature}도: 경고!")
elif temperature > 80:
    print(f"온도 {temperature}도: 정상")
else:
    print(f"온도 {temperature}도: 낮음")


# 3-2. 품질 등급 판정
print("\n3-2. 품질 등급 판정")
defect_rate = 0.025  # 2.5%

if defect_rate <= 0.01:
    grade = "A"
    message = "우수"
    action = "현 상태 유지"
elif defect_rate <= 0.03:
    grade = "B"
    message = "양호"
    action = "모니터링 강화"
elif defect_rate <= 0.05:
    grade = "C"
    message = "주의"
    action = "원인 분석 필요"
else:
    grade = "D"
    message = "개선 필요"
    action = "즉시 대책 수립"

print(f"불량률: {defect_rate:.1%}")
print(f"등급: {grade} ({message})")
print(f"조치: {action}")


# 3-3. 복합 조건
print("\n3-3. 복합 조건")
temp = 88
pressure = 95
vibration = 0.5

# AND 조건
if temp > 85 and pressure < 100:
    print("조건1 충족: 온도 높고 압력 정상")

# OR 조건
if temp > 90 or vibration > 0.8:
    print("조건2 충족: 온도 또는 진동 이상")
else:
    print("조건2 미충족: 정상 범위")

# 복합 조건
if (temp > 85 and temp < 95) and vibration < 0.6:
    print("조건3 충족: 안정 운영 구간")


# ============================================================
# 실습 4: 반복문 (for)
# ============================================================

print("\n[실습 4] 반복문 (for)")
print("-" * 40)

# 4-1. 기본 반복
print("\n4-1. 기본 반복")
temperatures = [82, 85, 88, 95, 84]

for temp in temperatures:
    print(f"온도: {temp}도")


# 4-2. enumerate 사용
print("\n4-2. enumerate 사용")
for i, temp in enumerate(temperatures):
    print(f"{i+1}번째 측정: {temp}도")


# 4-3. range 사용
print("\n4-3. range 사용")
print("0부터 4까지:")
for i in range(5):
    print(f"  {i}")

print("1부터 5까지:")
for i in range(1, 6):
    print(f"  {i}")


# 4-4. 조건과 함께 사용
print("\n4-4. 조건과 함께 사용")
temperatures = [82, 85, 88, 95, 84, 91, 86, 83]
threshold = 90

print("=== 온도 모니터링 ===")
warning_count = 0

for i, temp in enumerate(temperatures):
    hour = 9 + i
    if temp > threshold:
        print(f"[경고] {hour}시: {temp}도 - 기준 초과!")
        warning_count += 1
    else:
        print(f"[정상] {hour}시: {temp}도")

print(f"\n총 경고 횟수: {warning_count}회")


# 4-5. 리스트 컴프리헨션
print("\n4-5. 리스트 컴프리헨션")

# 제곱 계산
squares = [x ** 2 for x in range(5)]
print(f"제곱: {squares}")

# 조건 필터링
temps = [82, 85, 88, 95, 84, 91, 86]
high_temps = [t for t in temps if t > 90]
print(f"90도 초과: {high_temps}")

# 변환 + 조건
fahrenheit = [t * 9/5 + 32 for t in temps if t > 85]
print(f"85도 초과 화씨 변환: {fahrenheit}")


# ============================================================
# 실습 5: 딕셔너리 활용
# ============================================================

print("\n[실습 5] 딕셔너리 활용")
print("-" * 40)

# 5-1. 중첩 딕셔너리
print("\n5-1. 중첩 딕셔너리 (설비 관리)")
equipment = {
    "EQ-001": {"온도": 82, "진동": 0.3, "압력": 100, "상태": "정상"},
    "EQ-002": {"온도": 91, "진동": 0.8, "압력": 95,  "상태": "주의"},
    "EQ-003": {"온도": 85, "진동": 0.4, "압력": 102, "상태": "정상"},
    "EQ-004": {"온도": 96, "진동": 1.2, "압력": 88,  "상태": "경고"}
}

print("=== 설비 상태 점검 ===")
for eq_id, data in equipment.items():
    print(f"{eq_id}: 온도 {data['온도']}도, 진동 {data['진동']}mm/s, 상태 {data['상태']}")


# 5-2. 이상 설비 필터링
print("\n5-2. 이상 설비 필터링")
abnormal = [eq for eq, data in equipment.items() if data["상태"] != "정상"]
print(f"이상 설비: {abnormal}")

# 5-3. 통계 계산
print("\n5-3. 설비별 통계")
temps = [data["온도"] for data in equipment.values()]
print(f"평균 온도: {sum(temps)/len(temps):.1f}도")
print(f"최고 온도: {max(temps)}도")


# ============================================================
# 실습 6: 함수 정의
# ============================================================

print("\n[실습 6] 함수 정의")
print("-" * 40)

# 6-1. 기본 함수
def calculate_defect_rate(total, defects):
    """불량률 계산 함수"""
    if total == 0:
        return 0
    return defects / total


# 6-2. 등급 판정 함수
def get_quality_grade(defect_rate):
    """불량률로 등급 판정"""
    if defect_rate <= 0.01:
        return "A", "우수"
    elif defect_rate <= 0.03:
        return "B", "양호"
    elif defect_rate <= 0.05:
        return "C", "주의"
    else:
        return "D", "개선필요"


# 6-3. 온도 상태 판정 함수
def check_temperature(temp, threshold=90):
    """온도 상태 확인"""
    if temp > threshold:
        return "경고", True
    else:
        return "정상", False


# 함수 사용 예시
print("\n함수 사용 예시:")
rate = calculate_defect_rate(1200, 30)
grade, message = get_quality_grade(rate)
temp_status, is_warning = check_temperature(92)

print(f"불량률: {rate:.1%}")
print(f"등급: {grade} ({message})")
print(f"온도 상태: {temp_status}")


# ============================================================
# 실습 7: 종합 예제 - 일일 품질 보고서
# ============================================================

print("\n[실습 7] 종합 예제 - 일일 품질 보고서")
print("-" * 40)

# 생산 데이터
production_data = {
    "날짜": "2024-12-19",
    "라인": "LINE-01",
    "교대": 2,
    "생산량": 1200,
    "양품": 1170,
    "불량": 30,
    "평균온도": 85.2,
    "평균습도": 45,
    "최고온도": 92,
    "최저온도": 78
}

# 계산
defect_rate = production_data["불량"] / production_data["생산량"]
good_rate = 1 - defect_rate

# 등급 판정
if defect_rate <= 0.02:
    grade = "A"
    grade_msg = "우수"
elif defect_rate <= 0.04:
    grade = "B"
    grade_msg = "양호"
else:
    grade = "C"
    grade_msg = "주의"

# 온도 상태
temp_status = "정상" if production_data["최고온도"] <= 90 else "주의"

# 보고서 출력
print("\n" + "=" * 50)
print("         일일 품질 보고서")
print("=" * 50)
print(f"날짜: {production_data['날짜']}")
print(f"라인: {production_data['라인']} / {production_data['교대']}교대")
print("-" * 50)
print(f"총 생산량: {production_data['생산량']:>10,}개")
print(f"양  품  수: {production_data['양품']:>10,}개")
print(f"불  량  수: {production_data['불량']:>10,}개")
print("-" * 50)
print(f"양 품 률: {good_rate*100:>10.1f}%")
print(f"불 량 률: {defect_rate*100:>10.1f}%")
print(f"품질 등급: {grade:>10} ({grade_msg})")
print("-" * 50)
print(f"평균 온도: {production_data['평균온도']:>10.1f}°C")
print(f"최고 온도: {production_data['최고온도']:>10}°C ({temp_status})")
print(f"평균 습도: {production_data['평균습도']:>10}%")
print("=" * 50)


# ============================================================
# 실습 8: 다중 라인 분석
# ============================================================

print("\n[실습 8] 다중 라인 분석")
print("-" * 40)

# 여러 라인 데이터
lines_data = {
    "LINE-01": {"생산량": 1200, "불량": 24, "온도": 85},
    "LINE-02": {"생산량": 1150, "불량": 46, "온도": 88},
    "LINE-03": {"생산량": 1300, "불량": 26, "온도": 84},
    "LINE-04": {"생산량": 1180, "불량": 59, "온도": 91},
    "LINE-05": {"생산량": 1250, "불량": 25, "온도": 86}
}

print("\n=== 라인별 성과 분석 ===")
print(f"{'라인':<10} {'생산량':>8} {'불량률':>8} {'등급':>6} {'온도':>6}")
print("-" * 42)

total_production = 0
total_defects = 0
best_line = None
best_rate = 1.0

for line_id, data in lines_data.items():
    rate = data["불량"] / data["생산량"]
    grade, _ = get_quality_grade(rate)
    temp_mark = "*" if data["온도"] > 90 else ""

    print(f"{line_id:<10} {data['생산량']:>8,} {rate:>7.1%} {grade:>6} {data['온도']:>5}°C{temp_mark}")

    total_production += data["생산량"]
    total_defects += data["불량"]

    if rate < best_rate:
        best_rate = rate
        best_line = line_id

print("-" * 42)
overall_rate = total_defects / total_production
print(f"{'전체':>10} {total_production:>8,} {overall_rate:>7.1%}")
print(f"\n최우수 라인: {best_line} (불량률 {best_rate:.1%})")
print("* 표시: 온도 90°C 초과")


# ============================================================
# 실습 9: 시간대별 모니터링
# ============================================================

print("\n[실습 9] 시간대별 모니터링")
print("-" * 40)

# 시간대별 데이터
hourly_data = [
    {"시간": "09:00", "온도": 82, "습도": 45, "생산": 150},
    {"시간": "10:00", "온도": 85, "습도": 47, "생산": 148},
    {"시간": "11:00", "온도": 88, "습도": 50, "생산": 145},
    {"시간": "12:00", "온도": 92, "습도": 52, "생산": 140},  # 점심
    {"시간": "13:00", "온도": 95, "습도": 55, "생산": 138},
    {"시간": "14:00", "온도": 91, "습도": 53, "생산": 142},
    {"시간": "15:00", "온도": 88, "습도": 50, "생산": 146},
    {"시간": "16:00", "온도": 85, "습도": 48, "생산": 149},
    {"시간": "17:00", "온도": 83, "습도": 46, "생산": 147}
]

temp_threshold = 90
humidity_threshold = 52

print("\n=== 시간대별 공정 현황 ===")
print(f"{'시간':<8} {'온도':>6} {'습도':>6} {'생산':>6} {'상태':<10}")
print("-" * 40)

warnings = []
for data in hourly_data:
    status_list = []
    if data["온도"] > temp_threshold:
        status_list.append("온도↑")
    if data["습도"] > humidity_threshold:
        status_list.append("습도↑")

    status = ", ".join(status_list) if status_list else "정상"

    if status_list:
        warnings.append(data["시간"])

    print(f"{data['시간']:<8} {data['온도']:>5}°C {data['습도']:>5}% {data['생산']:>5}개 {status:<10}")

print("-" * 40)

# 통계
temps = [d["온도"] for d in hourly_data]
productions = [d["생산"] for d in hourly_data]

print(f"평균 온도: {sum(temps)/len(temps):.1f}°C")
print(f"총 생산량: {sum(productions):,}개")
print(f"경고 시간대: {', '.join(warnings) if warnings else '없음'}")


# ============================================================
# 핵심 요약
# ============================================================

print("\n" + "=" * 60)
print("[2차시 핵심 요약]")
print("=" * 60)

summary = """
1. Python 개발 환경
   - Anaconda: 패키지 관리 및 가상환경
   - Jupyter Notebook: 대화형 코드 실행 환경
   - 핵심 단축키: Shift+Enter (실행)

2. 5가지 자료형
   - int: 정수 (1000, 30)
   - float: 실수 (85.7, 0.025)
   - str: 문자열 ("정상", "LINE-01")
   - list: 리스트 [1, 2, 3]
   - dict: 딕셔너리 {"키": 값}

3. f-string 포맷팅
   - 기본: f'{변수}'
   - 소수점: f'{값:.2f}'
   - 퍼센트: f'{값:.1%}'
   - 콤마: f'{값:,}'

4. 제어문
   - if/elif/else: 조건 분기
   - for: 반복 처리
   - enumerate: 인덱스와 값 동시 접근

5. 함수 정의
   - def 함수명(매개변수):
   - return 반환값
"""

print(summary)

print("=" * 60)
print("다음 차시: 제조 데이터 다루기 기초 (NumPy, Pandas)")
print("=" * 60)
