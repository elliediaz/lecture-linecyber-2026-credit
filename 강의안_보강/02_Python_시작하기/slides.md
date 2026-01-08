---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 2차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# Python 시작하기

## 2차시 | Part I. AI 윤리와 환경 구축

**제조 AI 개발을 위한 첫 번째 도구 준비**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **Anaconda와 Jupyter Notebook**을 설치하고 실행한다
2. **Python 기본 데이터 종류**(숫자, 문자, 목록 등)를 이해한다
3. **조건문과 반복문**으로 간단한 프로그램을 작성한다
4. **제조 현장 데이터**를 Python으로 다루는 기초를 익힌다

---

# 오늘의 진행 순서

## 이론 + 실습 (25-30분)

| 순서 | 내용 | 시간 |
|------|------|------|
| 1 | 왜 Python인가? | 3분 |
| 2 | 개발 환경 구축 | 5분 |
| 3 | Python 기본 문법 | 7분 |
| 4 | 실습: 제조 데이터 다루기 | 15분 |

---

# 왜 Python인가?

## AI 개발의 사실상 표준 언어

| 특징 | 설명 |
|------|------|
| **쉬운 문법** | 영어처럼 읽히는 직관적 코드 |
| **풍부한 라이브러리** | NumPy, Pandas, TensorFlow, PyTorch |
| **거대한 커뮤니티** | 검색하면 답이 바로 나옴 |
| **빠른 개발 속도** | 아이디어를 빠르게 구현 가능 |

> **Python은 AI/데이터 분석 분야에서 90% 이상 사용**

---

# Python 사용 현황

## 글로벌 기업들의 Python 활용

| 기업 | 활용 분야 |
|------|-----------|
| **Google** | TensorFlow, YouTube 백엔드 |
| **Netflix** | 추천 시스템, 데이터 분석 |
| **Instagram** | 서버 백엔드 전체 |
| **Spotify** | 음악 추천 알고리즘 |
| **삼성전자** | AI 연구, 품질 예측 |
| **현대자동차** | 자율주행, 예지정비 |

---

# 다른 언어와 비교

## Python이 AI에 적합한 이유

```
┌─────────────────────────────────────────────────────────────┐
│                    언어별 특징 비교                          │
├─────────────────────────────────────────────────────────────┤
│  Python    │ 쉬운 문법, AI 라이브러리 풍부, 빠른 개발       │
│  Java      │ 기업용, 복잡한 문법, AI 라이브러리 부족        │
│  C++       │ 빠른 속도, 어려운 문법, 개발 시간 오래 걸림    │
│  R         │ 통계 특화, 범용성 부족                         │
│  Julia     │ 빠른 속도, 생태계 아직 작음                    │
└─────────────────────────────────────────────────────────────┘
```

> **결론**: AI/ML 분야에서는 Python이 최선의 선택

---

# Python 코드 맛보기

## 다른 언어와 비교

```java
// Java로 "Hello World" 출력
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
```

```python
# Python으로 "Hello World" 출력
print("Hello World")
```

> Python은 **단 1줄**로 같은 결과를 얻을 수 있습니다!

---

# 개발 환경 선택

## Anaconda vs 일반 Python

| 구분 | 일반 Python | Anaconda |
|------|-------------|----------|
| **패키지 관리** | pip (충돌 빈번) | conda (안정적) |
| **데이터 과학 패키지** | 수동 설치 필요 | 기본 포함 |
| **환경 관리** | 어려움 | 가상환경 쉽게 생성 |
| **Jupyter** | 별도 설치 | 기본 포함 |

### 결론
> **초보자는 Anaconda 강력 추천!**

---

# Anaconda 설치 방법

## Step-by-Step 가이드

### 1단계: 다운로드
- https://www.anaconda.com/download 접속
- Windows 64-bit 버전 선택

### 2단계: 설치 (약 10분 소요)
- 설치 파일 실행
- "Just Me" 선택 (개인 사용)
- 설치 경로: 기본값 유지

### 3단계: 확인
```bash
# 명령 프롬프트에서 실행
conda --version
python --version
```

---

# 설치 시 주의사항

## 자주 발생하는 문제

### 문제 1: 설치 중 멈춤
→ **해결**: 관리자 권한으로 실행, 백신 일시 중지

### 문제 2: conda 명령어 인식 안 됨
→ **해결**: Anaconda Prompt 사용 (일반 CMD 아님)

### 문제 3: 용량 부족
→ **해결**: Miniconda로 대체 (최소 설치)

### 문제 4: 방화벽 차단
→ **해결**: IT 담당자에게 포트 오픈 요청 (8888)

---

# Jupyter Notebook

## 코드 작성과 실행을 한 곳에서

```
┌─────────────────────────────────────────────────────────────┐
│ Jupyter Notebook                                             │
├─────────────────────────────────────────────────────────────┤
│ [File] [Edit] [View] [Cell] [Kernel] [Help]                 │ ← 메뉴바
├─────────────────────────────────────────────────────────────┤
│ In [1]: print("제조 AI 시작!")                              │ ← 코드 셀
├─────────────────────────────────────────────────────────────┤
│ 제조 AI 시작!                                               │ ← 출력 결과
├─────────────────────────────────────────────────────────────┤
│ In [2]: 1200 * 0.975                                        │ ← 다음 셀
├─────────────────────────────────────────────────────────────┤
│ Out[2]: 1170.0                                              │ ← 계산 결과
└─────────────────────────────────────────────────────────────┘
```

---

# Jupyter Notebook 장점

## 왜 Jupyter를 사용하는가?

| 장점 | 설명 |
|------|------|
| **즉각적 결과 확인** | 코드 실행 결과를 바로 확인 |
| **시각화 통합** | 그래프를 셀 아래에 바로 표시 |
| **문서화 용이** | 코드 + 설명 + 결과 한 파일에 |
| **공유 편리** | .ipynb 파일로 쉽게 공유 |
| **웹 기반** | 브라우저만 있으면 어디서든 |

> **데이터 분석가의 80%가 Jupyter 사용**

---

# Jupyter 핵심 단축키

## 꼭 기억해야 할 단축키

| 단축키 | 기능 |
|--------|------|
| `Shift + Enter` | 셀 실행 후 다음 셀로 이동 |
| `Ctrl + Enter` | 셀 실행 (현재 셀 유지) |
| `Esc + A` | 위에 새 셀 추가 |
| `Esc + B` | 아래에 새 셀 추가 |
| `Esc + D, D` | 현재 셀 삭제 |
| `Esc + M` | 마크다운 셀로 변경 |
| `Esc + Y` | 코드 셀로 변경 |

---

# Python 기본 데이터 종류

## 5가지 핵심 자료형

```python
# 1. 정수 (int) - 생산량, 개수
production_count = 1000

# 2. 실수 (float) - 온도, 불량률
temperature = 85.7

# 3. 문자열 (str) - 제품명, 상태
status = "정상"

# 4. 리스트 (list) - 여러 데이터 모음
daily_temp = [82, 85, 88, 84]

# 5. 딕셔너리 (dict) - 이름표 붙은 데이터
sensor = {"온도": 85, "습도": 45}
```

---

# 자료형 1: 정수 (int)

## 소수점 없는 숫자

```python
# 제조 현장 정수 데이터 예시
daily_production = 1200    # 일 생산량
defect_count = 30          # 불량 개수
line_number = 3            # 라인 번호
shift = 2                  # 근무 교대 (1, 2, 3교대)

# 정수 연산
total_week = daily_production * 5     # 5일 생산량
remaining = 1000 - 750                # 남은 목표

# 정수 나눗셈
packs = 1200 // 12        # 12개씩 포장하면 100팩
remainder = 1200 % 12     # 나머지: 0개
```

---

# 자료형 2: 실수 (float)

## 소수점 있는 숫자

```python
# 제조 현장 실수 데이터 예시
temperature = 85.7         # 온도 (°C)
defect_rate = 0.025        # 불량률 (2.5%)
pressure = 101.325         # 기압 (kPa)
vibration = 0.42           # 진동 (mm/s)

# 실수 계산
production = 1200
good_count = production * (1 - defect_rate)    # 1170.0

# 반올림
import math
rounded = round(85.736, 1)    # 85.7
floored = math.floor(85.7)    # 85
ceiled = math.ceil(85.1)      # 86
```

---

# 자료형 3: 문자열 (str)

## 텍스트 데이터

```python
# 기본 문자열
product_name = "센서 모듈 A"
line_id = "LINE-01"
status = "정상"

# 문자열 연결
full_id = line_id + "-" + status          # "LINE-01-정상"

# f-string (가장 많이 사용!)
temp = 85.7
rate = 0.025
message = f"온도: {temp}도, 불량률: {rate:.1%}"
# 결과: "온도: 85.7도, 불량률: 2.5%"

# 문자열 메서드
upper = "hello".upper()       # "HELLO"
lower = "HELLO".lower()       # "hello"
replaced = "A-B-C".replace("-", "_")  # "A_B_C"
```

---

# f-string 상세 가이드

## 포맷팅 옵션

```python
value = 1234.5678

# 소수점 자릿수
print(f"{value:.2f}")       # "1234.57" (소수점 2자리)
print(f"{value:.0f}")       # "1235" (소수점 없음)

# 퍼센트 표시
rate = 0.0253
print(f"{rate:.1%}")        # "2.5%"
print(f"{rate:.2%}")        # "2.53%"

# 천 단위 콤마
big_num = 1234567
print(f"{big_num:,}")       # "1,234,567"

# 정렬
name = "온도"
print(f"{name:<10}")        # "온도        " (왼쪽 정렬)
print(f"{name:>10}")        # "        온도" (오른쪽 정렬)
print(f"{name:^10}")        # "    온도    " (가운데 정렬)
```

---

# 자료형 4: 리스트 (list)

## 여러 데이터를 순서대로 저장

```python
# 일주일 생산량 데이터
weekly_production = [1200, 1150, 1300, 1180, 1250]

# 인덱스로 접근 (0부터 시작!)
first = weekly_production[0]      # 1200 (첫 번째)
last = weekly_production[-1]      # 1250 (마지막)

# 슬라이싱
first_three = weekly_production[0:3]    # [1200, 1150, 1300]

# 값 추가/수정
weekly_production.append(1280)          # 맨 뒤에 추가
weekly_production[0] = 1220             # 첫 번째 값 수정

# 기본 통계
total = sum(weekly_production)          # 합계
count = len(weekly_production)          # 개수
average = total / count                 # 평균
max_val = max(weekly_production)        # 최대값
min_val = min(weekly_production)        # 최소값
```

---

# 리스트 활용 예제

## 센서 데이터 분석

```python
# 1시간 간격 온도 측정값
temperatures = [82, 85, 88, 95, 84, 91, 86, 83]

# 기본 통계
print(f"최고 온도: {max(temperatures)}도")
print(f"최저 온도: {min(temperatures)}도")
print(f"평균 온도: {sum(temperatures)/len(temperatures):.1f}도")

# 정렬
sorted_temps = sorted(temperatures)           # 오름차순
sorted_desc = sorted(temperatures, reverse=True)  # 내림차순

# 특정 값 개수 세기
count_85 = temperatures.count(85)             # 85도가 몇 번?

# 특정 값 위치 찾기
idx = temperatures.index(95)                  # 95도의 인덱스
print(f"최고 온도는 {idx+1}시에 발생")
```

---

# 자료형 5: 딕셔너리 (dict)

## 이름표를 붙여 데이터 저장

```python
# 센서 데이터
sensor_data = {
    "온도": 85.2,
    "압력": 101.3,
    "습도": 45,
    "상태": "정상"
}

# 값 접근
temp = sensor_data["온도"]          # 85.2
status = sensor_data["상태"]        # "정상"

# 값 추가/수정
sensor_data["진동"] = 0.42          # 새 항목 추가
sensor_data["온도"] = 86.0          # 값 수정

# 안전한 접근 (키가 없어도 에러 안 남)
speed = sensor_data.get("속도", 0)  # 없으면 0 반환
```

---

# 딕셔너리 활용 예제

## 설비 상태 관리

```python
# 중첩 딕셔너리 (딕셔너리 안에 딕셔너리)
equipment = {
    "EQ-001": {"온도": 82, "진동": 0.3, "상태": "정상"},
    "EQ-002": {"온도": 91, "진동": 0.8, "상태": "주의"},
    "EQ-003": {"온도": 85, "진동": 0.4, "상태": "정상"}
}

# 특정 설비 데이터 접근
eq1_temp = equipment["EQ-001"]["온도"]     # 82

# 모든 키 확인
eq_ids = equipment.keys()      # dict_keys(['EQ-001', ...])

# 모든 값 확인
eq_data = equipment.values()   # dict_values([{...}, {...}, ...])

# 키-값 쌍으로 반복
for eq_id, data in equipment.items():
    print(f"{eq_id}: {data['상태']}")
```

---

# 조건문 if

## "만약 ~라면 ~해라"

```python
# 기본 구조
temperature = 92

if temperature > 90:
    print("경고: 온도 초과!")
    print("냉각 필요")
elif temperature > 80:
    print("정상 범위")
else:
    print("온도 낮음")

# 핵심 문법
# 1. 조건 뒤에 콜론(:) 필수
# 2. 다음 줄은 들여쓰기 (4칸 스페이스)
# 3. elif = else if의 줄임말
# 4. else는 나머지 모든 경우
```

---

# 조건문 예제

## 품질 등급 판정

```python
defect_rate = 0.025  # 2.5%

# 불량률에 따른 등급 판정
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

print(f"등급: {grade} ({message})")
print(f"조치: {action}")
```

---

# 복합 조건문

## and, or, not 사용

```python
temperature = 88
pressure = 95

# AND 조건: 둘 다 만족해야
if temperature > 85 and pressure < 100:
    print("조건 1 충족")

# OR 조건: 하나만 만족해도
if temperature > 90 or pressure > 110:
    print("경고: 이상 감지!")

# NOT 조건: 부정
status = "정상"
if not status == "정상":
    print("문제 발생!")

# 복합 조건
if (temperature > 85 and temperature < 90) or pressure > 100:
    print("주의 필요")
```

---

# 반복문 for

## 같은 작업을 여러 번 실행

```python
# 리스트 순회
temperatures = [82, 85, 88, 95, 84]

for temp in temperatures:
    print(f"온도: {temp}도")

# 인덱스와 함께 순회 (enumerate)
for i, temp in enumerate(temperatures):
    print(f"{i+1}번째 측정: {temp}도")

# 범위 순회 (range)
for i in range(5):          # 0, 1, 2, 3, 4
    print(f"반복 {i}")

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(f"반복 {i}")
```

---

# 반복문 예제

## 센서 데이터 모니터링

```python
temperatures = [82, 85, 88, 95, 84, 91, 86]
threshold = 90  # 경고 기준

print("=== 온도 모니터링 ===")
warning_count = 0

for i, temp in enumerate(temperatures):
    hour = i + 1
    if temp > threshold:
        print(f"[경고] {hour}시: {temp}도 - 기준 초과!")
        warning_count += 1
    else:
        print(f"[정상] {hour}시: {temp}도")

# 결과 요약
print(f"\n총 경고 횟수: {warning_count}회")
print(f"경고 비율: {warning_count/len(temperatures)*100:.1f}%")
```

---

# 반복문과 리스트

## 리스트 컴프리헨션 (고급)

```python
# 일반 반복문
squares = []
for x in range(5):
    squares.append(x ** 2)
# 결과: [0, 1, 4, 9, 16]

# 리스트 컴프리헨션 (한 줄로!)
squares = [x ** 2 for x in range(5)]

# 조건 추가
temps = [82, 85, 88, 95, 84, 91, 86]
high_temps = [t for t in temps if t > 90]
# 결과: [95, 91]

# 변환 + 조건
# 90도 이상인 온도를 화씨로 변환
fahrenheit = [t * 9/5 + 32 for t in temps if t > 90]
```

---

# 이론 정리

## 핵심 포인트

### Python 환경
- **Anaconda**: 패키지 관리 및 가상환경
- **Jupyter Notebook**: 대화형 코드 실행 환경

### 5가지 자료형
- **int/float**: 숫자 (정수/실수)
- **str**: 문자열 (텍스트)
- **list**: 순서 있는 데이터 모음
- **dict**: 이름표 있는 데이터 모음

### 제어문
- **if**: 조건에 따른 분기
- **for**: 반복 실행

---

# - 실습편 -

## 2차시

**Python 기초 실습: 제조 데이터 다루기**

---

# 실습 개요

## 제조 현장 데이터로 Python 연습

### 실습 목표
1. Jupyter Notebook에서 코드 실행
2. 변수에 데이터 저장하고 계산
3. 조건문으로 품질 판정
4. 반복문으로 데이터 분석

### 실습 환경 확인
```bash
# Anaconda Prompt에서 실행
jupyter notebook
```

---

# 실습 1: 변수와 기본 계산

## 생산량 데이터 다루기

```python
# 센서 측정값 입력
temperature = 85.2      # 온도 (°C)
pressure = 101.3        # 압력 (kPa)
humidity = 45           # 습도 (%)

# 생산 데이터
daily_production = 1200
defect_rate = 0.025     # 2.5%

# 불량 수 계산
defect_count = daily_production * defect_rate
good_count = daily_production - defect_count

# 결과 출력
print(f"=== 일일 생산 현황 ===")
print(f"총 생산량: {daily_production:,}개")
print(f"양품: {good_count:,.0f}개")
print(f"불량: {defect_count:,.0f}개")
print(f"양품률: {(1-defect_rate)*100:.1f}%")
```

---

# 실습 2: 조건문 활용

## 품질 등급 자동 판정

```python
# 입력 데이터
defect_rate = 0.025  # 불량률 2.5%
temperature = 88     # 온도
pressure = 102       # 압력

# 품질 등급 판정
if defect_rate <= 0.01:
    grade, msg = "A", "우수"
elif defect_rate <= 0.03:
    grade, msg = "B", "양호"
elif defect_rate <= 0.05:
    grade, msg = "C", "주의"
else:
    grade, msg = "D", "개선 필요"

# 온도 이상 확인
temp_status = "정상" if temperature <= 90 else "경고"

# 결과 출력
print(f"품질 등급: {grade} ({msg})")
print(f"온도 상태: {temp_status}")
```

---

# 실습 3: 반복문으로 데이터 분석

## 시간별 온도 모니터링

```python
# 1시간 간격 온도 데이터 (9시~17시)
temperatures = [82, 85, 88, 95, 92, 89, 91, 86, 84]
hours = ["09:00", "10:00", "11:00", "12:00", "13:00",
         "14:00", "15:00", "16:00", "17:00"]
threshold = 90

print("=== 시간별 온도 모니터링 ===")
warnings = []

for i, temp in enumerate(temperatures):
    status = "[경고]" if temp > threshold else "[정상]"
    print(f"{hours[i]} | {temp:>3}도 | {status}")

    if temp > threshold:
        warnings.append(hours[i])

# 요약
print(f"\n평균 온도: {sum(temperatures)/len(temperatures):.1f}도")
print(f"최고 온도: {max(temperatures)}도")
print(f"경고 시간대: {', '.join(warnings) if warnings else '없음'}")
```

---

# 실습 4: 딕셔너리로 설비 관리

## 다중 설비 상태 점검

```python
# 설비별 센서 데이터
equipment = {
    "EQ-001": {"온도": 82, "진동": 0.3, "압력": 100, "상태": "정상"},
    "EQ-002": {"온도": 91, "진동": 0.8, "압력": 95,  "상태": "주의"},
    "EQ-003": {"온도": 85, "진동": 0.4, "압력": 102, "상태": "정상"},
    "EQ-004": {"온도": 96, "진동": 1.2, "압력": 88,  "상태": "경고"}
}

print("=== 설비 상태 점검 ===")
for eq_id, data in equipment.items():
    print(f"\n{eq_id}:")
    print(f"  온도: {data['온도']}도")
    print(f"  진동: {data['진동']}mm/s")
    print(f"  상태: {data['상태']}")

# 이상 설비 필터링
abnormal = [eq for eq, data in equipment.items()
            if data["상태"] != "정상"]
print(f"\n이상 설비: {', '.join(abnormal)}")
```

---

# 실습 5: 종합 예제

## 일일 품질 보고서 생성

```python
# 생산 데이터
production_data = {
    "날짜": "2024-12-19",
    "라인": "LINE-01",
    "생산량": 1200,
    "양품": 1170,
    "불량": 30,
    "평균온도": 85.2,
    "평균습도": 45
}

# 계산
defect_rate = production_data["불량"] / production_data["생산량"]

# 등급 판정
if defect_rate <= 0.02:
    grade = "A"
else:
    grade = "B" if defect_rate <= 0.04 else "C"

# 보고서 출력
print("=" * 50)
print("         일일 품질 보고서")
print("=" * 50)
print(f"날짜: {production_data['날짜']}")
print(f"라인: {production_data['라인']}")
print("-" * 50)
print(f"총 생산량: {production_data['생산량']:>10,}개")
print(f"양  품  수: {production_data['양품']:>10,}개")
print(f"불  량  수: {production_data['불량']:>10,}개")
print("-" * 50)
print(f"양 품 률: {(1-defect_rate)*100:>10.1f}%")
print(f"불 량 률: {defect_rate*100:>10.1f}%")
print(f"품질 등급: {grade:>10}")
print("=" * 50)
```

---

# 실습 6: 데이터 처리 함수 만들기

## 재사용 가능한 코드

```python
def calculate_defect_rate(total, defects):
    """불량률 계산 함수"""
    return defects / total if total > 0 else 0

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

# 함수 사용
rate = calculate_defect_rate(1200, 30)
grade, message = get_quality_grade(rate)

print(f"불량률: {rate:.1%}")
print(f"등급: {grade} ({message})")
```

---

# 자주 하는 실수

## 초보자 주의사항

```python
# 1. 들여쓰기 오류
if True:
print("에러!")      # IndentationError

# 2. 콜론(:) 누락
if True
    print("에러!")  # SyntaxError

# 3. = vs == 혼동
x = 10
if x = 10:          # 에러! (할당)
if x == 10:         # 정답! (비교)

# 4. 인덱스 범위 초과
data = [1, 2, 3]
print(data[3])      # IndexError (0,1,2만 존재)

# 5. 문자열 따옴표 불일치
name = "홍길동'     # SyntaxError
```

---

# 디버깅 팁

## 에러 해결 방법

### 1. 에러 메시지 읽기
```
IndexError: list index out of range
→ 리스트 범위를 벗어난 인덱스 접근
```

### 2. print로 확인
```python
temperatures = [82, 85, 88]
print(f"리스트 길이: {len(temperatures)}")
print(f"현재 값: {temperatures}")
```

### 3. 검색하기
- 에러 메시지를 그대로 구글에 검색
- Stack Overflow에 거의 다 답이 있음

---

# 실습 정리

## 핵심 체크포인트

### 환경 설정
- [ ] Anaconda 설치 완료
- [ ] Jupyter Notebook 실행 확인
- [ ] 단축키 익히기 (Shift+Enter)

### Python 기초
- [ ] 5가지 자료형 이해 (int, float, str, list, dict)
- [ ] f-string 사용법
- [ ] 조건문 if/elif/else
- [ ] 반복문 for/enumerate
- [ ] 함수 정의 def

---

# 다음 차시 예고

## 3차시: 제조 데이터 다루기 기초

### 학습 내용
- **NumPy**: 수치 계산의 핵심 도구
- **Pandas**: 표 형태 데이터 다루기
- CSV 파일 불러오기 및 기본 조작

### 미리보기
```python
import pandas as pd
df = pd.read_csv('sensor_data.csv')
print(df.head())        # 처음 5행 보기
print(df.describe())    # 기본 통계
```

---

# 정리 및 Q&A

## 오늘의 핵심

1. **환경 설정**: Anaconda + Jupyter Notebook
2. **자료형**: 정수, 실수, 문자열, 리스트, 딕셔너리
3. **제어문**: if 조건문, for 반복문
4. **함수**: def로 재사용 가능한 코드 작성

### 문제 해결
- 설치 오류 → 관리자 권한으로 실행
- 한글 깨짐 → UTF-8 인코딩 확인
- 에러 발생 → 에러 메시지 읽고 검색

---

# 감사합니다

## 2차시: Python 시작하기

**수고하셨습니다!**
**다음 시간에 NumPy와 Pandas를 배워봅시다!**
