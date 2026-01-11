# 2차시: Python 시작하기

## 학습 목표

본 차시를 마치면 다음을 수행할 수 있음:

1. **Anaconda와 Jupyter Notebook**을 설치하고 실행함
2. **Python 기본 데이터 종류**(숫자, 문자, 목록 등)를 이해함
3. **조건문과 반복문**으로 간단한 프로그램을 작성함
4. **제조 현장 데이터**를 Python으로 다루는 기초를 익힘

---

## 강의 구성

| 순서 | 내용 | 시간 |
|------|------|------|
| Part 1 | 왜 Python인가? | 3분 |
| Part 2 | 개발 환경 구축 | 5분 |
| Part 3 | Python 기본 문법 | 7분 |
| Part 4 | 실습: 제조 데이터 다루기 | 15분 |

---

## Part 1: 왜 Python인가?

### 개념 설명

Python은 AI 개발의 사실상 표준 언어임. 다음과 같은 특징으로 인해 데이터 분석과 AI 분야에서 가장 널리 사용됨.

**Python의 주요 특징**

| 특징 | 설명 |
|------|------|
| 쉬운 문법 | 영어처럼 읽히는 직관적 코드 |
| 풍부한 라이브러리 | NumPy, Pandas, TensorFlow, PyTorch |
| 거대한 커뮤니티 | 검색하면 답이 바로 나옴 |
| 빠른 개발 속도 | 아이디어를 빠르게 구현 가능 |

> Python은 AI/데이터 분석 분야에서 90% 이상 사용됨

**글로벌 기업들의 Python 활용**

| 기업 | 활용 분야 |
|------|-----------|
| Google | TensorFlow, YouTube 백엔드 |
| Netflix | 추천 시스템, 데이터 분석 |
| Instagram | 서버 백엔드 전체 |
| Spotify | 음악 추천 알고리즘 |
| 삼성전자 | AI 연구, 품질 예측 |
| 현대자동차 | 자율주행, 예지정비 |

**언어별 특징 비교**

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

**코드 비교 예시**

Java로 "Hello World" 출력:
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
```

Python으로 "Hello World" 출력:
```python
print("Hello World")
```

> Python은 단 1줄로 같은 결과를 얻을 수 있음

---

## Part 2: 개발 환경 구축

### 개념 설명

**Anaconda vs 일반 Python**

| 구분 | 일반 Python | Anaconda |
|------|-------------|----------|
| 패키지 관리 | pip (충돌 빈번) | conda (안정적) |
| 데이터 과학 패키지 | 수동 설치 필요 | 기본 포함 |
| 환경 관리 | 어려움 | 가상환경 쉽게 생성 |
| Jupyter | 별도 설치 | 기본 포함 |

> 초보자는 Anaconda를 강력히 권장함

**Anaconda 설치 방법**

1단계: 다운로드
- https://www.anaconda.com/download 접속
- Windows 64-bit 버전 선택

2단계: 설치 (약 10분 소요)
- 설치 파일 실행
- "Just Me" 선택 (개인 사용)
- 설치 경로: 기본값 유지

3단계: 확인
```bash
# 명령 프롬프트에서 실행
conda --version
python --version
```

**설치 시 주의사항**

| 문제 | 해결 방법 |
|------|----------|
| 설치 중 멈춤 | 관리자 권한으로 실행, 백신 일시 중지 |
| conda 명령어 인식 안 됨 | Anaconda Prompt 사용 (일반 CMD 아님) |
| 용량 부족 | Miniconda로 대체 (최소 설치) |
| 방화벽 차단 | IT 담당자에게 포트 오픈 요청 (8888) |

**Jupyter Notebook 구조**

```
┌─────────────────────────────────────────────────────────────┐
│ Jupyter Notebook                                             │
├─────────────────────────────────────────────────────────────┤
│ [File] [Edit] [View] [Cell] [Kernel] [Help]                 │ <- 메뉴바
├─────────────────────────────────────────────────────────────┤
│ In [1]: print("제조 AI 시작!")                              │ <- 코드 셀
├─────────────────────────────────────────────────────────────┤
│ 제조 AI 시작!                                               │ <- 출력 결과
├─────────────────────────────────────────────────────────────┤
│ In [2]: 1200 * 0.975                                        │ <- 다음 셀
├─────────────────────────────────────────────────────────────┤
│ Out[2]: 1170.0                                              │ <- 계산 결과
└─────────────────────────────────────────────────────────────┘
```

**Jupyter Notebook 장점**

| 장점 | 설명 |
|------|------|
| 즉각적 결과 확인 | 코드 실행 결과를 바로 확인 |
| 시각화 통합 | 그래프를 셀 아래에 바로 표시 |
| 문서화 용이 | 코드 + 설명 + 결과 한 파일에 |
| 공유 편리 | .ipynb 파일로 쉽게 공유 |
| 웹 기반 | 브라우저만 있으면 어디서든 |

> 데이터 분석가의 80%가 Jupyter를 사용함

**핵심 단축키**

| 단축키 | 기능 |
|--------|------|
| Shift + Enter | 셀 실행 후 다음 셀로 이동 |
| Ctrl + Enter | 셀 실행 (현재 셀 유지) |
| Esc + A | 위에 새 셀 추가 |
| Esc + B | 아래에 새 셀 추가 |
| Esc + D, D | 현재 셀 삭제 |
| Esc + M | 마크다운 셀로 변경 |
| Esc + Y | 코드 셀로 변경 |

---

## Part 3: Python 기본 문법

### 개념 설명

Python에서 사용하는 5가지 핵심 자료형을 이해해야 함.

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 5가지 핵심 자료형                   │
├─────────────────────────────────────────────────────────────┤
│  int (정수)     │ 생산량, 개수 등 소수점 없는 숫자           │
│  float (실수)   │ 온도, 불량률 등 소수점 있는 숫자           │
│  str (문자열)   │ 제품명, 상태 등 텍스트 데이터              │
│  list (리스트)  │ 여러 데이터를 순서대로 저장                │
│  dict (딕셔너리)│ 이름표를 붙여 데이터 저장                  │
└─────────────────────────────────────────────────────────────┘
```

### 실습 코드

#### 3-1. 정수 (int)

```python
# 제조 현장 정수 데이터 예시
daily_production = 1200    # 일 생산량
defect_count = 30          # 불량 개수
line_number = 3            # 라인 번호

print(f"일 생산량: {daily_production}개")
print(f"불량 개수: {defect_count}개")
print(f"라인 번호: {line_number}번")

# 정수 연산
total_week = daily_production * 5
remaining = 1000 - 750
print(f"5일 생산량: {total_week}개")
print(f"남은 목표: {remaining}개")

# 정수 나눗셈
packs = 1200 // 12        # 12개씩 포장하면 100팩
remainder = 1200 % 12     # 나머지: 0개
```

**결과 해설**: 정수형은 소수점이 없는 숫자를 저장함. 제조 현장에서 생산량, 개수, 라인 번호 등에 사용됨. `//`는 몫, `%`는 나머지를 계산함.

#### 3-2. 실수 (float)

```python
# 제조 현장 실수 데이터 예시
temperature = 85.7         # 온도 (°C)
defect_rate = 0.025        # 불량률 (2.5%)
pressure = 101.325         # 기압 (kPa)

print(f"온도: {temperature}°C")
print(f"불량률: {defect_rate} ({defect_rate * 100}%)")
print(f"압력: {pressure} kPa")

# 실수 계산
daily_production = 1200
good_count = daily_production * (1 - defect_rate)
print(f"양품 수: {good_count:.0f}개")
```

**결과 해설**: 실수형은 소수점이 있는 숫자를 저장함. 온도, 불량률, 압력 등 정밀한 측정값에 사용됨. `:.0f`는 소수점 없이 출력하는 포맷 지정자임.

#### 3-3. 문자열 (str)

```python
# 기본 문자열
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
```

**결과 해설**: 문자열은 텍스트 데이터를 저장함. `+` 연산자로 문자열을 연결할 수 있고, `.upper()`, `.lower()`, `.replace()` 등의 메서드로 변환 가능함.

#### 3-4. f-string 포맷팅

```python
value = 1234.5678
rate = 0.0253
big_num = 1234567

# 소수점 자릿수
print(f"소수점 2자리: {value:.2f}")    # "1234.57"
print(f"소수점 없음: {value:.0f}")     # "1235"

# 퍼센트 표시
print(f"퍼센트 1자리: {rate:.1%}")     # "2.5%"
print(f"퍼센트 2자리: {rate:.2%}")     # "2.53%"

# 천 단위 콤마
print(f"콤마 표시: {big_num:,}")       # "1,234,567"

# 정렬
name = "온도"
print(f"왼쪽 정렬: [{name:<10}]")      # "[온도        ]"
print(f"오른쪽 정렬: [{name:>10}]")    # "[        온도]"
print(f"가운데 정렬: [{name:^10}]")    # "[    온도    ]"
```

**결과 해설**: f-string은 Python 3.6 이상에서 사용 가능한 문자열 포맷팅 방법임. `:.2f`(소수점 2자리), `:.1%`(퍼센트), `:,`(천 단위 콤마), `:<10`(정렬) 등 다양한 포맷 옵션을 제공함.

#### 3-5. 리스트 (list)

```python
# 일주일 생산량 데이터
weekly_production = [1200, 1150, 1300, 1180, 1250]
print(f"주간 생산량: {weekly_production}")

# 인덱스 접근 (0부터 시작!)
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
```

**결과 해설**: 리스트는 여러 데이터를 순서대로 저장함. 인덱스는 0부터 시작하며, `-1`은 마지막 요소를 의미함. `append()`로 요소 추가, `sum()`, `len()`, `max()`, `min()`으로 기본 통계 계산 가능함.

#### 3-6. 딕셔너리 (dict)

```python
# 센서 데이터
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

# 안전한 접근 (키가 없어도 에러 안 남)
speed = sensor_data.get("속도", 0)
print(f"속도 (없으면 0): {speed}")
```

**결과 해설**: 딕셔너리는 키-값 쌍으로 데이터를 저장함. 키로 값에 접근하며, 존재하지 않는 키에 접근 시 `.get()` 메서드를 사용하면 기본값을 반환받을 수 있음.

---

## Part 4: 제어문과 실습

### 개념 설명

**조건문 구조**

```
┌─────────────────────────────────────────────────────────────┐
│                       조건문 흐름                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────┐                               │
│                    │ 조건 확인 │                              │
│                    └────┬────┘                               │
│                         │                                    │
│            ┌────────────┼────────────┐                       │
│            │            │            │                       │
│        ┌───▼───┐   ┌───▼───┐   ┌───▼───┐                    │
│        │ 참(A) │   │ 참(B) │   │ 거짓  │                     │
│        │ 처리  │   │ 처리  │   │ 처리  │                     │
│        └───────┘   └───────┘   └───────┘                    │
│                                                              │
│  if 조건A:          elif 조건B:       else:                  │
│      실행문              실행문           실행문              │
└─────────────────────────────────────────────────────────────┘
```

**반복문 구조**

```
┌─────────────────────────────────────────────────────────────┐
│                       반복문 흐름                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│              ┌─────────────┐                                 │
│              │ 리스트/범위 │                                 │
│              └──────┬──────┘                                 │
│                     │                                        │
│              ┌──────▼──────┐                                 │
│              │ 요소 하나씩 │◄────────────┐                   │
│              │   꺼내기    │             │                   │
│              └──────┬──────┘             │                   │
│                     │                    │                   │
│              ┌──────▼──────┐             │                   │
│              │   실행문    │─────────────┘                   │
│              └─────────────┘                                 │
│                                                              │
│  for item in 리스트:                                         │
│      실행문                                                  │
└─────────────────────────────────────────────────────────────┘
```

### 실습 코드

#### 4-1. 조건문 기본

```python
# 기본 조건문
temperature = 92

if temperature > 90:
    print(f"온도 {temperature}도: 경고!")
elif temperature > 80:
    print(f"온도 {temperature}도: 정상")
else:
    print(f"온도 {temperature}도: 낮음")
```

**결과 해설**: `if` 문은 조건이 참일 때 실행됨. `elif`(else if의 줄임)는 앞 조건이 거짓이고 해당 조건이 참일 때 실행됨. `else`는 모든 조건이 거짓일 때 실행됨. 콜론(`:`)과 들여쓰기가 필수임.

#### 4-2. 품질 등급 판정

```python
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
```

**결과 해설**: 불량률 2.5%는 0.01 초과, 0.03 이하이므로 B등급(양호)으로 판정됨. 실제 제조 현장에서 품질 등급을 자동 판정하는 로직에 활용됨.

#### 4-3. 복합 조건

```python
temp = 88
pressure = 95
vibration = 0.5

# AND 조건: 둘 다 만족해야 함
if temp > 85 and pressure < 100:
    print("조건1 충족: 온도 높고 압력 정상")

# OR 조건: 하나만 만족해도 됨
if temp > 90 or vibration > 0.8:
    print("조건2 충족: 온도 또는 진동 이상")
else:
    print("조건2 미충족: 정상 범위")

# 복합 조건
if (temp > 85 and temp < 95) and vibration < 0.6:
    print("조건3 충족: 안정 운영 구간")
```

**결과 해설**: `and`는 두 조건이 모두 참일 때, `or`는 하나라도 참일 때 전체가 참이 됨. 괄호를 사용하여 복잡한 조건을 그룹화할 수 있음.

#### 4-4. 반복문 기본

```python
temperatures = [82, 85, 88, 95, 84]

# 기본 반복
for temp in temperatures:
    print(f"온도: {temp}도")

print()

# enumerate 사용 (인덱스와 값 동시 접근)
for i, temp in enumerate(temperatures):
    print(f"{i+1}번째 측정: {temp}도")

print()

# range 사용
print("0부터 4까지:")
for i in range(5):
    print(f"  {i}")
```

**결과 해설**: `for` 문은 리스트의 각 요소를 순차적으로 처리함. `enumerate()`를 사용하면 인덱스와 값을 동시에 얻을 수 있음. `range(5)`는 0부터 4까지의 숫자를 생성함.

#### 4-5. 조건과 반복 결합

```python
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
```

**결과 해설**: 반복문 안에서 조건문을 사용하여 각 데이터를 판정함. 9시부터 시간별로 온도를 확인하고, 90도 초과 시 경고를 출력함. `warning_count += 1`은 경고 횟수를 1씩 증가시킴.

#### 4-6. 리스트 컴프리헨션

```python
# 제곱 계산 - 일반 방식
squares = []
for x in range(5):
    squares.append(x ** 2)
print(f"제곱 (일반): {squares}")

# 제곱 계산 - 리스트 컴프리헨션
squares = [x ** 2 for x in range(5)]
print(f"제곱 (컴프리헨션): {squares}")

# 조건 필터링
temps = [82, 85, 88, 95, 84, 91, 86]
high_temps = [t for t in temps if t > 90]
print(f"90도 초과: {high_temps}")

# 변환 + 조건
fahrenheit = [t * 9/5 + 32 for t in temps if t > 85]
print(f"85도 초과 화씨 변환: {fahrenheit}")
```

**결과 해설**: 리스트 컴프리헨션은 반복문을 한 줄로 표현하는 Python 문법임. `[표현식 for 변수 in 리스트 if 조건]` 형태로 사용함. 코드가 간결해지고 가독성이 높아짐.

---

## Part 5: 종합 실습

### 실습 코드

#### 5-1. 설비 상태 관리 (딕셔너리 활용)

```python
# 중첩 딕셔너리 (설비 관리)
equipment = {
    "EQ-001": {"온도": 82, "진동": 0.3, "압력": 100, "상태": "정상"},
    "EQ-002": {"온도": 91, "진동": 0.8, "압력": 95,  "상태": "주의"},
    "EQ-003": {"온도": 85, "진동": 0.4, "압력": 102, "상태": "정상"},
    "EQ-004": {"온도": 96, "진동": 1.2, "압력": 88,  "상태": "경고"}
}

print("=== 설비 상태 점검 ===")
for eq_id, data in equipment.items():
    print(f"{eq_id}: 온도 {data['온도']}도, 진동 {data['진동']}mm/s, 상태 {data['상태']}")

# 이상 설비 필터링
abnormal = [eq for eq, data in equipment.items() if data["상태"] != "정상"]
print(f"\n이상 설비: {abnormal}")

# 통계 계산
temps = [data["온도"] for data in equipment.values()]
print(f"평균 온도: {sum(temps)/len(temps):.1f}도")
print(f"최고 온도: {max(temps)}도")
```

**결과 해설**: 중첩 딕셔너리로 여러 설비의 데이터를 구조화함. `.items()`로 키-값 쌍을 순회하고, 리스트 컴프리헨션으로 이상 설비를 필터링함. `.values()`로 모든 값을 가져와 통계를 계산함.

#### 5-2. 함수 정의

```python
def calculate_defect_rate(total, defects):
    """불량률 계산 함수"""
    if total == 0:
        return 0
    return defects / total


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


def check_temperature(temp, threshold=90):
    """온도 상태 확인"""
    if temp > threshold:
        return "경고", True
    else:
        return "정상", False


# 함수 사용 예시
rate = calculate_defect_rate(1200, 30)
grade, message = get_quality_grade(rate)
temp_status, is_warning = check_temperature(92)

print(f"불량률: {rate:.1%}")
print(f"등급: {grade} ({message})")
print(f"온도 상태: {temp_status}")
```

**결과 해설**: `def` 키워드로 함수를 정의함. 함수는 재사용 가능한 코드 블록임. `"""..."""`는 독스트링으로 함수 설명을 작성함. `return`으로 결과를 반환하며, 여러 값을 튜플로 반환할 수 있음.

#### 5-3. 일일 품질 보고서

```python
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
print("=" * 50)
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
print(f"평균 온도: {production_data['평균온도']:>10.1f}C")
print(f"최고 온도: {production_data['최고온도']:>10}C ({temp_status})")
print(f"평균 습도: {production_data['평균습도']:>10}%")
print("=" * 50)
```

**결과 해설**: 딕셔너리에 저장된 생산 데이터를 활용하여 품질 보고서를 생성함. f-string의 정렬 옵션(`:>10`)으로 깔끔한 출력 형식을 만듦. 삼항 연산자(`"정상" if 조건 else "주의"`)로 간결하게 조건 처리함.

#### 5-4. 다중 라인 분석

```python
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

    print(f"{line_id:<10} {data['생산량']:>8,} {rate:>7.1%} {grade:>6} {data['온도']:>5}C{temp_mark}")

    total_production += data["생산량"]
    total_defects += data["불량"]

    if rate < best_rate:
        best_rate = rate
        best_line = line_id

print("-" * 42)
overall_rate = total_defects / total_production
print(f"{'전체':>10} {total_production:>8,} {overall_rate:>7.1%}")
print(f"\n최우수 라인: {best_line} (불량률 {best_rate:.1%})")
print("* 표시: 온도 90C 초과")
```

**결과 해설**: 여러 라인의 데이터를 순회하며 분석함. 누적 합계를 계산하고, 최우수 라인을 찾음. `_`는 사용하지 않는 반환값을 무시할 때 관례적으로 사용하는 변수명임.

---

## 자주 하는 실수와 해결 방법

### 초보자 주의사항

```python
# 1. 들여쓰기 오류
if True:
print("에러!")      # IndentationError - 들여쓰기 필요

# 2. 콜론(:) 누락
if True             # SyntaxError - 콜론 필요
    print("에러!")

# 3. = vs == 혼동
x = 10
if x = 10:          # 에러! (= 는 할당)
if x == 10:         # 정답! (== 는 비교)

# 4. 인덱스 범위 초과
data = [1, 2, 3]
print(data[3])      # IndexError (0,1,2만 존재)

# 5. 문자열 따옴표 불일치
name = "홍길동'     # SyntaxError - 따옴표 종류 일치 필요
```

### 디버깅 팁

1. **에러 메시지 읽기**: Python이 알려주는 에러 유형과 위치를 확인함
2. **print로 확인**: 중간 값을 출력하여 문제 지점을 찾음
3. **검색하기**: 에러 메시지를 구글에 검색하면 Stack Overflow에서 해결책을 찾을 수 있음

---

## 핵심 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    2차시 핵심 요약                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Python 개발 환경                                         │
│     - Anaconda: 패키지 관리 및 가상환경                      │
│     - Jupyter Notebook: 대화형 코드 실행 환경                │
│     - 핵심 단축키: Shift+Enter (실행)                        │
│                                                              │
│  2. 5가지 자료형                                             │
│     - int: 정수 (1000, 30)                                   │
│     - float: 실수 (85.7, 0.025)                              │
│     - str: 문자열 ("정상", "LINE-01")                        │
│     - list: 리스트 [1, 2, 3]                                 │
│     - dict: 딕셔너리 {"키": 값}                              │
│                                                              │
│  3. f-string 포맷팅                                          │
│     - 기본: f'{변수}'                                        │
│     - 소수점: f'{값:.2f}'                                    │
│     - 퍼센트: f'{값:.1%}'                                    │
│     - 콤마: f'{값:,}'                                        │
│                                                              │
│  4. 제어문                                                   │
│     - if/elif/else: 조건 분기                                │
│     - for: 반복 처리                                         │
│     - enumerate: 인덱스와 값 동시 접근                       │
│                                                              │
│  5. 함수 정의                                                │
│     - def 함수명(매개변수):                                  │
│     - return 반환값                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 다음 차시 예고

### 3차시: 제조 데이터 다루기 기초

- **NumPy**: 수치 계산의 핵심 도구
- **Pandas**: 표 형태 데이터 다루기
- CSV 파일 불러오기 및 기본 조작

```python
import pandas as pd
df = pd.read_csv('sensor_data.csv')
print(df.head())        # 처음 5행 보기
print(df.describe())    # 기본 통계
```
