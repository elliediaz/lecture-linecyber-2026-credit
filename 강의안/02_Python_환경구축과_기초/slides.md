---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 2차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section {
    font-family: 'Malgun Gothic', sans-serif;
  }
  h1 {
    color: #2563eb;
  }
  h2 {
    color: #1e40af;
  }
  code {
    background-color: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
  }
  pre {
    background-color: #1e293b;
    color: #e2e8f0;
  }
---

# Python 환경 구축과 기초

## 2차시 | AI 기초체력훈련 (Pre AI-Campus)

**AI 개발을 위한 첫 번째 도구 준비**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **Anaconda와 Jupyter Notebook**을 설치하고 실행한다
2. **Python 기본 자료형**(정수, 실수, 문자열, 리스트, 딕셔너리)을 이해한다
3. **조건문과 반복문**을 활용한 간단한 프로그램을 작성한다

---

# 왜 Python인가?

## AI 개발의 표준 언어

| 장점 | 설명 |
|------|------|
| 쉬운 문법 | 영어처럼 읽히는 직관적 코드 |
| 풍부한 라이브러리 | NumPy, Pandas, TensorFlow 등 |
| 거대한 커뮤니티 | 질문하면 답이 금방 나옴 |
| 빠른 프로토타이핑 | 아이디어를 빠르게 구현 |

> **Python은 AI/ML 분야에서 90% 이상 점유율**

---

# 개발 환경 선택

## Anaconda를 추천하는 이유

```
일반 Python 설치:
- 패키지 충돌 문제
- 환경 관리 어려움
- 수동 설치 필요

Anaconda 설치:
- 데이터 과학 패키지 포함 (NumPy, Pandas 등)
- 가상환경으로 프로젝트 분리
- Jupyter Notebook 기본 포함
```

---

# Anaconda 설치

## 설치 순서

### 1단계: 다운로드
- https://www.anaconda.com/download 접속
- Windows 64-bit 버전 다운로드 (약 800MB)

### 2단계: 설치
- 설치 파일 실행
- "Add to PATH" 옵션 체크 (권장)
- 설치 완료까지 약 10분 소요

---

# Anaconda 설치

## 설치 확인

### 명령 프롬프트(CMD)에서 확인
```bash
# Anaconda 버전 확인
conda --version

# Python 버전 확인
python --version
```

### 예상 출력
```
conda 24.x.x
Python 3.11.x
```

---

# 가상환경 생성

## 왜 가상환경이 필요한가?

```
프로젝트 A: TensorFlow 2.10 필요
프로젝트 B: TensorFlow 2.15 필요

→ 같은 컴퓨터에서 두 버전을 동시에?
→ 가상환경으로 해결!
```

### 가상환경 생성 명령어
```bash
# ai_training이라는 이름의 환경 생성
conda create -n ai_training python=3.11

# 환경 활성화
conda activate ai_training
```

---

# Jupyter Notebook

## Jupyter란?

- **Ju**lia + **Pyt**hon + **R** = Jupyter
- 웹 브라우저에서 코드 작성 및 실행
- 코드와 결과를 함께 문서화
- 데이터 분석/AI 개발의 표준 도구

### 실행 방법
```bash
# 가상환경 활성화 후
jupyter notebook
```

---

# Jupyter Notebook 화면 구성

## 주요 인터페이스

```
┌─────────────────────────────────────────┐
│ [File] [Edit] [View] [Insert] [Cell]    │  ← 메뉴바
├─────────────────────────────────────────┤
│ In [1]: print("Hello, AI!")             │  ← 코드 셀
├─────────────────────────────────────────┤
│ Hello, AI!                              │  ← 출력 결과
├─────────────────────────────────────────┤
│ In [2]: █                               │  ← 새 셀
└─────────────────────────────────────────┘
```

---

# Jupyter 핵심 단축키

## 꼭 알아야 할 단축키

| 단축키 | 기능 |
|--------|------|
| `Shift + Enter` | 셀 실행 후 다음 셀로 이동 |
| `Ctrl + Enter` | 셀 실행 (현재 셀 유지) |
| `Esc + A` | 위에 새 셀 추가 |
| `Esc + B` | 아래에 새 셀 추가 |
| `Esc + DD` | 현재 셀 삭제 |
| `Esc + M` | 마크다운 셀로 변환 |

---

# Python 자료형 개요

## 기본 자료형 5가지

```python
# 1. 정수 (int)
count = 100

# 2. 실수 (float)
temperature = 36.5

# 3. 문자열 (str)
product_name = "반도체 A형"

# 4. 리스트 (list)
defect_rates = [0.02, 0.015, 0.03]

# 5. 딕셔너리 (dict)
sensor_data = {"온도": 25, "습도": 60}
```

---

# 정수와 실수

## int와 float

```python
# 정수: 소수점 없는 숫자
production_count = 1000
line_number = 3

# 실수: 소수점 있는 숫자
defect_rate = 0.025
temperature = 85.7

# 사칙연산
total = 100 + 50      # 150
average = 300 / 4     # 75.0
remainder = 10 % 3    # 1 (나머지)
power = 2 ** 10       # 1024 (거듭제곱)
```

---

# 문자열

## str (String)

```python
# 문자열 생성
product = "센서 모듈 A"
message = '불량률 0.02%'

# 문자열 연결
full_name = "라인" + "01"  # "라인01"

# 문자열 포매팅 (f-string) - 매우 중요!
line = 3
rate = 0.025
result = f"라인 {line}번 불량률: {rate:.2%}"
# "라인 3번 불량률: 2.50%"
```

---

# 리스트

## list - 순서가 있는 데이터 모음

```python
# 리스트 생성
daily_production = [1200, 1150, 1300, 1180, 1250]

# 인덱싱 (0부터 시작!)
first_day = daily_production[0]   # 1200
last_day = daily_production[-1]   # 1250

# 슬라이싱
first_three = daily_production[0:3]  # [1200, 1150, 1300]

# 리스트 수정
daily_production.append(1280)  # 끝에 추가
daily_production[0] = 1210     # 값 변경
```

---

# 딕셔너리

## dict - 키-값 쌍으로 데이터 저장

```python
# 딕셔너리 생성
sensor = {
    "온도": 85.2,
    "습도": 45,
    "압력": 1.2,
    "상태": "정상"
}

# 값 접근
temp = sensor["온도"]       # 85.2
status = sensor.get("상태") # "정상"

# 값 추가/수정
sensor["진동"] = 0.02       # 새 키 추가
sensor["온도"] = 86.0       # 값 수정
```

---

# 자료형 변환

## 타입 캐스팅 (Type Casting)

```python
# 문자열 → 숫자
user_input = "1000"
count = int(user_input)      # 1000 (정수)
rate = float("0.025")        # 0.025 (실수)

# 숫자 → 문자열
production = 1200
message = "생산량: " + str(production) + "개"

# 리스트 ↔ 기타
data = list(range(5))        # [0, 1, 2, 3, 4]
```

---

# 조건문 if

## 기본 구조

```python
defect_rate = 0.03

if defect_rate > 0.05:
    print("경고: 불량률이 높습니다!")
    print("라인 점검이 필요합니다.")
elif defect_rate > 0.02:
    print("주의: 불량률을 모니터링하세요.")
else:
    print("정상: 품질 기준 충족")
```

### 핵심 포인트
- 들여쓰기(4칸 스페이스)가 **코드 블록**을 결정
- 콜론(`:`) 잊지 말기!

---

# 비교 연산자

## 조건문에서 사용하는 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `==` | 같다 | `x == 10` |
| `!=` | 다르다 | `x != 0` |
| `>` | 크다 | `rate > 0.05` |
| `<` | 작다 | `temp < 80` |
| `>=` | 크거나 같다 | `count >= 100` |
| `<=` | 작거나 같다 | `score <= 60` |

---

# 논리 연산자

## 여러 조건 조합하기

```python
temperature = 85
humidity = 55

# and: 두 조건 모두 만족
if temperature > 80 and humidity > 50:
    print("경고: 온습도 모두 높음")

# or: 하나라도 만족
if temperature > 90 or humidity > 70:
    print("경고: 즉시 점검 필요")

# not: 조건 반전
if not (temperature > 90):
    print("온도 정상 범위")
```

---

# 반복문 for

## 기본 구조

```python
# 리스트 순회
production = [1200, 1150, 1300]

for daily in production:
    print(f"생산량: {daily}개")

# range() 활용
for i in range(5):
    print(f"라인 {i+1} 점검 중...")

# 출력:
# 라인 1 점검 중...
# 라인 2 점검 중...
# ...
```

---

# 반복문 while

## 조건이 참인 동안 반복

```python
# 품질 검사 시뮬레이션
defect_count = 0
check_count = 0

while defect_count < 3:
    check_count += 1
    # 임의로 불량 발생 가정
    if check_count % 10 == 0:
        defect_count += 1
        print(f"불량 발견! 누적: {defect_count}개")

print(f"총 {check_count}개 검사 완료")
```

### 주의: 무한 루프 조심!

---

# 실습 예제 1

## 품질 판정 프로그램

```python
# 불량률 기준 품질 등급 판정
defect_rate = 0.025  # 2.5%

if defect_rate <= 0.01:
    grade = "A"
    message = "우수"
elif defect_rate <= 0.03:
    grade = "B"
    message = "양호"
elif defect_rate <= 0.05:
    grade = "C"
    message = "주의"
else:
    grade = "D"
    message = "불량"

print(f"품질 등급: {grade} ({message})")
```

---

# 실습 예제 2

## 일일 생산량 분석

```python
# 5일간 생산량 데이터
production = [1200, 1150, 1300, 1180, 1250]

# 기본 통계 계산
total = sum(production)
average = total / len(production)
max_value = max(production)
min_value = min(production)

print(f"총 생산량: {total}개")
print(f"일평균: {average:.1f}개")
print(f"최대: {max_value}개, 최소: {min_value}개")
```

---

# 실습 예제 3

## 센서 데이터 이상 탐지

```python
# 센서 측정값 (온도)
temperatures = [82, 85, 88, 95, 84, 91, 86]
threshold = 90  # 경고 기준

print("=== 온도 이상 탐지 ===")
for i, temp in enumerate(temperatures):
    if temp > threshold:
        print(f"[경고] {i+1}번째 측정: {temp}°C")
    else:
        print(f"[정상] {i+1}번째 측정: {temp}°C")
```

---

# 자주 하는 실수

## 주의할 점

```python
# 1. 들여쓰기 오류
if True:
print("에러!")  # IndentationError

# 2. 콜론 누락
if True
    print("에러!")  # SyntaxError

# 3. = vs == 혼동
if x = 10:   # 할당 (에러)
if x == 10:  # 비교 (정답)

# 4. 인덱스 범위 초과
data = [1, 2, 3]
print(data[3])  # IndexError (0, 1, 2만 존재)
```

---

# 학습 정리

## 오늘 배운 내용

### 1. 개발 환경
- Anaconda 설치 및 가상환경 생성
- Jupyter Notebook 사용법

### 2. Python 자료형
- int, float, str, list, dict

### 3. 제어문
- 조건문: if, elif, else
- 반복문: for, while

---

# 다음 차시 예고

## 3차시: 데이터 다루기 기초

- **NumPy**: 수치 계산의 핵심 라이브러리
- **Pandas**: 데이터프레임으로 표 형태 데이터 다루기
- 제조 데이터 불러오기 및 기본 조작

### 과제 (선택)
- Anaconda, Jupyter Notebook 설치 완료
- 오늘 배운 코드 직접 실행해보기

---

# Q&A

## 질문이 있으신가요?

### 설치 문제 해결
- Anaconda 설치 오류: 관리자 권한으로 실행
- Jupyter 실행 안됨: `pip install notebook` 시도
- 한글 깨짐: 파일 저장 시 UTF-8 인코딩 확인

### 추가 학습 자료
- Python 공식 튜토리얼: https://docs.python.org/ko/3/tutorial/
- Anaconda 공식 문서: https://docs.anaconda.com/

---

# 감사합니다

## AI 기초체력훈련 2차시

**Python 환경 구축과 기초**

다음 시간에 만나요!
