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

---

# 왜 Python인가?

## AI 개발의 표준 언어

| 장점 | 설명 |
|------|------|
| 쉬운 문법 | 영어처럼 읽히는 직관적 코드 |
| 풍부한 라이브러리 | NumPy, Pandas, TensorFlow 등 |
| 거대한 커뮤니티 | 검색하면 답이 바로 나옴 |
| 빠른 개발 | 아이디어를 빠르게 구현 |

> **Python은 AI/데이터 분석 분야에서 90% 이상 사용**

---

# 개발 환경 선택

## Anaconda 추천 이유

| 일반 Python | Anaconda |
|-------------|----------|
| 패키지 충돌 문제 | 데이터 과학 패키지 포함 |
| 환경 관리 어려움 | 가상환경으로 프로젝트 분리 |
| 수동 설치 필요 | Jupyter Notebook 기본 포함 |

### 설치 방법
1. https://www.anaconda.com/download 접속
2. Windows 64-bit 버전 다운로드
3. 설치 파일 실행 (약 10분 소요)

---

# Jupyter Notebook

## 코드 작성과 실행을 한 곳에서

```
┌─────────────────────────────────────────┐
│ [File] [Edit] [View] [Cell]             │ ← 메뉴바
├─────────────────────────────────────────┤
│ In [1]: print("제조 AI 시작!")          │ ← 코드 셀
├─────────────────────────────────────────┤
│ 제조 AI 시작!                           │ ← 출력 결과
└─────────────────────────────────────────┘
```

### 핵심 단축키
- `Shift + Enter`: 셀 실행
- `Esc + A`: 위에 새 셀 추가
- `Esc + B`: 아래에 새 셀 추가

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

# 숫자 다루기

## 정수(int)와 실수(float)

```python
# 제조 현장 데이터 예시
production = 1200       # 일 생산량
defect_rate = 0.025     # 불량률 2.5%
temperature = 85.7      # 공정 온도

# 기본 계산
total = production * 5          # 5일 생산량: 6000
defect_count = production * defect_rate  # 불량 수: 30.0

# 출력
print(f"5일 총 생산량: {total}개")
print(f"예상 불량 수: {defect_count}개")
```

---

# 문자열 다루기

## 텍스트 데이터

```python
# 기본 문자열
product_name = "센서 모듈 A"
line_id = "LINE-01"

# 문자열 연결
full_name = product_name + " / " + line_id

# f-string으로 데이터 삽입 (매우 중요!)
temp = 85.7
rate = 0.025
message = f"온도: {temp}도, 불량률: {rate:.1%}"
# 결과: "온도: 85.7도, 불량률: 2.5%"
```

---

# 리스트 (목록)

## 여러 데이터를 순서대로 저장

```python
# 일주일 생산량 데이터
weekly_production = [1200, 1150, 1300, 1180, 1250]

# 값 접근 (0번부터 시작!)
first_day = weekly_production[0]    # 1200
last_day = weekly_production[-1]    # 1250

# 값 추가
weekly_production.append(1280)

# 기본 통계
total = sum(weekly_production)
average = total / len(weekly_production)
```

---

# 딕셔너리 (사전)

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
temp = sensor_data["온도"]        # 85.2
status = sensor_data["상태"]      # "정상"

# 값 추가/수정
sensor_data["진동"] = 0.42        # 새 항목 추가
sensor_data["온도"] = 86.0        # 값 수정
```

---

# 이론 정리

## 핵심 포인트

### Python 환경
- Anaconda로 패키지 관리
- Jupyter Notebook으로 코드 실행

### 5가지 자료형
- **정수/실수**: 숫자 데이터
- **문자열**: 텍스트 데이터
- **리스트**: 순서 있는 데이터 모음
- **딕셔너리**: 이름표 있는 데이터 모음

---

# - 실습편 -

## 2차시

**Python 기초 실습**

---

# 실습 개요

## 제조 현장 데이터 다루기

### 실습 목표
- Jupyter Notebook에서 Python 코드 실행
- 제조 데이터를 변수에 저장하고 계산
- 조건문과 반복문으로 데이터 처리

### 실습 환경
```bash
# Jupyter Notebook 실행
jupyter notebook
```

---

# 실습 1: 변수와 계산

## 생산량 데이터 다루기

```python
# 센서 측정값 입력
temperature = 85.2      # 온도
pressure = 101.3        # 압력
humidity = 45           # 습도

# 계산 예시: 일 생산량과 불량률
daily_production = 1200
defect_rate = 0.025

# 불량 수 계산
defect_count = daily_production * defect_rate
good_count = daily_production - defect_count

print(f"일 생산량: {daily_production}개")
print(f"양품: {good_count}개, 불량: {defect_count}개")
```

---

# 실습 2: 조건문

## 품질 등급 판정

```python
# 불량률에 따른 품질 등급 판정
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
    message = "개선 필요"

print(f"품질 등급: {grade} ({message})")
```

---

# 실습 3: 반복문 for

## 센서 데이터 분석

```python
# 온도 측정값 (1시간 간격)
temperatures = [82, 85, 88, 95, 84, 91, 86]
threshold = 90  # 경고 기준

print("=== 온도 모니터링 ===")
for i, temp in enumerate(temperatures):
    if temp > threshold:
        print(f"[경고] {i+1}시: {temp}도 - 기준 초과!")
    else:
        print(f"[정상] {i+1}시: {temp}도")

# 평균 온도 계산
average = sum(temperatures) / len(temperatures)
print(f"\n평균 온도: {average:.1f}도")
```

---

# 실습 4: 딕셔너리 활용

## 설비 상태 관리

```python
# 설비별 센서 데이터
equipment = {
    "EQ-001": {"온도": 82, "진동": 0.3, "상태": "정상"},
    "EQ-002": {"온도": 91, "진동": 0.8, "상태": "주의"},
    "EQ-003": {"온도": 85, "진동": 0.4, "상태": "정상"}
}

# 설비별 상태 확인
print("=== 설비 상태 점검 ===")
for eq_id, data in equipment.items():
    temp = data["온도"]
    status = data["상태"]
    print(f"{eq_id}: 온도 {temp}도, 상태 {status}")
```

---

# 실습 5: 종합 예제

## 일일 품질 보고서 생성

```python
# 일일 생산 데이터
production_data = {
    "라인": "LINE-01",
    "생산량": 1200,
    "양품": 1170,
    "불량": 30,
    "평균온도": 85.2
}

# 불량률 계산
defect_rate = production_data["불량"] / production_data["생산량"]

# 보고서 출력
print("=" * 40)
print("       일일 품질 보고서")
print("=" * 40)
print(f"라인: {production_data['라인']}")
print(f"생산량: {production_data['생산량']}개")
print(f"양품률: {(1-defect_rate)*100:.1f}%")
print(f"불량률: {defect_rate*100:.1f}%")
print("=" * 40)
```

---

# 자주 하는 실수

## 주의할 점

```python
# 1. 들여쓰기 오류
if True:
print("에러!")  # IndentationError

# 2. 콜론(:) 누락
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

# 실습 정리

## 핵심 체크포인트

### 환경 설정
- [ ] Anaconda 설치 완료
- [ ] Jupyter Notebook 실행 확인

### Python 기초
- [ ] 변수에 데이터 저장
- [ ] 조건문으로 분기 처리
- [ ] 반복문으로 여러 데이터 처리
- [ ] 딕셔너리로 구조화된 데이터 관리

---

# 다음 차시 예고

## 3차시: 제조 데이터 다루기 기초

### 학습 내용
- **NumPy**: 수치 계산의 핵심 도구
- **Pandas**: 표 형태 데이터 다루기
- CSV 파일 불러오기 및 기본 조작

### 준비물
- Anaconda 설치 완료
- 오늘 실습 코드 복습

---

# 정리 및 Q&A

## 오늘의 핵심

1. **환경 설정**: Anaconda + Jupyter Notebook
2. **자료형**: 정수, 실수, 문자열, 리스트, 딕셔너리
3. **제어문**: if 조건문, for 반복문

### 문제 해결
- 설치 오류: 관리자 권한으로 실행
- 한글 깨짐: UTF-8 인코딩 확인

---

# 감사합니다

## 2차시: Python 시작하기

**다음 시간에 제조 데이터를 다뤄봅시다!**
