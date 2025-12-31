---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 21차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# AI API의 이해와 활용

## 21차시 | AI 기초체력훈련 (Pre AI-Campus)

**Part IV 시작: 외부 AI 서비스 활용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **API**의 개념을 이해한다
2. **REST API**로 데이터를 주고받는다
3. **AI API**를 활용한다

---

# Part IV 시작!

## AI 활용의 마지막 단계

```
Part I: Python 기초 ✅
Part II: 통계/전처리 ✅
Part III: 머신러닝/딥러닝 ✅
Part IV: AI 활용 ← 지금!
  - 21차시: AI API 활용
  - 22차시: Streamlit 웹앱
  - 23차시: FastAPI 모델 서빙
  - 24차시: 특성 중요도 분석
  - 25차시: 모델 저장과 배포
```

---

# API란?

## Application Programming Interface

> 프로그램 간에 **데이터를 주고받는 약속**

```
  내 프로그램              외부 서비스
       │                      │
       │  ─────요청───────→    │
       │     (Request)         │
       │                       │
       │  ←────응답─────────   │
       │     (Response)        │
```

### 비유: 레스토랑의 웨이터
- 손님(프로그램)이 메뉴(요청)를 주문
- 웨이터(API)가 주방(서버)에 전달
- 음식(응답)을 손님에게 전달

---

# REST API

## 가장 많이 사용되는 API 방식

### HTTP 메서드
| 메서드 | 설명 | 예시 |
|--------|------|------|
| GET | 데이터 조회 | 정보 가져오기 |
| POST | 데이터 생성 | 새 데이터 보내기 |
| PUT | 데이터 수정 | 기존 데이터 변경 |
| DELETE | 데이터 삭제 | 데이터 삭제 |

> AI API는 주로 **POST**를 사용

---

# Python requests 라이브러리

## API 호출하기

```python
import requests

# GET 요청
response = requests.get('https://api.example.com/data')
print(response.status_code)  # 200 = 성공
print(response.json())       # JSON 응답 파싱

# POST 요청
data = {'text': '분석할 텍스트'}
response = requests.post(
    'https://api.example.com/analyze',
    json=data
)
result = response.json()
```

---

# JSON 형식

## 데이터 교환의 표준

```json
{
    "name": "품질 예측 모델",
    "version": "1.0",
    "prediction": {
        "result": "정상",
        "confidence": 0.95
    },
    "features": [85, 50, 100, 1.0]
}
```

### Python에서 JSON 다루기
```python
import json

# 딕셔너리 → JSON 문자열
json_str = json.dumps(data)

# JSON 문자열 → 딕셔너리
data = json.loads(json_str)
```

---

# 공공 API 활용

## 무료로 사용 가능한 API들

### 날씨 API (OpenWeatherMap)
```python
API_KEY = 'your_api_key'
city = 'Seoul'
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}'

response = requests.get(url)
weather = response.json()
print(f"온도: {weather['main']['temp']}K")
```

### 다른 유용한 API
- 공공데이터포털 (data.go.kr)
- 네이버 API
- 카카오 API

---

# AI API 종류

## 클라우드 AI 서비스

### 이미지/비전
- Google Vision API
- AWS Rekognition
- Azure Computer Vision

### 자연어 처리
- OpenAI API (GPT)
- Google Natural Language
- Naver Clova

### 음성
- Google Speech-to-Text
- AWS Transcribe

---

# AI API 활용 예시

## 텍스트 분석

```python
import requests

# 감성 분석 API 호출 (예시)
response = requests.post(
    'https://api.example.com/sentiment',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={'text': '이 제품 정말 좋아요!'}
)

result = response.json()
print(result)
# {'sentiment': 'positive', 'score': 0.92}
```

---

# API Key 관리

## 보안 주의!

### 잘못된 방법 ❌
```python
API_KEY = 'sk-1234567890abcdef'  # 코드에 직접 작성
```

### 올바른 방법 ✅
```python
import os

# 환경 변수에서 읽기
API_KEY = os.environ.get('MY_API_KEY')

# 또는 .env 파일 사용
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('MY_API_KEY')
```

---

# 에러 처리

## API 호출 시 주의사항

```python
import requests

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # 에러면 예외 발생
    data = response.json()

except requests.exceptions.Timeout:
    print("시간 초과!")

except requests.exceptions.HTTPError as e:
    print(f"HTTP 에러: {e}")

except requests.exceptions.RequestException as e:
    print(f"요청 실패: {e}")
```

---

# 직접 API 만들기

## 다음 차시 미리보기

```python
# FastAPI로 예측 API 만들기
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

@app.post('/predict')
def predict(data: dict):
    features = [data['temp'], data['humidity']]
    prediction = model.predict([features])[0]
    return {'prediction': int(prediction)}
```

> 22차시에서 Streamlit, 23차시에서 FastAPI를 배웁니다!

---

# 실습 정리

## API 사용 기본 패턴

```python
import requests

# 1. URL과 헤더 설정
url = 'https://api.example.com/endpoint'
headers = {'Authorization': f'Bearer {API_KEY}'}

# 2. 요청 데이터 준비
data = {'key': 'value'}

# 3. API 호출
response = requests.post(url, headers=headers, json=data)

# 4. 응답 처리
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f"에러: {response.status_code}")
```

---

# 정리

## 핵심 개념

| 개념 | 설명 |
|------|------|
| API | 프로그램 간 데이터 교환 인터페이스 |
| REST API | HTTP 기반 API 방식 |
| requests | Python HTTP 라이브러리 |
| JSON | 데이터 교환 형식 |
| API Key | 인증 키 (보안 주의!) |

---

# 다음 차시 예고

## 22차시: Streamlit으로 웹앱 만들기

- Streamlit 소개
- 대화형 웹앱 구축
- 모델 예측 UI 만들기

> 코드 몇 줄로 **웹 애플리케이션**을 만듭니다!

---

# 감사합니다

## AI 기초체력훈련 21차시

**AI API의 이해와 활용**

외부 AI 서비스를 활용하는 법을 배웠습니다!
