---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 20차시'
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

# AI API의 이해와 활용

## 20차시 | Part IV. AI 서비스화와 활용

**만들어진 AI를 API로 활용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **API**의 개념을 이해한다
2. **AI API**를 호출하는 방법을 익힌다
3. 외부 AI 서비스를 **활용**한다

---

# API란?

## Application Programming Interface

> 프로그램과 프로그램이 **소통하는 방법**

```
     우리 프로그램          외부 서비스
    ┌───────────┐         ┌───────────┐
    │           │  요청   │           │
    │  Python   │ ──────→ │  AI 서버  │
    │  코드     │ ←────── │           │
    │           │  응답   │           │
    └───────────┘         └───────────┘
```

> 데이터를 보내면 결과를 받아오는 **약속**

---

# 웹 API 작동 방식

## HTTP 요청/응답

```python
# 요청 (Request)
POST https://api.example.com/predict
{
    "temperature": 85,
    "humidity": 50
}

# 응답 (Response)
{
    "defect_probability": 0.15,
    "status": "normal"
}
```

> JSON 형식으로 데이터 주고받기

---

# AI API의 장점

## 왜 API를 사용하나?

### 직접 개발
- 데이터 수집/정제 필요
- 모델 학습 시간 필요
- GPU 등 인프라 필요

### AI API 사용
- 즉시 사용 가능
- 검증된 고성능 모델
- 유지보수 불필요

> 복잡한 AI를 **간단한 코드**로 활용!

---

# AI API 종류

## 다양한 AI 서비스

| 분야 | 예시 서비스 |
|------|------------|
| 이미지 | 객체 인식, OCR, 얼굴 인식 |
| 텍스트 | 번역, 감정 분석, 요약 |
| 음성 | 음성 인식, TTS |
| 생성 AI | ChatGPT, Claude |

### 제조 분야
- 비전 검사 API
- 예측 유지보수 API
- 문서 OCR API

---

# requests 라이브러리

## Python HTTP 요청

```python
import requests

# GET 요청
response = requests.get('https://api.example.com/data')

# POST 요청
response = requests.post(
    'https://api.example.com/predict',
    json={'temperature': 85, 'humidity': 50}
)

# 응답 확인
print(response.status_code)  # 200
print(response.json())       # 결과 딕셔너리
```

---

# HTTP 메서드

## GET vs POST

| 메서드 | 용도 | 데이터 전송 |
|--------|------|-------------|
| GET | 데이터 조회 | URL 파라미터 |
| POST | 데이터 전송/처리 | 요청 본문 (Body) |

```python
# GET: 데이터 조회
requests.get('https://api.example.com/status')

# POST: 예측 요청
requests.post('https://api.example.com/predict',
              json={'data': [1, 2, 3]})
```

---

# 응답 상태 코드

## HTTP Status Code

| 코드 | 의미 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 401 | 인증 실패 |
| 404 | 찾을 수 없음 |
| 500 | 서버 오류 |

```python
if response.status_code == 200:
    print("성공!")
else:
    print(f"오류: {response.status_code}")
```

---

# API 키 인증

## 보안 인증

```python
import requests

# API 키를 헤더에 포함
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.post(
    'https://api.example.com/predict',
    headers=headers,
    json={'data': [85, 50, 100]}
)
```

> API 키는 **절대 코드에 직접 쓰지 말 것!**

---

# 환경 변수로 API 키 관리

## 보안 Best Practice

```python
import os

# 환경 변수에서 API 키 읽기
API_KEY = os.environ.get('MY_API_KEY')

headers = {
    'Authorization': f'Bearer {API_KEY}'
}
```

```bash
# 터미널에서 설정
export MY_API_KEY="your-secret-key"
```

> .env 파일 사용도 추천

---

# 이론 정리

## AI API 핵심

| 개념 | 설명 |
|------|------|
| API | 프로그램 간 소통 방법 |
| requests | Python HTTP 라이브러리 |
| GET | 데이터 조회 |
| POST | 데이터 전송/처리 |
| 상태 코드 | 200=성공, 400=오류 |
| API 키 | 인증용 비밀 키 |

---

# - 실습편 -

## 20차시

**AI API 호출 실습**

---

# 실습 개요

## API 호출 연습

### 목표
- requests로 API 호출
- JSON 데이터 처리
- 공개 API 활용

### 실습 환경
```python
import requests
import json
import os
```

---

# 실습 1: 기본 GET 요청

## 공개 API 호출

```python
import requests

# 공개 API (JSONPlaceholder)
url = 'https://jsonplaceholder.typicode.com/posts/1'

response = requests.get(url)

print(f"상태 코드: {response.status_code}")
print(f"응답 데이터:")
print(response.json())
```

---

# 실습 2: POST 요청

## 데이터 전송

```python
url = 'https://jsonplaceholder.typicode.com/posts'

# 전송할 데이터
data = {
    'title': '불량 예측 결과',
    'body': '온도 85, 습도 50에서 불량 확률 15%',
    'userId': 1
}

response = requests.post(url, json=data)

print(f"상태 코드: {response.status_code}")
print(f"생성된 데이터: {response.json()}")
```

---

# 실습 3: 응답 처리

## JSON 파싱

```python
# 응답 JSON 파싱
data = response.json()

# 개별 값 접근
title = data['title']
body = data['body']

print(f"제목: {title}")
print(f"내용: {body}")

# 타입 확인
print(f"응답 타입: {type(data)}")  # dict
```

---

# 실습 4: 오류 처리

## try-except

```python
import requests

def call_api(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 오류 시 예외 발생
        return response.json()
    except requests.exceptions.Timeout:
        print("타임아웃 발생")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 오류: {e}")
    except Exception as e:
        print(f"오류: {e}")
    return None
```

---

# 실습 5: 헤더와 인증

## Authorization 헤더

```python
# API 키 설정 (실습용)
API_KEY = "demo-key-12345"

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

response = requests.get(
    'https://api.example.com/data',
    headers=headers
)
```

---

# 실습 6: 날씨 API 예시

## 실제 API 활용

```python
# OpenWeatherMap 예시 (API 키 필요)
API_KEY = os.environ.get('WEATHER_API_KEY')
city = 'Seoul'

url = f'https://api.openweathermap.org/data/2.5/weather'
params = {
    'q': city,
    'appid': API_KEY,
    'units': 'metric'
}

response = requests.get(url, params=params)
data = response.json()

print(f"도시: {data['name']}")
print(f"온도: {data['main']['temp']}°C")
```

---

# 실습 7: 로컬 AI 모델 API화

## Flask 간단 예시

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# 모델 로드
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000)
```

---

# 실습 8: API 클라이언트 클래스

## 재사용 가능한 코드

```python
class AIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}

    def predict(self, data):
        response = requests.post(
            f'{self.base_url}/predict',
            headers=self.headers,
            json=data
        )
        return response.json()

# 사용
client = AIClient('https://api.example.com', 'key')
result = client.predict({'temp': 85})
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] requests.get() / requests.post()
- [ ] response.status_code 확인
- [ ] response.json() 파싱
- [ ] headers로 API 키 전송
- [ ] try-except 오류 처리
- [ ] 환경 변수로 키 관리

---

# 다음 차시 예고

## 21차시: LLM API와 프롬프트 작성법

### 학습 내용
- ChatGPT, Claude API 사용법
- 프롬프트 엔지니어링
- 제조 현장 활용 예시

> 대규모 언어 모델 **API 활용**!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **API**: 프로그램 간 소통 약속
2. **requests**: Python HTTP 라이브러리
3. **GET/POST**: 조회 vs 전송
4. **JSON**: 데이터 교환 형식
5. **API 키**: 인증, 환경 변수로 관리
6. **오류 처리**: try-except 필수

---

# 감사합니다

## 20차시: AI API의 이해와 활용

**외부 AI 서비스를 코드로 활용하는 법을 배웠습니다!**
