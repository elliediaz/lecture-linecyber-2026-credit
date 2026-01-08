---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [25차시] AI API의 이해와 활용

## 외부 서비스와 데이터 주고받기

---

# 학습 목표

1. **REST API**의 개념과 구조를 이해한다
2. **requests** 라이브러리로 API를 호출한다
3. **JSON** 데이터를 처리한다

---

# 지난 시간 복습

- **joblib**: 모델 저장/로드
- **Pipeline**: 전처리 + 모델 통합
- **배포 체크리스트**: 모델, 데이터, 환경, 테스트, 문서

**오늘**: 저장된 모델을 API로 서비스하기 위한 기초

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | REST API 개념 |
| 대주제 2 | 10분 | requests 라이브러리 |
| 대주제 3 | 8분 | JSON 데이터 처리 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## REST API 개념과 구조

---

# API란?

**API (Application Programming Interface)**

- 프로그램 간 통신 방법을 정의한 규약
- "다른 프로그램의 기능을 빌려 쓰는 방법"

```
내 프로그램  ←→  API  ←→  외부 서비스
  (클라이언트)         (서버)
```

---

# 일상 속 API 예시

| 서비스 | API 활용 |
|-------|---------|
| **날씨 앱** | 기상청 API로 날씨 데이터 조회 |
| **지도 앱** | 구글/네이버 지도 API |
| **결제 시스템** | 카드사 결제 API |
| **AI 서비스** | OpenAI, Claude API |

---

# REST API란?

**REST (Representational State Transfer)**

- 웹에서 자원을 다루는 설계 원칙
- HTTP 프로토콜 기반
- URL로 자원 식별, HTTP 메서드로 동작 정의

```
https://api.example.com/products/123
       └─ 서버 주소 ─┘  └─ 자원 ─┘
```

---

# HTTP 메서드

| 메서드 | 동작 | 설명 |
|-------|-----|------|
| **GET** | 조회 | 데이터 가져오기 |
| **POST** | 생성 | 새 데이터 만들기 |
| **PUT** | 수정 | 데이터 전체 수정 |
| **DELETE** | 삭제 | 데이터 삭제 |

**ML 예측 요청**: 대부분 POST 사용

---

# REST API 구성 요소

```
[요청 (Request)]
- URL: https://api.example.com/predict
- Method: POST
- Headers: Content-Type: application/json
- Body: {"temperature": 200, "pressure": 50}

[응답 (Response)]
- Status Code: 200 OK
- Body: {"prediction": "normal", "confidence": 0.95}
```

---

# HTTP 상태 코드

| 코드 | 의미 | 설명 |
|-----|------|------|
| **200** | OK | 성공 |
| **201** | Created | 생성 성공 |
| **400** | Bad Request | 잘못된 요청 |
| **401** | Unauthorized | 인증 필요 |
| **404** | Not Found | 자원 없음 |
| **500** | Server Error | 서버 오류 |

---

# API 인증 방식

| 방식 | 특징 |
|-----|------|
| **API Key** | 간단, 헤더에 키 포함 |
| **Bearer Token** | OAuth 토큰 방식 |
| **Basic Auth** | ID/Password 인코딩 |

```
Headers: {
    "Authorization": "Bearer sk-xxxxxx"
}
```

---

# 제조업 API 활용 예시

| 용도 | API |
|-----|-----|
| **품질 예측** | 자체 ML 모델 API |
| **설비 모니터링** | IoT 플랫폼 API |
| **이상 분석** | LLM API로 원인 분석 |
| **보고서 생성** | 문서 생성 API |

---

<!-- _class: lead -->
# 대주제 2
## requests 라이브러리

---

# requests 라이브러리

**Python에서 가장 많이 쓰는 HTTP 라이브러리**

```bash
pip install requests
```

```python
import requests

# GET 요청
response = requests.get('https://api.example.com/data')

# POST 요청
response = requests.post('https://api.example.com/predict',
                         json={'temperature': 200})
```

---

# GET 요청 기본

```python
import requests

# 공개 API 호출
url = 'https://httpbin.org/get'
response = requests.get(url)

# 응답 확인
print(f"상태 코드: {response.status_code}")
print(f"응답 본문: {response.text}")
```

---

# GET 요청 - 파라미터

```python
# URL 파라미터 전달
url = 'https://api.example.com/search'
params = {
    'query': 'temperature',
    'limit': 10
}

response = requests.get(url, params=params)
# 실제 URL: https://api.example.com/search?query=temperature&limit=10
```

---

# POST 요청 기본

```python
import requests

url = 'https://httpbin.org/post'

# JSON 데이터 전송
data = {
    'temperature': 200,
    'pressure': 50,
    'speed': 100
}

response = requests.post(url, json=data)
print(response.json())
```

---

# POST 요청 - 헤더 추가

```python
url = 'https://api.example.com/predict'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your-api-key'
}

data = {'temperature': 200, 'pressure': 50}

response = requests.post(url, json=data, headers=headers)
```

---

# 응답 객체 속성

```python
response = requests.get(url)

# 주요 속성
response.status_code    # 상태 코드 (200, 404 등)
response.text          # 응답 본문 (문자열)
response.json()        # JSON을 딕셔너리로 변환
response.headers       # 응답 헤더
response.elapsed       # 응답 시간
response.ok            # 성공 여부 (200~299면 True)
```

---

# 응답 확인 패턴

```python
response = requests.get(url)

# 방법 1: ok 속성
if response.ok:
    data = response.json()
else:
    print(f"에러: {response.status_code}")

# 방법 2: raise_for_status()
try:
    response.raise_for_status()  # 에러면 예외 발생
    data = response.json()
except requests.HTTPError as e:
    print(f"HTTP 에러: {e}")
```

---

# 타임아웃 설정

```python
try:
    # 5초 안에 응답 없으면 타임아웃
    response = requests.get(url, timeout=5)
except requests.Timeout:
    print("요청 시간 초과!")
except requests.ConnectionError:
    print("연결 실패!")
```

**실무에서는 항상 타임아웃 설정 권장**

---

# 세션 사용하기

```python
# 세션 생성 (연결 재사용, 성능 향상)
session = requests.Session()

# 공통 헤더 설정
session.headers.update({
    'Authorization': 'Bearer your-api-key'
})

# 여러 요청에 재사용
response1 = session.get(url1)
response2 = session.post(url2, json=data)

session.close()
```

---

# 실습: 공개 API 호출

```python
import requests

# JSONPlaceholder (테스트용 공개 API)
url = 'https://jsonplaceholder.typicode.com/posts/1'

response = requests.get(url)
print(f"상태: {response.status_code}")

data = response.json()
print(f"제목: {data['title']}")
print(f"내용: {data['body'][:50]}...")
```

---

<!-- _class: lead -->
# 대주제 3
## JSON 데이터 처리

---

# JSON이란?

**JSON (JavaScript Object Notation)**

- 데이터 교환 형식
- 사람이 읽기 쉽고, 기계가 파싱하기 쉬움
- API 통신의 표준 형식

```json
{
    "temperature": 200,
    "pressure": 50,
    "sensors": ["temp_01", "press_01"]
}
```

---

# Python과 JSON 대응

| Python | JSON |
|--------|------|
| dict | object { } |
| list | array [ ] |
| str | string |
| int, float | number |
| True, False | true, false |
| None | null |

---

# json 모듈 기본

```python
import json

# Python → JSON 문자열 (직렬화)
data = {'temperature': 200, 'status': True}
json_str = json.dumps(data)
print(json_str)  # '{"temperature": 200, "status": true}'

# JSON 문자열 → Python (역직렬화)
json_str = '{"temperature": 200, "status": true}'
data = json.loads(json_str)
print(data['temperature'])  # 200
```

---

# JSON 파일 읽기/쓰기

```python
import json

# 파일에 쓰기
data = {'model': 'RandomForest', 'accuracy': 0.92}
with open('config.json', 'w') as f:
    json.dump(data, f, indent=2)

# 파일에서 읽기
with open('config.json', 'r') as f:
    loaded = json.load(f)
print(loaded)
```

---

# 한글 처리

```python
import json

# 한글 포함 데이터
data = {'설비명': '프레스 1호기', '상태': '정상'}

# ensure_ascii=False로 한글 유지
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)
# {
#   "설비명": "프레스 1호기",
#   "상태": "정상"
# }
```

---

# requests와 JSON

```python
import requests

# POST 요청 시 json 파라미터 사용
response = requests.post(url, json=data)
# 자동으로 Content-Type: application/json 설정
# 자동으로 json.dumps() 처리

# 응답 JSON 파싱
result = response.json()  # json.loads() 자동 처리
print(result['prediction'])
```

---

# 중첩 JSON 다루기

```python
# 중첩된 JSON 구조
data = {
    "request_id": "req_001",
    "input": {
        "sensors": {
            "temperature": 200,
            "pressure": 50
        },
        "metadata": {
            "machine_id": "M001",
            "timestamp": "2026-01-08T10:00:00"
        }
    }
}

# 접근
temp = data['input']['sensors']['temperature']
```

---

# JSON 스키마 검증 (간단 버전)

```python
def validate_prediction_input(data):
    """예측 입력 JSON 검증"""
    required = ['temperature', 'pressure', 'speed']

    # 필수 필드 확인
    for field in required:
        if field not in data:
            return False, f"Missing: {field}"

    # 타입 확인
    for field in required:
        if not isinstance(data[field], (int, float)):
            return False, f"Invalid type: {field}"

    return True, "Valid"

# 사용
is_valid, msg = validate_prediction_input({'temperature': 200})
print(f"{is_valid}: {msg}")  # False: Missing: pressure
```

---

# API 응답 처리 패턴

```python
def call_prediction_api(data):
    """예측 API 호출"""
    url = 'https://api.example.com/predict'

    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()

        result = response.json()

        # 응답 구조 확인
        if 'prediction' in result:
            return {
                'success': True,
                'prediction': result['prediction'],
                'confidence': result.get('confidence', None)
            }
        else:
            return {'success': False, 'error': 'Invalid response'}

    except requests.RequestException as e:
        return {'success': False, 'error': str(e)}
```

---

# 실습: 예측 요청 시뮬레이션

```python
import json

# 요청 데이터 생성
request_data = {
    "temperature": 210,
    "pressure": 55,
    "speed": 105,
    "humidity": 60,
    "vibration": 5.5
}

# JSON 변환 (API 전송 시뮬레이션)
json_request = json.dumps(request_data, indent=2)
print("=== 요청 ===")
print(json_request)

# 응답 시뮬레이션
response_data = {
    "prediction": "normal",
    "defect_probability": 0.15,
    "confidence": 0.85
}
print("\n=== 응답 ===")
print(json.dumps(response_data, indent=2))
```

---

# 에러 응답 처리

```python
# 에러 응답 예시
error_response = {
    "error": {
        "code": "INVALID_INPUT",
        "message": "Temperature out of range",
        "details": {
            "field": "temperature",
            "received": 500,
            "allowed_range": [100, 300]
        }
    }
}

# 에러 처리
if 'error' in error_response:
    error = error_response['error']
    print(f"에러 코드: {error['code']}")
    print(f"메시지: {error['message']}")
```

---

# 대용량 JSON 처리

```python
# 대용량 데이터를 스트리밍으로 처리
import json

def process_large_json(filepath):
    """대용량 JSON 라인별 처리"""
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # 처리 로직
                yield record

# JSON Lines 형식 (각 줄이 JSON 객체)
# {"id": 1, "value": 100}
# {"id": 2, "value": 200}
```

---

# 실무 팁

| 상황 | 권장 방법 |
|-----|----------|
| API 키 관리 | 환경 변수 사용 |
| 타임아웃 | 항상 설정 (5-30초) |
| 재시도 | 실패 시 지수 백오프 |
| 로깅 | 요청/응답 기록 |
| 테스트 | Mock 서버 활용 |

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **REST API 개념**
   - HTTP 메서드 (GET, POST, PUT, DELETE)
   - 상태 코드, 인증 방식

2. **requests 라이브러리**
   - get(), post() 요청
   - 응답 처리, 타임아웃

3. **JSON 처리**
   - json.dumps(), json.loads()
   - requests와 연동

---

# 핵심 코드

```python
import requests
import json

# GET 요청
response = requests.get(url, params={'key': 'value'})

# POST 요청
response = requests.post(url, json=data, headers=headers)

# 응답 처리
if response.ok:
    result = response.json()

# JSON 변환
json_str = json.dumps(data, ensure_ascii=False)
data = json.loads(json_str)
```

---

# 체크리스트

- [ ] REST API 개념 이해
- [ ] HTTP 메서드 구분
- [ ] requests.get() 사용
- [ ] requests.post() 사용
- [ ] JSON 직렬화/역직렬화
- [ ] 에러 처리 구현

---

# 다음 차시 예고

## [25차시] LLM API와 프롬프트 작성법

- OpenAI / Claude API 사용법
- 프롬프트 엔지니어링 기초
- 제조업 활용 예시

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습: 공개 API 호출 및 JSON 처리

