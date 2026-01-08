# [25차시] AI API의 이해와 활용 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 25차시 |
| 주제 | AI API의 이해와 활용 |
| 시간 | 30분 (이론 15분 + 실습 13분 + 정리 2분) |
| 학습 목표 | REST API 개념, requests 라이브러리, JSON 처리 |

---

## 학습 목표

1. REST API의 개념과 구조를 이해한다
2. requests 라이브러리로 API를 호출한다
3. JSON 데이터를 처리한다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | REST API 개념 |
| 대주제 2 | 5분 | requests 라이브러리 |
| 대주제 3 | 5분 | JSON 데이터 처리 |
| 실습 | 11분 | API 호출 실습 |
| 정리 | 2분 | 요약 및 다음 차시 예고 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 모델을 저장하고 Pipeline을 구성하는 방법을 배웠습니다."

> "저장된 모델을 실제로 서비스하려면 어떻게 해야 할까요? API로 만들어서 다른 프로그램이 호출할 수 있게 해야 합니다."

> "오늘은 API가 뭔지, 어떻게 호출하는지 기초를 배웁니다."

---

### 대주제 1: REST API 개념 (5분)

#### 슬라이드 4-6: API란

> "API는 프로그램 간에 통신하는 방법입니다. 쉽게 말해 '다른 프로그램의 기능을 빌려 쓰는 방법'이에요."

> "여러분이 날씨 앱을 만든다고 해보세요. 직접 기상 관측을 할 순 없죠. 기상청 API를 호출해서 날씨 정보를 가져오는 겁니다."

> "마찬가지로 품질 예측 모델을 API로 만들면, 다른 시스템에서 호출해서 예측 결과를 받을 수 있어요."

---

#### 슬라이드 7-9: REST API

> "REST API는 웹에서 가장 많이 쓰는 API 방식입니다. HTTP 프로토콜을 사용해요."

> "핵심은 HTTP 메서드입니다. GET은 데이터 조회, POST는 데이터 생성이에요."

> "머신러닝 예측 요청은 대부분 POST를 씁니다. 입력 데이터를 보내고 예측 결과를 받으니까요."

```
POST /predict
Body: {"temperature": 200, "pressure": 50}

Response: {"prediction": "normal"}
```

---

#### 슬라이드 10-12: 상태 코드와 인증

> "API 응답에는 상태 코드가 있습니다. 200은 성공, 400은 잘못된 요청, 500은 서버 에러."

> "404 에러 많이 보셨죠? 찾는 페이지가 없다는 뜻이에요. API도 마찬가지입니다."

> "대부분의 API는 인증이 필요합니다. API 키를 헤더에 넣어서 보내죠."

---

### 대주제 2: requests 라이브러리 (5분)

#### 슬라이드 13-15: requests 소개

> "Python에서 API를 호출할 때 requests 라이브러리를 씁니다. 가장 인기 있는 HTTP 라이브러리예요."

```python
import requests

response = requests.get('https://api.example.com/data')
response = requests.post('https://api.example.com/predict', json=data)
```

> "get은 조회, post는 데이터 전송이에요. json 파라미터로 데이터를 쉽게 보낼 수 있습니다."

---

#### 슬라이드 16-18: 요청과 응답

> "요청을 보내면 response 객체가 돌아옵니다."

```python
response = requests.get(url)
print(response.status_code)  # 200
print(response.json())       # JSON을 딕셔너리로 변환
```

> "status_code로 성공 여부를 확인하고, json()으로 응답 데이터를 딕셔너리로 받습니다."

> "타임아웃은 꼭 설정하세요. 안 그러면 서버가 응답 안 할 때 무한 대기합니다."

```python
response = requests.get(url, timeout=5)  # 5초 타임아웃
```

---

#### 슬라이드 19-21: 에러 처리

> "API 호출은 항상 실패할 수 있어요. 네트워크 문제, 서버 문제 등등."

```python
try:
    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()  # 에러면 예외 발생
    result = response.json()
except requests.RequestException as e:
    print(f"API 호출 실패: {e}")
```

> "try-except로 감싸고, raise_for_status()를 호출하면 에러를 쉽게 잡을 수 있습니다."

---

### 대주제 3: JSON 데이터 처리 (5분)

#### 슬라이드 22-24: JSON 기초

> "JSON은 데이터 교환 형식입니다. API 통신의 표준이에요."

> "Python 딕셔너리와 거의 같은 구조입니다. 중괄호로 객체, 대괄호로 배열을 표현해요."

```python
import json

# 딕셔너리 → JSON 문자열
data = {'temperature': 200, 'status': True}
json_str = json.dumps(data)

# JSON 문자열 → 딕셔너리
data = json.loads(json_str)
```

---

#### 슬라이드 25-27: requests와 JSON

> "requests 라이브러리를 쓰면 JSON 변환이 자동입니다."

```python
# 요청 시 json 파라미터 사용
response = requests.post(url, json=data)
# 자동으로 json.dumps() 해서 보냄

# 응답 시 json() 메서드 사용
result = response.json()
# 자동으로 json.loads() 해서 반환
```

> "파일에 저장할 때는 json.dump()와 json.load()를 씁니다. s가 없으면 파일용이에요."

---

### 실습편 (11분)

#### 슬라이드 28-30: 공개 API 호출

```python
import requests

# httpbin.org - 테스트용 공개 API
url = 'https://httpbin.org/get'
response = requests.get(url)

print(f"상태: {response.status_code}")
print(f"응답: {response.json()}")
```

> "httpbin.org는 HTTP 테스트용 공개 사이트입니다. 보낸 요청을 그대로 돌려줘요."

---

#### 슬라이드 31-33: POST 요청 실습

```python
url = 'https://httpbin.org/post'

# 품질 예측 요청 시뮬레이션
data = {
    'temperature': 210,
    'pressure': 55,
    'speed': 105
}

response = requests.post(url, json=data)
print(response.json()['json'])  # 보낸 데이터 확인
```

---

#### 슬라이드 34-36: 예측 API 시뮬레이션

```python
def simulate_prediction_api(input_data):
    """로컬 예측 시뮬레이션"""
    # 실제로는 requests.post()로 API 호출

    # 간단한 규칙 기반 예측
    if input_data['temperature'] > 250:
        return {'prediction': 'defect', 'confidence': 0.9}
    else:
        return {'prediction': 'normal', 'confidence': 0.85}

result = simulate_prediction_api({'temperature': 200})
print(result)
```

---

#### 슬라이드 37-39: JSON 파일 처리

```python
import json

# 설정 파일 저장
config = {
    'api_url': 'https://api.example.com/predict',
    'timeout': 10,
    'features': ['temperature', 'pressure', 'speed']
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 설정 파일 로드
with open('config.json', 'r') as f:
    loaded_config = json.load(f)
```

---

### 정리 (2분)

#### 슬라이드 40-41: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**REST API**는 HTTP 기반 통신 방식입니다. GET은 조회, POST는 데이터 전송."

> "**requests**로 Python에서 API를 호출합니다. json 파라미터로 데이터 전송, json() 메서드로 응답 파싱."

> "**JSON**은 API 통신의 표준 형식입니다. dumps/loads로 변환하고, requests에서는 자동 처리됩니다."

---

#### 슬라이드 42-43: 다음 차시 예고

> "다음 시간에는 LLM API를 배웁니다. OpenAI나 Claude API로 대화형 AI를 활용하는 방법을 다룹니다."

> "오늘 수업 마무리합니다. 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: GET과 POST 중 언제 뭘 써야 하나요?

> "데이터를 가져올 때는 GET, 데이터를 보내거나 생성할 때는 POST입니다. ML 예측처럼 입력 데이터를 보내야 하면 POST를 씁니다."

### Q2: API 키는 어디서 받나요?

> "각 API 서비스 사이트에서 회원가입 후 발급받습니다. 대부분 무료 티어가 있어요."

### Q3: requests 말고 다른 라이브러리도 있나요?

> "httpx는 async 지원이 좋고, aiohttp는 비동기 전용입니다. 하지만 입문자는 requests로 충분합니다."

### Q4: JSON과 XML 차이는 뭔가요?

> "둘 다 데이터 교환 형식입니다. JSON이 더 가볍고 읽기 쉬워서 요즘은 대부분 JSON을 씁니다."

---

## 참고 자료

### 공식 문서
- [requests 문서](https://requests.readthedocs.io/)
- [JSON 공식 사이트](https://www.json.org/)
- [httpbin.org](https://httpbin.org/) - 테스트용 API

### 관련 차시
- 23차시: 모델 저장과 실무 배포 준비
- 25차시: LLM API와 프롬프트 작성법

---

## 체크리스트

수업 전:
- [ ] requests 설치 확인
- [ ] 인터넷 연결 확인
- [ ] httpbin.org 접속 테스트

수업 중:
- [ ] HTTP 메서드 설명
- [ ] GET/POST 차이 강조
- [ ] 에러 처리 중요성 설명

수업 후:
- [ ] 실습 코드 배포
- [ ] API 키 관리 주의사항 안내

