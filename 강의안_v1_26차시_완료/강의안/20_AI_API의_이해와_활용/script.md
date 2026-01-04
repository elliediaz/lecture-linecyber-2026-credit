# [20차시] AI API의 이해와 활용 - 강사 스크립트

## 강의 정보
- **차시**: 20차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 20차시를 시작하겠습니다.
>
> 지난 시간에 Keras로 딥러닝 모델을 만들었습니다. MLP로 제조 품질을 예측했죠.
>
> 오늘은 조금 다른 접근입니다. 직접 모델을 만드는 게 아니라 **만들어진 AI를 API로 활용**하는 방법을 배웁니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, API의 개념을 이해합니다.
> 둘째, AI API를 호출하는 방법을 익힙니다.
> 셋째, 외부 AI 서비스를 활용합니다.

---

### 핵심 내용 (8분)

#### API란? [2min]

> **API**는 Application Programming Interface의 약자예요.
>
> 쉽게 말하면 프로그램과 프로그램이 **소통하는 방법**입니다.
>
> 우리가 Python 코드에서 외부 AI 서버로 데이터를 보내면, 서버가 처리해서 결과를 보내줘요.
>
> 음식점 비유를 하면, 메뉴판(API 문서)을 보고 주문(요청)하면 음식(결과)이 나오는 거예요.

#### AI API의 장점 [1.5min]

> 직접 AI 모델을 개발하려면 데이터 수집, 정제, 학습, GPU 인프라가 필요해요. 시간도 오래 걸리죠.
>
> AI API를 쓰면 이 모든 게 필요 없어요. **즉시 사용 가능**하고, 검증된 고성능 모델을 쓸 수 있어요.
>
> 복잡한 AI를 몇 줄 코드로 활용할 수 있습니다.

#### requests 라이브러리 [2min]

> Python에서 API를 호출할 때 **requests** 라이브러리를 씁니다.
>
> ```python
> import requests
>
> response = requests.get('https://api.example.com/data')
> response = requests.post('https://api.example.com/predict', json=data)
> ```
>
> **GET**은 데이터를 조회할 때, **POST**는 데이터를 보내서 처리할 때 씁니다.
>
> AI 예측 요청은 보통 POST를 써요. 입력 데이터를 보내야 하니까요.

#### 응답 처리 [1.5min]

> API 응답에서 중요한 건 **상태 코드**와 **데이터**예요.
>
> 상태 코드 200은 성공, 400번대는 클라이언트 오류, 500번대는 서버 오류입니다.
>
> ```python
> if response.status_code == 200:
>     data = response.json()  # JSON으로 파싱
> ```
>
> 대부분의 API는 JSON 형식으로 응답해요.

#### API 키 관리 [1min]

> AI API는 보통 **API 키**로 인증합니다.
>
> 중요한 건 API 키를 코드에 직접 쓰면 안 된다는 거예요. 보안 문제가 생겨요.
>
> **환경 변수**로 관리하세요.
>
> ```python
> import os
> API_KEY = os.environ.get('MY_API_KEY')
> ```

---

## 실습편 (15-20분)

### 실습 소개 [2min]

> 이제 실습 시간입니다. requests로 API를 호출해봅니다.
>
> **실습 목표**입니다.
> 1. requests로 GET/POST 요청합니다.
> 2. JSON 응답을 처리합니다.
> 3. 오류 처리를 합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import requests
> import json
> ```

### 실습 1: GET 요청 [2min]

> 첫 번째 실습입니다. 공개 API를 호출합니다.
>
> JSONPlaceholder라는 테스트용 API를 씁니다. 가입 없이 사용 가능해요.
>
> ```python
> response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
> print(response.json())
> ```

### 실습 2: POST 요청 [2min]

> 두 번째 실습입니다. 데이터를 전송합니다.
>
> ```python
> data = {'title': '테스트', 'body': '내용'}
> response = requests.post(url, json=data)
> ```
>
> json 파라미터를 쓰면 자동으로 JSON 형식으로 변환해줘요.

### 실습 3: 응답 처리 [2min]

> 세 번째 실습입니다. 응답을 파싱합니다.
>
> ```python
> data = response.json()
> title = data['title']
> ```
>
> 딕셔너리처럼 접근하면 됩니다.

### 실습 4: 오류 처리 [3min]

> 네 번째 실습입니다. 중요한 부분이에요.
>
> 네트워크 오류, 타임아웃, 서버 오류 등 여러 문제가 생길 수 있어요.
>
> ```python
> try:
>     response = requests.get(url, timeout=5)
>     response.raise_for_status()
> except requests.exceptions.Timeout:
>     print("타임아웃!")
> ```
>
> try-except로 안전하게 처리하세요.

### 실습 5: 헤더와 인증 [2min]

> 다섯 번째 실습입니다. API 키를 전송합니다.
>
> ```python
> headers = {
>     'Authorization': f'Bearer {API_KEY}'
> }
> response = requests.get(url, headers=headers)
> ```
>
> Authorization 헤더에 API 키를 넣는 게 가장 흔한 방식이에요.

### 실습 6: 환경 변수 [2min]

> 여섯 번째 실습입니다. API 키를 안전하게 관리합니다.
>
> ```python
> import os
> API_KEY = os.environ.get('MY_API_KEY')
> ```
>
> .env 파일을 쓰려면 python-dotenv 라이브러리를 설치하면 돼요.

### 실습 7: API 클라이언트 클래스 [3min]

> 마지막 실습입니다. 재사용 가능한 클래스를 만듭니다.
>
> ```python
> class AIClient:
>     def __init__(self, base_url, api_key):
>         self.base_url = base_url
>         self.headers = {'Authorization': f'Bearer {api_key}'}
>
>     def predict(self, data):
>         response = requests.post(
>             f'{self.base_url}/predict',
>             headers=self.headers,
>             json=data
>         )
>         return response.json()
> ```
>
> 이렇게 만들면 여러 곳에서 편하게 재사용할 수 있어요.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **API**는 프로그램 간 소통 방법이에요. 요청을 보내고 응답을 받습니다.
>
> **requests** 라이브러리로 Python에서 API를 호출해요. GET은 조회, POST는 전송/처리입니다.
>
> **응답 처리**는 status_code 확인하고 response.json()으로 파싱합니다.
>
> **API 키**는 환경 변수로 안전하게 관리하세요.

#### 다음 차시 예고 [1min]

> 다음 21차시에서는 **LLM API와 프롬프트 작성법**을 배웁니다.
>
> ChatGPT, Claude 같은 대규모 언어 모델 API를 사용하는 방법을 알아봅니다. 프롬프트를 잘 작성하는 기법도 배워요.

#### 마무리 [0.5min]

> 외부 AI 서비스를 코드로 활용하는 법을 배웠습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- 인터넷 연결 확인

### 주의사항
- API 키 보안 강조
- 공개 API로 실습 (가입 불필요)
- 오류 처리 중요성

### 예상 질문
1. "API 키를 어디서 받나요?"
   → 각 서비스 웹사이트에서 가입 후 발급. OpenAI, Claude 등

2. "requests 설치가 안 돼요"
   → pip install requests

3. "응답이 안 와요"
   → 인터넷 연결 확인, URL 오타 확인, 타임아웃 설정

4. "API 호출 비용이 있나요?"
   → 서비스마다 다름. 대부분 무료 티어 있음. 문서 확인 필요
