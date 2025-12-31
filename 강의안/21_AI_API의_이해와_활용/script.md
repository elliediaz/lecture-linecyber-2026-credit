# [21차시] AI API의 이해와 활용 - 강사 스크립트

## 강의 정보
- **차시**: 21차시 (25분)
- **유형**: 이론 + 실습
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 Part IV 시작 [1.5분]

> 안녕하세요, 21차시를 시작하겠습니다.
>
> 오늘부터 **Part IV: AI 활용**이 시작됩니다! 지금까지 데이터 분석, 머신러닝, 딥러닝을 배웠는데요, 이제 이것들을 **실무에서 활용**하는 방법을 배웁니다.
>
> 오늘은 **API**의 개념과 외부 AI 서비스를 활용하는 방법을 다룹니다.

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, API의 개념을 이해합니다.
> 둘째, REST API로 데이터를 주고받습니다.
> 셋째, AI API를 활용합니다.

---

## 전개 (19분)

### 섹션 1: API 개념 (5min)

#### API란 [2min]

> **API(Application Programming Interface)**는 프로그램 간에 데이터를 주고받는 약속입니다.
>
> 비유하자면 레스토랑의 **웨이터**예요. 손님(프로그램)이 메뉴(요청)를 웨이터(API)에게 주면, 웨이터가 주방(서버)에 전달하고, 음식(응답)을 가져다줍니다.
>
> 우리가 직접 주방에 들어갈 필요 없이, 웨이터를 통해 주문하고 음식을 받죠.

#### REST API [3min]

> **REST API**는 가장 많이 사용되는 API 방식입니다. HTTP 프로토콜을 사용해요.
>
> 주요 메서드:
> - **GET**: 데이터 가져오기 (조회)
> - **POST**: 데이터 보내기 (생성)
> - **PUT**: 데이터 수정
> - **DELETE**: 데이터 삭제
>
> AI API는 주로 **POST**를 사용해서 데이터를 보내고 결과를 받습니다.

---

### 섹션 2: Python으로 API 호출 (7min)

#### requests 라이브러리 [3min]

> *(코드 시연)*
>
> ```python
> import requests
>
> # GET 요청
> response = requests.get('https://api.example.com/data')
> print(response.status_code)  # 200이면 성공
> print(response.json())
> ```
>
> **requests** 라이브러리로 API를 호출합니다. response.json()으로 JSON 응답을 파이썬 딕셔너리로 변환해요.

#### POST 요청 [2min]

> *(코드 시연)*
>
> ```python
> data = {'temperature': 85, 'humidity': 50}
>
> response = requests.post(
>     'https://api.example.com/predict',
>     json=data,
>     headers={'Authorization': 'Bearer YOUR_KEY'}
> )
>
> result = response.json()
> print(result)  # {'prediction': 0, 'confidence': 0.95}
> ```
>
> POST 요청으로 데이터를 보내고 예측 결과를 받습니다.

#### 에러 처리 [2min]

> API 호출은 실패할 수 있어요. 네트워크 문제, 잘못된 요청 등.
>
> ```python
> try:
>     response = requests.get(url, timeout=10)
>     response.raise_for_status()
> except requests.exceptions.RequestException as e:
>     print(f"에러: {e}")
> ```
>
> try-except로 에러를 처리하고, timeout으로 대기 시간을 제한합니다.

---

### 섹션 3: AI API 활용 (5min)

#### AI API 종류 [2min]

> 다양한 클라우드 AI 서비스가 있어요.
>
> **이미지**: Google Vision, AWS Rekognition
> **텍스트**: OpenAI GPT, Google NLP
> **음성**: Google Speech-to-Text
>
> 이런 서비스들은 복잡한 모델을 직접 만들지 않아도 API 호출로 AI 기능을 사용할 수 있게 해줍니다.

#### API Key 보안 [2min]

> API Key는 **절대 코드에 직접 작성하면 안 돼요!**
>
> ```python
> # 잘못된 방법
> API_KEY = 'sk-1234...'  # 코드에 노출!
>
> # 올바른 방법
> import os
> API_KEY = os.environ.get('MY_API_KEY')
> ```
>
> 환경 변수나 .env 파일을 사용해서 분리하세요. 코드가 공개되면 API Key가 유출됩니다.

#### 비용 주의 [1min]

> 클라우드 AI API는 **사용량에 따라 비용**이 발생해요. 무료 할당량이 있는 경우도 있지만, 초과하면 과금됩니다.
>
> 테스트할 때는 소량으로, 사용량 제한을 설정해두는 것이 좋습니다.

---

### 섹션 4: 우리만의 API 만들기 소개 (2min)

> 다음 차시들에서 **직접 API를 만들어볼 거예요**.
>
> - 22차시: Streamlit으로 웹앱 만들기
> - 23차시: FastAPI로 예측 API 만들기
>
> 학습한 모델을 다른 사람이 사용할 수 있게 서비스로 만드는 겁니다!

---

## 정리 (3분)

### 핵심 내용 요약 [1.5min]

> 오늘 배운 핵심 내용을 정리하면:
>
> 1. **API**: 프로그램 간 데이터 교환 인터페이스
> 2. **REST API**: HTTP 기반, GET/POST/PUT/DELETE
> 3. **requests**: Python HTTP 라이브러리
> 4. **JSON**: 데이터 교환 형식
> 5. **API Key**: 환경 변수로 관리 (보안!)
>
> 외부 AI 서비스를 활용하면 복잡한 모델 없이도 AI 기능을 사용할 수 있어요!

### 다음 차시 예고 [1min]

> 다음 22차시에서는 **Streamlit**을 배웁니다.
>
> 코드 몇 줄로 웹 애플리케이션을 만들 수 있어요. 모델 예측 결과를 시각화하는 대시보드를 만들어봅니다.

### 마무리 인사 [0.5분]

> API의 세계에 입문했습니다. 수고하셨습니다!

---

## 강의 노트

### 예상 질문
1. "API Key가 뭔가요?"
   → 서비스 이용을 위한 인증 키. 비밀번호 같은 거

2. "무료 API도 있나요?"
   → 공공데이터포털, 일부 서비스의 무료 할당량 등

3. "POST와 GET 차이가 뭔가요?"
   → GET은 데이터 조회, POST는 데이터 전송/생성

### 시간 조절 팁
- 시간 부족: AI API 종류 간략히
- 시간 여유: 실제 공공 API 호출 실습
