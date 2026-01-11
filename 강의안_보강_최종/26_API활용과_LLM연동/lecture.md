# 26차시: API 활용과 LLM 연동

## 학습 목표

1. **API**의 개념과 **REST API** 통신 방식을 이해함
2. **requests** 라이브러리로 외부 데이터를 호출함
3. **LLM API**를 활용하여 제조 데이터 분석을 보조함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 8분 | API와 REST API 개념 |
| 대주제 2 | 8분 | LLM API 호출과 응답 처리 |
| 대주제 3 | 7분 | 프롬프트 작성과 제조업 활용 |
| 정리 | 2분 | 핵심 요약 |

---

## 대주제 1: API와 REST API 개념

### 1.1 API란?

**API (Application Programming Interface)**
- 프로그램 간 통신 방법을 정의한 규약
- "다른 프로그램의 기능을 빌려 쓰는 방법"

```mermaid
flowchart LR
    A[내 프로그램<br/>클라이언트] <--> B[API] <--> C[외부 서비스<br/>서버]
```

**일상 속 API 예시**

| 서비스 | API 활용 |
|-------|---------|
| 날씨 앱 | 기상청 API로 날씨 데이터 조회 |
| 지도 앱 | 구글/네이버 지도 API |
| AI 서비스 | OpenAI, Claude API |

---

### 1.2 REST API란?

**REST (Representational State Transfer)**
- 웹에서 자원을 다루는 설계 원칙
- HTTP 프로토콜 기반
- URL로 자원 식별, HTTP 메서드로 동작 정의

```
https://api.example.com/products/123
       [서버 주소]        [자원]
```

**HTTP 메서드**

| 메서드 | 동작 | 설명 |
|-------|-----|------|
| **GET** | 조회 | 데이터 가져오기 |
| **POST** | 생성 | 새 데이터 만들기 |
| **PUT** | 수정 | 데이터 전체 수정 |
| **DELETE** | 삭제 | 데이터 삭제 |

ML 예측 요청은 대부분 **POST** 사용

---

### 1.3 REST API 구성 요소

```mermaid
sequenceDiagram
    participant C as 클라이언트
    participant S as 서버
    C->>S: 요청 (Request)
    Note right of S: URL: /predict<br/>Method: POST<br/>Headers: Content-Type<br/>Body: JSON
    S->>C: 응답 (Response)
    Note left of C: Status: 200 OK<br/>Body: JSON 결과
```

**요청 예시**
- URL: `https://api.example.com/predict`
- Method: POST
- Headers: `Content-Type: application/json`
- Body: `{"temperature": 200, "pressure": 50}`

**응답 예시**
- Status Code: 200 OK
- Body: `{"prediction": "normal", "confidence": 0.95}`

---

### 1.4 HTTP 상태 코드

| 코드 | 의미 | 설명 |
|-----|------|------|
| **200** | OK | 성공 |
| **201** | Created | 생성 성공 |
| **400** | Bad Request | 잘못된 요청 |
| **401** | Unauthorized | 인증 필요 |
| **404** | Not Found | 자원 없음 |
| **500** | Server Error | 서버 오류 |

---

### 1.5 API 인증 방식

| 방식 | 특징 |
|-----|------|
| **API Key** | 간단, 헤더에 키 포함 |
| **Bearer Token** | OAuth 토큰 방식 |
| **Basic Auth** | ID/Password 인코딩 |

```python
headers = {
    "Authorization": "Bearer sk-xxxxxx"
}
```

---

### 1.6 requests 라이브러리

Python에서 가장 많이 쓰는 HTTP 라이브러리

```python
# 라이브러리 설치
# pip install requests
```

---

### 1.7 실습: GET 요청 기본

```python
import requests

# 공개 API 호출 (JSONPlaceholder - 테스트용 API)
url = 'https://jsonplaceholder.typicode.com/posts/1'
response = requests.get(url)

# 응답 확인
print(f"상태 코드: {response.status_code}")
print(f"응답 본문: {response.text[:100]}...")
```

**실행 결과**
```
상태 코드: 200
응답 본문: {
  "userId": 1,
  "id": 1,
  "title": "sunt aut facere repellat provident...
```

---

### 1.8 실습: POST 요청 기본

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
result = response.json()

print(f"상태 코드: {response.status_code}")
print(f"전송된 JSON: {result.get('json', {})}")
```

**실행 결과**
```
상태 코드: 200
전송된 JSON: {'temperature': 200, 'pressure': 50, 'speed': 100}
```

---

### 1.9 응답 객체 속성

```python
response = requests.get(url)

# 주요 속성
response.status_code    # 상태 코드 (200, 404 등)
response.text          # 응답 본문 (문자열)
response.json()        # JSON을 딕셔너리로 변환
response.headers       # 응답 헤더
response.ok            # 성공 여부 (200~299면 True)
```

---

### 1.10 에러 처리 패턴

```python
import requests

def safe_api_call(url, data=None):
    """안전한 API 호출 함수"""
    try:
        if data:
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)

        response.raise_for_status()  # 에러면 예외 발생
        return {'success': True, 'data': response.json()}

    except requests.Timeout:
        return {'success': False, 'error': '요청 시간 초과'}
    except requests.ConnectionError:
        return {'success': False, 'error': '연결 실패'}
    except requests.HTTPError as e:
        return {'success': False, 'error': f'HTTP 에러: {e}'}

# 사용 예시
result = safe_api_call('https://jsonplaceholder.typicode.com/posts/1')
print(result)
```

---

### 1.11 JSON 데이터 처리

**JSON (JavaScript Object Notation)**
- 데이터 교환 형식
- 사람이 읽기 쉽고, 기계가 파싱하기 쉬움
- API 통신의 표준 형식

**Python과 JSON 대응**

| Python | JSON |
|--------|------|
| dict | object { } |
| list | array [ ] |
| str | string |
| int, float | number |
| True, False | true, false |
| None | null |

---

### 1.12 실습: JSON 직렬화와 역직렬화

```python
import json

# Python -> JSON 문자열 (직렬화)
data = {
    'temperature': 200,
    'status': True,
    'sensors': ['temp_01', 'press_01']
}
json_str = json.dumps(data, indent=2, ensure_ascii=False)
print("=== 직렬화 결과 ===")
print(json_str)

# JSON 문자열 -> Python (역직렬화)
json_str = '{"temperature": 200, "status": true}'
parsed_data = json.loads(json_str)
print("\n=== 역직렬화 결과 ===")
print(f"온도: {parsed_data['temperature']}")
print(f"상태: {parsed_data['status']}")
```

**실행 결과**
```
=== 직렬화 결과 ===
{
  "temperature": 200,
  "status": true,
  "sensors": ["temp_01", "press_01"]
}

=== 역직렬화 결과 ===
온도: 200
상태: True
```

---

### 1.13 제조업 API 활용 예시

| 용도 | API |
|-----|-----|
| 품질 예측 | 자체 ML 모델 API |
| 설비 모니터링 | IoT 플랫폼 API |
| 이상 분석 | LLM API로 원인 분석 |
| 보고서 생성 | 문서 생성 API |

---

## 대주제 2: LLM API 호출과 응답 처리

### 2.1 LLM이란?

**LLM (Large Language Model)**
- 대규모 텍스트로 학습한 언어 모델
- 자연어 이해, 생성, 번역, 요약 등 수행
- 대표: GPT-4, Claude, Gemini, Llama

```
입력: "온도가 230도로 높은데 원인이 뭘까?"
출력: "온도 상승의 원인으로 냉각 시스템 고장,
       센서 오작동, 과부하 운전 등을 고려할 수 있습니다."
```

---

### 2.2 주요 LLM 서비스 비교

| 서비스 | 제공사 | 특징 |
|-------|-------|------|
| **GPT-4** | OpenAI | 가장 널리 사용, 범용 |
| **Claude** | Anthropic | 긴 문서 처리, 안전성 |
| **Gemini** | Google | 멀티모달, 검색 연동 |
| **Llama** | Meta | 오픈소스, 자체 배포 가능 |

**LLM API의 장점**

| 장점 | 설명 |
|-----|------|
| 자연어 인터페이스 | 복잡한 코딩 없이 명령 |
| 범용성 | 분류, 요약, 생성 모두 가능 |
| 빠른 프로토타이핑 | 학습 없이 바로 사용 |

---

### 2.3 API 키 발급 절차

**OpenAI**
1. https://platform.openai.com 접속
2. 회원가입 및 로그인
3. API Keys 메뉴 -> "Create new secret key" 클릭
4. 키 형식: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx`

**Claude (Anthropic)**
1. https://console.anthropic.com 접속
2. 회원가입 및 로그인
3. API Keys 메뉴 -> "Create Key" 클릭
4. 키 형식: `sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx`

---

### 2.4 API 키 안전 관리

**절대 하면 안 되는 것**
```python
# 코드에 직접 입력 금지!
api_key = "sk-proj-xxxxx"  # 위험!
```

**안전한 방법**
```python
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 읽기
api_key = os.environ.get('OPENAI_API_KEY')
print(f"API 키 로드 완료: {api_key[:10]}..." if api_key else "API 키 없음")
```

**.env 파일 예시**
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

---

### 2.5 OpenAI API 기본 구조

```python
from openai import OpenAI
import os

# 클라이언트 초기화
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# API 호출
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "당신은 제조 전문가입니다."},
        {"role": "user", "content": "온도 이상의 원인을 분석해주세요."}
    ]
)

# 응답 추출
answer = response.choices[0].message.content
print(answer)
```

---

### 2.6 Claude API 기본 구조

```python
import anthropic
import os

# 클라이언트 초기화
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# API 호출
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "온도 이상의 원인을 분석해주세요."}
    ]
)

# 응답 추출
answer = response.content[0].text
print(answer)
```

---

### 2.7 메시지 역할

| 역할 | 설명 | 사용 |
|-----|------|-----|
| **system** | AI의 역할/성격 정의 | 처음에 한 번 |
| **user** | 사용자 질문 | 매 요청마다 |
| **assistant** | AI 응답 | 대화 이력 유지 |

```python
messages = [
    {"role": "system", "content": "제조업 품질 전문가"},
    {"role": "user", "content": "불량 원인?"},
    {"role": "assistant", "content": "온도를 확인하세요."},
    {"role": "user", "content": "온도 외에 다른 원인은?"}
]
```

---

### 2.8 주요 파라미터

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| **model** | 모델 선택 | gpt-4, claude-3-sonnet |
| **max_tokens** | 응답 최대 길이 | 1000~4000 |
| **temperature** | 창의성 (0~1) | 분석: 0.3, 창작: 0.8 |
| **top_p** | 단어 선택 다양성 | 0.9~1.0 |

**temperature 이해**
- 0.0: 결정적 (같은 질문에 항상 같은 답변) -> 정확한 분석에 적합
- 0.7: 중간 -> 일반 대화, 설명에 적합
- 1.0: 창의적 (다양한 답변) -> 아이디어 생성에 적합

---

### 2.9 실습: LLM API 호출 시뮬레이션

실제 API 키 없이 구조를 이해하기 위한 시뮬레이션

```python
def simulate_llm_call(messages, model="gpt-4", temperature=0.7):
    """
    LLM API 호출 시뮬레이션
    실제 환경에서는 openai 라이브러리 사용
    """
    # 요청 정보 출력
    print("=== LLM API 요청 ===")
    print(f"모델: {model}")
    print(f"Temperature: {temperature}")
    print(f"메시지 수: {len(messages)}")

    for msg in messages:
        print(f"  [{msg['role']}] {msg['content'][:50]}...")

    # 시뮬레이션 응답
    simulated_response = {
        'content': '온도 이상의 주요 원인: 1) 냉각 시스템 고장 2) 센서 오작동 3) 과부하 운전',
        'usage': {
            'prompt_tokens': 150,
            'completion_tokens': 80,
            'total_tokens': 230
        }
    }

    print("\n=== LLM API 응답 ===")
    print(f"응답: {simulated_response['content']}")
    print(f"토큰 사용량: {simulated_response['usage']}")

    return simulated_response

# 사용 예시
messages = [
    {"role": "system", "content": "당신은 제조업 품질 관리 전문가입니다."},
    {"role": "user", "content": "온도가 230도로 정상 범위(180-220도)를 초과했습니다. 원인을 분석해주세요."}
]

response = simulate_llm_call(messages, temperature=0.3)
```

---

### 2.10 비용 구조

| 모델 | 입력 (1K 토큰) | 출력 (1K 토큰) |
|-----|---------------|---------------|
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5 Turbo | $0.0005 | $0.0015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |

**토큰 참고**: 대략 한글 1글자 = 1~2토큰

---

### 2.11 실습: 비용 추정 함수

```python
def estimate_cost(prompt_tokens, completion_tokens, model="gpt-4"):
    """
    API 호출 비용 추정

    Parameters:
    -----------
    prompt_tokens : int - 입력 토큰 수
    completion_tokens : int - 출력 토큰 수
    model : str - 모델명

    Returns:
    --------
    dict : 비용 추정 결과
    """
    # 토큰당 비용 (USD per 1K tokens)
    pricing = {
        'gpt-4': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015}
    }

    rates = pricing.get(model, pricing['gpt-4'])

    input_cost = (prompt_tokens / 1000) * rates['input']
    output_cost = (completion_tokens / 1000) * rates['output']
    total_cost = input_cost + output_cost

    return {
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_cost_usd': round(total_cost, 6),
        'total_cost_krw': round(total_cost * 1400, 2)  # 예상 환율
    }

# 사용 예시
cost = estimate_cost(500, 200, 'gpt-4')
print(f"모델: {cost['model']}")
print(f"입력 토큰: {cost['prompt_tokens']}")
print(f"출력 토큰: {cost['completion_tokens']}")
print(f"총 비용: ${cost['total_cost_usd']} (약 {cost['total_cost_krw']}원)")
```

**실행 결과**
```
모델: gpt-4
입력 토큰: 500
출력 토큰: 200
총 비용: $0.011 (약 15.4원)
```

---

## 대주제 3: 프롬프트 작성과 제조업 활용

### 3.1 프롬프트 엔지니어링이란?

**정의**: AI에게 원하는 결과를 얻기 위한 질문/지시 작성 기술

```
나쁜 프롬프트:
"불량 분석해줘"

좋은 프롬프트:
"제조 라인에서 온도 250도, 압력 70kPa 조건에서
발생한 표면 스크래치 불량의 원인을 3가지 제시하고
각각의 해결책을 설명해주세요."
```

---

### 3.2 프롬프트 기본 원칙

| 원칙 | 설명 |
|-----|------|
| **명확성** | 구체적이고 명확하게 |
| **맥락 제공** | 배경 정보 포함 |
| **형식 지정** | 원하는 출력 형태 명시 |
| **예시 제공** | 원하는 답변 예시 (Few-shot) |
| **단계적 사고** | 복잡한 문제는 단계 분리 (CoT) |

```mermaid
flowchart LR
    A[역할 지정] --> B[맥락 제공]
    B --> C[질문/요청]
    C --> D[형식 지정]
    D --> E[제약 조건]
```

---

### 3.3 역할 지정 (System Prompt)

```python
system_prompt = """
당신은 20년 경력의 제조업 품질 관리 전문가입니다.
반도체 제조 공정에 대한 깊은 이해를 가지고 있습니다.
데이터 기반으로 분석하고, 실용적인 해결책을 제시합니다.
기술적이지만 이해하기 쉽게 설명합니다.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "..."}
]
```

---

### 3.4 맥락 제공과 형식 지정

```python
user_prompt = """
[상황]
자동차 부품 제조 라인에서 용접 품질 불량이 증가했습니다.

[데이터]
- 불량률: 지난 주 2% -> 이번 주 5%
- 용접 전류: 평균 180A (기준: 175-185A)
- 용접 시간: 평균 2.3초 (기준: 2.0-2.5초)
- 온도: 평균 850도 (기준: 800-900도)

[질문]
불량률 증가의 가능한 원인과 조치 방안을 제시해주세요.

[형식]
## 원인 분석
1. [원인 1]: 근거와 영향
2. [원인 2]: 근거와 영향

## 권장 조치
1. 즉시 조치: ...
2. 단기 대책: ...
"""
```

---

### 3.5 Few-shot 학습

예시를 통해 원하는 출력 패턴을 학습시킴

```python
user_prompt = """
센서 데이터 이상 여부를 판단해주세요.

예시:
입력: 온도 210도, 압력 55kPa
출력: 정상 (모든 값이 정상 범위)

입력: 온도 280도, 압력 45kPa
출력: 이상 (온도 초과: 280 > 250)

이제 분석해주세요:
입력: 온도 245도, 압력 72kPa
출력:
"""
```

---

### 3.6 Chain of Thought (단계적 사고)

```python
user_prompt = """
다음 문제를 단계별로 분석해주세요.

문제: 생산 라인 A에서 불량률이 갑자기 증가했습니다.

단계 1: 데이터 확인
- 어떤 데이터를 확인해야 하는지

단계 2: 패턴 분석
- 불량이 언제, 어떤 조건에서 발생하는지

단계 3: 원인 가설
- 가능한 원인 목록

단계 4: 검증 방법
- 각 가설을 검증하는 방법

단계 5: 조치 방안
- 우선순위별 조치 사항
"""
```

---

### 3.7 JSON 출력 요청

구조화된 데이터가 필요할 때 JSON 형식 지정

```python
user_prompt = """
센서 데이터를 분석하고 결과를 JSON 형식으로 반환해주세요.

입력 데이터:
- 온도: 235도
- 압력: 65kPa
- 진동: 8.5mm/s

JSON 형식:
{
    "status": "normal" 또는 "warning" 또는 "critical",
    "anomalies": ["이상 항목 목록"],
    "risk_score": 0-100 점수,
    "recommendations": ["권장 조치"]
}
"""
```

---

### 3.8 프롬프트 최적화 팁

| 팁 | 설명 |
|---|------|
| **반복 실험** | 다양한 표현으로 테스트 |
| **제약 조건** | 답변 길이, 형식 제한 |
| **부정 지시** | "~하지 마세요" 사용 |
| **우선순위** | 중요한 내용 먼저 |
| **검증 요청** | "확실하지 않으면 모른다고 답변" |

**피해야 할 안티패턴**
- "잘 분석해줘" (모호함)
- "모든 것을 설명해줘" (범위 없음)
- "가장 좋은 방법은?" (기준 없음)
- 매우 긴 프롬프트 (집중력 저하)

---

### 3.9 실습: 제조업 활용 - 불량 원인 분석

```python
def create_defect_analysis_prompt(sensor_data, defect_type):
    """불량 원인 분석 프롬프트 생성"""
    system_prompt = """
당신은 20년 경력의 제조업 품질 관리 전문가입니다.
데이터 기반으로 분석하고, 실용적인 해결책을 제시합니다.
"""

    user_prompt = f"""
[제조 공정 불량 분석]

불량 유형: {defect_type}
센서 데이터:
- 온도: {sensor_data['temperature']}도
- 압력: {sensor_data['pressure']}kPa
- 속도: {sensor_data['speed']}rpm

위 데이터를 바탕으로:
1. 가장 가능성 높은 원인 3가지
2. 각 원인의 해결 방법
3. 예방을 위한 권장 사항

을 분석해주세요.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# 사용 예시
sensor_data = {
    'temperature': 235,
    'pressure': 72,
    'speed': 1500
}

messages = create_defect_analysis_prompt(sensor_data, "표면 스크래치")
print("=== 프롬프트 미리보기 ===")
for msg in messages:
    print(f"[{msg['role']}]")
    print(msg['content'][:200])
    print()
```

---

### 3.10 실습: 품질 보고서 자동 생성

```python
def create_quality_report_prompt(report_data):
    """품질 보고서 생성 프롬프트"""
    prompt = f"""
다음 데이터로 품질 보고서를 작성해주세요.

[데이터]
- 기간: {report_data['period']}
- 생산량: {report_data['production']:,}개
- 불량률: {report_data['defect_rate']}%
- 주요 불량: {', '.join(report_data['top_defects'])}

[형식]
## 1. 요약 (3줄 이내)
## 2. 주요 지표 분석
## 3. 개선 필요 사항
## 4. 다음 주 권장 조치

비즈니스 리포트 형식으로 간결하게 작성해주세요.
"""
    return prompt

# 사용 예시
report_data = {
    'period': '2026년 1월 1주차',
    'production': 10500,
    'defect_rate': 3.2,
    'top_defects': ['표면 스크래치', '치수 불량', '용접 불량']
}

prompt = create_quality_report_prompt(report_data)
print("=== 생성된 프롬프트 ===")
print(prompt)
```

---

### 3.11 LLM + ML 하이브리드 분석

```mermaid
flowchart TB
    A[센서 데이터] --> B[ML 모델]
    B --> C[예측 결과<br/>불량 확률: 0.85]
    C --> D[LLM 해석]
    D --> E[권장 조치<br/>자연어 설명]

    subgraph ML
    B
    end

    subgraph LLM
    D
    end
```

---

### 3.12 실습: 하이브리드 분석 구현

```python
def hybrid_analysis(sensor_data, ml_prediction, ml_probability):
    """
    ML 예측 결과를 LLM으로 해석

    Parameters:
    -----------
    sensor_data : dict - 센서 데이터
    ml_prediction : str - ML 예측 결과 ('normal' 또는 'defect')
    ml_probability : float - 예측 확률

    Returns:
    --------
    dict : 통합 분석 결과
    """
    prompt = f"""
[ML 모델 예측 결과 해석]

입력 데이터:
- 온도: {sensor_data['temperature']}도
- 압력: {sensor_data['pressure']}kPa
- 진동: {sensor_data['vibration']}mm/s

예측 결과: {ml_prediction}
예측 확률: {ml_probability:.1%}

이 결과를 바탕으로:
1. 예측 결과의 의미 (한 문장)
2. 주의가 필요한 센서값
3. 권장 조치 사항 (우선순위별)

을 현장 작업자가 이해할 수 있게 설명해주세요.
"""

    # 시뮬레이션 응답
    interpretation = """
1. 의미: 현재 설비 상태는 불량 발생 가능성이 높아 주의가 필요합니다.

2. 주의 센서값:
   - 온도 (235도): 정상 범위(180-220도) 초과
   - 진동 (8.5mm/s): 경고 수준 도달

3. 권장 조치:
   - 즉시: 온도 센서 및 냉각 시스템 점검
   - 단기: 진동 원인 조사 (베어링, 정렬 상태)
   - 예방: 설비 정기 점검 주기 단축
"""

    return {
        'sensor_data': sensor_data,
        'prediction': ml_prediction,
        'probability': ml_probability,
        'interpretation': interpretation
    }

# 사용 예시
sensor_data = {
    'temperature': 235,
    'pressure': 68,
    'vibration': 8.5
}

result = hybrid_analysis(sensor_data, 'defect', 0.85)
print("=== 하이브리드 분석 결과 ===")
print(f"예측: {result['prediction']} (확률: {result['probability']:.1%})")
print(f"\n해석:\n{result['interpretation']}")
```

---

### 3.13 주의사항

| 항목 | 주의점 | 대응 방안 |
|-----|--------|----------|
| **환각** | 사실이 아닌 내용 생성 가능 | 중요 정보 검증 필수 |
| **일관성** | 같은 질문에 다른 답변 | temperature=0 사용 |
| **보안** | 민감 정보 전송 위험 | 데이터 익명화 |
| **비용** | 예상치 못한 비용 발생 | 사용량 모니터링 |
| **지연** | 응답 시간 불규칙 | 타임아웃 설정 |

---

### 3.14 비용 최적화 전략

| 전략 | 방법 |
|-----|------|
| **모델 선택** | 간단한 작업은 저렴한 모델 (GPT-3.5) |
| **프롬프트 최적화** | 불필요한 내용 제거, 간결하게 |
| **캐싱** | 동일 질문 결과 저장하여 재사용 |
| **배치 처리** | 여러 요청 묶어서 처리 |
| **토큰 제한** | max_tokens 적절히 설정 |

---

## 실습 종합: 완전한 예제

### 공개 API 호출 + 데이터 처리

```python
import requests
import json

def fetch_and_process_data():
    """
    공개 API에서 데이터를 가져와 처리하는 완전한 예제
    """
    # 1. API 호출
    url = 'https://jsonplaceholder.typicode.com/posts'
    params = {'_limit': 3}

    print("=== 1. API 호출 ===")
    response = requests.get(url, params=params, timeout=10)

    if not response.ok:
        print(f"에러: {response.status_code}")
        return None

    # 2. JSON 파싱
    data = response.json()
    print(f"가져온 데이터 수: {len(data)}개")

    # 3. 데이터 처리
    print("\n=== 2. 데이터 처리 ===")
    for item in data:
        print(f"ID: {item['id']}, 제목: {item['title'][:30]}...")

    # 4. 결과 저장 (시뮬레이션)
    result = {
        'total_items': len(data),
        'ids': [item['id'] for item in data],
        'status': 'success'
    }

    print("\n=== 3. 결과 JSON ===")
    print(json.dumps(result, indent=2))

    return result

# 실행
result = fetch_and_process_data()
```

---

### LLM 활용 종합 예제

```python
def complete_manufacturing_analysis():
    """
    제조업 데이터 분석 완전한 예제
    (API 호출 시뮬레이션)
    """
    # 1. 센서 데이터 수집 (시뮬레이션)
    sensor_data = {
        'temperature': 232,
        'pressure': 68,
        'vibration': 7.2,
        'humidity': 45,
        'speed': 1450
    }

    print("=== 1. 센서 데이터 ===")
    for key, value in sensor_data.items():
        print(f"  {key}: {value}")

    # 2. 정상 범위 정의
    normal_ranges = {
        'temperature': (180, 220),
        'pressure': (50, 70),
        'vibration': (0, 5),
        'humidity': (30, 60),
        'speed': (1400, 1600)
    }

    # 3. 이상 탐지
    print("\n=== 2. 이상 탐지 ===")
    anomalies = []
    for key, value in sensor_data.items():
        min_val, max_val = normal_ranges[key]
        if value < min_val or value > max_val:
            anomalies.append(f"{key}: {value} (범위: {min_val}-{max_val})")
            print(f"  [이상] {key}: {value} (정상 범위: {min_val}-{max_val})")

    if not anomalies:
        print("  모든 센서값 정상")

    # 4. LLM 분석 요청 생성
    print("\n=== 3. LLM 분석 프롬프트 ===")
    prompt = f"""
센서 데이터 분석 결과:
- 총 센서 수: {len(sensor_data)}개
- 이상 탐지: {len(anomalies)}건
- 이상 항목: {', '.join(anomalies) if anomalies else '없음'}

위 결과를 바탕으로 설비 상태를 평가하고 권장 조치를 제안해주세요.
"""
    print(prompt)

    # 5. 시뮬레이션 응답
    print("=== 4. LLM 분석 결과 (시뮬레이션) ===")
    simulated_response = """
[설비 상태 평가]
- 종합 상태: 주의 필요
- 위험 수준: 중간

[이상 항목 분석]
1. 온도 (232도): 정상 범위 초과
   - 원인: 냉각 시스템 효율 저하 또는 부하 증가
   - 조치: 냉각수 온도 및 유량 점검

2. 진동 (7.2mm/s): 정상 범위 초과
   - 원인: 베어링 마모 또는 축 정렬 불량
   - 조치: 베어링 상태 점검, 정렬 확인

[권장 조치]
1. 즉시: 냉각 시스템 및 베어링 점검
2. 단기: 정밀 진단 실시
3. 예방: 정기 점검 주기 단축 검토
"""
    print(simulated_response)

    return {
        'sensor_data': sensor_data,
        'anomalies': anomalies,
        'analysis': simulated_response
    }

# 실행
result = complete_manufacturing_analysis()
```

---

## 핵심 정리

### 1. API와 REST API
- HTTP 메서드: GET(조회), POST(생성), PUT(수정), DELETE(삭제)
- 상태 코드: 200(성공), 400(잘못된 요청), 401(인증 필요), 500(서버 에러)
- requests 라이브러리: `get()`, `post()`, `response.json()`
- JSON: `json.dumps()` (직렬화), `json.loads()` (역직렬화)

### 2. LLM API 호출
- OpenAI: `client.chat.completions.create()`
- Claude: `client.messages.create()`
- API 키는 환경 변수로 관리 (`.env` 파일)
- 메시지 역할: system(역할), user(질문), assistant(응답)

### 3. 프롬프트 엔지니어링
- 역할 지정: system 메시지로 AI 역할 정의
- 맥락 제공: 상황, 데이터, 배경 정보 포함
- 형식 지정: 원하는 출력 형태 명시
- Few-shot: 예시로 원하는 패턴 학습
- Chain of Thought: 단계적 사고 유도

### 4. 제조업 활용
- 불량 원인 분석
- 품질 보고서 자동 생성
- 이상 알림 해석
- ML + LLM 하이브리드 분석

---

## 핵심 코드

```python
import requests
import json
from openai import OpenAI
import os

# REST API 호출
response = requests.get(url, params={'key': 'value'})
response = requests.post(url, json=data, headers=headers, timeout=10)
if response.ok:
    result = response.json()

# JSON 처리
json_str = json.dumps(data, ensure_ascii=False, indent=2)
data = json.loads(json_str)

# LLM API 호출
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "제조 전문가"},
        {"role": "user", "content": "불량 원인 분석..."}
    ],
    temperature=0.3
)
answer = response.choices[0].message.content
```

---

## 체크리스트

- [ ] REST API 개념 이해
- [ ] HTTP 메서드 (GET, POST) 구분
- [ ] requests.get(), requests.post() 사용
- [ ] JSON 직렬화/역직렬화
- [ ] 에러 처리 구현 (try-except, timeout)
- [ ] LLM API 키 발급 및 환경 변수 설정
- [ ] 메시지 역할 (system, user, assistant) 이해
- [ ] 역할 지정 프롬프트 작성
- [ ] 맥락 제공 및 형식 지정
- [ ] 제조업 활용 사례 실습

