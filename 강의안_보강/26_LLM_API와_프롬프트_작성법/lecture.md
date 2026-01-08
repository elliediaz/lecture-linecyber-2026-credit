# 26차시: LLM API와 프롬프트 작성법

## 학습 목표

1. **OpenAI/Claude API** 사용법을 익힘
2. **프롬프트 엔지니어링** 기초를 이해함
3. 제조업 **활용 사례**를 실습함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | LLM과 API 소개 |
| 대주제 2 | 12분 | 프롬프트 엔지니어링 |
| 대주제 3 | 6분 | 제조업 활용 사례 |
| 정리 | 2분 | 핵심 요약 |

---

## 대주제 1: LLM과 API 소개

### 1.1 LLM이란?

**LLM (Large Language Model)**
- 대규모 텍스트로 학습한 언어 모델
- 자연어 이해, 생성, 번역, 요약 등 수행
- 대표: GPT-4, Claude, Gemini, Llama

```
입력: "온도가 230도로 높은데 원인이 뭘까?"
출력: "온도 상승의 원인으로 냉각 시스템 고장,
       센서 오작동, 과부하 운전 등을 고려할 수 있습니다."
```

### 1.2 주요 LLM 서비스

| 서비스 | 제공사 | 특징 |
|-------|-------|------|
| **GPT-4** | OpenAI | 가장 널리 사용, 범용 |
| **Claude** | Anthropic | 긴 문서 처리, 안전성 |
| **Gemini** | Google | 멀티모달, 검색 연동 |
| **Llama** | Meta | 오픈소스, 자체 배포 |

### 1.3 LLM API의 장점

| 장점 | 설명 |
|-----|------|
| **자연어 인터페이스** | 복잡한 코딩 없이 명령 |
| **범용성** | 분류, 요약, 생성 모두 가능 |
| **빠른 프로토타이핑** | 학습 없이 바로 사용 |
| **지속적 개선** | 모델 업데이트 자동 적용 |

### 1.4 API 키 발급 (OpenAI)

1. https://platform.openai.com 접속
2. 회원가입 및 로그인
3. API Keys 메뉴
4. "Create new secret key" 클릭
5. 키 안전하게 저장

```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

주의: 키는 한 번만 표시됨

### 1.5 API 키 발급 (Claude)

1. https://console.anthropic.com 접속
2. 회원가입 및 로그인
3. API Keys 메뉴
4. "Create Key" 클릭
5. 키 안전하게 저장

```
sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 1.6 API 키 관리 (중요)

**절대 하면 안 되는 것**
```python
# 코드에 직접 입력 금지!
api_key = "sk-proj-xxxxx"  # 위험!
```

**안전한 방법**
```python
import os

# 환경 변수에서 읽기
api_key = os.environ.get('OPENAI_API_KEY')

# 또는 .env 파일 사용
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
```

### 1.7 OpenAI API 기본 구조

```python
from openai import OpenAI

client = OpenAI(api_key='your-api-key')

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "당신은 제조 전문가입니다."},
        {"role": "user", "content": "온도 이상의 원인을 분석해주세요."}
    ]
)

answer = response.choices[0].message.content
print(answer)
```

### 1.8 Claude API 기본 구조

```python
import anthropic

client = anthropic.Anthropic(api_key='your-api-key')

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "온도 이상의 원인을 분석해주세요."}
    ]
)

answer = response.content[0].text
print(answer)
```

### 1.9 메시지 역할

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

### 1.10 API 파라미터

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| **model** | 모델 선택 | gpt-4, claude-3 |
| **max_tokens** | 응답 최대 길이 | 1000~4000 |
| **temperature** | 창의성 (0~1) | 분석: 0.3, 창작: 0.8 |
| **top_p** | 단어 선택 다양성 | 0.9~1.0 |

### 1.11 temperature 이해

```
temperature = 0.0 (결정적)
-> 같은 질문에 항상 같은 답변
-> 정확한 분석, 사실 확인에 적합

temperature = 0.7 (중간)
-> 적당한 변화
-> 일반 대화, 설명에 적합

temperature = 1.0 (창의적)
-> 다양한 답변
-> 아이디어 생성, 창작에 적합
```

### 1.12 비용 구조

| 모델 | 입력 (1K 토큰) | 출력 (1K 토큰) |
|-----|---------------|---------------|
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5 Turbo | $0.0005 | $0.0015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |

토큰: 대략 한글 1글자 = 1~2토큰

---

## 대주제 2: 프롬프트 엔지니어링

### 2.1 프롬프트 엔지니어링이란?

**정의**: AI에게 원하는 결과를 얻기 위한 질문/지시 작성 기술

```
나쁜 프롬프트:
"불량 분석해줘"

좋은 프롬프트:
"제조 라인에서 온도 250도, 압력 70kPa 조건에서
발생한 표면 스크래치 불량의 원인을 3가지 제시하고
각각의 해결책을 설명해주세요."
```

### 2.2 프롬프트 기본 원칙

| 원칙 | 설명 |
|-----|------|
| **명확성** | 구체적이고 명확하게 |
| **맥락 제공** | 배경 정보 포함 |
| **형식 지정** | 원하는 출력 형태 명시 |
| **예시 제공** | 원하는 답변 예시 |
| **단계적 사고** | 복잡한 문제는 단계 분리 |

### 2.3 역할 지정 (Role)

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

### 2.4 맥락 제공

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
"""
```

### 2.5 출력 형식 지정

```python
user_prompt = """
품질 이상 원인을 분석해주세요.

다음 형식으로 답변해주세요:

## 원인 분석
1. [원인 1]
   - 근거: ...
   - 영향: ...

2. [원인 2]
   - 근거: ...
   - 영향: ...

## 권장 조치
1. 즉시 조치: ...
2. 단기 대책: ...
3. 장기 개선: ...
"""
```

### 2.6 Few-shot 학습

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

### 2.7 Chain of Thought (단계적 사고)

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

### 2.8 JSON 출력 요청

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

### 2.9 프롬프트 최적화 팁

| 팁 | 설명 |
|---|------|
| **반복 실험** | 다양한 표현으로 테스트 |
| **제약 조건** | 답변 길이, 형식 제한 |
| **부정 지시** | "~하지 마세요" 사용 |
| **우선순위** | 중요한 내용 먼저 |
| **검증 요청** | "확실하지 않으면 모른다고" |

### 2.10 프롬프트 안티패턴

**피해야 할 것**
```
- "잘 분석해줘" (모호함)
- "모든 것을 설명해줘" (범위 없음)
- "가장 좋은 방법은?" (기준 없음)
- 매우 긴 프롬프트 (집중력 저하)
```

**좋은 방법**
```
- 구체적인 요청
- 명확한 범위
- 평가 기준 제시
- 핵심만 간결하게
```

---

## 대주제 3: 제조업 활용 사례

### 3.1 활용 사례 1: 불량 원인 분석

```python
def analyze_defect(sensor_data, defect_type):
    prompt = f"""
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
    return call_llm(prompt)
```

### 3.2 활용 사례 2: 보고서 자동 생성

```python
def generate_quality_report(data):
    prompt = f"""
    다음 데이터로 품질 보고서를 작성해주세요.

    [데이터]
    - 기간: {data['period']}
    - 생산량: {data['production']}
    - 불량률: {data['defect_rate']}%
    - 주요 불량: {data['top_defects']}

    [형식]
    1. 요약 (3줄 이내)
    2. 주요 지표 분석
    3. 개선 필요 사항
    4. 다음 주 권장 조치

    비즈니스 리포트 형식으로 작성해주세요.
    """
    return call_llm(prompt)
```

### 3.3 활용 사례 3: 이상 탐지 알림 해석

```python
def explain_anomaly(alert):
    prompt = f"""
    [이상 탐지 알림 해석]

    알림 내용: {alert['message']}
    발생 설비: {alert['equipment']}
    발생 시각: {alert['timestamp']}
    관련 센서값: {alert['sensor_values']}

    현장 작업자가 이해할 수 있도록:
    1. 무슨 문제인지 쉽게 설명
    2. 즉시 확인해야 할 것
    3. 응급 조치 방법

    을 간단명료하게 안내해주세요.
    """
    return call_llm(prompt)
```

### 3.4 활용 사례 4: 문서 요약

```python
def summarize_manual(manual_text):
    prompt = f"""
    다음 설비 매뉴얼을 요약해주세요.

    [매뉴얼 내용]
    {manual_text}

    [요청]
    1. 핵심 절차 5단계로 요약
    2. 주의사항 3가지
    3. 트러블슈팅 가이드

    현장에서 빠르게 참고할 수 있도록
    간결하게 작성해주세요.
    """
    return call_llm(prompt)
```

### 3.5 활용 사례 5: 데이터 해석

```python
def interpret_data(df_summary):
    prompt = f"""
    다음 통계 분석 결과를 해석해주세요.

    [통계 요약]
    {df_summary}

    [질문]
    1. 주목할 만한 패턴이 있는가?
    2. 이상치는 무엇을 의미하는가?
    3. 개선이 필요한 영역은?

    데이터 분석가 관점에서 해석해주세요.
    """
    return call_llm(prompt)
```

### 3.6 LLM + ML 모델 조합

```python
def hybrid_analysis(sensor_data):
    # 1. ML 모델로 예측
    ml_prediction = ml_model.predict(sensor_data)
    ml_probability = ml_model.predict_proba(sensor_data)

    # 2. LLM으로 해석 및 권장 사항
    prompt = f"""
    [ML 모델 예측 결과 해석]

    입력 데이터: {sensor_data}
    예측 결과: {ml_prediction}
    확률: {ml_probability}

    이 결과를 바탕으로:
    1. 예측 결과의 의미
    2. 주의가 필요한 센서값
    3. 권장 조치 사항

    을 설명해주세요.
    """
    interpretation = call_llm(prompt)

    return {
        'prediction': ml_prediction,
        'probability': ml_probability,
        'interpretation': interpretation
    }
```

### 3.7 비용 최적화 전략

| 전략 | 방법 |
|-----|------|
| **모델 선택** | 간단한 작업은 저렴한 모델 |
| **프롬프트 최적화** | 불필요한 내용 제거 |
| **캐싱** | 동일 질문 결과 저장 |
| **배치 처리** | 여러 요청 묶어서 처리 |
| **토큰 제한** | max_tokens 적절히 설정 |

### 3.8 주의사항

| 항목 | 주의점 |
|-----|--------|
| **환각** | 사실 확인 필요 |
| **일관성** | 같은 질문에 다른 답변 |
| **보안** | 민감 정보 전송 주의 |
| **비용** | 사용량 모니터링 |
| **지연** | 응답 시간 고려 |

---

## 실습: LLM API 호출 (시뮬레이션)

### API 호출 래퍼 함수

```python
import os
from typing import Dict, List

def call_openai(messages: List[Dict], model: str = "gpt-4",
                temperature: float = 0.7, max_tokens: int = 1000) -> Dict:
    """
    OpenAI API 호출

    Parameters:
    -----------
    messages : list
        대화 메시지 목록
    model : str
        모델 이름
    temperature : float
        창의성 파라미터 (0~1)
    max_tokens : int
        최대 응답 토큰 수

    Returns:
    --------
    dict : API 응답
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        'content': response.choices[0].message.content,
        'usage': {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    }
```

### 응답에서 텍스트 추출

```python
def get_response_text(response: Dict) -> str:
    """API 응답에서 텍스트 추출"""
    return response['content']
```

### 비용 추정 함수

```python
def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = "gpt-4") -> Dict:
    """
    API 호출 비용 추정

    Parameters:
    -----------
    prompt_tokens : int
        입력 토큰 수
    completion_tokens : int
        출력 토큰 수
    model : str
        모델명

    Returns:
    --------
    dict : 비용 추정 결과
    """
    # 토큰당 비용 (USD per 1M tokens)
    pricing = {
        'gpt-4': {'input': 30.0, 'output': 60.0},
        'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'claude-3-opus': {'input': 15.0, 'output': 75.0},
        'claude-3-sonnet': {'input': 3.0, 'output': 15.0}
    }

    rates = pricing.get(model, pricing['gpt-4'])

    input_cost = (prompt_tokens / 1_000_000) * rates['input']
    output_cost = (completion_tokens / 1_000_000) * rates['output']
    total_cost = input_cost + output_cost

    return {
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_cost_usd': total_cost,
        'total_cost_krw': total_cost * 1350  # 예상 환율
    }

# 사용 예시
cost = estimate_cost(500, 200, 'gpt-4')
print(f"총 비용: ${cost['total_cost_usd']:.6f}")
```

---

## 핵심 정리

### 1. LLM API 사용법
- OpenAI: `client.chat.completions.create()`
- Claude: `client.messages.create()`
- API 키는 환경 변수로 관리

### 2. 프롬프트 엔지니어링
- 역할 지정: system 메시지로 AI 역할 정의
- 맥락 제공: 상황, 데이터, 배경 정보
- 형식 지정: 원하는 출력 형태 명시
- Few-shot: 예시로 원하는 패턴 학습
- Chain of Thought: 단계적 사고 유도

### 3. 제조업 활용
- 불량 원인 분석
- 품질 보고서 자동 생성
- 이상 알림 해석
- ML + LLM 하이브리드

### 4. 주의사항
- 환각: 사실 확인 필요
- 비용: 모델 선택, 토큰 관리
- 보안: 민감 정보 주의

### 핵심 코드

```python
from openai import OpenAI
import os

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
usage = response.usage  # 토큰 사용량
```

---

## 체크리스트

- [ ] API 키 발급 및 환경 변수 설정
- [ ] 기본 API 호출 테스트
- [ ] 역할 지정 프롬프트 작성
- [ ] 출력 형식 지정 연습
- [ ] 제조업 활용 사례 실습
