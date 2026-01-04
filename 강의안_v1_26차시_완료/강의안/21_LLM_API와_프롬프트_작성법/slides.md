---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 21차시'
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

# LLM API와 프롬프트 작성법

## 21차시 | Part IV. AI 서비스화와 활용

**대규모 언어 모델 활용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **LLM API**를 호출하는 방법을 익힌다
2. 효과적인 **프롬프트**를 작성한다
3. 제조 현장에서 LLM을 **활용**한다

---

# LLM이란?

## Large Language Model

> 대규모 텍스트로 학습된 **언어 이해/생성** AI 모델

### 대표 모델
- **OpenAI GPT**: ChatGPT, GPT-4
- **Anthropic Claude**: Claude 3
- **Google**: Gemini
- **Meta**: LLaMA

> 자연어로 질문하면 자연어로 대답!

---

# LLM의 능력

## 다양한 작업 수행

| 작업 | 예시 |
|------|------|
| 텍스트 생성 | 보고서, 이메일 작성 |
| 요약 | 긴 문서 핵심 정리 |
| 번역 | 한영, 영한 번역 |
| 질의응답 | 매뉴얼 기반 답변 |
| 코드 생성 | Python 코드 작성 |
| 분석 | 데이터 해석, 인사이트 |

---

# 제조 현장 활용

## LLM in Manufacturing

### 문서 작업
- 작업 지시서 자동 생성
- 품질 보고서 요약
- 매뉴얼 Q&A

### 데이터 분석
- 불량 원인 분석 지원
- 센서 데이터 해석
- 이상 패턴 설명

### 커뮤니케이션
- 다국어 번역
- 기술 문서 작성

---

# OpenAI API

## ChatGPT API 사용

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "안녕하세요!"}
    ]
)

print(response.choices[0].message.content)
```

---

# 메시지 구조

## System / User / Assistant

```python
messages = [
    {"role": "system", "content": "당신은 제조 품질 전문가입니다."},
    {"role": "user", "content": "온도가 높으면 불량이 증가하나요?"},
    {"role": "assistant", "content": "네, 일반적으로..."},
    {"role": "user", "content": "적정 온도는?"}
]
```

| 역할 | 설명 |
|------|------|
| system | AI의 역할/성격 설정 |
| user | 사용자 질문 |
| assistant | AI 응답 |

---

# Claude API

## Anthropic Claude 사용

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "제조 불량의 주요 원인을 알려주세요."}
    ]
)

print(message.content[0].text)
```

---

# 프롬프트란?

## Prompt

> AI에게 전달하는 **지시/질문 텍스트**

### 좋은 프롬프트의 조건
1. **명확한 지시**: 무엇을 원하는지 구체적으로
2. **맥락 제공**: 배경 정보 포함
3. **출력 형식 지정**: 원하는 형태 명시
4. **예시 제공**: 몇 가지 예시로 안내

> 프롬프트 품질 = 결과 품질

---

# 프롬프트 기본 패턴

## 구조화된 프롬프트

```
[역할 설정]
당신은 제조 품질 관리 전문가입니다.

[맥락]
온도 센서 데이터가 90°C를 초과했습니다.

[질문]
이 상황에서 취해야 할 조치를 3가지 알려주세요.

[출력 형식]
번호 목록으로 작성해주세요.
```

---

# 프롬프트 예시 1

## 불량 원인 분석

```
당신은 제조 공정 분석 전문가입니다.

다음은 불량이 발생한 시점의 센서 데이터입니다:
- 온도: 92°C (정상 범위: 80-90°C)
- 습도: 65% (정상 범위: 40-60%)
- 속도: 105 RPM (정상 범위: 95-105 RPM)

이 데이터를 분석하여 불량의 가장 가능성 높은 원인을
추론해주세요. 각 변수별로 영향을 분석해주세요.
```

---

# 프롬프트 예시 2

## 보고서 요약

```
다음 품질 보고서를 3문장으로 요약해주세요.
핵심 수치와 주요 문제점을 포함해주세요.

[보고서 내용]
2024년 1월 생산량 10,500대 중 불량률 2.3%...
주요 불량 유형: 외관 결함 45%, 기능 결함 30%...
...

[출력 형식]
- 요약:
- 주요 수치:
- 개선 필요 사항:
```

---

# 프롬프트 예시 3

## 코드 생성

```
Python으로 다음 기능을 구현해주세요:

1. CSV 파일에서 센서 데이터 읽기
2. 온도 컬럼의 평균, 최대, 최소 계산
3. 90도 초과 데이터 필터링
4. 결과를 새 CSV로 저장

pandas 라이브러리를 사용하고,
주석으로 각 단계를 설명해주세요.
```

---

# Few-shot 프롬프트

## 예시 제공

```
센서 데이터를 분석해서 상태를 판단해주세요.

예시 1:
입력: 온도=85, 습도=50
출력: 정상 (모든 값이 허용 범위 내)

예시 2:
입력: 온도=95, 습도=70
출력: 경고 (온도와 습도 모두 높음)

분석할 데이터:
입력: 온도=88, 습도=65
출력:
```

---

# 프롬프트 팁

## 더 나은 결과를 위해

1. **구체적으로**: "좋은 코드" → "에러 처리 포함된 Python 코드"
2. **단계별로**: 복잡한 작업은 나눠서 요청
3. **역할 부여**: "당신은 전문가입니다"
4. **형식 지정**: "표로 정리해주세요"
5. **제약 추가**: "100단어 이내로"

---

# 이론 정리

## LLM API 핵심

| 개념 | 설명 |
|------|------|
| LLM | 대규모 언어 모델 |
| API 호출 | client.chat.completions.create() |
| 메시지 구조 | system, user, assistant |
| 프롬프트 | AI에게 전달하는 지시문 |
| Few-shot | 예시 제공 방식 |

---

# - 실습편 -

## 21차시

**LLM API 호출 실습**

---

# 실습 개요

## LLM 활용 실습

### 목표
- LLM API 호출
- 효과적인 프롬프트 작성
- 제조 분야 활용

### 실습 환경
```python
from openai import OpenAI
# 또는
import anthropic
```

---

# 실습 1: 기본 API 호출

## OpenAI GPT

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "제조업에서 AI의 활용 사례 3가지를 알려주세요."}
    ]
)

print(response.choices[0].message.content)
```

---

# 실습 2: 시스템 프롬프트

## 역할 설정

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "당신은 20년 경력의 제조 품질 전문가입니다. 전문적이지만 쉽게 설명해주세요."},
        {"role": "user", "content": "SPC(통계적 공정 관리)가 무엇인가요?"}
    ]
)
```

---

# 실습 3: 데이터 분석 요청

## 센서 데이터 해석

```python
sensor_data = """
시간: 09:00, 온도: 85°C, 압력: 2.1bar, 진동: 0.5mm/s
시간: 10:00, 온도: 92°C, 압력: 2.3bar, 진동: 1.2mm/s
시간: 11:00, 온도: 95°C, 압력: 2.5bar, 진동: 2.1mm/s
"""

prompt = f"""
다음 센서 데이터를 분석해주세요:
{sensor_data}

1. 이상 징후가 있나요?
2. 있다면 어떤 조치가 필요한가요?
"""
```

---

# 실습 4: 보고서 생성

## 자동 보고서 작성

```python
data_summary = {
    "date": "2024-01-15",
    "production": 1050,
    "defects": 23,
    "defect_rate": "2.19%",
    "main_defect": "외관 결함"
}

prompt = f"""
다음 데이터로 일일 품질 보고서를 작성해주세요:
{data_summary}

형식:
- 제목
- 요약 (2-3문장)
- 주요 지표 표
- 개선 권고사항
"""
```

---

# 실습 5: 코드 생성

## Python 코드 요청

```python
prompt = """
pandas를 사용해서 다음 기능을 하는 Python 함수를 작성해주세요:

함수명: analyze_sensor_data
입력: CSV 파일 경로
기능:
1. 파일 읽기
2. 결측치 확인 및 처리
3. 기초 통계 계산 (평균, 표준편차, 최대, 최소)
4. 이상치 탐지 (평균 ± 3*표준편차)

주석을 상세히 달아주세요.
"""
```

---

# 실습 6: 대화 기록 유지

## 멀티턴 대화

```python
messages = [
    {"role": "system", "content": "제조 품질 상담사입니다."}
]

# 첫 질문
messages.append({"role": "user", "content": "불량률이 갑자기 올랐어요."})
response1 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
messages.append({"role": "assistant", "content": response1.choices[0].message.content})

# 추가 질문
messages.append({"role": "user", "content": "온도가 원인일 수 있을까요?"})
response2 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
```

---

# 실습 7: 응답 파싱

## JSON 형식 요청

```python
prompt = """
다음 센서값을 분석하고 JSON 형식으로 응답해주세요:
온도: 92°C, 습도: 65%

응답 형식:
{
    "status": "정상/경고/위험",
    "issues": ["이슈1", "이슈2"],
    "recommendations": ["권고1", "권고2"]
}
"""

# 응답 파싱
import json
result = json.loads(response.choices[0].message.content)
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] API 클라이언트 설정 (API 키)
- [ ] 메시지 구조 이해 (system/user/assistant)
- [ ] 시스템 프롬프트로 역할 설정
- [ ] 구조화된 프롬프트 작성
- [ ] 대화 기록 유지 (멀티턴)
- [ ] JSON 응답 파싱

---

# 다음 차시 예고

## 22차시: Streamlit으로 웹앱 만들기

### 학습 내용
- Streamlit 기초
- 대시보드 구축
- AI 모델 웹앱화

> AI를 **웹 서비스**로 만들기!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **LLM**: 자연어 이해/생성 AI
2. **API 호출**: messages 형식으로 대화
3. **시스템 프롬프트**: AI 역할 설정
4. **좋은 프롬프트**: 구체적, 맥락 제공, 형식 지정
5. **Few-shot**: 예시로 안내
6. **멀티턴**: 대화 기록 유지

---

# 감사합니다

## 21차시: LLM API와 프롬프트 작성법

**대규모 언어 모델을 제조 현장에 활용하는 법을 배웠습니다!**
