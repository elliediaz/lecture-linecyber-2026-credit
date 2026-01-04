# [21차시] LLM API와 프롬프트 작성법 - 강사 스크립트

## 강의 정보
- **차시**: 21차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 21차시를 시작하겠습니다.
>
> 지난 시간에 API의 개념과 requests 라이브러리를 배웠습니다. GET/POST 요청, 응답 처리 방법을 익혔죠.
>
> 오늘은 그 연장선에서 **대규모 언어 모델(LLM) API**를 다룹니다. ChatGPT, Claude 같은 AI를 코드에서 활용하는 방법입니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, LLM API를 호출하는 방법을 익힙니다.
> 둘째, 효과적인 프롬프트를 작성합니다.
> 셋째, 제조 현장에서 LLM을 활용합니다.

---

### 핵심 내용 (8분)

#### LLM이란? [2min]

> **LLM**은 Large Language Model, 대규모 언어 모델입니다.
>
> 엄청난 양의 텍스트로 학습된 AI예요. 인터넷의 책, 문서, 대화 등을 학습해서 언어를 이해하고 생성합니다.
>
> 대표적인 모델로 OpenAI의 **GPT**(ChatGPT), Anthropic의 **Claude**, Google의 **Gemini**, Meta의 **LLaMA**가 있어요.
>
> 특징은 **자연어로 대화**한다는 거예요. 코드나 특수 문법 없이 그냥 한국어나 영어로 질문하면 자연어로 대답합니다.

#### LLM의 활용 [1.5min]

> LLM은 다양한 작업을 수행할 수 있어요.
>
> 텍스트 생성, 요약, 번역, 질의응답, 코드 작성, 데이터 분석까지 가능합니다.
>
> 제조 현장에서는 어떻게 쓸 수 있을까요?
>
> 문서 작업으로는 작업 지시서 생성, 품질 보고서 요약, 매뉴얼 Q&A가 있어요.
> 데이터 분석에서는 불량 원인 분석 지원, 센서 데이터 해석이 가능합니다.
> 다국어 번역, 기술 문서 작성에도 활용됩니다.

#### OpenAI API [2min]

> 가장 유명한 LLM API인 OpenAI를 먼저 보겠습니다.
>
> ```python
> from openai import OpenAI
>
> client = OpenAI(api_key="your-api-key")
>
> response = client.chat.completions.create(
>     model="gpt-4",
>     messages=[
>         {"role": "system", "content": "You are a helpful assistant."},
>         {"role": "user", "content": "안녕하세요!"}
>     ]
> )
> ```
>
> `messages` 리스트에 대화 내용을 넣어요. `role`은 세 가지가 있습니다.
>
> **system**은 AI의 역할을 설정해요. "당신은 제조 전문가입니다" 같은 지시를 줍니다.
> **user**는 사용자 질문이에요.
> **assistant**는 AI의 응답입니다. 대화 기록을 유지할 때 사용해요.

#### Claude API [1min]

> Anthropic의 Claude도 비슷한 구조입니다.
>
> ```python
> import anthropic
>
> client = anthropic.Anthropic(api_key="your-api-key")
>
> message = client.messages.create(
>     model="claude-3-sonnet-20240229",
>     messages=[{"role": "user", "content": "제조 불량의 주요 원인을 알려주세요."}]
> )
> ```
>
> 라이브러리만 다르고 개념은 같아요. 메시지 리스트를 보내고 응답을 받습니다.

#### 프롬프트 작성법 [1.5min]

> **프롬프트**는 AI에게 전달하는 지시문입니다.
>
> 프롬프트 품질이 결과 품질을 결정해요. 잘 쓰면 좋은 답변, 못 쓰면 이상한 답변이 나옵니다.
>
> 좋은 프롬프트의 조건 네 가지입니다.
>
> 첫째, **명확한 지시**. 무엇을 원하는지 구체적으로 씁니다.
> 둘째, **맥락 제공**. 배경 정보를 포함합니다.
> 셋째, **출력 형식 지정**. 원하는 형태를 명시합니다.
> 넷째, **예시 제공**. 몇 가지 예시로 안내합니다.
>
> "좋은 코드 써줘"보다 "에러 처리가 포함된 Python 코드로 CSV 파일을 읽는 함수를 작성해주세요"가 훨씬 좋은 프롬프트예요.

---

## 실습편 (15-20분)

### 실습 소개 [2min]

> 이제 실습 시간입니다. LLM API를 직접 호출해봅니다.
>
> **실습 목표**입니다.
> 1. OpenAI 또는 Claude API를 호출합니다.
> 2. 시스템 프롬프트로 역할을 설정합니다.
> 3. 제조 분야에 맞는 프롬프트를 작성합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> from openai import OpenAI
> # 또는
> import anthropic
> ```
>
> API 키가 필요합니다. 환경 변수로 설정하세요.

### 실습 1: 기본 API 호출 [3min]

> 첫 번째 실습입니다. 기본 API를 호출합니다.
>
> ```python
> from openai import OpenAI
> import os
>
> client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
>
> response = client.chat.completions.create(
>     model="gpt-3.5-turbo",
>     messages=[
>         {"role": "user", "content": "제조업에서 AI의 활용 사례 3가지를 알려주세요."}
>     ]
> )
>
> print(response.choices[0].message.content)
> ```
>
> `response.choices[0].message.content`로 응답 텍스트를 가져옵니다.

### 실습 2: 시스템 프롬프트 [2min]

> 두 번째 실습입니다. 시스템 프롬프트로 AI에게 역할을 부여합니다.
>
> ```python
> messages=[
>     {"role": "system", "content": "당신은 20년 경력의 제조 품질 전문가입니다. 전문적이지만 쉽게 설명해주세요."},
>     {"role": "user", "content": "SPC(통계적 공정 관리)가 무엇인가요?"}
> ]
> ```
>
> 시스템 프롬프트에 역할, 성격, 답변 스타일을 지정하면 일관된 응답을 받을 수 있어요.

### 실습 3: 데이터 분석 요청 [3min]

> 세 번째 실습입니다. 센서 데이터를 LLM에게 분석 요청합니다.
>
> ```python
> sensor_data = """
> 시간: 09:00, 온도: 85°C, 압력: 2.1bar, 진동: 0.5mm/s
> 시간: 10:00, 온도: 92°C, 압력: 2.3bar, 진동: 1.2mm/s
> 시간: 11:00, 온도: 95°C, 압력: 2.5bar, 진동: 2.1mm/s
> """
>
> prompt = f"""
> 다음 센서 데이터를 분석해주세요:
> {sensor_data}
>
> 1. 이상 징후가 있나요?
> 2. 있다면 어떤 조치가 필요한가요?
> """
> ```
>
> f-string으로 데이터를 프롬프트에 포함시킵니다. 구조화된 질문이 좋은 답변을 이끌어냅니다.

### 실습 4: 보고서 생성 [3min]

> 네 번째 실습입니다. LLM에게 보고서 작성을 요청합니다.
>
> ```python
> data_summary = {
>     "date": "2024-01-15",
>     "production": 1050,
>     "defects": 23,
>     "defect_rate": "2.19%",
>     "main_defect": "외관 결함"
> }
>
> prompt = f"""
> 다음 데이터로 일일 품질 보고서를 작성해주세요:
> {data_summary}
>
> 형식:
> - 제목
> - 요약 (2-3문장)
> - 주요 지표 표
> - 개선 권고사항
> """
> ```
>
> 출력 형식을 명확히 지정하면 원하는 구조의 답변을 받을 수 있어요.

### 실습 5: 멀티턴 대화 [3min]

> 다섯 번째 실습입니다. 대화 기록을 유지해서 이어지는 대화를 합니다.
>
> ```python
> messages = [
>     {"role": "system", "content": "제조 품질 상담사입니다."}
> ]
>
> # 첫 질문
> messages.append({"role": "user", "content": "불량률이 갑자기 올랐어요."})
> response1 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
> messages.append({"role": "assistant", "content": response1.choices[0].message.content})
>
> # 추가 질문 (이전 대화 기억)
> messages.append({"role": "user", "content": "온도가 원인일 수 있을까요?"})
> response2 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
> ```
>
> `messages` 리스트에 대화 기록을 계속 추가하면 AI가 맥락을 기억합니다.

### 실습 6: JSON 형식 응답 [2min]

> 여섯 번째 실습입니다. 구조화된 JSON 형식으로 응답을 받습니다.
>
> ```python
> prompt = """
> 다음 센서값을 분석하고 JSON 형식으로 응답해주세요:
> 온도: 92°C, 습도: 65%
>
> 응답 형식:
> {
>     "status": "정상/경고/위험",
>     "issues": ["이슈1", "이슈2"],
>     "recommendations": ["권고1", "권고2"]
> }
> """
>
> import json
> result = json.loads(response.choices[0].message.content)
> ```
>
> JSON으로 받으면 후처리가 쉬워요. 프로그램에서 바로 활용할 수 있습니다.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **LLM**은 대규모 언어 모델로, 자연어로 대화하는 AI입니다.
>
> **API 호출**은 `client.chat.completions.create()`로 합니다. messages 리스트에 대화를 담아 보내요.
>
> **메시지 구조**는 system(역할 설정), user(질문), assistant(응답) 세 가지 역할이 있습니다.
>
> **좋은 프롬프트**는 구체적이고, 맥락을 제공하고, 출력 형식을 지정합니다.
>
> **멀티턴 대화**는 messages 리스트에 기록을 유지해서 맥락을 이어갑니다.

#### 다음 차시 예고 [1min]

> 다음 22차시에서는 **Streamlit으로 웹앱 만들기**를 배웁니다.
>
> 지금까지 만든 AI 모델과 LLM을 웹 앱으로 만들어서 누구나 사용할 수 있게 합니다. Python 코드 몇 줄로 대시보드를 만들 수 있어요.

#### 마무리 [0.5min]

> 대규모 언어 모델을 API로 활용하는 법을 배웠습니다. 프롬프트 작성이 중요하다는 것, 꼭 기억하세요! 수고하셨습니다.

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- API 키 (OpenAI 또는 Claude)

### 주의사항
- API 키 보안 강조 (환경 변수 사용)
- API 호출 비용 안내
- 프롬프트 품질의 중요성

### 예상 질문
1. "API 키는 어디서 받나요?"
   → OpenAI: platform.openai.com, Anthropic: console.anthropic.com

2. "비용이 얼마나 드나요?"
   → GPT-3.5-turbo는 저렴, GPT-4는 비쌈. 토큰당 과금. 테스트용으로 충분히 저렴

3. "Claude랑 GPT 뭐가 더 좋아요?"
   → 용도에 따라 다름. 둘 다 훌륭함. 직접 비교해보시길

4. "한국어도 잘 되나요?"
   → 네, 둘 다 한국어 지원. 다만 영어가 약간 더 정확할 수 있음

5. "회사에서 써도 되나요?"
   → 보안 정책 확인 필요. API 사용은 대개 가능, 데이터 유출 주의
