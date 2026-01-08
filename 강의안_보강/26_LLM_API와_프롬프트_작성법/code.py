"""
[26차시] LLM API와 프롬프트 작성법 - 실습 코드

학습 목표:
1. OpenAI/Claude API 사용법을 익힌다
2. 프롬프트 엔지니어링 기초를 이해한다
3. 제조업 활용 사례를 실습한다

실습 환경: Python 3.8+, openai, anthropic (선택)

참고: 실제 API 호출에는 API 키가 필요합니다.
      이 코드는 시뮬레이션 모드로 동작합니다.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

print("=" * 60)
print("[26차시] LLM API와 프롬프트 작성법")
print("=" * 60)

# ============================================================
# 1. API 키 관리 (중요!)
# ============================================================
print("\n" + "=" * 60)
print("1. API 키 관리 (중요!)")
print("=" * 60)

print("""
[API 키 관리 원칙]

절대 하면 안 되는 것:
  api_key = "sk-proj-xxxxx"  # 코드에 직접 입력 금지!

안전한 방법 1: 환경 변수
  export OPENAI_API_KEY="sk-proj-xxxxx"  # 터미널에서
  api_key = os.environ.get('OPENAI_API_KEY')  # 코드에서

안전한 방법 2: .env 파일
  # .env 파일 (git에 포함하지 않음)
  OPENAI_API_KEY=sk-proj-xxxxx

  # 코드
  from dotenv import load_dotenv
  load_dotenv()
  api_key = os.environ.get('OPENAI_API_KEY')
""")

# 환경 변수에서 API 키 읽기 시도
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# API 키 존재 여부 확인
has_openai = bool(OPENAI_API_KEY)
has_anthropic = bool(ANTHROPIC_API_KEY)

print(f"\nAPI 키 상태:")
print(f"  OpenAI: {'설정됨' if has_openai else '미설정 (시뮬레이션 모드)'}")
print(f"  Anthropic: {'설정됨' if has_anthropic else '미설정 (시뮬레이션 모드)'}")

# ============================================================
# 2. 실제 API 요청/응답 형식
# ============================================================
print("\n" + "=" * 60)
print("2. 실제 API 요청/응답 형식")
print("=" * 60)

print("""
[OpenAI API 요청 형식]

POST https://api.openai.com/v1/chat/completions
Headers:
  Authorization: Bearer sk-proj-xxxxx
  Content-Type: application/json

Body:
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}

[OpenAI API 응답 형식]

{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "gpt-4-0613",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
""")

# ============================================================
# 3. LLM 응답 시뮬레이터
# ============================================================
print("\n" + "=" * 60)
print("3. LLM 응답 시뮬레이터 (API 키 없을 때 사용)")
print("=" * 60)

class LLMSimulator:
    """API 키 없이 LLM 응답을 시뮬레이션"""

    def __init__(self):
        self.response_templates = {
            'analysis': """## 분석 결과

### 가능한 원인
1. **온도 상승**: 냉각 시스템 효율 저하 가능성
   - 냉각수 순환 펌프 점검 필요
   - 열교환기 효율 확인

2. **압력 변동**: 밸브 동작 불안정
   - 압력 조절 밸브 캘리브레이션 필요
   - 배관 누설 점검

3. **속도 변화**: 모터 베어링 마모
   - 진동 측정으로 베어링 상태 확인
   - 윤활유 점검

### 권장 조치
- 즉시: 냉각수 순환 확인 및 온도 모니터링 강화
- 단기: 센서 캘리브레이션 및 밸브 점검
- 장기: 예방 정비 일정 수립

### 위험도 평가
현재 상태는 **주의** 수준입니다. 24시간 내 점검을 권장합니다.""",

            'report': """## 품질 보고서

### 요약
금주 생산 라인 품질 지표는 전반적으로 양호하나,
일부 구간에서 개선이 필요합니다.

### 주요 지표
| 항목 | 이번 주 | 지난 주 | 변화 |
|------|--------|--------|------|
| 불량률 | 2.3% | 2.1% | +0.2% |
| 가동률 | 94.5% | 95.2% | -0.7% |
| 생산량 | 15,234개 | 14,890개 | +2.3% |

### 개선 필요 사항
1. 3번 라인 불량률 상승 원인 조사 필요
2. 설비 B의 예방 정비 일정 검토

### 권장 조치
주간 품질 회의에서 3번 라인 집중 검토 필요.""",

            'json': '{"status": "warning", "risk_score": 65, "anomalies": ["온도 상승", "진동 증가"], "recommendations": ["냉각 시스템 점검", "베어링 상태 확인"], "priority": "medium", "confidence": 0.85}'
        }

        # 토큰 사용량 시뮬레이션
        self.usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

    def generate(self, prompt: str, response_type: str = 'analysis') -> Dict:
        """프롬프트에 따른 시뮬레이션 응답 생성"""
        time.sleep(0.5)  # API 호출 시간 시뮬레이션

        # 프롬프트 토큰 추정 (대략 4글자 = 1토큰)
        prompt_tokens = len(prompt) // 4

        if 'json' in prompt.lower() or 'JSON' in prompt:
            content = self.response_templates['json']
        elif '보고서' in prompt or 'report' in prompt.lower():
            content = self.response_templates['report']
        else:
            content = self.response_templates['analysis']

        completion_tokens = len(content) // 4

        self.usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }

        # OpenAI API 응답 형식과 유사하게 반환
        return {
            'id': f'sim-{int(time.time())}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'gpt-4-simulation',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content
                },
                'finish_reason': 'stop'
            }],
            'usage': self.usage
        }

# 시뮬레이터 인스턴스
simulator = LLMSimulator()

# ============================================================
# 4. OpenAI API 구조 이해
# ============================================================
print("\n" + "=" * 60)
print("4. OpenAI API 구조 이해")
print("=" * 60)

print("""
[OpenAI API 기본 구조 - 공식 SDK 사용]

from openai import OpenAI

client = OpenAI(api_key='your-api-key')

response = client.chat.completions.create(
    model="gpt-4",                    # 모델 선택
    messages=[                        # 대화 메시지
        {"role": "system", "content": "역할 정의"},
        {"role": "user", "content": "사용자 질문"}
    ],
    temperature=0.7,                  # 창의성 (0~1)
    max_tokens=1000                   # 최대 응답 길이
)

answer = response.choices[0].message.content
usage = response.usage  # 토큰 사용량
""")

# OpenAI 스타일 래퍼 함수
def call_openai(messages: List[Dict], model: str = "gpt-4",
                temperature: float = 0.7, max_tokens: int = 1000) -> Dict:
    """
    OpenAI API 호출 (시뮬레이션 포함)

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
    dict : API 응답 (OpenAI 형식)
    """
    if has_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                'id': response.id,
                'model': response.model,
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': response.choices[0].message.content
                    },
                    'finish_reason': response.choices[0].finish_reason
                }],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"OpenAI API 에러: {e}")
            return simulator.generate(messages[-1]['content'])
    else:
        # 시뮬레이션 모드
        return simulator.generate(messages[-1]['content'])

# 응답에서 텍스트 추출 헬퍼 함수
def get_response_text(response: Dict) -> str:
    """API 응답에서 텍스트 추출"""
    return response['choices'][0]['message']['content']

# ============================================================
# 5. Claude API 구조 이해
# ============================================================
print("\n" + "=" * 60)
print("5. Claude API 구조 이해")
print("=" * 60)

print("""
[Claude API 기본 구조]

import anthropic

client = anthropic.Anthropic(api_key='your-api-key')

response = client.messages.create(
    model="claude-3-opus-20240229",   # 모델 선택
    max_tokens=1024,                   # 최대 응답 길이
    system="역할 정의",                # 시스템 프롬프트
    messages=[                         # 대화 메시지
        {"role": "user", "content": "사용자 질문"}
    ]
)

answer = response.content[0].text
usage = response.usage  # 토큰 사용량

[Claude API 응답 형식]

{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "응답 내용"}],
    "model": "claude-3-opus-20240229",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 20,
        "output_tokens": 10
    }
}
""")

# Claude 스타일 래퍼 함수
def call_claude(messages: List[Dict], system: str = None,
                model: str = "claude-3-sonnet-20240229",
                max_tokens: int = 1000) -> Dict:
    """
    Claude API 호출 (시뮬레이션 포함)
    """
    if has_anthropic:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "",
                messages=messages
            )
            return {
                'id': response.id,
                'model': response.model,
                'content': [{'type': 'text', 'text': response.content[0].text}],
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
        except Exception as e:
            print(f"Claude API 에러: {e}")
            return simulator.generate(messages[-1]['content'])
    else:
        # 시뮬레이션 모드
        sim_response = simulator.generate(messages[-1]['content'])
        return {
            'id': sim_response['id'],
            'model': 'claude-simulation',
            'content': [{'type': 'text', 'text': get_response_text(sim_response)}],
            'usage': {
                'input_tokens': sim_response['usage']['prompt_tokens'],
                'output_tokens': sim_response['usage']['completion_tokens']
            }
        }

# ============================================================
# 6. 기본 API 호출 테스트
# ============================================================
print("\n" + "=" * 60)
print("6. 기본 API 호출 테스트")
print("=" * 60)

# 기본 메시지
messages = [
    {"role": "system", "content": "당신은 제조업 품질 전문가입니다."},
    {"role": "user", "content": "온도가 250도로 높은데 원인이 뭘까요?"}
]

print("\n요청:")
for msg in messages:
    print(f"  [{msg['role']}]: {msg['content'][:50]}...")

print("\n응답 (시뮬레이션):")
response = call_openai(messages, temperature=0.3)
content = get_response_text(response)
print(content[:500] + "..." if len(content) > 500 else content)

print(f"\n토큰 사용량:")
print(f"  프롬프트: {response['usage']['prompt_tokens']}")
print(f"  응답: {response['usage']['completion_tokens']}")
print(f"  합계: {response['usage']['total_tokens']}")

# ============================================================
# 7. 프롬프트 엔지니어링 - 역할 지정
# ============================================================
print("\n" + "=" * 60)
print("7. 프롬프트 엔지니어링 - 역할 지정")
print("=" * 60)

# 상세한 역할 정의
system_prompt = """당신은 20년 경력의 제조업 품질 관리 전문가입니다.

전문 분야:
- 반도체 제조 공정
- 통계적 공정 관리 (SPC)
- 불량 원인 분석 (Root Cause Analysis)

응답 스타일:
- 데이터 기반으로 분석
- 실용적인 해결책 제시
- 기술적이지만 이해하기 쉽게 설명
- 우선순위를 명확히 제시
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "최근 웨이퍼 불량률이 2%에서 5%로 증가했습니다. 원인을 분석해주세요."}
]

print("\nSystem 프롬프트:")
print("-" * 40)
print(system_prompt[:200] + "...")

print("\n응답:")
response = call_openai(messages, temperature=0.3)
content = get_response_text(response)
print(content[:500] + "..." if len(content) > 500 else content)

# ============================================================
# 8. 프롬프트 엔지니어링 - 맥락 제공
# ============================================================
print("\n" + "=" * 60)
print("8. 프롬프트 엔지니어링 - 맥락 제공")
print("=" * 60)

context_prompt = """
[상황]
자동차 부품 제조 라인에서 용접 품질 불량이 증가했습니다.

[데이터]
- 기간: 2026년 1월 첫째 주
- 불량률 변화: 지난 주 2% -> 이번 주 5%
- 용접 전류: 평균 180A (기준: 175-185A)
- 용접 시간: 평균 2.3초 (기준: 2.0-2.5초)
- 용접 온도: 평균 850도 (기준: 800-900도)
- 불량 유형: 용접 강도 부족 70%, 기포 발생 20%, 기타 10%

[환경 변화]
- 이번 주부터 신규 원자재 공급업체 변경
- 월요일 오전 설비 정기 점검 실시

[질문]
불량률 증가의 가능한 원인과 우선순위별 조치 방안을 제시해주세요.
"""

messages = [
    {"role": "system", "content": "당신은 용접 공정 전문가입니다."},
    {"role": "user", "content": context_prompt}
]

print("\n맥락이 포함된 프롬프트:")
print("-" * 40)
print(context_prompt[:400] + "...")

print("\n응답:")
response = call_openai(messages, temperature=0.3)
content = get_response_text(response)
print(content[:500] + "..." if len(content) > 500 else content)

# ============================================================
# 9. 프롬프트 엔지니어링 - 출력 형식 지정
# ============================================================
print("\n" + "=" * 60)
print("9. 프롬프트 엔지니어링 - 출력 형식 지정")
print("=" * 60)

format_prompt = """
품질 이상 원인을 분석해주세요.

입력 데이터:
- 온도: 260도 (기준: 200-250도)
- 압력: 72kPa (기준: 40-70kPa)
- 진동: 8.5mm/s (기준: 0-7mm/s)

다음 형식으로 정확히 답변해주세요:

## 이상 항목 요약
| 항목 | 측정값 | 기준 | 상태 |
|------|--------|------|------|
| ... | ... | ... | ... |

## 원인 분석
1. **[원인 1 제목]**
   - 근거: ...
   - 영향: ...

2. **[원인 2 제목]**
   - 근거: ...
   - 영향: ...

## 권장 조치
1. 즉시 조치 (4시간 내): ...
2. 단기 대책 (1주 내): ...
3. 장기 개선 (1개월 내): ...

## 위험도
- 현재 수준: [높음/중간/낮음]
- 점수: [0-100]
"""

messages = [
    {"role": "system", "content": "당신은 품질 관리 전문가입니다."},
    {"role": "user", "content": format_prompt}
]

print("\n형식 지정 프롬프트:")
print("-" * 40)
print(format_prompt[:300] + "...")

print("\n응답:")
response = call_openai(messages, temperature=0.3)
content = get_response_text(response)
print(content[:600] + "..." if len(content) > 600 else content)

# ============================================================
# 10. 프롬프트 엔지니어링 - Few-shot 학습
# ============================================================
print("\n" + "=" * 60)
print("10. 프롬프트 엔지니어링 - Few-shot 학습")
print("=" * 60)

fewshot_prompt = """
센서 데이터의 이상 여부를 판단해주세요.

[기준값]
- 온도: 180-250도
- 압력: 40-70kPa
- 진동: 0-7mm/s

[예시 1]
입력: 온도 210도, 압력 55kPa, 진동 5mm/s
출력: 정상 - 모든 값이 기준 범위 내

[예시 2]
입력: 온도 280도, 압력 45kPa, 진동 6mm/s
출력: 이상 - 온도 초과 (280 > 250)

[예시 3]
입력: 온도 200도, 압력 75kPa, 진동 9mm/s
출력: 이상 - 압력 초과 (75 > 70), 진동 초과 (9 > 7)

이제 분석해주세요:
입력: 온도 245도, 압력 72kPa, 진동 6.5mm/s
출력:
"""

messages = [
    {"role": "user", "content": fewshot_prompt}
]

print("\nFew-shot 프롬프트:")
print("-" * 40)
print(fewshot_prompt[:400] + "...")

print("\n응답:")
response = call_openai(messages, temperature=0.1)
content = get_response_text(response)
print(content[:300] + "..." if len(content) > 300 else content)

# ============================================================
# 11. 프롬프트 엔지니어링 - JSON 출력
# ============================================================
print("\n" + "=" * 60)
print("11. 프롬프트 엔지니어링 - JSON 출력")
print("=" * 60)

json_prompt = """
센서 데이터를 분석하고 결과를 JSON 형식으로만 반환해주세요.

입력 데이터:
- 온도: 255도 (기준: 200-250도)
- 압력: 68kPa (기준: 40-70kPa)
- 진동: 8.2mm/s (기준: 0-7mm/s)
- 습도: 55% (기준: 40-60%)

다음 JSON 형식으로 정확히 반환해주세요. JSON 외 다른 텍스트는 포함하지 마세요:

{
    "timestamp": "분석 시각 (ISO 형식)",
    "overall_status": "normal 또는 warning 또는 critical",
    "risk_score": 0에서 100 사이 정수,
    "anomalies": [
        {
            "sensor": "센서명",
            "value": 측정값,
            "threshold": {"min": 최소값, "max": 최대값},
            "deviation": "초과량 또는 미달량"
        }
    ],
    "recommendations": ["권장 조치 목록"],
    "priority": "high 또는 medium 또는 low"
}
"""

messages = [
    {"role": "system", "content": "JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요."},
    {"role": "user", "content": json_prompt}
]

print("\nJSON 출력 프롬프트:")
print("-" * 40)
print(json_prompt[:300] + "...")

print("\n응답:")
response = call_openai(messages, temperature=0.1)
content = get_response_text(response)
print(content)

# JSON 파싱 시도
try:
    # 시뮬레이션 응답이 JSON인 경우 파싱
    if content.strip().startswith('{'):
        parsed = json.loads(content)
        print("\n파싱된 JSON:")
        print(f"  상태: {parsed.get('status', parsed.get('overall_status', 'N/A'))}")
        print(f"  위험 점수: {parsed.get('risk_score', 'N/A')}")
        if 'anomalies' in parsed:
            print(f"  이상 항목: {parsed.get('anomalies', [])}")
except json.JSONDecodeError:
    print("\n(JSON 파싱 불가 - 텍스트 응답)")

# ============================================================
# 12. 제조업 활용 - 불량 원인 분석
# ============================================================
print("\n" + "=" * 60)
print("12. 제조업 활용 - 불량 원인 분석")
print("=" * 60)

def analyze_defect(sensor_data: Dict, defect_type: str, defect_description: str) -> Dict:
    """
    불량 원인 분석 함수

    Parameters:
    -----------
    sensor_data : dict
        센서 데이터
    defect_type : str
        불량 유형
    defect_description : str
        불량 상세 설명

    Returns:
    --------
    dict : 분석 결과 (API 응답 포함)
    """
    prompt = f"""
[제조 공정 불량 원인 분석]

## 불량 정보
- 불량 유형: {defect_type}
- 상세 설명: {defect_description}

## 센서 데이터 (불량 발생 시점)
- 온도: {sensor_data.get('temperature', 'N/A')}도
- 압력: {sensor_data.get('pressure', 'N/A')}kPa
- 속도: {sensor_data.get('speed', 'N/A')}rpm
- 진동: {sensor_data.get('vibration', 'N/A')}mm/s
- 습도: {sensor_data.get('humidity', 'N/A')}%

## 요청사항
위 데이터를 바탕으로 다음을 분석해주세요:

1. **가장 가능성 높은 원인 3가지**
   - 각 원인의 근거
   - 센서 데이터와의 연관성

2. **각 원인별 해결 방법**
   - 즉시 조치
   - 근본 원인 해결

3. **재발 방지를 위한 권장 사항**
   - 모니터링 강화 항목
   - 예방 정비 제안
"""

    messages = [
        {"role": "system", "content": "당신은 제조 공정 품질 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai(messages, temperature=0.3)

    return {
        'analysis': get_response_text(response),
        'usage': response['usage'],
        'model': response.get('model', 'unknown')
    }

# 테스트
sensor_data = {
    'temperature': 265,
    'pressure': 72,
    'speed': 1450,
    'vibration': 8.5,
    'humidity': 58
}

print("\n불량 분석 테스트:")
print(f"센서 데이터: {sensor_data}")
print(f"불량 유형: 표면 스크래치")

result = analyze_defect(
    sensor_data,
    defect_type="표면 스크래치",
    defect_description="제품 표면에 0.5mm 깊이의 선형 스크래치 발생"
)

print("\n분석 결과:")
print(result['analysis'][:600] + "..." if len(result['analysis']) > 600 else result['analysis'])

# ============================================================
# 13. 제조업 활용 - 품질 보고서 생성
# ============================================================
print("\n" + "=" * 60)
print("13. 제조업 활용 - 품질 보고서 생성")
print("=" * 60)

def generate_quality_report(data: Dict) -> Dict:
    """
    품질 보고서 자동 생성

    Parameters:
    -----------
    data : dict
        보고서 데이터

    Returns:
    --------
    dict : 생성된 보고서와 메타데이터
    """
    prompt = f"""
다음 데이터를 바탕으로 주간 품질 보고서를 작성해주세요.

[보고서 데이터]
- 기간: {data.get('period', 'N/A')}
- 생산량: {data.get('production', 'N/A')}개
- 불량 수: {data.get('defect_count', 'N/A')}개
- 불량률: {data.get('defect_rate', 'N/A')}%
- 주요 불량 유형: {data.get('top_defects', [])}
- 라인별 불량률: {data.get('line_defects', {})}
- 전주 대비: {data.get('change', 'N/A')}

[보고서 형식]
## 주간 품질 보고서

### 1. 요약 (Executive Summary)
- 핵심 내용을 3줄 이내로

### 2. 주요 지표
- 표 형식으로 정리

### 3. 이상 항목 분석
- 기준 초과 항목 분석

### 4. 개선 필요 사항
- 우선순위별 정리

### 5. 다음 주 권장 조치
- 구체적인 액션 아이템
"""

    messages = [
        {"role": "system", "content": "당신은 품질 관리 보고서 작성 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai(messages, temperature=0.5)

    return {
        'report': get_response_text(response),
        'usage': response['usage'],
        'generated_at': datetime.now().isoformat()
    }

# 테스트
report_data = {
    'period': '2026년 1월 첫째 주',
    'production': 15000,
    'defect_count': 345,
    'defect_rate': 2.3,
    'top_defects': ['치수 불량 (45%)', '표면 불량 (30%)', '용접 불량 (25%)'],
    'line_defects': {'라인A': 1.8, '라인B': 2.5, '라인C': 2.8},
    'change': '+0.2% (전주 2.1%)'
}

print("\n보고서 데이터:")
for key, value in report_data.items():
    print(f"  {key}: {value}")

result = generate_quality_report(report_data)

print("\n생성된 보고서:")
print(result['report'][:800] + "..." if len(result['report']) > 800 else result['report'])

# ============================================================
# 14. 제조업 활용 - 이상 알림 해석
# ============================================================
print("\n" + "=" * 60)
print("14. 제조업 활용 - 이상 알림 해석")
print("=" * 60)

def explain_anomaly_alert(alert: Dict) -> Dict:
    """
    이상 탐지 알림을 현장 작업자가 이해할 수 있게 해석

    Parameters:
    -----------
    alert : dict
        알림 정보

    Returns:
    --------
    dict : 해석된 설명과 메타데이터
    """
    prompt = f"""
[이상 탐지 알림 해석 요청]

다음 시스템 알림을 현장 작업자가 이해할 수 있도록 쉽게 설명해주세요.

## 알림 정보
- 알림 코드: {alert.get('code', 'N/A')}
- 알림 메시지: {alert.get('message', 'N/A')}
- 발생 설비: {alert.get('equipment', 'N/A')}
- 발생 시각: {alert.get('timestamp', 'N/A')}
- 관련 센서값: {alert.get('sensor_values', {})}
- 심각도: {alert.get('severity', 'N/A')}

## 요청사항
현장 작업자를 위해 다음 형식으로 설명해주세요:

### 무슨 문제인가요?
[기술 용어 없이 쉽게 설명]

### 지금 당장 확인해야 할 것
1. ...
2. ...

### 응급 조치 방법
1. ...
2. ...

### 담당자 연락이 필요한 경우
[어떤 상황에서 연락해야 하는지]
"""

    messages = [
        {"role": "system", "content": "당신은 현장 작업자를 돕는 안내원입니다. 쉬운 말로 설명하세요."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai(messages, temperature=0.3)

    return {
        'explanation': get_response_text(response),
        'usage': response['usage']
    }

# 테스트
alert = {
    'code': 'TEMP_HIGH_001',
    'message': 'Temperature threshold exceeded in cooling zone',
    'equipment': 'CNC 가공기 3호기',
    'timestamp': '2026-01-08 14:23:45',
    'sensor_values': {'temperature': 285, 'threshold': 250, 'duration': '5분'},
    'severity': 'HIGH'
}

print("\n이상 알림:")
for key, value in alert.items():
    print(f"  {key}: {value}")

result = explain_anomaly_alert(alert)

print("\n현장 설명:")
print(result['explanation'][:600] + "..." if len(result['explanation']) > 600 else result['explanation'])

# ============================================================
# 15. ML 모델 + LLM 조합
# ============================================================
print("\n" + "=" * 60)
print("15. ML 모델 + LLM 조합")
print("=" * 60)

def hybrid_analysis(sensor_data: Dict, ml_prediction: str, ml_probability: float) -> Dict:
    """
    ML 예측 결과를 LLM으로 해석

    Parameters:
    -----------
    sensor_data : dict
        센서 데이터
    ml_prediction : str
        ML 모델 예측 결과
    ml_probability : float
        예측 확률

    Returns:
    --------
    dict : 종합 분석 결과
    """
    prompt = f"""
[ML 모델 예측 결과 해석]

## ML 모델 예측
- 예측 결과: {ml_prediction}
- 확신도: {ml_probability:.1%}

## 입력된 센서 데이터
- 온도: {sensor_data.get('temperature', 'N/A')}도
- 압력: {sensor_data.get('pressure', 'N/A')}kPa
- 속도: {sensor_data.get('speed', 'N/A')}rpm
- 진동: {sensor_data.get('vibration', 'N/A')}mm/s

## 정상 기준값
- 온도: 180-250도
- 압력: 40-70kPa
- 속도: 1400-1600rpm
- 진동: 0-7mm/s

## 요청
다음 내용을 분석해주세요:

1. **예측 결과 해석**
   - ML 모델이 왜 이런 예측을 했는지

2. **주의가 필요한 센서값**
   - 어떤 값이 예측에 가장 큰 영향을 미쳤는지

3. **권장 조치 사항**
   - 확률에 따른 대응 수준
"""

    messages = [
        {"role": "system", "content": "당신은 ML 모델 결과를 해석하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai(messages, temperature=0.3)

    return {
        'sensor_data': sensor_data,
        'ml_prediction': ml_prediction,
        'ml_probability': ml_probability,
        'interpretation': get_response_text(response),
        'usage': response['usage']
    }

# 테스트
sensor_data = {
    'temperature': 260,
    'pressure': 68,
    'speed': 1480,
    'vibration': 7.8
}

print("\n하이브리드 분석 테스트:")
print(f"센서 데이터: {sensor_data}")
print(f"ML 예측: 불량 (확률 85%)")

result = hybrid_analysis(sensor_data, '불량', 0.85)

print("\nML + LLM 종합 분석:")
print(result['interpretation'][:600] + "..." if len(result['interpretation']) > 600 else result['interpretation'])

# ============================================================
# 16. 대화형 분석 (멀티턴)
# ============================================================
print("\n" + "=" * 60)
print("16. 대화형 분석 (멀티턴)")
print("=" * 60)

class ConversationalAnalyzer:
    """대화형 품질 분석 에이전트"""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or """
당신은 제조업 품질 분석 전문가입니다.
사용자와 대화하면서 품질 문제를 분석합니다.
필요한 정보가 부족하면 추가 질문을 합니다.
"""
        self.conversation_history = []
        self.total_tokens = 0
        self.reset()

    def reset(self):
        """대화 초기화"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.total_tokens = 0

    def chat(self, user_message: str) -> Dict:
        """
        대화 진행

        Parameters:
        -----------
        user_message : str
            사용자 메시지

        Returns:
        --------
        dict : AI 응답과 메타데이터
        """
        # 사용자 메시지 추가
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 응답 생성
        response = call_openai(self.conversation_history, temperature=0.5)
        content = get_response_text(response)

        # 응답 기록
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })

        # 토큰 누적
        self.total_tokens += response['usage']['total_tokens']

        return {
            'response': content,
            'usage': response['usage'],
            'total_tokens': self.total_tokens,
            'turns': len([m for m in self.conversation_history if m['role'] == 'user'])
        }

# 대화형 분석 시뮬레이션
print("\n대화형 분석 예시:")
print("-" * 40)

analyzer = ConversationalAnalyzer()

# 첫 번째 메시지
result1 = analyzer.chat("3번 라인에서 불량이 증가하고 있어요.")
print(f"\n사용자: 3번 라인에서 불량이 증가하고 있어요.")
print(f"AI: {result1['response'][:200]}...")
print(f"(턴: {result1['turns']}, 누적 토큰: {result1['total_tokens']})")

# 두 번째 메시지 (후속 질문)
result2 = analyzer.chat("불량률이 2%에서 5%로 증가했고, 주로 표면 스크래치입니다.")
print(f"\n사용자: 불량률이 2%에서 5%로 증가했고, 주로 표면 스크래치입니다.")
print(f"AI: {result2['response'][:200]}...")
print(f"(턴: {result2['turns']}, 누적 토큰: {result2['total_tokens']})")

# ============================================================
# 17. 비용 관리
# ============================================================
print("\n" + "=" * 60)
print("17. 비용 관리")
print("=" * 60)

print("""
[LLM API 비용 관리]

1. 토큰 비용 (2024년 기준, 변동 가능)
   - GPT-4: 입력 $30/1M 토큰, 출력 $60/1M 토큰
   - GPT-3.5 Turbo: 입력 $0.50/1M 토큰, 출력 $1.50/1M 토큰
   - Claude 3 Opus: 입력 $15/1M 토큰, 출력 $75/1M 토큰
   - Claude 3 Sonnet: 입력 $3/1M 토큰, 출력 $15/1M 토큰

2. 비용 절감 전략
   - 프롬프트 최적화: 불필요한 내용 제거
   - 캐싱: 동일 요청 결과 재사용
   - 모델 선택: 작업에 맞는 모델 사용
   - max_tokens 제한: 필요한 만큼만 응답

3. 비용 추정 함수
""")

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

    model_key = model.lower()
    for key in pricing:
        if key in model_key:
            model_key = key
            break

    if model_key not in pricing:
        model_key = 'gpt-4'  # 기본값

    rates = pricing[model_key]

    input_cost = (prompt_tokens / 1_000_000) * rates['input']
    output_cost = (completion_tokens / 1_000_000) * rates['output']
    total_cost = input_cost + output_cost

    return {
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'input_cost_usd': input_cost,
        'output_cost_usd': output_cost,
        'total_cost_usd': total_cost,
        'total_cost_krw': total_cost * 1350  # 예상 환율
    }

# 비용 추정 예시
print("\n비용 추정 예시:")
cost = estimate_cost(500, 200, 'gpt-4')
print(f"  모델: {cost['model']}")
print(f"  입력 토큰: {cost['prompt_tokens']}")
print(f"  출력 토큰: {cost['completion_tokens']}")
print(f"  총 비용: ${cost['total_cost_usd']:.6f} (약 {cost['total_cost_krw']:.2f}원)")

# ============================================================
# 18. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("18. 핵심 정리")
print("=" * 60)

print("""
[26차시 핵심 정리]

1. LLM API 사용법
   - OpenAI: client.chat.completions.create()
   - Claude: client.messages.create()
   - API 키는 환경 변수로 관리

2. 프롬프트 엔지니어링
   - 역할 지정: system 메시지로 AI 역할 정의
   - 맥락 제공: 상황, 데이터, 배경 정보
   - 형식 지정: 원하는 출력 형태 명시
   - Few-shot: 예시로 원하는 패턴 학습
   - Chain of Thought: 단계적 사고 유도

3. 제조업 활용
   - 불량 원인 분석
   - 품질 보고서 자동 생성
   - 이상 알림 해석
   - ML + LLM 하이브리드

4. 주의사항
   - 환각: 사실 확인 필요
   - 비용: 모델 선택, 토큰 관리
   - 보안: 민감 정보 주의

5. 핵심 코드
   ```python
   from openai import OpenAI
   client = OpenAI()

   response = client.chat.completions.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": "역할"},
           {"role": "user", "content": "질문"}
       ],
       temperature=0.3
   )
   answer = response.choices[0].message.content
   usage = response.usage  # 토큰 사용량
   ```
""")

print("\n다음 차시 예고: Streamlit으로 웹앱 만들기")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
