# [21차시] LLM API와 프롬프트 작성법 - 실습 코드

import os
import json

print("=" * 60)
print("21차시: LLM API와 프롬프트 작성법")
print("대규모 언어 모델을 API로 활용합니다!")
print("=" * 60)
print()

# ============================================================
# 참고: API 키 설정
# ============================================================
print("=" * 50)
print("API 키 설정 방법")
print("=" * 50)

print("""
API 키는 환경 변수로 설정하세요!

[터미널에서 설정]
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

[Python에서 읽기]
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

[.env 파일 사용]
pip install python-dotenv

# .env 파일
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Python
from dotenv import load_dotenv
load_dotenv()
""")
print()


# ============================================================
# 실습 1: OpenAI API 기본 호출 (시뮬레이션)
# ============================================================
print("=" * 50)
print("실습 1: OpenAI API 기본 호출")
print("=" * 50)

# 실제 코드 (API 키 필요)
openai_code = '''
from openai import OpenAI
import os

# 클라이언트 생성
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# API 호출
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "제조업에서 AI의 활용 사례 3가지를 알려주세요."}
    ]
)

# 응답 출력
print(response.choices[0].message.content)
'''

print("OpenAI API 호출 코드:")
print("-" * 50)
print(openai_code)

# 시뮬레이션 응답
print("\n[시뮬레이션 응답]")
print("""
제조업에서 AI 활용 사례 3가지:

1. **품질 검사 자동화**: 비전 AI로 제품 외관 결함을 실시간 검출합니다.

2. **예측 유지보수**: 설비 센서 데이터를 분석해 고장을 미리 예측합니다.

3. **공정 최적화**: 생산 데이터를 분석해 최적의 공정 조건을 찾습니다.
""")
print()


# ============================================================
# 실습 2: 시스템 프롬프트로 역할 설정
# ============================================================
print("=" * 50)
print("실습 2: 시스템 프롬프트로 역할 설정")
print("=" * 50)

system_prompt_code = '''
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "당신은 20년 경력의 제조 품질 전문가입니다. 전문적이지만 쉽게 설명해주세요."
        },
        {
            "role": "user",
            "content": "SPC(통계적 공정 관리)가 무엇인가요?"
        }
    ]
)
'''

print("시스템 프롬프트 사용 코드:")
print("-" * 50)
print(system_prompt_code)

print("\n[시뮬레이션 응답]")
print("""
SPC(Statistical Process Control)는 공정을 통계적으로 관리하는 방법입니다.

쉽게 말하면, 생산 중에 데이터를 계속 측정하고 그래프(관리도)로 그려서
"이 공정이 정상인가?"를 판단하는 거예요.

예를 들어, 부품 두께를 계속 측정해서 평균에서 너무 벗어나면
"뭔가 이상하다!"고 알려주는 겁니다. 불량이 나오기 전에 미리 조치할 수 있죠.
""")
print()


# ============================================================
# 실습 3: Claude API 호출 (시뮬레이션)
# ============================================================
print("=" * 50)
print("실습 3: Claude API 호출")
print("=" * 50)

claude_code = '''
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "제조 불량의 주요 원인을 알려주세요."}
    ]
)

print(message.content[0].text)
'''

print("Claude API 호출 코드:")
print("-" * 50)
print(claude_code)

print("\n[시뮬레이션 응답]")
print("""
제조 불량의 주요 원인은 크게 4가지로 분류할 수 있습니다:

1. **설비 요인**: 장비 마모, 정밀도 저하, 유지보수 부족
2. **재료 요인**: 원자재 품질 불량, 규격 미달
3. **작업 요인**: 작업자 숙련도, 표준 작업 미준수
4. **환경 요인**: 온도, 습도, 진동 등 환경 조건 변화

이 중 설비 요인이 가장 높은 비율을 차지하는 경우가 많습니다.
""")
print()


# ============================================================
# 실습 4: 센서 데이터 분석 요청
# ============================================================
print("=" * 50)
print("실습 4: 센서 데이터 분석 요청")
print("=" * 50)

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

print("프롬프트:")
print("-" * 50)
print(prompt)

print("\n[시뮬레이션 응답]")
print("""
## 센서 데이터 분석 결과

### 1. 이상 징후 분석

**예, 이상 징후가 있습니다.**

| 항목 | 09:00 | 10:00 | 11:00 | 추세 |
|------|-------|-------|-------|------|
| 온도 | 85°C | 92°C | 95°C | 상승 (+10°C) |
| 압력 | 2.1bar | 2.3bar | 2.5bar | 상승 (+0.4bar) |
| 진동 | 0.5mm/s | 1.2mm/s | 2.1mm/s | 급상승 (4배) |

**진동 수치가 가장 우려됩니다.** 2시간 만에 4배 증가했습니다.

### 2. 권고 조치

1. **즉시**: 설비 가동 중단 검토
2. **점검**: 베어링, 축 정렬 상태 확인
3. **모니터링**: 다음 30분간 집중 모니터링
4. **기록**: 이상 발생 시점 및 조건 기록
""")
print()


# ============================================================
# 실습 5: 보고서 자동 생성
# ============================================================
print("=" * 50)
print("실습 5: 보고서 자동 생성")
print("=" * 50)

data_summary = {
    "date": "2024-01-15",
    "production": 1050,
    "defects": 23,
    "defect_rate": "2.19%",
    "main_defect": "외관 결함"
}

prompt = f"""
다음 데이터로 일일 품질 보고서를 작성해주세요:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

형식:
- 제목
- 요약 (2-3문장)
- 주요 지표 표
- 개선 권고사항
"""

print("프롬프트:")
print("-" * 50)
print(prompt)

print("\n[시뮬레이션 응답]")
print("""
# 일일 품질 보고서

## 2024년 1월 15일

### 요약
금일 생산량 1,050대 중 23건의 불량이 발생하여 불량률 2.19%를 기록했습니다.
주요 불량 유형은 외관 결함이며, 목표 불량률 2% 대비 0.19%p 초과했습니다.

### 주요 지표

| 항목 | 수치 |
|------|------|
| 생산량 | 1,050대 |
| 불량 수 | 23건 |
| 불량률 | 2.19% |
| 주요 불량 | 외관 결함 |

### 개선 권고사항

1. 외관 검사 기준 재점검
2. 작업자 품질 교육 실시
3. 표면 처리 공정 조건 확인
""")
print()


# ============================================================
# 실습 6: 멀티턴 대화
# ============================================================
print("=" * 50)
print("실습 6: 멀티턴 대화 (대화 기록 유지)")
print("=" * 50)

multiturn_code = '''
messages = [
    {"role": "system", "content": "제조 품질 상담사입니다."}
]

# 첫 번째 질문
messages.append({"role": "user", "content": "불량률이 갑자기 올랐어요."})
response1 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
assistant_reply1 = response1.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_reply1})

print("AI:", assistant_reply1)

# 두 번째 질문 (이전 대화 기억함)
messages.append({"role": "user", "content": "온도가 원인일 수 있을까요?"})
response2 = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
assistant_reply2 = response2.choices[0].message.content

print("AI:", assistant_reply2)
'''

print("멀티턴 대화 코드:")
print("-" * 50)
print(multiturn_code)

print("\n[시뮬레이션 대화]")
print("-" * 50)
print("사용자: 불량률이 갑자기 올랐어요.")
print()
print("AI: 불량률이 갑자기 상승했군요. 몇 가지 확인이 필요합니다.")
print("    - 언제부터 상승했나요?")
print("    - 특정 제품이나 라인에서만 발생하나요?")
print("    - 최근 변경된 것이 있나요? (재료, 설비, 작업자 등)")
print()
print("사용자: 온도가 원인일 수 있을까요?")
print()
print("AI: 네, 온도는 불량의 주요 원인 중 하나입니다.")
print("    특히 플라스틱이나 금속 가공에서 온도 변화는 치수 변형,")
print("    표면 품질 저하를 일으킬 수 있습니다.")
print("    최근 온도 데이터를 확인해보시고, 정상 범위를 벗어난")
print("    시점이 있는지 살펴보세요.")
print()


# ============================================================
# 실습 7: JSON 형식 응답 요청
# ============================================================
print("=" * 50)
print("실습 7: JSON 형식 응답 요청")
print("=" * 50)

json_prompt = """
다음 센서값을 분석하고 JSON 형식으로 응답해주세요:
온도: 92°C, 습도: 65%

응답 형식:
{
    "status": "정상/경고/위험",
    "issues": ["이슈1", "이슈2"],
    "recommendations": ["권고1", "권고2"]
}
"""

print("프롬프트:")
print("-" * 50)
print(json_prompt)

# 시뮬레이션 JSON 응답
simulated_json = {
    "status": "경고",
    "issues": [
        "온도가 정상 범위(80-90°C) 초과",
        "습도가 권장 범위(40-60%) 초과"
    ],
    "recommendations": [
        "냉각 시스템 점검",
        "환기 시스템 가동 또는 제습기 사용"
    ]
}

print("\n[시뮬레이션 JSON 응답]")
print(json.dumps(simulated_json, ensure_ascii=False, indent=2))

print("\n[JSON 파싱 후 활용]")
print(f"상태: {simulated_json['status']}")
print(f"이슈 수: {len(simulated_json['issues'])}건")
for i, issue in enumerate(simulated_json['issues'], 1):
    print(f"  {i}. {issue}")
print()


# ============================================================
# 실습 8: 프롬프트 템플릿
# ============================================================
print("=" * 50)
print("실습 8: 프롬프트 템플릿 만들기")
print("=" * 50)

def create_analysis_prompt(sensor_data: dict) -> str:
    """센서 데이터 분석을 위한 프롬프트 생성"""
    return f"""
당신은 제조 공정 분석 전문가입니다.

다음 센서 데이터를 분석해주세요:
- 온도: {sensor_data.get('temperature', 'N/A')}°C
- 습도: {sensor_data.get('humidity', 'N/A')}%
- 압력: {sensor_data.get('pressure', 'N/A')}bar

정상 범위:
- 온도: 80-90°C
- 습도: 40-60%
- 압력: 2.0-2.5bar

다음 형식으로 분석 결과를 제공해주세요:
1. 각 센서별 상태 (정상/주의/경고)
2. 종합 판단
3. 권고 조치
"""

# 템플릿 사용
sensor_reading = {
    "temperature": 88,
    "humidity": 55,
    "pressure": 2.3
}

prompt = create_analysis_prompt(sensor_reading)
print("생성된 프롬프트:")
print("-" * 50)
print(prompt)
print()


# ============================================================
# 실습 9: 에러 처리
# ============================================================
print("=" * 50)
print("실습 9: API 호출 에러 처리")
print("=" * 50)

error_handling_code = '''
from openai import OpenAI, OpenAIError
import os

def safe_llm_call(prompt: str) -> str:
    """안전한 LLM API 호출"""
    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            timeout=30  # 30초 타임아웃
        )

        return response.choices[0].message.content

    except OpenAIError as e:
        print(f"OpenAI API 오류: {e}")
        return None
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
        return None

# 사용 예시
result = safe_llm_call("제조 품질 개선 방법 3가지")
if result:
    print(result)
else:
    print("API 호출 실패")
'''

print("에러 처리 코드:")
print("-" * 50)
print(error_handling_code)
print()


# ============================================================
# 실습 10: LLM 클라이언트 클래스
# ============================================================
print("=" * 50)
print("실습 10: 재사용 가능한 LLM 클라이언트 클래스")
print("=" * 50)

llm_client_code = '''
class ManufacturingLLM:
    """제조 현장용 LLM 클라이언트"""

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = "당신은 제조 공정 전문가입니다."

    def analyze_sensor(self, sensor_data: dict) -> str:
        """센서 데이터 분석"""
        prompt = f"센서 데이터: {sensor_data}\\n이상 여부를 분석해주세요."
        return self._call(prompt)

    def generate_report(self, data: dict) -> str:
        """품질 보고서 생성"""
        prompt = f"데이터: {data}\\n품질 보고서를 작성해주세요."
        return self._call(prompt)

    def troubleshoot(self, problem: str) -> str:
        """문제 해결 지원"""
        prompt = f"문제: {problem}\\n원인과 해결책을 알려주세요."
        return self._call(prompt)

    def _call(self, prompt: str) -> str:
        """API 호출"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# 사용 예시
# llm = ManufacturingLLM(api_key=os.environ.get('OPENAI_API_KEY'))
#
# # 센서 분석
# result = llm.analyze_sensor({"temp": 92, "humidity": 65})
# print(result)
#
# # 문제 해결
# result = llm.troubleshoot("생산 라인 2에서 불량률이 5%로 상승")
# print(result)
'''

print("LLM 클라이언트 클래스:")
print("-" * 50)
print(llm_client_code)
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              LLM API 활용 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ OpenAI API                                         │
│     from openai import OpenAI                         │
│     client = OpenAI(api_key=...)                      │
│     response = client.chat.completions.create(        │
│         model="gpt-3.5-turbo",                        │
│         messages=[...]                                │
│     )                                                 │
│                                                        │
│  ▶ 메시지 구조                                         │
│     messages = [                                      │
│         {{"role": "system", "content": "역할 설정"}},  │
│         {{"role": "user", "content": "질문"}},         │
│         {{"role": "assistant", "content": "응답"}}     │
│     ]                                                 │
│                                                        │
│  ▶ 좋은 프롬프트                                       │
│     1. 명확한 지시 (구체적으로)                        │
│     2. 맥락 제공 (배경 정보)                           │
│     3. 출력 형식 지정                                  │
│     4. 예시 제공 (Few-shot)                           │
│                                                        │
│  ▶ 멀티턴 대화                                         │
│     messages 리스트에 대화 기록 계속 추가              │
│     AI가 이전 맥락을 기억함                            │
│                                                        │
│  ▶ API 키 보안                                         │
│     ❌ 코드에 직접 쓰지 말 것!                         │
│     ✅ 환경 변수 또는 .env 파일 사용                   │
│                                                        │
│  ★ 프롬프트 품질 = 결과 품질                          │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: Streamlit으로 웹앱 만들기
""")

print("=" * 60)
print("21차시 실습 완료!")
print("대규모 언어 모델을 제조 현장에 활용하는 법을 배웠습니다!")
print("=" * 60)
