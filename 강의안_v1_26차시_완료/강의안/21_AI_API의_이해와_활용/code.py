"""
[21차시] AI API의 이해와 활용 - 실습 코드
학습목표: API 개념 이해, requests 라이브러리 사용, JSON 데이터 처리
"""

import json
import os

# requests 라이브러리 (설치: pip install requests)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests 라이브러리가 없습니다. pip install requests")

# ============================================================
# 1. API 개념
# ============================================================
print("=" * 50)
print("1. API란?")
print("=" * 50)

print("""
API (Application Programming Interface)
= 프로그램 간 데이터를 주고받는 약속/인터페이스

┌─────────────┐                    ┌─────────────┐
│  내 프로그램  │  ←─── API ───→   │  외부 서비스  │
│   (클라이언트) │                   │   (서버)     │
└─────────────┘                    └─────────────┘

▶ 비유: 레스토랑의 웨이터
   - 손님(클라이언트)이 메뉴(요청)를 웨이터(API)에게 전달
   - 웨이터가 주방(서버)에 주문 전달
   - 음식(응답)을 손님에게 전달
""")

# ============================================================
# 2. JSON 형식
# ============================================================
print("\n" + "=" * 50)
print("2. JSON 형식")
print("=" * 50)

# Python 딕셔너리
data = {
    "model_name": "품질 예측 모델",
    "version": "1.0",
    "prediction": {
        "result": "정상",
        "confidence": 0.95
    },
    "features": [85, 50, 100, 1.0]
}

print("▶ Python 딕셔너리:")
print(data)

# 딕셔너리 → JSON 문자열
json_string = json.dumps(data, ensure_ascii=False, indent=2)
print("\n▶ JSON 문자열로 변환:")
print(json_string)

# JSON 문자열 → 딕셔너리
parsed_data = json.loads(json_string)
print("\n▶ 다시 딕셔너리로:")
print(f"   모델명: {parsed_data['model_name']}")
print(f"   예측 결과: {parsed_data['prediction']['result']}")

# ============================================================
# 3. HTTP 요청 기본
# ============================================================
print("\n" + "=" * 50)
print("3. HTTP 메서드")
print("=" * 50)

print("""
▶ REST API의 주요 HTTP 메서드:

  메서드    │  용도          │  예시
  ─────────────────────────────────────────
  GET      │  데이터 조회    │  목록 가져오기
  POST     │  데이터 생성    │  예측 요청
  PUT      │  데이터 수정    │  정보 업데이트
  DELETE   │  데이터 삭제    │  항목 삭제

★ AI API는 주로 POST를 사용 (데이터를 보내고 결과를 받음)
""")

if REQUESTS_AVAILABLE:
    # ============================================================
    # 4. requests 라이브러리 사용법
    # ============================================================
    print("\n" + "=" * 50)
    print("4. requests 라이브러리")
    print("=" * 50)

    # GET 요청 예시 (공개 API)
    print("▶ GET 요청 예시 (JSONPlaceholder API):")
    try:
        response = requests.get(
            'https://jsonplaceholder.typicode.com/todos/1',
            timeout=10
        )
        print(f"   상태 코드: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   응답 데이터: {data}")
    except requests.exceptions.RequestException as e:
        print(f"   요청 실패: {e}")

    # POST 요청 예시
    print("\n▶ POST 요청 예시:")
    try:
        post_data = {
            'title': '예측 요청',
            'body': '온도=85, 습도=50',
            'userId': 1
        }
        response = requests.post(
            'https://jsonplaceholder.typicode.com/posts',
            json=post_data,
            timeout=10
        )
        print(f"   상태 코드: {response.status_code}")
        if response.status_code == 201:  # 201 = Created
            result = response.json()
            print(f"   생성된 ID: {result.get('id')}")
    except requests.exceptions.RequestException as e:
        print(f"   요청 실패: {e}")

    # ============================================================
    # 5. 에러 처리
    # ============================================================
    print("\n" + "=" * 50)
    print("5. 에러 처리")
    print("=" * 50)

    def safe_api_call(url, method='GET', data=None, timeout=10):
        """안전한 API 호출 함수"""
        try:
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            else:
                return None, f"지원하지 않는 메서드: {method}"

            response.raise_for_status()  # 4xx, 5xx 에러면 예외 발생
            return response.json(), None

        except requests.exceptions.Timeout:
            return None, "시간 초과 (Timeout)"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP 에러: {e}"
        except requests.exceptions.ConnectionError:
            return None, "연결 실패 (네트워크 확인)"
        except requests.exceptions.RequestException as e:
            return None, f"요청 실패: {e}"

    # 테스트
    data, error = safe_api_call('https://jsonplaceholder.typicode.com/posts/1')
    if error:
        print(f"▶ 에러 발생: {error}")
    else:
        print(f"▶ 성공: {data['title'][:30]}...")

    # 잘못된 URL 테스트
    data, error = safe_api_call('https://invalid-url-12345.com/api', timeout=3)
    print(f"▶ 잘못된 URL 테스트: {error}")

# ============================================================
# 6. API Key 관리
# ============================================================
print("\n" + "=" * 50)
print("6. API Key 보안")
print("=" * 50)

print("""
⚠️ API Key를 코드에 직접 작성하면 안됩니다!

❌ 잘못된 방법:
   API_KEY = 'sk-1234567890abcdef'  # 코드에 노출!

✅ 올바른 방법 1: 환경 변수
   import os
   API_KEY = os.environ.get('MY_API_KEY')

✅ 올바른 방법 2: .env 파일
   # .env 파일 내용
   MY_API_KEY=sk-1234567890abcdef

   # Python 코드
   from dotenv import load_dotenv
   load_dotenv()
   API_KEY = os.getenv('MY_API_KEY')

★ .env 파일은 .gitignore에 추가해서 Git에 올리지 않음!
""")

# 환경 변수 읽기 예시
api_key = os.environ.get('MY_API_KEY', '설정되지 않음')
print(f"▶ 환경 변수 MY_API_KEY: {api_key}")

# ============================================================
# 7. 예측 API 시뮬레이션
# ============================================================
print("\n" + "=" * 50)
print("7. 예측 API 시뮬레이션")
print("=" * 50)

def mock_prediction_api(temperature, humidity, speed):
    """가상의 예측 API (실제 API 호출 시뮬레이션)"""
    # 실제로는 requests.post()를 사용
    import random

    # 간단한 규칙 기반 예측
    defect_prob = 0.1 + 0.02 * (temperature - 85) + 0.01 * (humidity - 50)
    defect_prob = max(0, min(1, defect_prob))

    return {
        "status": "success",
        "prediction": {
            "result": "불량" if defect_prob > 0.3 else "정상",
            "confidence": round(1 - defect_prob if defect_prob <= 0.3 else defect_prob, 3)
        },
        "input": {
            "temperature": temperature,
            "humidity": humidity,
            "speed": speed
        }
    }

# API 호출 시뮬레이션
test_cases = [
    (85, 50, 100),   # 정상 조건
    (92, 65, 100),   # 높은 온도, 습도
    (80, 45, 100),   # 낮은 온도, 습도
]

print("▶ 예측 API 호출 결과:")
for temp, hum, speed in test_cases:
    result = mock_prediction_api(temp, hum, speed)
    print(f"   입력: 온도={temp}, 습도={hum}, 속도={speed}")
    print(f"   결과: {result['prediction']['result']} "
          f"(신뢰도: {result['prediction']['confidence']:.1%})")
    print()

# ============================================================
# 8. 핵심 요약
# ============================================================
print("=" * 50)
print("8. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                   API 활용 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ requests 라이브러리                                 │
│     import requests                                   │
│     response = requests.get(url)                      │
│     response = requests.post(url, json=data)         │
│     result = response.json()                          │
│                                                        │
│  ▶ 상태 코드                                           │
│     200: 성공                                          │
│     201: 생성 성공                                     │
│     400: 잘못된 요청                                   │
│     401: 인증 실패                                     │
│     404: 찾을 수 없음                                  │
│     500: 서버 에러                                     │
│                                                        │
│  ▶ API Key 보안                                        │
│     - 환경 변수 또는 .env 파일 사용                    │
│     - 코드에 직접 작성 금지!                           │
│                                                        │
│  ▶ 에러 처리                                           │
│     try-except로 예외 처리                             │
│     timeout 설정 권장                                  │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: Streamlit으로 웹앱 만들기
""")
