# [20차시] AI API의 이해와 활용 - 실습 코드

import requests
import json
import os
from pprint import pprint

print("=" * 60)
print("20차시: AI API의 이해와 활용")
print("외부 AI 서비스를 코드로 활용합니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 기본 GET 요청
# ============================================================
print("=" * 50)
print("실습 1: 기본 GET 요청")
print("=" * 50)

# 공개 테스트 API (JSONPlaceholder)
url = 'https://jsonplaceholder.typicode.com/posts/1'

print(f"요청 URL: {url}")
print("요청 중...")

response = requests.get(url)

print(f"\n상태 코드: {response.status_code}")
print(f"응답 데이터:")
pprint(response.json())
print()


# ============================================================
# 실습 2: POST 요청
# ============================================================
print("=" * 50)
print("실습 2: POST 요청")
print("=" * 50)

url = 'https://jsonplaceholder.typicode.com/posts'

# 전송할 데이터
data = {
    'title': '불량 예측 결과',
    'body': '온도 85, 습도 50에서 불량 확률 15%',
    'userId': 1
}

print(f"요청 URL: {url}")
print(f"전송 데이터: {data}")
print("요청 중...")

response = requests.post(url, json=data)

print(f"\n상태 코드: {response.status_code}")
print(f"생성된 데이터:")
pprint(response.json())
print()


# ============================================================
# 실습 3: 응답 처리
# ============================================================
print("=" * 50)
print("실습 3: 응답 처리 (JSON 파싱)")
print("=" * 50)

# GET 요청
url = 'https://jsonplaceholder.typicode.com/users/1'
response = requests.get(url)

# JSON 파싱
data = response.json()

print("응답 데이터 타입:", type(data))
print("\n개별 값 접근:")
print(f"  이름: {data['name']}")
print(f"  이메일: {data['email']}")
print(f"  회사: {data['company']['name']}")
print()


# ============================================================
# 실습 4: 여러 데이터 조회
# ============================================================
print("=" * 50)
print("실습 4: 여러 데이터 조회")
print("=" * 50)

url = 'https://jsonplaceholder.typicode.com/posts'
response = requests.get(url)

posts = response.json()

print(f"총 게시물 수: {len(posts)}")
print("\n처음 5개 게시물:")
for post in posts[:5]:
    print(f"  [{post['id']}] {post['title'][:30]}...")
print()


# ============================================================
# 실습 5: URL 파라미터
# ============================================================
print("=" * 50)
print("실습 5: URL 파라미터")
print("=" * 50)

# 방법 1: URL에 직접 포함
url = 'https://jsonplaceholder.typicode.com/posts?userId=1'
response = requests.get(url)
print(f"userId=1 게시물 수: {len(response.json())}")

# 방법 2: params 사용 (권장)
url = 'https://jsonplaceholder.typicode.com/posts'
params = {'userId': 1}
response = requests.get(url, params=params)
print(f"params 사용 게시물 수: {len(response.json())}")
print()


# ============================================================
# 실습 6: 상태 코드 확인
# ============================================================
print("=" * 50)
print("실습 6: 상태 코드 확인")
print("=" * 50)

# 정상 요청
response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
print(f"정상 요청: {response.status_code}")

# 존재하지 않는 리소스
response = requests.get('https://jsonplaceholder.typicode.com/posts/99999')
print(f"없는 리소스: {response.status_code}")

print("\n상태 코드 의미:")
print("  200: 성공")
print("  400: 잘못된 요청")
print("  401: 인증 실패")
print("  404: 찾을 수 없음")
print("  500: 서버 오류")
print()


# ============================================================
# 실습 7: 오류 처리
# ============================================================
print("=" * 50)
print("실습 7: 오류 처리")
print("=" * 50)

def safe_api_call(url, timeout=5):
    """안전한 API 호출 함수"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # 4xx, 5xx 오류 시 예외 발생
        return response.json()
    except requests.exceptions.Timeout:
        print("⚠️ 타임아웃 발생")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"⚠️ HTTP 오류: {e}")
        return None
    except requests.exceptions.ConnectionError:
        print("⚠️ 연결 오류")
        return None
    except Exception as e:
        print(f"⚠️ 알 수 없는 오류: {e}")
        return None

# 정상 호출
print("정상 URL 호출:")
result = safe_api_call('https://jsonplaceholder.typicode.com/posts/1')
if result:
    print(f"  성공! 제목: {result['title'][:30]}...")

# 잘못된 URL 호출
print("\n잘못된 URL 호출:")
result = safe_api_call('https://invalid-url-example.com/api')
print()


# ============================================================
# 실습 8: 헤더 설정
# ============================================================
print("=" * 50)
print("실습 8: 헤더 설정")
print("=" * 50)

# 커스텀 헤더
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'MyApp/1.0'
}

response = requests.get(
    'https://jsonplaceholder.typicode.com/posts/1',
    headers=headers
)

print("요청 헤더:")
for key, value in headers.items():
    print(f"  {key}: {value}")

print(f"\n응답 상태: {response.status_code}")
print()


# ============================================================
# 실습 9: API 키 인증 (시뮬레이션)
# ============================================================
print("=" * 50)
print("실습 9: API 키 인증 (시뮬레이션)")
print("=" * 50)

# API 키 설정 (실제로는 환경 변수에서!)
# os.environ['MY_API_KEY'] = 'demo-key-12345'
API_KEY = os.environ.get('MY_API_KEY', 'demo-key-12345')

# Authorization 헤더
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

print("인증 헤더 설정:")
print(f"  Authorization: Bearer {'*' * 10}")  # 키 숨김
print(f"  Content-Type: application/json")

# 실제 요청 (테스트 API는 인증 불필요)
response = requests.get(
    'https://jsonplaceholder.typicode.com/posts/1',
    headers=headers
)
print(f"\n응답 상태: {response.status_code}")
print()


# ============================================================
# 실습 10: 환경 변수 관리
# ============================================================
print("=" * 50)
print("실습 10: 환경 변수 관리")
print("=" * 50)

print("환경 변수에서 API 키 읽기:")
print("-" * 40)

# 방법 1: os.environ.get (기본값 지정 가능)
api_key = os.environ.get('MY_API_KEY', 'default-key')
print(f"os.environ.get: {api_key}")

# 방법 2: os.environ[] (키 없으면 KeyError)
try:
    api_key = os.environ['MY_API_KEY']
except KeyError:
    print("os.environ[]: 키가 없습니다 (KeyError)")

print("\n설정 방법 (터미널):")
print("  export MY_API_KEY='your-secret-key'")

print("\n.env 파일 사용 (python-dotenv):")
print("  pip install python-dotenv")
print("  from dotenv import load_dotenv")
print("  load_dotenv()")
print()


# ============================================================
# 실습 11: API 클라이언트 클래스
# ============================================================
print("=" * 50)
print("실습 11: API 클라이언트 클래스")
print("=" * 50)

class AIClient:
    """재사용 가능한 API 클라이언트"""

    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def get(self, endpoint):
        """GET 요청"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"오류: {e}")
            return None

    def post(self, endpoint, data):
        """POST 요청"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"오류: {e}")
            return None


# 클라이언트 사용
client = AIClient('https://jsonplaceholder.typicode.com')

print("GET 요청:")
result = client.get('posts/1')
if result:
    print(f"  제목: {result['title'][:30]}...")

print("\nPOST 요청:")
result = client.post('posts', {'title': '테스트', 'body': '내용', 'userId': 1})
if result:
    print(f"  생성된 ID: {result['id']}")
print()


# ============================================================
# 실습 12: 배치 요청
# ============================================================
print("=" * 50)
print("실습 12: 배치 요청 (여러 데이터 처리)")
print("=" * 50)

# 여러 제조 데이터에 대해 예측 요청 시뮬레이션
manufacturing_data = [
    {'temperature': 85, 'humidity': 50, 'speed': 100},
    {'temperature': 90, 'humidity': 60, 'speed': 105},
    {'temperature': 80, 'humidity': 45, 'speed': 95},
]

print("제조 데이터 배치 처리 시뮬레이션:")
print("-" * 50)

for i, data in enumerate(manufacturing_data):
    # 실제로는 AI API 호출
    # result = client.post('predict', data)

    # 시뮬레이션 결과
    prob = 0.05 + 0.03 * (data['temperature'] - 80) / 5
    result = {'probability': prob, 'status': 'defect' if prob > 0.1 else 'normal'}

    print(f"데이터 {i+1}: 온도={data['temperature']}, 습도={data['humidity']}")
    print(f"  → 불량 확률: {result['probability']:.1%}, 상태: {result['status']}")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print(f"""
┌───────────────────────────────────────────────────────┐
│              AI API 활용 핵심 정리                     │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 기본 요청                                           │
│     import requests                                    │
│     response = requests.get(url)                       │
│     response = requests.post(url, json=data)           │
│                                                        │
│  ▶ 응답 처리                                           │
│     status_code = response.status_code  # 200 = 성공   │
│     data = response.json()  # JSON → dict              │
│                                                        │
│  ▶ 헤더와 인증                                         │
│     headers = {{'Authorization': f'Bearer {{API_KEY}}'}}│
│     requests.get(url, headers=headers)                 │
│                                                        │
│  ▶ 오류 처리                                           │
│     try:                                               │
│         response = requests.get(url, timeout=5)        │
│         response.raise_for_status()                    │
│     except requests.exceptions.Timeout:                │
│         print("타임아웃!")                              │
│                                                        │
│  ▶ API 키 보안                                         │
│     API_KEY = os.environ.get('MY_API_KEY')             │
│     ❌ 코드에 직접 쓰지 말 것!                          │
│     ✅ 환경 변수 또는 .env 파일 사용                    │
│                                                        │
│  ★ 재사용 가능한 클라이언트 클래스 만들기!             │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: LLM API와 프롬프트 작성법
""")

print("=" * 60)
print("20차시 실습 완료!")
print("외부 AI 서비스를 코드로 활용하는 법을 배웠습니다!")
print("=" * 60)
