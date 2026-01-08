"""
[25차시] AI API의 이해와 활용 - 실습 코드

학습 목표:
1. REST API의 개념과 구조를 이해한다
2. requests 라이브러리로 API를 호출한다
3. JSON 데이터를 처리한다

실습 환경: Python 3.8+, requests

참고: 이 코드는 실제 공개 API를 호출합니다.
"""

import json
import time
from datetime import datetime

# requests 설치 확인
try:
    import requests
    print(f"requests 버전: {requests.__version__}")
except ImportError:
    print("requests 설치 필요: pip install requests")
    exit(1)

print("=" * 60)
print("[25차시] AI API의 이해와 활용")
print("=" * 60)

# ============================================================
# 1. REST API 기본 개념 이해
# ============================================================
print("\n" + "=" * 60)
print("1. REST API 기본 개념 이해")
print("=" * 60)

print("""
REST API 구성 요소:
1. URL (Endpoint): API 주소
2. HTTP 메서드: GET, POST, PUT, DELETE
3. 헤더 (Headers): 인증, 데이터 타입 등 메타 정보
4. 바디 (Body): 전송할 데이터 (주로 JSON)

HTTP 메서드:
- GET: 데이터 조회
- POST: 데이터 생성 (ML 예측에 주로 사용)
- PUT: 데이터 수정
- DELETE: 데이터 삭제
""")

# ============================================================
# 2. 기본 GET 요청 - httpbin.org
# ============================================================
print("\n" + "=" * 60)
print("2. 기본 GET 요청 - httpbin.org")
print("=" * 60)

# httpbin.org - HTTP 테스트용 공개 API
url = 'https://httpbin.org/get'

print(f"\n요청 URL: {url}")
print("요청 메서드: GET")

try:
    response = requests.get(url, timeout=10)

    print(f"\n응답 상태 코드: {response.status_code}")
    print(f"응답 성공 여부: {response.ok}")
    print(f"응답 시간: {response.elapsed.total_seconds():.3f}초")

    # JSON 응답 파싱
    data = response.json()
    print(f"\n응답 데이터 (일부):")
    print(f"  - origin (요청 IP): {data.get('origin', 'N/A')}")
    print(f"  - headers: {list(data.get('headers', {}).keys())[:3]}...")

except requests.RequestException as e:
    print(f"요청 실패: {e}")

# ============================================================
# 3. 공개 API 호출 - OpenWeatherMap 형식 예제
# ============================================================
print("\n" + "=" * 60)
print("3. 공개 API 호출 - 날씨 API 형식 예제")
print("=" * 60)

# 참고: 실제 OpenWeatherMap API 사용시 API 키 필요
# 여기서는 httpbin.org를 통해 요청 구조만 설명

print("""
[날씨 API 요청 형식 예시 - OpenWeatherMap]

URL: https://api.openweathermap.org/data/2.5/weather
Parameters:
  - q: 도시명 (예: Seoul)
  - appid: API 키
  - units: metric (섭씨) 또는 imperial (화씨)

예시 요청:
GET https://api.openweathermap.org/data/2.5/weather?q=Seoul&units=metric&appid=YOUR_KEY

예시 응답:
{
    "main": {
        "temp": 15.5,
        "humidity": 60,
        "pressure": 1013
    },
    "weather": [{"description": "clear sky"}],
    "name": "Seoul"
}
""")

# httpbin.org로 파라미터 전달 테스트
url = 'https://httpbin.org/get'
params = {
    'q': 'Seoul',
    'units': 'metric',
    'appid': 'demo-api-key'
}

print(f"\n요청 URL: {url}")
print(f"파라미터: {params}")

try:
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    print(f"\n실제 요청 URL: {response.url}")
    print(f"전달된 파라미터: {data.get('args', {})}")

except requests.RequestException as e:
    print(f"요청 실패: {e}")

# ============================================================
# 4. POST 요청 - JSON 데이터 전송
# ============================================================
print("\n" + "=" * 60)
print("4. POST 요청 - JSON 데이터 전송")
print("=" * 60)

url = 'https://httpbin.org/post'

# 품질 예측을 위한 입력 데이터 (ML API 요청 형식)
prediction_input = {
    'temperature': 210,
    'pressure': 55,
    'speed': 105,
    'humidity': 60,
    'vibration': 5.5,
    'timestamp': datetime.now().isoformat()
}

print(f"\n요청 URL: {url}")
print(f"요청 메서드: POST")
print(f"전송 데이터: {prediction_input}")

try:
    # json 파라미터 사용 시 자동으로:
    # 1. Content-Type: application/json 헤더 추가
    # 2. json.dumps() 처리
    response = requests.post(url, json=prediction_input, timeout=10)

    print(f"\n응답 상태 코드: {response.status_code}")

    data = response.json()
    print(f"서버가 받은 JSON: {data.get('json', {})}")

except requests.RequestException as e:
    print(f"요청 실패: {e}")

# ============================================================
# 5. 헤더 추가하기 (API 인증)
# ============================================================
print("\n" + "=" * 60)
print("5. 헤더 추가하기 (API 인증)")
print("=" * 60)

url = 'https://httpbin.org/post'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer fake-api-key-12345',
    'X-Request-ID': 'req-001',
    'X-Client-Version': '1.0.0'
}

data = {'temperature': 200, 'pressure': 50}

print(f"\n요청 헤더: {headers}")

try:
    response = requests.post(url, json=data, headers=headers, timeout=10)
    result = response.json()

    print(f"\n서버가 받은 헤더:")
    for key, value in result.get('headers', {}).items():
        if key.startswith('X-') or key == 'Authorization':
            print(f"  {key}: {value}")

except requests.RequestException as e:
    print(f"요청 실패: {e}")

# ============================================================
# 6. 응답 객체 상세 분석
# ============================================================
print("\n" + "=" * 60)
print("6. 응답 객체 상세 분석")
print("=" * 60)

url = 'https://httpbin.org/get'

try:
    response = requests.get(url, timeout=10)

    print("\n응답 객체 속성:")
    print(f"  status_code: {response.status_code}")
    print(f"  ok (200-299): {response.ok}")
    print(f"  reason: {response.reason}")
    print(f"  url: {response.url}")
    print(f"  encoding: {response.encoding}")
    print(f"  elapsed: {response.elapsed}")

    print("\n응답 헤더 (일부):")
    for key in ['Content-Type', 'Content-Length', 'Date']:
        if key in response.headers:
            print(f"  {key}: {response.headers[key]}")

    print("\n응답 본문 타입:")
    print(f"  response.text: {type(response.text).__name__} (문자열)")
    print(f"  response.json(): {type(response.json()).__name__} (딕셔너리)")
    print(f"  response.content: {type(response.content).__name__} (바이트)")

except requests.RequestException as e:
    print(f"요청 실패: {e}")

# ============================================================
# 7. 에러 처리
# ============================================================
print("\n" + "=" * 60)
print("7. 에러 처리")
print("=" * 60)

# 7.1 존재하지 않는 URL (404)
print("\n7.1 404 에러 테스트:")
try:
    response = requests.get('https://httpbin.org/status/404', timeout=10)
    print(f"  상태 코드: {response.status_code}")
    print(f"  ok: {response.ok}")

    # raise_for_status()는 4xx, 5xx 에러 시 예외 발생
    response.raise_for_status()

except requests.HTTPError as e:
    print(f"  HTTPError 발생: {e}")
except requests.RequestException as e:
    print(f"  요청 실패: {e}")

# 7.2 서버 에러 (500)
print("\n7.2 500 에러 테스트:")
try:
    response = requests.get('https://httpbin.org/status/500', timeout=10)
    response.raise_for_status()
except requests.HTTPError as e:
    print(f"  HTTPError 발생: {e}")

# 7.3 타임아웃
print("\n7.3 타임아웃 테스트:")
try:
    # 10초 지연되는 응답에 1초 타임아웃
    response = requests.get('https://httpbin.org/delay/10', timeout=1)
except requests.Timeout:
    print("  타임아웃 발생!")
except requests.RequestException as e:
    print(f"  요청 실패: {e}")

# 7.4 연결 에러
print("\n7.4 연결 에러 테스트:")
try:
    response = requests.get('https://invalid-domain-12345.com', timeout=3)
except requests.ConnectionError:
    print("  연결 실패! (도메인 없음)")
except requests.RequestException as e:
    print(f"  요청 실패: {e}")

# ============================================================
# 8. JSON 기본 처리
# ============================================================
print("\n" + "=" * 60)
print("8. JSON 기본 처리")
print("=" * 60)

# 8.1 Python → JSON (직렬화)
print("\n8.1 Python → JSON (dumps):")
python_data = {
    'temperature': 200,
    'pressure': 50,
    'is_normal': True,
    'tags': ['sensor', 'quality'],
    'metadata': None
}

json_string = json.dumps(python_data)
print(f"  Python: {python_data}")
print(f"  JSON:   {json_string}")

# 8.2 JSON → Python (역직렬화)
print("\n8.2 JSON → Python (loads):")
json_string = '{"temperature": 200, "status": true, "error": null}'
python_data = json.loads(json_string)
print(f"  JSON:   {json_string}")
print(f"  Python: {python_data}")
print(f"  타입:   {type(python_data)}")

# 8.3 들여쓰기와 한글 처리
print("\n8.3 들여쓰기와 한글 처리:")
data = {
    '설비명': '프레스 1호기',
    '상태': '정상',
    'temperature': 200
}

# ensure_ascii=False로 한글 유지
json_pretty = json.dumps(data, ensure_ascii=False, indent=2)
print(json_pretty)

# ============================================================
# 9. JSON 파일 처리
# ============================================================
print("\n" + "=" * 60)
print("9. JSON 파일 처리")
print("=" * 60)

# 9.1 JSON 파일 저장
config = {
    'api_url': 'https://api.example.com/predict',
    'timeout': 10,
    'features': ['temperature', 'pressure', 'speed', 'humidity', 'vibration'],
    'model_version': 'v1.0.0',
    'created_at': datetime.now().isoformat()
}

config_path = 'api_config.json'
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"\n설정 파일 저장: {config_path}")
print("내용:")
print(json.dumps(config, ensure_ascii=False, indent=2))

# 9.2 JSON 파일 로드
print("\n파일에서 로드:")
with open(config_path, 'r', encoding='utf-8') as f:
    loaded_config = json.load(f)

print(f"  api_url: {loaded_config['api_url']}")
print(f"  timeout: {loaded_config['timeout']}")
print(f"  features: {loaded_config['features']}")

# ============================================================
# 10. 세션 사용하기
# ============================================================
print("\n" + "=" * 60)
print("10. 세션 사용하기")
print("=" * 60)

# 세션 생성 (연결 재사용으로 성능 향상)
session = requests.Session()

# 공통 헤더 설정
session.headers.update({
    'Authorization': 'Bearer fake-api-key',
    'X-Client-ID': 'quality-predictor'
})

print("\n세션 공통 헤더 설정 완료")
print(f"헤더: {dict(session.headers)}")

# 여러 요청에 재사용
urls = [
    'https://httpbin.org/get',
    'https://httpbin.org/headers'
]

print("\n세션으로 여러 요청:")
for url in urls:
    try:
        response = session.get(url, timeout=10)
        print(f"  {url}: {response.status_code}")
    except requests.RequestException as e:
        print(f"  {url}: 실패 - {e}")

session.close()
print("\n세션 종료")

# ============================================================
# 11. 실제 공개 API 호출 - JSONPlaceholder
# ============================================================
print("\n" + "=" * 60)
print("11. 실제 공개 API 호출 - JSONPlaceholder")
print("=" * 60)

# JSONPlaceholder - 테스트용 REST API
print("\nJSONPlaceholder는 무료 REST API 테스트 서비스입니다.")

try:
    # 게시물 조회
    print("\n11.1 게시물 조회 (GET):")
    response = requests.get('https://jsonplaceholder.typicode.com/posts/1', timeout=10)
    post = response.json()
    print(f"  게시물 ID: {post['id']}")
    print(f"  제목: {post['title'][:40]}...")
    print(f"  본문: {post['body'][:50]}...")

    # 여러 게시물 조회
    print("\n11.2 여러 게시물 조회 (GET with params):")
    response = requests.get(
        'https://jsonplaceholder.typicode.com/posts',
        params={'userId': 1, '_limit': 3},
        timeout=10
    )
    posts = response.json()
    print(f"  조회된 게시물 수: {len(posts)}")
    for p in posts:
        print(f"    - ID {p['id']}: {p['title'][:30]}...")

    # 게시물 생성 (시뮬레이션 - 실제로 저장되지 않음)
    print("\n11.3 게시물 생성 (POST):")
    new_post = {
        'title': 'Quality Report - Temperature Anomaly',
        'body': 'Temperature exceeded threshold at 14:30',
        'userId': 1
    }
    response = requests.post(
        'https://jsonplaceholder.typicode.com/posts',
        json=new_post,
        timeout=10
    )
    created = response.json()
    print(f"  생성된 게시물 ID: {created.get('id')}")
    print(f"  제목: {created.get('title')}")

    # 게시물 수정 (PUT)
    print("\n11.4 게시물 수정 (PUT):")
    update_data = {
        'id': 1,
        'title': 'Updated Quality Report',
        'body': 'Issue resolved at 15:00',
        'userId': 1
    }
    response = requests.put(
        'https://jsonplaceholder.typicode.com/posts/1',
        json=update_data,
        timeout=10
    )
    updated = response.json()
    print(f"  수정된 게시물 ID: {updated.get('id')}")
    print(f"  새 제목: {updated.get('title')}")

    # 게시물 삭제 (DELETE)
    print("\n11.5 게시물 삭제 (DELETE):")
    response = requests.delete(
        'https://jsonplaceholder.typicode.com/posts/1',
        timeout=10
    )
    print(f"  삭제 응답 상태: {response.status_code}")

except requests.RequestException as e:
    print(f"  API 호출 실패: {e}")

# ============================================================
# 12. 실제 공개 API 호출 - GitHub API
# ============================================================
print("\n" + "=" * 60)
print("12. 실제 공개 API 호출 - GitHub API")
print("=" * 60)

print("\nGitHub API는 공개 리포지토리 정보를 무료로 조회할 수 있습니다.")

try:
    # 공개 리포지토리 정보 조회
    print("\n12.1 리포지토리 정보 조회:")
    response = requests.get(
        'https://api.github.com/repos/python/cpython',
        timeout=10
    )
    repo = response.json()
    print(f"  리포지토리: {repo.get('full_name')}")
    print(f"  설명: {repo.get('description', 'N/A')[:50]}...")
    print(f"  스타 수: {repo.get('stargazers_count'):,}")
    print(f"  포크 수: {repo.get('forks_count'):,}")
    print(f"  주요 언어: {repo.get('language')}")

    # 사용자 정보 조회
    print("\n12.2 사용자 정보 조회:")
    response = requests.get(
        'https://api.github.com/users/torvalds',
        timeout=10
    )
    user = response.json()
    print(f"  사용자: {user.get('login')}")
    print(f"  이름: {user.get('name', 'N/A')}")
    print(f"  공개 저장소 수: {user.get('public_repos')}")
    print(f"  팔로워 수: {user.get('followers'):,}")

except requests.RequestException as e:
    print(f"  API 호출 실패: {e}")

# ============================================================
# 13. API 호출 래퍼 함수
# ============================================================
print("\n" + "=" * 60)
print("13. API 호출 래퍼 함수")
print("=" * 60)

def call_api(url, method='GET', data=None, headers=None, timeout=10, retries=3):
    """
    API 호출 래퍼 함수

    Parameters:
    -----------
    url : str
        API 엔드포인트
    method : str
        HTTP 메서드 (GET, POST 등)
    data : dict
        전송할 데이터 (POST 시)
    headers : dict
        추가 헤더
    timeout : int
        타임아웃 (초)
    retries : int
        재시도 횟수

    Returns:
    --------
    dict : 응답 결과
    """
    default_headers = {'Content-Type': 'application/json'}
    if headers:
        default_headers.update(headers)

    for attempt in range(retries):
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=data, headers=default_headers, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, headers=default_headers, timeout=timeout)
            else:
                return {'success': False, 'error': f'Unsupported method: {method}'}

            response.raise_for_status()

            return {
                'success': True,
                'status_code': response.status_code,
                'data': response.json() if response.text else None,
                'elapsed': response.elapsed.total_seconds()
            }

        except requests.Timeout:
            print(f"  시도 {attempt + 1}/{retries}: 타임아웃")
            if attempt == retries - 1:
                return {'success': False, 'error': 'Timeout'}

        except requests.HTTPError as e:
            return {'success': False, 'error': str(e), 'status_code': response.status_code}

        except requests.RequestException as e:
            print(f"  시도 {attempt + 1}/{retries}: {e}")
            if attempt == retries - 1:
                return {'success': False, 'error': str(e)}

        # 재시도 전 대기 (지수 백오프)
        time.sleep(2 ** attempt)

# 사용 예시
print("\n래퍼 함수 테스트:")

# GET 요청
result = call_api('https://httpbin.org/get', method='GET')
print(f"  GET 요청: success={result['success']}, elapsed={result.get('elapsed', 'N/A')}")

# POST 요청
result = call_api('https://httpbin.org/post', method='POST', data={'temperature': 200})
print(f"  POST 요청: success={result['success']}")

# ============================================================
# 14. 응답 데이터 검증
# ============================================================
print("\n" + "=" * 60)
print("14. 응답 데이터 검증")
print("=" * 60)

def validate_prediction_response(response_data):
    """
    예측 API 응답 검증

    Parameters:
    -----------
    response_data : dict
        API 응답 데이터

    Returns:
    --------
    tuple : (is_valid, message)
    """
    # 필수 필드 확인
    required_fields = ['prediction', 'defect_probability']

    for field in required_fields:
        if field not in response_data:
            return False, f"Missing field: {field}"

    # 값 유효성 확인
    valid_predictions = ['normal', 'defect', 'warning']
    if response_data['prediction'] not in valid_predictions:
        return False, f"Invalid prediction: {response_data['prediction']}"

    prob = response_data['defect_probability']
    if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
        return False, f"Invalid probability: {prob}"

    return True, "Valid response"

# 테스트
print("\n응답 검증 테스트:")

# 정상 응답
valid_response = {'prediction': 'normal', 'defect_probability': 0.15}
is_valid, msg = validate_prediction_response(valid_response)
print(f"  정상 응답: {is_valid} - {msg}")

# 필드 누락
invalid_response1 = {'prediction': 'normal'}
is_valid, msg = validate_prediction_response(invalid_response1)
print(f"  필드 누락: {is_valid} - {msg}")

# 잘못된 값
invalid_response2 = {'prediction': 'unknown', 'defect_probability': 0.5}
is_valid, msg = validate_prediction_response(invalid_response2)
print(f"  잘못된 값: {is_valid} - {msg}")

# ============================================================
# 15. 품질 예측 API 클라이언트 예제
# ============================================================
print("\n" + "=" * 60)
print("15. 품질 예측 API 클라이언트 예제")
print("=" * 60)

class QualityPredictionClient:
    """품질 예측 API 클라이언트"""

    def __init__(self, base_url='https://api.example.com', api_key=None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

    def predict(self, sensor_data: dict, timeout: int = 10) -> dict:
        """
        품질 예측 요청

        Parameters:
        -----------
        sensor_data : dict
            센서 데이터
        timeout : int
            타임아웃 (초)

        Returns:
        --------
        dict : 예측 결과
        """
        # 실제 API 호출 (여기서는 httpbin.org로 시뮬레이션)
        try:
            response = self.session.post(
                'https://httpbin.org/post',
                json={
                    'endpoint': f'{self.base_url}/predict',
                    'payload': sensor_data
                },
                timeout=timeout
            )
            response.raise_for_status()

            # 시뮬레이션: 간단한 규칙 기반 예측
            temp = sensor_data.get('temperature', 200)
            pressure = sensor_data.get('pressure', 50)

            if temp > 250 or pressure > 70:
                prediction = 'defect'
                defect_prob = 0.85
            elif temp > 220 or pressure > 60:
                prediction = 'warning'
                defect_prob = 0.45
            else:
                prediction = 'normal'
                defect_prob = 0.1

            return {
                'success': True,
                'prediction': prediction,
                'defect_probability': defect_prob,
                'confidence': 1 - abs(0.5 - defect_prob),
                'input': sensor_data,
                'timestamp': datetime.now().isoformat()
            }

        except requests.RequestException as e:
            return {
                'success': False,
                'error': str(e)
            }

    def batch_predict(self, sensor_data_list: list) -> list:
        """배치 예측"""
        results = []
        for data in sensor_data_list:
            result = self.predict(data)
            results.append(result)
        return results

    def close(self):
        self.session.close()

# 클라이언트 사용 예시
print("\n품질 예측 클라이언트 테스트:")

client = QualityPredictionClient(api_key='demo-key')

# 단일 예측
single_input = {
    'temperature': 210,
    'pressure': 55,
    'speed': 100,
    'humidity': 60,
    'vibration': 5
}

result = client.predict(single_input)
print(f"\n단일 예측:")
print(f"  입력: {single_input}")
print(f"  예측: {result['prediction']}")
print(f"  불량 확률: {result['defect_probability']:.1%}")

# 배치 예측
batch_input = [
    {'temperature': 200, 'pressure': 50},
    {'temperature': 230, 'pressure': 65},
    {'temperature': 260, 'pressure': 75}
]

print(f"\n배치 예측:")
results = client.batch_predict(batch_input)
for i, r in enumerate(results):
    print(f"  샘플 {i+1}: {r['prediction']} (불량 확률: {r['defect_probability']:.1%})")

client.close()

# ============================================================
# 16. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("16. 핵심 정리")
print("=" * 60)

print("""
[25차시 핵심 정리]

1. REST API 개념
   - HTTP 메서드: GET(조회), POST(생성), PUT(수정), DELETE(삭제)
   - 상태 코드: 200(성공), 400(잘못된 요청), 404(없음), 500(서버 에러)
   - 인증: API Key, Bearer Token

2. requests 라이브러리
   - GET: requests.get(url, params=params)
   - POST: requests.post(url, json=data)
   - 응답: response.status_code, response.json()
   - 에러 처리: try-except, raise_for_status()
   - 타임아웃: timeout 파라미터 필수

3. JSON 처리
   - 직렬화: json.dumps(python_dict)
   - 역직렬화: json.loads(json_string)
   - 파일: json.dump() / json.load()
   - 한글: ensure_ascii=False

4. 실제 공개 API 예시
   - httpbin.org: HTTP 테스트
   - JSONPlaceholder: REST API 테스트
   - GitHub API: 리포지토리/사용자 정보 조회

5. 핵심 코드
   ```python
   import requests
   import json

   # API 호출
   response = requests.post(url, json=data, timeout=10)
   if response.ok:
       result = response.json()

   # JSON 변환
   json_str = json.dumps(data, ensure_ascii=False)
   data = json.loads(json_str)
   ```
""")

# 임시 파일 정리
import os
if os.path.exists('api_config.json'):
    os.remove('api_config.json')

print("\n다음 차시 예고: LLM API와 프롬프트 작성법")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
