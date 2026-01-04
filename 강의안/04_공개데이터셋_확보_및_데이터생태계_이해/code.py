"""
[4차시] 공개 데이터셋 확보 및 데이터 생태계 이해 - 실습 코드

학습목표:
1. AI 학습에 필요한 공개 데이터셋의 종류와 특성을 파악함
2. AI 허브, Kaggle, UCI Repository 등 주요 데이터 플랫폼을 활용함
3. 데이터셋 다운로드 및 기본 구조 확인 방법을 습득함
"""

# ============================================================
# 실습 환경 설정
# ============================================================

import pandas as pd
import requests

# 선택적 라이브러리 (사전 설치 필요)
# pip install ucimlrepo
# pip install kaggle

print("=" * 50)
print("4차시: 공개 데이터셋 확보 및 데이터 생태계 이해")
print("=" * 50)


# ============================================================
# 실습 1: 공공데이터포털 API 활용
# ============================================================

print("\n[실습 1] 공공데이터포털 API 활용")
print("-" * 40)

# API 키 발급 방법:
# 1. data.go.kr 접속 및 회원가입
# 2. 로그인 후 '오픈API' 메뉴 클릭
# 3. 원하는 데이터셋 선택 → '활용신청'
# 4. 승인 후 마이페이지에서 API 키 확인

# API 호출 예시 (실제 키 필요)
def fetch_public_data(api_key, endpoint, params=None):
    """공공데이터포털 API 호출 함수"""
    if params is None:
        params = {}

    params["serviceKey"] = api_key

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return None


# 예시 코드 (실제 API 키로 교체 필요)
"""
API_KEY = "YOUR_API_KEY_HERE"
url = "https://api.odcloud.kr/api/..."

params = {
    "page": 1,
    "perPage": 10
}

data = fetch_public_data(API_KEY, url, params)
if data:
    print(data)
"""

print("✓ 공공데이터포털 API: data.go.kr에서 키 발급 필요")
print("✓ requests.get()으로 데이터 호출")
print("✓ JSON 형식으로 응답 받음")


# ============================================================
# 실습 2: Kaggle 데이터셋 다운로드
# ============================================================

print("\n[실습 2] Kaggle 데이터셋 다운로드")
print("-" * 40)

# 방법 1: 웹에서 직접 다운로드
# 1. kaggle.com 접속 및 로그인
# 2. Datasets 메뉴 클릭
# 3. 검색창에 키워드 입력 (예: "manufacturing")
# 4. 원하는 데이터셋 선택 → Download 버튼

# 방법 2: Kaggle API 사용
# 1. kaggle.com → Account → API → Create New API Token
# 2. kaggle.json 파일을 ~/.kaggle/ 에 저장
# 3. 터미널에서 명령어 실행

# Kaggle CLI 명령어 예시:
"""
# 데이터셋 검색
kaggle datasets list -s manufacturing

# 데이터셋 다운로드
kaggle datasets download -d uciml/iris

# 압축 해제
unzip iris.zip
"""

# 다운로드한 데이터 로드 예시
def load_kaggle_data(filepath):
    """Kaggle에서 다운로드한 CSV 파일 로드"""
    try:
        df = pd.read_csv(filepath)
        print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
        return df
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filepath}")
        return None


print("✓ Kaggle: kaggle.com에서 직접 다운로드 가능")
print("✓ Kaggle API로 명령어 다운로드도 지원")
print("✓ 다양한 제조 관련 데이터셋 보유")


# ============================================================
# 실습 3: UCI ML Repository 데이터셋
# ============================================================

print("\n[실습 3] UCI ML Repository 데이터셋")
print("-" * 40)

# ucimlrepo 라이브러리 사용법
# pip install ucimlrepo

try:
    from ucimlrepo import fetch_ucirepo

    # Iris 데이터셋 (ID: 53) - 가장 유명한 벤치마크 데이터
    print("UCI Iris 데이터셋 로드 중...")
    iris = fetch_ucirepo(id=53)

    # 특성(X)과 타겟(y) 분리
    X = iris.data.features
    y = iris.data.targets

    print(f"\n특성 데이터 (X):")
    print(X.head())
    print(f"\n타겟 데이터 (y):")
    print(y.head())

    print(f"\n메타데이터:")
    print(f"- 이름: {iris.metadata.get('name', 'N/A')}")
    print(f"- 샘플 수: {iris.metadata.get('num_instances', 'N/A')}")
    print(f"- 특성 수: {iris.metadata.get('num_features', 'N/A')}")

except ImportError:
    print("ucimlrepo 라이브러리가 설치되지 않았습니다.")
    print("설치: pip install ucimlrepo")

    # 대안: 직접 URL에서 로드
    print("\n대안: URL에서 직접 로드")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    try:
        iris_df = pd.read_csv(url, header=None, names=columns)
        print(iris_df.head())
    except Exception as e:
        print(f"URL 로드 실패: {e}")

print("\n✓ UCI: fetch_ucirepo(id=...)로 간단히 로드")
print("✓ X(특성)와 y(타겟)가 자동 분리됨")
print("✓ 메타데이터로 데이터셋 정보 확인 가능")


# ============================================================
# 실습 4: 데이터 구조 확인 체크리스트
# ============================================================

print("\n[실습 4] 데이터 구조 확인 체크리스트")
print("-" * 40)

# 샘플 제조 데이터 생성
sample_data = {
    'temperature': [85, 87, 92, 88, 90, 86, 91, 89, 94, 85],
    'humidity': [50, 52, 55, 48, 51, 49, 53, 50, 56, 47],
    'speed': [100, 102, 98, 105, 101, 99, 103, 100, 97, 104],
    'pressure': [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.0, 0.92, 1.08],
    'defect': [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(sample_data)

print("샘플 제조 데이터:")
print(df)

# 데이터 확인 5단계
print("\n" + "=" * 40)
print("데이터 확인 5단계")
print("=" * 40)

# 1. 데이터 크기
print("\n1. 데이터 크기 (shape)")
print(f"   → {df.shape[0]}행 x {df.shape[1]}열")

# 2. 컬럼 정보
print("\n2. 컬럼 목록 (columns)")
print(f"   → {list(df.columns)}")

# 3. 데이터 타입
print("\n3. 데이터 타입 (dtypes)")
for col, dtype in df.dtypes.items():
    print(f"   {col}: {dtype}")

# 4. 결측치 확인
print("\n4. 결측치 확인 (isnull().sum())")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   → 결측치 없음")
else:
    print(missing)

# 5. 기본 통계
print("\n5. 기본 통계 (describe())")
print(df.describe())


# ============================================================
# 실습 5: 제조 데이터 예시
# ============================================================

print("\n[실습 5] 제조 데이터 예시")
print("-" * 40)

# 조금 더 큰 제조 데이터 생성
import random
random.seed(42)

n_samples = 100

manufacturing_data = {
    'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
    'temperature': [random.gauss(88, 3) for _ in range(n_samples)],
    'humidity': [random.gauss(50, 5) for _ in range(n_samples)],
    'speed': [random.gauss(100, 5) for _ in range(n_samples)],
    'pressure': [random.gauss(1.0, 0.1) for _ in range(n_samples)],
    'vibration': [random.gauss(0.5, 0.1) for _ in range(n_samples)],
}

# 불량 여부 (온도 92 이상 또는 압력 0.85 미만이면 불량)
manufacturing_data['defect'] = [
    1 if temp > 92 or press < 0.85 else 0
    for temp, press in zip(manufacturing_data['temperature'],
                           manufacturing_data['pressure'])
]

df_mfg = pd.DataFrame(manufacturing_data)

print(f"제조 센서 데이터: {df_mfg.shape[0]}개 샘플")
print(df_mfg.head(10))

print("\n기술통계:")
print(df_mfg.describe())

print(f"\n불량률: {df_mfg['defect'].mean() * 100:.1f}%")


# ============================================================
# 데이터 저장 및 불러오기
# ============================================================

print("\n[보너스] 데이터 저장 및 불러오기")
print("-" * 40)

# CSV로 저장
# df_mfg.to_csv('manufacturing_sample.csv', index=False)
# print("✓ manufacturing_sample.csv 저장 완료")

# CSV 불러오기
# loaded_df = pd.read_csv('manufacturing_sample.csv')
# print("✓ 데이터 불러오기 완료")

print("# 저장: df.to_csv('filename.csv', index=False)")
print("# 불러오기: pd.read_csv('filename.csv')")


# ============================================================
# 핵심 요약
# ============================================================

print("\n" + "=" * 50)
print("4차시 핵심 요약")
print("=" * 50)

summary = """
1. 공개 데이터 생태계
   - 공공: 공공데이터포털 (data.go.kr)
   - AI 전문: AI 허브 (aihub.or.kr)
   - 글로벌: Kaggle (kaggle.com)
   - 학술: UCI ML Repository

2. 데이터 확보 방법
   - API 키 발급 → requests로 호출
   - 웹에서 직접 다운로드
   - ucimlrepo 라이브러리 활용

3. 데이터 확인 5단계
   - shape: 크기
   - columns: 컬럼 목록
   - dtypes: 데이터 타입
   - isnull().sum(): 결측치
   - describe(): 기술통계
"""

print(summary)

print("=" * 50)
print("다음 차시: 기초 기술통계량과 탐색적 시각화")
print("=" * 50)
