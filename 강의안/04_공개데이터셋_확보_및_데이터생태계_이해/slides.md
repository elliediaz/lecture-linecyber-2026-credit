---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 4차시'
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

# 공개 데이터셋 확보 및 데이터 생태계 이해

## 4차시 | Part I. AI 윤리와 환경 구축

**AI 학습을 위한 공개 데이터 플랫폼 활용하기**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **공개 데이터셋**의 종류와 특성을 파악한다
2. **AI 허브, Kaggle, UCI Repository** 등 주요 플랫폼을 활용한다
3. **데이터셋 다운로드** 및 기본 구조 확인 방법을 습득한다

---

# 오늘의 진행 순서

## 이론 + 실습 (25-30분)

| 순서 | 내용 | 시간 |
|------|------|------|
| 1 | 데이터 생태계 개요 | 3분 |
| 2 | 주요 데이터 플랫폼 소개 | 5분 |
| 3 | 플랫폼별 활용 방법 | 7분 |
| 4 | 실습: 데이터셋 다운로드 | 15분 |

---

# 데이터 생태계란?

## AI 학습에는 데이터가 필수!

```
┌─────────────────────────────────────────────────────┐
│              AI 학습의 핵심 요소                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│   데이터 수집 → 전처리 → 모델 학습 → 평가           │
│       ↑                                              │
│   가장 중요한 첫 단계!                               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

> "데이터 없이는 AI도 없다"

---

# 공개 데이터의 종류

## 어디서 데이터를 구할 수 있을까?

| 유형 | 설명 | 예시 |
|------|------|------|
| **공공데이터** | 정부/공공기관 제공 | 공공데이터포털 |
| **연구 데이터** | 학술 연구용 | UCI Repository |
| **경진대회 데이터** | AI 대회용 | Kaggle |
| **AI 학습용 데이터** | AI 모델 학습 전용 | AI 허브 |
| **기업 데이터** | 기업 제공 오픈소스 | 제조사 공개 데이터 |

---

# 주요 데이터 플랫폼

## 1. 공공데이터포털 (data.go.kr)

- 대한민국 정부 공식 데이터 포털
- **43만+ 데이터셋** 제공
- API 키 발급으로 실시간 데이터 접근
- 제조업 관련 데이터 다수

```
주요 제조 데이터:
- 스마트공장 센서 데이터
- 제조업 통계 데이터
- 품질 검사 이력 데이터
```

---

# 주요 데이터 플랫폼

## 2. AI 허브 (aihub.or.kr)

- 한국지능정보사회진흥원(NIA) 운영
- **AI 학습용 데이터** 전문
- 다양한 형태: 이미지, 텍스트, 음성
- 제조 분야 특화 데이터셋 보유

```
제조 AI 데이터셋:
- 제조 불량 이미지 데이터
- 설비 진동 센서 데이터
- 공정 품질 데이터
```

---

# 주요 데이터 플랫폼

## 3. Kaggle (kaggle.com)

- 세계 최대 데이터 과학 커뮤니티
- **20만+ 데이터셋** 무료 제공
- 경진대회 및 노트북 공유
- 다양한 제조 데이터셋

```python
# Kaggle API로 데이터 다운로드
!kaggle datasets download -d dataset-name
```

---

# 주요 데이터 플랫폼

## 4. UCI ML Repository

- UC Irvine 대학 운영
- **600+ 머신러닝 데이터셋**
- 학술 연구에 많이 활용
- 분류/회귀 벤치마크 데이터

```python
# ucimlrepo 라이브러리 활용
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=53)  # Iris 데이터
```

---

# 제조 분야 추천 데이터셋

## 실습에 활용할 수 있는 데이터

| 데이터셋 | 출처 | 활용 |
|----------|------|------|
| 품질 예측 데이터 | AI 허브 | 불량 분류 |
| 설비 센서 데이터 | 공공데이터포털 | 이상 탐지 |
| 반도체 공정 데이터 | Kaggle | 수율 예측 |
| Steel Plates Faults | UCI | 결함 분류 |

---

# - 실습편 -

## 4차시

**데이터 플랫폼 활용 실습**

---

# 실습 환경

## 필요 라이브러리

```python
# 기본 라이브러리
import pandas as pd
import requests

# Kaggle API (선택)
# pip install kaggle

# UCI 데이터 (선택)
# pip install ucimlrepo
```

---

# 실습 1: 공공데이터포털 API

## API 키 발급 및 데이터 조회

```python
import requests
import pandas as pd

# API 키 (발급 후 사용)
API_KEY = "YOUR_API_KEY"

# 공공데이터 API 호출 예시
url = "https://api.odcloud.kr/api/..."
params = {
    "serviceKey": API_KEY,
    "page": 1,
    "perPage": 10
}

response = requests.get(url, params=params)
data = response.json()
print(data)
```

---

# 실습 2: Kaggle 데이터셋

## 데이터셋 검색 및 다운로드

```python
# 방법 1: 웹에서 직접 다운로드
# kaggle.com → Datasets → 검색 → Download

# 방법 2: Kaggle API 사용
# 1. kaggle.com에서 API 토큰 발급
# 2. ~/.kaggle/kaggle.json에 저장
# 3. 명령어로 다운로드

# !kaggle datasets download -d uciml/iris

# 다운로드 후 로드
df = pd.read_csv('iris.csv')
print(df.head())
print(df.info())
```

---

# 실습 3: UCI 데이터셋

## ucimlrepo 라이브러리 활용

```python
from ucimlrepo import fetch_ucirepo

# Iris 데이터셋 (ID: 53)
iris = fetch_ucirepo(id=53)

# 특성(X)과 타겟(y) 분리
X = iris.data.features
y = iris.data.targets

print("특성 데이터:")
print(X.head())

print("\n메타데이터:")
print(iris.metadata)
```

---

# 실습 4: 제조 데이터 로드

## 실습용 제조 데이터 확인

```python
import pandas as pd

# 샘플 제조 데이터 생성
data = {
    'temperature': [85, 87, 92, 88, 90],
    'humidity': [50, 52, 55, 48, 51],
    'speed': [100, 102, 98, 105, 101],
    'pressure': [1.0, 1.1, 0.9, 1.05, 0.95],
    'defect': [0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df)
print("\n기술통계:")
print(df.describe())
```

---

# 데이터 구조 확인 체크리스트

## 데이터를 받으면 확인할 것들

```python
# 1. 데이터 크기
print(df.shape)

# 2. 컬럼 정보
print(df.columns)

# 3. 데이터 타입
print(df.dtypes)

# 4. 결측치 확인
print(df.isnull().sum())

# 5. 기본 통계
print(df.describe())
```

---

# 핵심 요약

## 4차시 정리

1. **공개 데이터 생태계**: 공공, 연구, 경진대회, AI 학습용
2. **주요 플랫폼 4가지**:
   - 공공데이터포털: 정부 공식, API 제공
   - AI 허브: AI 학습 전문, 제조 특화
   - Kaggle: 글로벌, 경진대회
   - UCI: 학술 벤치마크
3. **데이터 확인**: shape, columns, dtypes, isnull

---

# 다음 차시 예고

## 5차시: 기초 기술통계량과 탐색적 시각화

- 평균, 중앙값, 표준편차 이해
- Matplotlib으로 그래프 그리기
- 히스토그램, 상자그림 해석

> 데이터를 확보했으니, 이제 분석을 시작합니다!

---

# 감사합니다

## 4차시: 공개 데이터셋 확보 및 데이터 생태계 이해

**제조데이터를 활용한 AI 이해와 예측 모델 구축**

수고하셨습니다!
