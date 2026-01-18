---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 12차시'
footer: '공공데이터를 활용한 AI 예측 모델 구축'
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
  table { font-size: 0.9em; }
  .highlight { background-color: #fef3c7; padding: 10px; border-radius: 8px; }
  .important { background-color: #fee2e2; padding: 10px; border-radius: 8px; }
  .tip { background-color: #d1fae5; padding: 10px; border-radius: 8px; }
---

# 머신러닝 소개와 문제 유형

## 12차시 | Part III. 문제 중심 모델링 실습

**드디어 AI 모델을 만듭니다!**

---

# Part III 시작!

## 지금까지의 여정

```
Part I:  Python 기초, 데이터 다루기 (1-4차시)
Part II: 통계, 시각화, 전처리, EDA (5-10차시)

         ↓ 모든 준비 완료!

Part III: 머신러닝 (오늘부터!)
```

<div class="tip">

데이터 준비가 끝났으니 이제 **예측 모델**을 만듭니다!

</div>

---

# 학습 목표

이 차시를 마치면 다음을 할 수 있습니다:

| 번호 | 학습 목표 |
|:----:|----------|
| 1 | **머신러닝의 개념**을 설명한다 |
| 2 | **지도학습과 비지도학습**을 구분한다 |
| 3 | **분류와 회귀 문제**를 구분한다 |

---

# 오늘의 학습 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Part 1    │    │   Part 2    │    │   Part 3    │
│  머신러닝   │ →  │  지도학습   │ → │  분류/회귀  │
│    개념     │    │  비지도학습  │    │    구분     │
└─────────────┘    └─────────────┘    └─────────────┘
   정의와 원리       학습 유형          문제 유형
   전통 vs ML        정답 유무          출력 형태
```

---

<!-- _class: lead -->

# Part 1

## 머신러닝의 개념

---

# 머신러닝이란?

## Machine Learning (기계 학습)

> 명시적으로 프로그래밍하지 않아도
> **데이터로부터 스스로 패턴을 학습**하는 알고리즘

### 핵심 아이디어
- 사람이 규칙을 정하는 게 아니라
- **데이터가 규칙을 알려줌**

---

# 전통적 프로그래밍 vs 머신러닝

## 접근 방식의 차이

| 구분 | 전통적 프로그래밍 | 머신러닝 |
|------|------------------|----------|
| **규칙** | 사람이 직접 작성 | 데이터에서 학습 |
| **예시** | `if 온도 > 90: 불량` | 데이터가 알려줌: `온도 > 87.3` |
| **유연성** | 규칙 변경 시 코드 수정 | 새 데이터로 재학습 |
| **복잡성** | 규칙이 많아지면 한계 | 복잡한 패턴도 학습 |

---

# 전통적 프로그래밍

## 규칙을 직접 작성

```python
# 사람이 규칙을 정의
def predict_defect(temperature, humidity, speed):
    if temperature > 90:
        return "불량"
    if humidity > 70 and speed > 120:
        return "불량"
    if temperature > 85 and humidity > 60:
        return "불량"
    return "정상"
```

### 문제점
- 규칙이 많아지면 관리 어려움
- 새로운 패턴 발견 어려움
- 미묘한 경계값 설정 어려움

---

# 머신러닝 방식

## 데이터에서 규칙 학습

```python
# 데이터만 제공
X = [[온도, 습도, 속도], ...]
y = [정상, 불량, 정상, ...]

# 모델이 스스로 규칙 학습
model.fit(X, y)

# 학습된 규칙으로 예측
model.predict([[85, 50, 100]])  # → "정상"
```

### 장점
- 복잡한 패턴도 자동 발견
- 새 데이터로 재학습 가능
- 미묘한 경계도 학습

---

# 머신러닝의 핵심 개념

## 학습 = 패턴 찾기

```
     [데이터]              [학습]              [예측]
        │                    │                   │
        ▼                    ▼                   ▼
  온도, 습도, 속도    →    모델 학습    →    불량 여부
  (특성, Features)        (패턴 발견)       (타겟, Target)
```

---

# 핵심 용어 정리

## 반드시 알아야 할 용어

| 용어 | 영어 | 설명 |
|------|------|------|
| **특성** | Feature | 입력 데이터, 예측에 사용하는 정보 |
| **타겟** | Target | 예측하려는 값, 정답 |
| **모델** | Model | 학습된 패턴, 예측 기계 |
| **학습** | Training | 데이터에서 패턴을 찾는 과정 |
| **예측** | Prediction | 새 데이터에 패턴 적용 |

---

# 머신러닝이 적합한 경우

## 언제 ML을 사용하나?

<div class="highlight">

### ML이 적합한 상황
1. **규칙이 복잡**하거나 명확하지 않을 때
2. **데이터가 충분**할 때 (수백~수천 건 이상)
3. **패턴이 변화**할 때 (재학습으로 적응)
4. **미묘한 판단**이 필요할 때

### ML이 부적합한 상황
- 명확한 규칙이 있을 때 (if-else로 충분)
- 데이터가 매우 적을 때
- 100% 정확도가 필요할 때

</div>

---

# 제조 현장의 ML 활용

## 실제 적용 사례

| 분야 | 기존 방식 | ML 적용 |
|------|----------|---------|
| **품질 검사** | 육안/규칙 기반 | 이미지 분류 모델 |
| **설비 유지보수** | 정기 점검 | 고장 예측 모델 |
| **불량 예측** | 경험 기반 판단 | 센서 데이터 분석 |
| **생산 계획** | 수작업 예측 | 수요 예측 모델 |

---

<!-- _class: lead -->

# Part 2

## 지도학습과 비지도학습

---

# 머신러닝의 종류

## 3가지 주요 유형

```
                머신러닝
                   │
      ┌────────────┼────────────┐
      │            │            │
   지도학습     비지도학습    강화학습
 (Supervised)  (Unsupervised) (Reinforcement)
      │            │            │
   정답 있음    정답 없음     보상 기반
```

<div class="tip">

이 과정에서는 **지도학습**에 집중합니다!

</div>

---

# 지도학습 (Supervised Learning)

## 정답이 있는 데이터로 학습

> "선생님이 정답을 알려주며 가르치는 것"

### 특징
- **레이블(정답)**이 있는 데이터 필요
- 입력(X)과 출력(y)의 관계 학습
- 새 입력에 대해 출력 예측

---

# 지도학습 예시

## 코드로 보는 지도학습

```python
# 학습 데이터 (특성 + 정답)
X = [[80, 45, 100],   # 온도, 습도, 속도
     [85, 50, 110],
     [92, 65, 105],
     ...]
y = ["정상", "정상", "불량", ...]  # 정답 (레이블)

# 모델이 X와 y의 관계를 학습
model.fit(X, y)

# 새 데이터로 예측 (정답 모름)
model.predict([[87, 55, 108]])  # → "정상" (예측)
```

---

# 지도학습 제조 현장 예시

## 실무 적용

| 문제 | 입력 (특성) | 출력 (타겟) |
|------|------------|------------|
| 불량 분류 | 온도, 습도, 속도 | 불량/정상 |
| 생산량 예측 | 설비, 원료, 인력 | 생산량(숫자) |
| 품질 등급 | 성분, 외관, 강도 | A/B/C 등급 |
| 고장 예측 | 진동, 온도, 가동시간 | 고장 여부 |

---

# 비지도학습 (Unsupervised Learning)

## 정답 없이 패턴 발견

> "정답 없이 스스로 패턴을 찾는 것"

### 특징
- 레이블(정답) **없음**
- 데이터의 **구조/패턴** 발견
- 군집화, 차원 축소, 이상 탐지

---

# 비지도학습 예시

## 코드로 보는 비지도학습

```python
# 학습 데이터 (특성만, 정답 없음!)
X = [[80, 45, 100],
     [85, 50, 110],
     [120, 80, 150],  # 다른 패턴
     ...]

# 모델이 데이터의 구조를 발견
model.fit(X)

# 결과: "이 데이터는 3개 그룹으로 나뉩니다"
labels = model.predict(X)  # [0, 0, 1, 0, 1, ...]
```

---

# 비지도학습 제조 현장 예시

## 실무 적용

| 문제 | 방법 | 설명 |
|------|------|------|
| **제품 군집화** | 클러스터링 | 유사 제품 그룹 발견 |
| **이상 탐지** | 이상탐지 알고리즘 | 정상 패턴 벗어난 데이터 |
| **차원 축소** | PCA | 변수 개수 줄이기 |
| **고객 세분화** | 클러스터링 | 고객 그룹 분류 |

---

# 지도학습 vs 비지도학습

## 핵심 차이

| 구분 | 지도학습 | 비지도학습 |
|------|----------|-----------|
| **정답** | 있음 (레이블) | 없음 |
| **목표** | 예측 | 구조 발견 |
| **대표 문제** | 분류, 회귀 | 군집화, 차원축소 |
| **제조 예시** | 불량 예측 | 이상 탐지 |

---

# 강화학습 (참고)

## 보상 기반 학습

> "시행착오를 통해 최적 행동 학습"

### 특징
- 에이전트가 환경과 상호작용
- 보상을 최대화하는 행동 학습
- 게임, 로봇, 자율주행에 활용

<div class="highlight">

제조 현장에서는 지도학습이 가장 많이 사용됩니다.

</div>

---

<!-- _class: lead -->

# Part 3

## 분류와 회귀 문제 구분

---

# 지도학습의 두 가지 문제

## 분류 vs 회귀

```
             지도학습
                │
        ┌───────┴───────┐
        │               │
      분류            회귀
 (Classification)  (Regression)
        │               │
    범주 예측        숫자 예측
```

---

# 분류 (Classification)

## 범주(카테고리)를 예측

```
입력: 온도 85도, 습도 50%, 속도 100
출력: "정상" 또는 "불량"  ← 범주!
```

### 특징
- 출력이 **정해진 범주** 중 하나
- 이산적(discrete) 값
- 확률로 해석 가능

---

# 분류 문제 예시

## 제조 현장 실무

| 문제 | 입력 | 출력 (범주) |
|------|------|-----------|
| 불량 분류 | 센서 데이터 | 불량 / 정상 |
| 품질 등급 | 측정값 | A / B / C |
| 고장 유형 | 진동 패턴 | 모터 / 베어링 / 전기 |
| 설비 상태 | 가동 데이터 | 정상 / 점검필요 / 교체필요 |

---

# 분류 유형

## 이진 분류 vs 다중 분류

### 이진 분류 (Binary Classification)
- 2개 클래스 중 하나 선택
- 예: 불량/정상, 합격/불합격

### 다중 분류 (Multi-class Classification)
- 3개 이상 클래스 중 하나 선택
- 예: A/B/C 등급, 불량유형 분류

---

# 회귀 (Regression)

## 연속적인 숫자를 예측

```
입력: 온도 85도, 습도 50%, 속도 100
출력: 1,247개  ← 숫자!
```

### 특징
- 출력이 **연속적인 숫자**
- 어떤 값이든 나올 수 있음
- 오차를 최소화

---

# 회귀 문제 예시

## 제조 현장 실무

| 문제 | 입력 | 출력 (숫자) |
|------|------|-----------|
| 생산량 예측 | 설비, 인력 | 1,247개 |
| 불량률 예측 | 공정 조건 | 3.5% |
| 설비 수명 | 사용 이력 | 87일 |
| 소요 시간 | 작업 조건 | 4.2시간 |

---

# 분류 vs 회귀 구분법

## "출력이 뭔가요?"

| 질문 | 분류 | 회귀 |
|------|:----:|:----:|
| 이 제품 불량인가요? | ✅ | |
| 불량률이 몇 %? | | ✅ |
| 어떤 등급인가요? | ✅ | |
| 생산량이 얼마? | | ✅ |
| 고장 유형은? | ✅ | |
| 남은 수명은? | | ✅ |

---

# 구분 팁

## 간단한 질문으로 구분

<div class="highlight">

### 분류
- **"~인가요?"** (예/아니오)
- **"어떤 종류인가요?"** (범주 선택)

### 회귀
- **"얼마나?"** (숫자)
- **"몇 개?"** (개수)
- **"몇 %?"** (비율)

</div>

---

# sklearn 소개

## 머신러닝의 표준 라이브러리

```python
# scikit-learn
import sklearn
print(sklearn.__version__)
```

### 왜 sklearn인가?
- **일관된 API**: fit, predict, score 패턴
- **다양한 알고리즘**: 분류, 회귀, 군집화 등
- **전처리/평가 도구** 포함
- **풍부한 문서**와 커뮤니티

---

# sklearn 기본 흐름

## 4단계 패턴

```python
# 1. 데이터 준비
X = df[['온도', '습도', '속도']]  # 특성
y = df['불량여부']                # 타겟

# 2. 모델 생성
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# 3. 학습 (fit)
model.fit(X, y)

# 4. 예측 (predict)
predictions = model.predict(X_new)
```

---

# 학습/테스트 데이터 분리

## 왜 분리하나요?

```
       전체 데이터
           │
    ┌──────┴──────┐
    │             │
 학습 데이터   테스트 데이터
   (80%)        (20%)
    │             │
 패턴 학습    성능 평가
```

<div class="important">

시험 문제를 미리 보면 안 되듯이,
모델도 **처음 보는 데이터**로 평가해야 합니다!

</div>

---

# train_test_split

## 데이터 분리 함수

```python
from sklearn.model_selection import train_test_split

# 80% 학습, 20% 테스트
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 테스트 비율 (20%)
    random_state=42    # 재현성 (같은 결과)
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
```

---

# 분류 모델 예시

## DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성
model = DecisionTreeClassifier(random_state=42)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.2%}")
```

---

# 회귀 모델 예시

## LinearRegression

```python
from sklearn.linear_model import LinearRegression

# 모델 생성
model = LinearRegression()

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# R² 점수 평가
r2 = model.score(X_test, y_test)
print(f"R² 점수: {r2:.3f}")
```

---

<!-- _class: lead -->

# 실습편

## sklearn 기초 실습

---

# 실습 개요

## 제조 데이터로 분류/회귀 체험

### 목표
1. 분류 모델로 불량 여부 예측
2. 회귀 모델로 생산량 예측
3. sklearn 기본 패턴 익히기

### 핵심
- fit → predict → score 패턴

---

# 실습 1: 데이터 생성

## 제조 데이터 준비

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

df = pd.DataFrame({
    'temperature': np.random.normal(85, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'speed': np.random.normal(100, 15, n),
})

# 불량 여부 (분류 타겟)
defect_prob = 0.1 + 0.02 * (df['temperature'] - 80)
df['defect'] = (np.random.random(n) < defect_prob).astype(int)

# 생산량 (회귀 타겟)
df['production'] = 1000 + 5*df['speed'] - 3*df['temperature'] + \
                   np.random.normal(0, 50, n)
```

---

# 실습 2: 특성과 타겟 분리

## X와 y 정의

```python
# 특성 (입력)
X = df[['temperature', 'humidity', 'speed']]

# 분류 타겟
y_clf = df['defect']  # 0 또는 1
print(f"불량 비율: {y_clf.mean():.1%}")

# 회귀 타겟
y_reg = df['production']  # 연속 숫자
print(f"생산량 범위: {y_reg.min():.0f} ~ {y_reg.max():.0f}")
```

---

# 실습 3: 학습/테스트 분리

## train_test_split 사용

```python
from sklearn.model_selection import train_test_split

# 분류용 데이터 분리
X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# 회귀용 데이터 분리
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print(f"학습: {len(X_train)}개, 테스트: {len(X_test)}개")
```

---

# 실습 4: 분류 모델

## 불량 여부 예측

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성 및 학습
model_clf = DecisionTreeClassifier(random_state=42)
model_clf.fit(X_train, y_train_clf)

# 예측
y_pred_clf = model_clf.predict(X_test)

# 정확도
accuracy = model_clf.score(X_test, y_test_clf)
print(f"분류 정확도: {accuracy:.1%}")
```

---

# 실습 5: 회귀 모델

## 생산량 예측

```python
from sklearn.linear_model import LinearRegression

# 모델 생성 및 학습
model_reg = LinearRegression()
model_reg.fit(X_train, y_train_reg)

# 예측
y_pred_reg = model_reg.predict(X_test)

# R² 점수
r2 = model_reg.score(X_test, y_test_reg)
print(f"R² 점수: {r2:.3f}")
```

---

# 실습 6: 새 데이터 예측

## 실제 활용

```python
# 새로운 제품 데이터
new_product = pd.DataFrame({
    'temperature': [87],
    'humidity': [55],
    'speed': [105]
})

# 분류: 불량 여부
defect_pred = model_clf.predict(new_product)
print(f"불량 예측: {'불량' if defect_pred[0] == 1 else '정상'}")

# 회귀: 생산량
prod_pred = model_reg.predict(new_product)
print(f"생산량 예측: {prod_pred[0]:.0f}개")
```

---

# 실습 정리

## sklearn 기본 패턴 (모든 모델 공통!)

```python
# 1. 모델 생성
model = ModelName()

# 2. 학습
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가
score = model.score(X_test, y_test)
```

---

# 핵심 정리

## 12차시 요약

| 개념 | 설명 |
|------|------|
| **머신러닝** | 데이터에서 패턴을 자동 학습 |
| **지도학습** | 정답(레이블) 있는 데이터로 학습 |
| **비지도학습** | 정답 없이 구조/패턴 발견 |
| **분류** | 범주 예측 ("~인가요?") |
| **회귀** | 숫자 예측 ("얼마나?") |
| **sklearn** | fit → predict → score 패턴 |

---

# 다음 차시 예고

## 12차시: 분류 모델 (1) - 의사결정나무

### 학습 내용
- 의사결정나무 원리와 구조
- DecisionTreeClassifier 상세 사용법
- 트리 시각화와 해석

<div class="tip">

첫 번째 AI 분류 모델을 본격적으로 만듭니다!

</div>

---

# 감사합니다

## 12차시: 머신러닝 소개와 문제 유형

**Part III가 시작되었습니다!**

Q&A
