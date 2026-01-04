---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 10차시'
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

# 머신러닝 소개와 문제 유형

## 10차시 | Part III. 문제 중심 모델링 실습

**드디어 AI 모델을 만듭니다!**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **머신러닝의 개념**을 설명한다
2. **지도학습과 비지도학습**을 구분한다
3. **분류와 회귀** 문제를 구분한다

---

# Part III 시작!

## 지금까지 배운 것

```
Part I: Python 기초, 데이터 다루기
Part II: 통계, 시각화, 전처리, EDA

         ↓ 모든 준비 완료!

Part III: 머신러닝 (오늘부터!)
```

> 데이터 준비가 끝났으니 이제 **예측 모델**을 만듭니다!

---

# 머신러닝이란?

## Machine Learning (기계 학습)

> 명시적으로 프로그래밍하지 않아도
> **데이터로부터 스스로 학습**하는 알고리즘

### 전통적 프로그래밍 vs 머신러닝

| 전통적 프로그래밍 | 머신러닝 |
|------------------|----------|
| 규칙을 **직접 작성** | 데이터에서 규칙을 **학습** |
| if 온도 > 90 → 불량 | 데이터가 알려줌: 온도 > 87 → 불량 |

---

# 머신러닝의 핵심 개념

## 학습 = 패턴 찾기

```
     [데이터]           [학습]           [예측]
        │                 │                │
        ▼                 ▼                ▼
  온도, 습도, 속도  →  모델 학습  →  불량 여부 예측
  (특성, Features)      (패턴)        (타겟, Target)
```

### 핵심 용어
- **특성(Feature)**: 입력 데이터 (X)
- **타겟(Target)**: 예측하려는 값 (y)
- **모델(Model)**: 학습된 패턴

---

# 머신러닝의 종류

## 3가지 유형

```
              머신러닝
                 │
    ┌────────────┼────────────┐
    │            │            │
 지도학습     비지도학습    강화학습
(Supervised) (Unsupervised) (Reinforcement)
    │            │            │
 정답 있음    정답 없음     보상 기반
```

> 이 과정에서는 **지도학습**에 집중합니다!

---

# 지도학습 (Supervised Learning)

## 정답이 있는 데이터로 학습

```python
# 학습 데이터
X = [[온도, 습도, 속도], ...]  # 특성
y = [정상, 불량, 정상, ...]    # 정답 (레이블)

# 모델이 X와 y의 관계를 학습
model.fit(X, y)

# 새 데이터로 예측
model.predict([[85, 45, 120]])  # → 정상
```

### 제조 현장 예시
- 제품 **불량/정상** 분류
- **생산량** 예측
- 설비 **고장** 예측

---

# 비지도학습 (Unsupervised Learning)

## 정답 없이 패턴 발견

```python
# 학습 데이터 (정답 없음!)
X = [[온도, 습도, 속도], ...]

# 모델이 데이터의 구조를 발견
model.fit(X)

# 결과: 3개 그룹으로 나뉨
```

### 제조 현장 예시
- 제품 군집화 (유사 제품 그룹)
- 이상 탐지 (정상 패턴 학습 후 이상 감지)
- 차원 축소

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

### 제조 현장 실무 예시
- 제품 **불량/정상** 분류
- 품질 **A등급/B등급/C등급** 분류
- 설비 **정상/점검필요/교체필요** 판정

> 출력이 **정해진 범주** 중 하나

---

# 회귀 (Regression)

## 연속적인 숫자를 예측

```
입력: 온도 85도, 습도 50%, 속도 100
출력: 1,247개  ← 숫자!
```

### 제조 현장 실무 예시
- **생산량** 예측
- **불량률** 예측 (0.03 = 3%)
- **설비 수명** 예측 (남은 시간)

> 출력이 **연속적인 숫자**

---

# 분류 vs 회귀 구분법

## "출력이 뭔가요?"

| 질문 | 분류 | 회귀 |
|------|------|------|
| 불량인가요? | ✅ 예/아니오 | |
| 불량률이 몇 %? | | ✅ 숫자 |
| 어떤 등급인가요? | ✅ A/B/C | |
| 생산량이 얼마? | | ✅ 숫자 |

### 구분 팁
- **"~인가요?"** → 분류
- **"얼마나?"** → 회귀

---

# sklearn 소개

## 머신러닝의 표준 라이브러리

```python
# scikit-learn 설치 (보통 이미 설치됨)
# pip install scikit-learn

import sklearn
print(sklearn.__version__)
```

### 왜 sklearn인가?
- 일관된 API (fit, predict, score)
- 다양한 알고리즘 제공
- 전처리, 평가 도구 포함
- 풍부한 문서와 커뮤니티

---

# sklearn 기본 흐름

## 4단계 패턴

```python
# 1. 데이터 준비
X = df[['온도', '습도', '속도']]
y = df['불량여부']

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

> 시험 문제를 미리 보면 안 되듯이,
> 모델도 **처음 보는 데이터**로 평가해야 합니다!

---

# train_test_split

## 데이터 분리 함수

```python
from sklearn.model_selection import train_test_split

# 80% 학습, 20% 테스트
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 테스트 비율
    random_state=42   # 재현성
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
```

---

# 이론 정리

## 핵심 개념

| 개념 | 설명 |
|------|------|
| 머신러닝 | 데이터에서 패턴을 학습 |
| 지도학습 | 정답이 있는 데이터로 학습 |
| 분류 | 범주 예측 (예/아니오) |
| 회귀 | 숫자 예측 (얼마나?) |
| sklearn | 머신러닝 표준 라이브러리 |

---

# - 실습편 -

## 10차시

**sklearn 기초 실습**

---

# 실습 개요

## 제조 데이터로 분류/회귀 체험

### 목표
- 분류 모델: 불량 여부 예측
- 회귀 모델: 생산량 예측
- sklearn 기본 패턴 익히기

### 실습 환경
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
```

---

# 실습 1: 데이터 생성

## 제조 데이터 준비

```python
np.random.seed(42)
n = 200

df = pd.DataFrame({
    '온도': np.random.normal(85, 5, n),
    '습도': np.random.normal(50, 10, n),
    '속도': np.random.normal(100, 15, n),
})

# 불량 여부 (분류 타겟)
defect_prob = 0.1 + 0.02 * (df['온도'] - 80)
df['불량여부'] = (np.random.random(n) < defect_prob).astype(int)

# 생산량 (회귀 타겟)
df['생산량'] = 1000 + 5*df['속도'] - 3*df['온도'] + np.random.normal(0, 50, n)
```

---

# 실습 2: 특성과 타겟 분리

## X와 y 정의

```python
# 분류 문제
X_clf = df[['온도', '습도', '속도']]  # 특성
y_clf = df['불량여부']                 # 타겟 (0 또는 1)

print(f"타겟 분포:\n{y_clf.value_counts()}")

# 회귀 문제
X_reg = df[['온도', '습도', '속도']]
y_reg = df['생산량']                   # 타겟 (숫자)

print(f"생산량 범위: {y_reg.min():.0f} ~ {y_reg.max():.0f}")
```

---

# 실습 3: 학습/테스트 분리

## train_test_split 사용

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf,
    test_size=0.2,
    random_state=42
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
```

---

# 실습 4: 분류 모델 (의사결정트리)

## 불량 여부 예측

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성
model = DecisionTreeClassifier(random_state=42)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도
accuracy = (y_pred == y_test).mean()
print(f"정확도: {accuracy:.1%}")
```

---

# 실습 5: 회귀 모델 (선형회귀)

## 생산량 예측

```python
from sklearn.linear_model import LinearRegression

# 데이터 분리 (회귀용)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 모델 생성 및 학습
model_reg = LinearRegression()
model_reg.fit(X_train_r, y_train_r)

# 예측
y_pred_r = model_reg.predict(X_test_r)

# R² 점수
r2 = model_reg.score(X_test_r, y_test_r)
print(f"R² 점수: {r2:.3f}")
```

---

# 실습 6: 분류 vs 회귀 구분 연습

## 문제 유형 구분

```
Q: 제품이 불량인가요?
→ 분류 (예/아니오)

Q: 불량률이 몇 %일까요?
→ 회귀 (숫자)

Q: 다음 달 생산량은?
→ 회귀 (숫자)

Q: 품질 등급은?
→ 분류 (A/B/C)
```

---

# 실습 정리

## sklearn 기본 패턴

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

> 모든 sklearn 모델이 이 패턴을 따릅니다!

---

# 다음 차시 예고

## 11차시: 분류 모델 (1) - 의사결정나무

### 학습 내용
- 의사결정나무 원리
- DecisionTreeClassifier 상세 사용법
- 불량 분류 모델 구축

> 첫 번째 AI 모델을 본격적으로 만듭니다!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **머신러닝**: 데이터에서 패턴을 학습
2. **지도학습**: 정답 있는 데이터로 학습 (분류/회귀)
3. **분류**: 범주 예측 ("~인가요?")
4. **회귀**: 숫자 예측 ("얼마나?")
5. **sklearn**: fit → predict → score 패턴

---

# 감사합니다

## 10차시: 머신러닝 소개와 문제 유형

**Part III가 시작되었습니다!**
