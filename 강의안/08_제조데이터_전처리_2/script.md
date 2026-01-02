# [8차시] 제조 데이터 전처리 (2) - 강사 스크립트

## 강의 정보
- **차시**: 8차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 8차시를 시작하겠습니다.
>
> 지난 시간에 결측치와 이상치 처리를 배웠죠? 비어있는 값을 채우고, 극단값을 처리하는 방법을 익혔습니다.
>
> 오늘은 전처리의 두 번째 단계, **스케일링과 인코딩**을 배웁니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, 스케일링(정규화, 표준화)의 필요성을 이해합니다.
> 둘째, 범주형 데이터를 인코딩하는 방법을 적용합니다.
> 셋째, sklearn의 전처리 도구를 활용합니다.
>
> 머신러닝 모델에 데이터를 넣기 전에 꼭 해야 하는 작업들입니다.

---

### 핵심 내용 (8분)

#### 왜 스케일링이 필요한가? [1.5분]

> 먼저 **스케일링**이 왜 필요한지 알아봅시다.
>
> 제조 데이터를 보면, 온도는 80~100 정도고, 생산량은 1000~1500 정도입니다.
> 범위가 20 대 500으로 25배 차이가 나죠.
>
> 이게 왜 문제일까요?
>
> 스케일이 큰 변수가 모델에 과도한 영향을 미칩니다. 마치 큰 소리가 작은 소리를 덮어버리는 것처럼요.
> 또 KNN 같은 거리 기반 알고리즘에서는 스케일 차이 때문에 결과가 왜곡됩니다.
>
> 그래서 스케일을 맞춰줘야 합니다.

#### 표준화 (StandardScaler) [1.5분]

> **표준화**는 데이터를 평균 0, 표준편차 1로 변환합니다.
>
> Z = (값 - 평균) / 표준편차
>
> 지난 시간에 배운 Z-score와 같은 공식이에요.
>
> sklearn에서는 `StandardScaler`를 사용합니다.
>
> ```python
> from sklearn.preprocessing import StandardScaler
> scaler = StandardScaler()
> X_scaled = scaler.fit_transform(X)
> ```
>
> 세 줄이면 됩니다. 대부분의 머신러닝 모델에서 이 방법을 먼저 시도합니다.

#### 정규화 (MinMaxScaler) [1분]

> **정규화**는 데이터를 0~1 범위로 변환합니다.
>
> 최소값이 0이 되고, 최대값이 1이 되죠.
>
> sklearn에서는 `MinMaxScaler`를 사용합니다.
>
> ```python
> from sklearn.preprocessing import MinMaxScaler
> scaler = MinMaxScaler()
> X_scaled = scaler.fit_transform(X)
> ```
>
> 신경망 모델에서 주로 사용합니다. 입력값의 범위를 제한해서 학습을 안정화시키거든요.

#### 범주형 데이터 [1분]

> 이제 **인코딩**을 배워봅시다.
>
> 제조 데이터에는 숫자가 아닌 데이터도 있죠. 라인 A, B, C. 등급 상, 중, 하 같은 거요.
>
> 문제는 대부분의 머신러닝 모델이 **숫자만 입력받을 수 있다**는 겁니다.
>
> 그래서 범주형 데이터를 숫자로 바꿔야 해요. 이걸 **인코딩**이라고 합니다.

#### 레이블 인코딩 [1분]

> **레이블 인코딩**은 범주를 0, 1, 2 같은 숫자로 바꿉니다.
>
> 상 → 0, 중 → 1, 하 → 2 이렇게요.
>
> sklearn에서는 `LabelEncoder`를 사용합니다.
>
> 주의할 점이 있어요. 숫자로 바꾸면 크기 관계가 생깁니다. 0 < 1 < 2
>
> 등급처럼 순서가 있는 범주에는 괜찮지만, 라인 A, B, C처럼 순서가 없는 범주에는 부적합합니다.

#### 원-핫 인코딩 [1.5분]

> **원-핫 인코딩**은 각 범주를 별도의 열로 만듭니다.
>
> 라인이 A, B, C 세 가지면 라인_A, 라인_B, 라인_C 세 개의 열이 생깁니다.
> 해당 범주면 1, 아니면 0이 들어가요.
>
> pandas에서는 `get_dummies`를 사용하면 간편합니다.
>
> ```python
> df_encoded = pd.get_dummies(df, columns=['라인'])
> ```
>
> 순서가 없는 범주에는 이 방법을 사용하세요.
>
> 단점은 열 수가 늘어난다는 거예요. 범주가 100개면 열이 100개 생기죠. 이럴 때는 다른 방법을 고려해야 합니다.

---

## 실습편 (15-20분)

### 실습 소개 [2분]

> 이제 실습 시간입니다. 오늘은 스케일링과 인코딩을 직접 해보겠습니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from sklearn.preprocessing import StandardScaler, MinMaxScaler
> from sklearn.preprocessing import LabelEncoder, OneHotEncoder
> ```
>
> sklearn의 preprocessing 모듈에서 다양한 도구를 가져옵니다.

### 실습 1: 데이터 준비 [2분]

> 첫 번째 실습입니다. 샘플 데이터를 만들어봅시다.
>
> ```python
> np.random.seed(42)
> n = 100
>
> df = pd.DataFrame({
>     '온도': np.random.normal(85, 5, n),
>     '생산량': np.random.normal(1200, 50, n),
>     '습도': np.random.normal(60, 10, n),
>     '라인': np.random.choice(['A', 'B', 'C'], n),
>     '등급': np.random.choice(['상', '중', '하'], n)
> })
> ```
>
> 수치형 3개, 범주형 2개 변수가 있는 데이터입니다.

### 실습 2: 스케일 차이 확인 [2분]

> 두 번째 실습입니다. 변수별 범위를 확인해봅시다.
>
> ```python
> for col in ['온도', '생산량', '습도']:
>     print(f"{col}: {df[col].min():.1f} ~ {df[col].max():.1f}")
> ```
>
> 생산량의 범위가 온도나 습도보다 훨씬 크죠? 이게 스케일링이 필요한 이유입니다.

### 실습 3: 표준화 적용 [2분]

> 세 번째 실습입니다. StandardScaler를 적용해봅시다.
>
> ```python
> from sklearn.preprocessing import StandardScaler
>
> numeric_cols = ['온도', '생산량', '습도']
> X = df[numeric_cols].values
>
> scaler_std = StandardScaler()
> X_std = scaler_std.fit_transform(X)
>
> df_std = pd.DataFrame(X_std, columns=numeric_cols)
> print(df_std.describe())
> ```
>
> 평균이 0에 가깝고, 표준편차가 1에 가깝게 변했죠?

### 실습 4: 정규화 적용 [2분]

> 네 번째 실습입니다. MinMaxScaler를 적용해봅시다.
>
> ```python
> from sklearn.preprocessing import MinMaxScaler
>
> scaler_mm = MinMaxScaler()
> X_mm = scaler_mm.fit_transform(X)
>
> df_mm = pd.DataFrame(X_mm, columns=numeric_cols)
> print(df_mm.describe())
> ```
>
> 최소값이 0, 최대값이 1로 바뀌었습니다.

### 실습 5: 레이블 인코딩 [2분]

> 다섯 번째 실습입니다. 등급을 레이블 인코딩해봅시다.
>
> ```python
> from sklearn.preprocessing import LabelEncoder
>
> le = LabelEncoder()
> df['등급_숫자'] = le.fit_transform(df['등급'])
>
> print(df[['등급', '등급_숫자']].drop_duplicates())
> ```
>
> 상, 중, 하가 각각 숫자로 바뀌었습니다. 알파벳 순서대로 0, 1, 2가 됩니다.

### 실습 6: 원-핫 인코딩 [2분]

> 여섯 번째 실습입니다. 라인을 원-핫 인코딩해봅시다.
>
> ```python
> df_onehot = pd.get_dummies(df, columns=['라인'], prefix='라인')
> print(df_onehot[['라인_A', '라인_B', '라인_C']].head())
> ```
>
> 라인_A, 라인_B, 라인_C 세 개의 열이 생겼습니다. 해당 라인이면 1, 아니면 0입니다.

### 실습 7: 역변환 [2분]

> 마지막 실습입니다. 스케일링을 역변환해봅시다.
>
> ```python
> X_original = scaler_std.inverse_transform(X_std)
> print("원본:", df[numeric_cols].iloc[0].values.round(2))
> print("복원:", X_original[0].round(2))
> ```
>
> 모델 예측 후 원래 스케일로 복원할 때 사용합니다. 예측값이 표준화된 값이면 해석하기 어렵거든요.

---

### 정리 (3분)

#### 핵심 요약 [1.5분]

> 오늘 배운 내용을 정리하겠습니다.
>
> **스케일링**: 변수 간 스케일 차이를 맞춥니다.
> - StandardScaler는 평균 0, 표준편차 1로 변환
> - MinMaxScaler는 0~1 범위로 변환
>
> **인코딩**: 범주형 데이터를 숫자로 변환합니다.
> - LabelEncoder는 순서 있는 범주에 사용
> - OneHotEncoder나 get_dummies는 순서 없는 범주에 사용

#### 주의사항 [0.5분]

> 중요한 주의사항입니다.
>
> **fit은 학습 데이터에만** 적용합니다. 테스트 데이터에는 transform만 해야 합니다.
>
> 테스트 데이터에도 fit을 하면 데이터 누수가 발생해서 모델 성능이 과대 평가됩니다.

#### 다음 차시 예고 [0.5분]

> 다음 9차시에서는 **제조 데이터 탐색 분석 종합**을 배웁니다.
>
> 1~8차시에서 배운 내용을 총정리하면서, 처음부터 끝까지 EDA를 해보겠습니다.

#### 마무리 [0.5분]

> 오늘 스케일링과 인코딩을 배웠습니다.
>
> 이제 결측치, 이상치, 스케일링, 인코딩까지 전처리의 핵심을 모두 익혔습니다. 다음 시간에 종합 실습으로 복습하겠습니다.
>
> 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- sklearn 설치 확인

### 주의사항
- fit_transform vs transform 차이 강조
- 원-핫 인코딩 후 열 수 증가 주의
- 테스트 데이터 누수 방지

### 예상 질문
1. "StandardScaler와 MinMaxScaler 중 뭘 써야 하나요?"
   → 일반적으로 StandardScaler 먼저. 신경망은 MinMaxScaler 권장

2. "원-핫 인코딩하면 열이 너무 많아지는데?"
   → 타겟 인코딩, 빈도 인코딩 등 대안 있음. 일단은 원-핫으로 시작

3. "fit_transform과 transform의 차이가 뭔가요?"
   → fit은 평균/표준편차 등을 학습. transform은 학습된 값으로 변환만

4. "인코딩된 데이터를 다시 원래대로 돌릴 수 있나요?"
   → LabelEncoder.inverse_transform() 사용. 원-핫은 argmax로 역변환
