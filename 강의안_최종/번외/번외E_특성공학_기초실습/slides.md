---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# 번외E: 특성 공학 기초 실습
## 좋은 특성이 좋은 모델을 만든다!

**AI 기초체력훈련 과정**
제조 AI 예측 모델 개발

---

## 학습 목표

1. **특성 공학**의 개념과 중요성 이해
2. 수치형 특성 **변환** 기법 실습
3. 범주형 특성 **인코딩** 고급 기법 실습
4. **특성 선택**으로 중요한 특성 찾기
5. 특성 공학 전후 **성능 비교**

---

## 왜 특성 공학이 중요한가?

> "Garbage in, garbage out"
> (쓰레기를 넣으면 쓰레기가 나온다)

| 상황 | 결과 |
|------|------|
| 좋은 특성 + 단순 모델 | 좋은 성능 |
| 나쁜 특성 + 복잡한 모델 | 나쁜 성능 |

**핵심**: 모델보다 **데이터/특성**이 더 중요!

---

## 특성 공학이란?

```
원본 데이터 → [특성 공학] → 개선된 특성 → 더 좋은 모델
```

| 활동 | 설명 |
|------|------|
| 변환 | 로그, 제곱근, 비율 등 |
| 생성 | 기존 특성 조합으로 새 특성 |
| 선택 | 불필요한 특성 제거 |
| 인코딩 | 범주형 → 수치형 변환 |

---

## 오늘 실습할 데이터셋: Tips

| 항목 | 내용 |
|------|------|
| **데이터셋** | Tips (seaborn 내장) |
| **크기** | 244건 |
| **목표** | tip 금액 예측 (회귀) |
| **특성** | 총 금액, 성별, 요일, 시간 등 |

---

# Part 1: 특성 공학 개념
## 좋은 특성의 조건

---

## 좋은 특성의 3가지 조건

| 조건 | 설명 | 예시 |
|------|------|------|
| **정보성** | 타겟과 관련성 | 온도 → 불량률 |
| **독립성** | 다른 특성과 중복 안 됨 | 키, 몸무게 중 하나 |
| **이해 가능** | 해석 가능한 의미 | 팁 비율 |

---

## 특성 공학의 종류

```
┌─────────────────────────────────────────────────────────────┐
│           특성 공학 (Feature Engineering)                   │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│  수치형 변환 │  범주형 인코딩│  특성 생성   │  특성 선택      │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ 로그 변환   │ Label       │ 비율 계산   │ 상관관계 기반   │
│ 표준화      │ One-Hot     │ 날짜 분해   │ 중요도 기반     │
│ 구간화      │ 빈도 인코딩  │ 상호작용    │ 분산 기반       │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

---

# Part 2: 수치형 특성 변환
## 로그 변환, 비율, 구간화

---

## 데이터 로드

```python
import pandas as pd
import numpy as np
import seaborn as sns

# Tips 데이터셋 로드
df = sns.load_dataset('tips')
print(df.head())
print(df.shape)  # (244, 7)
```

---

## 원본 데이터 확인

| 컬럼 | 타입 | 설명 |
|------|------|------|
| total_bill | float | 총 금액 |
| tip | float | 팁 금액 (타겟) |
| sex | category | 성별 |
| smoker | category | 흡연 여부 |
| day | category | 요일 |
| time | category | 시간 |
| size | int | 인원수 |

---

## 2-1. 로그 변환

**문제**: 금액 데이터는 오른쪽으로 치우친 분포 (right-skewed)

```python
# 로그 변환 전
print(f"왜도(skewness): {df['total_bill'].skew():.2f}")

# 로그 변환
df['log_total_bill'] = np.log1p(df['total_bill'])

# 로그 변환 후
print(f"왜도(skewness): {df['log_total_bill'].skew():.2f}")
```

---

## 로그 변환 효과

| 변환 | 왜도 |
|------|------|
| 변환 전 | 1.13 (치우침) |
| 변환 후 | 0.12 (정규 분포에 가까움) |

**log1p 사용 이유**: log(0) = -∞ 방지

```python
np.log1p(x)  # = np.log(x + 1)
```

---

## 2-2. 비율 특성 생성

**아이디어**: 팁 금액보다 **팁 비율**이 더 의미있을 수 있음

```python
# 팁 비율 = 팁 / 총 금액
df['tip_ratio'] = df['tip'] / df['total_bill']

print(df['tip_ratio'].describe())
```

| 통계 | 값 |
|------|-----|
| 평균 | 0.161 (16.1%) |
| 표준편차 | 0.061 |
| 최소 | 0.035 |
| 최대 | 0.710 |

---

## 2-3. 1인당 금액 계산

**아이디어**: 총 금액보다 **1인당 금액**이 더 유용할 수 있음

```python
# 1인당 금액
df['per_person'] = df['total_bill'] / df['size']

print(df['per_person'].describe())
```

**해석**: 1인당 평균 $12.5 지출

---

## 2-4. 구간화 (Binning)

**아이디어**: 연속형 → 범주형으로 변환

```python
# 총 금액을 3구간으로 분류
df['bill_category'] = pd.cut(
    df['total_bill'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)

print(df['bill_category'].value_counts())
```

| 구간 | 수 |
|------|-----|
| Low | 168 |
| Medium | 66 |
| High | 10 |

---

## 구간화 방법 비교

| 방법 | 함수 | 특징 |
|------|------|------|
| 등간격 | pd.cut() | 값 범위를 균등 분할 |
| 등빈도 | pd.qcut() | 데이터 개수를 균등 분할 |

```python
# 등빈도: 각 구간에 동일한 수의 데이터
df['bill_qcat'] = pd.qcut(df['total_bill'], q=3,
                          labels=['Low', 'Medium', 'High'])
```

---

# Part 3: 범주형 특성 인코딩
## 빈도 인코딩, 타겟 인코딩

---

## 기본 인코딩 복습

| 방법 | 코드 | 결과 |
|------|------|------|
| Label | LabelEncoder | Male=0, Female=1 |
| One-Hot | get_dummies | sex_Male, sex_Female |

**한계**: 범주가 많으면 차원 폭발

---

## 3-1. 빈도 인코딩 (Frequency Encoding)

**아이디어**: 범주를 **출현 빈도**로 대체

```python
# 요일별 빈도 계산
day_freq = df['day'].value_counts(normalize=True)
print(day_freq)

# 빈도로 인코딩
df['day_freq'] = df['day'].map(day_freq)
```

| 요일 | 빈도 |
|------|------|
| Sat | 0.358 |
| Sun | 0.311 |
| Thur | 0.255 |
| Fri | 0.078 |

---

## 빈도 인코딩의 장점

| 장점 | 설명 |
|------|------|
| 차원 유지 | 컬럼 수 증가 안 함 |
| 정보 보존 | 빈도 정보 반영 |
| 범주 많아도 OK | 고카디널리티 대응 |

---

## 3-2. 타겟 인코딩 (Target Encoding)

**아이디어**: 범주를 **타겟 평균**으로 대체

```python
# 요일별 팁 평균
day_target = df.groupby('day')['tip'].mean()
print(day_target)

# 타겟 인코딩
df['day_target'] = df['day'].map(day_target)
```

| 요일 | 평균 팁 |
|------|--------|
| Fri | 2.73 |
| Sat | 2.99 |
| Sun | 3.26 |
| Thur | 2.77 |

---

## 타겟 인코딩 주의사항

**위험**: 데이터 누출 (Data Leakage)

| 문제 | 해결책 |
|------|--------|
| 테스트 데이터로 인코딩 | 학습 데이터만 사용 |
| 과적합 위험 | K-Fold 기반 인코딩 |

```python
# 올바른 방법: 학습 데이터로만 계산
train_target_mean = df_train.groupby('day')['tip'].mean()
df_test['day_target'] = df_test['day'].map(train_target_mean)
```

---

# Part 4: 날짜/시간 특성
## 시간 정보에서 특성 추출

---

## 날짜/시간 특성 추출 예시

```python
# 예시: datetime 컬럼이 있다면
df['datetime'] = pd.to_datetime(df['datetime'])

# 다양한 특성 추출
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour
df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
```

---

## 시간 특성의 종류

| 특성 | 코드 | 예시 |
|------|------|------|
| 연도 | .dt.year | 2024 |
| 월 | .dt.month | 1~12 |
| 요일 | .dt.dayofweek | 0(월)~6(일) |
| 시간 | .dt.hour | 0~23 |
| 분기 | .dt.quarter | 1~4 |

---

## 주기적 특성 처리

**문제**: 12월(12)과 1월(1)은 가까운데 숫자로는 멀음

**해결**: 삼각함수 변환

```python
# 월을 주기적 특성으로 변환
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

## Tips 데이터에 시간 특성 적용

```python
# time: 점심/저녁 → 이진 특성
df['is_dinner'] = (df['time'] == 'Dinner').astype(int)

# day: 주말 여부
df['is_weekend'] = df['day'].isin(['Sat', 'Sun']).astype(int)

print(df[['time', 'is_dinner', 'day', 'is_weekend']].head())
```

---

# Part 5: 특성 선택
## 중요한 특성만 남기기

---

## 왜 특성 선택이 필요한가?

| 이유 | 설명 |
|------|------|
| **차원의 저주** | 특성 많으면 과적합 |
| **학습 속도** | 특성 적으면 빠름 |
| **해석 용이** | 중요 특성만 남기면 이해 쉬움 |

---

## 5-1. 상관관계 기반 선택

```python
# 수치형 특성과 타겟의 상관관계
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['tip'].abs()
correlations = correlations.sort_values(ascending=False)

print(correlations)
```

---

## 상관관계 결과

| 특성 | 상관계수 |
|------|----------|
| tip | 1.000 |
| total_bill | 0.676 |
| size | 0.489 |
| per_person | 0.394 |
| log_total_bill | 0.628 |
| tip_ratio | 0.051 |

**해석**: total_bill이 tip과 가장 상관관계 높음

---

## 5-2. 상관관계 기반 필터링

```python
# 상관관계 0.3 이상인 특성만 선택
threshold = 0.3
selected = correlations[correlations > threshold].index.tolist()
selected.remove('tip')  # 타겟 제외

print(f"선택된 특성: {selected}")
```

---

## 5-3. 다중공선성 제거

**문제**: 서로 상관관계 높은 특성들 (예: total_bill, log_total_bill)

```python
# 특성 간 상관관계 확인
feature_corr = df[selected].corr()
print(feature_corr)
```

**해결**: 중복되는 특성 중 하나 제거

---

# Part 6: 성능 비교
## 특성 공학 전 vs 후

---

## 비교 실험 설계

| 실험 | 사용 특성 |
|------|----------|
| **기본** | total_bill, size (원본) |
| **공학 후** | log_total_bill, per_person, is_dinner, is_weekend |

---

## 실험 코드

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 기본 특성
X_basic = df[['total_bill', 'size']]
y = df['tip']

# 공학 후 특성
X_engineered = df[['log_total_bill', 'per_person',
                   'is_dinner', 'is_weekend', 'size']]
```

---

## 성능 비교 결과

```python
# 각각 학습 및 평가
for name, X in [('기본', X_basic), ('공학 후', X_engineered)]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.3f}, R²={r2:.3f}")
```

---

## 비교 결과

| 특성 | RMSE | R² |
|------|------|-----|
| 기본 | 1.02 | 0.43 |
| **공학 후** | **0.98** | **0.47** |

**결론**: 특성 공학으로 성능 향상!

---

## 특성 공학 체크리스트

| 단계 | 체크 항목 |
|------|----------|
| 수치형 | 로그 변환, 스케일링 필요? |
| 비율 | 의미있는 비율 계산 가능? |
| 구간화 | 연속형 → 범주형 변환? |
| 인코딩 | 빈도/타겟 인코딩 적용? |
| 날짜 | 연/월/일/요일/시간 분해? |
| 선택 | 불필요한 특성 제거? |

---

## 핵심 정리

1. **특성 공학**: 모델보다 데이터/특성이 더 중요
2. **수치형 변환**: 로그, 비율, 구간화
3. **범주형 인코딩**: 빈도, 타겟 인코딩 (고급)
4. **날짜 특성**: 연/월/일/시간/주말 추출
5. **특성 선택**: 상관관계, 다중공선성 확인
6. **성능 비교**: 공학 전후 비교 필수

---

## 다음 단계

- **번외 F**: sklearn 패턴 마스터 (Pipeline으로 자동화)
- **12-17차시**: 다양한 모델에 특성 공학 적용

---

# 실습 시간

Tips 데이터셋으로 다양한
특성 공학 기법을 실습해봅시다!

---

# Q&A

질문이 있으신가요?
