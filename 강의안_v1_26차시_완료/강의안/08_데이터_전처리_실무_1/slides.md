---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 8차시'
footer: '© 2026 AI 기초체력훈련'
---

# 데이터 전처리 실무 (1)

## 8차시 | AI 기초체력훈련 (Pre AI-Campus)

**결측치와 이상치 처리**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **결측치**를 탐지하고 적절히 처리한다
2. **이상치**를 탐지하는 다양한 방법을 적용한다
3. 상황에 맞는 **전처리 전략**을 선택한다

---

# 왜 전처리가 중요한가?

## 데이터 과학의 현실

```
전체 프로젝트 시간 배분:

데이터 수집·전처리: 60~80%
모델링: 10~20%
평가·배포: 10~20%
```

> "Garbage In, Garbage Out"
> 나쁜 데이터 → 나쁜 모델

---

# 결측치 (Missing Values)

## 비어있는 값

### 결측치 발생 원인
- 센서 오류 / 통신 두절
- 데이터 입력 누락
- 시스템 장애
- 의도적 미응답

### pandas에서의 표현
```python
np.nan, None, pd.NA
```

---

# 결측치 탐지

## pandas 메서드

```python
# 결측치 확인
df.isnull().sum()       # 열별 결측치 수
df.isnull().sum().sum() # 전체 결측치 수

# 결측치 비율
df.isnull().mean() * 100

# 결측치 있는 행 확인
df[df.isnull().any(axis=1)]

# 결측치 시각화
import missingno as msno
msno.matrix(df)
```

---

# 결측치 처리 전략

## 1. 삭제

```python
# 결측치 있는 행 삭제
df_clean = df.dropna()

# 특정 열에 결측치 있으면 삭제
df_clean = df.dropna(subset=['중요한_열'])

# 결측치가 많은 열 삭제
df_clean = df.drop(columns=['결측_많은_열'])
```

### 언제 사용?
- 결측치가 매우 적을 때 (< 5%)
- 무작위 결측일 때

---

# 결측치 처리 전략

## 2. 대체 (Imputation)

```python
# 평균으로 대체
df['온도'].fillna(df['온도'].mean(), inplace=True)

# 중앙값으로 대체 (이상치에 강건)
df['온도'].fillna(df['온도'].median(), inplace=True)

# 최빈값으로 대체 (범주형)
df['등급'].fillna(df['등급'].mode()[0], inplace=True)

# 앞/뒤 값으로 대체 (시계열)
df['센서값'].fillna(method='ffill', inplace=True)  # 앞 값
df['센서값'].fillna(method='bfill', inplace=True)  # 뒤 값
```

---

# 결측치 처리 전략

## 3. 보간 (Interpolation)

```python
# 선형 보간 (시계열에 유용)
df['온도'] = df['온도'].interpolate(method='linear')

# 다항식 보간
df['온도'] = df['온도'].interpolate(method='polynomial', order=2)
```

---

# 결측치 처리 가이드

## 상황별 전략

| 결측 비율 | 권장 방법 |
|----------|----------|
| < 5% | 삭제 또는 평균/중앙값 대체 |
| 5~20% | 중앙값 대체, 보간, 예측 모델 |
| > 20% | 열 삭제 고려, 결측 자체를 피처로 |

### 핵심 원칙
- 결측 패턴 먼저 파악 (무작위? 체계적?)
- 도메인 지식 활용
- 대체 전후 분포 비교

---

# 이상치 (Outliers)

## 비정상적으로 극단적인 값

### 이상치 발생 원인
- 측정 오류
- 데이터 입력 실수
- 실제 극단적 사건
- 시스템 이상

### 중요 질문
> 이상치가 **오류**인가, **중요한 신호**인가?

---

# 이상치 탐지 방법 1

## IQR (사분위수 범위)

```python
Q1 = df['생산량'].quantile(0.25)
Q3 = df['생산량'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 마스크
outliers = (df['생산량'] < lower) | (df['생산량'] > upper)
```

---

# 이상치 탐지 방법 2

## Z-score

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['생산량']))
outliers = z_scores > 3  # |Z| > 3이면 이상치
```

### 기준
- |Z| > 2: 상위/하위 5% (주의)
- |Z| > 3: 상위/하위 0.3% (이상치)

---

# 이상치 처리 전략

## 옵션들

```python
# 1. 삭제
df_clean = df[~outliers]

# 2. 상/하한 클리핑
df['생산량'] = df['생산량'].clip(lower, upper)

# 3. 대체 (중앙값)
df.loc[outliers, '생산량'] = df['생산량'].median()

# 4. 별도 처리 (플래그)
df['is_outlier'] = outliers
```

---

# 이상치 처리 주의사항

## 신중하게 결정할 것

```
❌ 무조건 제거하지 말 것!

✓ 도메인 전문가와 상의
✓ 원인 파악 먼저
✓ 분석 목적 고려
✓ 전후 비교
```

### 예시
- 불량품 예측 → 이상치가 핵심 정보!
- 평균 생산량 추정 → 이상치 제거 고려

---

# 실습: 제조 데이터 전처리

## 종합 예제

```python
# 1. 데이터 로드
# 2. 결측치 탐지 및 시각화
# 3. 결측치 처리 (전략 선택)
# 4. 이상치 탐지 (IQR, Z-score)
# 5. 이상치 처리 (전략 선택)
# 6. 처리 전후 비교
```

---

# 학습 정리

## 오늘 배운 내용

### 1. 결측치 처리
- 탐지: isnull(), 시각화
- 처리: 삭제, 대체, 보간

### 2. 이상치 처리
- 탐지: IQR, Z-score
- 처리: 삭제, 클리핑, 대체, 플래그

### 3. 핵심 원칙
- 데이터 이해 먼저
- 도메인 지식 활용
- 전후 비교 필수

---

# 다음 차시 예고

## 9차시: 데이터 전처리 실무 (2)

- 스케일링 (정규화, 표준화)
- 범주형 데이터 인코딩
- 피처 엔지니어링 기초

---

# 감사합니다

## AI 기초체력훈련 8차시

**데이터 전처리 실무 (1)**

다음 시간에 만나요!
