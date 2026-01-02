---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 7차시'
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

# 제조 데이터 전처리 (1)

## 7차시 | Part II. 기초 수리와 데이터 분석

**결측치와 이상치 처리**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **결측치**를 탐지하고 적절히 처리한다
2. **이상치**를 탐지하는 방법을 적용한다
3. 상황에 맞는 **전처리 전략**을 선택한다

---

# 왜 전처리가 중요한가?

## 데이터 분석의 현실

```
전체 프로젝트 시간 배분:

데이터 수집·전처리: 60~80%
모델링: 10~20%
평가·배포: 10~20%
```

> **"Garbage In, Garbage Out"**
> 나쁜 데이터 → 나쁜 모델

---

# 결측치 (Missing Values)

## 비어있는 값

### 제조 현장에서 발생 원인
- **센서 오류**: 통신 두절, 배터리 방전
- **입력 누락**: 수기 입력 시 실수
- **시스템 장애**: PLC, MES 연결 끊김
- **정상 결측**: 특정 조건에서만 측정

```python
# pandas에서의 표현
np.nan, None, pd.NA
```

---

# 결측치 탐지

## pandas 메서드

```python
# 열별 결측치 수
df.isnull().sum()

# 결측치 비율 (%)
(df.isnull().sum() / len(df) * 100).round(2)

# 결측치 있는 행 확인
df[df.isnull().any(axis=1)]

# 전체 정보 확인
df.info()
```

---

# 결측치 처리 전략 1: 삭제

## 결측치가 적을 때

```python
# 결측치 있는 행 삭제
df_clean = df.dropna()

# 특정 열에 결측치 있으면 삭제
df_clean = df.dropna(subset=['생산량', '온도'])

# 모든 값이 결측인 행만 삭제
df_clean = df.dropna(how='all')
```

### 사용 시점
- 결측치 비율 < 5%
- 무작위 결측일 때

---

# 결측치 처리 전략 2: 대체

## Imputation

```python
# 평균으로 대체
df['온도'].fillna(df['온도'].mean(), inplace=True)

# 중앙값으로 대체 (이상치에 강건)
df['온도'].fillna(df['온도'].median(), inplace=True)

# 최빈값으로 대체 (범주형)
df['등급'].fillna(df['등급'].mode()[0], inplace=True)

# 앞/뒤 값으로 대체 (시계열)
df['센서값'] = df['센서값'].ffill()  # 앞 값
df['센서값'] = df['센서값'].bfill()  # 뒤 값
```

---

# 결측치 처리 가이드

## 비율별 권장 전략

| 결측 비율 | 권장 방법 |
|----------|----------|
| < 5% | 삭제 또는 평균/중앙값 대체 |
| 5 ~ 20% | 중앙값 대체, 보간 |
| > 20% | 열 삭제 고려 |

### 핵심 원칙
- 결측 패턴 먼저 파악 (무작위? 체계적?)
- 도메인 지식 활용
- 대체 전후 분포 비교

---

# 이상치 (Outliers)

## 비정상적으로 극단적인 값

### 제조 현장에서 발생 원인
- **측정 오류**: 센서 고장, 캘리브레이션 불량
- **입력 실수**: 단위 오류, 오타
- **실제 극단값**: 공정 이탈, 설비 고장
- **시스템 이상**: 리셋 값, 오버플로우

> 중요 질문: 이상치가 **오류**인가, **중요한 신호**인가?

---

# 이상치 탐지 방법 1: IQR

## 사분위수 범위

```python
Q1 = df['생산량'].quantile(0.25)  # 25%
Q3 = df['생산량'].quantile(0.75)  # 75%
IQR = Q3 - Q1

# 경계값 계산
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 탐지
outliers = (df['생산량'] < lower) | (df['생산량'] > upper)
print(f"이상치 개수: {outliers.sum()}")
```

> 상자그림(Boxplot)과 동일한 기준

---

# 이상치 탐지 방법 2: Z-score

## 표준점수

```python
mean = df['생산량'].mean()
std = df['생산량'].std()

# Z-score 계산
z_scores = (df['생산량'] - mean) / std

# 이상치 탐지 (|Z| > 3)
outliers = np.abs(z_scores) > 3
print(f"이상치 개수: {outliers.sum()}")
```

| Z-score | 해석 |
|---------|------|
| \|Z\| > 2 | 상위/하위 5% (주의) |
| \|Z\| > 3 | 상위/하위 0.3% (이상치) |

---

# 이상치 처리 전략

## 상황별 선택

```python
# 1. 삭제
df_clean = df[~outliers]

# 2. 상/하한 클리핑
df['생산량'] = df['생산량'].clip(lower, upper)

# 3. 대체 (중앙값)
median = df['생산량'].median()
df.loc[outliers, '생산량'] = median

# 4. 별도 플래그
df['이상치_여부'] = outliers
```

---

# 이상치 처리 주의사항

## 무조건 제거하지 말 것!

```
분석 목적에 따라 다름:

불량품 예측 → 이상치가 핵심 정보!
평균 생산량 추정 → 이상치 제거 고려
이상 탐지 시스템 → 이상치를 찾는 게 목적
```

### 체크리스트
- 도메인 전문가와 상의
- 원인 파악 먼저
- 전후 분포 비교

---

# 이론 정리

## 핵심 포인트

### 결측치 처리
- **탐지**: isnull(), info()
- **처리**: 삭제(dropna), 대체(fillna)

### 이상치 처리
- **탐지**: IQR, Z-score
- **처리**: 삭제, 클리핑, 대체, 플래그

### 핵심 원칙
- 데이터 이해 먼저
- 도메인 지식 활용
- 전후 비교 필수

---

# - 실습편 -

## 7차시

**결측치와 이상치 처리 실습**

---

# 실습 개요

## 제조 데이터 전처리

### 실습 목표
- 결측치 탐지 및 처리
- 이상치 탐지 및 처리
- 전처리 전후 비교

### 실습 환경
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

# 실습 1: 결측치 있는 데이터 생성

## 샘플 데이터

```python
np.random.seed(42)
n = 100

df = pd.DataFrame({
    '일자': pd.date_range('2024-01-01', periods=n),
    '온도': np.random.normal(85, 5, n),
    '생산량': np.random.normal(1200, 50, n),
    '불량수': np.random.poisson(30, n),
    '라인': np.random.choice(['A', 'B', 'C'], n)
})

# 결측치 삽입 (10%)
missing_idx = np.random.choice(n, 10, replace=False)
df.loc[missing_idx, '온도'] = np.nan

missing_idx2 = np.random.choice(n, 5, replace=False)
df.loc[missing_idx2, '생산량'] = np.nan

print(df.head(10))
```

---

# 실습 2: 결측치 탐지

## 탐색

```python
# 열별 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())
print()

# 결측 비율
missing_ratio = (df.isnull().sum() / len(df) * 100).round(2)
print("=== 결측 비율 (%) ===")
print(missing_ratio)

# 결측치 시각화
plt.figure(figsize=(10, 4))
df.isnull().sum().plot(kind='bar', color='coral')
plt.ylabel('결측치 수')
plt.title('열별 결측치 현황')
plt.show()
```

---

# 실습 3: 결측치 처리

## 전략 적용

```python
# 원본 보존
df_clean = df.copy()

# 온도: 중앙값으로 대체
temp_median = df_clean['온도'].median()
df_clean['온도'].fillna(temp_median, inplace=True)
print(f"온도 결측치를 {temp_median:.1f}도로 대체")

# 생산량: 평균으로 대체
prod_mean = df_clean['생산량'].mean()
df_clean['생산량'].fillna(prod_mean, inplace=True)
print(f"생산량 결측치를 {prod_mean:.0f}개로 대체")

# 결과 확인
print(f"\n처리 후 결측치: {df_clean.isnull().sum().sum()}개")
```

---

# 실습 4: 이상치 삽입

## 테스트용 이상치

```python
# 이상치 삽입
df_clean.loc[5, '생산량'] = 2000   # 매우 높음
df_clean.loc[15, '생산량'] = 500   # 매우 낮음
df_clean.loc[25, '온도'] = 120     # 비정상 온도

print("이상치가 삽입된 행:")
print(df_clean.loc[[5, 15, 25], ['온도', '생산량']])
```

---

# 실습 5: IQR로 이상치 탐지

## 생산량 이상치

```python
Q1 = df_clean['생산량'].quantile(0.25)
Q3 = df_clean['생산량'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = (df_clean['생산량'] < lower) | (df_clean['생산량'] > upper)

print("=== IQR 방법 ===")
print(f"Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
print(f"하한: {lower:.0f}, 상한: {upper:.0f}")
print(f"이상치 개수: {outliers_iqr.sum()}개")
print(f"이상치 인덱스: {df_clean[outliers_iqr].index.tolist()}")
```

---

# 실습 6: Z-score로 이상치 탐지

## 온도 이상치

```python
mean = df_clean['온도'].mean()
std = df_clean['온도'].std()
z_scores = (df_clean['온도'] - mean) / std

outliers_z = np.abs(z_scores) > 2

print("=== Z-score 방법 ===")
print(f"평균: {mean:.1f}, 표준편차: {std:.1f}")
print(f"이상치 개수 (|Z|>2): {outliers_z.sum()}개")

# 이상치 상세
print("\n이상치 상세:")
outlier_df = df_clean[outliers_z][['온도']].copy()
outlier_df['Z-score'] = z_scores[outliers_z]
print(outlier_df)
```

---

# 실습 7: 이상치 시각화

## 상자그림

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 생산량 상자그림
axes[0].boxplot(df_clean['생산량'])
axes[0].set_ylabel('생산량')
axes[0].set_title('생산량 분포 (이상치 포함)')

# 온도 상자그림
axes[1].boxplot(df_clean['온도'])
axes[1].set_ylabel('온도')
axes[1].set_title('온도 분포 (이상치 포함)')

plt.tight_layout()
plt.show()
```

---

# 실습 8: 이상치 처리

## 클리핑 적용

```python
# 생산량: 클리핑
df_final = df_clean.copy()
df_final['생산량'] = df_final['생산량'].clip(lower, upper)

# 온도: Z-score 기준 이상치를 중앙값으로 대체
temp_median = df_final['온도'].median()
df_final.loc[outliers_z, '온도'] = temp_median

print("=== 처리 결과 ===")
print(f"생산량 범위: {df_final['생산량'].min():.0f} ~ {df_final['생산량'].max():.0f}")
print(f"온도 범위: {df_final['온도'].min():.1f} ~ {df_final['온도'].max():.1f}")
```

---

# 실습 9: 전후 비교

## 분포 변화 확인

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 처리 전 생산량
axes[0, 0].hist(df_clean['생산량'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('처리 전 생산량')

# 처리 후 생산량
axes[0, 1].hist(df_final['생산량'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('처리 후 생산량')

# 처리 전 온도
axes[1, 0].hist(df_clean['온도'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('처리 전 온도')

# 처리 후 온도
axes[1, 1].hist(df_final['온도'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('처리 후 온도')

plt.tight_layout()
plt.show()
```

---

# 실습 10: 전처리 요약

## 종합 리포트

```python
print("=" * 50)
print("         데이터 전처리 요약 리포트")
print("=" * 50)

print(f"\n[원본 데이터]")
print(f"행 수: {len(df)}")
print(f"결측치: 온도 {df['온도'].isnull().sum()}개, 생산량 {df['생산량'].isnull().sum()}개")

print(f"\n[처리 내용]")
print(f"- 온도 결측치: 중앙값({temp_median:.1f})으로 대체")
print(f"- 생산량 결측치: 평균({prod_mean:.0f})으로 대체")
print(f"- 생산량 이상치: 클리핑({lower:.0f}~{upper:.0f})")
print(f"- 온도 이상치: 중앙값으로 대체")

print(f"\n[최종 데이터]")
print(f"결측치: {df_final.isnull().sum().sum()}개")
print("=" * 50)
```

---

# 실습 정리

## 핵심 체크포인트

### 결측치
- [ ] isnull().sum()으로 탐지
- [ ] fillna()로 대체
- [ ] 비율에 따라 전략 선택

### 이상치
- [ ] IQR 또는 Z-score로 탐지
- [ ] clip()으로 클리핑
- [ ] 도메인에 따라 처리 결정

---

# 다음 차시 예고

## 8차시: 제조 데이터 전처리 (2)

### 학습 내용
- 데이터 정규화와 표준화
- 범주형 데이터 인코딩
- 피처 엔지니어링 기초

### 준비물
- 오늘 배운 코드 복습
- sklearn 설치 확인

---

# 정리 및 Q&A

## 오늘의 핵심

1. **결측치**: 탐지(isnull) → 처리(dropna/fillna)
2. **이상치**: 탐지(IQR/Z-score) → 처리(clip/대체)
3. **원칙**: 데이터 이해 → 도메인 지식 → 전후 비교

### 자주 하는 실수
- 결측치를 0으로 채우는 것 (의미 왜곡)
- 이상치를 무조건 제거하는 것

---

# 감사합니다

## 7차시: 제조 데이터 전처리 (1)

**다음 시간에 스케일링과 인코딩을 배워봅시다!**
