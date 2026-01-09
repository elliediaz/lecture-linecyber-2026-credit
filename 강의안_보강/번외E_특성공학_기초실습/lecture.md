# 번외E: 특성 공학 기초 실습

## 학습 목표

| 목표 | 설명 |
|------|------|
| 특성 공학 이해 | 좋은 특성이 좋은 모델을 만드는 원리 |
| 수치형 변환 | 로그 변환, 비율 계산, 구간화 |
| 범주형 인코딩 | 빈도 인코딩, 타겟 인코딩 |
| 특성 선택 | 상관관계 기반 선택 |
| 성능 비교 | 특성 공학 전후 비교 |

---

## 핵심 개념: 좋은 특성이 좋은 모델을 만듦

```
복잡한 모델 < 좋은 특성 + 단순한 모델
```

특성 공학(Feature Engineering)은 원본 데이터를 모델이 더 잘 학습할 수 있는 형태로 변환하는 과정임

---

## 좋은 특성의 3가지 조건

| 조건 | 설명 | 예시 |
|------|------|------|
| 정보성 | 타겟과 관련 있어야 함 | 팁 금액 ↔ 총 결제 금액 |
| 독립성 | 다른 특성과 중복 없어야 함 | total_bill과 log_total_bill 중 하나만 |
| 이해 가능 | 해석 가능해야 함 | tip_ratio = 팁 비율 |

---

## 실습 환경 설정

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("번외E: 특성 공학 기초 실습")
print("=" * 60)
```

---

# Part 1: 데이터 로드 및 확인

## Tips 데이터셋 로드

```python
# Tips 데이터셋 로드
df = sns.load_dataset('tips')

print(f"[데이터 크기]")
print(f"행: {df.shape[0]}, 열: {df.shape[1]}")
```

**출력**: 행: 244, 열: 7

---

## 데이터 구조 확인

```python
print(df.dtypes)
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| total_bill | float64 | 총 결제 금액 |
| tip | float64 | 팁 금액 (타겟) |
| sex | category | 성별 |
| smoker | category | 흡연 여부 |
| day | category | 요일 |
| time | category | 시간대 |
| size | int64 | 인원수 |

---

## 데이터 미리보기

```python
print(df.head())
```

| total_bill | tip | sex | smoker | day | time | size |
|------------|-----|-----|--------|-----|------|------|
| 16.99 | 1.01 | Female | No | Sun | Dinner | 2 |
| 10.34 | 1.66 | Male | No | Sun | Dinner | 3 |
| 21.01 | 3.50 | Male | No | Sun | Dinner | 3 |
| 23.68 | 3.31 | Male | No | Sun | Dinner | 2 |
| 24.59 | 3.61 | Female | No | Sun | Dinner | 4 |

---

## 기초 통계량

```python
print(df.describe())
```

| 항목 | total_bill | tip | size |
|------|------------|-----|------|
| mean | 19.79 | 3.00 | 2.57 |
| std | 8.90 | 1.38 | 0.95 |
| min | 3.07 | 1.00 | 1 |
| max | 50.81 | 10.00 | 6 |

---

# Part 2: 수치형 특성 변환

## 작업용 복사본 생성

```python
df_eng = df.copy()
```

데이터 전처리 시 원본 보존을 위해 항상 복사본을 사용함

---

## 2-1. 로그 변환

### 왜 로그 변환을 사용하는가?

금액 데이터는 보통 오른쪽으로 치우친 분포를 가짐. 로그 변환으로 정규분포에 가깝게 만들 수 있음

```python
print(f"변환 전 왜도(skewness): {df_eng['total_bill'].skew():.3f}")

# log1p = log(x + 1) → 0 처리 안전
df_eng['log_total_bill'] = np.log1p(df_eng['total_bill'])

print(f"변환 후 왜도(skewness): {df_eng['log_total_bill'].skew():.3f}")
```

| 상태 | 왜도 | 해석 |
|------|------|------|
| 변환 전 | 1.126 | 오른쪽 치우침 |
| 변환 후 | 0.092 | 정규분포에 가까움 |

왜도가 0에 가까울수록 정규분포에 가까움

---

## log1p를 사용하는 이유

```python
# log(0) = -inf (무한대)
# log1p(0) = log(1) = 0 → 안전!
```

| 함수 | 계산 | 장점 |
|------|------|------|
| np.log(x) | log(x) | - |
| np.log1p(x) | log(x + 1) | 0 값 처리 안전 |
| np.expm1(x) | exp(x) - 1 | log1p의 역변환 |

---

## 2-2. 비율 특성 생성

### 팁 비율 계산

```python
df_eng['tip_ratio'] = df_eng['tip'] / df_eng['total_bill']

print(f"팁 비율 평균: {df_eng['tip_ratio'].mean():.1%}")
print(f"팁 비율 범위: {df_eng['tip_ratio'].min():.1%} ~ {df_eng['tip_ratio'].max():.1%}")
```

| 항목 | 값 |
|------|-----|
| 평균 | 16.1% |
| 최소 | 3.6% |
| 최대 | 71.0% |

팁 금액 자체보다 **팁 비율**이 더 의미있는 특성이 될 수 있음

---

### 1인당 금액 계산

```python
df_eng['per_person'] = df_eng['total_bill'] / df_eng['size']

print(f"1인당 금액 평균: ${df_eng['per_person'].mean():.2f}")
```

**출력**: 1인당 금액 평균: $8.04

1인당 금액은 인원수의 영향을 제거한 특성임

---

## 2-3. 구간화 (Binning)

### 등간격 구간화: pd.cut

```python
df_eng['bill_cut'] = pd.cut(
    df_eng['total_bill'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)

print("등간격(cut) 분포:")
print(df_eng['bill_cut'].value_counts())
```

| 구간 | 개수 |
|------|------|
| Low | 168 |
| Medium | 64 |
| High | 12 |

등간격: 값의 범위를 균등하게 나눔

---

### 등빈도 구간화: pd.qcut

```python
df_eng['bill_qcut'] = pd.qcut(
    df_eng['total_bill'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

print("등빈도(qcut) 분포:")
print(df_eng['bill_qcut'].value_counts())
```

| 구간 | 개수 |
|------|------|
| Low | 82 |
| Medium | 81 |
| High | 81 |

등빈도: 각 구간에 데이터가 균등하게 배분됨

---

## cut vs qcut 비교

| 방법 | 기준 | 결과 | 사용 시점 |
|------|------|------|----------|
| pd.cut | 값 범위 균등 | 불균등 분포 | 의미있는 경계가 있을 때 |
| pd.qcut | 데이터 수 균등 | 균등 분포 | 순위/등급 나눌 때 |

---

# Part 3: 범주형 특성 인코딩

## 3-1. 빈도 인코딩 (Frequency Encoding)

### 요일별 빈도 계산

```python
day_freq = df_eng['day'].value_counts(normalize=True)
print("요일별 빈도:")
print(day_freq)
```

| 요일 | 빈도 |
|------|------|
| Sat | 0.36 |
| Sun | 0.31 |
| Thur | 0.26 |
| Fri | 0.08 |

---

### 빈도로 인코딩

```python
df_eng['day_freq'] = df_eng['day'].map(day_freq)

print("인코딩 결과 (샘플):")
print(df_eng[['day', 'day_freq']].head(10))
```

| day | day_freq |
|-----|----------|
| Sun | 0.311 |
| Sun | 0.311 |
| Sat | 0.358 |

빈도 인코딩: 범주를 해당 범주의 출현 빈도로 대체함

---

## 3-2. 타겟 인코딩 (Target Encoding)

### 요일별 평균 팁 계산

```python
day_target = df_eng.groupby('day')['tip'].mean()
print("요일별 평균 팁:")
print(day_target)
```

| 요일 | 평균 팁 |
|------|---------|
| Fri | 2.73 |
| Sat | 2.99 |
| Sun | 3.26 |
| Thur | 2.77 |

---

### 타겟 인코딩 적용

```python
df_eng['day_target'] = df_eng['day'].map(day_target)

print("인코딩 결과 (샘플):")
print(df_eng[['day', 'day_target']].head(10))
```

| day | day_target |
|-----|------------|
| Sun | 3.26 |
| Sun | 3.26 |
| Sat | 2.99 |

타겟 인코딩: 범주를 해당 범주의 타겟 평균으로 대체함

---

## 타겟 인코딩 주의사항

```
⚠️ 데이터 누출(Data Leakage) 주의!
```

| 상황 | 결과 |
|------|------|
| 전체 데이터로 평균 계산 | 테스트 정보가 학습에 포함됨 (잘못됨) |
| 학습 데이터로만 평균 계산 | 올바른 방법 |

타겟 인코딩 시 반드시 **학습 데이터로만** 평균을 계산해야 함

---

## 3-3. One-Hot Encoding (참고)

```python
df_onehot = pd.get_dummies(df_eng[['day', 'time']], drop_first=True)
print(df_onehot.head())
```

| day_Sat | day_Sun | day_Thur | time_Lunch |
|---------|---------|----------|------------|
| 0 | 1 | 0 | 0 |
| 0 | 1 | 0 | 0 |
| 1 | 0 | 0 | 0 |

One-Hot: 범주가 적을 때 사용, 범주가 많으면 빈도/타겟 인코딩 권장

---

# Part 4: 날짜/시간 특성

## 시간 관련 특성 생성

Tips 데이터에는 datetime이 없으므로 기존 컬럼을 활용함

```python
# 저녁 식사 여부
df_eng['is_dinner'] = (df_eng['time'] == 'Dinner').astype(int)
print(f"저녁 식사 비율: {df_eng['is_dinner'].mean():.1%}")

# 주말 여부
df_eng['is_weekend'] = df_eng['day'].isin(['Sat', 'Sun']).astype(int)
print(f"주말 비율: {df_eng['is_weekend'].mean():.1%}")
```

| 특성 | 비율 |
|------|------|
| 저녁 식사 | 72.5% |
| 주말 | 66.8% |

---

## 생성된 특성 확인

```python
print(df_eng[['day', 'time', 'is_dinner', 'is_weekend']].head(10))
```

| day | time | is_dinner | is_weekend |
|-----|------|-----------|------------|
| Sun | Dinner | 1 | 1 |
| Sun | Dinner | 1 | 1 |
| Sat | Dinner | 1 | 1 |
| Sat | Lunch | 0 | 1 |
| Thur | Lunch | 0 | 0 |

이진 특성으로 변환하면 모델이 더 쉽게 패턴을 학습함

---

## 날짜 특성 추출 (datetime이 있을 경우)

```python
# datetime 컬럼이 있다면:
# df['year'] = df['datetime'].dt.year
# df['month'] = df['datetime'].dt.month
# df['day'] = df['datetime'].dt.day
# df['dayofweek'] = df['datetime'].dt.dayofweek
# df['hour'] = df['datetime'].dt.hour
# df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
```

datetime에서 다양한 특성을 추출할 수 있음

---

# Part 5: 특성 선택

## 5-1. 타겟과의 상관관계 분석

```python
# 수치형 특성 선택
numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
print(f"수치형 특성: {numeric_cols}")

# 상관관계 계산
correlations = df_eng[numeric_cols].corr()['tip'].abs()
correlations = correlations.sort_values(ascending=False)

print("\n타겟(tip)과의 상관계수:")
for col, corr in correlations.items():
    bar = "█" * int(corr * 30)
    print(f"{col:18} {corr:.3f} {bar}")
```

---

## 상관관계 결과

| 특성 | 상관계수 | 해석 |
|------|----------|------|
| tip | 1.000 | 자기 자신 |
| total_bill | 0.676 | 강한 양의 상관 |
| log_total_bill | 0.652 | 강한 양의 상관 |
| per_person | 0.550 | 중간 양의 상관 |
| size | 0.489 | 중간 양의 상관 |
| day_target | 0.067 | 약한 상관 |

total_bill과 log_total_bill 모두 높은 상관관계를 보임

---

## 5-2. 상관관계 기반 특성 선택

```python
threshold = 0.3
selected_features = correlations[correlations > threshold].index.tolist()
selected_features.remove('tip')  # 타겟 제외

print(f"임계값: {threshold}")
print(f"선택된 특성: {selected_features}")
```

**출력**: 선택된 특성: ['total_bill', 'log_total_bill', 'per_person', 'size']

상관계수 0.3 이상인 특성만 선택함

---

## 5-3. 다중공선성 확인

```python
feature_corr = df_eng[selected_features].corr()
print(feature_corr.round(2))
```

| | total_bill | log_total_bill | per_person | size |
|-|------------|----------------|------------|------|
| total_bill | 1.00 | **0.99** | 0.72 | 0.60 |
| log_total_bill | **0.99** | 1.00 | 0.63 | 0.54 |
| per_person | 0.72 | 0.63 | 1.00 | -0.02 |
| size | 0.60 | 0.54 | -0.02 | 1.00 |

total_bill과 log_total_bill의 상관관계가 0.99로 매우 높음 → **하나만 사용 권장**

---

# Part 6: 성능 비교

## 타겟 설정

```python
y = df_eng['tip']
```

---

## 6-1. 기본 특성으로 학습

```python
X_basic = df_eng[['total_bill', 'size']]
print(f"특성: {list(X_basic.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

model_basic = LinearRegression()
model_basic.fit(X_train, y_train)
y_pred_basic = model_basic.predict(X_test)

rmse_basic = np.sqrt(mean_squared_error(y_test, y_pred_basic))
r2_basic = r2_score(y_test, y_pred_basic)

print(f"RMSE: {rmse_basic:.3f}")
print(f"R²: {r2_basic:.3f}")
```

| 지표 | 값 |
|------|-----|
| RMSE | 1.02 |
| R² | 0.46 |

---

## 6-2. 공학 후 특성으로 학습

```python
X_engineered = df_eng[['log_total_bill', 'per_person', 'size',
                       'is_dinner', 'is_weekend', 'day_freq']]
print(f"특성: {list(X_engineered.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

model_eng = LinearRegression()
model_eng.fit(X_train, y_train)
y_pred_eng = model_eng.predict(X_test)

rmse_eng = np.sqrt(mean_squared_error(y_test, y_pred_eng))
r2_eng = r2_score(y_test, y_pred_eng)

print(f"RMSE: {rmse_eng:.3f}")
print(f"R²: {r2_eng:.3f}")
```

| 지표 | 값 |
|------|-----|
| RMSE | 0.98 |
| R² | 0.50 |

---

## 6-3. 성능 비교

```python
print(f"{'특성':15} {'RMSE':>10} {'R²':>10}")
print("-" * 40)
print(f"{'기본':15} {rmse_basic:>10.3f} {r2_basic:>10.3f}")
print(f"{'공학 후':15} {rmse_eng:>10.3f} {r2_eng:>10.3f}")

improvement = (rmse_basic - rmse_eng) / rmse_basic * 100
print(f"\nRMSE 개선율: {improvement:.1f}%")
```

| 특성 | RMSE | R² |
|------|------|-----|
| 기본 | 1.020 | 0.458 |
| **공학 후** | **0.982** | **0.498** |

**RMSE 개선율: 3.7%** - 특성 공학으로 성능이 향상됨

---

# 시각화

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 로그 변환 효과
ax1 = axes[0, 0]
ax1.hist(df_eng['total_bill'], bins=30, alpha=0.5, label='Original', color='blue')
ax1.hist(df_eng['log_total_bill'] * 10, bins=30, alpha=0.5, label='Log (scaled)', color='orange')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Log Transformation Effect')
ax1.legend()

# 2. 상관관계 히트맵
ax2 = axes[0, 1]
corr_subset = df_eng[['tip', 'total_bill', 'log_total_bill', 'per_person', 'size']].corr()
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, center=0)
ax2.set_title('Correlation Heatmap')

# 3. 요일별 팁 분포
ax3 = axes[1, 0]
df_eng.boxplot(column='tip', by='day', ax=ax3)
ax3.set_xlabel('Day')
ax3.set_ylabel('Tip')
ax3.set_title('Tip Distribution by Day')
plt.suptitle('')

# 4. 성능 비교
ax4 = axes[1, 1]
models = ['Basic', 'Engineered']
rmse_values = [rmse_basic, rmse_eng]
r2_values = [r2_basic, r2_eng]

x = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x - width/2, rmse_values, width, label='RMSE', color='#e74c3c')
bars2 = ax4.bar(x + width/2, r2_values, width, label='R²', color='#3498db')

ax4.set_ylabel('Score')
ax4.set_title('Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()

plt.tight_layout()
plt.savefig('feature_engineering_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

# 생성된 특성 요약

## 수치형 변환

| 특성 | 계산 | 의미 |
|------|------|------|
| log_total_bill | log1p(total_bill) | 로그 변환된 금액 |
| tip_ratio | tip / total_bill | 팁 비율 |
| per_person | total_bill / size | 1인당 금액 |

---

## 구간화

| 특성 | 방법 | 의미 |
|------|------|------|
| bill_cut | pd.cut (등간격) | 금액 등급 (값 기준) |
| bill_qcut | pd.qcut (등빈도) | 금액 등급 (순위 기준) |

---

## 범주형 인코딩

| 특성 | 방법 | 의미 |
|------|------|------|
| day_freq | 빈도 인코딩 | 요일의 출현 빈도 |
| day_target | 타겟 인코딩 | 요일별 평균 팁 |

---

## 시간 특성

| 특성 | 조건 | 의미 |
|------|------|------|
| is_dinner | time == 'Dinner' | 저녁 식사 여부 |
| is_weekend | day in ['Sat', 'Sun'] | 주말 여부 |

---

# 특성 공학 체크리스트

## 수치형 특성

| 체크 | 항목 |
|------|------|
| □ | 분포 확인 (왜도) → 로그 변환 필요? |
| □ | 스케일링 필요? (StandardScaler, MinMaxScaler) |
| □ | 구간화가 의미있을까? |

---

## 비율/조합 특성

| 체크 | 항목 |
|------|------|
| □ | 의미있는 비율 계산 가능? |
| □ | 특성 간 상호작용 특성? |

---

## 범주형 특성

| 체크 | 항목 |
|------|------|
| □ | 범주 개수 확인 → One-Hot vs 빈도/타겟 인코딩 |
| □ | 순서 있는 범주? → Label Encoding |

---

## 날짜/시간

| 체크 | 항목 |
|------|------|
| □ | 연/월/일/요일/시간 분해 |
| □ | 주말/공휴일 여부 |
| □ | 주기적 특성? → 삼각함수 변환 |

---

## 특성 선택

| 체크 | 항목 |
|------|------|
| □ | 타겟과의 상관관계 확인 |
| □ | 특성 간 다중공선성 확인 |
| □ | 분산 낮은 특성 제거? |

---

# 핵심 정리

1. **특성 공학의 중요성**: 좋은 특성이 좋은 모델을 만듦
2. **수치형 변환**: 로그, 비율, 구간화로 분포 개선
3. **범주형 인코딩**: 빈도/타겟 인코딩으로 고차원 방지
4. **특성 선택**: 상관관계로 불필요한 특성 제거
5. **다중공선성**: 상관 높은 특성은 하나만 사용
6. **성능 검증**: 특성 공학 전후 비교 필수

---

## 다음 학습

- **번외 F**: sklearn 핵심 패턴 마스터 (Pipeline, ColumnTransformer)
- **12차시~**: 머신러닝 모델 심화 학습
