# 23차시: 모델 해석과 변수별 영향력 분석

## 학습 목표

1. **모델 해석**의 필요성과 종류를 이해함
2. **Feature Importance**를 계산하고 해석함
3. **Permutation Importance**를 활용함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | 모델 해석의 필요성 |
| 대주제 2 | 10분 | Feature Importance |
| 대주제 3 | 8분 | Permutation Importance |
| 정리 | 2분 | 핵심 요약 |

---

## 지난 시간 복습

- **CNN**: 이미지 처리, 합성곱, 풀링
- **RNN/LSTM**: 시계열 처리, 장기 의존성
- **고급 아키텍처**: ResNet, Transformer

**오늘**: 모델이 왜 그런 예측을 했는지 이해하기

---

# 대주제 1: 모델 해석의 필요성

## 1.1 왜 모델 해석이 필요한가?

**블랙박스 문제**:
```
입력 [온도, 압력, 속도] -> 모델 -> 예측: 불량
                          ?
```

- 왜 불량이라고 예측했는가?
- 어떤 변수가 중요했는가?
- 예측을 신뢰할 수 있는가?

---

## 1.2 모델 해석이 중요한 이유

| 관점 | 이유 |
|-----|------|
| **신뢰** | 예측 근거를 설명해야 의사결정 가능 |
| **디버깅** | 잘못된 패턴 학습 발견 |
| **개선** | 중요 변수 파악 -> 데이터 수집 방향 |
| **규제** | EU AI Act 등 설명 의무화 추세 |

---

## 1.3 제조업에서의 예시

**상황**: 품질 예측 모델이 "불량"이라고 예측

**질문**:
- 온도 때문인가? 압력 때문인가?
- 어떤 조건을 바꾸면 정상이 되는가?
- 이 예측을 믿어도 되는가?

**모델 해석이 답을 줌**

---

## 1.4 해석 가능한 모델 vs 복잡한 모델

| 모델 | 해석 가능성 | 성능 |
|-----|-----------|------|
| 선형 회귀 | 높음 (계수) | 낮음 |
| 의사결정나무 | 높음 (규칙) | 중간 |
| RandomForest | 중간 (중요도) | 높음 |
| 신경망 | 낮음 (블랙박스) | 높음 |

**해석 방법으로 복잡한 모델도 이해 가능**

---

## 1.5 모델 해석 방법 분류

| 분류 | 방법 | 특징 |
|-----|------|------|
| **모델 내장** | Feature Importance | 트리 모델 전용 |
| **모델 무관** | Permutation Importance | 모든 모델 가능 |
| **인스턴스별** | SHAP, LIME | 개별 예측 설명 |

---

## 1.6 전역 vs 지역 해석

**전역 해석 (Global)**:
- 모델 전체에서 변수 중요도
- "온도가 가장 중요한 변수"

**지역 해석 (Local)**:
- 특정 예측에 대한 설명
- "이 샘플은 압력이 높아서 불량"

---

## 실습 코드: 데이터셋 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("[23차시] 모델 해석과 변수별 영향력 분석")
print("=" * 60)

# sklearn의 Wine 데이터셋 로드
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("\n[Wine 데이터셋 개요]")
print(f"  샘플 수: {len(X)}")
print(f"  특성 수: {X.shape[1]}")
print(f"  클래스: {list(wine.target_names)}")
print(f"  클래스 분포: {np.bincount(y)}")

print("\n[특성 목록]")
for i, name in enumerate(wine.feature_names):
    print(f"  {i+1:2d}. {name}")
```

---

# 대주제 2: Feature Importance

## 2.1 Feature Importance란?

**정의**: 모델이 예측할 때 각 변수가 얼마나 기여했는가

**트리 모델에서**:
- 변수로 분할할 때 불순도 감소량
- 많이 사용되고, 불순도를 많이 줄이면 중요

---

## 2.2 RandomForest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature Importance 추출
importances = model.feature_importances_
```

`feature_importances_`: 각 변수의 중요도 (합 = 1)

---

## 실습 코드: 데이터 분할 및 모델 학습

```python
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomForest 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 성능 확인
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n[모델 성능]")
print(f"  학습 정확도: {train_acc:.2%}")
print(f"  테스트 정확도: {test_acc:.2%}")

# 분류 리포트
print("\n[분류 리포트]")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

---

## 2.3 Feature Importance 추출

```python
# Feature Importance 추출
feature_importance = model.feature_importances_
feature_names = wine.feature_names

# 정렬
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n[Feature Importance]")
print("-" * 50)
for _, row in fi_df.iterrows():
    bar = '*' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

print("\n[해석]")
top_features = fi_df.head(3)['Feature'].tolist()
print(f"  - 가장 중요한 특성: {top_features[0]}")
print(f"  - 상위 3개 특성이 전체 중요도의 "
      f"{fi_df.head(3)['Importance'].sum():.1%} 차지")
```

---

## 2.4 Feature Importance 시각화

```python
fig, ax = plt.subplots(figsize=(12, 8))

# 수평 막대 그래프
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors[::-1])

ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance - Wine Classification (RandomForest)', fontsize=14)
ax.invert_yaxis()  # 높은 순으로 위에서 아래로

# 값 표시
for i, (feature, importance) in enumerate(zip(fi_df['Feature'], fi_df['Importance'])):
    ax.text(importance + 0.005, i, f'{importance:.3f}', va='center')

plt.tight_layout()
plt.savefig('feature_importance_wine.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2.5 Feature Importance 해석

```
온도      ********************  0.35
압력      **************        0.25
진동      **********            0.18
속도      ********              0.12
습도      ******                0.10
```

**해석**:
- 온도가 가장 중요 (35%)
- 온도, 압력만으로 60% 설명
- 습도는 상대적으로 덜 중요

---

## 2.6 제조업 인사이트

**Feature Importance 결과**:
- 온도 > 압력 > 진동 > 속도 > 습도

**조치**:
1. 온도 모니터링 강화
2. 압력 센서 정밀도 개선
3. 습도 센서는 비용 대비 효과 낮음

---

## 2.7 Feature Importance 주의점

**한계**:
1. **상관 변수 문제**: 상관된 변수 간 중요도 분산
2. **스케일 영향**: 일부 모델에서 스케일 영향
3. **트리 모델 전용**: 다른 모델에는 적용 어려움

**해결**: Permutation Importance 사용

---

## 2.8 상관 변수 문제 예시

```
온도와 습도가 높은 상관 (r = 0.9)

Feature Importance:
온도: 0.25
습도: 0.20  <- 합쳐서 0.45인데 분산됨

실제로는 "온도" 하나로도 충분한 정보
```

---

# 대주제 3: Permutation Importance

## 3.1 Permutation Importance란?

**아이디어**:
변수 값을 무작위로 섞으면 예측 성능이 얼마나 떨어지는가?

**중요한 변수** -> 섞으면 성능 크게 하락
**중요하지 않은 변수** -> 섞어도 성능 유지

---

## 3.2 Permutation Importance 원리

```
원본 데이터:
온도  압력  속도  -> 정확도 90%

온도 섞기:
랜덤  압력  속도  -> 정확도 65%
                   |
          중요도 = 90% - 65% = 25%
```

---

## 3.3 sklearn에서 구현

```python
from sklearn.inspection import permutation_importance

# Permutation Importance 계산
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,      # 반복 횟수
    random_state=42
)

# 결과
importances = result.importances_mean
std = result.importances_std
```

---

## 실습 코드: Permutation Importance 계산

```python
from sklearn.inspection import permutation_importance

# Permutation Importance 계산
print("\nPermutation Importance 계산 중 (n_repeats=10)...")
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# 결과 정리
pi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\n[Permutation Importance]")
print("-" * 60)
for _, row in pi_df.iterrows():
    bar = '*' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} +/- {row['Std']:.3f} {bar}")
```

---

## 3.4 Permutation Importance 파라미터

| 파라미터 | 의미 |
|---------|------|
| `estimator` | 학습된 모델 |
| `X`, `y` | 테스트 데이터 |
| `n_repeats` | 반복 횟수 (안정성) |
| `scoring` | 평가 지표 |

---

## 3.5 Permutation Importance 시각화

```python
fig, ax = plt.subplots(figsize=(12, 8))

# 수평 막대 그래프 with 에러바
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(pi_df)))
ax.barh(pi_df['Feature'], pi_df['Importance'],
        xerr=pi_df['Std'], color=colors[::-1], capsize=3)

ax.set_xlabel('Importance (Performance Drop)', fontsize=12)
ax.set_title('Permutation Importance - Wine Classification', fontsize=14)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('permutation_importance_wine.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3.6 Feature vs Permutation Importance

| 항목 | Feature Importance | Permutation Importance |
|-----|-------------------|------------------------|
| 모델 | 트리 모델만 | 모든 모델 |
| 데이터 | 학습 데이터 기반 | 테스트 데이터 기반 |
| 상관 변수 | 분산됨 | 그래도 분산됨 |
| 계산 비용 | 빠름 | 느림 (n_repeats) |

---

## 실습 코드: Feature vs Permutation Importance 비교

```python
# 비교 테이블
comparison = pd.DataFrame({
    'Feature': feature_names,
    'Feature_Importance': feature_importance,
    'Permutation_Importance': perm_importance.importances_mean
})

# 순위 계산
comparison['FI_Rank'] = comparison['Feature_Importance'].rank(ascending=False).astype(int)
comparison['PI_Rank'] = comparison['Permutation_Importance'].rank(ascending=False).astype(int)
comparison['Rank_Diff'] = abs(comparison['FI_Rank'] - comparison['PI_Rank'])

comparison = comparison.sort_values('Feature_Importance', ascending=False)

print("\n[비교 테이블]")
print("-" * 80)
print(f"{'Feature':25s} | {'FI':8s} | {'PI':8s} | FI순위 | PI순위 | 차이")
print("-" * 80)
for _, row in comparison.iterrows():
    print(f"{row['Feature']:25s} | {row['Feature_Importance']:.3f}    | "
          f"{row['Permutation_Importance']:.3f}    | "
          f"{int(row['FI_Rank']):6d} | {int(row['PI_Rank']):6d} | {int(row['Rank_Diff']):4d}")

print("\n[해석]")
rank_diff_sum = comparison['Rank_Diff'].sum()
if rank_diff_sum <= 10:
    print("  -> 두 방법의 순위가 비교적 일치 - 결과 신뢰도 높음")
else:
    print("  -> 순위 차이 존재 - 상관 변수 확인 필요")
```

---

## 3.7 두 방법 함께 사용

```python
# 1. Feature Importance (빠르게)
fi = model.feature_importances_

# 2. Permutation Importance (정확하게)
pi = permutation_importance(model, X_test, y_test)

# 3. 비교
for name, f, p in zip(feature_names, fi, pi.importances_mean):
    print(f"{name}: FI={f:.3f}, PI={p:.3f}")
```

**두 방법의 순위가 비슷하면 신뢰도 상승**

---

## 3.8 비교 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Feature Importance
fi_sorted = fi_df.sort_values('Importance', ascending=True)
axes[0].barh(fi_sorted['Feature'], fi_sorted['Importance'], color='steelblue')
axes[0].set_title('Feature Importance', fontsize=12)
axes[0].set_xlabel('Importance')

# Permutation Importance
pi_sorted = pi_df.sort_values('Importance', ascending=True)
axes[1].barh(pi_sorted['Feature'], pi_sorted['Importance'], color='seagreen')
axes[1].set_title('Permutation Importance', fontsize=12)
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('importance_comparison_wine.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3.9 다른 모델과 비교

```python
from sklearn.ensemble import GradientBoostingClassifier

# GradientBoosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

gb_fi = gb_model.feature_importances_
gb_pi = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42)

print("\n[GradientBoosting Feature Importance - Top 5]")
gb_fi_sorted = sorted(zip(feature_names, gb_fi), key=lambda x: -x[1])[:5]
for feature, fi in gb_fi_sorted:
    print(f"  {feature:25s}: {fi:.3f}")

print("\n[모델 간 중요도 비교 - Top 5]")
model_comparison = pd.DataFrame({
    'Feature': feature_names,
    'RF_FI': feature_importance,
    'GB_FI': gb_fi,
    'RF_PI': perm_importance.importances_mean,
    'GB_PI': gb_pi.importances_mean
})
model_comparison = model_comparison.nlargest(5, 'RF_FI')
print(model_comparison.to_string(index=False))
```

---

## 3.10 상관 분석

```python
# 상관 행렬
corr_matrix = X.corr()

# 높은 상관관계 찾기 (|r| > 0.7)
print("\n[높은 상관관계 (|r| > 0.7)]")
found_high_corr = False
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            print(f"  {feature_names[i]} - {feature_names[j]}: {r:.2f}")
            found_high_corr = True

if not found_high_corr:
    print("  -> 높은 상관관계 없음")

# 상관 행렬 히트맵
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_yticklabels(feature_names)
plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Feature Correlation Matrix - Wine Dataset')
plt.tight_layout()
plt.savefig('correlation_matrix_wine.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3.11 SHAP 소개 (참고)

**SHAP (SHapley Additive exPlanations)**:
- 게임 이론 기반 해석 방법
- 개별 예측에 대한 변수 기여도
- 전역 + 지역 해석 모두 가능

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 3.12 SHAP 예시

```
샘플 1 예측: 불량 (확률 0.85)

기여도:
온도 (+0.25)  <- 높아서 불량 방향
압력 (+0.15)  <- 높아서 불량 방향
속도 (-0.05)  <- 낮아서 정상 방향
```

**왜 이 샘플이 불량인지** 구체적 설명

---

## 3.13 부분 의존성 플롯 (간단 버전)

```python
# 가장 중요한 변수의 부분 의존성
top_feature = fi_df.iloc[0]['Feature']
print(f"\n'{top_feature}'의 부분 의존성 분석:")

# 값 범위에 따른 예측 확률
feature_idx = list(feature_names).index(top_feature)
feature_values = np.linspace(X[top_feature].min(), X[top_feature].max(), 20)
X_temp = X_test.copy()
predictions_per_class = []

for val in feature_values:
    X_temp[top_feature] = val
    pred_proba = model.predict_proba(X_temp).mean(axis=0)
    predictions_per_class.append(pred_proba)

predictions_per_class = np.array(predictions_per_class)

print(f"\n  {top_feature} 값 변화에 따른 클래스별 예측 확률:")
sample_indices = [0, 5, 10, 15, 19]
for idx in sample_indices:
    probs = predictions_per_class[idx]
    print(f"    값 {feature_values[idx]:.1f}: "
          f"class_0={probs[0]:.2f}, class_1={probs[1]:.2f}, class_2={probs[2]:.2f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
for i, class_name in enumerate(wine.target_names):
    ax.plot(feature_values, predictions_per_class[:, i],
            label=class_name, linewidth=2)
ax.set_xlabel(top_feature, fontsize=12)
ax.set_ylabel('Predicted Probability', fontsize=12)
ax.set_title(f'Partial Dependence Plot: {top_feature}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('partial_dependence_wine.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3.14 모델 해석 워크플로우

```
1. 모델 학습
   |
2. Feature Importance (빠른 탐색)
   |
3. Permutation Importance (검증)
   |
4. SHAP (개별 예측 설명, 필요시)
   |
5. 비즈니스 인사이트 도출
```

---

## 3.15 비즈니스 인사이트 도출

```python
# 상위 중요 변수 추출
top_features = fi_df.head(3)['Feature'].tolist()
low_features = fi_df.tail(3)['Feature'].tolist()

print("\n[분석 결과]")
print("-" * 50)
print(f"  가장 중요한 특성 Top 3: {top_features}")
print(f"  상대적으로 덜 중요한 특성: {low_features}")

print("\n[Wine Classification 맥락에서의 해석]")
print("-" * 50)

# 주요 특성에 대한 해석
interpretations = {
    'proline': '프롤린(아미노산) - 와인 품종별 특성 반영',
    'flavanoids': '플라보노이드 - 와인의 색과 맛에 영향',
    'color_intensity': '색 강도 - 품종 구분에 중요',
    'od280/od315_of_diluted_wines': '희석된 와인의 흡광도 비율',
    'alcohol': '알코올 함량 - 와인 특성의 기본 지표'
}

for i, (_, row) in enumerate(fi_df.iterrows()):
    feature = row['Feature']
    importance = row['Importance']
    if importance > 0.1:
        interpretation = interpretations.get(feature, '품종 분류에 기여')
        print(f"  {feature}:")
        print(f"    - 중요도: {importance:.3f}")
        print(f"    - 해석: {interpretation}")
```

---

## 3.16 비즈니스 활용

**분석 결과**:
- 온도가 가장 중요 (25%)
- 압력이 두 번째 (18%)

**조치**:
1. 온도 모니터링 주기 단축
2. 온도 이상 시 즉시 알림
3. 압력 센서 교정 주기 확인
4. 불필요한 습도 센서 비용 절감 검토

---

# 핵심 정리

## 오늘 배운 내용

1. **모델 해석의 필요성**
   - 신뢰, 디버깅, 개선, 규제 대응
   - 블랙박스 -> 설명 가능한 AI

2. **Feature Importance**
   - 트리 모델 내장 기능
   - 빠르지만 상관 변수 문제

3. **Permutation Importance**
   - 모든 모델에 적용 가능
   - 테스트 데이터 기반, 신뢰도 높음

---

## 핵심 코드

```python
# Feature Importance (트리 모델)
model.feature_importances_

# Permutation Importance (모든 모델)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
result.importances_mean
result.importances_std
```

---

## 체크리스트

- [ ] Feature Importance 추출
- [ ] Permutation Importance 계산
- [ ] 두 방법 결과 비교
- [ ] 중요 변수 시각화
- [ ] 비즈니스 인사이트 도출

---

## 사용한 데이터셋

- **sklearn.datasets.load_wine**
  - 178개 샘플, 13개 특성, 3개 클래스
  - 와인 품종 분류 (class_0, class_1, class_2)

- **UCI Wine Quality (화이트 와인)**
  - 4,898개 샘플 (화이트 와인)
  - 알코올, 산도, 당분 등 11개 화학적 특성
  - 품질 점수 (3-9)로 분류

---

## 다음 차시 예고

### [24차시] 모델 저장과 실무 배포 준비

- joblib으로 모델 저장/로드
- Pipeline 구성
- 배포 체크리스트
