"""
[23차시] 모델 해석과 변수별 영향력 분석 - 실습 코드

학습 목표:
1. 모델 해석의 필요성과 종류를 이해한다
2. Feature Importance를 계산하고 해석한다
3. Permutation Importance를 활용한다

실습 환경: Python 3.8+, scikit-learn, matplotlib

데이터셋:
- sklearn.datasets.load_wine (와인 품질 분류)
- UCI Wine Quality Dataset (화이트 와인)
"""

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

# ============================================================
# 1. 실제 데이터셋 로드 (Wine Classification)
# ============================================================
print("\n" + "=" * 60)
print("1. 실제 데이터셋 로드 (Wine Classification)")
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

print("\n[데이터 미리보기]")
print(X.head())

print("\n[기술 통계]")
print(X.describe().round(2))

# ============================================================
# 2. 데이터 분할 및 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("2. 데이터 분할 및 모델 학습")
print("=" * 60)

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

# ============================================================
# 3. Feature Importance 추출
# ============================================================
print("\n" + "=" * 60)
print("3. Feature Importance 추출")
print("=" * 60)

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
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

print("\n[해석]")
top_features = fi_df.head(3)['Feature'].tolist()
print(f"  - 가장 중요한 특성: {top_features[0]}")
print(f"  - 상위 3개 특성이 전체 중요도의 "
      f"{fi_df.head(3)['Importance'].sum():.1%} 차지")

# ============================================================
# 4. Feature Importance 시각화
# ============================================================
print("\n" + "=" * 60)
print("4. Feature Importance 시각화")
print("=" * 60)

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
plt.close()
print("  → 'feature_importance_wine.png' 저장됨")

# ============================================================
# 5. Permutation Importance 계산
# ============================================================
print("\n" + "=" * 60)
print("5. Permutation Importance 계산")
print("=" * 60)

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
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} ± {row['Std']:.3f} {bar}")

# ============================================================
# 6. Permutation Importance 시각화
# ============================================================
print("\n" + "=" * 60)
print("6. Permutation Importance 시각화")
print("=" * 60)

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
plt.close()
print("  → 'permutation_importance_wine.png' 저장됨")

# ============================================================
# 7. Feature vs Permutation Importance 비교
# ============================================================
print("\n" + "=" * 60)
print("7. Feature vs Permutation Importance 비교")
print("=" * 60)

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
    print("  → 두 방법의 순위가 비교적 일치 - 결과 신뢰도 높음")
else:
    print("  → 순위 차이 존재 - 상관 변수 확인 필요")

# 비교 시각화
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
plt.close()
print("\n  → 'importance_comparison_wine.png' 저장됨")

# ============================================================
# 8. 다른 모델과 비교
# ============================================================
print("\n" + "=" * 60)
print("8. 다른 모델과 비교")
print("=" * 60)

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

# ============================================================
# 9. 상관 분석
# ============================================================
print("\n" + "=" * 60)
print("9. 상관 분석 (중요도 해석 보조)")
print("=" * 60)

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
    print("  → 높은 상관관계 없음")

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
plt.close()
print("\n  → 'correlation_matrix_wine.png' 저장됨")

# ============================================================
# 10. 비즈니스 인사이트 도출
# ============================================================
print("\n" + "=" * 60)
print("10. 비즈니스 인사이트 도출")
print("=" * 60)

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

# ============================================================
# 11. 부분 의존성 플롯 (간단 버전)
# ============================================================
print("\n" + "=" * 60)
print("11. 부분 의존성 플롯 (간단 버전)")
print("=" * 60)

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
plt.close()
print("\n  → 'partial_dependence_wine.png' 저장됨")

# ============================================================
# 12. UCI Wine Quality 데이터셋 추가 실습
# ============================================================
print("\n" + "=" * 60)
print("12. UCI Wine Quality 데이터셋 (화이트 와인)")
print("=" * 60)

# UCI 화이트 와인 품질 데이터셋 로드
try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    wine_quality = pd.read_csv(url, sep=';')
    print("\n[UCI Wine Quality 데이터 로드 성공]")
except Exception as e:
    print(f"\n[데이터 로드 실패: {e}]")
    print("오프라인 모드로 샘플 데이터 생성")
    np.random.seed(42)
    wine_quality = pd.DataFrame({
        'fixed acidity': np.random.normal(6.8, 0.8, 500),
        'volatile acidity': np.random.normal(0.28, 0.1, 500),
        'citric acid': np.random.normal(0.33, 0.12, 500),
        'residual sugar': np.random.normal(6.4, 5.0, 500),
        'chlorides': np.random.normal(0.045, 0.02, 500),
        'free sulfur dioxide': np.random.normal(35, 17, 500),
        'total sulfur dioxide': np.random.normal(138, 42, 500),
        'density': np.random.normal(0.994, 0.003, 500),
        'pH': np.random.normal(3.18, 0.15, 500),
        'sulphates': np.random.normal(0.49, 0.11, 500),
        'alcohol': np.random.normal(10.5, 1.2, 500),
        'quality': np.random.choice([5, 6, 7], 500, p=[0.3, 0.5, 0.2])
    })

print(f"  샘플 수: {len(wine_quality)}")
print(f"  특성 수: {wine_quality.shape[1] - 1}")
print(f"\n  품질 분포:")
print(wine_quality['quality'].value_counts().sort_index())

# 이진 분류로 변환 (품질 7 이상 = 좋은 와인)
wine_quality['good_quality'] = (wine_quality['quality'] >= 7).astype(int)

X_wq = wine_quality.drop(['quality', 'good_quality'], axis=1)
y_wq = wine_quality['good_quality']

print(f"\n  좋은 와인 비율: {y_wq.mean():.1%}")

# 학습
X_train_wq, X_test_wq, y_train_wq, y_test_wq = train_test_split(
    X_wq, y_wq, test_size=0.2, random_state=42, stratify=y_wq
)

model_wq = RandomForestClassifier(n_estimators=100, random_state=42)
model_wq.fit(X_train_wq, y_train_wq)

# Feature Importance
fi_wq = model_wq.feature_importances_
fi_wq_df = pd.DataFrame({
    'Feature': X_wq.columns,
    'Importance': fi_wq
}).sort_values('Importance', ascending=False)

print("\n[화이트 와인 품질 예측 - Feature Importance Top 5]")
for _, row in fi_wq_df.head(5).iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

# ============================================================
# 13. 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("13. 핵심 정리")
print("=" * 60)

print("""
[23차시 핵심 정리]

1. 모델 해석의 필요성
   - 신뢰: 예측 근거 설명
   - 디버깅: 잘못된 패턴 발견
   - 개선: 데이터 수집 방향
   - 규제: 설명 의무화 대응

2. Feature Importance
   - model.feature_importances_
   - 트리 모델에서 분할 시 불순도 감소량
   - 빠르지만 상관 변수 문제

3. Permutation Importance
   - permutation_importance(model, X, y, n_repeats=10)
   - 변수 섞어서 성능 하락 측정
   - 모든 모델에 적용 가능

4. 활용
   - 두 방법 순위 비교 → 신뢰도 확인
   - 중요 변수 → 데이터 품질 관리 강화
   - 덜 중요한 변수 → 수집 비용 효율성 검토

5. 사용한 데이터셋
   - sklearn.datasets.load_wine: 178샘플, 13특성, 3클래스
   - UCI Wine Quality: 4898샘플 (화이트 와인)
     - 알코올, 산도, 당분 등 11개 화학적 특성
     - 품질 점수 (3-9)로 분류

6. 핵심 코드
   ```python
   # Feature Importance (트리 모델)
   model.feature_importances_

   # Permutation Importance (모든 모델)
   from sklearn.inspection import permutation_importance
   result = permutation_importance(model, X_test, y_test, n_repeats=10)
   result.importances_mean
   result.importances_std
   ```
""")

print("\n다음 차시 예고: 24차시 - 모델 저장과 실무 배포 준비")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
