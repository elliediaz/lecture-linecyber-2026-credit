"""
번외D: ML 워크플로우 종합 실습
==============================
데이터에서 예측까지 한 번에!

데이터셋: Titanic (seaborn 내장)
목표: 승객의 생존 여부 예측 (이진 분류)

워크플로우:
1. 문제 정의
2. 데이터 탐색 (EDA)
3. 데이터 전처리
4. 모델 학습
5. 모델 평가
6. 결과 해석
"""

# ============================================================
# 라이브러리 임포트
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("번외D: ML 워크플로우 종합 실습")
print("=" * 60)

# ============================================================
# Part 1: 문제 정의
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 문제 정의")
print("=" * 60)

print("""
[문제 정의]
- 문제: 타이타닉 승객의 생존 여부 예측
- 유형: 이진 분류 (Binary Classification)
- 타겟: survived (0=사망, 1=생존)
- 평가: 정확도, 정밀도, 재현율, F1 Score
""")

# ============================================================
# Part 2: 데이터 탐색 (EDA)
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 데이터 탐색 (EDA)")
print("=" * 60)

# 데이터 로드
df = sns.load_dataset('titanic')
print(f"\n[데이터 크기]")
print(f"행: {df.shape[0]}, 열: {df.shape[1]}")

# 데이터 구조 확인
print(f"\n[데이터 타입 및 결측치]")
print(df.info())

# 기초 통계량
print(f"\n[기초 통계량]")
print(df.describe())

# 결측치 확인
print(f"\n[결측치 현황]")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '결측치 비율(%)': missing_pct
})
print(missing_df[missing_df['결측치 수'] > 0])

# 타겟 변수 분포
print(f"\n[타겟 변수 분포]")
print(df['survived'].value_counts())
print(f"\n생존율: {df['survived'].mean():.1%}")

# 주요 특성과 생존의 관계
print(f"\n[성별에 따른 생존율]")
print(df.groupby('sex')['survived'].mean().round(3))

print(f"\n[객실 등급에 따른 생존율]")
print(df.groupby('pclass')['survived'].mean().round(3))

# ============================================================
# Part 3: 데이터 전처리
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 데이터 전처리")
print("=" * 60)

# 작업용 복사본 생성
df_processed = df.copy()

# 3-1. 결측치 처리
print("\n[결측치 처리]")

# age: 중앙값으로 대체 (이상치에 강건)
age_median = df_processed['age'].median()
df_processed['age'].fillna(age_median, inplace=True)
print(f"- age 결측치 → 중앙값({age_median})으로 대체")

# embarked: 최빈값으로 대체
embarked_mode = df_processed['embarked'].mode()[0]
df_processed['embarked'].fillna(embarked_mode, inplace=True)
print(f"- embarked 결측치 → 최빈값('{embarked_mode}')으로 대체")

# deck: 결측치 너무 많음 → 제거
df_processed.drop(columns=['deck'], inplace=True)
print("- deck 컬럼 → 결측치 77% 이상으로 제거")

# 결측치 재확인
print(f"\n[전처리 후 결측치]")
print(f"결측치 총합: {df_processed.isnull().sum().sum()}")

# 3-2. 범주형 변수 인코딩
print("\n[범주형 인코딩]")

# 사용할 범주형 컬럼
cat_columns = ['sex', 'embarked']
print(f"인코딩 대상: {cat_columns}")

# One-Hot Encoding (drop_first=True로 다중공선성 방지)
df_processed = pd.get_dummies(
    df_processed,
    columns=cat_columns,
    drop_first=True
)
print("- One-Hot Encoding 적용 완료")

# 3-3. 특성 선택
print("\n[특성 선택]")

# 사용할 특성 목록
features = [
    'pclass',      # 객실 등급
    'age',         # 나이
    'sibsp',       # 형제/배우자 수
    'parch',       # 부모/자녀 수
    'fare',        # 요금
    'sex_male',    # 성별 (1=남성)
    'embarked_Q',  # 승선항 Q
    'embarked_S'   # 승선항 S
]

X = df_processed[features]
y = df_processed['survived']

print(f"선택된 특성: {features}")
print(f"특성 수: {len(features)}")

# 3-4. 데이터 분할
print("\n[데이터 분할]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% 테스트용
    random_state=42,    # 재현성
    stratify=y          # 클래스 비율 유지
)

print(f"학습 데이터: {len(X_train)}건")
print(f"테스트 데이터: {len(X_test)}건")
print(f"학습 데이터 생존율: {y_train.mean():.1%}")
print(f"테스트 데이터 생존율: {y_test.mean():.1%}")

# ============================================================
# Part 4: 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 모델 학습")
print("=" * 60)

# 3가지 모델 정의
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 모델 학습 및 평가
print("\n[모델 비교]")
results = {}

for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred
    }

    print(f"{name}: {accuracy:.3f}")

# 최고 성능 모델 선택
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
best_accuracy = results[best_name]['accuracy']
y_pred_best = results[best_name]['y_pred']

print(f"\n[최고 성능 모델]")
print(f"모델: {best_name}")
print(f"정확도: {best_accuracy:.3f}")

# ============================================================
# Part 5: 모델 평가
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 모델 평가")
print("=" * 60)

# 혼동 행렬
print("\n[혼동 행렬]")
cm = confusion_matrix(y_test, y_pred_best)
print(f"              예측:사망  예측:생존")
print(f"실제:사망     {cm[0,0]:^8}  {cm[0,1]:^8}")
print(f"실제:생존     {cm[1,0]:^8}  {cm[1,1]:^8}")

# 분류 보고서
print("\n[Classification Report]")
print(classification_report(y_test, y_pred_best,
                           target_names=['사망(0)', '생존(1)']))

# 각 지표 해석
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("\n[지표 해석]")
print(f"- 정밀도 (Precision): {precision:.3f}")
print(f"  → 생존으로 예측한 것 중 실제 생존 비율")
print(f"- 재현율 (Recall): {recall:.3f}")
print(f"  → 실제 생존자 중 찾아낸 비율")
print(f"- F1 Score: {f1:.3f}")
print(f"  → 정밀도와 재현율의 조화평균")

# ============================================================
# Part 6: 결과 해석
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 결과 해석")
print("=" * 60)

# 특성 중요도 (Random Forest)
if hasattr(best_model, 'feature_importances_'):
    print("\n[특성 중요도]")
    importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"{row['feature']:15} {row['importance']:.3f} {bar}")

    print("\n[비즈니스 인사이트]")
    print("1. 성별이 가장 중요한 예측 요인")
    print("   → '여성과 아이 먼저' 구조 정책 반영")
    print("2. 요금과 나이도 중요")
    print("   → 상위 등급 승객이 더 좋은 위치에 배정")
    print("3. 객실 등급(pclass)도 영향")
    print("   → 1등석 승객의 생존율이 높음")

# ============================================================
# 시각화
# ============================================================
print("\n" + "=" * 60)
print("시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 모델 성능 비교
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(model_names, accuracies, color=colors)
ax1.set_ylim(0.7, 0.9)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Comparison')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

# 2. 혼동 행렬 히트맵
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')

# 3. 특성 중요도
ax3 = axes[1, 0]
if hasattr(best_model, 'feature_importances_'):
    importance_df_sorted = importance_df.sort_values('importance', ascending=True)
    ax3.barh(importance_df_sorted['feature'], importance_df_sorted['importance'],
             color='#3498db')
    ax3.set_xlabel('Importance')
    ax3.set_title('Feature Importance (Random Forest)')

# 4. 성별/등급별 생존율
ax4 = axes[1, 1]
survival_by_sex_class = df.groupby(['sex', 'pclass'])['survived'].mean().unstack()
survival_by_sex_class.plot(kind='bar', ax=ax4, color=['#3498db', '#2ecc71', '#e74c3c'])
ax4.set_xlabel('Sex')
ax4.set_ylabel('Survival Rate')
ax4.set_title('Survival Rate by Sex and Class')
ax4.legend(title='Class')
ax4.set_xticklabels(['Female', 'Male'], rotation=0)

plt.tight_layout()
plt.savefig('ml_workflow_results.png', dpi=150, bbox_inches='tight')
print("시각화 저장 완료: ml_workflow_results.png")
plt.show()

# ============================================================
# 전체 워크플로우 요약
# ============================================================
print("\n" + "=" * 60)
print("전체 워크플로우 요약")
print("=" * 60)

print("""
[ML 워크플로우 6단계]

1. 문제 정의
   - 분류 vs 회귀 판단
   - 타이타닉: 이진 분류

2. 데이터 탐색 (EDA)
   - shape, info, describe
   - 결측치 확인
   - 특성별 분석

3. 데이터 전처리
   - 결측치: 중앙값/최빈값 대체
   - 인코딩: One-Hot Encoding
   - 특성 선택: 8개 특성

4. 모델 학습
   - 3가지 모델 비교
   - Random Forest 최고 성능

5. 모델 평가
   - 정확도: 82.1%
   - 정밀도, 재현율, F1

6. 결과 해석
   - 특성 중요도 분석
   - 비즈니스 인사이트 도출
""")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
