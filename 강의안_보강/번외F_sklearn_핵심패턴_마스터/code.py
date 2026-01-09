"""
번외F: sklearn 핵심 패턴 마스터
==============================
fit, transform, Pipeline 완전 정복

데이터셋: Iris (sklearn), Titanic (seaborn)

내용:
1. fit / predict / transform 완전 이해
2. fit_transform() 올바른 사용법
3. Pipeline으로 워크플로우 자동화
4. ColumnTransformer로 컬럼별 전처리
5. 모델 저장 및 배포
"""

# ============================================================
# 라이브러리 임포트
# ============================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("번외F: sklearn 핵심 패턴 마스터")
print("=" * 60)

# ============================================================
# Part 1: 기본 패턴 이해 - fit / predict / transform
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 기본 패턴 이해")
print("=" * 60)

# Iris 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

print(f"\n[데이터 정보]")
print(f"샘플 수: {X.shape[0]}")
print(f"특성 수: {X.shape[1]}")
print(f"특성 이름: {feature_names}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n학습 데이터: {len(X_train)}건")
print(f"테스트 데이터: {len(X_test)}건")

# ============================================================
# Part 1-1: fit()과 transform() 분리
# ============================================================
print("\n" + "-" * 40)
print("[1-1. fit()과 transform() 분리]")
print("-" * 40)

# 스케일러 생성
scaler = StandardScaler()

# fit(): 학습 데이터로 평균/표준편차 계산
scaler.fit(X_train)

print(f"\n[fit() 후 저장된 값]")
print(f"평균: {scaler.mean_}")
print(f"표준편차: {scaler.scale_}")

# transform(): 저장된 값으로 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n[변환 결과]")
print(f"학습 데이터 변환 후 평균: {X_train_scaled.mean(axis=0).round(2)}")
print(f"테스트 데이터 변환 후 평균: {X_test_scaled.mean(axis=0).round(2)}")
print("→ 테스트도 학습 기준으로 변환됨 (0이 아닐 수 있음)")

# ============================================================
# Part 1-2: fit_transform() 사용법
# ============================================================
print("\n" + "-" * 40)
print("[1-2. fit_transform() 사용법]")
print("-" * 40)

# 새 스케일러로 시연
scaler2 = StandardScaler()

# fit_transform(): fit + transform 한 번에 (학습용에서만!)
X_train_scaled2 = scaler2.fit_transform(X_train)

# 테스트는 transform만!
X_test_scaled2 = scaler2.transform(X_test)

print(f"학습: fit_transform() 사용")
print(f"테스트: transform()만 사용")
print(f"결과 동일 확인: {np.allclose(X_train_scaled, X_train_scaled2)}")

# ============================================================
# Part 1-3: 잘못된 사용법 시연
# ============================================================
print("\n" + "-" * 40)
print("[1-3. 잘못된 사용법 (하지 마세요!)]")
print("-" * 40)

scaler_wrong = StandardScaler()

# 잘못된 방법: 테스트에도 fit
X_train_wrong = scaler_wrong.fit_transform(X_train)
X_test_wrong = scaler_wrong.fit_transform(X_test)  # ❌ 잘못!

print("❌ 잘못된 방법:")
print(f"  학습 기준 평균: {X_train_wrong.mean(axis=0).round(2)}")
print(f"  테스트 기준 평균: {X_test_wrong.mean(axis=0).round(2)}")
print("  → 둘 다 0이지만 서로 다른 기준!")

print("\n✅ 올바른 방법:")
print(f"  학습 기준 평균: {X_train_scaled.mean(axis=0).round(2)}")
print(f"  테스트 기준 평균: {X_test_scaled.mean(axis=0).round(2)}")
print("  → 학습 기준으로 테스트도 변환!")

# ============================================================
# Part 2: 모델 학습과 예측
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 모델 학습과 예측")
print("=" * 60)

# 모델 생성
model = LogisticRegression(max_iter=200)

# fit(): 모델 학습
model.fit(X_train_scaled, y_train)

# predict(): 예측
y_pred = model.predict(X_test_scaled)

# score(): 평가
accuracy = model.score(X_test_scaled, y_test)

print(f"\n[모델 평가]")
print(f"정확도: {accuracy:.3f}")

# 학습된 파라미터 확인
print(f"\n[학습된 파라미터]")
print(f"계수 shape: {model.coef_.shape}")
print(f"절편: {model.intercept_}")

# ============================================================
# Part 3: Pipeline 사용
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Pipeline 사용")
print("=" * 60)

# 수동 방식의 문제점
print("\n[수동 방식의 문제점]")
print("1. 전처리와 모델을 따로 관리")
print("2. fit/transform 순서 실수 가능")
print("3. 배포 시 모든 객체를 따로 저장/로드")

# Pipeline 생성
print("\n[Pipeline 생성]")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200))
])

print("파이프라인 구조:")
for name, step in pipeline.named_steps.items():
    print(f"  - {name}: {type(step).__name__}")

# Pipeline 학습
pipeline.fit(X_train, y_train)

# Pipeline 예측 (내부에서 스케일링 자동!)
y_pred_pipe = pipeline.predict(X_test)

# 평가
accuracy_pipe = pipeline.score(X_test, y_test)
print(f"\n[Pipeline 평가]")
print(f"정확도: {accuracy_pipe:.3f}")
print(f"수동 방식과 동일: {accuracy == accuracy_pipe}")

# ============================================================
# Part 3-2: make_pipeline 사용
# ============================================================
print("\n" + "-" * 40)
print("[3-2. make_pipeline 사용]")
print("-" * 40)

# 이름 자동 생성
pipe_auto = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)

# 자동 생성된 이름 확인
print("자동 생성된 이름:")
for name in pipe_auto.named_steps.keys():
    print(f"  - {name}")

# ============================================================
# Part 3-3: Pipeline 단계 접근
# ============================================================
print("\n" + "-" * 40)
print("[3-3. Pipeline 단계 접근]")
print("-" * 40)

# 학습 후 접근
pipe_auto.fit(X_train, y_train)

# 스케일러의 저장된 값
print(f"스케일러 평균: {pipe_auto.named_steps['standardscaler'].mean_}")

# 모델의 학습된 파라미터
print(f"모델 계수 shape: {pipe_auto.named_steps['logisticregression'].coef_.shape}")

# ============================================================
# Part 4: ColumnTransformer - 컬럼별 전처리
# ============================================================
print("\n" + "=" * 60)
print("Part 4: ColumnTransformer - 컬럼별 전처리")
print("=" * 60)

# Titanic 데이터 로드
df = sns.load_dataset('titanic')

print(f"\n[Titanic 데이터]")
print(f"샘플 수: {len(df)}")
print(df.head())

# 특성과 타겟 분리
X_df = df[['pclass', 'age', 'fare', 'sex', 'embarked']].copy()
y_df = df['survived']

# 결측치 처리
X_df['age'].fillna(X_df['age'].median(), inplace=True)
X_df['embarked'].fillna('S', inplace=True)

print(f"\n[컬럼 유형]")
print(X_df.dtypes)

# 컬럼 유형 분리
numeric_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

print(f"\n수치형: {numeric_features}")
print(f"범주형: {categorical_features}")

# ============================================================
# Part 4-1: ColumnTransformer 정의
# ============================================================
print("\n" + "-" * 40)
print("[4-1. ColumnTransformer 정의]")
print("-" * 40)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

print("ColumnTransformer 구조:")
for name, transformer, cols in preprocessor.transformers:
    print(f"  - {name}: {type(transformer).__name__} → {cols}")

# ============================================================
# Part 4-2: 전체 Pipeline 구성
# ============================================================
print("\n" + "-" * 40)
print("[4-2. 전체 Pipeline 구성]")
print("-" * 40)

# 전처리 + 모델 파이프라인
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("전체 파이프라인:")
print("  1. preprocessor (ColumnTransformer)")
print("     - num: StandardScaler → ['age', 'fare']")
print("     - cat: OneHotEncoder → ['pclass', 'sex', 'embarked']")
print("  2. classifier (RandomForestClassifier)")

# 데이터 분할
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
)

# 학습
full_pipeline.fit(X_train_df, y_train_df)

# 평가
y_pred_df = full_pipeline.predict(X_test_df)
accuracy_df = accuracy_score(y_test_df, y_pred_df)

print(f"\n[평가 결과]")
print(f"정확도: {accuracy_df:.3f}")

# ============================================================
# Part 4-3: 새 데이터로 예측
# ============================================================
print("\n" + "-" * 40)
print("[4-3. 새 데이터로 예측]")
print("-" * 40)

# 새로운 승객 데이터
new_passenger = pd.DataFrame({
    'pclass': [1, 3],
    'age': [30, 25],
    'fare': [100, 10],
    'sex': ['female', 'male'],
    'embarked': ['C', 'S']
})

print("새 승객 데이터:")
print(new_passenger)

# 전처리 + 예측 한 번에!
predictions = full_pipeline.predict(new_passenger)
print(f"\n생존 예측: {predictions}")
print("  0: 사망, 1: 생존")

# ============================================================
# Part 5: GridSearchCV + Pipeline
# ============================================================
print("\n" + "=" * 60)
print("Part 5: GridSearchCV + Pipeline")
print("=" * 60)

# 파라미터 그리드 (파이프라인 단계명__파라미터)
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None]
}

print("[파라미터 그리드]")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# GridSearchCV
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_df, y_train_df)

print(f"\n[GridSearchCV 결과]")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 CV 점수: {grid_search.best_score_:.3f}")

# 테스트 평가
best_accuracy = grid_search.score(X_test_df, y_test_df)
print(f"테스트 정확도: {best_accuracy:.3f}")

# ============================================================
# Part 6: 모델 저장 및 배포
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 모델 저장 및 배포")
print("=" * 60)

# 최적 모델 저장
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'titanic_pipeline.pkl')
print("✅ 모델 저장 완료: titanic_pipeline.pkl")

# 모델 로드
loaded_model = joblib.load('titanic_pipeline.pkl')
print("✅ 모델 로드 완료")

# 로드된 모델로 예측
loaded_predictions = loaded_model.predict(new_passenger)
print(f"\n[로드된 모델로 예측]")
print(f"예측 결과: {loaded_predictions}")

# ============================================================
# Part 7: 시각화
# ============================================================
print("\n" + "=" * 60)
print("시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. fit/transform 개념
ax1 = axes[0, 0]
data = ['fit()', 'transform()', 'predict()', 'fit_transform()']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax1.barh(data, [1, 1, 1, 1], color=colors)
ax1.set_xlim(0, 1.5)
ax1.set_title('sklearn Core Methods')
for i, (bar, label) in enumerate(zip(bars, ['Learn from data', 'Apply learned params', 'Make predictions', 'fit + transform'])):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, label, va='center')
ax1.set_xticks([])

# 2. Pipeline vs Manual
ax2 = axes[0, 1]
methods = ['Manual\n(3 objects)', 'Pipeline\n(1 object)']
complexity = [3, 1]
ax2.bar(methods, complexity, color=['#e74c3c', '#2ecc71'])
ax2.set_ylabel('Number of objects to manage')
ax2.set_title('Pipeline Simplifies Workflow')

# 3. Titanic 모델 성능
ax3 = axes[1, 0]
models = ['Before\nTuning', 'After\nTuning']
accuracies = [accuracy_df, best_accuracy]
bars = ax3.bar(models, accuracies, color=['#3498db', '#2ecc71'])
ax3.set_ylabel('Accuracy')
ax3.set_title('Model Performance')
ax3.set_ylim(0.7, 0.9)
for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center')

# 4. ColumnTransformer 구조
ax4 = axes[1, 1]
ax4.text(0.5, 0.8, 'ColumnTransformer', fontsize=14, ha='center', fontweight='bold')
ax4.text(0.25, 0.5, 'Numeric\n(StandardScaler)', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.5))
ax4.text(0.75, 0.5, 'Categorical\n(OneHotEncoder)', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.5))
ax4.text(0.5, 0.2, 'Combined Features', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.5))
ax4.arrow(0.25, 0.4, 0.15, -0.12, head_width=0.03, head_length=0.02, fc='black')
ax4.arrow(0.75, 0.4, -0.15, -0.12, head_width=0.03, head_length=0.02, fc='black')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('ColumnTransformer Structure')

plt.tight_layout()
plt.savefig('sklearn_patterns.png', dpi=150, bbox_inches='tight')
print("시각화 저장 완료: sklearn_patterns.png")
plt.show()

# ============================================================
# 핵심 정리
# ============================================================
print("\n" + "=" * 60)
print("핵심 정리")
print("=" * 60)

print("""
[sklearn 핵심 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. fit()은 학습 데이터에서만!
   - 스케일러: 평균/표준편차 계산
   - 인코더: 범주 목록 학습
   - 모델: 패턴 학습

2. transform()/predict()는 fit 후에!
   - 학습에서 계산된 값으로 변환/예측
   - 테스트에도 동일한 기준 적용

3. fit_transform()은 학습에서만!
   - fit() + transform() 편의 메서드
   - 테스트에서 절대 사용 금지

4. Pipeline으로 자동화!
   - 전처리 + 모델을 하나로
   - fit/transform 자동 관리
   - 저장/로드 간편

5. ColumnTransformer로 유연하게!
   - 컬럼별 다른 전처리
   - Pipeline과 결합 가능
""")

# ============================================================
# 체크리스트
# ============================================================
print("\n" + "=" * 60)
print("체크리스트")
print("=" * 60)

print("""
[실무 적용 체크리스트]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ 데이터 분할 후 전처리 (전처리 후 분할 X)
□ 학습 데이터로만 fit() 호출
□ 테스트 데이터는 transform()만
□ Pipeline으로 전처리 + 모델 묶기
□ ColumnTransformer로 컬럼별 처리
□ joblib로 전체 파이프라인 저장
□ 새 데이터는 파이프라인에 바로 입력
""")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)

# 임시 파일 정리
import os
if os.path.exists('titanic_pipeline.pkl'):
    os.remove('titanic_pipeline.pkl')
    print("(임시 파일 정리 완료)")
