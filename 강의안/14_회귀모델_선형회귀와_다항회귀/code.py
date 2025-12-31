"""
[14차시] 회귀 모델: 선형회귀와 다항회귀 - 실습 코드
학습목표: 선형회귀, 다항회귀 실습, 회귀 모델 평가 지표 이해
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 제조 데이터 생성
# ============================================================
print("=" * 50)
print("1. 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 300

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
speed = np.random.normal(100, 15, n_samples)

# 생산량 (숫자 타겟 - 회귀 문제)
# 온도와 속도에 선형적, 습도와는 약간 비선형 관계
production = (
    1000 +
    5 * speed -
    3 * (temperature - 85) -
    0.05 * (humidity - 50) ** 2 +  # 비선형 항
    np.random.normal(0, 30, n_samples)  # 노이즈
)

df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '생산량': production
})

print(df.head())
print(f"\n데이터 크기: {df.shape}")
print(f"생산량 평균: {df['생산량'].mean():.1f}")
print(f"생산량 표준편차: {df['생산량'].std():.1f}")

# ============================================================
# 2. 데이터 준비
# ============================================================
print("\n" + "=" * 50)
print("2. 데이터 준비")
print("=" * 50)

X = df[['온도', '습도', '속도']]
y = df['생산량']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# ============================================================
# 3. 선형회귀
# ============================================================
print("\n" + "=" * 50)
print("3. 선형회귀 (Linear Regression)")
print("=" * 50)

# 모델 학습
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 계수 확인
print("▶ 학습 완료!")
print(f"\n   회귀 계수 (기울기):")
for name, coef in zip(X.columns, lr_model.coef_):
    print(f"   - {name}: {coef:.3f}")
print(f"   절편: {lr_model.intercept_:.3f}")

# 해석
print("\n▶ 해석:")
print(f"   - 속도가 1 증가하면 생산량이 {lr_model.coef_[2]:.1f}개 증가")
print(f"   - 온도가 1도 상승하면 생산량이 {lr_model.coef_[0]:.1f}개 변화")

# 예측
y_pred_lr = lr_model.predict(X_test)

# ============================================================
# 4. 회귀 모델 평가 지표
# ============================================================
print("\n" + "=" * 50)
print("4. 회귀 모델 평가 지표")
print("=" * 50)

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred_lr)
print(f"MAE (평균 절대 오차): {mae:.2f}")

# MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred_lr)
print(f"MSE (평균 제곱 오차): {mse:.2f}")

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")

# R² (결정계수)
r2 = r2_score(y_test, y_pred_lr)
print(f"R² (결정계수): {r2:.4f}")

print("\n▶ 해석:")
print(f"   - RMSE {rmse:.1f}: 평균적으로 약 {rmse:.0f}개 정도 오차")
print(f"   - R² {r2:.3f}: 모델이 데이터 변동의 {r2*100:.1f}%를 설명")

# ============================================================
# 5. 다항회귀 (Polynomial Regression)
# ============================================================
print("\n" + "=" * 50)
print("5. 다항회귀 (Polynomial Regression)")
print("=" * 50)

# PolynomialFeatures 이해
print("▶ PolynomialFeatures 동작 예시")
poly_demo = PolynomialFeatures(degree=2, include_bias=False)
X_demo = np.array([[2, 3]])
X_poly_demo = poly_demo.fit_transform(X_demo)
print(f"   원래: {X_demo}")
print(f"   변환: {X_poly_demo}")
print(f"   특성: {poly_demo.get_feature_names_out(['x1', 'x2'])}")

# Pipeline으로 다항회귀 구현
print("\n▶ 2차 다항회귀 학습")
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

poly_pipe.fit(X_train, y_train)
y_pred_poly = poly_pipe.predict(X_test)

# 평가
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"   RMSE: {rmse_poly:.2f}")
print(f"   R²: {r2_poly:.4f}")

# ============================================================
# 6. 차수(degree)에 따른 성능 비교
# ============================================================
print("\n" + "=" * 50)
print("6. 차수(degree)에 따른 성능")
print("=" * 50)

degrees = [1, 2, 3, 4, 5]
results = []

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('linear', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)

    results.append({
        'degree': d,
        '학습 R²': f'{train_r2:.4f}',
        '테스트 R²': f'{test_r2:.4f}'
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\n★ degree가 너무 높으면 과대적합 발생 가능!")

# ============================================================
# 7. 트리 기반 회귀 모델
# ============================================================
print("\n" + "=" * 50)
print("7. 트리 기반 회귀 모델")
print("=" * 50)

# 의사결정트리 회귀
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)
dt_pred = dt_reg.predict(X_test)
dt_r2 = r2_score(y_test, dt_pred)

# 랜덤포레스트 회귀
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)

print(f"의사결정트리 회귀 R²: {dt_r2:.4f}")
print(f"랜덤포레스트 회귀 R²: {rf_r2:.4f}")

# ============================================================
# 8. 모델 비교
# ============================================================
print("\n" + "=" * 50)
print("8. 회귀 모델 비교")
print("=" * 50)

models = {
    '선형회귀': lr_model,
    '다항회귀(2차)': poly_pipe,
    '의사결정트리': dt_reg,
    '랜덤포레스트': rf_reg
}

print(f"{'모델':<20} {'테스트 R²':<12} {'RMSE':<12}")
print("-" * 45)

for name, model in models.items():
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:<20} {r2:<12.4f} {rmse:<12.2f}")

# ============================================================
# 9. 예측값 vs 실제값 시각화
# ============================================================
print("\n" + "=" * 50)
print("9. 시각화")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

model_preds = [
    ('Linear Regression', y_pred_lr),
    ('Polynomial (d=2)', y_pred_poly),
    ('Decision Tree', dt_pred),
    ('Random Forest', rf_pred)
]

for ax, (name, pred) in zip(axes.flat, model_preds):
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, pred):.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig('regression_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("▶ regression_comparison.png 저장됨")

# ============================================================
# 10. 핵심 요약
# ============================================================
print("\n" + "=" * 50)
print("10. 핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                  회귀 모델 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 선형회귀                                            │
│     model = LinearRegression()                         │
│     model.coef_     → 각 특성의 기울기                  │
│     model.intercept_ → 절편                            │
│                                                        │
│  ▶ 다항회귀                                            │
│     pipe = Pipeline([                                  │
│         ('poly', PolynomialFeatures(degree=2)),        │
│         ('linear', LinearRegression())                 │
│     ])                                                 │
│                                                        │
│  ▶ 트리 기반 회귀                                      │
│     DecisionTreeRegressor                              │
│     RandomForestRegressor                              │
│                                                        │
│  ▶ 평가 지표                                           │
│     - RMSE: 작을수록 좋음 (원래 단위)                   │
│     - R²: 1에 가까울수록 좋음 (0~1)                     │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 모델 평가와 교차검증
""")
