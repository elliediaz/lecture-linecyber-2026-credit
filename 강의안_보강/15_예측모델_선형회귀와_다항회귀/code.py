"""
[15차시] 예측 모델 - 선형/다항회귀 - 실습 코드
===============================================

학습 목표:
1. 회귀 문제의 개념을 이해한다
2. 선형회귀 모델을 사용한다
3. 다항회귀로 비선형 관계를 학습한다

실습 내용:
- California Housing 데이터셋으로 주택 가격 예측
- LinearRegression 사용
- PolynomialFeatures + Pipeline
- 차수별 성능 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Part 1: 데이터 로딩
# ============================================================

print("=" * 60)
print("Part 1: California Housing 데이터셋 로딩")
print("=" * 60)

# California Housing 데이터셋 로딩
print("\n[California Housing 데이터셋 로딩 중...]")
try:
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target  # 중간 주택 가격 (단위: $100,000)
    print("California Housing 데이터셋 로딩 완료!")
except Exception as e:
    print(f"데이터셋 로딩 실패: {e}")
    raise

print(f"\n[데이터 확인]")
print(f"데이터 크기: {df.shape}")
print(f"특성 이름: {list(housing.feature_names)}")
print(f"\n특성 설명:")
print("  - MedInc: 중위 소득 (만 달러 단위)")
print("  - HouseAge: 주택 연식 (년)")
print("  - AveRooms: 평균 방 개수")
print("  - AveBedrms: 평균 침실 개수")
print("  - Population: 인구")
print("  - AveOccup: 평균 거주자 수")
print("  - Latitude: 위도")
print("  - Longitude: 경도")

print(f"\n기술통계:")
print(df.describe())

print(f"\n타겟 (주택 가격) 정보:")
print(f"  평균: ${df['MedHouseVal'].mean()*100000:,.0f}")
print(f"  범위: ${df['MedHouseVal'].min()*100000:,.0f} ~ ${df['MedHouseVal'].max()*100000:,.0f}")


# ============================================================
# Part 2: 데이터 분할
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 데이터 분할")
print("=" * 60)

# 주요 특성 선택
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
X = df[feature_columns]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"주택 가격 범위: ${y.min()*100000:,.0f} ~ ${y.max()*100000:,.0f}")


# ============================================================
# Part 3: 선형회귀
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 선형회귀 (LinearRegression)")
print("=" * 60)

# 모델 생성 및 학습
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 예측
y_pred_linear = linear_model.predict(X_test)

# 평가
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

print(f"\n[선형회귀 결과]")
print(f"R² 점수: {r2_linear:.3f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f} (약 ${rmse_linear*100000:,.0f})")
print(f"MAE: {mae_linear:.4f} (약 ${mae_linear*100000:,.0f})")

# 계수 확인
print(f"\n[회귀 계수]")
print(f"절편 (intercept): {linear_model.intercept_:.4f}")
for col, coef in zip(feature_columns, linear_model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f"  {col}: {sign}{coef:.4f}")

print(f"\n[계수 해석]")
print(f"  -> 중위 소득(MedInc) 1단위 증가 시 주택 가격 ${linear_model.coef_[0]*100000:,.0f} 증가")
print(f"  -> 주택 연식(HouseAge) 1년 증가 시 주택 가격 ${linear_model.coef_[1]*100000:,.0f} 변화")


# ============================================================
# Part 4: 다항회귀
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 다항회귀 (PolynomialFeatures)")
print("=" * 60)

# 스케일링 포함 파이프라인 (다항회귀에서 중요)
# 2차 다항회귀
poly_model_2 = Pipeline([
    ('scaler', StandardScaler()),  # 스케일링 추가
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

poly_model_2.fit(X_train, y_train)
y_pred_poly2 = poly_model_2.predict(X_test)

r2_poly2 = r2_score(y_test, y_pred_poly2)
rmse_poly2 = np.sqrt(mean_squared_error(y_test, y_pred_poly2))
mae_poly2 = mean_absolute_error(y_test, y_pred_poly2)

print(f"\n[2차 다항회귀 결과]")
print(f"R² 점수: {r2_poly2:.3f}")
print(f"RMSE: {rmse_poly2:.4f} (약 ${rmse_poly2*100000:,.0f})")
print(f"MAE: {mae_poly2:.4f}")

# 3차 다항회귀
poly_model_3 = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linear', LinearRegression())
])

poly_model_3.fit(X_train, y_train)
y_pred_poly3 = poly_model_3.predict(X_test)

r2_poly3 = r2_score(y_test, y_pred_poly3)
rmse_poly3 = np.sqrt(mean_squared_error(y_test, y_pred_poly3))

print(f"\n[3차 다항회귀 결과]")
print(f"R² 점수: {r2_poly3:.3f}")
print(f"RMSE: {rmse_poly3:.4f} (약 ${rmse_poly3*100000:,.0f})")


# ============================================================
# Part 5: 차수별 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 차수별 성능 비교")
print("=" * 60)

degrees = [1, 2, 3]
train_scores = []
test_scores = []
rmse_scores = []

print(f"\n{'차수':>4} {'학습 R²':>10} {'테스트 R²':>12} {'RMSE':>12}")
print("-" * 45)

for deg in degrees:
    if deg == 1:
        # 선형회귀
        poly = LinearRegression()
    else:
        poly = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('linear', LinearRegression())
        ])
    poly.fit(X_train, y_train)

    train_score = poly.score(X_train, y_train)
    test_score = poly.score(X_test, y_test)
    y_pred_temp = poly.predict(X_test)
    rmse_temp = np.sqrt(mean_squared_error(y_test, y_pred_temp))

    train_scores.append(train_score)
    test_scores.append(test_score)
    rmse_scores.append(rmse_temp)

    print(f"{deg:>4} {train_score:>10.3f} {test_score:>12.3f} {rmse_temp:>12.4f}")

# 최적 차수
best_idx = np.argmax(test_scores)
print(f"\n최적 차수: {degrees[best_idx]} (테스트 R² = {test_scores[best_idx]:.3f})")


# ============================================================
# Part 6: 모델 비교 요약
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 모델 비교 요약")
print("=" * 60)

print(f"\n{'모델':<20} {'R² 점수':>10} {'RMSE':>12} {'MAE':>10}")
print("-" * 55)
print(f"{'선형회귀':<20} {r2_linear:>10.3f} {rmse_linear:>12.4f} {mae_linear:>10.4f}")
print(f"{'다항회귀 (degree=2)':<20} {r2_poly2:>10.3f} {rmse_poly2:>12.4f} {mae_poly2:>10.4f}")
print(f"{'다항회귀 (degree=3)':<20} {r2_poly3:>10.3f} {rmse_poly3:>12.4f} {'-':>10}")


# ============================================================
# Part 7: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 선형회귀: 실제 vs 예측
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_linear, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('실제 주택 가격')
ax1.set_ylabel('예측 주택 가격')
ax1.set_title(f'선형회귀 (R²={r2_linear:.3f})')
ax1.grid(True, alpha=0.3)

# 2. 다항회귀(2차): 실제 vs 예측
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_poly2, alpha=0.3, s=10, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('실제 주택 가격')
ax2.set_ylabel('예측 주택 가격')
ax2.set_title(f'다항회귀 degree=2 (R²={r2_poly2:.3f})')
ax2.grid(True, alpha=0.3)

# 3. 차수별 R² 점수
ax3 = axes[1, 0]
ax3.plot(degrees, train_scores, 'o-', label='학습', color='blue')
ax3.plot(degrees, test_scores, 'o-', label='테스트', color='orange')
ax3.axvline(x=degrees[best_idx], color='red', linestyle='--',
            label=f'최적 (degree={degrees[best_idx]})')
ax3.set_xlabel('다항 차수 (degree)')
ax3.set_ylabel('R² 점수')
ax3.set_title('차수별 R² 점수')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(degrees)

# 4. 잔차 분포
ax4 = axes[1, 1]
residuals_linear = y_test - y_pred_linear
residuals_poly2 = y_test - y_pred_poly2
ax4.hist(residuals_linear, bins=50, alpha=0.7, label='선형', color='blue')
ax4.hist(residuals_poly2, bins=50, alpha=0.7, label='다항(2차)', color='orange')
ax4.axvline(x=0, color='red', linestyle='--', lw=2)
ax4.set_xlabel('잔차 (실제 - 예측)')
ax4.set_ylabel('빈도')
ax4.set_title('잔차 분포')
ax4.legend()

plt.tight_layout()
plt.savefig('15_regression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 15_regression_comparison.png")


# ============================================================
# Part 8: 새 데이터 예측
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 새 데이터 예측")
print("=" * 60)

# 새 주택 조건
new_data = pd.DataFrame({
    'MedInc': [3.0, 5.0, 8.0, 10.0, 2.0],     # 중위 소득
    'HouseAge': [20, 15, 5, 10, 30],           # 주택 연식
    'AveRooms': [5.0, 6.0, 7.0, 8.0, 4.5],     # 평균 방 수
    'AveOccup': [3.0, 2.5, 2.0, 2.2, 3.5]      # 평균 거주자 수
})

print("\n[새 주택 조건]")
print(new_data)

# 예측
linear_pred = linear_model.predict(new_data)
poly2_pred = poly_model_2.predict(new_data)

print("\n[주택 가격 예측 결과]")
print(f"{'조건':>4} {'선형회귀':>15} {'다항(2차)':>15}")
print("-" * 40)
for i in range(len(new_data)):
    print(f"{i+1:>4} ${linear_pred[i]*100000:>13,.0f} ${poly2_pred[i]*100000:>13,.0f}")


# ============================================================
# Part 9: 단변량 회귀 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 9: 단변량 회귀 시각화 (MedInc vs 주택 가격)")
print("=" * 60)

# 중위 소득만 사용한 회귀
X_income = df[['MedInc']]
y_price = df['MedHouseVal']

X_inc_train, X_inc_test, y_inc_train, y_inc_test = train_test_split(
    X_income, y_price, test_size=0.2, random_state=42
)

# 선형회귀
linear_1d = LinearRegression()
linear_1d.fit(X_inc_train, y_inc_train)

# 다항회귀 (2차, 3차)
poly2_1d = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly2_1d.fit(X_inc_train, y_inc_train)

poly3_1d = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
poly3_1d.fit(X_inc_train, y_inc_train)

# 시각화
plt.figure(figsize=(12, 5))

# 샘플링해서 시각화 (데이터가 많으므로)
sample_idx = np.random.choice(len(X_income), size=2000, replace=False)
X_sample = X_income.iloc[sample_idx]
y_sample = y_price.iloc[sample_idx]

X_plot = np.linspace(X_income.min()[0], X_income.max()[0], 100).reshape(-1, 1)

plt.scatter(X_sample, y_sample, alpha=0.2, s=10, label='데이터')
plt.plot(X_plot, linear_1d.predict(X_plot), 'g-', lw=2,
         label=f'선형 (R²={linear_1d.score(X_inc_test, y_inc_test):.2f})')
plt.plot(X_plot, poly2_1d.predict(X_plot), 'b-', lw=2,
         label=f'다항 deg=2 (R²={poly2_1d.score(X_inc_test, y_inc_test):.2f})')
plt.plot(X_plot, poly3_1d.predict(X_plot), 'r-', lw=2,
         label=f'다항 deg=3 (R²={poly3_1d.score(X_inc_test, y_inc_test):.2f})')

plt.xlabel('중위 소득 (MedInc)')
plt.ylabel('주택 가격 (단위: $100,000)')
plt.title('중위 소득 vs 주택 가격: 선형 vs 다항회귀')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('15_univariate_regression.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n단변량 회귀 시각화 저장: 15_univariate_regression.png")


# ============================================================
# Part 10: 특성별 영향력 분석
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 특성별 영향력 분석")
print("=" * 60)

# 각 특성의 회귀 계수 시각화
coef_df = pd.DataFrame({
    '특성': feature_columns,
    '계수': linear_model.coef_
})
coef_df['절대값'] = abs(coef_df['계수'])
coef_df = coef_df.sort_values('절대값', ascending=True)

plt.figure(figsize=(10, 5))
colors = ['red' if c < 0 else 'blue' for c in coef_df['계수']]
plt.barh(coef_df['특성'], coef_df['계수'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('회귀 계수')
plt.title('선형회귀 특성별 영향력\n(파랑=양의 영향, 빨강=음의 영향)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('15_feature_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n특성별 영향력 시각화 저장: 15_feature_coefficients.png")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 15차시 선형/다항회귀")
print("=" * 60)

print("""
1. 회귀 문제란?
   - 연속적인 숫자 예측
   - 분류: "~인가요?" vs 회귀: "얼마나?"
   - 평가: MSE, RMSE, MAE, R² 점수

2. California Housing 데이터셋
   - 20,640개 캘리포니아 주택 데이터
   - 8개 특성 (소득, 연식, 방 수 등)
   - 타겟: 중간 주택 가격 (단위: $100,000)

3. 선형회귀 (LinearRegression)
   - 수식: y = w1*x1 + w2*x2 + ... + b
   - 장점: 해석이 쉬움, 빠름
   - 계수(coef_): 각 특성의 영향력
   - 절편(intercept_): 기준값

4. 다항회귀 (PolynomialFeatures)
   - 특성의 거듭제곱 추가 (x², x³, ...)
   - Pipeline으로 구성:
     1) StandardScaler: 스케일링 (선택)
     2) PolynomialFeatures: 다항 특성 생성
     3) LinearRegression: 선형회귀
   - 곡선 관계 학습 가능

5. 차수(degree) 선택
   - 낮은 차수부터 시작 (2~3)
   - 학습/테스트 점수 비교
   - 과대적합 주의: 학습 높고 테스트 낮으면 위험

6. sklearn 사용법
   # 선형회귀
   model = LinearRegression()
   model.fit(X_train, y_train)
   model.predict(X_test)
   model.score(X_test, y_test)  # R²

   # 다항회귀
   poly_model = Pipeline([
       ('scaler', StandardScaler()),
       ('poly', PolynomialFeatures(degree=2)),
       ('linear', LinearRegression())
   ])
""")

print("\n다음 차시 예고: 16차시 - 모델 평가와 반복 검증")
