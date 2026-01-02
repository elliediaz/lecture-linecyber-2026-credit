# [13차시] 예측 모델: 선형회귀와 다항회귀 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("13차시: 예측 모델 - 선형회귀와 다항회귀")
print("숫자를 예측하는 회귀 모델을 배웁니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 제조 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 300

# 특성 생성
temperature = np.random.normal(85, 5, n_samples)  # 온도
humidity = np.random.normal(50, 10, n_samples)    # 습도
speed = np.random.normal(100, 15, n_samples)      # 속도

# 생산량 (선형 관계 + 노이즈)
production = (10 * temperature + 3 * humidity + 2 * speed
              + np.random.normal(0, 20, n_samples))

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '생산량': production
})

print("데이터 샘플:")
print(df.head(10))
print(f"\n데이터 크기: {df.shape}")
print(f"생산량 평균: {df['생산량'].mean():.1f}개")
print(f"생산량 범위: {df['생산량'].min():.1f} ~ {df['생산량'].max():.1f}개")
print()


# ============================================================
# 실습 2: 데이터 준비
# ============================================================
print("=" * 50)
print("실습 2: 데이터 준비")
print("=" * 50)

# 특성(X)과 타겟(y) 분리
X = df[['온도', '습도', '속도']]
y = df['생산량']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print()


# ============================================================
# 실습 3: 선형회귀 학습
# ============================================================
print("=" * 50)
print("실습 3: 선형회귀 학습")
print("=" * 50)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)
print("▶ model.fit(X_train, y_train) - 학습 완료!")

# 계수 확인
print(f"\n기울기 (coef_):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"\n절편 (intercept_): {model.intercept_:.3f}")

# 해석
print("\n★ 해석:")
print(f"  온도가 1도 올라가면 생산량 {model.coef_[0]:.1f}개 증가")
print(f"  습도가 1% 올라가면 생산량 {model.coef_[1]:.1f}개 증가")
print(f"  속도가 1 올라가면 생산량 {model.coef_[2]:.1f}개 증가")
print()


# ============================================================
# 실습 4: 모델 평가
# ============================================================
print("=" * 50)
print("실습 4: 모델 평가")
print("=" * 50)

# 예측
y_pred = model.predict(X_test)

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("평가 지표:")
print(f"  MSE: {mse:.2f}")
print(f"  RMSE: {rmse:.2f} (평균 {rmse:.1f}개 오차)")
print(f"  R²: {r2:.4f} ({r2*100:.1f}% 설명력)")

# 해석
if r2 > 0.9:
    print("\n★ R² > 0.9: 매우 좋은 모델!")
elif r2 > 0.7:
    print("\n△ R² > 0.7: 괜찮은 모델")
else:
    print("\n⚠️ R² < 0.7: 개선 필요")
print()


# ============================================================
# 실습 5: 실제 vs 예측 비교
# ============================================================
print("=" * 50)
print("실습 5: 실제 vs 예측 비교")
print("=" * 50)

# 처음 10개 비교
comparison = pd.DataFrame({
    '실제 생산량': y_test[:10].values,
    '예측 생산량': y_pred[:10],
    '오차': y_test[:10].values - y_pred[:10]
})
print(comparison.round(1).to_string(index=False))

# 시각화
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', label='완벽한 예측')
ax.set_xlabel('실제 생산량')
ax.set_ylabel('예측 생산량')
ax.set_title(f'실제 vs 예측 (R² = {r2:.3f})')
ax.legend()
plt.tight_layout()
plt.show()
print()


# ============================================================
# 실습 6: 다항회귀
# ============================================================
print("=" * 50)
print("실습 6: 다항회귀")
print("=" * 50)

# Pipeline으로 다항회귀 구성
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_pipe.fit(X_train, y_train)
y_pred_poly = poly_pipe.predict(X_test)

# 평가
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print("다항회귀 (degree=2) 결과:")
print(f"  R²: {r2_poly:.4f}")
print(f"  RMSE: {rmse_poly:.2f}")
print()


# ============================================================
# 실습 7: 선형 vs 다항 비교
# ============================================================
print("=" * 50)
print("실습 7: 선형 vs 다항 비교")
print("=" * 50)

print("성능 비교:")
print("-" * 50)
print(f"{'모델':<20} {'R²':<15} {'RMSE':<15}")
print("-" * 50)
print(f"{'선형회귀':<20} {r2:<15.4f} {rmse:<15.2f}")
print(f"{'다항회귀 (deg=2)':<20} {r2_poly:<15.4f} {rmse_poly:<15.2f}")
print("-" * 50)
print()


# ============================================================
# 실습 8: degree 실험
# ============================================================
print("=" * 50)
print("실습 8: degree에 따른 성능 변화")
print("=" * 50)

degrees = [1, 2, 3, 4, 5]
results = []

for degree in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)

    results.append({
        'degree': degree,
        '학습 R²': train_r2,
        '테스트 R²': test_r2,
        '차이': train_r2 - test_r2
    })

    print(f"degree={degree}: 학습 R²={train_r2:.4f}, 테스트 R²={test_r2:.4f}")

print("\n★ degree가 높아지면 학습 R²는 증가하지만,")
print("   테스트 R²는 어느 순간 감소할 수 있음 (과대적합)")
print()


# ============================================================
# 실습 9: 트리 기반 회귀
# ============================================================
print("=" * 50)
print("실습 9: 트리 기반 회귀")
print("=" * 50)

# 랜덤포레스트 회귀
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

rf_r2 = rf_reg.score(X_test, y_test)
rf_pred = rf_reg.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("랜덤포레스트 회귀 결과:")
print(f"  R²: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.2f}")

# 모든 모델 비교
print("\n모델 비교:")
print("-" * 50)
print(f"{'모델':<25} {'R²':<15} {'RMSE':<15}")
print("-" * 50)
print(f"{'선형회귀':<25} {r2:<15.4f} {rmse:<15.2f}")
print(f"{'다항회귀 (deg=2)':<25} {r2_poly:<15.4f} {rmse_poly:<15.2f}")
print(f"{'랜덤포레스트 회귀':<25} {rf_r2:<15.4f} {rf_rmse:<15.2f}")
print("-" * 50)
print()


# ============================================================
# 실습 10: 새 데이터 예측
# ============================================================
print("=" * 50)
print("실습 10: 새 데이터 예측")
print("=" * 50)

# 새 데이터
test_conditions = [
    [80, 45, 95],    # 온도 낮음
    [85, 50, 100],   # 평균 조건
    [90, 55, 105],   # 온도 높음
    [88, 52, 110],   # 속도 높음
]

print("예측 결과:")
print("-" * 60)
for condition in test_conditions:
    pred = model.predict([condition])[0]
    print(f"온도={condition[0]:5.1f}, 습도={condition[1]:5.1f}, 속도={condition[2]:5.1f}")
    print(f"   → 예측 생산량: {pred:.0f}개")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│               선형회귀 & 다항회귀 핵심 정리              │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 회귀 = 숫자 예측                                    │
│     분류: 범주 예측 (정상/불량)                         │
│     회귀: 숫자 예측 (1,247개)                          │
│                                                        │
│  ▶ 선형회귀                                            │
│     model = LinearRegression()                        │
│     model.coef_: 각 특성의 기울기                      │
│     model.intercept_: 절편                            │
│                                                        │
│  ▶ 다항회귀                                            │
│     PolynomialFeatures(degree=2) + LinearRegression   │
│     곡선 관계 표현 가능                                │
│     degree=2~3 권장 (과대적합 주의)                    │
│                                                        │
│  ▶ 평가 지표                                           │
│     MSE: 오차² 평균                                    │
│     RMSE: √MSE (원래 단위로 해석)                      │
│     R²: 0~1, 1에 가까울수록 좋음                       │
│                                                        │
│  ★ 먼저 LinearRegression으로 시작하세요!               │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 모델 평가와 반복 검증 (교차검증)
""")

print("=" * 60)
print("13차시 실습 완료!")
print("숫자 예측의 기초를 다졌습니다!")
print("=" * 60)
