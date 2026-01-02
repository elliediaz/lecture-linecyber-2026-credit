# [6차시] 상관분석과 예측의 기초 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 실습 1: 데이터 준비
# ============================================================
print("=" * 50)
print("실습 1: 데이터 준비")
print("=" * 50)

np.random.seed(42)

# 온도 데이터 (섭씨)
temperature = np.array([75, 78, 80, 82, 85, 87, 90, 92, 95, 98])

# 불량률 (%, 온도와 양의 상관관계 + 노이즈)
defect_rate = 1.5 + 0.05 * temperature + np.random.normal(0, 0.2, 10)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '불량률': defect_rate
})

print(df)
print(f"\n데이터 크기: {df.shape}")

# ============================================================
# 실습 2: 산점도 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 2: 산점도 시각화")
print("=" * 50)

plt.figure(figsize=(10, 6))
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7, c='steelblue')

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률의 관계')
plt.grid(True, alpha=0.3)
plt.show()

print("점들이 오른쪽 위로 향하면 양의 상관관계입니다.")

# ============================================================
# 실습 3: 상관계수 계산
# ============================================================
print("\n" + "=" * 50)
print("실습 3: 상관계수 계산")
print("=" * 50)

# numpy로 계산
r_numpy = np.corrcoef(df['온도'], df['불량률'])[0, 1]
print(f"상관계수 (numpy): {r_numpy:.4f}")

# pandas로 계산
r_pandas = df['온도'].corr(df['불량률'])
print(f"상관계수 (pandas): {r_pandas:.4f}")

# 해석
if abs(r_numpy) > 0.7:
    strength = "강한"
elif abs(r_numpy) > 0.3:
    strength = "중간"
else:
    strength = "약한"
direction = "양의" if r_numpy > 0 else "음의"
print(f"\n해석: {strength} {direction} 상관관계")

# ============================================================
# 실습 4: 상관행렬
# ============================================================
print("\n" + "=" * 50)
print("실습 4: 상관행렬")
print("=" * 50)

# 추가 변수 생성
df['습도'] = 60 + np.random.normal(0, 5, 10)
df['생산량'] = 1200 - 5 * df['온도'] + np.random.normal(0, 20, 10)

# 상관행렬
corr_matrix = df.corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))

# ============================================================
# 실습 5: 상관행렬 히트맵
# ============================================================
print("\n" + "=" * 50)
print("실습 5: 상관행렬 히트맵")
print("=" * 50)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='상관계수')

# 레이블 추가
labels = corr_matrix.columns
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=12,
                color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

plt.title('변수 간 상관관계 히트맵')
plt.tight_layout()
plt.show()

# ============================================================
# 실습 6: 선형회귀 모델 학습
# ============================================================
print("\n" + "=" * 50)
print("실습 6: 선형회귀 모델 학습")
print("=" * 50)

# 데이터 준비 (2D 배열로 변환)
X = df[['온도']].values  # 독립변수 (2D)
y = df['불량률'].values   # 종속변수 (1D)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 결과 확인
print("=== 회귀 분석 결과 ===")
print(f"절편 (β₀): {model.intercept_:.4f}")
print(f"기울기 (β₁): {model.coef_[0]:.4f}")
print(f"\n회귀식: 불량률 = {model.intercept_:.2f} + {model.coef_[0]:.4f} × 온도")
print(f"\n해석: 온도가 1도 상승하면 불량률이 {model.coef_[0]:.4f}% 증가")

# ============================================================
# 실습 7: 회귀선 시각화
# ============================================================
print("\n" + "=" * 50)
print("실습 7: 회귀선 시각화")
print("=" * 50)

plt.figure(figsize=(10, 6))

# 원본 데이터
plt.scatter(df['온도'], df['불량률'], s=100, alpha=0.7, label='실제 데이터')

# 회귀선
X_line = np.linspace(70, 100, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='회귀선')

plt.xlabel('온도 (도)')
plt.ylabel('불량률 (%)')
plt.title('온도와 불량률: 선형회귀')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# 실습 8: 모델 평가
# ============================================================
print("\n" + "=" * 50)
print("실습 8: 모델 평가")
print("=" * 50)

# 예측값
y_pred = model.predict(X)

# R² (결정계수)
r2 = r2_score(y, y_pred)

# RMSE (평균제곱근오차)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("=== 모델 평가 ===")
print(f"R² (결정계수): {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print()
print(f"해석: 온도가 불량률 변동의 {r2*100:.1f}%를 설명합니다.")

if r2 >= 0.9:
    print("평가: 매우 좋은 모델입니다.")
elif r2 >= 0.7:
    print("평가: 좋은 모델입니다.")
elif r2 >= 0.5:
    print("평가: 보통 수준의 모델입니다.")
else:
    print("평가: 개선이 필요한 모델입니다.")

# ============================================================
# 실습 9: 새로운 데이터 예측
# ============================================================
print("\n" + "=" * 50)
print("실습 9: 새로운 데이터 예측")
print("=" * 50)

# 새로운 온도에서 불량률 예측
new_temps = np.array([[85], [90], [95], [100]])
predictions = model.predict(new_temps)

print("=== 불량률 예측 ===")
for temp, pred in zip(new_temps.flatten(), predictions):
    print(f"온도 {temp}도 → 예측 불량률: {pred:.2f}%")

# 목표 불량률을 위한 온도 역산
target_defect = 5.0
required_temp = (target_defect - model.intercept_) / model.coef_[0]
print(f"\n불량률 {target_defect}% 이하 유지를 위한 최대 온도: {required_temp:.1f}도")

# ============================================================
# 실습 10: 잔차 분석
# ============================================================
print("\n" + "=" * 50)
print("실습 10: 잔차 분석")
print("=" * 50)

# 잔차 계산
residuals = y - y_pred

plt.figure(figsize=(12, 4))

# 잔차 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.7, s=80)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.grid(True, alpha=0.3)

# 잔차 히스토그램
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=5, edgecolor='black', alpha=0.7)
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 분포')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"잔차 평균: {residuals.mean():.6f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.4f}")

# ============================================================
# 추가 실습: 다중 회귀 미리보기
# ============================================================
print("\n" + "=" * 50)
print("추가: 다중 회귀 미리보기")
print("=" * 50)

# 온도와 습도 두 변수로 불량률 예측
X_multi = df[['온도', '습도']].values
y_multi = df['불량률'].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("=== 다중 회귀 결과 ===")
print(f"절편: {model_multi.intercept_:.4f}")
print(f"온도 계수: {model_multi.coef_[0]:.4f}")
print(f"습도 계수: {model_multi.coef_[1]:.4f}")
print(f"\n회귀식: 불량률 = {model_multi.intercept_:.2f} + "
      f"{model_multi.coef_[0]:.4f}×온도 + {model_multi.coef_[1]:.4f}×습도")

r2_multi = model_multi.score(X_multi, y_multi)
print(f"\nR² (다중회귀): {r2_multi:.4f}")
print(f"R² (단순회귀): {r2:.4f}")
