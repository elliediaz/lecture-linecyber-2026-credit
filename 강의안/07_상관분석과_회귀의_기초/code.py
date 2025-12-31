"""
[7차시] 상관분석과 회귀의 기초 - 실습 코드

변수 간 관계 분석과 예측 모델의 기초를 실습합니다.

학습목표:
- 상관계수의 의미와 해석 방법 이해
- 단순선형회귀의 개념과 원리 이해
- sklearn으로 회귀 모델 구현
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("7차시: 상관분석과 회귀의 기초")
print("=" * 60)
print()


# =============================================================
# 1. 상관계수 계산
# =============================================================

print("=" * 60)
print("1. 상관계수 계산")
print("=" * 60)

# 샘플 데이터 생성
np.random.seed(42)
n = 100

# 온도와 불량률 (양의 상관)
temperature = np.random.normal(85, 5, n)
defect_rate = 0.01 + 0.003 * (temperature - 80) + np.random.normal(0, 0.005, n)
defect_rate = np.clip(defect_rate, 0, 1)

# 가동시간과 생산량 (양의 상관)
runtime = np.random.normal(8, 1, n)
production = 150 * runtime + np.random.normal(0, 50, n)

# 상관계수 계산
r_temp_defect = np.corrcoef(temperature, defect_rate)[0, 1]
r_runtime_prod = np.corrcoef(runtime, production)[0, 1]

print("[피어슨 상관계수]")
print(f"온도 - 불량률: r = {r_temp_defect:.4f}")
print(f"가동시간 - 생산량: r = {r_runtime_prod:.4f}")
print()

# 상관계수 해석
def interpret_correlation(r):
    r_abs = abs(r)
    if r_abs >= 0.7:
        strength = "강한"
    elif r_abs >= 0.3:
        strength = "중간"
    else:
        strength = "약한"
    direction = "양의" if r > 0 else "음의"
    return f"{strength} {direction} 상관"

print("[해석]")
print(f"온도 - 불량률: {interpret_correlation(r_temp_defect)}")
print(f"가동시간 - 생산량: {interpret_correlation(r_runtime_prod)}")
print()


# =============================================================
# 2. 상관계수 시각화
# =============================================================

print("=" * 60)
print("2. 상관계수 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 온도 vs 불량률
axes[0].scatter(temperature, defect_rate * 100, alpha=0.5, c='steelblue')
axes[0].set_xlabel('온도 (°C)')
axes[0].set_ylabel('불량률 (%)')
axes[0].set_title(f'온도 vs 불량률\nr = {r_temp_defect:.3f}')
axes[0].grid(True, alpha=0.3)

# 추세선 추가
z = np.polyfit(temperature, defect_rate * 100, 1)
p = np.poly1d(z)
temp_sorted = np.sort(temperature)
axes[0].plot(temp_sorted, p(temp_sorted), 'r--', linewidth=2, label='추세선')
axes[0].legend()

# 가동시간 vs 생산량
axes[1].scatter(runtime, production, alpha=0.5, c='coral')
axes[1].set_xlabel('가동시간 (시간)')
axes[1].set_ylabel('생산량 (개)')
axes[1].set_title(f'가동시간 vs 생산량\nr = {r_runtime_prod:.3f}')
axes[1].grid(True, alpha=0.3)

# 추세선 추가
z = np.polyfit(runtime, production, 1)
p = np.poly1d(z)
runtime_sorted = np.sort(runtime)
axes[1].plot(runtime_sorted, p(runtime_sorted), 'r--', linewidth=2, label='추세선')
axes[1].legend()

plt.tight_layout()
plt.savefig('01_correlation_scatter.png', dpi=150)
plt.show()

print("'01_correlation_scatter.png' 저장 완료")
print()


# =============================================================
# 3. 상관행렬 (여러 변수)
# =============================================================

print("=" * 60)
print("3. 상관행렬 (여러 변수)")
print("=" * 60)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '습도': np.random.normal(50, 10, n),
    '가동시간': runtime,
    '생산량': production,
    '불량률': defect_rate * 100
})

# 상관행렬
corr_matrix = df.corr()
print("[상관행렬]")
print(corr_matrix.round(3))
print()

# 히트맵 시각화
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, label='상관계수')

# 레이블 설정
labels = corr_matrix.columns
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.yticks(range(len(labels)), labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=10)

plt.title('변수 간 상관행렬')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150)
plt.show()

print("'02_correlation_heatmap.png' 저장 완료")
print()


# =============================================================
# 4. 단순선형회귀 - sklearn
# =============================================================

print("=" * 60)
print("4. 단순선형회귀 - sklearn")
print("=" * 60)

# 데이터 준비 (온도 → 불량률 예측)
X = temperature.reshape(-1, 1)  # 2D 배열로 변환
y = defect_rate * 100  # 퍼센트로 변환

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 결과 출력
print("[회귀 모델 결과]")
print(f"절편 (β₀): {model.intercept_:.4f}")
print(f"기울기 (β₁): {model.coef_[0]:.4f}")
print()
print(f"회귀식: 불량률 = {model.intercept_:.4f} + {model.coef_[0]:.4f} × 온도")
print()

# 해석
print("[해석]")
print(f"온도가 1°C 상승하면 불량률이 약 {model.coef_[0]:.3f}%p 증가")
print()

# 예측
test_temps = np.array([[80], [85], [90], [95]])
predictions = model.predict(test_temps)

print("[예측]")
for temp, pred in zip(test_temps.flatten(), predictions):
    print(f"온도 {temp}°C → 예측 불량률: {pred:.2f}%")
print()


# =============================================================
# 5. 회귀 모델 평가 (R²)
# =============================================================

print("=" * 60)
print("5. 회귀 모델 평가 (R²)")
print("=" * 60)

# 예측값
y_pred = model.predict(X)

# R² (결정계수)
r2 = r2_score(y, y_pred)
r2_from_model = model.score(X, y)

# MSE, RMSE
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"R² (결정계수): {r2:.4f}")
print(f"R² (model.score): {r2_from_model:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print()

print("[R² 해석]")
print(f"온도가 불량률 변동의 {r2*100:.1f}%를 설명함")
if r2 >= 0.7:
    print("→ 설명력이 높은 모델")
elif r2 >= 0.4:
    print("→ 중간 정도의 설명력")
else:
    print("→ 설명력이 낮음, 다른 변수 추가 필요")
print()


# =============================================================
# 6. 회귀 결과 시각화
# =============================================================

print("=" * 60)
print("6. 회귀 결과 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 회귀선 그래프
axes[0].scatter(X, y, alpha=0.5, c='steelblue', label='실제 데이터')
axes[0].plot(np.sort(X.flatten()), model.predict(np.sort(X, axis=0)),
             'r-', linewidth=2, label='회귀선')
axes[0].set_xlabel('온도 (°C)')
axes[0].set_ylabel('불량률 (%)')
axes[0].set_title(f'온도 vs 불량률 회귀분석\nR² = {r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 95% 신뢰구간 추가 (근사)
X_sorted = np.sort(X.flatten())
y_pred_sorted = model.predict(X_sorted.reshape(-1, 1))
se = np.std(y - y_pred)  # 잔차의 표준오차
ci = 1.96 * se
axes[0].fill_between(X_sorted, y_pred_sorted - ci, y_pred_sorted + ci,
                     alpha=0.2, color='red', label='95% 신뢰구간')

# 잔차 그래프
residuals = y - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, c='steelblue')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('예측값')
axes[1].set_ylabel('잔차 (실제 - 예측)')
axes[1].set_title('잔차 그래프')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_regression_results.png', dpi=150)
plt.show()

print("'03_regression_results.png' 저장 완료")
print()


# =============================================================
# 7. 잔차 분석
# =============================================================

print("=" * 60)
print("7. 잔차 분석")
print("=" * 60)

print("[잔차 기본 통계]")
print(f"잔차 평균: {np.mean(residuals):.6f} (0에 가까워야 함)")
print(f"잔차 표준편차: {np.std(residuals):.4f}")
print()

# 잔차의 정규성 검정
stat, p_value = stats.shapiro(residuals)
print(f"[정규성 검정 (Shapiro-Wilk)]")
print(f"통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value > 0.05:
    print("→ 잔차가 정규분포를 따른다고 볼 수 있음")
else:
    print("→ 잔차가 정규분포를 따르지 않을 수 있음")
print()

# 잔차 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 히스토그램
axes[0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('잔차')
axes[0].set_ylabel('빈도')
axes[0].set_title('잔차 분포')

# Q-Q 플롯
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (정규성 확인)')

plt.tight_layout()
plt.savefig('04_residual_analysis.png', dpi=150)
plt.show()

print("'04_residual_analysis.png' 저장 완료")
print()


# =============================================================
# 8. 다중회귀 미리보기
# =============================================================

print("=" * 60)
print("8. 다중회귀 미리보기")
print("=" * 60)

# 여러 변수로 예측
X_multi = df[['온도', '습도', '가동시간']].values
y_multi = df['불량률'].values

# 모델 학습
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("[다중회귀 결과]")
print(f"절편: {model_multi.intercept_:.4f}")
print()
print("계수:")
for name, coef in zip(['온도', '습도', '가동시간'], model_multi.coef_):
    print(f"  {name}: {coef:.4f}")
print()

# R² 비교
r2_multi = model_multi.score(X_multi, y_multi)
print(f"[모델 비교]")
print(f"단순회귀 (온도만) R²: {r2:.4f}")
print(f"다중회귀 (온도+습도+가동시간) R²: {r2_multi:.4f}")
print(f"→ 변수 추가로 설명력 {(r2_multi-r2)*100:.1f}%p 향상")
print()


# =============================================================
# 9. 상관 vs 인과 예시
# =============================================================

print("=" * 60)
print("9. 상관 vs 인과 주의사항")
print("=" * 60)

# 허위 상관 예시 생성
np.random.seed(123)
months = np.arange(1, 13)
# 기온 (계절 변화)
avg_temp = 10 + 15 * np.sin((months - 4) * np.pi / 6)
# 아이스크림 판매 (기온에 비례)
ice_cream = 100 + 8 * avg_temp + np.random.normal(0, 10, 12)
# 익사 사고 (기온에 비례)
drowning = 5 + 0.3 * avg_temp + np.random.normal(0, 1, 12)

r_spurious = np.corrcoef(ice_cream, drowning)[0, 1]

print("[허위 상관 예시]")
print(f"아이스크림 판매량 - 익사 사고 수 상관계수: r = {r_spurious:.3f}")
print()
print("분석:")
print("- 높은 양의 상관관계가 있음")
print("- 하지만 아이스크림이 익사를 유발하는 것이 아님!")
print("- 숨겨진 변수 '기온'이 둘 다에 영향")
print()
print("교훈:")
print("- 상관관계 ≠ 인과관계")
print("- 상관분석은 관계의 '존재'만 알려줌")
print("- 인과관계 확인에는 실험 설계나 도메인 지식 필요")
print()

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 기온 vs 아이스크림
axes[0].scatter(avg_temp, ice_cream, c='coral')
axes[0].set_xlabel('평균 기온 (°C)')
axes[0].set_ylabel('아이스크림 판매량')
axes[0].set_title(f'기온 vs 아이스크림\nr = {np.corrcoef(avg_temp, ice_cream)[0,1]:.3f}')

# 기온 vs 익사
axes[1].scatter(avg_temp, drowning, c='steelblue')
axes[1].set_xlabel('평균 기온 (°C)')
axes[1].set_ylabel('익사 사고 수')
axes[1].set_title(f'기온 vs 익사 사고\nr = {np.corrcoef(avg_temp, drowning)[0,1]:.3f}')

# 아이스크림 vs 익사 (허위 상관)
axes[2].scatter(ice_cream, drowning, c='purple')
axes[2].set_xlabel('아이스크림 판매량')
axes[2].set_ylabel('익사 사고 수')
axes[2].set_title(f'아이스크림 vs 익사 (허위 상관!)\nr = {r_spurious:.3f}')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_spurious_correlation.png', dpi=150)
plt.show()

print("'05_spurious_correlation.png' 저장 완료")
print()

print("=" * 60)
print("7차시 실습 완료!")
print("=" * 60)
