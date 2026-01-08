"""
[8차시] 상관분석과 예측의 기초 - 실습 코드

학습 목표:
1. 상관계수의 의미와 해석 방법을 이해한다
2. 단순선형회귀의 개념과 원리를 이해한다
3. sklearn으로 예측 모델을 구현한다

실습 환경:
- Python 3.8+
- NumPy, Pandas, Matplotlib, scikit-learn

데이터:
- MPG 데이터셋 (자동차 연비 데이터)
- 출처: seaborn-data repository
- 변수: mpg(연비), cylinders(실린더), displacement(배기량),
        horsepower(마력), weight(무게), acceleration(가속력) 등
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("        8차시: 상관분석과 예측의 기초 실습")
print("=" * 60)


# ============================================================
# Part 1: 상관계수의 의미와 해석 방법
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 상관계수의 의미와 해석 방법")
print("=" * 60)


# 실습 1-1: 데이터 준비 (MPG 데이터셋 로드)
print("\n[실습 1-1] 데이터 준비 - MPG 데이터셋")
print("-" * 40)

# MPG 데이터셋 로드 (자동차 연비 데이터)
# 출처: https://github.com/mwaskom/seaborn-data
try:
    mpg_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
    df_full = pd.read_csv(mpg_url)
    print("MPG 데이터셋 로드 성공!")
except Exception as e:
    print(f"온라인 로드 실패: {e}")
    print("대체 데이터를 생성합니다...")
    # 대체 데이터 생성 (온라인 접속 불가 시)
    np.random.seed(42)
    n = 50
    df_full = pd.DataFrame({
        'mpg': 20 + np.random.normal(0, 5, n),
        'weight': 3000 + np.random.normal(0, 500, n),
        'horsepower': 100 + np.random.normal(0, 30, n),
        'displacement': 200 + np.random.normal(0, 80, n),
        'cylinders': np.random.choice([4, 6, 8], n),
        'acceleration': 15 + np.random.normal(0, 2, n)
    })

# 결측치 제거 및 수치형 변수만 선택
df = df_full[['mpg', 'weight', 'horsepower', 'displacement',
              'cylinders', 'acceleration']].dropna()

print(f"\n데이터 형태: {df.shape}")
print(f"\n처음 10행:")
print(df.head(10))

# 데이터 설명
print("\n=== 변수 설명 ===")
print("mpg: 연비 (miles per gallon) - 높을수록 연비 좋음")
print("weight: 자동차 무게 (파운드)")
print("horsepower: 마력 - 엔진 출력")
print("displacement: 배기량 (세제곱인치)")
print("cylinders: 실린더 수 (4, 6, 8개)")
print("acceleration: 가속력 (0-60mph 도달 시간, 초)")

print(f"\n기술 통계:")
print(df.describe().round(2))


# 실습 1-2: 산점도 시각화 (무게 vs 연비)
print("\n[실습 1-2] 산점도 시각화 - 무게와 연비 관계")
print("-" * 40)

plt.figure(figsize=(10, 6))
plt.scatter(df['weight'], df['mpg'], s=100, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=1.5)

plt.xlabel('Weight (lbs)', fontsize=12)
plt.ylabel('MPG (miles per gallon)', fontsize=12)
plt.title('Weight vs MPG Relationship', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('01_scatter_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("그래프가 01_scatter_plot.png로 저장되었습니다.")
print("-> 점들이 오른쪽 아래로 향함: 음의 상관관계 예상 (무게 증가 -> 연비 감소)")


# 실습 1-3: 상관계수 계산
print("\n[실습 1-3] 상관계수 계산")
print("-" * 40)

# 방법 1: numpy
r_numpy = np.corrcoef(df['weight'], df['mpg'])[0, 1]

# 방법 2: pandas
r_pandas = df['weight'].corr(df['mpg'])

print(f"상관계수 (numpy): {r_numpy:.4f}")
print(f"상관계수 (pandas): {r_pandas:.4f}")

# 해석 함수
def interpret_correlation(r):
    """상관계수 해석"""
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "강한"
    elif abs_r >= 0.5:
        strength = "중간"
    elif abs_r >= 0.3:
        strength = "보통"
    else:
        strength = "약한"

    direction = "양의" if r > 0 else "음의" if r < 0 else "없는"
    return f"{strength} {direction} 상관관계"


print(f"\n해석: {interpret_correlation(r_numpy)}")
print(f"       (|r| = {abs(r_numpy):.2f})")
print(f"\n의미: 자동차 무게가 무거울수록 연비(mpg)가 낮아지는 강한 경향")


# 실습 1-4: 다양한 상관관계 예시 (실제 데이터 기반)
print("\n[실습 1-4] 다양한 상관관계 예시")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 강한 음의 상관 (무게 vs 연비)
r1 = df['weight'].corr(df['mpg'])
axes[0].scatter(df['weight'], df['mpg'], alpha=0.7)
axes[0].set_title(f'Weight vs MPG (r = {r1:.2f})')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('MPG')

# 강한 양의 상관 (무게 vs 배기량)
r2 = df['weight'].corr(df['displacement'])
axes[1].scatter(df['weight'], df['displacement'], alpha=0.7, color='coral')
axes[1].set_title(f'Weight vs Displacement (r = {r2:.2f})')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('Displacement')

# 약한 상관 (가속력 vs 연비)
r3 = df['acceleration'].corr(df['mpg'])
axes[2].scatter(df['acceleration'], df['mpg'], alpha=0.7, color='gray')
axes[2].set_title(f'Acceleration vs MPG (r = {r3:.2f})')
axes[2].set_xlabel('Acceleration')
axes[2].set_ylabel('MPG')

plt.tight_layout()
plt.savefig('02_correlation_types.png', dpi=150, bbox_inches='tight')
plt.show()

print("그래프가 02_correlation_types.png로 저장되었습니다.")
print(f"\n상관계수 요약:")
print(f"  - 무게 vs 연비: {r1:.2f} ({interpret_correlation(r1)})")
print(f"  - 무게 vs 배기량: {r2:.2f} ({interpret_correlation(r2)})")
print(f"  - 가속력 vs 연비: {r3:.2f} ({interpret_correlation(r3)})")


# 실습 1-5: 상관행렬
print("\n[실습 1-5] 상관행렬")
print("-" * 40)

# 상관행렬
corr_matrix = df.corr()
print("=== 상관행렬 ===")
print(corr_matrix.round(3))


# 실습 1-6: 상관행렬 히트맵
print("\n[실습 1-6] 상관행렬 히트맵")
print("-" * 40)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# 컬러바
cbar = ax.figure.colorbar(im, ax=ax)
cbar.set_label('Correlation Coefficient', fontsize=12)

# 레이블
labels = corr_matrix.columns
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

# 값 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        value = corr_matrix.iloc[i, j]
        color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                fontsize=10, color=color)

ax.set_title('MPG Dataset Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("그래프가 03_correlation_heatmap.png로 저장되었습니다.")

print("\n=== 주요 상관관계 해석 ===")
print("- mpg와 weight: 강한 음의 상관 (무거울수록 연비 나쁨)")
print("- mpg와 horsepower: 강한 음의 상관 (마력 높을수록 연비 나쁨)")
print("- weight와 displacement: 강한 양의 상관 (배기량 클수록 무거움)")


# ============================================================
# Part 2: 단순선형회귀의 개념과 원리
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 단순선형회귀의 개념과 원리")
print("=" * 60)


# 실습 2-1: 최소제곱법 직접 구현
print("\n[실습 2-1] 최소제곱법 직접 구현 - 무게로 연비 예측")
print("-" * 40)

# 데이터 (무게 -> 연비 예측)
X = df['weight'].values
Y = df['mpg'].values

# 최소제곱법 공식
x_mean = X.mean()
y_mean = Y.mean()

# 기울기 (beta1)
numerator = np.sum((X - x_mean) * (Y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
beta1 = numerator / denominator

# 절편 (beta0)
beta0 = y_mean - beta1 * x_mean

print("=== 최소제곱법 직접 계산 ===")
print(f"절편 (beta0): {beta0:.4f}")
print(f"기울기 (beta1): {beta1:.6f}")
print(f"\n회귀식: MPG = {beta0:.2f} + ({beta1:.6f}) * Weight")
print(f"해석: 무게가 1파운드 증가하면 연비가 {abs(beta1):.6f} mpg 감소")
print(f"      무게가 1000파운드 증가하면 연비가 약 {abs(beta1*1000):.2f} mpg 감소")


# 실습 2-2: 잔차와 SSE 계산
print("\n[실습 2-2] 잔차와 오차 계산")
print("-" * 40)

# 예측값
Y_pred_manual = beta0 + beta1 * X

# 잔차
residuals_manual = Y - Y_pred_manual

# 오차 제곱합 (SSE)
SSE = np.sum(residuals_manual ** 2)

# 총 제곱합 (SST)
SST = np.sum((Y - y_mean) ** 2)

# R^2
R2_manual = 1 - SSE / SST

print("=== 오차 분석 ===")
print(f"SSE (잔차제곱합): {SSE:.4f}")
print(f"SST (총제곱합): {SST:.4f}")
print(f"R^2 (결정계수): {R2_manual:.4f}")
print(f"\n해석: 무게가 연비 변동의 {R2_manual*100:.1f}%를 설명")


# 실습 2-3: 잔차 출력 (처음 10개)
print("\n[실습 2-3] 잔차 상세 (처음 10개 데이터)")
print("-" * 40)

print("=== 각 데이터의 잔차 ===")
for i in range(min(10, len(X))):
    print(f"무게 {X[i]:.0f}lbs: 실제 {Y[i]:.1f}mpg, 예측 {Y_pred_manual[i]:.1f}mpg, 잔차 {residuals_manual[i]:+.1f}")

print(f"\n잔차 합계: {residuals_manual.sum():.6f} (0에 가까워야 함)")
print(f"잔차 평균: {residuals_manual.mean():.6f}")


# ============================================================
# Part 3: sklearn으로 예측 모델 구현
# ============================================================
print("\n" + "=" * 60)
print("Part 3: sklearn으로 예측 모델 구현")
print("=" * 60)


# 실습 3-1: sklearn LinearRegression
print("\n[실습 3-1] sklearn LinearRegression")
print("-" * 40)

# 데이터 준비 (2D 배열로 변환 필수!)
X_2d = df[['weight']].values  # shape: (n, 1)
y = df['mpg'].values           # shape: (n,)

print(f"X shape: {X_2d.shape}")
print(f"y shape: {y.shape}")

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_2d, y)

print("\n=== sklearn 회귀 분석 결과 ===")
print(f"절편 (intercept_): {model.intercept_:.4f}")
print(f"기울기 (coef_): {model.coef_[0]:.6f}")
print(f"\n회귀식: MPG = {model.intercept_:.2f} + ({model.coef_[0]:.6f}) * Weight")

# 직접 계산과 비교
print("\n=== 직접 계산과 비교 ===")
print(f"절편 차이: {abs(model.intercept_ - beta0):.10f}")
print(f"기울기 차이: {abs(model.coef_[0] - beta1):.10f}")
print("-> sklearn과 직접 계산 결과가 동일함")


# 실습 3-2: 예측
print("\n[실습 3-2] 예측")
print("-" * 40)

# 기존 데이터 예측
y_pred = model.predict(X_2d)

# 새로운 데이터 예측 (다양한 무게)
new_weights = np.array([[2000], [2500], [3000], [3500], [4000], [4500]])
new_predictions = model.predict(new_weights)

print("=== 새로운 무게 예측 ===")
for weight, pred in zip(new_weights.flatten(), new_predictions):
    print(f"무게 {weight:,}lbs -> 예측 연비: {pred:.1f} mpg")


# 실습 3-3: 모델 평가
print("\n[실습 3-3] 모델 평가")
print("-" * 40)

# 평가 지표 계산
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print("=== 모델 평가 지표 ===")
print(f"R^2 (결정계수): {r2:.4f}")
print(f"MSE (평균제곱오차): {mse:.4f}")
print(f"RMSE (평균제곱근오차): {rmse:.4f}")
print(f"MAE (평균절대오차): {mae:.4f}")
print()

# R^2 해석
if r2 >= 0.9:
    quality = "매우 좋음"
elif r2 >= 0.7:
    quality = "좋음"
elif r2 >= 0.5:
    quality = "보통"
else:
    quality = "개선 필요"

print(f"모델 품질: {quality}")
print(f"해석: 무게가 연비 변동의 {r2*100:.1f}%를 설명합니다.")
print(f"      평균적으로 약 {mae:.1f} mpg 오차로 예측합니다.")


# 실습 3-4: 회귀선 시각화
print("\n[실습 3-4] 회귀선 시각화")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 회귀선
X_line = np.linspace(df['weight'].min()-100, df['weight'].max()+100, 100).reshape(-1, 1)
y_line = model.predict(X_line)

axes[0].scatter(df['weight'], df['mpg'], s=100, alpha=0.7,
                label='Actual Data', color='steelblue', edgecolor='black')
axes[0].plot(X_line, y_line, 'r-', linewidth=2, label='Regression Line')

# 회귀식 표시
eq_text = f'MPG = {model.intercept_:.2f} + ({model.coef_[0]:.6f}) * Weight\nR^2 = {r2:.4f}'
axes[0].text(df['weight'].max()-800, df['mpg'].max()-2, eq_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_xlabel('Weight (lbs)', fontsize=12)
axes[0].set_ylabel('MPG', fontsize=12)
axes[0].set_title('Weight vs MPG: Linear Regression', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 실제 vs 예측
axes[1].scatter(y, y_pred, s=100, alpha=0.7, color='green', edgecolor='black')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2,
             label='Perfect Prediction (y=x)')
axes[1].set_xlabel('Actual MPG', fontsize=12)
axes[1].set_ylabel('Predicted MPG', fontsize=12)
axes[1].set_title('Actual vs Predicted', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_regression_line.png', dpi=150, bbox_inches='tight')
plt.show()

print("그래프가 04_regression_line.png로 저장되었습니다.")


# 실습 3-5: 잔차 분석
print("\n[실습 3-5] 잔차 분석")
print("-" * 40)

residuals = y - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 잔차 플롯
axes[0].scatter(y_pred, residuals, s=80, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted MPG', fontsize=12)
axes[0].set_ylabel('Residual', fontsize=12)
axes[0].set_title('Residual Plot\n(No pattern = Good)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 잔차 히스토그램
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution\n(Should be normal)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# 잔차 Q-Q 플롯 (간단 버전)
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[2].scatter(osm, osr, s=80, alpha=0.7, color='steelblue', edgecolor='black')
axes[2].plot(osm, slope*osm + intercept, 'r-', linewidth=2)
axes[2].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[2].set_ylabel('Sample Quantiles', fontsize=12)
axes[2].set_title('Q-Q Plot\n(Should follow the line)', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("그래프가 05_residual_analysis.png로 저장되었습니다.")

print(f"\n=== 잔차 통계 ===")
print(f"잔차 평균: {residuals.mean():.6f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.4f}")
print(f"잔차 최소: {residuals.min():.4f}")
print(f"잔차 최대: {residuals.max():.4f}")


# 실습 3-6: 역 예측 (목표 연비를 위한 무게)
print("\n[실습 3-6] 역 예측")
print("-" * 40)

def predict_weight_for_target_mpg(target_mpg, model):
    """목표 연비를 위한 최대 무게 계산"""
    # Y = beta0 + beta1*X -> X = (Y - beta0) / beta1
    required_weight = (target_mpg - model.intercept_) / model.coef_[0]
    return required_weight


print("=== 목표 연비를 위한 최대 무게 계산 ===")
for target in [20.0, 25.0, 30.0, 35.0, 40.0]:
    weight = predict_weight_for_target_mpg(target, model)
    print(f"목표 연비 {target} mpg -> 무게 {weight:.0f} lbs 이하 유지")


# 실습 3-7: 다중선형회귀 미리보기
print("\n[실습 3-7] 다중선형회귀 미리보기")
print("-" * 40)

# 여러 독립변수 사용
X_multi = df[['weight', 'horsepower', 'displacement']].values
y_multi = df['mpg'].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("=== 다중선형회귀 결과 ===")
print(f"절편: {model_multi.intercept_:.4f}")
print("\n각 변수의 기울기 (영향력):")
for name, coef in zip(['weight', 'horsepower', 'displacement'], model_multi.coef_):
    print(f"  {name}: {coef:+.6f}")

# 다중회귀 R^2
y_pred_multi = model_multi.predict(X_multi)
r2_multi = r2_score(y_multi, y_pred_multi)
print(f"\nR^2 (다중회귀): {r2_multi:.4f}")
print(f"R^2 (단순회귀): {r2:.4f}")
print(f"R^2 개선: {r2_multi - r2:.4f}")
print("\n-> 변수 추가로 설명력 향상!")


# ============================================================
# 종합 분석 리포트
# ============================================================
print("\n" + "=" * 60)
print("종합 분석 리포트")
print("=" * 60)

print("""
=== 분석 요약 ===

1. 상관분석 결과
   - 무게와 연비의 상관계수: {:.4f}
   - 해석: {}

2. 선형회귀 모델
   - 회귀식: MPG = {:.2f} + ({:.6f}) * Weight
   - 해석: 무게 1000lbs 증가 -> 연비 {:.2f} mpg 감소

3. 모델 평가
   - R^2 (결정계수): {:.4f}
   - RMSE: {:.4f}
   - 모델 품질: {}

4. 실무 활용
   - 연비 30 mpg 달성을 위한 최대 무게: {:.0f} lbs
   - 권장: 차량 경량화를 통한 연비 개선

5. 주의사항
   - 상관관계 =/= 인과관계
   - 다른 변수(마력, 배기량 등)의 영향도 고려 필요
   - 모델의 예측 범위 외 데이터는 신뢰도 낮음
""".format(
    r_numpy,
    interpret_correlation(r_numpy),
    model.intercept_,
    model.coef_[0],
    abs(model.coef_[0]) * 1000,
    r2,
    rmse,
    quality,
    predict_weight_for_target_mpg(30, model)
))


# ============================================================
# 연습 문제
# ============================================================
print("\n" + "=" * 60)
print("연습 문제")
print("=" * 60)

print("""
[연습 1] MPG 데이터에서 horsepower와 mpg의 상관계수를 계산하고 해석하세요.

[연습 2] horsepower로 mpg를 예측하는 선형회귀 모델을 학습하고 회귀식을 출력하세요.

[연습 3] horsepower가 200일 때 예상 mpg를 예측하세요.

[연습 4] mpg를 25 이상 달성하려면 horsepower가 얼마 이하여야 하는지 계산하세요.
""")


# 연습 1 정답
print("\n[연습 1 정답]")
r_hp_mpg = df['horsepower'].corr(df['mpg'])
print(f"상관계수: {r_hp_mpg:.4f}")
print(f"해석: {interpret_correlation(r_hp_mpg)}")

# 연습 2 정답
print("\n[연습 2 정답]")
X_hp = df[['horsepower']].values
y_mpg = df['mpg'].values
model_hp = LinearRegression()
model_hp.fit(X_hp, y_mpg)
print(f"회귀식: MPG = {model_hp.intercept_:.2f} + ({model_hp.coef_[0]:.4f}) * Horsepower")
print(f"해석: 마력 1 증가 -> 연비 {abs(model_hp.coef_[0]):.4f} mpg 감소")

# 연습 3 정답
print("\n[연습 3 정답]")
pred_200hp = model_hp.predict([[200]])[0]
print(f"Horsepower 200 -> 예상 MPG: {pred_200hp:.2f}")

# 연습 4 정답
print("\n[연습 4 정답]")
target_mpg = 25
required_hp = (target_mpg - model_hp.intercept_) / model_hp.coef_[0]
print(f"MPG {target_mpg} 이상을 위한 최대 Horsepower: {required_hp:.1f}")


print("\n" + "=" * 60)
print("8차시 실습을 완료했습니다!")
print("다음 시간: 9차시 - 제조 데이터 전처리 (1)")
print("=" * 60)
