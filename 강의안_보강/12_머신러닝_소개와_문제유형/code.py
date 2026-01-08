"""
[12차시] 머신러닝 소개와 문제 유형 - 실습 코드
=============================================

학습 목표:
1. 머신러닝의 개념을 설명한다
2. 지도학습과 비지도학습을 구분한다
3. 분류와 회귀 문제를 구분한다

실습 내용:
- 실제 공개 데이터셋 활용 (Iris, California Housing)
- sklearn 기본 패턴 (fit, predict, score)
- 분류 모델 실습 (DecisionTreeClassifier)
- 회귀 모델 실습 (LinearRegression)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.datasets import load_iris, fetch_california_housing

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Part 1: 전통적 프로그래밍 vs 머신러닝 비교
# ============================================================

print("=" * 60)
print("Part 1: 전통적 프로그래밍 vs 머신러닝")
print("=" * 60)

# 전통적 방식: 규칙 기반 (꽃 분류 예시)
def traditional_predict(sepal_length, sepal_width, petal_length, petal_width):
    """전통적 방식: 사람이 직접 규칙 작성"""
    if petal_length < 2.5:
        return "setosa"
    elif petal_width < 1.8:
        return "versicolor"
    else:
        return "virginica"

# 테스트
test_cases = [
    (5.0, 3.5, 1.5, 0.2),  # 예상: setosa
    (6.0, 2.8, 4.5, 1.5),  # 예상: versicolor
    (7.0, 3.0, 6.0, 2.5),  # 예상: virginica
]

print("\n[전통적 방식 - 규칙 기반]")
for sl, sw, pl, pw in test_cases:
    result = traditional_predict(sl, sw, pl, pw)
    print(f"  꽃받침={sl}, 꽃잎={pl} → {result}")

print("\n문제점:")
print("  - 규칙이 많아지면 관리 어려움")
print("  - 새로운 패턴 발견 어려움")
print("  - 미묘한 경계값 설정 어려움")


# ============================================================
# Part 2: 실제 공개 데이터셋 불러오기
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 실제 공개 데이터셋 불러오기")
print("=" * 60)

# 분류용 데이터: Iris 데이터셋
print("\n[Iris 데이터셋 로딩 중...]")
try:
    iris = load_iris()
    df_clf = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_clf['target'] = iris.target
    df_clf['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Iris 데이터셋 로딩 완료!")
except Exception as e:
    print(f"Iris 데이터셋 로딩 실패: {e}")
    raise

print(f"\n[Iris 데이터셋 정보]")
print(f"데이터 크기: {df_clf.shape}")
print(f"특성 이름: {iris.feature_names}")
print(f"클래스: {list(iris.target_names)}")
print(f"\n처음 5행:")
print(df_clf.head())

# 회귀용 데이터: California Housing 데이터셋
print("\n[California Housing 데이터셋 로딩 중...]")
try:
    housing = fetch_california_housing()
    df_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
    df_reg['MedHouseVal'] = housing.target  # 중간 주택 가격 (단위: $100,000)
    print("California Housing 데이터셋 로딩 완료!")
except Exception as e:
    print(f"California Housing 데이터셋 로딩 실패: {e}")
    raise

print(f"\n[California Housing 데이터셋 정보]")
print(f"데이터 크기: {df_reg.shape}")
print(f"특성 이름: {list(housing.feature_names)}")
print(f"\n처음 5행:")
print(df_reg.head())


# ============================================================
# Part 3: 특성(X)과 타겟(y) 분리
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 특성과 타겟 분리")
print("=" * 60)

# 분류용 특성과 타겟
feature_columns_clf = ['sepal length (cm)', 'sepal width (cm)',
                       'petal length (cm)', 'petal width (cm)']
X_clf = df_clf[feature_columns_clf]
y_clf = df_clf['target']

# 회귀용 특성과 타겟 (주요 특성만 선택)
feature_columns_reg = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
X_reg = df_reg[feature_columns_reg]
y_reg = df_reg['MedHouseVal']

print("\n[분류 데이터 - Iris]")
print(f"  특성 열: {list(X_clf.columns)}")
print(f"  크기: {X_clf.shape}")
print(f"  타겟 클래스 분포:")
for i, name in enumerate(iris.target_names):
    count = (y_clf == i).sum()
    print(f"    - {name}: {count}개 ({count/len(y_clf):.1%})")

print("\n[회귀 데이터 - California Housing]")
print(f"  특성 열: {list(X_reg.columns)}")
print(f"  크기: {X_reg.shape}")
print(f"  타겟 (주택 가격) 통계:")
print(f"    - 평균: ${y_reg.mean()*100000:,.0f}")
print(f"    - 범위: ${y_reg.min()*100000:,.0f} ~ ${y_reg.max()*100000:,.0f}")


# ============================================================
# Part 4: 학습/테스트 데이터 분리
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 학습/테스트 데이터 분리")
print("=" * 60)

# 분류용 데이터 분리
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf,
    test_size=0.2,      # 20%는 테스트용
    random_state=42,    # 재현성
    stratify=y_clf      # 클래스 비율 유지
)

# 회귀용 데이터 분리
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

print(f"\n[분류 데이터 분리 결과 (Iris)]")
print(f"  전체 데이터: {len(X_clf)}개")
print(f"  학습 데이터: {len(X_train_clf)}개 ({len(X_train_clf)/len(X_clf):.0%})")
print(f"  테스트 데이터: {len(X_test_clf)}개 ({len(X_test_clf)/len(X_clf):.0%})")

print(f"\n[회귀 데이터 분리 결과 (California Housing)]")
print(f"  전체 데이터: {len(X_reg)}개")
print(f"  학습 데이터: {len(X_train_reg)}개 ({len(X_train_reg)/len(X_reg):.0%})")
print(f"  테스트 데이터: {len(X_test_reg)}개 ({len(X_test_reg)/len(X_reg):.0%})")


# ============================================================
# Part 5: 분류 모델 (DecisionTreeClassifier)
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 분류 모델 - DecisionTreeClassifier (Iris)")
print("=" * 60)

# 1. 모델 생성
clf_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5  # 과대적합 방지
)

# 2. 학습 (fit)
clf_model.fit(X_train_clf, y_train_clf)
print("\n[학습 완료]")

# 3. 예측 (predict)
y_pred_clf = clf_model.predict(X_test_clf)
print(f"\n[예측 결과 샘플]")
print(f"  실제: {list(y_test_clf[:10].values)}")
print(f"  예측: {list(y_pred_clf[:10])}")

# 예측을 꽃 이름으로 변환
print(f"\n[예측 결과 (꽃 이름)]")
for i in range(min(5, len(y_pred_clf))):
    actual = iris.target_names[y_test_clf.iloc[i]]
    predicted = iris.target_names[y_pred_clf[i]]
    match = "O" if actual == predicted else "X"
    print(f"  샘플 {i+1}: 실제={actual}, 예측={predicted} [{match}]")

# 4. 평가 (score)
accuracy = clf_model.score(X_test_clf, y_test_clf)
print(f"\n[분류 성능]")
print(f"  정확도(Accuracy): {accuracy:.1%}")

# 혼동 행렬
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\n[혼동 행렬]")
print(f"              setosa  versicolor  virginica")
for i, name in enumerate(iris.target_names):
    print(f"  {name:>10}:  {cm[i,0]:5}      {cm[i,1]:5}      {cm[i,2]:5}")

# 특성 중요도
print(f"\n[특성 중요도]")
for col, imp in zip(feature_columns_clf, clf_model.feature_importances_):
    print(f"  {col}: {imp:.3f}")


# ============================================================
# Part 6: 회귀 모델 (LinearRegression)
# ============================================================

print("\n" + "=" * 60)
print("Part 6: 회귀 모델 - LinearRegression (California Housing)")
print("=" * 60)

# 1. 모델 생성
reg_model = LinearRegression()

# 2. 학습 (fit)
reg_model.fit(X_train_reg, y_train_reg)
print("\n[학습 완료]")

# 3. 예측 (predict)
y_pred_reg = reg_model.predict(X_test_reg)
print(f"\n[예측 결과 샘플 (주택 가격, 단위: $100,000)]")
print(f"  실제: {list(y_test_reg[:5].round(2))}")
print(f"  예측: {list(y_pred_reg[:5].round(2))}")

# 4. 평가 (score)
r2 = reg_model.score(X_test_reg, y_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)

print(f"\n[회귀 성능]")
print(f"  R² 점수: {r2:.3f}")
print(f"  MSE: {mse:.3f}")
print(f"  RMSE: {rmse:.3f} (약 ${rmse*100000:,.0f})")

# 회귀 계수
print(f"\n[회귀 계수]")
print(f"  절편: {reg_model.intercept_:.4f}")
for col, coef in zip(feature_columns_reg, reg_model.coef_):
    print(f"  {col}: {coef:.4f}")

print(f"\n[계수 해석]")
print(f"  → 중위 소득(MedInc) 1단위 증가 시 주택 가격 ${reg_model.coef_[0]*100000:,.0f} 증가")


# ============================================================
# Part 7: 새 데이터 예측
# ============================================================

print("\n" + "=" * 60)
print("Part 7: 새 데이터 예측")
print("=" * 60)

# 새로운 꽃 데이터 (분류)
new_flowers = pd.DataFrame({
    'sepal length (cm)': [5.0, 6.5, 7.2],
    'sepal width (cm)': [3.5, 2.8, 3.0],
    'petal length (cm)': [1.5, 4.5, 6.0],
    'petal width (cm)': [0.2, 1.5, 2.2]
})

print("\n[새 꽃 데이터]")
print(new_flowers)

# 분류 예측
flower_predictions = clf_model.predict(new_flowers)
flower_proba = clf_model.predict_proba(new_flowers)

print("\n[꽃 종류 예측 (분류)]")
for i, (pred, proba) in enumerate(zip(flower_predictions, flower_proba)):
    species = iris.target_names[pred]
    print(f"  꽃 {i+1}: {species}")
    print(f"         (확률: setosa {proba[0]:.1%}, versicolor {proba[1]:.1%}, virginica {proba[2]:.1%})")

# 새로운 주택 데이터 (회귀)
new_houses = pd.DataFrame({
    'MedInc': [3.0, 5.0, 8.0],      # 중위 소득
    'HouseAge': [20, 15, 5],         # 집 연식
    'AveRooms': [5.0, 6.0, 7.0],     # 평균 방 수
    'AveOccup': [3.0, 2.5, 2.0]      # 평균 거주자 수
})

print("\n[새 주택 데이터]")
print(new_houses)

# 회귀 예측
price_predictions = reg_model.predict(new_houses)

print("\n[주택 가격 예측 (회귀)]")
for i, pred in enumerate(price_predictions):
    print(f"  주택 {i+1}: ${pred*100000:,.0f}")


# ============================================================
# Part 8: 시각화
# ============================================================

print("\n" + "=" * 60)
print("Part 8: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 분류: 꽃잎 길이-너비별 종 분포
ax1 = axes[0, 0]
colors_map = {0: 'red', 1: 'green', 2: 'blue'}
colors = [colors_map[t] for t in y_clf]
ax1.scatter(df_clf['petal length (cm)'], df_clf['petal width (cm)'],
            c=colors, alpha=0.6, s=30)
ax1.set_xlabel('Petal Length (cm)')
ax1.set_ylabel('Petal Width (cm)')
ax1.set_title('Iris: 꽃잎 크기별 종 분포\n(빨강=setosa, 초록=versicolor, 파랑=virginica)')

# 2. 분류: 혼동 행렬 히트맵
ax2 = axes[0, 1]
im = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(['setosa', 'versicolor', 'virginica'], rotation=45)
ax2.set_yticklabels(['setosa', 'versicolor', 'virginica'])
ax2.set_xlabel('예측')
ax2.set_ylabel('실제')
ax2.set_title(f'분류 혼동 행렬 (정확도: {accuracy:.1%})')
for i in range(3):
    for j in range(3):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
plt.colorbar(im, ax=ax2)

# 3. 회귀: 실제 vs 예측
ax3 = axes[1, 0]
ax3.scatter(y_test_reg, y_pred_reg, alpha=0.3, s=10)
ax3.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         'r--', linewidth=2)
ax3.set_xlabel('실제 주택 가격')
ax3.set_ylabel('예측 주택 가격')
ax3.set_title(f'California Housing: 실제 vs 예측 (R²={r2:.3f})')

# 4. 회귀: 잔차 분포
ax4 = axes[1, 1]
residuals = y_test_reg - y_pred_reg
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('잔차 (실제 - 예측)')
ax4.set_ylabel('빈도')
ax4.set_title(f'회귀 잔차 분포 (RMSE={rmse:.3f})')

plt.tight_layout()
plt.savefig('12_ml_intro_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장: 12_ml_intro_visualization.png")


# ============================================================
# Part 9: sklearn 일관된 패턴 정리
# ============================================================

print("\n" + "=" * 60)
print("Part 9: sklearn 일관된 패턴 정리")
print("=" * 60)

print("""
[sklearn 기본 패턴 - 모든 모델 공통!]

1. 모델 생성
   model = ModelClass(parameters)

2. 학습 (fit)
   model.fit(X_train, y_train)

3. 예측 (predict)
   y_pred = model.predict(X_test)

4. 평가 (score)
   score = model.score(X_test, y_test)

[분류 모델]
- DecisionTreeClassifier
- RandomForestClassifier
- LogisticRegression
- SVM (SVC)
- KNeighborsClassifier

[회귀 모델]
- LinearRegression
- DecisionTreeRegressor
- RandomForestRegressor
- Ridge, Lasso
- SVR

모든 모델이 동일한 fit/predict/score 패턴을 따릅니다!
""")


# ============================================================
# Part 10: 다양한 모델 비교
# ============================================================

print("\n" + "=" * 60)
print("Part 10: 다양한 모델 비교")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 분류 모델들 비교 (Iris 데이터)
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

print("\n[분류 모델 비교 - Iris 데이터셋]")
print(f"{'모델':<20} {'정확도':>10}")
print("-" * 32)

for name, model in classifiers.items():
    model.fit(X_train_clf, y_train_clf)
    accuracy = model.score(X_test_clf, y_test_clf)
    print(f"{name:<20} {accuracy:>10.1%}")


# 회귀 모델들 비교 (California Housing 데이터)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

regressors = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50),
}

print("\n[회귀 모델 비교 - California Housing 데이터셋]")
print(f"{'모델':<20} {'R² 점수':>10} {'RMSE':>10}")
print("-" * 42)

for name, model in regressors.items():
    model.fit(X_train_reg, y_train_reg)
    r2 = model.score(X_test_reg, y_test_reg)
    y_pred = model.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    print(f"{name:<20} {r2:>10.3f} {rmse:>10.3f}")


# ============================================================
# 핵심 정리
# ============================================================

print("\n" + "=" * 60)
print("핵심 정리: 12차시 머신러닝 소개와 문제 유형")
print("=" * 60)

print("""
1. 머신러닝이란?
   - 데이터에서 스스로 패턴을 학습하는 알고리즘
   - 전통적 프로그래밍: 사람이 규칙 작성
   - 머신러닝: 데이터가 규칙을 알려줌

2. 실제 공개 데이터셋
   - Iris: 붓꽃 분류 (150개, 3종류)
   - California Housing: 주택 가격 예측 (20,640개)
   - sklearn.datasets에서 쉽게 불러오기

3. 핵심 용어
   - 특성 (Feature): 입력 데이터
   - 타겟 (Target): 예측하려는 값
   - 모델 (Model): 학습된 패턴
   - 학습 (Training): 패턴 찾기
   - 예측 (Prediction): 패턴 적용

4. 지도학습 문제 유형
   - 분류: 범주 예측 ("~인가요?") → Iris
   - 회귀: 숫자 예측 ("얼마나?") → California Housing

5. sklearn 기본 패턴
   model.fit(X, y)      # 학습
   model.predict(X)     # 예측
   model.score(X, y)    # 평가

6. 학습/테스트 분리
   - 일반적으로 80% 학습, 20% 테스트
   - 처음 보는 데이터로 평가해야 실제 성능 확인
""")

print("\n다음 차시 예고: 13차시 - 분류 모델: 의사결정나무")
