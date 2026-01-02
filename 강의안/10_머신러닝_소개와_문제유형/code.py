# [10차시] 머신러닝 소개와 문제 유형 - 실습 코드

import numpy as np
import pandas as pd

# 한글 출력 설정
print("=" * 60)
print("10차시: 머신러닝 소개와 문제 유형")
print("Part III 시작: 드디어 AI 모델을 만듭니다!")
print("=" * 60)
print()


# ============================================================
# 실습 1: 제조 데이터 생성
# ============================================================
print("=" * 50)
print("실습 1: 제조 데이터 생성")
print("=" * 50)

np.random.seed(42)
n_samples = 200

# 특성(Feature) 생성
temperature = np.random.normal(85, 5, n_samples)  # 온도
humidity = np.random.normal(50, 10, n_samples)    # 습도
speed = np.random.normal(100, 15, n_samples)      # 속도

# 타겟(Target) 생성 - 불량 여부 (온도 높으면 불량 확률 증가)
defect_prob = 0.1 + 0.02 * (temperature - 80)
defect = (np.random.random(n_samples) < defect_prob).astype(int)

# 생산량 (회귀용 타겟)
production = 1000 + 5 * speed - 3 * temperature + np.random.normal(0, 50, n_samples)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '불량여부': defect,        # 분류 타겟 (0: 정상, 1: 불량)
    '생산량': production        # 회귀 타겟
})

print("데이터 샘플:")
print(df.head(10))
print(f"\n데이터 크기: {df.shape}")
print()


# ============================================================
# 실습 2: 특성(X)과 타겟(y) 구분
# ============================================================
print("=" * 50)
print("실습 2: 특성(X)과 타겟(y) 구분")
print("=" * 50)

# 분류 문제: 불량 여부 예측
X_clf = df[['온도', '습도', '속도']]  # 특성 (Feature)
y_clf = df['불량여부']                 # 타겟 (Target) - 범주

print("▶ 분류 문제 (Classification)")
print(f"  - 특성(X): {list(X_clf.columns)}")
print(f"  - 타겟(y): 불량여부 (0=정상, 1=불량)")
print(f"  - 타겟 분포:")
print(y_clf.value_counts())

# 회귀 문제: 생산량 예측
X_reg = df[['온도', '습도', '속도']]  # 특성 (Feature)
y_reg = df['생산량']                   # 타겟 (Target) - 숫자

print("\n▶ 회귀 문제 (Regression)")
print(f"  - 특성(X): {list(X_reg.columns)}")
print(f"  - 타겟(y): 생산량 (연속적인 숫자)")
print(f"  - 타겟 통계: 평균 {y_reg.mean():.1f}, 표준편차 {y_reg.std():.1f}")
print()


# ============================================================
# 실습 3: 학습/테스트 데이터 분리
# ============================================================
print("=" * 50)
print("실습 3: 학습/테스트 데이터 분리")
print("=" * 50)

from sklearn.model_selection import train_test_split

# train_test_split 사용
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf,
    test_size=0.2,      # 20%를 테스트용으로
    random_state=42     # 재현성을 위한 시드
)

print(f"전체 데이터: {len(df)}개")
print(f"학습 데이터: {len(X_train)}개 (80%)")
print(f"테스트 데이터: {len(X_test)}개 (20%)")

print("\n★ 왜 분리하나요?")
print("  - 모델이 '외운' 것인지 '이해한' 것인지 확인하기 위해")
print("  - 처음 보는 데이터(테스트)로 성능을 평가합니다")
print()


# ============================================================
# 실습 4: 분류 모델 맛보기 (의사결정트리)
# ============================================================
print("=" * 50)
print("실습 4: 분류 모델 맛보기 (의사결정트리)")
print("=" * 50)

from sklearn.tree import DecisionTreeClassifier

# 모델 생성
clf_model = DecisionTreeClassifier(random_state=42)

# 학습 (fit)
clf_model.fit(X_train, y_train)
print("▶ model.fit(X_train, y_train) - 학습 완료!")

# 예측 (predict)
y_pred = clf_model.predict(X_test)
print("▶ model.predict(X_test) - 예측 완료!")

# 결과 확인
print(f"\n실제값 (처음 10개): {list(y_test[:10].values)}")
print(f"예측값 (처음 10개): {list(y_pred[:10])}")

# 정확도 (간단히)
accuracy = (y_pred == y_test).mean()
print(f"\n정확도: {accuracy:.1%}")

# score() 메서드 사용
accuracy_score = clf_model.score(X_test, y_test)
print(f"score() 메서드 정확도: {accuracy_score:.1%}")
print()


# ============================================================
# 실습 5: 회귀 모델 맛보기 (선형회귀)
# ============================================================
print("=" * 50)
print("실습 5: 회귀 모델 맛보기 (선형회귀)")
print("=" * 50)

from sklearn.linear_model import LinearRegression

# 학습/테스트 분리 (회귀용)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 모델 생성
reg_model = LinearRegression()

# 학습
reg_model.fit(X_train_r, y_train_r)
print("▶ model.fit(X_train, y_train) - 학습 완료!")

# 예측
y_pred_r = reg_model.predict(X_test_r)
print("▶ model.predict(X_test) - 예측 완료!")

# 결과 확인
print(f"\n실제값 (처음 5개): {list(y_test_r[:5].round(1).values)}")
print(f"예측값 (처음 5개): {list(y_pred_r[:5].round(1))}")

# R² 점수
r2 = reg_model.score(X_test_r, y_test_r)
print(f"\nR² 점수: {r2:.3f}")
print("(1에 가까울수록 좋음)")
print()


# ============================================================
# 실습 6: 분류 vs 회귀 구분 연습
# ============================================================
print("=" * 50)
print("실습 6: 분류 vs 회귀 구분 연습")
print("=" * 50)

problems = [
    ("제품이 불량인가요?", "분류", "예/아니오로 답함"),
    ("불량률이 몇 %일까요?", "회귀", "숫자로 답함 (예: 3.5%)"),
    ("고객이 이탈할까요?", "분류", "이탈/유지 중 하나"),
    ("다음 달 매출은 얼마?", "회귀", "숫자로 답함 (예: 1억원)"),
    ("이메일이 스팸인가요?", "분류", "스팸/정상 중 하나"),
    ("생산량이 얼마나 될까요?", "회귀", "숫자로 답함 (예: 1,500개)"),
    ("품질 등급은 무엇인가요?", "분류", "A/B/C 중 하나 (다중 분류)"),
    ("설비 고장까지 남은 시간?", "회귀", "숫자로 답함 (예: 48시간)"),
]

print("문제를 분류/회귀로 구분해보세요:\n")
for question, answer, explanation in problems:
    print(f"Q: {question}")
    print(f"   → {answer} ({explanation})")
    print()


# ============================================================
# 실습 7: sklearn 패턴 정리
# ============================================================
print("=" * 50)
print("실습 7: sklearn 기본 패턴 정리")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────┐
│              sklearn 기본 패턴                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. from sklearn.xxx import ModelName           │
│                                                  │
│  2. model = ModelName()                         │
│                                                  │
│  3. model.fit(X_train, y_train)   ← 학습        │
│                                                  │
│  4. model.predict(X_test)          ← 예측        │
│                                                  │
│  5. model.score(X_test, y_test)    ← 평가        │
│                                                  │
└─────────────────────────────────────────────────┘

★ 모든 sklearn 모델이 이 패턴을 따릅니다!
  → 한번 배우면 다른 모델도 같은 방식으로 사용
""")


# ============================================================
# 실습 8: 다양한 sklearn 모델 미리보기
# ============================================================
print("=" * 50)
print("실습 8: 다양한 sklearn 모델 미리보기")
print("=" * 50)

print("""
[분류 모델]
- DecisionTreeClassifier: 의사결정트리 (11차시)
- RandomForestClassifier: 랜덤포레스트 (12차시)
- LogisticRegression: 로지스틱 회귀
- SVC: 서포트 벡터 머신

[회귀 모델]
- LinearRegression: 선형회귀 (13차시)
- DecisionTreeRegressor: 의사결정트리 회귀
- RandomForestRegressor: 랜덤포레스트 회귀

모두 같은 패턴: fit() → predict() → score()
""")


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                     머신러닝 핵심                       │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 머신러닝 = 데이터에서 패턴을 학습                      │
│                                                        │
│  ▶ 지도학습: 정답이 있는 데이터로 학습                    │
│     - 분류: 범주 예측 ("~인가요?" → 예/아니오)            │
│     - 회귀: 숫자 예측 ("얼마나?" → 1,247개)              │
│                                                        │
│  ▶ sklearn 패턴                                         │
│     - fit(X, y): 학습                                   │
│     - predict(X): 예측                                  │
│     - score(X, y): 평가                                 │
│                                                        │
│  ▶ 학습/테스트 분리: 처음 보는 데이터로 평가              │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 의사결정나무로 불량 분류 모델 구축!
""")

print("=" * 60)
print("10차시 실습 완료!")
print("Part III가 시작되었습니다!")
print("=" * 60)
