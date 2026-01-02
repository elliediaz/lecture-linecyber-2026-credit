# [11차시] 분류 모델 (1): 의사결정나무 - 실습 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("11차시: 분류 모델 (1) - 의사결정나무")
print("첫 번째 AI 모델을 본격적으로 만들어봅시다!")
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

# 불량 여부 (온도와 습도가 높으면 불량 확률 증가)
defect_prob = 0.05 + 0.03 * (temperature - 80) / 5 + 0.02 * (humidity - 40) / 10
defect = (np.random.random(n_samples) < defect_prob).astype(int)

# DataFrame 생성
df = pd.DataFrame({
    '온도': temperature,
    '습도': humidity,
    '속도': speed,
    '불량여부': defect
})

print("데이터 샘플:")
print(df.head(10))
print(f"\n데이터 크기: {df.shape}")
print(f"불량 비율: {df['불량여부'].mean():.1%}")
print()


# ============================================================
# 실습 2: 데이터 준비 (특성과 타겟)
# ============================================================
print("=" * 50)
print("실습 2: 데이터 준비")
print("=" * 50)

# 특성(X)과 타겟(y) 분리
X = df[['온도', '습도', '속도']]
y = df['불량여부']

print(f"특성(X) shape: {X.shape}")
print(f"타겟(y) shape: {y.shape}")
print(f"\n특성 변수: {list(X.columns)}")
print(f"타겟 클래스: 0=정상, 1=불량")
print()


# ============================================================
# 실습 3: 학습/테스트 데이터 분리
# ============================================================
print("=" * 50)
print("실습 3: 학습/테스트 분리")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20%를 테스트용으로
    random_state=42,    # 재현성
    stratify=y          # 클래스 비율 유지
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"\n학습 데이터 불량 비율: {y_train.mean():.1%}")
print(f"테스트 데이터 불량 비율: {y_test.mean():.1%}")
print()


# ============================================================
# 실습 4: 의사결정나무 모델 생성 및 학습
# ============================================================
print("=" * 50)
print("실습 4: 의사결정나무 학습")
print("=" * 50)

# 모델 생성
model = DecisionTreeClassifier(
    max_depth=5,        # 트리 깊이 제한 (과대적합 방지)
    random_state=42
)

# 학습
model.fit(X_train, y_train)
print("▶ model.fit(X_train, y_train) - 학습 완료!")

# 트리 정보
print(f"\n트리 깊이: {model.get_depth()}")
print(f"리프 노드 수: {model.get_n_leaves()}")
print()


# ============================================================
# 실습 5: 예측하기
# ============================================================
print("=" * 50)
print("실습 5: 예측하기")
print("=" * 50)

# 새 데이터 예측
print("▶ 새 데이터 예측")
new_data = [[90, 55, 100]]  # 온도 90, 습도 55, 속도 100
prediction = model.predict(new_data)
print(f"   입력: 온도=90, 습도=55, 속도=100")
print(f"   예측: {'불량' if prediction[0] == 1 else '정상'}")

# 확률 예측
proba = model.predict_proba(new_data)
print(f"\n▶ 예측 확률")
print(f"   정상 확률: {proba[0][0]:.1%}")
print(f"   불량 확률: {proba[0][1]:.1%}")

# 테스트 데이터 예측
y_pred = model.predict(X_test)
print(f"\n▶ 테스트 데이터 {len(y_pred)}개 예측 완료")
print()


# ============================================================
# 실습 6: 모델 평가
# ============================================================
print("=" * 50)
print("실습 6: 모델 평가")
print("=" * 50)

# 정확도 계산
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"학습 정확도: {train_acc:.1%}")
print(f"테스트 정확도: {test_acc:.1%}")

# 과대적합 확인
diff = train_acc - test_acc
if diff > 0.1:
    print(f"\n⚠️ 과대적합 의심! (차이: {diff:.1%})")
    print("   → max_depth를 줄이거나 min_samples_leaf를 늘려보세요")
else:
    print(f"\n✅ 적절한 일반화 (차이: {diff:.1%})")

# 상세 평가
print("\n▶ 분류 리포트")
print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
print()


# ============================================================
# 실습 7: 트리 시각화
# ============================================================
print("=" * 50)
print("실습 7: 트리 시각화")
print("=" * 50)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model,
          feature_names=['온도', '습도', '속도'],
          class_names=['정상', '불량'],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title('의사결정나무 시각화', fontsize=14)
plt.tight_layout()
plt.show()

print("▶ 트리 시각화 완료!")
print("   → 왜 그렇게 예측했는지 눈으로 확인 가능")
print()


# ============================================================
# 실습 8: 특성 중요도
# ============================================================
print("=" * 50)
print("실습 8: 특성 중요도")
print("=" * 50)

importance = pd.DataFrame({
    '특성': X.columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print("특성 중요도:")
print(importance.to_string(index=False))

# 중요도 시각화
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3B82F6', '#10B981', '#F59E0B']
bars = ax.barh(importance['특성'], importance['중요도'], color=colors)
ax.set_xlabel('중요도')
ax.set_title('특성 중요도')
ax.invert_yaxis()

for bar, val in zip(bars, importance['중요도']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center')

plt.tight_layout()
plt.show()
print()


# ============================================================
# 실습 9: max_depth에 따른 성능 변화
# ============================================================
print("=" * 50)
print("실습 9: max_depth에 따른 성능 변화")
print("=" * 50)

depths = [1, 2, 3, 5, 10, None]  # None = 제한 없음
results = []

for depth in depths:
    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_model.fit(X_train, y_train)
    train_score = temp_model.score(X_train, y_train)
    test_score = temp_model.score(X_test, y_test)
    depth_str = str(depth) if depth else '무제한'
    results.append({
        'max_depth': depth_str,
        '학습 정확도': f'{train_score:.1%}',
        '테스트 정확도': f'{test_score:.1%}',
        '차이': f'{(train_score - test_score):.1%}'
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\n★ max_depth가 너무 크면 학습/테스트 차이가 커짐 (과대적합)")
print()


# ============================================================
# 실습 10: 의사결정 규칙 추출
# ============================================================
print("=" * 50)
print("실습 10: 의사결정 규칙 추출")
print("=" * 50)

rules = export_text(model, feature_names=['온도', '습도', '속도'])
# 처음 20줄만 출력
rules_lines = rules.split('\n')[:20]
print('\n'.join(rules_lines))
if len(rules.split('\n')) > 20:
    print("... (이하 생략)")
print()


# ============================================================
# 실습 11: 새로운 데이터 예측 연습
# ============================================================
print("=" * 50)
print("실습 11: 새로운 데이터 예측 연습")
print("=" * 50)

test_samples = [
    [80, 45, 100],   # 정상적인 조건
    [92, 65, 90],    # 고온, 고습
    [85, 50, 110],   # 평균적인 조건
    [95, 70, 85],    # 극단적 조건
]

print("예측 결과:")
print("-" * 60)
for sample in test_samples:
    pred = model.predict([sample])
    proba = model.predict_proba([sample])
    result = '불량' if pred[0] == 1 else '정상'
    print(f"온도={sample[0]:5.1f}, 습도={sample[1]:5.1f}, 속도={sample[2]:5.1f}")
    print(f"   → {result} (불량 확률: {proba[0][1]:.1%})")
print()


# ============================================================
# 핵심 요약
# ============================================================
print("=" * 50)
print("핵심 요약")
print("=" * 50)

print("""
┌───────────────────────────────────────────────────────┐
│                의사결정나무 핵심 정리                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ▶ 모델 생성 및 학습                                    │
│     model = DecisionTreeClassifier(max_depth=5)        │
│     model.fit(X_train, y_train)                        │
│                                                        │
│  ▶ 예측                                                │
│     y_pred = model.predict(X_test)                    │
│     proba = model.predict_proba(X_test)               │
│                                                        │
│  ▶ 평가                                                │
│     accuracy = model.score(X_test, y_test)            │
│                                                        │
│  ▶ 시각화                                              │
│     plot_tree(model, feature_names=[...])             │
│                                                        │
│  ▶ 과대적합 방지                                        │
│     → max_depth 설정 (보통 3~10)                       │
│     → 학습/테스트 정확도 차이 확인                      │
│                                                        │
│  ★ 장점: 해석 가능! 왜 그렇게 예측했는지 설명 가능       │
│                                                        │
└───────────────────────────────────────────────────────┘

다음 차시: 랜덤포레스트 (여러 트리의 앙상블)
""")

print("=" * 60)
print("11차시 실습 완료!")
print("첫 번째 AI 모델을 성공적으로 만들었습니다!")
print("=" * 60)
