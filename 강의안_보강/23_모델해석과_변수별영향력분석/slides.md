---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [23차시] 모델 해석과 변수별 영향력 분석

## 왜 그런 예측을 했는지 이해하기

---

# 학습 목표

1. **모델 해석**의 필요성과 종류를 이해한다
2. **Feature Importance**를 계산하고 해석한다
3. **Permutation Importance**를 활용한다

---

# 지난 시간 복습

- **CNN**: 이미지 처리, 합성곱, 풀링
- **RNN/LSTM**: 시계열 처리, 장기 의존성
- **고급 아키텍처**: ResNet, Transformer

**오늘**: 모델이 왜 그런 예측을 했는지 이해하기

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | 모델 해석의 필요성 |
| 대주제 2 | 10분 | Feature Importance |
| 대주제 3 | 8분 | Permutation Importance |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## 모델 해석의 필요성

---

# 왜 모델 해석이 필요한가?

**블랙박스 문제**:
```
입력 [온도, 압력, 속도] → 모델 → 예측: 불량
                          ?
```

- 왜 불량이라고 예측했는가?
- 어떤 변수가 중요했는가?
- 예측을 신뢰할 수 있는가?

---

# 모델 해석이 중요한 이유

| 관점 | 이유 |
|-----|------|
| **신뢰** | 예측 근거를 설명해야 의사결정 가능 |
| **디버깅** | 잘못된 패턴 학습 발견 |
| **개선** | 중요 변수 파악 → 데이터 수집 방향 |
| **규제** | EU AI Act 등 설명 의무화 추세 |

---

# 제조업에서의 예시

**상황**: 품질 예측 모델이 "불량"이라고 예측

**질문**:
- 온도 때문인가? 압력 때문인가?
- 어떤 조건을 바꾸면 정상이 되는가?
- 이 예측을 믿어도 되는가?

**모델 해석이 답을 줌**

---

# 해석 가능한 모델 vs 복잡한 모델

| 모델 | 해석 가능성 | 성능 |
|-----|-----------|------|
| 선형 회귀 | 높음 (계수) | 낮음 |
| 의사결정나무 | 높음 (규칙) | 중간 |
| RandomForest | 중간 (중요도) | 높음 |
| 신경망 | 낮음 (블랙박스) | 높음 |

**해석 방법으로 복잡한 모델도 이해 가능**

---

# 모델 해석 방법 분류

| 분류 | 방법 | 특징 |
|-----|------|------|
| **모델 내장** | Feature Importance | 트리 모델 전용 |
| **모델 무관** | Permutation Importance | 모든 모델 가능 |
| **인스턴스별** | SHAP, LIME | 개별 예측 설명 |

---

# 전역 vs 지역 해석

**전역 해석 (Global)**:
- 모델 전체에서 변수 중요도
- "온도가 가장 중요한 변수"

**지역 해석 (Local)**:
- 특정 예측에 대한 설명
- "이 샘플은 압력이 높아서 불량"

---

<!-- _class: lead -->
# 대주제 2
## Feature Importance

---

# Feature Importance란?

**정의**: 모델이 예측할 때 각 변수가 얼마나 기여했는가

**트리 모델에서**:
- 변수로 분할할 때 불순도 감소량
- 많이 사용되고, 불순도를 많이 줄이면 중요

---

# RandomForest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature Importance 추출
importances = model.feature_importances_
```

`feature_importances_`: 각 변수의 중요도 (합 = 1)

---

# Feature Importance 시각화

```python
import matplotlib.pyplot as plt

# 정렬
indices = np.argsort(importances)[::-1]

# 막대 그래프
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [feature_names[i] for i in indices],
           rotation=45)
plt.title('Feature Importance')
plt.ylabel('Importance')
```

---

# Feature Importance 해석

```
온도      ████████████████████  0.35
압력      ██████████████        0.25
진동      ██████████            0.18
속도      ████████              0.12
습도      ██████                0.10
```

**해석**:
- 온도가 가장 중요 (35%)
- 온도, 압력만으로 60% 설명
- 습도는 상대적으로 덜 중요

---

# 제조업 인사이트

**Feature Importance 결과**:
- 온도 > 압력 > 진동 > 속도 > 습도

**조치**:
1. 온도 모니터링 강화
2. 압력 센서 정밀도 개선
3. 습도 센서는 비용 대비 효과 낮음

---

# Feature Importance 주의점

**한계**:
1. **상관 변수 문제**: 상관된 변수 간 중요도 분산
2. **스케일 영향**: 일부 모델에서 스케일 영향
3. **트리 모델 전용**: 다른 모델에는 적용 어려움

**해결**: Permutation Importance 사용

---

# 상관 변수 문제 예시

```
온도와 습도가 높은 상관 (r = 0.9)

Feature Importance:
온도: 0.25
습도: 0.20  ← 합쳐서 0.45인데 분산됨

실제로는 "온도" 하나로도 충분한 정보
```

---

<!-- _class: lead -->
# 대주제 3
## Permutation Importance

---

# Permutation Importance란?

**아이디어**:
변수 값을 무작위로 섞으면 예측 성능이 얼마나 떨어지는가?

**중요한 변수** → 섞으면 성능 크게 하락
**중요하지 않은 변수** → 섞어도 성능 유지

---

# Permutation Importance 원리

```
원본 데이터:
온도  압력  속도  → 정확도 90%

온도 섞기:
랜덤  압력  속도  → 정확도 65%
                   ↓
          중요도 = 90% - 65% = 25%
```

---

# sklearn에서 구현

```python
from sklearn.inspection import permutation_importance

# Permutation Importance 계산
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,      # 반복 횟수
    random_state=42
)

# 결과
importances = result.importances_mean
std = result.importances_std
```

---

# Permutation Importance 파라미터

| 파라미터 | 의미 |
|---------|------|
| `estimator` | 학습된 모델 |
| `X`, `y` | 테스트 데이터 |
| `n_repeats` | 반복 횟수 (안정성) |
| `scoring` | 평가 지표 |

---

# 결과 해석

```python
for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]}: "
          f"{result.importances_mean[i]:.3f} "
          f"± {result.importances_std[i]:.3f}")
```

```
온도: 0.250 ± 0.015
압력: 0.180 ± 0.020
진동: 0.120 ± 0.010
속도: 0.050 ± 0.008
습도: 0.010 ± 0.005
```

---

# Feature vs Permutation Importance

| 항목 | Feature Importance | Permutation Importance |
|-----|-------------------|------------------------|
| 모델 | 트리 모델만 | 모든 모델 |
| 데이터 | 학습 데이터 기반 | 테스트 데이터 기반 |
| 상관 변수 | 분산됨 | 그래도 분산됨 |
| 계산 비용 | 빠름 | 느림 (n_repeats) |

---

# 두 방법 함께 사용

```python
# 1. Feature Importance (빠르게)
fi = model.feature_importances_

# 2. Permutation Importance (정확하게)
pi = permutation_importance(model, X_test, y_test)

# 3. 비교
for name, f, p in zip(feature_names, fi, pi.importances_mean):
    print(f"{name}: FI={f:.3f}, PI={p:.3f}")
```

**두 방법의 순위가 비슷하면 신뢰도 ↑**

---

# 실습: 품질 예측 모델 해석

```python
# 데이터 준비
X = df[['temperature', 'pressure', 'speed', 'humidity', 'vibration']]
y = df['defect']

# 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature Importance
fi = model.feature_importances_

# Permutation Importance
pi = permutation_importance(model, X_test, y_test, n_repeats=10)
```

---

# 시각화 코드

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature Importance
axes[0].barh(feature_names, fi)
axes[0].set_title('Feature Importance')

# Permutation Importance
axes[1].barh(feature_names, pi.importances_mean)
axes[1].errorbar(pi.importances_mean, feature_names,
                 xerr=pi.importances_std, fmt='o')
axes[1].set_title('Permutation Importance')
```

---

# SHAP 소개 (참고)

**SHAP (SHapley Additive exPlanations)**:
- 게임 이론 기반 해석 방법
- 개별 예측에 대한 변수 기여도
- 전역 + 지역 해석 모두 가능

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

# SHAP 예시

```
샘플 1 예측: 불량 (확률 0.85)

기여도:
온도 (+0.25)  ← 높아서 불량 방향
압력 (+0.15)  ← 높아서 불량 방향
속도 (-0.05)  ← 낮아서 정상 방향
```

**왜 이 샘플이 불량인지** 구체적 설명

---

# 모델 해석 워크플로우

```
1. 모델 학습
   ↓
2. Feature Importance (빠른 탐색)
   ↓
3. Permutation Importance (검증)
   ↓
4. SHAP (개별 예측 설명, 필요시)
   ↓
5. 비즈니스 인사이트 도출
```

---

# 비즈니스 활용

**분석 결과**:
- 온도가 가장 중요 (25%)
- 압력이 두 번째 (18%)

**조치**:
1. 온도 모니터링 주기 단축
2. 온도 이상 시 즉시 알림
3. 압력 센서 교정 주기 확인
4. 불필요한 습도 센서 비용 절감 검토

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **모델 해석의 필요성**
   - 신뢰, 디버깅, 개선, 규제 대응
   - 블랙박스 → 설명 가능한 AI

2. **Feature Importance**
   - 트리 모델 내장 기능
   - 빠르지만 상관 변수 문제

3. **Permutation Importance**
   - 모든 모델에 적용 가능
   - 테스트 데이터 기반, 신뢰도 높음

---

# 핵심 코드

```python
# Feature Importance (트리 모델)
model.feature_importances_

# Permutation Importance (모든 모델)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
result.importances_mean
result.importances_std
```

---

# 체크리스트

- [ ] Feature Importance 추출
- [ ] Permutation Importance 계산
- [ ] 두 방법 결과 비교
- [ ] 중요 변수 시각화
- [ ] 비즈니스 인사이트 도출

---

# 다음 차시 예고

## [23차시] 모델 저장과 실무 배포 준비

- joblib으로 모델 저장/로드
- Pipeline 구성
- 배포 체크리스트

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습: 품질 예측 모델의 변수 중요도 분석
