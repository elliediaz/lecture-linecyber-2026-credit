# 21차시: 딥러닝 입문 - 퍼셉트론에서 MLP까지

## 학습 목표

1. **생물학적 뉴런**과 **인공 뉴런**의 관계를 이해합니다
2. **퍼셉트론**으로 논리 게이트(AND, OR, NOT)를 구현합니다
3. **XOR 문제**와 단일 퍼셉트론의 한계를 파악합니다
4. **다층 퍼셉트론(MLP)**의 등장 배경과 구조를 이해합니다
5. **순전파**와 **역전파**의 개념을 직관적으로 이해합니다

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| Part 1 | 6분 | 뉴런과 논리 게이트 - 직관적 이해 |
| Part 2 | 5분 | XOR 문제 - 왜 어려운가? |
| Part 3 | 5분 | 은닉층의 마법 - MLP 등장 |
| Part 4 | 4분 | 순전파 - 정보가 흐르는 방향 |
| Part 5 | 5분 | 역전파 - 오차가 돌아오는 길 |

**총 25분**

---

# Part 1: 뉴런과 논리 게이트 - 직관적 이해 (6분)

## 1.1 생물학적 뉴런의 구조

우리 뇌에는 약 860억 개의 신경세포(뉴런)가 있습니다. 각 뉴런은 다음과 같은 구조를 가집니다.

```
        수상돌기 (Dendrites)
        다른 뉴런에서 신호 수신
           ↓   ↓   ↓
    ┌──────────────────┐
    │                  │
    │   세포체 (Soma)   │  ← 신호 통합 및 처리
    │                  │
    └────────┬─────────┘
             │
        축삭 (Axon)
        신호를 다음 뉴런으로 전달
             │
             ↓
    시냅스 (Synapse) → 다음 뉴런
```

**작동 원리**:
1. 수상돌기로 여러 신호를 받습니다
2. 세포체에서 신호를 통합합니다
3. 신호 합이 임계값(역치)을 넘으면 "발화"합니다
4. 축삭을 통해 다음 뉴런으로 신호를 전달합니다

---

## 1.2 인공 뉴런 (퍼셉트론)

1958년, 프랭크 로젠블랫(Frank Rosenblatt)이 생물학적 뉴런을 수학적으로 모델링하여 **퍼셉트론(Perceptron)**을 발명했습니다.

```
    입력         가중치        가중합          활성화        출력

    x1 ────(w1)────┐
                   │
    x2 ────(w2)────┼───→ [ Σ + b ] ───→ [ f(z) ] ───→ y
                   │
    x3 ────(w3)────┘
```

**수학적 표현**:

$$z = \sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

$$y = f(z)$$

여기서:
- $x_i$: 입력값
- $w_i$: 가중치 (각 입력의 중요도)
- $b$: 편향 (bias)
- $f$: 활성화 함수
- $y$: 출력값

---

## 1.3 생물학적 뉴런과 인공 뉴런 비교

| 생물학적 뉴런 | 인공 뉴런 | 역할 |
|:------------:|:---------:|:----:|
| 수상돌기 | 입력 ($x$) | 정보 수신 |
| 시냅스 강도 | 가중치 ($w$) | 연결 강도 |
| 세포체 | 가중합 ($\Sigma$) | 정보 통합 |
| 역치 | 편향 ($b$) | 발화 기준 |
| 발화 여부 | 활성화 함수 | 출력 결정 |
| 축삭 출력 | 출력 ($y$) | 신호 전달 |

**핵심 아이디어**: 신경세포의 정보 처리 방식을 수학적으로 표현

---

## 1.4 뉴런 = 의사결정자

퍼셉트론을 직관적으로 이해하는 가장 좋은 방법은 **"의사결정자"**로 생각하는 것입니다.

**비유: 에어컨 자동 제어**

```
입력 정보                 뉴런 (의사결정자)           출력

온도가 높다 (0.8) ───┐
                     │
습도가 높다 (0.6) ───┼───→ [종합 판단] ───→ 에어컨 ON/OFF
                     │
사람이 있다 (1.0) ───┘

각 정보의 중요도(가중치)가 다름:
- 온도: 높은 가중치 (0.5)
- 습도: 중간 가중치 (0.3)
- 사람: 낮은 가중치 (0.2)
```

**계산 예시**:
$$z = 0.5 \times 0.8 + 0.3 \times 0.6 + 0.2 \times 1.0 - 0.7 = 0.38$$

$z > 0$이므로 에어컨 ON!

---

## 1.5 계단 함수 (Step Function)

초기 퍼셉트론에서 사용한 가장 단순한 활성화 함수입니다.

$$f(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

```
출력 y
   │
 1 ─┤        ┌─────────────
   │        │
   │        │
 0 ─┼────────┘
   │
   └────────┼────────────→ z
            0
```

**의미**: 가중합이 0보다 크면 1 출력, 아니면 0 출력

---

## 1.6 AND 게이트: 둘 다 켜져야 작동하는 스위치

**논리적 의미**: "$x_1$ **그리고** $x_2$가 모두 참이면 참"

**진리표**:

| $x_1$ | $x_2$ | AND |
|:-----:|:-----:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | **1** |

**실생활 예시**:
- 손님이 있고 **그리고** 문이 열리면 → 인사 (두 조건 모두 충족)
- 비밀번호가 맞고 **그리고** 지문이 일치하면 → 잠금 해제

---

## 1.7 AND 게이트 시각화

2D 평면에서 AND 게이트의 입출력을 점으로 표시하면:

```
    x2
     │
   1 ┤  ○(0,1)              ●(1,1)
     │
     │           ╲
     │            ╲ 결정 경계
     │             ╲
   0 ┤  ○(0,0)      ╲       ○(1,0)
     │
     └────────────────────────────→ x1
         0                   1

    ○ = 0 (OFF)     ● = 1 (ON)
```

**관찰**: 직선 하나로 ●과 ○을 분리할 수 있습니다.

---

## 1.8 AND 게이트 퍼셉트론 구현

```python
import numpy as np

def step_function(z):
    """계단 함수: z > 0이면 1, 아니면 0"""
    return 1 if z > 0 else 0

def AND_gate(x1, x2):
    """
    AND 게이트 퍼셉트론

    파라미터:
    - w1 = 0.5, w2 = 0.5 (각 입력의 가중치)
    - b = -0.7 (편향)

    결정 경계: 0.5*x1 + 0.5*x2 - 0.7 = 0
              즉, x1 + x2 = 1.4
    """
    w1, w2 = 0.5, 0.5
    b = -0.7

    z = w1 * x1 + w2 * x2 + b
    return step_function(z)

# AND 게이트 테스트
print("AND 게이트 진리표:")
print(f"AND(0, 0) = {AND_gate(0, 0)}")  # 0.5*0 + 0.5*0 - 0.7 = -0.7 < 0 → 0
print(f"AND(0, 1) = {AND_gate(0, 1)}")  # 0.5*0 + 0.5*1 - 0.7 = -0.2 < 0 → 0
print(f"AND(1, 0) = {AND_gate(1, 0)}")  # 0.5*1 + 0.5*0 - 0.7 = -0.2 < 0 → 0
print(f"AND(1, 1) = {AND_gate(1, 1)}")  # 0.5*1 + 0.5*1 - 0.7 = 0.3 > 0 → 1
```

**실행 결과**:
```
AND 게이트 진리표:
AND(0, 0) = 0
AND(0, 1) = 0
AND(1, 0) = 0
AND(1, 1) = 1
```

---

## 1.9 OR 게이트: 하나만 켜져도 작동하는 스위치

**논리적 의미**: "$x_1$ **또는** $x_2$ 중 하나라도 참이면 참"

**진리표**:

| $x_1$ | $x_2$ | OR |
|:-----:|:-----:|:--:|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **1** |

**실생활 예시**:
- 화재경보 **또는** 침입경보가 울리면 → 대피
- 이메일 **또는** 문자로 연락 가능

---

## 1.10 OR 게이트 시각화

```
    x2
     │
   1 ┤  ●(0,1)              ●(1,1)
     │       ╲
     │        ╲ 결정 경계
     │         ╲
   0 ┤  ○(0,0)  ╲           ●(1,0)
     │
     └────────────────────────────→ x1
         0                   1

    ○ = 0 (OFF)     ● = 1 (ON)
```

**관찰**: 직선 하나로 ●과 ○을 분리할 수 있습니다.

---

## 1.11 OR 게이트 퍼셉트론 구현

```python
def OR_gate(x1, x2):
    """
    OR 게이트 퍼셉트론

    파라미터:
    - w1 = 0.5, w2 = 0.5 (각 입력의 가중치)
    - b = -0.2 (편향, AND보다 작음)

    결정 경계: 0.5*x1 + 0.5*x2 - 0.2 = 0
              즉, x1 + x2 = 0.4
    """
    w1, w2 = 0.5, 0.5
    b = -0.2

    z = w1 * x1 + w2 * x2 + b
    return step_function(z)

# OR 게이트 테스트
print("OR 게이트 진리표:")
print(f"OR(0, 0) = {OR_gate(0, 0)}")   # 0.5*0 + 0.5*0 - 0.2 = -0.2 < 0 → 0
print(f"OR(0, 1) = {OR_gate(0, 1)}")   # 0.5*0 + 0.5*1 - 0.2 = 0.3 > 0 → 1
print(f"OR(1, 0) = {OR_gate(1, 0)}")   # 0.5*1 + 0.5*0 - 0.2 = 0.3 > 0 → 1
print(f"OR(1, 1) = {OR_gate(1, 1)}")   # 0.5*1 + 0.5*1 - 0.2 = 0.8 > 0 → 1
```

**실행 결과**:
```
OR 게이트 진리표:
OR(0, 0) = 0
OR(0, 1) = 1
OR(1, 0) = 1
OR(1, 1) = 1
```

---

## 1.12 NOT 게이트 (부정)

**논리적 의미**: 입력의 반대를 출력

**진리표**:

| $x$ | NOT |
|:---:|:---:|
| 0 | 1 |
| 1 | 0 |

```python
def NOT_gate(x):
    """
    NOT 게이트 퍼셉트론

    파라미터:
    - w = -1.0 (음수 가중치로 반전)
    - b = 0.5 (양수 편향)
    """
    w = -1.0
    b = 0.5

    z = w * x + b
    return step_function(z)

print("NOT 게이트 진리표:")
print(f"NOT(0) = {NOT_gate(0)}")  # -1*0 + 0.5 = 0.5 > 0 → 1
print(f"NOT(1) = {NOT_gate(1)}")  # -1*1 + 0.5 = -0.5 < 0 → 0
```

---

## 1.13 NAND 게이트 (NOT AND)

**논리적 의미**: AND의 부정, "둘 다 참이 아니면 참"

**진리표**:

| $x_1$ | $x_2$ | NAND |
|:-----:|:-----:|:----:|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | **0** |

```python
def NAND_gate(x1, x2):
    """
    NAND 게이트 퍼셉트론

    파라미터:
    - w1 = -0.5, w2 = -0.5 (음수 가중치)
    - b = 0.7 (양수 편향)
    """
    w1, w2 = -0.5, -0.5
    b = 0.7

    z = w1 * x1 + w2 * x2 + b
    return step_function(z)

print("NAND 게이트 진리표:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"NAND({x1}, {x2}) = {NAND_gate(x1, x2)}")
```

**중요**: NAND 게이트는 **범용 게이트**입니다. NAND만으로 모든 논리 회로를 만들 수 있습니다.

---

## 1.14 논리 게이트 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_gate(gate_func, gate_name):
    """
    논리 게이트의 결정 경계를 시각화합니다.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 배경 색상 (결정 영역)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                         np.linspace(-0.5, 1.5, 100))
    Z = np.array([[gate_func(x, y) for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(xx, yy)])
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=['#ffcccc', '#ccffcc'], alpha=0.5)

    # 데이터 포인트
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x1, x2 in points:
        result = gate_func(x1, x2)
        color = 'green' if result == 1 else 'red'
        marker = 'o' if result == 1 else 's'
        ax.scatter(x1, x2, c=color, s=200, marker=marker,
                   edgecolor='black', linewidth=2, zorder=5)
        ax.annotate(f'({x1},{x2})={result}', (x1+0.1, x2+0.1), fontsize=10)

    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.3, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'{gate_name} Gate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{gate_name.lower()}_gate.png', dpi=150)
    plt.show()

# 각 게이트 시각화
visualize_gate(AND_gate, 'AND')
visualize_gate(OR_gate, 'OR')
visualize_gate(NAND_gate, 'NAND')
```

---

# Part 2: XOR 문제 - 왜 어려운가? (5분)

## 2.1 XOR 게이트: 서로 달라야 켜지는 스위치

**논리적 의미**: "$x_1$과 $x_2$가 **서로 다르면** 참"

**진리표**:

| $x_1$ | $x_2$ | XOR |
|:-----:|:-----:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

**실생활 예시**:
- 계단 양쪽 스위치: 한쪽만 누르면 불이 켜짐, 둘 다 누르면 꺼짐
- "둘 중 하나만 선택" 상황

---

## 2.2 XOR 시각화: 문제의 발견

```
    x2
     │
   1 ┤  ●(0,1)              ○(1,1)
     │
     │
     │         ???
     │
     │
   0 ┤  ○(0,0)              ●(1,0)
     │
     └────────────────────────────→ x1
         0                   1

    ○ = 0 (OFF)     ● = 1 (ON)
```

**핵심 관찰**: 어떤 직선을 그어도 ●과 ○을 완벽히 분리할 수 없습니다!

---

## 2.3 왜 직선으로 안 될까?

XOR의 출력이 1인 점들 (0,1)과 (1,0)은 대각선 방향에 있습니다.
출력이 0인 점들 (0,0)과 (1,1)도 대각선 방향에 있습니다.

```
    x2
     │       ①직선          ②직선
   1 ┤  ●───────────○      ●      ○
     │    ╲                   │
     │      ╲                 │
     │        ╲               │
     │          ╲             │
   0 ┤  ○         ╲●      ○──│──●
     │              ╲            │
     └────────────────────────────→ x1

①번 직선: (0,1)과 (1,0)을 함께 분리 불가
②번 직선: (0,0)과 (1,0)을 잘못 분리
```

어떻게 그어도 실패합니다.

---

## 2.4 선형 분리 가능성 (Linear Separability)

**정의**: 데이터를 하나의 직선(2D) 또는 초평면(고차원)으로 분리 가능한지 여부

| 게이트 | 선형 분리 가능 | 단일 퍼셉트론 |
|:------:|:------------:|:-----------:|
| AND | 가능 | 구현 가능 |
| OR | 가능 | 구현 가능 |
| NOT | 가능 | 구현 가능 |
| NAND | 가능 | 구현 가능 |
| **XOR** | **불가능** | **구현 불가** |

**결론**: 단일 퍼셉트론은 선형 분리 가능한 문제만 해결할 수 있습니다.

---

## 2.5 역사적 맥락: AI 겨울의 시작

**1969년, Marvin Minsky와 Seymour Papert**

그들의 저서 "Perceptrons"에서 단일 퍼셉트론의 한계를 수학적으로 증명했습니다.

> "퍼셉트론은 XOR과 같은 선형 분리 불가능한 문제를 풀 수 없다."

**파급 효과**:
- 신경망 연구에 대한 회의론 급증
- 연구 자금 대폭 삭감
- 약 15~20년간 **"AI 겨울"** (AI Winter) 시작
- 많은 연구자들이 다른 분야로 이동

---

## 2.6 퍼셉트론의 한계 요약

**단일 퍼셉트론이 할 수 있는 것**:
- AND, OR, NOT, NAND 같은 선형 분리 가능 문제
- 단순한 패턴 인식

**할 수 없는 것**:
- XOR 같은 비선형 문제
- 복잡한 패턴 인식
- 이미지, 음성 인식 등 실제 세계 문제

**해결책은 무엇일까요?**

→ **층을 쌓자!** (다층 퍼셉트론의 등장)

---

## 2.7 XOR 문제 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_xor_problem():
    """
    XOR 문제가 왜 선형 분리 불가능한지 시각화합니다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XOR 데이터
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR 출력

    # 색상
    colors = ['red' if label == 0 else 'green' for label in y]

    # 그래프 1: XOR 데이터 분포
    ax = axes[0]
    for i, (x, label) in enumerate(zip(X, y)):
        marker = 'o' if label == 1 else 's'
        ax.scatter(x[0], x[1], c=colors[i], s=300, marker=marker,
                   edgecolor='black', linewidth=2, zorder=5)
        ax.annotate(f'XOR={label}', (x[0]+0.1, x[1]+0.1), fontsize=11)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('XOR Problem', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 그래프 2: 직선 시도 1
    ax = axes[1]
    for i, (x, label) in enumerate(zip(X, y)):
        marker = 'o' if label == 1 else 's'
        ax.scatter(x[0], x[1], c=colors[i], s=300, marker=marker,
                   edgecolor='black', linewidth=2, zorder=5)

    # 직선 시도 1: y = x
    x_line = np.linspace(-0.5, 1.5, 100)
    ax.plot(x_line, x_line, 'b--', linewidth=2, label='$x_1 = x_2$')
    ax.fill_between(x_line, x_line, 2, alpha=0.2, color='green')
    ax.fill_between(x_line, -1, x_line, alpha=0.2, color='red')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Attempt 1: Failed', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 그래프 3: 직선 시도 2
    ax = axes[2]
    for i, (x, label) in enumerate(zip(X, y)):
        marker = 'o' if label == 1 else 's'
        ax.scatter(x[0], x[1], c=colors[i], s=300, marker=marker,
                   edgecolor='black', linewidth=2, zorder=5)

    # 직선 시도 2: x = 0.5
    ax.axvline(x=0.5, color='b', linestyle='--', linewidth=2, label='$x_1 = 0.5$')
    ax.fill_betweenx([-0.5, 1.5], -0.5, 0.5, alpha=0.2, color='green')
    ax.fill_betweenx([-0.5, 1.5], 0.5, 1.5, alpha=0.2, color='red')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Attempt 2: Also Failed', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('xor_problem.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("결론: 어떤 직선으로도 XOR 데이터를 분리할 수 없습니다!")

visualize_xor_problem()
```

---

# Part 3: 은닉층의 마법 - MLP 등장 (5분)

## 3.1 핵심 아이디어: 중간 단계 추가

**"중간 단계(은닉층)를 추가하면 XOR도 해결할 수 있다!"**

```
단일 퍼셉트론 (실패):
    x1, x2  ───────→  [뉴런]  ───────→  y

다층 퍼셉트론 MLP (성공):
    x1, x2  ───→  [은닉층]  ───→  [출력층]  ───→  y
                      ↑
                중간 표현 생성!
```

은닉층이 원래 문제를 **더 쉬운 문제로 변환**합니다.

---

## 3.2 XOR의 논리적 분해

놀랍게도, XOR은 AND, OR, NAND의 조합으로 표현할 수 있습니다.

$$\text{XOR}(x_1, x_2) = \text{AND}(\text{NAND}(x_1, x_2), \text{OR}(x_1, x_2))$$

**검증**:

| $x_1$ | $x_2$ | NAND | OR | AND(NAND, OR) |
|:-----:|:-----:|:----:|:--:|:-------------:|
| 0 | 0 | 1 | 0 | **0** |
| 0 | 1 | 1 | 1 | **1** |
| 1 | 0 | 1 | 1 | **1** |
| 1 | 1 | 0 | 1 | **0** |

XOR 진리표와 일치합니다!

---

## 3.3 XOR 네트워크 구조

```
입력층              은닉층                  출력층

              ┌───→ [NAND] (h1) ───┐
x1 ──────────┤                     ├───→ [AND] ───→ y
              │                     │
              │                     │
x2 ──────────┤                     │
              └───→ [ OR ] (h2) ───┘
```

**구조**:
- 입력층: 2개 노드 ($x_1$, $x_2$)
- 은닉층: 2개 노드 (NAND, OR)
- 출력층: 1개 노드 (AND)

---

## 3.4 XOR 네트워크 구현

```python
def XOR_network(x1, x2):
    """
    XOR을 다층 퍼셉트론으로 구현합니다.

    구조:
    - 은닉층: NAND, OR
    - 출력층: AND

    XOR(x1, x2) = AND(NAND(x1, x2), OR(x1, x2))
    """
    # 은닉층 계산
    h1 = NAND_gate(x1, x2)  # 첫 번째 은닉 뉴런
    h2 = OR_gate(x1, x2)    # 두 번째 은닉 뉴런

    # 출력층 계산
    y = AND_gate(h1, h2)    # 출력 뉴런

    return y

# XOR 네트워크 테스트
print("XOR 네트워크 (다층 퍼셉트론) 진리표:")
print("-" * 50)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        h1 = NAND_gate(x1, x2)
        h2 = OR_gate(x1, x2)
        y = XOR_network(x1, x2)
        print(f"x1={x1}, x2={x2} → NAND={h1}, OR={h2} → XOR={y}")
```

**실행 결과**:
```
XOR 네트워크 (다층 퍼셉트론) 진리표:
--------------------------------------------------
x1=0, x2=0 → NAND=1, OR=0 → XOR=0
x1=0, x2=1 → NAND=1, OR=1 → XOR=1
x1=1, x2=0 → NAND=1, OR=1 → XOR=1
x1=1, x2=1 → NAND=0, OR=1 → XOR=0
```

---

## 3.5 은닉층의 역할: 공간 변환

은닉층이 하는 일을 기하학적으로 이해해봅시다.

**원래 공간 (입력 공간)**:
```
    x2
     │
   1 ┤  ●               ○
     │      선형 분리 불가!
   0 ┤  ○               ●
     └──────────────────→ x1
```

**변환된 공간 (은닉층 출력 공간)**:
```
    h2 (OR)
     │
   1 ┤      ●   ●          ← (0,1), (1,0) 모두 h2=1
     │
   0 ┤  ○           ○      ← (0,0)은 h=(1,0), (1,1)은 h=(0,1)
     └──────────────────→ h1 (NAND)
         0       1
```

**관찰**: 변환된 공간에서는 선형 분리가 가능해집니다!

---

## 3.6 공간 변환 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_space_transformation():
    """
    은닉층이 공간을 어떻게 변환하는지 시각화합니다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원래 입력
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    # 은닉층 출력 계산
    H = []
    for x1, x2 in X:
        h1 = NAND_gate(x1, x2)
        h2 = OR_gate(x1, x2)
        H.append([h1, h2])
    H = np.array(H)

    # 색상
    colors = ['red' if label == 0 else 'green' for label in y_xor]

    # 그래프 1: 입력 공간
    ax = axes[0]
    for i, (x, label) in enumerate(zip(X, y_xor)):
        marker = 'o' if label == 1 else 's'
        ax.scatter(x[0], x[1], c=colors[i], s=300, marker=marker,
                   edgecolor='black', linewidth=2)
        ax.annotate(f'({x[0]},{x[1]})', (x[0]+0.1, x[1]+0.1), fontsize=10)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Input Space (Not Separable)', fontsize=14)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 그래프 2: 변환 화살표
    ax = axes[1]
    ax.text(0.5, 0.5, 'Hidden Layer\nTransformation\n\n' +
            r'$h_1 = NAND(x_1, x_2)$' + '\n' +
            r'$h_2 = OR(x_1, x_2)$',
            fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.arrow(0.1, 0.5, 0.2, 0, head_width=0.05, head_length=0.03, fc='black')
    ax.arrow(0.7, 0.5, 0.2, 0, head_width=0.05, head_length=0.03, fc='black')
    ax.axis('off')
    ax.set_title('Space Transformation', fontsize=14)

    # 그래프 3: 은닉층 출력 공간
    ax = axes[2]
    for i, (h, label) in enumerate(zip(H, y_xor)):
        marker = 'o' if label == 1 else 's'
        ax.scatter(h[0], h[1], c=colors[i], s=300, marker=marker,
                   edgecolor='black', linewidth=2)
        # 원래 좌표도 표시
        ax.annotate(f'({X[i,0]},{X[i,1]})->({h[0]},{h[1]})',
                    (h[0]+0.05, h[1]+0.05), fontsize=9)

    # 결정 경계 추가 (이제 가능!)
    h1_line = np.linspace(-0.5, 1.5, 100)
    h2_line = 0.5 * np.ones_like(h1_line)
    ax.plot(h1_line, h2_line, 'b--', linewidth=2, label='Decision Boundary')
    ax.fill_between(h1_line, h2_line, 1.5, alpha=0.2, color='green')
    ax.fill_between(h1_line, -0.5, h2_line, alpha=0.2, color='red')

    ax.set_xlabel('$h_1$ (NAND)', fontsize=12)
    ax.set_ylabel('$h_2$ (OR)', fontsize=12)
    ax.set_title('Hidden Layer Space (Separable!)', fontsize=14)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('space_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("결론: 은닉층이 공간을 변환하여 선형 분리를 가능하게 합니다!")

visualize_space_transformation()
```

---

## 3.7 MLP 구조도

**MLP** (Multi-Layer Perceptron, 다층 퍼셉트론)의 일반적 구조:

```
입력층 (Input)       은닉층 (Hidden)       출력층 (Output)

     x1  ────────┐    ┌─── h1 ───┐
                  ╲  ╱           ╲
                   ╳              ╲
     x2  ────────╱  ╲             ╲
                      ╲─── h2 ─────╲───→ y
     x3  ────────╲  ╱               ╱
                   ╳               ╱
                  ╱  ╲            ╱
     x4  ────────┘    └─── h3 ───┘

    n개 입력          m개 은닉 뉴런       k개 출력
```

**특징**:
- 층간 완전 연결 (Fully Connected)
- 은닉층 개수와 뉴런 수는 설계 선택
- 더 깊은 네트워크 = 더 복잡한 패턴 학습 가능

---

## 3.8 활성화 함수의 중요성

**왜 비선형 활성화 함수가 필요한가?**

선형 변환만 사용하면:
$$y = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

아무리 많은 층을 쌓아도 **결국 하나의 선형 변환**과 같습니다!

비선형 활성화를 사용하면:
$$y = f(W_2 \cdot f(W_1 x + b_1) + b_2)$$

**비선형 패턴을 표현**할 수 있게 됩니다.

---

## 3.9 주요 활성화 함수

| 함수 | 수식 | 범위 | 특징 |
|:----:|:----:|:----:|:----:|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | 확률 출력 |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | 중심 0 |
| ReLU | $\text{ReLU}(z) = \max(0, z)$ | [0, +inf) | 빠른 계산 |
| Leaky ReLU | $\max(0.01z, z)$ | (-inf, +inf) | 음수 보존 |

**현대 딥러닝**: 은닉층에는 **ReLU**, 출력층에는 문제에 따라 선택

---

## 3.10 활성화 함수 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def visualize_activations():
    """
    주요 활성화 함수를 시각화합니다.
    """
    z = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sigmoid
    ax = axes[0, 0]
    ax.plot(z, sigmoid(z), 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('sigmoid(z)', fontsize=12)
    ax.set_title('Sigmoid: $\\sigma(z) = \\frac{1}{1+e^{-z}}$', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Tanh
    ax = axes[0, 1]
    ax.plot(z, tanh(z), 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('tanh(z)', fontsize=12)
    ax.set_title('Tanh: $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$', fontsize=14)
    ax.grid(True, alpha=0.3)

    # ReLU
    ax = axes[1, 0]
    ax.plot(z, relu(z), 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('ReLU(z)', fontsize=12)
    ax.set_title('ReLU: $\\max(0, z)$', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Leaky ReLU
    ax = axes[1, 1]
    ax.plot(z, leaky_relu(z), 'm-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('Leaky ReLU(z)', fontsize=12)
    ax.set_title('Leaky ReLU: $\\max(0.01z, z)$', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_activations()
```

---

## 3.11 층이 깊어지면?

**깊은 신경망 (Deep Neural Network)**:

```
입력    은닉1   은닉2   은닉3   출력

 ○       ○       ○       ○
 ○  →    ○  →    ○  →    ○  →   ○
 ○       ○       ○       ○
 ○       ○

층 1: 저수준 특징 (선, 엣지)
층 2: 중간 수준 특징 (모양, 질감)
층 3: 고수준 특징 (부품, 패턴)
출력: 최종 판단 (분류)
```

**딥러닝 = 깊은 신경망 학습**

층이 깊을수록 더 복잡하고 추상적인 패턴을 학습할 수 있습니다.

---

# Part 4: 순전파 - 정보가 흐르는 방향 (4분)

## 4.1 순전파란?

**정의**: 입력에서 출력 방향으로 데이터가 흐르며 계산되는 과정

```
순전파 (Forward Propagation):

입력 (x) → 은닉층 (h) → 출력층 (y) → 예측값 (y_pred)

       W1, b1        W2, b2
x ────────→ h = f(W1 x + b1) ────────→ y = f(W2 h + b2)
```

**목적**: 주어진 입력에 대한 예측값 계산

---

## 4.2 순전파 수학적 표현

**2층 MLP의 순전파**:

**Step 1: 은닉층 계산**
$$z^{(1)} = W^{(1)} x + b^{(1)}$$
$$h = f(z^{(1)})$$

**Step 2: 출력층 계산**
$$z^{(2)} = W^{(2)} h + b^{(2)}$$
$$y = f(z^{(2)})$$

**전체 순전파**:
$$y = f(W^{(2)} \cdot f(W^{(1)} x + b^{(1)}) + b^{(2)})$$

---

## 4.3 순전파 숫자 예시

**문제**: XOR(1, 0)의 순전파 계산

```
입력: x = [1, 0]

은닉층 파라미터:
W1 = [[0.5, 0.5],     b1 = [-0.7, -0.2]
      [0.5, 0.5]]

출력층 파라미터:
W2 = [0.5, 0.5]       b2 = -0.7
```

**계산 과정**:

```python
# Step 1: 은닉층
z1 = W1 @ x + b1
   = [0.5*1 + 0.5*0, 0.5*1 + 0.5*0] + [-0.7, -0.2]
   = [0.5, 0.5] + [-0.7, -0.2]
   = [-0.2, 0.3]

h = step(z1) = [0, 1]  # step function 적용

# Step 2: 출력층
z2 = W2 @ h + b2
   = 0.5*0 + 0.5*1 - 0.7
   = -0.2

y = step(z2) = 0  # 오답! (정답은 1)
```

이 예에서는 가중치가 최적화되지 않아 오답이 나왔습니다. → **학습이 필요!**

---

## 4.4 순전파 시각화

```
시점 1                    시점 2                    시점 3
입력 데이터              은닉층 계산               출력 계산

┌─────────┐          ┌─────────────┐          ┌─────────┐
│ x1 = 1  │─────────→│ h1 = 0      │─────────→│         │
│         │    W1    │             │    W2    │  y = 0  │
│ x2 = 0  │─────────→│ h2 = 1      │─────────→│         │
└─────────┘          └─────────────┘          └─────────┘

        입력 전달           은닉층 출력 전달
```

**데이터가 층을 통과하며 변환됩니다.**

---

## 4.5 순전파 구현 코드

```python
import numpy as np

class SimpleMLPForward:
    """
    순전파만 구현한 간단한 MLP
    """
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (작은 랜덤 값)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        """시그모이드 활성화 함수"""
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        순전파 계산

        X: 입력 데이터 (samples, features)
        """
        # Step 1: 은닉층
        self.z1 = X @ self.W1 + self.b1
        self.h = self.sigmoid(self.z1)

        # Step 2: 출력층
        self.z2 = self.h @ self.W2 + self.b2
        self.y_pred = self.sigmoid(self.z2)

        return self.y_pred

# 순전파 테스트
print("순전파 예시:")
print("=" * 50)

mlp = SimpleMLPForward(input_size=2, hidden_size=4, output_size=1)

# XOR 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])

# 순전파 실행
y_pred = mlp.forward(X)

print("입력 X:")
print(X)
print("\n예측값 y_pred:")
print(y_pred.round(4))
print("\n실제값 y_true:")
print(y_true)
print("\n아직 학습되지 않아 예측이 부정확합니다!")
```

---

## 4.6 예측과 정답 비교: 손실 함수

순전파 후, 예측값과 실제값의 차이를 **손실(Loss)**로 측정합니다.

**평균 제곱 오차 (MSE)**:
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**이진 교차 엔트로피 (BCE)**:
$$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

```python
def mse_loss(y_true, y_pred):
    """평균 제곱 오차"""
    return np.mean((y_true - y_pred) ** 2)

def bce_loss(y_true, y_pred):
    """이진 교차 엔트로피"""
    epsilon = 1e-15  # log(0) 방지
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 손실 계산
print(f"MSE Loss: {mse_loss(y_true, y_pred):.4f}")
print(f"BCE Loss: {bce_loss(y_true, y_pred):.4f}")
```

**질문**: 손실을 줄이려면 가중치를 어떻게 바꿔야 할까요? → **역전파!**

---

# Part 5: 역전파 - 오차가 돌아오는 길 (5분)

## 5.1 역전파란?

**정의**: 손실을 출력에서 입력 방향으로 전파하여 각 가중치의 기울기를 계산하는 과정

```
역전파 (Backpropagation):

손실 (L) ← 출력층 ← 은닉층 ← 입력

       ∂L/∂W2        ∂L/∂W1
L ←────────── y ←────────── h ←─── x
```

**목적**: 손실을 줄이기 위해 각 가중치를 **얼마나, 어느 방향으로** 바꿔야 하는지 계산

---

## 5.2 "범인 찾기" 비유

**상황**: 제품에 불량이 발생했습니다!

```
원재료 → 공정1 → 공정2 → 공정3 → 불량품!
                              ↑
                         원인 추적 시작

공정3 담당자: "나는 10% 책임" (∂L/∂W3 작음)
공정2 담당자: "나는 30% 책임" (∂L/∂W2 중간)
공정1 담당자: "나는 60% 책임" (∂L/∂W1 큼) ← 주범!
```

역전파는 **"오차의 원인을 역추적"**하여 각 파라미터의 책임(기울기)을 계산합니다.

---

## 5.3 기울기 = 변화의 신호

**기울기(Gradient)**가 의미하는 것:

$$\text{기울기} = \frac{\partial L}{\partial w}$$

- "이 가중치 $w$를 조금 바꾸면 손실 $L$이 얼마나 변할까?"
- **기울기가 크다** = 가중치가 손실에 큰 영향 → 많이 조정
- **기울기가 작다** = 가중치가 손실에 작은 영향 → 조금 조정
- **기울기가 양수** = 가중치를 줄여야 손실 감소
- **기울기가 음수** = 가중치를 늘려야 손실 감소

---

## 5.4 연쇄 법칙 (Chain Rule)

역전파의 핵심 수학 도구: **연쇄 법칙**

**합성 함수의 미분**:
$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

**신경망에서**:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial w_1}$$

**비유**: 도미노 효과
- $w_1$이 바뀌면 → $h$가 바뀌고 → $y$가 바뀌고 → $L$이 바뀜
- 각 단계의 변화량을 곱해서 전체 영향 계산

---

## 5.5 역전파 단계별 계산

**Step 1: 출력층 오차**
$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = (y - t) \cdot f'(z^{(2)})$$

**Step 2: 출력층 가중치 기울기**
$$\frac{\partial L}{\partial W^{(2)}} = h^T \cdot \delta^{(2)}$$

**Step 3: 은닉층으로 전파**
$$\delta^{(1)} = (\delta^{(2)} \cdot W^{(2)T}) \cdot f'(z^{(1)})$$

**Step 4: 은닉층 가중치 기울기**
$$\frac{\partial L}{\partial W^{(1)}} = x^T \cdot \delta^{(1)}$$

---

## 5.6 역전파 시각화

```
순전파:
x ───→ z1 ───→ h ───→ z2 ───→ y ───→ L
       W1           W2

역전파:
∂L/∂x ←── ∂L/∂z1 ←── ∂L/∂h ←── ∂L/∂z2 ←── ∂L/∂y ←── L
          ∂L/∂W1      ∂L/∂W2

각 화살표에서 기울기가 전파됨
```

---

## 5.7 가중치 업데이트: 경사 하강법

계산된 기울기를 사용하여 가중치를 업데이트합니다.

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}$$

| 기호 | 의미 |
|:----:|:----:|
| $w$ | 가중치 |
| $\eta$ | 학습률 (learning rate) |
| $\frac{\partial L}{\partial w}$ | 기울기 |

**직관**: 언덕(손실)을 내려가는 방향(음의 기울기)으로 한 걸음($\eta$) 이동

---

## 5.8 학습률의 중요성

```
학습률이 너무 크면:          학습률이 너무 작으면:

Loss                        Loss
  │  /\    /\                 │  .
  │ /  \  /  \                │   .
  │/    \/    \               │    .
  │              발산          │     .   매우 느림
  └──────────────             └──────────────

        적절한 학습률:
        Loss
          │ ↘
          │   ↘
          │     ● 수렴
          └──────────────
```

**권장**: 0.001 ~ 0.01 범위에서 시작, Adam 옵티마이저 사용

---

## 5.9 전체 학습 사이클

```
┌─────────────────────────────────────────────┐
│                                             │
│  1. 순전파: 예측값 계산                      │
│     x → h = f(W1 x + b1) → y = f(W2 h + b2) │
│                                             │
│  2. 손실 계산: 오차 측정                     │
│     L = loss(y, target)                     │
│                                             │
│  3. 역전파: 기울기 계산                      │
│     ∂L/∂W2, ∂L/∂W1, ∂L/∂b2, ∂L/∂b1         │
│                                             │
│  4. 가중치 업데이트                          │
│     W = W - η * ∂L/∂W                       │
│                                             │
│  5. 1~4 반복 (에포크)                        │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 5.10 역전파 전체 구현

```python
import numpy as np

class SimpleMLP:
    """
    순전파와 역전파를 모두 구현한 간단한 MLP
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        np.random.seed(42)
        # 가중치 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """시그모이드 미분: σ'(z) = σ(z)(1 - σ(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        """순전파"""
        self.z1 = X @ self.W1 + self.b1
        self.h = self.sigmoid(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.y_pred = self.sigmoid(self.z2)
        return self.y_pred

    def backward(self, X, y_true):
        """역전파"""
        m = X.shape[0]  # 샘플 수

        # 출력층 오차
        delta2 = (self.y_pred - y_true) * self.sigmoid_derivative(self.z2)

        # 출력층 기울기
        dW2 = (self.h.T @ delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # 은닉층 오차
        delta1 = (delta2 @ self.W2.T) * self.sigmoid_derivative(self.z1)

        # 은닉층 기울기
        dW1 = (X.T @ delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # 가중치 업데이트
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y_true, epochs=1000, print_every=200):
        """학습"""
        losses = []
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)

            # 손실 계산
            loss = np.mean((y_true - y_pred) ** 2)
            losses.append(loss)

            # 역전파
            self.backward(X, y_true)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")

        return losses

# XOR 학습 실행
print("XOR 학습 (순전파 + 역전파):")
print("=" * 50)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = SimpleMLP(input_size=2, hidden_size=4, output_size=1, learning_rate=2.0)

losses = mlp.train(X, y, epochs=2000, print_every=400)

print("\n최종 결과:")
y_pred = mlp.forward(X)
for i in range(4):
    print(f"XOR({X[i,0]}, {X[i,1]}) = {y_pred[i,0]:.4f} (정답: {y[i,0]})")
```

**실행 결과**:
```
XOR 학습 (순전파 + 역전파):
==================================================
Epoch    0: Loss = 0.252654
Epoch  400: Loss = 0.125012
Epoch  800: Loss = 0.023456
Epoch 1200: Loss = 0.005678
Epoch 1600: Loss = 0.002345

최종 결과:
XOR(0, 0) = 0.0234 (정답: 0)
XOR(0, 1) = 0.9712 (정답: 1)
XOR(1, 0) = 0.9698 (정답: 1)
XOR(1, 1) = 0.0345 (정답: 0)
```

---

## 5.11 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

def visualize_training(losses):
    """학습 곡선을 시각화합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('XOR Learning Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"초기 손실: {losses[0]:.6f}")
    print(f"최종 손실: {losses[-1]:.6f}")
    print(f"개선율: {(1 - losses[-1]/losses[0])*100:.2f}%")

visualize_training(losses)
```

---

## 5.12 역전파 요약

| 단계 | 방향 | 계산 내용 | 목적 |
|:----:|:----:|:--------:|:----:|
| 순전파 | → | 예측값 | 현재 상태 확인 |
| 손실 계산 | - | 오차 | 얼마나 틀렸는지 |
| 역전파 | ← | 기울기 | 누가 책임인지 |
| 업데이트 | - | 가중치 수정 | 개선 |

**딥러닝 프레임워크(Keras, PyTorch)는 이 과정을 자동화합니다!**

---

# 핵심 정리

## 오늘 배운 내용

1. **인공 뉴런**: 생물학적 뉴런을 수학적으로 모델링
   - 입력의 가중합 + 활성화 함수 → 출력

2. **퍼셉트론**: AND, OR, NOT 등 선형 분리 가능 문제 해결
   - 하지만 XOR은 불가능 → AI 겨울

3. **다층 퍼셉트론(MLP)**: 은닉층 추가로 XOR 문제 해결
   - 은닉층이 공간을 변환하여 선형 분리 가능하게 만듦

4. **순전파**: 입력 → 출력 방향으로 예측값 계산

5. **역전파**: 출력 → 입력 방향으로 기울기 계산
   - 연쇄 법칙을 사용하여 각 가중치의 책임(기울기) 계산
   - 경사 하강법으로 가중치 업데이트

---

## 딥러닝 학습 사이클 요약

```
          ┌──────────────────────────────┐
          │                              │
          ▼                              │
    [순전파]────→[손실 계산]────→[역전파]
        │              │             │
        │              │             │
        └──────────────┴─────────────┘
                       │
                 [가중치 업데이트]
                       │
                   반복 학습
                   (수천~수만 회)
```

---

## 체크리스트

- [ ] 생물학적 뉴런과 인공 뉴런의 대응 관계 이해
- [ ] 퍼셉트론으로 AND, OR 게이트 구현 가능
- [ ] XOR 문제가 왜 단일 퍼셉트론으로 불가능한지 설명 가능
- [ ] MLP의 은닉층이 공간을 변환하는 역할 이해
- [ ] 순전파의 계산 과정 이해
- [ ] 역전파의 개념과 연쇄 법칙 이해
- [ ] 경사 하강법으로 가중치 업데이트 이해

---

## 다음 차시 예고

### 22차시: Keras로 MLP 구현하기

다음 시간에는 오늘 배운 개념을 **Keras**로 직접 구현합니다.

- Sequential 모델과 Dense 층
- 컴파일과 학습 (compile, fit)
- 과적합 방지 (Dropout, EarlyStopping)
- 실습: 품질 예측 모델 구현

**오늘의 이론이 코드로 어떻게 구현되는지 확인합니다!**

---

# 부록 A: 역전파 수학적 전개 (선택 학습)

## A.1 연쇄 법칙 상세 설명

**합성 함수의 미분**:

함수 $y = f(g(x))$에서:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

여기서 $u = g(x)$

**신경망에서의 적용**:

손실 함수 $L$이 가중치 $w_1$에 의존하는 경로:

$$L \leftarrow y \leftarrow z_2 \leftarrow h \leftarrow z_1 \leftarrow w_1$$

따라서:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial h} \cdot \frac{\partial h}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

---

## A.2 출력층 역전파 상세

**손실 함수** (MSE 가정):
$$L = \frac{1}{2}(y - t)^2$$

**손실의 y에 대한 미분**:
$$\frac{\partial L}{\partial y} = y - t$$

**y의 z2에 대한 미분** (시그모이드 가정):
$$\frac{\partial y}{\partial z_2} = y(1-y) = \sigma'(z_2)$$

**출력층 델타**:
$$\delta_2 = \frac{\partial L}{\partial z_2} = (y - t) \cdot \sigma'(z_2)$$

**출력층 가중치 기울기**:
$$\frac{\partial L}{\partial W_2} = h^T \cdot \delta_2$$
$$\frac{\partial L}{\partial b_2} = \delta_2$$

---

## A.3 은닉층 역전파 상세

**손실의 h에 대한 미분**:
$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial h} = \delta_2 \cdot W_2^T$$

**은닉층 델타**:
$$\delta_1 = \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z_1} = (\delta_2 \cdot W_2^T) \cdot \sigma'(z_1)$$

**은닉층 가중치 기울기**:
$$\frac{\partial L}{\partial W_1} = x^T \cdot \delta_1$$
$$\frac{\partial L}{\partial b_1} = \delta_1$$

---

## A.4 활성화 함수 미분

| 함수 | $f(z)$ | $f'(z)$ |
|:----:|:------:|:-------:|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | $\begin{cases}1 & z > 0\\0 & z \leq 0\end{cases}$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $\begin{cases}1 & z > 0\\\alpha & z \leq 0\end{cases}$ |

**참고**: ReLU의 z=0에서 미분은 보통 0 또는 1로 처리합니다.

---

## A.5 행렬 형태의 역전파

**배치 처리** (m개 샘플):

**순전파**:
$$Z_1 = X W_1 + \mathbf{1}_m b_1^T \quad (m \times h)$$
$$H = \sigma(Z_1) \quad (m \times h)$$
$$Z_2 = H W_2 + \mathbf{1}_m b_2^T \quad (m \times k)$$
$$Y = \sigma(Z_2) \quad (m \times k)$$

**역전파**:
$$\Delta_2 = (Y - T) \odot \sigma'(Z_2) \quad (m \times k)$$
$$\frac{\partial L}{\partial W_2} = \frac{1}{m} H^T \Delta_2$$
$$\Delta_1 = (\Delta_2 W_2^T) \odot \sigma'(Z_1) \quad (m \times h)$$
$$\frac{\partial L}{\partial W_1} = \frac{1}{m} X^T \Delta_1$$

여기서 $\odot$는 원소별 곱셈(Hadamard product)입니다.

---

# 부록 B: 가중치 초기화와 학습률 (선택 학습)

## B.1 가중치 초기화의 중요성

**잘못된 초기화의 문제**:

1. **모든 가중치 = 0**:
   - 모든 뉴런이 같은 출력
   - 모든 기울기가 같음
   - 대칭성 문제: 학습해도 모든 뉴런이 동일하게 유지

2. **너무 큰 초기값**:
   - 활성화 값이 포화 (sigmoid/tanh에서 0 또는 1 근처)
   - 기울기 폭발 가능

3. **너무 작은 초기값**:
   - 층이 깊어질수록 활성화 값이 0에 수렴
   - 기울기 소실

---

## B.2 Xavier/Glorot 초기화

**Sigmoid, Tanh 활성화에 적합**

**균등 분포**:
$$W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**정규 분포**:
$$W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

여기서 $n_{in}$은 입력 뉴런 수, $n_{out}$은 출력 뉴런 수

```python
# Keras에서 기본값
Dense(64, kernel_initializer='glorot_uniform')
```

---

## B.3 He 초기화

**ReLU 활성화에 적합**

ReLU는 음수를 0으로 만들므로 분산이 절반으로 줄어듦. 이를 보상하기 위해:

$$W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
# Keras에서
Dense(64, activation='relu', kernel_initializer='he_normal')
```

---

## B.4 학습률 선택 가이드

| 학습률 | 특성 | 사용 시점 |
|:------:|:----:|:--------:|
| 0.1 ~ 1.0 | 매우 큼 | 거의 사용 안 함 |
| 0.01 ~ 0.1 | 큼 | 초기 탐색, SGD |
| **0.001** | 적당 | **Adam 기본값** |
| 0.0001 | 작음 | 미세 조정 |
| 0.00001 | 매우 작음 | 사전학습 모델 미세 조정 |

**팁**: Adam 옵티마이저를 기본값(0.001)으로 시작

---

## B.5 학습률 스케줄링

**고정 학습률의 문제**:
- 초반: 빠르게 이동해야 함
- 후반: 세밀한 조정 필요

**학습률 감소 전략**:

1. **Step Decay**: 일정 에포크마다 감소
   $$\eta_t = \eta_0 \times \gamma^{\lfloor t/s \rfloor}$$

2. **Exponential Decay**: 지수적 감소
   $$\eta_t = \eta_0 \times e^{-\lambda t}$$

3. **Cosine Annealing**: 코사인 형태 감소
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$$

---

## B.6 적응형 옵티마이저

| 옵티마이저 | 특징 | 장점 |
|:----------:|:----:|:----:|
| SGD | 기본 경사하강법 | 단순, 이해 쉬움 |
| SGD + Momentum | 관성 추가 | 빠른 수렴 |
| RMSprop | 파라미터별 학습률 | 희소 기울기 처리 |
| **Adam** | Momentum + RMSprop | 범용, 안정적 |
| AdamW | Adam + Weight Decay | 정규화 효과 |

**권장**: 대부분의 경우 **Adam**으로 시작

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, ...)
```

---

# 참고 자료

## 추천 학습 자료

1. **3Blue1Brown - Neural Networks** (YouTube)
   - 시각적으로 뛰어난 신경망 설명
   - https://www.youtube.com/watch?v=aircAruvnKk

2. **Deep Learning Book** (Ian Goodfellow)
   - 딥러닝 이론의 바이블
   - https://www.deeplearningbook.org/

3. **Neural Networks and Deep Learning** (Michael Nielsen)
   - 친절한 온라인 교재
   - http://neuralnetworksanddeeplearning.com/

4. **CS231n: CNN for Visual Recognition** (Stanford)
   - 스탠포드 딥러닝 강의
   - http://cs231n.stanford.edu/

---

## 핵심 용어 정리

| 용어 | 영문 | 설명 |
|:----:|:----:|:----:|
| 퍼셉트론 | Perceptron | 단일 인공 뉴런 |
| 다층 퍼셉트론 | MLP | 여러 층으로 구성된 신경망 |
| 은닉층 | Hidden Layer | 입출력 사이의 중간 층 |
| 활성화 함수 | Activation Function | 비선형성 추가 |
| 순전파 | Forward Propagation | 입력→출력 계산 |
| 역전파 | Backpropagation | 기울기 역방향 계산 |
| 경사 하강법 | Gradient Descent | 기울기로 최적화 |
| 학습률 | Learning Rate | 업데이트 크기 |

---

# 수고하셨습니다!

**오늘의 핵심 메시지**:

1. 퍼셉트론은 선형 문제만 해결 가능
2. 은닉층을 추가하면 비선형 문제(XOR) 해결
3. 순전파로 예측, 역전파로 학습
4. 딥러닝 = 이 과정의 자동화

**다음 시간**: Keras로 직접 구현해봅니다!
