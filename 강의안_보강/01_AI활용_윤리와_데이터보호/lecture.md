# [1차시] AI 활용 윤리와 데이터 보호

## 학습 목표

이 차시를 마치면 다음을 수행할 수 있음:

1. **AI 윤리 4대 원칙**을 설명하고 제조 현장에 적용함
2. **데이터 보안 사고 유형**과 구체적인 예방 방법을 이해함
3. **AI 생성물의 저작권 문제**와 법적 쟁점을 파악함
4. **윤리 점검 체크리스트**를 활용하여 AI 프로젝트를 검토함

---

## 강의 구성

| 파트 | 대주제 | 시간 |
|:----:|--------|:----:|
| 1 | AI 시대와 제조업의 변화 | 5분 |
| 2 | AI 윤리 4대 원칙 심화 | 10분 |
| 3 | 데이터 보안 사고 사례 분석 | 8분 |
| 4 | AI 저작권과 법적 이슈 | 5분 |
| 5 | 정리 및 Q&A | 2분 |

---

## Part 1. AI 시대와 제조업의 변화

### 개념 설명

AI의 역사는 1956년 다트머스 회의에서 용어가 탄생한 이래 지속적으로 발전해왔음. 2022년 ChatGPT 등장으로 생성형 AI가 대중화되었으며, 2024년에는 제조업에서 AI 본격 도입이 이루어지고 있음.

```
1956년: AI 용어 탄생 (다트머스 회의)
    |
1980년대: 전문가 시스템의 시대
    |
1997년: Deep Blue, 체스 세계 챔피언 격파
    |
2016년: AlphaGo, 이세돌 9단 격파
    |
2022년: ChatGPT 등장, 생성형 AI 대중화
    |
2024년: 제조업 AI 본격 도입
```

### 제조업 AI 도입 현황 (2024년)

| 구분 | 비율 |
|------|------|
| AI 도입 완료 | 23% |
| 도입 진행 중 | 35% |
| 도입 검토 중 | 28% |
| 계획 없음 | 14% |

58%의 기업이 AI 도입 중이거나 완료된 상태임.

### 제조 현장의 AI 활용 분야

| 분야 | AI 활용 사례 | 기대 효과 |
|------|-------------|----------|
| 품질 관리 | 불량품 자동 검출, 품질 예측 | 불량률 30% 감소 |
| 설비 관리 | 고장 예측, 예지 정비 | 다운타임 50% 감소 |
| 공정 최적화 | 최적 파라미터 자동 설정 | 생산성 20% 향상 |
| 수요 예측 | AI 기반 생산 계획 | 재고 비용 25% 절감 |
| 에너지 관리 | 소비 패턴 분석, 최적화 | 에너지 15% 절감 |

### AI의 양면성

```
       AI의 혜택                    AI의 위험
    +-------------+            +-------------+
    | 생산성 향상  |            | 편향된 판단  |
    | 품질 개선   |     VS     | 개인정보 침해 |
    | 비용 절감   |            | 책임 소재 불명 |
    | 안전 강화   |            | 일자리 위협  |
    +-------------+            +-------------+
```

강력한 도구에는 그에 맞는 책임이 따름.

---

## Part 2. AI 윤리 4대 원칙 심화

### 개념 설명

AI 윤리 4대 원칙은 국내(과학기술정보통신부 인공지능 윤리기준 2020)와 국제(EU AI Act, OECD AI 원칙)에서 공통적으로 강조하는 핵심 원칙임.

```
    +--------------------------------------+
    |          AI 윤리 4대 원칙             |
    +--------+--------+--------+-----------+
    | 공정성  | 투명성  | 책임성  | 안전성    |
    |Fairness|Trans-  |Account-|Safety     |
    |        |parency |ability |           |
    +--------+--------+--------+-----------+
```

### 원칙 1: 공정성 (Fairness)

**정의**: AI 시스템은 모든 사용자를 공평하게 대우하고, 특정 그룹에 불리한 결과를 내지 않아야 함.

**핵심 질문**:
- 학습 데이터에 편향이 있지 않은가?
- 특정 조건에서만 잘 작동하지 않는가?
- 결과가 특정 그룹에 불이익을 주지 않는가?

**공정성 위반 사례: Amazon 채용 AI (2018년)**

| 항목 | 내용 |
|------|------|
| 문제 | 여성 지원자에게 불리한 점수 부여 |
| 원인 | 과거 10년간 남성 위주 채용 데이터로 학습 |
| 결과 | 시스템 전면 폐기 |
| 교훈 | 역사적 편향이 AI에 그대로 반영됨 |

**공정성 확보 방안**:

| 단계 | 점검 항목 |
|------|----------|
| 데이터 수집 | 모든 라인/시간대에서 균등 수집 |
| 데이터 분석 | 편향 여부 통계적 검증 |
| 모델 학습 | 공정성 지표 모니터링 |
| 배포 후 | 정기적 편향 감사 (Bias Audit) |

### 원칙 2: 투명성 (Transparency)

**정의**: AI의 의사결정 과정이 설명 가능해야 하며, 왜 그런 결과가 나왔는지 이해할 수 있어야 함.

```
+--------------------------------------------+
|               투명성                        |
+---------------------+----------------------+
|   설명 가능성        |    해석 가능성        |
|  (Explainability)   |  (Interpretability)   |
+---------------------+----------------------+
| 개별 예측의 근거 설명 | 모델 전체 동작 이해    |
| "왜 이 제품이 불량?" | "어떤 변수가 중요?"    |
| LIME, SHAP 등       | 특성 중요도, 규칙 추출 |
+---------------------+----------------------+
```

**나쁜 예 vs 좋은 예**:

나쁜 예 (블랙박스):
```
AI: "이 제품은 불량입니다."
작업자: "왜요? 어디가 문제인가요?"
AI: "..."
```

좋은 예 (설명 가능 AI):
```
AI: "이 제품은 불량입니다."
    - 이유: 표면 스크래치 0.7mm 검출 (기준: 0.5mm 이하)
    - 위치: 우측 상단 모서리
    - 신뢰도: 94%
```

### 원칙 3: 책임성 (Accountability)

**정의**: AI 시스템의 결과에 대한 책임 주체가 명확해야 하며, 문제 발생 시 대응 체계가 갖춰져야 함.

**책임 소재의 복잡성** (AI가 잘못 판단해서 불량품이 출하되었다면?):

| 후보 | 책임 여부 | 조건 |
|------|----------|------|
| AI 개발사 | O | 모델 오류, 학습 결함 |
| 도입 기업 | O | 운영 관리 책임, 검증 미흡 |
| 담당 부서 | O | 모니터링 소홀, 경고 무시 |
| 담당자 | 삼각형 | 고의/과실 여부에 따라 |
| AI 자체 | X | 법적 주체 아님 |

현행법상 AI는 책임 주체가 될 수 없음.

### 원칙 4: 안전성 / 프라이버시

**정의**: 개인정보 및 기업 비밀을 보호하고, 데이터 수집/저장/활용 시 적법한 동의를 받아야 함.

**데이터 보안 3대 영역**:

| 영역 | 보호 대상 | 핵심 조치 |
|------|----------|----------|
| 개인정보 | 작업자 정보, 연락처 | 비식별화, 동의 획득 |
| 기업 비밀 | 공정 데이터, 파라미터 | 암호화, 접근 제한 |
| 접근 권한 | 시스템 전체 | 최소 권한 원칙 |

---

## Part 3. 데이터 보안 사고 사례 분석

### 개념 설명

제조업 데이터 유출 유형을 분석하면 내부자에 의한 유출이 가장 많음.

| 유형 | 비율 | 주요 원인 |
|------|------|----------|
| 내부자 유출 | 45% | 퇴직자, 불만 직원 |
| 외부 해킹 | 30% | 랜섬웨어, 피싱 |
| 실수에 의한 유출 | 20% | 이메일 오발송, 설정 오류 |
| 협력사 경유 | 5% | 공급망 취약점 |

### 사례 1: S전자 ChatGPT 유출 사건 (2023년)

| 항목 | 내용 |
|------|------|
| 사건 개요 | 직원들이 ChatGPT에 반도체 소스코드 입력 |
| 유출 데이터 | 반도체 설비 측정 데이터, 소스코드 |
| 문제점 | OpenAI 서버로 데이터 전송, 학습 데이터로 활용 가능성 |
| 대응 | 사내 ChatGPT 사용 전면 금지 (이후 자체 AI 개발) |
| 교훈 | 외부 AI 서비스에 기밀 정보 입력 금지 |

### 사례 2: 랜섬웨어 공격 - Colonial Pipeline 사태 (2021년)

| 항목 | 내용 |
|------|------|
| 사건 개요 | 미국 최대 송유관 회사 랜섬웨어 감염 |
| 피해 | 6일간 운영 중단, 동부 연료 공급 대란 |
| 몸값 | 440만 달러 (약 50억 원) 지불 |
| 원인 | 단일 비밀번호 유출 (VPN 계정) |
| 제조업 교훈 | OT(운영기술) 보안의 중요성 |

### 외부 AI 서비스 데이터 흐름

```
+-------------------------------------------------+
|           외부 AI 서비스 데이터 흐름              |
+-------------------------------------------------+
|                                                 |
|  사용자 --> [프롬프트 입력] --> 외부 AI 서버     |
|                |                                |
|         서버에 데이터 저장                       |
|                |                                |
|         모델 학습에 활용 가능                    |
|                |                                |
|         다른 사용자에게 노출 위험                |
|                                                 |
+-------------------------------------------------+
```

### 개인정보 보호 방안: 비식별화 기법

| 기법 | 설명 | 예시 |
|------|------|------|
| 가명처리 | 식별자를 가명으로 대체 | 홍길동 -> ID_001 |
| 총계처리 | 개별값을 집계값으로 | 개인 실적 -> 팀 평균 |
| 데이터 삭제 | 불필요 정보 제거 | 주민번호 삭제 |
| 범주화 | 구체적 값을 범주로 | 32세 -> 30대 |
| 데이터 마스킹 | 일부 문자 가림 | 010-****-5678 |

---

## Part 4. AI 저작권과 법적 이슈

### 개념 설명

AI 저작권의 핵심 쟁점은 법적으로 불명확한 영역이 많음.

| 쟁점 | 현황 |
|------|------|
| AI 생성물 저작권 | 인간 창작물로 볼 수 있는가? |
| 학습 데이터 저작권 | 저작물을 학습에 사용해도 되는가? |
| AI 출력물 유사성 | 기존 저작물과 유사하면? |

현재 대부분 국가에서 AI 자체는 저작권 주체가 될 수 없음.

### 오픈소스 라이선스 종류

| 라이선스 | 특징 | 주의사항 |
|----------|------|---------|
| MIT | 자유롭게 사용 가능 | 저작권 표시 필요 |
| Apache 2.0 | 상업적 사용 가능 | 특허권 명시 |
| GPL | 소스 공개 의무 | 파생작품도 GPL |
| LGPL | 라이브러리 사용 가능 | 수정 시 공개 |

AI가 생성한 코드에 어떤 라이선스가 적용되는지 확인 필수.

---

## 실습 코드

### AI 윤리 원칙 체크리스트 클래스

```python
class AIEthicsChecklist:
    """AI 윤리 점검을 위한 체크리스트 클래스"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.checklist = {
            "공정성": {
                "학습 데이터 편향 검토": False,
                "예측 결과 그룹별 비교": False,
                "불이익 집단 존재 여부 확인": False,
            },
            "투명성": {
                "모델 설명 가능성 확보": False,
                "AI 사용 여부 고지": False,
                "의사결정 과정 문서화": False,
            },
            "책임성": {
                "책임자 지정": False,
                "피해 구제 절차 마련": False,
                "외부 감사 가능 구조": False,
            },
            "안전성": {
                "오작동 테스트 완료": False,
                "비상정지 체계 구축": False,
                "모니터링 시스템 가동": False,
            },
        }

    def check_item(self, principle, item):
        """체크리스트 항목 체크"""
        if principle in self.checklist:
            if item in self.checklist[principle]:
                self.checklist[principle][item] = True
                print(f"[v] {principle} - {item} 완료")

    def get_completion_rate(self):
        """완료율 계산"""
        total = 0
        completed = 0
        for principle, items in self.checklist.items():
            for item, status in items.items():
                total += 1
                if status:
                    completed += 1
        return (completed / total) * 100

    def generate_report(self):
        """점검 보고서 생성"""
        print(f"\n{'=' * 50}")
        print(f"AI 윤리 점검 보고서: {self.project_name}")
        print("=" * 50)

        for principle, items in self.checklist.items():
            print(f"\n[{principle}]")
            for item, status in items.items():
                mark = "v" if status else " "
                print(f"  [{mark}] {item}")

        rate = self.get_completion_rate()
        print(f"\n완료율: {rate:.1f}%")

        if rate == 100:
            print("상태: 윤리 점검 완료 - 배포 가능")
        elif rate >= 75:
            print("상태: 추가 점검 필요")
        else:
            print("상태: 윤리 점검 미흡 - 배포 불가")
```

### 결과 해설

- AIEthicsChecklist 클래스는 AI 프로젝트의 윤리 점검 상태를 추적함
- 4대 원칙별로 세부 점검 항목을 관리함
- 완료율에 따라 배포 가능 여부를 판단함

---

### 데이터 편향 검사 함수

```python
def check_data_bias(data, group_column, target_column):
    """
    데이터셋의 그룹별 분포 편향을 검사하는 함수

    Parameters:
    - data: 딕셔너리 리스트 형태의 데이터
    - group_column: 그룹을 나누는 컬럼명
    - target_column: 타겟 변수 컬럼명

    Returns:
    - 그룹별 통계 및 편향 여부
    """
    # 그룹별 통계 계산
    group_stats = {}

    for row in data:
        group = row[group_column]
        target = row[target_column]

        if group not in group_stats:
            group_stats[group] = {"count": 0, "positive": 0}

        group_stats[group]["count"] += 1
        if target == 1:
            group_stats[group]["positive"] += 1

    # 비율 계산 및 출력
    print(f"\n그룹별 {target_column} 비율:")
    print("-" * 40)

    rates = []
    for group, stats in group_stats.items():
        rate = stats["positive"] / stats["count"] * 100
        rates.append(rate)
        print(f"  {group}: {stats['positive']}/{stats['count']} ({rate:.1f}%)")

    # 편향 판단 (최대 차이가 20%p 이상이면 편향 의심)
    max_diff = max(rates) - min(rates)
    print(f"\n최대 그룹 간 차이: {max_diff:.1f}%p")

    if max_diff > 20:
        print("주의: 그룹 간 유의미한 차이 발견 - 편향 검토 필요")
        return True
    else:
        print("결과: 그룹 간 큰 차이 없음")
        return False


# 샘플 채용 데이터 (가상)
hiring_data = [
    {"gender": "남성", "experience": 5, "hired": 1},
    {"gender": "남성", "experience": 3, "hired": 1},
    {"gender": "남성", "experience": 2, "hired": 0},
    {"gender": "남성", "experience": 7, "hired": 1},
    {"gender": "남성", "experience": 4, "hired": 1},
    {"gender": "여성", "experience": 5, "hired": 0},
    {"gender": "여성", "experience": 6, "hired": 1},
    {"gender": "여성", "experience": 3, "hired": 0},
    {"gender": "여성", "experience": 8, "hired": 0},
    {"gender": "여성", "experience": 4, "hired": 0},
]

print("[가상 채용 데이터 편향 검사]")
check_data_bias(hiring_data, "gender", "hired")
```

### 결과 해설

- 남성 채용률: 80%, 여성 채용률: 20%
- 최대 그룹 간 차이: 60%p
- 심각한 편향이 발견됨 -> 데이터 또는 프로세스 검토 필요

---

### 민감 정보 마스킹 함수

```python
import re

def mask_sensitive_data(text, mask_type="all"):
    """
    텍스트에서 민감 정보를 마스킹하는 함수

    Parameters:
    - text: 원본 텍스트
    - mask_type: 마스킹 유형 ('all', 'email', 'phone', 'id')

    Returns:
    - 마스킹된 텍스트
    """
    masked = text

    # 이메일 마스킹
    if mask_type in ["all", "email"]:
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        masked = re.sub(email_pattern, "[이메일 마스킹]", masked)

    # 전화번호 마스킹 (010-1234-5678 형식)
    if mask_type in ["all", "phone"]:
        phone_pattern = r"\d{2,3}-\d{3,4}-\d{4}"
        masked = re.sub(phone_pattern, "[전화번호 마스킹]", masked)

    # 주민번호 마스킹 (123456-1234567 형식)
    if mask_type in ["all", "id"]:
        id_pattern = r"\d{6}-\d{7}"
        masked = re.sub(id_pattern, "[주민번호 마스킹]", masked)

    return masked


# 테스트
sample_text = """
작업자 정보:
- 이름: 김철수
- 연락처: 010-1234-5678
- 이메일: kim@company.com
- 주민번호: 901234-1234567
- 담당 라인: A라인
"""

print("[원본 텍스트]")
print(sample_text)

print("[마스킹 후]")
print(mask_sensitive_data(sample_text))
```

### 결과 해설

- 이메일, 전화번호, 주민번호가 모두 마스킹됨
- 담당 라인 등 비민감 정보는 유지됨
- 정규표현식을 활용한 자동화된 민감정보 보호

---

### 데이터 접근 로그 기록 클래스

```python
from datetime import datetime

class DataAccessLogger:
    """데이터 접근 로그를 기록하는 클래스"""

    def __init__(self):
        self.logs = []

    def log_access(self, user, action, data_type, status="success"):
        """접근 로그 기록"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user,
            "action": action,
            "data_type": data_type,
            "status": status,
        }
        self.logs.append(log_entry)
        print(f"[LOG] {log_entry['timestamp']} | {user} | {action} | {data_type} | {status}")

    def get_suspicious_activity(self):
        """의심스러운 활동 감지"""
        suspicious = []
        user_counts = {}

        for log in self.logs:
            user = log["user"]
            user_counts[user] = user_counts.get(user, 0) + 1

            # 실패한 접근 시도
            if log["status"] == "denied":
                suspicious.append(log)

        # 짧은 시간 내 과다 접근 (예: 10회 이상)
        for user, count in user_counts.items():
            if count > 10:
                suspicious.append({"user": user, "issue": "과다 접근", "count": count})

        return suspicious

    def print_summary(self):
        """로그 요약 출력"""
        print(f"\n총 로그 수: {len(self.logs)}")
        print("-" * 40)

        # 액션별 통계
        actions = {}
        for log in self.logs:
            action = log["action"]
            actions[action] = actions.get(action, 0) + 1

        print("액션별 통계:")
        for action, count in actions.items():
            print(f"  {action}: {count}회")


# 사용 예시
logger = DataAccessLogger()

print("[데이터 접근 시뮬레이션]")
print("-" * 40)

# 정상 접근
logger.log_access("user_A", "READ", "품질 데이터")
logger.log_access("user_A", "WRITE", "품질 데이터")
logger.log_access("user_B", "READ", "센서 데이터")

# 거부된 접근
logger.log_access("user_C", "READ", "급여 데이터", "denied")
logger.log_access("user_C", "READ", "인사 데이터", "denied")

# 요약 출력
logger.print_summary()

# 의심스러운 활동
suspicious = logger.get_suspicious_activity()
if suspicious:
    print("\n주의: 의심스러운 활동 감지됨!")
    for item in suspicious:
        print(f"  {item}")
```

### 결과 해설

- 모든 데이터 접근이 로그로 기록됨
- denied 상태의 접근 시도가 의심 활동으로 분류됨
- 보안 감사 및 사고 추적에 활용 가능함

---

## 핵심 정리

### AI 윤리 4대 원칙

| 원칙 | 핵심 내용 |
|------|----------|
| 공정성 | 편향 없는 공평한 판단 |
| 투명성 | 설명 가능한 AI |
| 책임성 | 명확한 책임 주체 |
| 안전성 | 개인정보/기업비밀 보호 |

### 데이터 보안 3대 영역

- 개인정보 보호
- 기업 비밀 보호
- 접근 권한 관리

### 실무 적용

- 체크리스트로 프로젝트 시작 전 점검
- 외부 AI에 기밀 정보 입력 금지
- AI 사용 전 동의 수집
- 접근 로그 기록 및 모니터링

---

## 다음 차시 예고

### 2차시: Python 시작하기

- Anaconda 설치 및 환경 구성
- Jupyter Notebook 사용법
- Python 기본 문법 (변수, 조건문, 반복문)
- 첫 번째 Python 프로그램 작성

---

**1차시 AI 활용 윤리와 데이터 보호 완료**

**기술만큼이나 윤리 의식이 중요함**
