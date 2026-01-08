"""
[1차시] AI 활용 윤리와 데이터 보호 - 참조 코드 (보강판)

이 차시는 이론 중심이므로, 실습 코드보다는
AI 윤리와 보안 개념을 이해하기 위한 참조 코드를 제공합니다.

학습목표:
1. AI 윤리의 4가지 핵심 원칙을 코드로 이해
2. 데이터 보안 점검 방법 예시
3. 편향 검사 기초 개념
"""

# ============================================================
# 1. AI 윤리 원칙 체크리스트 (개념 이해용)
# ============================================================

print("=" * 60)
print("[1차시] AI 활용 윤리와 데이터 보호 - 참조 코드")
print("=" * 60)


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


# 사용 예시
print("\n[예시] AI 윤리 체크리스트 사용")
print("-" * 40)

project = AIEthicsChecklist("품질 예측 AI 모델 v1.0")

# 일부 항목 체크
project.check_item("공정성", "학습 데이터 편향 검토")
project.check_item("투명성", "AI 사용 여부 고지")
project.check_item("책임성", "책임자 지정")
project.check_item("안전성", "비상정지 체계 구축")

# 보고서 생성
project.generate_report()


# ============================================================
# 2. 데이터 편향 검사 기초 예시
# ============================================================

print("\n" + "=" * 60)
print("[예시] 데이터 편향 검사 기초")
print("=" * 60)

import random


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

print("\n[가상 채용 데이터 편향 검사]")
check_data_bias(hiring_data, "gender", "hired")


# 제조 라인별 데이터 (가상)
line_data = [
    {"line": "A라인", "shift": "주간", "defect": 0},
    {"line": "A라인", "shift": "야간", "defect": 0},
    {"line": "A라인", "shift": "주간", "defect": 1},
    {"line": "B라인", "shift": "주간", "defect": 0},
    {"line": "B라인", "shift": "야간", "defect": 0},
    {"line": "B라인", "shift": "주간", "defect": 0},
    {"line": "C라인", "shift": "주간", "defect": 1},
    {"line": "C라인", "shift": "야간", "defect": 1},
    {"line": "C라인", "shift": "주간", "defect": 0},
]

print("\n[제조 라인별 불량률 편향 검사]")
check_data_bias(line_data, "line", "defect")


# ============================================================
# 3. 민감 정보 마스킹 예시
# ============================================================

print("\n" + "=" * 60)
print("[예시] 민감 정보 마스킹")
print("=" * 60)

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

print("\n[원본 텍스트]")
print(sample_text)

print("[마스킹 후]")
print(mask_sensitive_data(sample_text))


# ============================================================
# 4. 데이터 접근 로그 기록 예시
# ============================================================

print("\n" + "=" * 60)
print("[예시] 데이터 접근 로그 기록")
print("=" * 60)

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

    def get_user_logs(self, user):
        """특정 사용자의 로그 조회"""
        return [log for log in self.logs if log["user"] == user]

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

print("\n[데이터 접근 시뮬레이션]")
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


# ============================================================
# 5. AI 모델 사용 동의서 체크 예시
# ============================================================

print("\n" + "=" * 60)
print("[예시] AI 모델 사용 동의 체크")
print("=" * 60)


class AIConsentManager:
    """AI 모델 사용 동의 관리 클래스"""

    REQUIRED_CONSENTS = [
        "개인정보 수집 및 이용 동의",
        "AI 분석 결과 활용 동의",
        "데이터 보관 기간 동의",
        "제3자 제공 동의 (선택)",
    ]

    def __init__(self):
        self.consents = {}

    def request_consent(self, user_id, consent_item, agreed):
        """동의 요청 및 기록"""
        if user_id not in self.consents:
            self.consents[user_id] = {}

        self.consents[user_id][consent_item] = {
            "agreed": agreed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def check_required_consents(self, user_id):
        """필수 동의 항목 확인"""
        if user_id not in self.consents:
            return False, "동의 기록 없음"

        user_consents = self.consents[user_id]
        missing = []

        for item in self.REQUIRED_CONSENTS:
            if "(선택)" not in item:  # 선택 항목 제외
                if item not in user_consents or not user_consents[item]["agreed"]:
                    missing.append(item)

        if missing:
            return False, f"미동의 항목: {missing}"
        return True, "모든 필수 항목 동의 완료"

    def can_use_ai(self, user_id):
        """AI 사용 가능 여부 확인"""
        valid, message = self.check_required_consents(user_id)
        if valid:
            print(f"[v] {user_id}: AI 사용 가능 - {message}")
            return True
        else:
            print(f"[x] {user_id}: AI 사용 불가 - {message}")
            return False


# 사용 예시
consent_manager = AIConsentManager()

print("\n[동의 수집 시뮬레이션]")
print("-" * 40)

# 사용자 A: 모든 필수 항목 동의
consent_manager.request_consent("user_A", "개인정보 수집 및 이용 동의", True)
consent_manager.request_consent("user_A", "AI 분석 결과 활용 동의", True)
consent_manager.request_consent("user_A", "데이터 보관 기간 동의", True)

# 사용자 B: 일부 항목 미동의
consent_manager.request_consent("user_B", "개인정보 수집 및 이용 동의", True)
consent_manager.request_consent("user_B", "AI 분석 결과 활용 동의", False)

print("\n[AI 사용 가능 여부 확인]")
consent_manager.can_use_ai("user_A")
consent_manager.can_use_ai("user_B")


# ============================================================
# 6. 핵심 요약
# ============================================================

print("\n" + "=" * 60)
print("[1차시 핵심 요약]")
print("=" * 60)

summary = """
AI 윤리 4원칙:
1. 공정성 (Fairness): 편향 없는 AI
   → 학습 데이터 편향 검사 필수

2. 투명성 (Transparency): 설명 가능한 AI
   → 의사결정 과정 문서화

3. 책임성 (Accountability): 책임질 수 있는 AI
   → 책임자 지정, 피해 구제 절차

4. 안전성 (Safety): 안전하게 작동하는 AI
   → 비상정지 체계, 모니터링

데이터 보안 3영역:
1. 개인정보 보호 → 마스킹, 익명화
2. 기업 기밀 보호 → 접근 통제
3. 시스템 접근 통제 → 로그 기록

실무 가이드:
- 외부 AI에 기밀 정보 입력 금지
- AI 사용 전 동의 수집
- 접근 로그 기록 및 모니터링
"""

print(summary)

print("=" * 60)
print("다음 차시: Python 시작하기")
print("=" * 60)
