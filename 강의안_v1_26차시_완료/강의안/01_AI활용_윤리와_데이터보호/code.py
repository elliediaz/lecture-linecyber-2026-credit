"""
[1차시] AI 활용 윤리 및 보호체계 - 참고 코드

이 차시는 이론 중심이지만, 다음 차시부터 사용할
데이터 보안 관련 기본 개념을 코드로 미리 살펴봅니다.

학습목표:
- 개인정보 비식별화의 기본 개념 이해
- 안전한 데이터 처리 방법 예시
"""

# =============================================================
# 1. 개인정보 비식별화 예시
# =============================================================

import hashlib

def hash_personal_info(value: str) -> str:
    """
    개인정보를 해시 함수로 비식별화합니다.

    주의: 실무에서는 salt를 추가하여 보안을 강화해야 합니다.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:16]


# 예시: 이름 비식별화
original_names = ["홍길동", "김철수", "이영희"]
anonymized_names = [hash_personal_info(name) for name in original_names]

print("=== 개인정보 비식별화 예시 ===")
print(f"원본: {original_names}")
print(f"비식별화: {anonymized_names}")
print()


# =============================================================
# 2. 잘못된 예시 vs 올바른 예시
# =============================================================

print("=== 데이터 처리 예시 ===")

# 잘못된 예시: 민감한 정보 직접 출력
# print("사원번호: 123456, 이름: 홍길동, 급여: 5000만원")  # 절대 금지!

# 올바른 예시: 필요한 정보만, 비식별화하여 사용
employee_data = {
    "id": "EMP_A1B2C3",  # 비식별화된 ID
    "department": "제조1팀",
    "performance_score": 85  # 개인 식별 불가능한 정보
}
print(f"분석용 데이터: {employee_data}")
print()


# =============================================================
# 3. 데이터 접근 권한 시뮬레이션
# =============================================================

class DataAccessControl:
    """간단한 데이터 접근 권한 관리 예시"""

    def __init__(self, role: str):
        self.role = role

        # 역할별 접근 가능 데이터
        self.permissions = {
            "operator": ["own_line_data"],
            "quality_manager": ["all_quality_data"],
            "data_analyst": ["anonymized_analysis_data"],
            "admin": ["all_data", "system_logs"]
        }

    def can_access(self, data_type: str) -> bool:
        """특정 데이터 유형에 접근 가능한지 확인"""
        allowed = self.permissions.get(self.role, [])
        return data_type in allowed or "all_data" in allowed

    def get_accessible_data(self) -> list:
        """접근 가능한 데이터 목록 반환"""
        return self.permissions.get(self.role, [])


print("=== 데이터 접근 권한 시뮬레이션 ===")

# 다양한 역할 테스트
roles = ["operator", "quality_manager", "data_analyst", "admin"]

for role in roles:
    access_control = DataAccessControl(role)
    accessible = access_control.get_accessible_data()
    print(f"역할: {role:20} → 접근 가능: {accessible}")

print()


# =============================================================
# 4. AI 윤리 체크리스트 (프로젝트 시작 전 확인용)
# =============================================================

def ai_ethics_checklist() -> dict:
    """AI 프로젝트 윤리 체크리스트"""

    checklist = {
        "공정성": [
            "[ ] 학습 데이터에 편향이 없는가?",
            "[ ] 특정 그룹에 불이익을 주지 않는가?",
            "[ ] 다양한 사용자 그룹으로 테스트했는가?"
        ],
        "투명성": [
            "[ ] AI 결정 이유를 설명할 수 있는가?",
            "[ ] 사용자에게 AI 사용 사실을 고지했는가?",
            "[ ] 모델 작동 방식을 문서화했는가?"
        ],
        "책임성": [
            "[ ] 문제 발생 시 책임자가 지정되어 있는가?",
            "[ ] 오류 발생 시 대응 절차가 있는가?",
            "[ ] 모델 성능을 정기적으로 모니터링하는가?"
        ],
        "프라이버시": [
            "[ ] 개인정보 수집 동의를 받았는가?",
            "[ ] 데이터 비식별화 처리를 했는가?",
            "[ ] 데이터 보관 기간을 정했는가?"
        ]
    }

    return checklist


print("=== AI 윤리 체크리스트 ===")
checklist = ai_ethics_checklist()
for category, items in checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")


# =============================================================
# 5. 외부 AI 서비스 사용 시 주의사항 (의사 코드)
# =============================================================

print("\n\n=== 외부 AI 서비스 사용 가이드 ===")

# 잘못된 예시 (실제로 실행하면 안 됨!)
"""
# 절대 하지 말아야 할 것!
import openai

# 기업 기밀 데이터를 외부 AI에 전송
openai.ChatCompletion.create(
    messages=[{
        "role": "user",
        "content": f"이 설계 도면을 분석해줘: {company_secret_data}"
    }]
)
"""

# 올바른 예시
print("""
[올바른 외부 AI 서비스 사용 절차]

1. 사내 보안 정책 확인
   - IT 보안팀 승인 여부 확인
   - 사용 가능한 AI 서비스 목록 확인

2. 데이터 검토
   - 전송할 데이터에 기밀 정보 포함 여부 확인
   - 필요시 비식별화 처리

3. 사용 기록
   - 어떤 데이터를 어떤 서비스에 전송했는지 기록
   - 이상 징후 발견 시 즉시 보고
""")


# =============================================================
# 요약
# =============================================================

print("\n" + "="*50)
print("1차시 핵심 정리")
print("="*50)
print("""
1. AI 윤리 4대 원칙: 공정성, 투명성, 책임성, 프라이버시
2. 데이터 보안: 비식별화, 접근 권한 관리, 외부 서비스 주의
3. 저작권: AI 생성물의 권리 불명확, 라이선스 확인 필수
""")
