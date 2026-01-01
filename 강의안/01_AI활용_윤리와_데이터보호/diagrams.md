# [1차시] AI 활용 윤리와 데이터 보호 - 다이어그램

## 1. AI 윤리 4대 원칙

```mermaid
mindmap
  root((AI 윤리<br>4대 원칙))
    공정성
      모든 사용자 공평 대우
      편향 없는 판단
      차별 금지
    투명성
      설명 가능한 AI
      블랙박스 해소
      결정 근거 제시
    책임성
      책임 주체 명확화
      대응 체계 구축
      오류 시 책임
    안전성
      개인정보 보호
      기업비밀 보호
      데이터 동의
```

## 2. 데이터 보안 3대 영역

```mermaid
flowchart TD
    subgraph 데이터보안["데이터 보안 3대 영역"]
        A[개인정보 보호]
        B[기업 비밀 보호]
        C[접근 권한 관리]
    end

    A --> A1[비식별화 처리]
    A --> A2[수집 동의]
    A --> A3[보유 기간 준수]

    B --> B1[암호화 저장]
    B --> B2[외부 서비스 주의]
    B --> B3[기밀 분류]

    C --> C1[최소 권한 원칙]
    C --> C2[역할별 접근 설정]
    C --> C3[접근 로그 기록]
```

## 3. 외부 AI 서비스 사용 판단

```mermaid
flowchart TD
    A[외부 AI 서비스 사용?] --> B{기업 데이터 포함?}
    B -->|Yes| C{어떤 데이터?}
    B -->|No| D[사용 가능]

    C -->|센서값/파라미터| E[사용 금지]
    C -->|고객사 정보| E
    C -->|설비 운영 코드| E

    D --> F[일반 문법 질문]
    D --> G[공개 알고리즘]
    D --> H[오픈소스 사용법]

    E --> I[사내 승인 도구 사용]
```

## 4. AI 프로젝트 윤리 점검 프로세스

```mermaid
flowchart LR
    A[프로젝트 시작] --> B[윤리 점검]
    B --> C{점검 통과?}
    C -->|No| D[위험 식별 및 대응]
    D --> B
    C -->|Yes| E[데이터 수집]
    E --> F[모델 개발]
    F --> G[배포 전 재점검]
    G --> H[운영 및 모니터링]
```

## 5. 데이터 보안 등급 분류

```mermaid
pie title 데이터 보안 등급별 처리 우선순위
    "최고 (고객사 정보)" : 30
    "상 (설비 파라미터, 작업자 정보)" : 35
    "중 (센서값, 품질 결과)" : 25
    "하 (일반 로그)" : 10
```

## 6. 제조 AI 프로젝트 책임 체계

```mermaid
flowchart TD
    subgraph 책임체계["AI 오류 발생 시 책임 체계"]
        A[AI 오류 발생] --> B{오류 유형?}
        B -->|모델 결함| C[AI 개발사]
        B -->|운영 미흡| D[도입 기업]
        B -->|검증 누락| E[담당자]
    end

    C --> F[모델 수정/보상]
    D --> G[프로세스 개선]
    E --> H[재교육/절차 보완]
```

## 7. AI 윤리 체크리스트 흐름

```mermaid
sequenceDiagram
    participant PM as 프로젝트 관리자
    participant DEV as 개발팀
    participant SEC as 보안팀
    participant LEG as 법무팀

    PM->>DEV: 프로젝트 요구사항 전달
    DEV->>DEV: 공정성/투명성 점검
    DEV->>SEC: 데이터 보안 검토 요청
    SEC->>SEC: 보안 등급 분류
    SEC->>DEV: 보안 조치 권고
    DEV->>LEG: 저작권/개인정보 검토
    LEG->>PM: 법률 검토 결과
    PM->>PM: 최종 승인
```

## 8. 이론-실습 구조

```mermaid
gantt
    title 1차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (인사/목표)     :a1, 00:00, 2m
    AI 윤리 4대 원칙     :a2, after a1, 4m
    데이터 보안/저작권   :a3, after a2, 4m

    section 실습편
    실습 소개            :b1, after a3, 2m
    윤리 체크리스트      :b2, after b1, 5m
    보안 위험 평가       :b3, after b2, 4m
    AI 서비스 가이드     :b4, after b3, 4m
    점검 문서 작성       :b5, after b4, 4m

    section 정리
    요약 및 예고         :c1, after b5, 3m
```
