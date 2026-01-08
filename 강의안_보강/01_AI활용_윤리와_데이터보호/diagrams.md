# [1차시] AI 활용 윤리와 데이터 보호 - 다이어그램 (보강판)

## 1. AI 학습의 전체 흐름

```mermaid
flowchart LR
    A["데이터 수집"] --> B["전처리"]
    B --> C["모델 학습"]
    C --> D["평가"]
    D --> E["배포"]
    E --> F["모니터링"]

    style A fill:#dbeafe,stroke:#3b82f6
    style B fill:#dcfce7,stroke:#22c55e
    style C fill:#fef3c7,stroke:#f59e0b
    style D fill:#fce7f3,stroke:#ec4899
    style E fill:#e0e7ff,stroke:#6366f1
    style F fill:#f3e8ff,stroke:#a855f7
```

## 2. AI 윤리 4원칙 (중심 구조)

```mermaid
flowchart TD
    CENTER["AI 윤리"]

    CENTER --> F["공정성<br>Fairness"]
    CENTER --> T["투명성<br>Transparency"]
    CENTER --> A["책임성<br>Accountability"]
    CENTER --> S["안전성<br>Safety"]

    style CENTER fill:#1e40af,color:#fff
    style F fill:#dbeafe
    style T fill:#dcfce7
    style A fill:#fef3c7
    style S fill:#fce7f3
```

## 3. 공정성 원칙 상세

```mermaid
flowchart TD
    A["공정성<br>Fairness"]

    A --> B["편향 없는 학습 데이터"]
    A --> C["차별 없는 예측 결과"]
    A --> D["균형 잡힌 평가 기준"]

    B --> B1["성별/인종/연령<br>균형 확보"]
    C --> C1["특정 집단<br>불이익 방지"]
    D --> D1["다양한 지표로<br>성능 평가"]

    style A fill:#dbeafe,stroke:#3b82f6
```

## 4. 투명성 원칙 상세

```mermaid
flowchart TD
    A["투명성<br>Transparency"]

    A --> B["설명 가능성<br>Explainability"]
    A --> C["정보 공개<br>Disclosure"]
    A --> D["추적 가능성<br>Traceability"]

    B --> B1["왜 그런 결정을<br>내렸는가?"]
    C --> C1["AI 사용 여부<br>고지"]
    D --> D1["의사결정 과정<br>기록"]

    style A fill:#dcfce7,stroke:#22c55e
```

## 5. 책임성 원칙 상세

```mermaid
flowchart TD
    A["책임성<br>Accountability"]

    A --> B["명확한 책임 주체"]
    A --> C["피해 구제 체계"]
    A --> D["감사 가능성"]

    B --> B1["AI 결과물<br>책임자 지정"]
    C --> C1["오류 발생 시<br>보상 절차"]
    D --> D1["외부 검증<br>가능한 구조"]

    style A fill:#fef3c7,stroke:#f59e0b
```

## 6. 안전성 원칙 상세

```mermaid
flowchart TD
    A["안전성<br>Safety"]

    A --> B["기술적 안전"]
    A --> C["운영 안전"]
    A --> D["비상 체계"]

    B --> B1["오작동 방지<br>검증 테스트"]
    C --> C1["지속적<br>모니터링"]
    D --> D1["비상정지<br>매뉴얼"]

    style A fill:#fce7f3,stroke:#ec4899
```

## 7. AI 편향 발생 과정

```mermaid
flowchart LR
    A["편향된<br>학습 데이터"] --> B["AI 모델<br>학습"]
    B --> C["편향된<br>예측 결과"]
    C --> D["불공정한<br>의사결정"]
    D --> E["사회적<br>문제 발생"]

    style A fill:#fecaca,stroke:#ef4444
    style E fill:#fecaca,stroke:#ef4444
```

## 8. 아마존 채용 AI 편향 사례

```mermaid
flowchart TD
    A["과거 10년<br>채용 데이터"]
    B["AI 학습"]
    C["패턴 인식:<br>남성 합격률 높음"]
    D["여성 지원자<br>불리한 점수"]
    E["시스템 폐기"]

    A --> B --> C --> D --> E

    style D fill:#fecaca,stroke:#ef4444
    style E fill:#dcfce7,stroke:#22c55e
```

## 9. 블랙박스 vs 화이트박스 모델

```mermaid
flowchart LR
    subgraph BlackBox["블랙박스 (딥러닝)"]
        B1["입력"] --> B2["????"]
        B2 --> B3["출력"]
    end

    subgraph WhiteBox["화이트박스 (의사결정나무)"]
        W1["입력"] --> W2["조건1"]
        W2 -->|Yes| W3["조건2"]
        W2 -->|No| W4["결과A"]
        W3 -->|Yes| W5["결과B"]
        W3 -->|No| W6["결과C"]
    end

    style B2 fill:#1e293b,color:#fff
    style WhiteBox fill:#dcfce7
```

## 10. 책임 소재 체계도

```mermaid
flowchart TD
    A["AI 의사결정 오류 발생"]

    A --> B["책임 분석"]

    B --> C["데이터 문제?"]
    B --> D["알고리즘 문제?"]
    B --> E["운영 문제?"]

    C --> C1["데이터팀 책임"]
    D --> D1["개발팀 책임"]
    E --> E1["운영팀 책임"]

    C1 & D1 & E1 --> F["최종: 경영진 책임"]

    style F fill:#fef3c7,stroke:#f59e0b
```

## 11. 데이터 보안 3대 영역

```mermaid
flowchart TD
    A["데이터 보안"]

    A --> B["개인정보<br>보호"]
    A --> C["기업 기밀<br>보호"]
    A --> D["시스템<br>접근 통제"]

    B --> B1["작업자 정보"]
    B --> B2["건강 데이터"]

    C --> C1["공정 조건"]
    C --> C2["레시피"]
    C --> C3["품질 데이터"]

    D --> D1["권한 관리"]
    D --> D2["로그 기록"]

    style A fill:#1e40af,color:#fff
```

## 12. 삼성전자 ChatGPT 사건 흐름

```mermaid
flowchart TD
    A["직원이 ChatGPT에<br>소스코드 입력"]
    B["외부 AI 서버로<br>데이터 전송"]
    C["기밀 유출<br>우려 발생"]
    D["ChatGPT<br>사용 금지"]
    E["내부 AI<br>플랫폼 구축"]

    A --> B --> C --> D --> E

    style C fill:#fecaca,stroke:#ef4444
    style E fill:#dcfce7,stroke:#22c55e
```

## 13. 내부자 위협 흐름

```mermaid
flowchart LR
    A["내부 직원"]
    B["기밀 데이터<br>접근"]
    C["외부 저장장치<br>복사"]
    D["퇴직"]
    E["경쟁사 이직"]
    F["기술 유출"]

    A --> B --> C --> D --> E --> F

    style F fill:#fecaca,stroke:#ef4444
```

## 14. 공급망 공격 경로

```mermaid
flowchart TD
    A["공격자"]
    B["협력사 서버<br>해킹"]
    C["협력사-본사<br>연결망"]
    D["본사 시스템<br>침투"]
    E["데이터 유출"]

    A --> B --> C --> D --> E

    style A fill:#fecaca,stroke:#ef4444
    style E fill:#fecaca,stroke:#ef4444
```

## 15. TSMC 바이러스 감염 사건

```mermaid
flowchart TD
    A["신규 장비 설치"]
    B["보안 검증 미흡"]
    C["바이러스 감염"]
    D["웨이퍼 팹 3곳<br>가동 중단"]
    E["3일간 생산 차질"]
    F["2,500억 원 손실"]

    A --> B --> C --> D --> E --> F

    style B fill:#fecaca,stroke:#ef4444
    style F fill:#fecaca,stroke:#ef4444
```

## 16. 보안 수준별 AI 활용 가이드

```mermaid
flowchart TD
    A["데이터 분류"]

    A --> B["기밀 데이터"]
    A --> C["내부 데이터"]
    A --> D["공개 데이터"]

    B --> B1["외부 AI 금지<br>사내 AI만 사용"]
    C --> C1["가이드라인 내<br>제한적 사용"]
    D --> D1["자유롭게<br>사용 가능"]

    style B fill:#fecaca,stroke:#ef4444
    style C fill:#fef3c7,stroke:#f59e0b
    style D fill:#dcfce7,stroke:#22c55e
```

## 17. AI 저작권 논쟁 구조

```mermaid
flowchart TD
    A["AI 생성물"]

    A --> B["학습 데이터<br>저작권"]
    A --> C["AI 모델<br>저작권"]
    A --> D["생성물<br>저작권"]

    B --> B1["원저작자 권리?"]
    C --> C1["개발사 권리?"]
    D --> D1["사용자 권리?<br>AI 권리?"]

    style A fill:#e0e7ff
    style D1 fill:#fef3c7
```

## 18. 주요 오픈소스 라이선스 비교

```mermaid
flowchart TD
    A["오픈소스 라이선스"]

    A --> MIT["MIT License"]
    A --> Apache["Apache 2.0"]
    A --> GPL["GPL"]
    A --> Commercial["상업용 라이선스"]

    MIT --> MIT1["가장 자유로움<br>출처 표기만"]
    Apache --> Apache1["특허권 조항 포함"]
    GPL --> GPL1["파생물도<br>오픈소스 공개"]
    Commercial --> Commercial1["비용 지불 필요"]

    style MIT fill:#dcfce7
    style GPL fill:#fef3c7
    style Commercial fill:#fecaca
```

## 19. Getty vs Stability AI 소송 구조

```mermaid
flowchart LR
    A["Getty Images<br>(이미지 제공업체)"]
    B["소송 제기"]
    C["Stability AI<br>(AI 개발사)"]

    A --> B --> C

    D["주장: 이미지 무단 학습"]
    E["현재: 소송 진행 중"]

    B --> D
    D --> E

    style A fill:#dbeafe
    style C fill:#fce7f3
```

## 20. AI 윤리 점검 프로세스

```mermaid
flowchart TD
    A["AI 프로젝트 시작"]
    B["윤리 영향 평가"]
    C["데이터 편향 검토"]
    D["보안 리스크 평가"]
    E["법적 검토"]
    F["이해관계자 협의"]
    G["승인"]
    H["개발 진행"]

    A --> B --> C --> D --> E --> F --> G --> H

    style B fill:#dbeafe
    style G fill:#dcfce7
```

## 21. 국제 AI 윤리 가이드라인 비교

```mermaid
flowchart TD
    A["국제 AI 윤리 기준"]

    A --> B["EU AI Act<br>(2024)"]
    A --> C["한국 AI 윤리기준<br>(2020)"]
    A --> D["OECD AI 원칙<br>(2019)"]

    B --> B1["위험도별<br>규제 차등"]
    C --> C1["사람 중심<br>AI 강조"]
    D --> D1["인류 가치<br>존중"]

    style A fill:#1e40af,color:#fff
```

## 22. AI 위험도 분류 (EU AI Act)

```mermaid
flowchart TD
    A["AI 시스템 위험도"]

    A --> B["금지<br>Unacceptable"]
    A --> C["고위험<br>High-risk"]
    A --> D["제한적<br>Limited"]
    A --> E["최소<br>Minimal"]

    B --> B1["사회적 점수 시스템<br>실시간 생체인식"]
    C --> C1["채용, 신용평가<br>의료 진단 AI"]
    D --> D1["챗봇<br>(고지 의무)"]
    E --> E1["스팸 필터<br>게임 AI"]

    style B fill:#fecaca,stroke:#ef4444
    style C fill:#fef3c7,stroke:#f59e0b
    style D fill:#dbeafe,stroke:#3b82f6
    style E fill:#dcfce7,stroke:#22c55e
```

## 23. 제조업 AI 윤리 적용 영역

```mermaid
flowchart TD
    A["제조업 AI 활용"]

    A --> B["품질 예측"]
    A --> C["설비 이상 탐지"]
    A --> D["작업자 모니터링"]
    A --> E["공정 최적화"]

    B --> B1["공정성: 라인별<br>편향 확인"]
    C --> C1["안전성: 오경보<br>비율 관리"]
    D --> D1["개인정보: 동의<br>및 익명화"]
    E --> E1["투명성: 변경<br>이유 설명"]

    style A fill:#1e40af,color:#fff
```

## 24. AI 도입 체크리스트

```mermaid
flowchart TD
    A["AI 도입 전 체크리스트"]

    A --> B["1. 학습 데이터<br>편향 검토"]
    A --> C["2. 설명 가능성<br>확보"]
    A --> D["3. 책임자<br>지정"]
    A --> E["4. 보안 리스크<br>평가"]
    A --> F["5. 법적 준수<br>확인"]

    style A fill:#dbeafe
```

## 25. 1차시 핵심 요약

```mermaid
flowchart LR
    subgraph 윤리["AI 윤리 4원칙"]
        A1["공정성"]
        A2["투명성"]
        A3["책임성"]
        A4["안전성"]
    end

    subgraph 보안["데이터 보안 3영역"]
        B1["개인정보"]
        B2["기업 기밀"]
        B3["접근 통제"]
    end

    subgraph 법률["법적 이슈"]
        C1["저작권"]
        C2["라이선스"]
    end

    style 윤리 fill:#dbeafe
    style 보안 fill:#dcfce7
    style 법률 fill:#fef3c7
```

## 26. 다음 차시 연결

```mermaid
flowchart LR
    A["1차시<br>AI 윤리와<br>데이터 보호"]
    B["2차시<br>Python<br>시작하기"]
    C["3차시<br>제조 데이터<br>다루기 기초"]
    D["4차시<br>공개 데이터셋<br>확보"]

    A --> B --> C --> D

    style A fill:#dbeafe,stroke:#3b82f6
    style B fill:#dcfce7,stroke:#22c55e
```

## 27. 퀴즈: AI 윤리 원칙 매칭

```mermaid
flowchart LR
    subgraph 상황["상황"]
        Q1["AI 결정 이유를<br>설명할 수 있다"]
        Q2["특정 집단에<br>불이익이 없다"]
        Q3["문제 발생 시<br>책임자가 있다"]
        Q4["오작동 시<br>비상정지 가능"]
    end

    subgraph 원칙["원칙"]
        A1["투명성"]
        A2["공정성"]
        A3["책임성"]
        A4["안전성"]
    end

    Q1 -.-> A1
    Q2 -.-> A2
    Q3 -.-> A3
    Q4 -.-> A4
```

## 28. 토론 주제 구조

```mermaid
flowchart TD
    A["토론 주제"]

    A --> B["토론 1:<br>AI가 작업자 불량률<br>높다고 분석하면?"]
    A --> C["토론 2:<br>보안 vs 편의성<br>어떻게 균형?"]
    A --> D["토론 3:<br>AI 오판단 시<br>책임은 누구?"]

    B --> B1["공정성 관점"]
    C --> C1["위험 수준별 접근"]
    D --> D1["책임 체계 정의"]

    style A fill:#e0e7ff
```
