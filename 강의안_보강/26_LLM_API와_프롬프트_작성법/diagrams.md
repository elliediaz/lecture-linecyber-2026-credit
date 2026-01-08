# [26차시] LLM API와 프롬프트 작성법 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["24차시<br>AI API"]
    B["26차시<br>LLM API"]
    C["26차시<br>Streamlit"]

    A --> B --> C

    B --> B1["LLM API 사용"]
    B --> B2["프롬프트 엔지니어링"]
    B --> B3["제조업 활용"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["26차시: LLM API와 프롬프트 작성법"]

    A --> B["대주제 1<br>LLM과 API 소개"]
    A --> C["대주제 2<br>프롬프트 엔지니어링"]
    A --> D["대주제 3<br>제조업 활용"]

    B --> B1["OpenAI/Claude<br>API 구조"]
    C --> C1["역할, 맥락<br>형식 지정"]
    D --> D1["불량 분석<br>보고서 생성"]

    style A fill:#1e40af,color:#fff
```

## 3. LLM 개념

```mermaid
flowchart LR
    A["자연어 입력"]
    B["LLM<br>(Large Language Model)"]
    C["자연어 출력"]

    A --> B --> C

    B --> B1["이해"]
    B --> B2["생성"]
    B --> B3["번역"]
    B --> B4["요약"]

    style B fill:#1e40af,color:#fff
```

## 4. 주요 LLM 서비스

```mermaid
flowchart TD
    A["LLM 서비스"]

    A --> B["GPT-4<br>(OpenAI)"]
    B --> B1["범용, 가장 널리 사용"]

    A --> C["Claude<br>(Anthropic)"]
    C --> C1["긴 문서, 안전성"]

    A --> D["Gemini<br>(Google)"]
    D --> D1["멀티모달, 검색 연동"]

    A --> E["Llama<br>(Meta)"]
    E --> E1["오픈소스, 자체 배포"]
```

## 5. API 키 관리

```mermaid
flowchart TD
    A["API 키 관리"]

    A --> B["위험한 방법"]
    B --> B1["코드에 직접 입력"]
    B1 --> B2["Git에 노출 위험"]

    A --> C["안전한 방법"]
    C --> C1["환경 변수"]
    C --> C2[".env 파일"]

    style B fill:#fecaca
    style C fill:#dcfce7
```

## 6. API 호출 구조

```mermaid
flowchart LR
    A["Client"]
    B["API 요청"]
    C["LLM Server"]
    D["응답"]

    A --> B --> C
    C --> D --> A

    B --> B1["model"]
    B --> B2["messages"]
    B --> B3["temperature"]

    style C fill:#1e40af,color:#fff
```

## 7. 메시지 역할

```mermaid
flowchart TD
    A["messages"]

    A --> B["system"]
    B --> B1["AI 역할/성격 정의<br>'당신은 전문가입니다'"]

    A --> C["user"]
    C --> C1["사용자 질문<br>'원인을 분석해주세요'"]

    A --> D["assistant"]
    D --> D1["AI 응답<br>대화 이력 유지"]

    style A fill:#1e40af,color:#fff
```

## 8. API 파라미터

```mermaid
flowchart TD
    A["API 파라미터"]

    A --> B["model"]
    B --> B1["gpt-4, claude-3"]

    A --> C["max_tokens"]
    C --> C1["응답 최대 길이"]

    A --> D["temperature"]
    D --> D1["0: 결정적<br>1: 창의적"]

    style A fill:#1e40af,color:#fff
```

## 9. Temperature 효과

```mermaid
flowchart LR
    A["temperature = 0"]
    B["temperature = 0.5"]
    C["temperature = 1.0"]

    A --> A1["항상 같은 답변<br>분석, 사실 확인"]
    B --> B1["적당한 변화<br>일반 대화"]
    C --> C1["다양한 답변<br>창작, 아이디어"]

    style A fill:#dbeafe
    style C fill:#fef3c7
```

## 10. 프롬프트 엔지니어링

```mermaid
flowchart TD
    A["프롬프트 엔지니어링"]

    A --> B["원칙 1: 명확성"]
    B --> B1["구체적으로"]

    A --> C["원칙 2: 맥락"]
    C --> C1["배경 정보 제공"]

    A --> D["원칙 3: 형식"]
    D --> D1["출력 형태 지정"]

    A --> E["원칙 4: 예시"]
    E --> E1["Few-shot 학습"]

    style A fill:#1e40af,color:#fff
```

## 11. 좋은 프롬프트 vs 나쁜 프롬프트

```mermaid
flowchart TD
    A["프롬프트 비교"]

    A --> B["나쁜 예"]
    B --> B1["'불량 분석해줘'"]
    B --> B2["모호함, 맥락 없음"]

    A --> C["좋은 예"]
    C --> C1["'온도 250도, 압력 70kPa에서<br>발생한 스크래치 원인 3가지와<br>해결책을 설명해주세요'"]
    C --> C2["구체적, 형식 지정"]

    style B fill:#fecaca
    style C fill:#dcfce7
```

## 12. 역할 지정 (Role)

```mermaid
flowchart TD
    A["역할 지정"]

    A --> B["system 메시지"]
    B --> B1["'당신은 20년 경력의<br>제조업 품질 관리 전문가입니다'"]

    B --> C["효과"]
    C --> C1["전문적 답변"]
    C --> C2["일관된 톤"]
    C --> C3["도메인 지식 활용"]

    style A fill:#1e40af,color:#fff
```

## 13. 맥락 제공 구조

```mermaid
flowchart TD
    A["맥락 제공"]

    A --> B["[상황]"]
    B --> B1["문제 배경"]

    A --> C["[데이터]"]
    C --> C1["센서값, 수치"]

    A --> D["[질문]"]
    D --> D1["원하는 분석"]

    style A fill:#1e40af,color:#fff
```

## 14. 출력 형식 지정

```mermaid
flowchart TD
    A["출력 형식"]

    A --> B["마크다운"]
    B --> B1["## 제목<br>1. 항목"]

    A --> C["JSON"]
    C --> C1["{status: normal}"]

    A --> D["테이블"]
    D --> D1["| 항목 | 값 |"]

    style A fill:#fef3c7
```

## 15. Few-shot 학습

```mermaid
flowchart TD
    A["Few-shot 학습"]

    A --> B["예시 1"]
    B --> B1["입력: 온도 210도<br>출력: 정상"]

    A --> C["예시 2"]
    C --> C1["입력: 온도 280도<br>출력: 이상"]

    A --> D["실제 질문"]
    D --> D1["입력: 온도 245도<br>출력: ?"]

    style D fill:#dbeafe
```

## 16. Chain of Thought

```mermaid
flowchart TD
    A["Chain of Thought"]

    A --> B["단계 1"]
    B --> B1["데이터 확인"]

    B1 --> C["단계 2"]
    C --> C1["패턴 분석"]

    C1 --> D["단계 3"]
    D --> D1["원인 가설"]

    D1 --> E["단계 4"]
    E --> E1["검증 방법"]

    E1 --> F["단계 5"]
    F --> F1["조치 방안"]

    style A fill:#1e40af,color:#fff
```

## 17. 제조업 활용 사례

```mermaid
flowchart TD
    A["제조업 LLM 활용"]

    A --> B["불량 원인 분석"]
    B --> B1["센서 데이터 + 불량 유형<br>→ 원인과 해결책"]

    A --> C["보고서 생성"]
    C --> C1["데이터 → 품질 보고서"]

    A --> D["이상 해석"]
    D --> D1["알림 → 쉬운 설명"]

    A --> E["문서 요약"]
    E --> E1["매뉴얼 → 핵심 정리"]

    style A fill:#1e40af,color:#fff
```

## 18. ML + LLM 하이브리드

```mermaid
flowchart TD
    A["센서 데이터"]
    B["ML 모델"]
    C["예측 결과<br>(불량 확률 85%)"]
    D["LLM"]
    E["해석 + 권장 사항"]

    A --> B --> C --> D --> E

    style B fill:#dbeafe
    style D fill:#dcfce7
```

## 19. LLM 주의사항

```mermaid
flowchart TD
    A["LLM 주의사항"]

    A --> B["환각"]
    B --> B1["사실 확인 필요"]

    A --> C["일관성"]
    C --> C1["같은 질문, 다른 답"]

    A --> D["보안"]
    D --> D1["민감 정보 주의"]

    A --> E["비용"]
    E --> E1["사용량 모니터링"]

    style A fill:#fef3c7
```

## 20. 비용 최적화

```mermaid
flowchart TD
    A["비용 최적화"]

    A --> B["모델 선택"]
    B --> B1["간단한 작업:<br>저렴한 모델"]

    A --> C["프롬프트 최적화"]
    C --> C1["불필요한 내용 제거"]

    A --> D["캐싱"]
    D --> D1["동일 질문 결과 저장"]

    A --> E["토큰 제한"]
    E --> E1["max_tokens 적절히"]

    style A fill:#dcfce7
```

## 21. API 호출 흐름

```mermaid
sequenceDiagram
    participant App as 애플리케이션
    participant API as LLM API
    participant LLM as LLM 모델

    App->>API: 메시지 전송
    API->>LLM: 추론 요청
    LLM->>API: 응답 생성
    API->>App: 응답 반환
```

## 22. 실습 흐름

```mermaid
flowchart TD
    A["1. 환경 설정"]
    B["2. API 키 설정"]
    C["3. 기본 호출"]
    D["4. 역할 지정"]
    E["5. 형식 지정"]
    F["6. 제조업 활용"]

    A --> B --> C --> D --> E --> F

    style A fill:#dbeafe
    style F fill:#dcfce7
```

## 23. 프롬프트 템플릿

```mermaid
flowchart TD
    A["프롬프트 템플릿"]

    A --> B["역할"]
    B --> B1["당신은 {role}입니다"]

    A --> C["맥락"]
    C --> C1["[상황] {context}"]

    A --> D["데이터"]
    D --> D1["[데이터] {data}"]

    A --> E["요청"]
    E --> E1["[질문] {question}"]

    A --> F["형식"]
    F --> F1["[형식] {format}"]

    style A fill:#1e40af,color:#fff
```

## 24. 응답 처리

```mermaid
flowchart TD
    A["API 응답"]
    B["response.choices[0]"]
    C["message.content"]
    D["텍스트 추출"]

    A --> B --> C --> D

    D --> E{"JSON 형식?"}
    E -->|Yes| F["json.loads()"]
    E -->|No| G["텍스트 처리"]

    style D fill:#dcfce7
```

## 25. 핵심 코드 구조

```mermaid
flowchart TD
    A["OpenAI API"]

    A --> B["client = OpenAI()"]
    A --> C["chat.completions.create()"]
    A --> D["messages = [...]"]
    A --> E["response.choices[0].message.content"]

    style A fill:#1e40af,color:#fff
```

## 26. 핵심 정리

```mermaid
flowchart TD
    A["26차시 핵심"]

    A --> B["LLM API"]
    B --> B1["OpenAI/Claude<br>자연어 처리"]

    A --> C["프롬프트"]
    C --> C1["역할, 맥락<br>형식 지정"]

    A --> D["제조업 활용"]
    D --> D1["분석, 보고서<br>ML 조합"]

    style A fill:#1e40af,color:#fff
```

## 27. 다음 차시 연결

```mermaid
flowchart LR
    A["26차시<br>LLM API"]
    B["26차시<br>Streamlit"]

    A --> B

    A --> A1["프롬프트"]
    A --> A2["API 호출"]

    B --> B1["웹 UI"]
    B --> B2["인터랙티브 앱"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

