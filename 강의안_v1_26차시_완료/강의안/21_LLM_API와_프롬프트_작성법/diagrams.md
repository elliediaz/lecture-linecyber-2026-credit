# [21차시] LLM API와 프롬프트 작성법 - 다이어그램

## 1. LLM이란?

```mermaid
flowchart TD
    subgraph LLM["대규모 언어 모델 (LLM)"]
        A["엄청난 양의<br>텍스트 데이터"]
        B["딥러닝 학습"]
        C["언어 이해/생성"]
    end

    A --> B --> C
    C --> D["자연어 입력"]
    C --> E["자연어 출력"]
```

## 2. LLM 종류

```mermaid
mindmap
  root((LLM))
    OpenAI
      GPT-4
      GPT-3.5
      ChatGPT
    Anthropic
      Claude 3 Opus
      Claude 3 Sonnet
      Claude 3 Haiku
    Google
      Gemini Pro
      Gemini Ultra
    Meta
      LLaMA 2
      LLaMA 3
```

## 3. LLM 능력

```mermaid
flowchart LR
    A["LLM"]

    A --> B["텍스트 생성"]
    A --> C["요약"]
    A --> D["번역"]
    A --> E["질의응답"]
    A --> F["코드 작성"]
    A --> G["데이터 분석"]
```

## 4. 제조 현장 LLM 활용

```mermaid
flowchart TD
    subgraph 문서작업["문서 작업"]
        A1["작업 지시서 생성"]
        A2["품질 보고서 요약"]
        A3["매뉴얼 Q&A"]
    end

    subgraph 데이터분석["데이터 분석"]
        B1["불량 원인 분석"]
        B2["센서 데이터 해석"]
        B3["이상 패턴 설명"]
    end

    subgraph 커뮤니케이션["커뮤니케이션"]
        C1["다국어 번역"]
        C2["기술 문서 작성"]
    end
```

## 5. API 호출 흐름

```mermaid
sequenceDiagram
    participant P as Python 코드
    participant A as OpenAI/Claude API
    participant L as LLM 모델

    P->>A: 메시지 전송
    Note over P,A: messages=[{role, content}]
    A->>L: 프롬프트 처리
    L->>A: 텍스트 생성
    A->>P: 응답 반환
    Note over P,A: response.choices[0]
```

## 6. 메시지 구조

```mermaid
flowchart TD
    A["messages 리스트"]

    A --> B["system<br>AI 역할 설정"]
    A --> C["user<br>사용자 질문"]
    A --> D["assistant<br>AI 응답"]

    B --> B1["당신은 제조<br>전문가입니다"]
    C --> C1["불량 원인을<br>알려주세요"]
    D --> D1["불량의 주요<br>원인은..."]
```

## 7. 메시지 역할별 기능

```mermaid
flowchart LR
    subgraph System["system 역할"]
        S1["역할 부여"]
        S2["성격 설정"]
        S3["제약 조건"]
    end

    subgraph User["user 역할"]
        U1["질문"]
        U2["요청"]
        U3["데이터"]
    end

    subgraph Assistant["assistant 역할"]
        A1["이전 응답"]
        A2["대화 기록"]
    end
```

## 8. OpenAI API 구조

```mermaid
flowchart TD
    A["from openai import OpenAI"]
    B["client = OpenAI(api_key=...)"]
    C["client.chat.completions.create()"]
    D["model='gpt-4'"]
    E["messages=[...]"]
    F["response"]
    G["response.choices[0].message.content"]

    A --> B --> C
    C --> D
    C --> E
    C --> F --> G
```

## 9. Claude API 구조

```mermaid
flowchart TD
    A["import anthropic"]
    B["client = anthropic.Anthropic(api_key=...)"]
    C["client.messages.create()"]
    D["model='claude-3-sonnet-...'"]
    E["messages=[...]"]
    F["message"]
    G["message.content[0].text"]

    A --> B --> C
    C --> D
    C --> E
    C --> F --> G
```

## 10. 프롬프트란?

```mermaid
flowchart LR
    A["프롬프트<br>(Prompt)"]
    B["AI에게 전달하는<br>지시/질문"]
    C["결과"]

    A --> B --> C

    style A fill:#fef3c7
    style C fill:#d1fae5
```

## 11. 좋은 프롬프트 조건

```mermaid
flowchart TD
    A["좋은 프롬프트"]

    A --> B["명확한 지시"]
    A --> C["맥락 제공"]
    A --> D["출력 형식 지정"]
    A --> E["예시 제공"]

    B --> B1["구체적으로<br>무엇을 원하는지"]
    C --> C1["배경 정보<br>포함"]
    D --> D1["원하는 형태<br>명시"]
    E --> E1["몇 가지<br>예시로 안내"]
```

## 12. 프롬프트 구조화

```mermaid
flowchart TD
    subgraph Prompt["구조화된 프롬프트"]
        A["[역할 설정]<br>당신은 제조 전문가입니다"]
        B["[맥락]<br>온도 센서가 90°C 초과"]
        C["[질문]<br>취해야 할 조치 3가지"]
        D["[출력 형식]<br>번호 목록으로"]
    end

    A --> B --> C --> D
```

## 13. 프롬프트 비교

```mermaid
flowchart LR
    subgraph Bad["나쁜 프롬프트"]
        A1["좋은 코드<br>써줘"]
    end

    subgraph Good["좋은 프롬프트"]
        B1["에러 처리 포함된<br>Python 코드로<br>CSV 파일을 읽는<br>함수를 작성해주세요"]
    end

    Bad -.->|"개선"| Good

    style Bad fill:#fee2e2
    style Good fill:#d1fae5
```

## 14. Few-shot 프롬프트

```mermaid
flowchart TD
    A["Few-shot 학습"]

    A --> B["예시 1"]
    A --> C["예시 2"]
    A --> D["새 입력"]
    D --> E["AI 추론"]

    B --> B1["입력: 온도=85<br>출력: 정상"]
    C --> C1["입력: 온도=95<br>출력: 경고"]
    D --> D1["입력: 온도=88<br>출력: ?"]
```

## 15. 멀티턴 대화

```mermaid
sequenceDiagram
    participant U as User
    participant A as Assistant
    participant M as messages[]

    U->>M: "불량률이 올랐어요"
    M->>A: messages 전송
    A->>M: "원인 분석이 필요합니다"
    U->>M: "온도가 원인일까요?"
    Note over M: 이전 대화 포함
    M->>A: 전체 messages 전송
    A->>M: "네, 온도 상승이..."
```

## 16. 대화 기록 유지

```mermaid
flowchart TD
    A["messages = []"]

    A --> B["system 추가"]
    B --> C["user 질문 1 추가"]
    C --> D["API 호출"]
    D --> E["assistant 응답 추가"]
    E --> F["user 질문 2 추가"]
    F --> G["API 호출<br>(전체 기록 포함)"]
```

## 17. JSON 응답

```mermaid
flowchart LR
    subgraph Input["프롬프트"]
        A["JSON 형식으로<br>응답해주세요"]
    end

    subgraph Output["응답"]
        B["{<br>'status': '경고',<br>'issues': [...],<br>'recommendations': [...]<br>}"]
    end

    subgraph Parse["파싱"]
        C["json.loads()"]
        D["Python dict"]
    end

    Input --> Output --> Parse
    C --> D
```

## 18. 강의 구조

```mermaid
gantt
    title 21차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    LLM이란?                :a2, after a1, 2m
    LLM 활용                :a3, after a2, 1.5m
    OpenAI API              :a4, after a3, 2m
    Claude API              :a5, after a4, 1m
    프롬프트 작성법           :a6, after a5, 1.5m

    section 실습편
    실습 소개               :b1, after a6, 2m
    기본 API 호출           :b2, after b1, 3m
    시스템 프롬프트          :b3, after b2, 2m
    데이터 분석 요청         :b4, after b3, 3m
    보고서 생성             :b5, after b4, 3m
    멀티턴 대화             :b6, after b5, 3m
    JSON 응답              :b7, after b6, 2m

    section 정리
    핵심 요약               :c1, after b7, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((LLM API<br>프롬프트))
    LLM
      GPT
      Claude
      자연어 대화
    API 호출
      messages 리스트
      system/user/assistant
      응답 파싱
    프롬프트
      명확한 지시
      맥락 제공
      출력 형식
      Few-shot
    활용
      보고서 생성
      데이터 분석
      멀티턴 대화
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>LLM API"]
    B["다음<br>Streamlit"]
    C["이후<br>FastAPI"]

    A --> B --> C

    A -.-> A1["프롬프트 작성"]
    B -.-> B1["웹앱 만들기"]
    C -.-> C1["API 서빙"]
```
