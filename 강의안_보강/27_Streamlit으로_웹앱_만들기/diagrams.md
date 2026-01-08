# [27차시] Streamlit으로 웹앱 만들기 - 다이어그램

## 1. 학습 흐름

```mermaid
flowchart LR
    A["25차시<br>LLM API"]
    B["27차시<br>Streamlit"]
    C["27차시<br>FastAPI"]

    A --> B --> C

    B --> B1["기본 위젯"]
    B --> B2["입력 위젯"]
    B --> B3["앱 개발"]

    style B fill:#1e40af,color:#fff
```

## 2. 대주제 구조

```mermaid
flowchart TD
    A["27차시: Streamlit으로 웹앱 만들기"]

    A --> B["대주제 1<br>기본 위젯"]
    A --> C["대주제 2<br>입력 위젯"]
    A --> D["대주제 3<br>앱 개발"]

    B --> B1["title, write<br>dataframe, chart"]
    C --> C1["button, slider<br>columns, sidebar"]
    D --> D1["모델 로드<br>예측, 배포"]

    style A fill:#1e40af,color:#fff
```

## 3. Streamlit 개념

```mermaid
flowchart LR
    A["Python 코드"]
    B["Streamlit"]
    C["웹앱"]

    A --> B --> C

    B --> B1["HTML/CSS/JS<br>불필요"]

    style B fill:#1e40af,color:#fff
```

## 4. Streamlit 실행 흐름

```mermaid
flowchart LR
    A["app.py 작성"]
    B["streamlit run app.py"]
    C["localhost:8501"]
    D["웹 브라우저"]

    A --> B --> C --> D

    style C fill:#dcfce7
```

## 5. 텍스트 계층 구조

```mermaid
flowchart TD
    A["텍스트 위젯"]

    A --> B["st.title"]
    B --> B1["가장 큰 제목"]

    A --> C["st.header"]
    C --> C1["섹션 제목"]

    A --> D["st.subheader"]
    D --> D1["소제목"]

    A --> E["st.write"]
    E --> E1["일반 내용"]

    style A fill:#1e40af,color:#fff
```

## 6. st.write 만능 출력

```mermaid
flowchart TD
    A["st.write()"]

    A --> B["문자열"]
    A --> C["숫자"]
    A --> D["리스트/딕셔너리"]
    A --> E["DataFrame"]
    A --> F["차트"]

    style A fill:#1e40af,color:#fff
```

## 7. 데이터 표시 위젯

```mermaid
flowchart TD
    A["데이터 표시"]

    A --> B["st.dataframe"]
    B --> B1["인터랙티브<br>정렬, 필터"]

    A --> C["st.table"]
    C --> C1["정적 테이블"]

    A --> D["st.metric"]
    D --> D1["KPI 표시<br>값 + 변화량"]

    style A fill:#1e40af,color:#fff
```

## 8. 차트 위젯

```mermaid
flowchart TD
    A["차트 위젯"]

    A --> B["st.line_chart"]
    B --> B1["라인 차트"]

    A --> C["st.bar_chart"]
    C --> C1["막대 차트"]

    A --> D["st.pyplot"]
    D --> D1["Matplotlib 차트"]

    A --> E["st.plotly_chart"]
    E --> E1["Plotly 인터랙티브"]

    style A fill:#1e40af,color:#fff
```

## 9. 입력 위젯 종류

```mermaid
flowchart TD
    A["입력 위젯"]

    A --> B["st.button"]
    B --> B1["클릭 이벤트"]

    A --> C["st.slider"]
    C --> C1["범위 선택"]

    A --> D["st.selectbox"]
    D --> D1["드롭다운"]

    A --> E["st.text_input"]
    E --> E1["텍스트 입력"]

    A --> F["st.number_input"]
    F --> F1["숫자 입력"]

    style A fill:#1e40af,color:#fff
```

## 10. 버튼 동작

```mermaid
flowchart TD
    A["st.button()"]
    B{"클릭?"}
    C["True 반환"]
    D["로직 실행"]
    E["False"]

    A --> B
    B -->|Yes| C --> D
    B -->|No| E

    style C fill:#dcfce7
```

## 11. 슬라이더 위젯

```mermaid
flowchart LR
    A["st.slider"]
    B["min_value"]
    C["max_value"]
    D["value"]
    E["반환값"]

    A --> B
    A --> C
    A --> D
    A --> E

    style A fill:#fef3c7
```

## 12. 레이아웃 - 열

```mermaid
flowchart LR
    subgraph Columns
        A["col1"]
        B["col2"]
        C["col3"]
    end

    A --> A1["온도 입력"]
    B --> B1["압력 입력"]
    C --> C1["속도 입력"]
```

## 13. 레이아웃 - 사이드바

```mermaid
flowchart LR
    subgraph Page
        subgraph Sidebar
            A["설정"]
            B["모델 선택"]
            C["임계값"]
        end
        subgraph Main
            D["입력 폼"]
            E["결과 표시"]
        end
    end
```

## 14. 레이아웃 - 탭

```mermaid
flowchart TD
    A["st.tabs"]

    A --> B["Tab 1: 데이터"]
    A --> C["Tab 2: 분석"]
    A --> D["Tab 3: 결과"]

    B --> B1["데이터 입력"]
    C --> C1["분석 설정"]
    D --> D1["예측 결과"]

    style A fill:#1e40af,color:#fff
```

## 15. 폼 동작

```mermaid
flowchart TD
    A["st.form"]
    B["입력 위젯들"]
    C["submit_button"]
    D{"제출?"}
    E["모든 입력<br>한 번에 처리"]

    A --> B --> C --> D
    D -->|Yes| E

    style E fill:#dcfce7
```

## 16. 상태 메시지

```mermaid
flowchart TD
    A["상태 메시지"]

    A --> B["st.success"]
    B --> B1["녹색 - 성공"]

    A --> C["st.info"]
    C --> C1["파란색 - 정보"]

    A --> D["st.warning"]
    D --> D1["노란색 - 경고"]

    A --> E["st.error"]
    E --> E1["빨간색 - 에러"]

    style B fill:#dcfce7
    style D fill:#fef3c7
    style E fill:#fecaca
```

## 17. 캐싱 데코레이터

```mermaid
flowchart TD
    A["@st.cache_resource"]
    B["함수 실행"]
    C{"캐시<br>있음?"}
    D["캐시 반환"]
    E["새로 실행"]
    F["캐시 저장"]

    A --> B --> C
    C -->|Yes| D
    C -->|No| E --> F

    style D fill:#dcfce7
```

## 18. 품질 예측 앱 구조

```mermaid
flowchart TD
    A["품질 예측 앱"]

    A --> B["페이지 설정"]
    B --> B1["제목, 아이콘, 레이아웃"]

    A --> C["모델 로드"]
    C --> C1["@st.cache_resource"]

    A --> D["센서 입력"]
    D --> D1["슬라이더, 숫자 입력"]

    A --> E["예측 버튼"]
    E --> E1["model.predict()"]

    A --> F["결과 표시"]
    F --> F1["success/error"]

    style A fill:#1e40af,color:#fff
```

## 19. 앱 실행 흐름

```mermaid
flowchart TD
    A["사용자 입력"]
    B["버튼 클릭"]
    C["DataFrame 생성"]
    D["모델 예측"]
    E["결과 표시"]

    A --> B --> C --> D --> E

    E --> E1{"불량?"}
    E1 -->|Yes| E2["st.error"]
    E1 -->|No| E3["st.success"]

    style E2 fill:#fecaca
    style E3 fill:#dcfce7
```

## 20. Streamlit Cloud 배포

```mermaid
flowchart TD
    A["1. GitHub 업로드"]
    B["2. share.streamlit.io"]
    C["3. 저장소 연결"]
    D["4. Deploy"]
    E["5. 공개 URL"]

    A --> B --> C --> D --> E

    style E fill:#dcfce7
```

## 21. 필요 파일 구조

```mermaid
flowchart TD
    A["프로젝트 폴더"]

    A --> B["app.py"]
    B --> B1["메인 앱 코드"]

    A --> C["requirements.txt"]
    C --> C1["의존성 목록"]

    A --> D["model.pkl"]
    D --> D1["학습된 모델"]

    A --> E[".streamlit/"]
    E --> E1["secrets.toml"]

    style A fill:#1e40af,color:#fff
```

## 22. Secrets 관리

```mermaid
flowchart TD
    A["비밀 정보"]

    A --> B["로컬"]
    B --> B1[".streamlit/secrets.toml"]

    A --> C["Cloud"]
    C --> C1["Settings > Secrets"]

    A --> D["코드"]
    D --> D1["st.secrets['KEY']"]

    style A fill:#fef3c7
```

## 23. 위젯 요약 - 출력

```mermaid
flowchart TD
    A["출력 위젯"]

    A --> B["st.title/header"]
    A --> C["st.write"]
    A --> D["st.dataframe"]
    A --> E["st.metric"]
    A --> F["st.pyplot/line_chart"]

    style A fill:#1e40af,color:#fff
```

## 24. 위젯 요약 - 입력

```mermaid
flowchart TD
    A["입력 위젯"]

    A --> B["st.button"]
    A --> C["st.slider"]
    A --> D["st.selectbox"]
    A --> E["st.text_input"]
    A --> F["st.file_uploader"]

    style A fill:#1e40af,color:#fff
```

## 25. 위젯 요약 - 레이아웃

```mermaid
flowchart TD
    A["레이아웃 위젯"]

    A --> B["st.columns"]
    A --> C["st.sidebar"]
    A --> D["st.tabs"]
    A --> E["st.expander"]
    A --> F["st.form"]

    style A fill:#1e40af,color:#fff
```

## 26. 핵심 정리

```mermaid
flowchart TD
    A["27차시 핵심"]

    A --> B["Streamlit"]
    B --> B1["Python만으로<br>웹앱 개발"]

    A --> C["위젯"]
    C --> C1["출력, 입력<br>레이아웃"]

    A --> D["배포"]
    D --> D1["Streamlit Cloud<br>무료 호스팅"]

    style A fill:#1e40af,color:#fff
```

## 27. 다음 차시 연결

```mermaid
flowchart LR
    A["27차시<br>Streamlit"]
    B["27차시<br>FastAPI"]

    A --> B

    A --> A1["웹 UI"]
    A --> A2["사용자 인터랙션"]

    B --> B1["REST API"]
    B --> B2["프로그램 호출"]

    style A fill:#dbeafe
    style B fill:#dcfce7
```

