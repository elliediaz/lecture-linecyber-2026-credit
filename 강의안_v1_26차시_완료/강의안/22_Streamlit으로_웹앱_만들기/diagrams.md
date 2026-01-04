# [22차시] Streamlit으로 웹앱 만들기 - 다이어그램

## 1. Streamlit이란?

```mermaid
flowchart LR
    A["Python 코드<br>(app.py)"]
    B["Streamlit"]
    C["웹 애플리케이션"]
    D["브라우저"]

    A --> B --> C --> D
```

## 2. 기존 방식 vs Streamlit

```mermaid
flowchart TD
    subgraph 기존["기존 웹 개발"]
        A1["HTML"]
        A2["CSS"]
        A3["JavaScript"]
        A4["Python 백엔드"]
    end

    subgraph Streamlit["Streamlit"]
        B1["Python만!"]
    end

    style Streamlit fill:#d1fae5
```

## 3. Streamlit 실행 흐름

```mermaid
flowchart TD
    A["app.py 작성"]
    B["streamlit run app.py"]
    C["브라우저 자동 열림"]
    D["http://localhost:8501"]
    E["코드 저장 시 자동 새로고침"]

    A --> B --> C --> D
    E -.-> C
```

## 4. 텍스트 출력 함수

```mermaid
flowchart LR
    subgraph 출력["텍스트 출력"]
        A["st.title()"]
        B["st.header()"]
        C["st.subheader()"]
        D["st.write()"]
        E["st.markdown()"]
    end
```

## 5. 입력 위젯 종류

```mermaid
mindmap
  root((입력 위젯))
    텍스트
      st.text_input
      st.text_area
    숫자
      st.number_input
      st.slider
    선택
      st.selectbox
      st.multiselect
      st.radio
    토글
      st.checkbox
      st.toggle
    버튼
      st.button
      st.download_button
```

## 6. 슬라이더 구조

```mermaid
flowchart LR
    A["st.slider"]
    B["레이블"]
    C["최소값"]
    D["최대값"]
    E["기본값"]
    F["반환값"]

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
```

## 7. 레이아웃 구조

```mermaid
flowchart TD
    subgraph Page["Streamlit 페이지"]
        subgraph Sidebar["st.sidebar"]
            S1["입력 위젯들"]
        end
        subgraph Main["메인 영역"]
            M1["col1"]
            M2["col2"]
        end
    end
```

## 8. 사이드바 패턴

```mermaid
flowchart LR
    subgraph Sidebar["사이드바"]
        A["설정"]
        B["슬라이더"]
        C["선택박스"]
    end

    subgraph Main["메인"]
        D["결과 표시"]
        E["차트"]
        F["테이블"]
    end

    Sidebar --> Main
```

## 9. 컬럼 레이아웃

```mermaid
flowchart TD
    A["st.columns(2)"]
    B["col1, col2"]

    subgraph 화면["화면"]
        C["왼쪽<br>with col1:"]
        D["오른쪽<br>with col2:"]
    end

    A --> B --> 화면
```

## 10. 시각화 옵션

```mermaid
flowchart TD
    A["시각화"]

    A --> B["st.pyplot<br>(Matplotlib)"]
    A --> C["st.line_chart<br>(내장)"]
    A --> D["st.bar_chart<br>(내장)"]
    A --> E["st.plotly_chart<br>(Plotly)"]
```

## 11. ML 모델 통합

```mermaid
flowchart TD
    A["학습된 모델<br>(model.pkl)"]
    B["@st.cache_resource"]
    C["load_model()"]
    D["model.predict()"]
    E["결과 표시"]

    A --> B --> C --> D --> E
```

## 12. 캐싱 종류

```mermaid
flowchart LR
    subgraph Data["@st.cache_data"]
        A1["DataFrame"]
        A2["CSV 파일"]
        A3["API 응답"]
    end

    subgraph Resource["@st.cache_resource"]
        B1["ML 모델"]
        B2["DB 연결"]
        B3["무거운 객체"]
    end
```

## 13. 파일 업로드

```mermaid
flowchart TD
    A["st.file_uploader"]
    B{파일 선택?}
    C["pd.read_csv"]
    D["데이터 처리"]
    E["결과 표시"]

    A --> B
    B -->|예| C --> D --> E
    B -->|아니오| F["대기"]
```

## 14. 세션 상태

```mermaid
flowchart TD
    A["st.session_state"]
    B["값 저장"]
    C["페이지 재실행"]
    D["값 유지"]

    A --> B
    C --> A
    A --> D
```

## 15. 품질 예측 앱 구조

```mermaid
flowchart TD
    subgraph Input["입력"]
        A["온도 슬라이더"]
        B["습도 슬라이더"]
        C["속도 슬라이더"]
    end

    subgraph Process["처리"]
        D["predict_quality()"]
    end

    subgraph Output["출력"]
        E["결과 메시지"]
        F["확률 표시"]
        G["차트"]
    end

    Input --> Process --> Output
```

## 16. 결과 표시 위젯

```mermaid
flowchart LR
    A["결과 표시"]

    A --> B["st.success<br>(초록)"]
    A --> C["st.error<br>(빨강)"]
    A --> D["st.warning<br>(노랑)"]
    A --> E["st.info<br>(파랑)"]
    A --> F["st.metric<br>(큰 숫자)"]
    A --> G["st.progress<br>(진행바)"]
```

## 17. 배포 흐름

```mermaid
flowchart LR
    A["로컬 개발"]
    B["GitHub 푸시"]
    C["Streamlit Cloud"]
    D["배포 완료"]
    E["공개 URL"]

    A --> B --> C --> D --> E
```

## 18. 강의 구조

```mermaid
gantt
    title 22차시 강의 구조 (25-30분)
    dateFormat  mm:ss
    axisFormat %M:%S

    section 이론편
    도입 (복습/목표)          :a1, 00:00, 2m
    Streamlit이란?          :a2, after a1, 2m
    왜 Streamlit?            :a3, after a2, 1m
    기본 사용법              :a4, after a3, 2m
    입력 위젯               :a5, after a4, 1.5m
    레이아웃                :a6, after a5, 1.5m

    section 실습편
    실습 소개               :b1, after a6, 2m
    기본 앱 구조            :b2, after b1, 2m
    입력 위젯               :b3, after b2, 2m
    예측 로직               :b4, after b3, 2m
    결과 표시               :b5, after b4, 2m
    시각화                 :b6, after b5, 2m
    파일 업로드             :b7, after b6, 2m
    ML 모델 연동            :b8, after b7, 2m
    세션 상태               :b9, after b8, 2m

    section 정리
    핵심 요약               :c1, after b9, 1.5m
    다음 차시 예고           :c2, after c1, 1.5m
```

## 19. 핵심 요약

```mermaid
mindmap
  root((Streamlit<br>웹앱))
    기본
      streamlit run
      st.title
      st.write
    위젯
      st.slider
      st.selectbox
      st.button
    레이아웃
      st.sidebar
      st.columns
    시각화
      st.pyplot
      st.line_chart
    최적화
      @st.cache_data
      @st.cache_resource
    배포
      Streamlit Cloud
```

## 20. 다음 단계

```mermaid
flowchart LR
    A["오늘<br>Streamlit<br>(웹 UI)"]
    B["다음<br>FastAPI<br>(백엔드 API)"]
    C["이후<br>모델 해석<br>배포"]

    A --> B --> C
```
