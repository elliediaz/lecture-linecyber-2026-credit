# [22차시] Streamlit으로 웹앱 만들기 - 강사 스크립트

## 강의 정보
- **차시**: 22차시 (25-30분)
- **유형**: 이론 + 실습
- **구성**: 이론 10분 + 실습 15-20분
- **대상**: 비전공자, AI 입문자, 제조업 종사자

---

## 이론편 (10분)

### 도입 (2분)

#### 인사 및 지난 시간 복습 [1분]

> 안녕하세요, 22차시를 시작하겠습니다.
>
> 지난 시간에 LLM API와 프롬프트 작성법을 배웠습니다. ChatGPT, Claude 같은 대규모 언어 모델을 API로 활용하는 방법을 익혔죠.
>
> 오늘은 지금까지 만든 AI 모델을 **웹 애플리케이션**으로 만들어봅니다. **Streamlit**이라는 라이브러리를 사용합니다.

#### 학습목표 안내 [1분]

> 오늘 수업을 마치면 다음 세 가지를 할 수 있습니다.
>
> 첫째, Streamlit의 기본 사용법을 익힙니다.
> 둘째, 대화형 위젯을 활용합니다.
> 셋째, ML 모델 예측 웹앱을 만듭니다.

---

### 핵심 내용 (8분)

#### Streamlit이란? [2min]

> **Streamlit**은 Python 코드만으로 웹 애플리케이션을 만드는 라이브러리예요.
>
> 보통 웹 앱을 만들려면 HTML, CSS, JavaScript를 알아야 해요. 프론트엔드와 백엔드를 따로 개발해야 하죠. 복잡하고 시간이 오래 걸려요.
>
> Streamlit은 이런 복잡한 과정 없이 **Python만으로** 웹 앱을 만들 수 있어요.
>
> 설치는 `pip install streamlit`, 실행은 `streamlit run app.py`입니다. 브라우저에서 바로 앱을 볼 수 있어요.

#### 왜 Streamlit? [1min]

> 데이터 과학자, 엔지니어가 AI 모델을 만들었어요. 이걸 다른 사람이 쓸 수 있게 하려면 어떻게 해야 할까요?
>
> 예전에는 웹 개발자에게 부탁해야 했어요. 하지만 Streamlit을 쓰면 본인이 직접 빠르게 웹 앱을 만들 수 있습니다.
>
> 프로토타입, 대시보드, 데모용으로 아주 적합해요.

#### 기본 사용법 [2min]

> 첫 번째 앱을 만들어볼게요.
>
> ```python
> import streamlit as st
>
> st.title('첫 번째 앱')
> st.write('안녕하세요!')
>
> name = st.text_input('이름을 입력하세요')
> if name:
>     st.write(f'환영합니다, {name}님!')
> ```
>
> `st.title`로 제목, `st.write`로 텍스트를 출력해요.
> `st.text_input`으로 사용자 입력을 받습니다.
>
> 이 코드를 저장하고 `streamlit run app.py`하면 브라우저에서 앱이 열려요.

#### 입력 위젯 [1.5min]

> Streamlit은 다양한 입력 위젯을 제공해요.
>
> **슬라이더**: `st.slider('온도', 0, 100, 50)` - 숫자 범위 선택
> **선택박스**: `st.selectbox('라인', ['A', 'B', 'C'])` - 드롭다운 선택
> **버튼**: `st.button('실행')` - 클릭 이벤트
> **체크박스**: `st.checkbox('옵션')` - ON/OFF 토글
>
> 이 위젯들을 조합해서 AI 모델 입력 UI를 만들 수 있어요.

#### 레이아웃 [1.5min]

> 화면 레이아웃도 쉽게 구성할 수 있어요.
>
> **사이드바**: `st.sidebar`를 쓰면 왼쪽에 사이드바가 생겨요. 보통 입력 위젯을 사이드바에, 결과를 메인에 표시합니다.
>
> **컬럼**: `col1, col2 = st.columns(2)`로 화면을 2분할할 수 있어요. `with col1:`로 각 컬럼에 내용을 배치합니다.
>
> **시각화**: `st.pyplot(fig)`로 Matplotlib 차트, `st.line_chart(data)`로 Streamlit 내장 차트를 표시해요.

---

## 실습편 (15-20분)

### 실습 소개 [2min]

> 이제 실습 시간입니다. 품질 예측 웹앱을 만들어봅니다.
>
> **실습 목표**입니다.
> 1. 센서값 입력 UI를 만듭니다.
> 2. 예측 결과를 표시합니다.
> 3. 시각화를 추가합니다.
>
> **실습 환경**을 확인해주세요.
>
> ```python
> import streamlit as st
> import pandas as pd
> import numpy as np
> ```

### 실습 1: 기본 앱 구조 [2min]

> 첫 번째 실습입니다. 앱의 기본 구조를 만듭니다.
>
> ```python
> st.set_page_config(page_title='품질 예측', page_icon='🏭', layout='wide')
> st.title('🏭 제조 품질 예측 시스템')
> st.write('센서 데이터를 입력하면 품질을 예측합니다.')
> ```
>
> `set_page_config`로 페이지 제목, 아이콘, 레이아웃을 설정해요. 이건 코드 맨 위에 써야 합니다.

### 실습 2: 입력 위젯 [2min]

> 두 번째 실습입니다. 사이드바에 센서 입력 위젯을 배치합니다.
>
> ```python
> st.sidebar.header('센서 입력')
> temperature = st.sidebar.slider('온도 (°C)', 70, 100, 85)
> humidity = st.sidebar.slider('습도 (%)', 30, 70, 50)
> speed = st.sidebar.slider('속도 (RPM)', 80, 120, 100)
> ```
>
> 슬라이더의 인자는 순서대로 레이블, 최소값, 최대값, 기본값이에요.

### 실습 3: 예측 로직 [2min]

> 세 번째 실습입니다. 간단한 예측 함수를 만듭니다.
>
> ```python
> def predict_quality(temp, humidity, speed):
>     score = 0
>     if temp > 90: score += 30
>     if humidity > 60: score += 20
>     if speed > 110: score += 15
>     return min(score / 100, 1.0)
> ```
>
> 온도, 습도, 속도가 임계값을 넘으면 점수가 올라가고, 이걸 불량 확률로 반환해요.
> 실제로는 여기에 ML 모델을 넣으면 됩니다.

### 실습 4: 결과 표시 [2min]

> 네 번째 실습입니다. 예측 결과를 메인 화면에 표시합니다.
>
> ```python
> if prob > 0.3:
>     st.error(f'⚠️ 불량 위험: {prob:.1%}')
> else:
>     st.success(f'✅ 정상: 불량 확률 {prob:.1%}')
>
> st.progress(prob)
> st.metric('확률', f'{prob:.1%}')
> ```
>
> `st.error`는 빨간 박스, `st.success`는 초록 박스로 표시해요.
> `st.progress`는 진행률 바, `st.metric`은 큰 숫자 표시에 좋습니다.

### 실습 5: 시각화 [2min]

> 다섯 번째 실습입니다. 차트를 추가합니다.
>
> ```python
> fig, ax = plt.subplots()
> values = [temperature, humidity, speed/1.2]
> labels = ['온도', '습도', '속도']
> ax.bar(labels, values)
> st.pyplot(fig)
> ```
>
> Matplotlib 차트를 `st.pyplot`으로 표시해요.
> 정상 범위를 넘은 항목은 빨간색으로 표시하면 더 직관적이죠.

### 실습 6: 파일 업로드 [2min]

> 여섯 번째 실습입니다. CSV 파일 업로드 기능을 추가합니다.
>
> ```python
> uploaded = st.file_uploader('CSV 파일', type='csv')
>
> if uploaded:
>     df = pd.read_csv(uploaded)
>     st.dataframe(df.head())
>     st.write(df.describe())
> ```
>
> 사용자가 파일을 올리면 데이터를 분석해서 보여줄 수 있어요.

### 실습 7: ML 모델 연동 [2min]

> 일곱 번째 실습입니다. 학습된 ML 모델을 연동합니다.
>
> ```python
> @st.cache_resource
> def load_model():
>     return joblib.load('model.pkl')
>
> if st.button('AI 예측'):
>     model = load_model()
>     prediction = model.predict([[temp, humidity, speed]])
> ```
>
> `@st.cache_resource`는 모델을 캐싱해서 매번 로드하지 않게 해요. 성능 최적화에 중요합니다.

### 실습 8: 세션 상태 [2min]

> 여덟 번째 실습입니다. 세션 상태로 값을 유지합니다.
>
> ```python
> if 'history' not in st.session_state:
>     st.session_state.history = []
>
> if st.button('저장'):
>     st.session_state.history.append(record)
> ```
>
> Streamlit은 기본적으로 코드가 매번 다시 실행돼요. `st.session_state`를 쓰면 값을 유지할 수 있습니다.

---

### 정리 (3분)

#### 핵심 요약 [1.5min]

> 오늘 배운 내용을 정리하겠습니다.
>
> **Streamlit**은 Python만으로 웹앱을 만드는 라이브러리예요. `streamlit run app.py`로 실행합니다.
>
> **입력 위젯**으로 slider, selectbox, button 등이 있어요. 사용자 입력을 받습니다.
>
> **레이아웃**은 sidebar, columns로 구성해요. 사이드바에 입력, 메인에 결과 패턴이 일반적이에요.
>
> **캐싱**은 `@st.cache_data`, `@st.cache_resource`로 해요. 성능 최적화에 중요합니다.

#### 다음 차시 예고 [1min]

> 다음 23차시에서는 **FastAPI로 예측 서비스 만들기**를 배웁니다.
>
> Streamlit은 웹 UI를 만들지만, 다른 프로그램에서 호출하려면 API가 필요해요.
> FastAPI로 REST API를 만들어서 AI 모델을 서비스화합니다.

#### 마무리 [0.5min]

> Python 코드 몇 줄로 품질 예측 웹앱을 만들었습니다. 수고하셨습니다!

---

## 강의 노트

### 준비물
- PPT 슬라이드 (slides.md)
- 실습 코드 파일 (code.py)
- Streamlit 설치 확인

### 주의사항
- Streamlit 버전에 따라 API가 조금 다를 수 있음
- cache_data vs cache_resource 구분 설명
- 배포는 Streamlit Cloud가 가장 쉬움

### 예상 질문
1. "저장하면 왜 화면이 새로고침되나요?"
   → Streamlit의 작동 방식. 코드 변경 시 자동 재실행

2. "모델 로드가 느려요"
   → @st.cache_resource 사용

3. "여러 페이지 만들 수 있나요?"
   → 네, pages/ 폴더에 파일 추가하면 멀티페이지 앱 가능

4. "외부에서 접속하려면?"
   → Streamlit Cloud 배포 또는 서버에서 --server.address 0.0.0.0 옵션
