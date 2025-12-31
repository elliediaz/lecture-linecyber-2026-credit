# [22차시] Streamlit으로 웹앱 만들기 - 강사 스크립트

## 강의 정보
- **차시**: 22차시 (25분)
- **유형**: 실습 중심
- **대상**: AI 기초체력훈련 수강생 (비전공자/입문자)

---

## 도입 (3분)

### 인사 및 지난 시간 복습 [1.5분]

> 안녕하세요, 22차시를 시작하겠습니다.
>
> 지난 시간에 API의 개념과 requests 라이브러리를 배웠습니다. 외부 서비스와 데이터를 주고받는 방법을 익혔죠.
>
> 오늘은 **Streamlit**으로 웹 애플리케이션을 만들어봅니다. HTML이나 JavaScript 없이 Python만으로 웹앱을 구축할 수 있어요!

### 학습목표 안내 [1.5분]

> 오늘 수업을 마치면 다음을 할 수 있습니다.
>
> 첫째, Streamlit의 기본 사용법을 익힙니다.
> 둘째, 대화형 위젯을 활용합니다.
> 셋째, ML 모델 예측 웹앱을 만듭니다.

---

## 전개 (19분)

### 섹션 1: Streamlit 소개 (4min)

#### Streamlit이란 [2min]

> **Streamlit**은 Python 코드만으로 웹 앱을 만드는 프레임워크입니다.
>
> 데이터 과학자나 ML 엔지니어가 프론트엔드를 몰라도 대시보드나 데모 앱을 빠르게 만들 수 있어요.
>
> ```bash
> pip install streamlit
> streamlit run app.py
> ```

#### 첫 번째 앱 [2min]

> *(코드 시연)*
>
> ```python
> import streamlit as st
>
> st.title('첫 번째 앱')
> st.write('Hello, Streamlit!')
> ```
>
> 이것만으로 웹페이지가 만들어집니다! 저장하면 자동으로 새로고침돼요.

---

### 섹션 2: 위젯과 레이아웃 (7min)

#### 입력 위젯 [3min]

> *(코드 시연)*
>
> ```python
> # 슬라이더
> temp = st.slider('온도', 0, 100, 50)
>
> # 입력 박스
> name = st.text_input('이름')
>
> # 선택박스
> line = st.selectbox('라인', ['A', 'B', 'C'])
>
> # 버튼
> if st.button('실행'):
>     st.write('버튼 클릭!')
> ```
>
> 슬라이더를 움직이면 바로 값이 반영됩니다. 대화형 앱이에요!

#### 사이드바 [2min]

> ```python
> st.sidebar.title('설정')
> temp = st.sidebar.slider('온도', 0, 100)
> ```
>
> 사이드바에 입력 위젯을 모아두면 깔끔합니다.

#### 컬럼 레이아웃 [2min]

> ```python
> col1, col2 = st.columns(2)
> with col1:
>     st.write('왼쪽')
> with col2:
>     st.write('오른쪽')
> ```
>
> 화면을 분할해서 정보를 배치할 수 있어요.

---

### 섹션 3: ML 모델 통합 (6min)

#### 모델 로드 [2min]

> *(코드 시연)*
>
> ```python
> import joblib
>
> @st.cache_resource  # 캐싱으로 한 번만 로드
> def load_model():
>     return joblib.load('model.pkl')
>
> model = load_model()
> ```
>
> @st.cache_resource를 사용하면 모델을 한 번만 로드해서 성능이 좋아져요.

#### 예측 앱 만들기 [4min]

> *(코드 시연)*
>
> ```python
> st.title('품질 예측')
>
> temp = st.slider('온도', 70, 100, 85)
> humidity = st.slider('습도', 30, 70, 50)
>
> if st.button('예측'):
>     pred = model.predict([[temp, humidity]])[0]
>     result = '불량' if pred == 1 else '정상'
>     st.success(f'결과: {result}')
> ```
>
> 슬라이더로 값을 조정하고 버튼을 누르면 예측 결과가 나옵니다!

---

### 섹션 4: 배포 (2min)

#### Streamlit Cloud [2min]

> Streamlit Cloud에서 무료로 배포할 수 있어요.
>
> 1. GitHub에 코드 푸시
> 2. share.streamlit.io 접속
> 3. 저장소 연결
> 4. 배포 완료!
>
> 다른 사람들이 웹 브라우저로 접속해서 사용할 수 있습니다.

---

## 정리 (3분)

### 핵심 내용 요약 [1.5min]

> 오늘 배운 핵심 내용:
>
> 1. **st.title, st.write**: 텍스트 출력
> 2. **st.slider, st.button**: 입력 위젯
> 3. **st.sidebar**: 사이드바 레이아웃
> 4. **@st.cache_resource**: 모델 캐싱
> 5. **Streamlit Cloud**: 무료 배포
>
> Python만으로 웹앱을 만들 수 있습니다!

### 다음 차시 예고 [1min]

> 다음 23차시에서는 **FastAPI**를 배웁니다.
>
> Streamlit은 UI/대시보드용이고, FastAPI는 백엔드 API를 만드는 프레임워크예요.

### 마무리 인사 [0.5분]

> 첫 웹앱을 만들어봤습니다. 수고하셨습니다!

---

## 강의 노트

### 예상 질문
1. "Streamlit vs Flask?"
   → Streamlit은 빠른 프로토타이핑, Flask는 더 자유도 높음

2. "무료로 배포 가능?"
   → Streamlit Cloud 무료 티어 있음

### 시간 조절 팁
- 시간 부족: 배포 부분 생략
- 시간 여유: 파일 업로드 기능 추가
