---
marp: true
theme: default
paginate: true
header: 'AI 기초체력훈련 | 22차시'
footer: '© 2026 AI 기초체력훈련'
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2563eb; }
  h2 { color: #1e40af; }
  code { background-color: #f1f5f9; }
---

# Streamlit으로 웹앱 만들기

## 22차시 | AI 기초체력훈련 (Pre AI-Campus)

**코드 몇 줄로 웹 애플리케이션 구축**

---

# 학습목표

이 차시를 마치면 다음을 할 수 있습니다:

1. **Streamlit**의 기본 사용법을 익힌다
2. **대화형 위젯**을 활용한다
3. **ML 모델 예측 웹앱**을 만든다

---

# Streamlit이란?

## 간단한 웹앱 프레임워크

> Python 코드만으로 **웹 애플리케이션**을 만드는 라이브러리

```bash
# 설치
pip install streamlit

# 실행
streamlit run app.py
```

### 장점
- HTML/CSS/JavaScript 몰라도 OK
- Python 코드만으로 UI 구성
- 실시간 반영 (저장하면 자동 새로고침)

---

# Hello, Streamlit!

## 첫 번째 앱

```python
# app.py
import streamlit as st

st.title('🎉 첫 번째 Streamlit 앱')
st.write('안녕하세요!')

name = st.text_input('이름을 입력하세요')
if name:
    st.write(f'환영합니다, {name}님!')
```

```bash
streamlit run app.py
# 브라우저에서 http://localhost:8501 접속
```

---

# 텍스트 출력

## st.write, st.title, st.markdown

```python
import streamlit as st

st.title('제목')
st.header('헤더')
st.subheader('서브헤더')
st.text('일반 텍스트')

st.write('Markdown과 **데이터** 모두 출력 가능!')
st.markdown('### Markdown 문법 *지원*')

# 데이터프레임도 출력 가능
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
st.write(df)
```

---

# 입력 위젯

## 사용자 입력 받기

```python
# 텍스트 입력
name = st.text_input('이름')

# 숫자 입력
age = st.number_input('나이', min_value=0, max_value=100)

# 슬라이더
temperature = st.slider('온도', 0, 100, 50)

# 선택박스
option = st.selectbox('라인 선택', ['A', 'B', 'C'])

# 체크박스
if st.checkbox('상세 정보 표시'):
    st.write('상세 정보...')

# 버튼
if st.button('예측 실행'):
    st.write('예측 중...')
```

---

# 사이드바

## 레이아웃 구성

```python
import streamlit as st

# 사이드바에 위젯 배치
st.sidebar.title('설정')
temperature = st.sidebar.slider('온도', 0, 100, 85)
humidity = st.sidebar.slider('습도', 0, 100, 50)

# 메인 화면
st.title('품질 예측 시스템')
st.write(f'입력값: 온도={temperature}, 습도={humidity}')
```

---

# 시각화

## 차트 표시

```python
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib 차트
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
st.pyplot(fig)

# Streamlit 내장 차트
data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 1, 5]})
st.line_chart(data)
st.bar_chart(data)
```

---

# ML 모델 통합

## 예측 앱 만들기

```python
import streamlit as st
import joblib
import numpy as np

# 모델 로드
model = joblib.load('model.pkl')

st.title('🏭 품질 예측 시스템')

# 입력
temp = st.slider('온도', 70, 100, 85)
humidity = st.slider('습도', 30, 70, 50)
speed = st.slider('속도', 80, 120, 100)

# 예측 버튼
if st.button('예측하기'):
    features = np.array([[temp, humidity, speed]])
    prediction = model.predict(features)[0]
    result = '🔴 불량' if prediction == 1 else '🟢 정상'
    st.success(f'예측 결과: {result}')
```

---

# 파일 업로드

## 데이터 업로드 기능

```python
import streamlit as st
import pandas as pd

st.title('데이터 분석')

uploaded_file = st.file_uploader('CSV 파일 업로드', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('데이터 미리보기:')
    st.dataframe(df.head())

    st.write('기술통계:')
    st.write(df.describe())
```

---

# 컬럼 레이아웃

## 화면 분할

```python
import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.header('왼쪽')
    temp = st.slider('온도', 0, 100)

with col2:
    st.header('오른쪽')
    humidity = st.slider('습도', 0, 100)

# 3개 컬럼
col1, col2, col3 = st.columns(3)
```

---

# 세션 상태

## 값 유지하기

```python
import streamlit as st

# 세션 상태 초기화
if 'count' not in st.session_state:
    st.session_state.count = 0

# 버튼 클릭 시 증가
if st.button('클릭'):
    st.session_state.count += 1

st.write(f'클릭 횟수: {st.session_state.count}')
```

---

# 캐싱

## 성능 최적화

```python
import streamlit as st
import pandas as pd

@st.cache_data  # 데이터 캐싱
def load_data():
    return pd.read_csv('large_data.csv')

@st.cache_resource  # 모델 캐싱
def load_model():
    import joblib
    return joblib.load('model.pkl')

# 처음만 로드, 이후 캐시 사용
df = load_data()
model = load_model()
```

---

# 실습: 품질 예측 대시보드

## 전체 코드

```python
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='품질 예측', page_icon='🏭')
st.title('🏭 제조 품질 예측 시스템')

# 사이드바
st.sidebar.header('입력 파라미터')
temp = st.sidebar.slider('온도 (°C)', 70, 100, 85)
humidity = st.sidebar.slider('습도 (%)', 30, 70, 50)
speed = st.sidebar.slider('속도', 80, 120, 100)

# 예측 (간단한 규칙)
prob = 0.1 + 0.02 * (temp - 85) + 0.01 * (humidity - 50)
prediction = '불량 위험' if prob > 0.3 else '정상'

# 결과 표시
st.metric('예측 결과', prediction)
st.progress(min(prob, 1.0))
```

---

# 배포

## Streamlit Cloud

```yaml
# requirements.txt
streamlit
pandas
numpy
scikit-learn
joblib
```

1. GitHub에 코드 푸시
2. share.streamlit.io 접속
3. 저장소 연결
4. 배포 완료!

> 무료로 웹앱 호스팅 가능

---

# 다음 차시 예고

## 23차시: FastAPI로 모델 서빙

- FastAPI 소개
- REST API 만들기
- 모델 예측 API 구축

> Streamlit은 UI, FastAPI는 **백엔드 API**!

---

# 감사합니다

## AI 기초체력훈련 22차시

**Streamlit으로 웹앱 만들기**

Python만으로 웹앱을 만들었습니다!
