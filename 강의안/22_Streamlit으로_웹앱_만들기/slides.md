---
marp: true
theme: default
paginate: true
header: '제조 AI 과정 | 22차시'
footer: '제조데이터를 활용한 AI 이해와 예측 모델 구축'
style: |
  section {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    background-color: #f8fafc;
  }
  h1 { color: #1e40af; font-size: 2.2em; }
  h2 { color: #2563eb; font-size: 1.6em; }
  h3 { color: #3b82f6; }
  code { background-color: #e2e8f0; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1e293b; color: #e2e8f0; }
---

# Streamlit으로 웹앱 만들기

## 22차시 | Part IV. AI 서비스화와 활용

**Python만으로 웹 애플리케이션 구축하기**

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

# 왜 Streamlit?

## 기존 방식 vs Streamlit

| 구분 | 기존 웹 개발 | Streamlit |
|------|-------------|-----------|
| 언어 | HTML+CSS+JS+Python | Python만 |
| 학습곡선 | 높음 | 낮음 |
| 개발 시간 | 길다 | 빠르다 |
| 용도 | 대규모 서비스 | 프로토타입, 대시보드 |

> 데이터 과학자, 엔지니어가 **빠르게 AI 앱**을 만들 때 적합

---

# Hello, Streamlit!

## 첫 번째 앱

```python
# app.py
import streamlit as st

st.title('첫 번째 Streamlit 앱')
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
df = pd.DataFrame({'온도': [85, 90], '불량': [0, 1]})
st.write(df)
```

---

# 입력 위젯

## 사용자 입력 받기

```python
# 텍스트 입력
name = st.text_input('작업자 이름')

# 숫자 입력
temp = st.number_input('온도', min_value=0, max_value=150)

# 슬라이더
humidity = st.slider('습도', 0, 100, 50)

# 선택박스
line = st.selectbox('라인 선택', ['A라인', 'B라인', 'C라인'])

# 체크박스
if st.checkbox('상세 정보 표시'):
    st.write('상세 정보...')

# 버튼
if st.button('예측 실행'):
    st.write('예측 중...')
```

---

# 입력 위젯 정리

## 주요 위젯

| 위젯 | 용도 | 예시 |
|------|------|------|
| `st.text_input` | 텍스트 입력 | 이름, 코멘트 |
| `st.number_input` | 숫자 입력 | 온도, 수량 |
| `st.slider` | 범위 선택 | 0~100 값 |
| `st.selectbox` | 단일 선택 | 라인 선택 |
| `st.multiselect` | 다중 선택 | 여러 옵션 |
| `st.checkbox` | 체크박스 | 옵션 ON/OFF |
| `st.button` | 버튼 클릭 | 실행 트리거 |

---

# 사이드바

## 레이아웃 구성

```python
import streamlit as st

# 사이드바에 위젯 배치
st.sidebar.title('설정')
temperature = st.sidebar.slider('온도', 70, 100, 85)
humidity = st.sidebar.slider('습도', 30, 70, 50)

# 메인 화면
st.title('품질 예측 시스템')
st.write(f'입력값: 온도={temperature}°C, 습도={humidity}%')
```

> 사이드바에 입력, 메인에 결과 표시하는 패턴

---

# 컬럼 레이아웃

## 화면 분할

```python
import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.header('센서 입력')
    temp = st.slider('온도', 70, 100)
    humidity = st.slider('습도', 30, 70)

with col2:
    st.header('예측 결과')
    st.metric('불량 확률', '15%')
    st.metric('상태', '정상')

# 3개 컬럼
col1, col2, col3 = st.columns(3)
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
ax.plot([1, 2, 3], [85, 90, 88])
ax.set_ylabel('온도')
st.pyplot(fig)

# Streamlit 내장 차트
data = pd.DataFrame({
    '온도': [85, 90, 88, 92],
    '습도': [50, 55, 52, 60]
})
st.line_chart(data)
st.bar_chart(data)
```

---

# 이론 정리

## Streamlit 핵심

| 개념 | 설명 |
|------|------|
| streamlit run | 앱 실행 명령 |
| st.write | 텍스트/데이터 출력 |
| st.slider | 슬라이더 위젯 |
| st.button | 버튼 위젯 |
| st.sidebar | 사이드바 레이아웃 |
| st.columns | 컬럼 레이아웃 |

---

# - 실습편 -

## 22차시

**품질 예측 웹앱 만들기**

---

# 실습 개요

## 품질 예측 대시보드

### 목표
- 센서값 입력 UI 만들기
- 예측 결과 표시
- 시각화 추가

### 실습 환경
```python
import streamlit as st
import pandas as pd
import numpy as np
```

---

# 실습 1: 기본 앱 구조

## 타이틀과 텍스트

```python
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title='품질 예측',
    page_icon='🏭',
    layout='wide'
)

# 타이틀
st.title('🏭 제조 품질 예측 시스템')
st.write('센서 데이터를 입력하면 품질을 예측합니다.')
```

---

# 실습 2: 입력 위젯

## 센서값 입력

```python
# 사이드바 입력
st.sidebar.header('센서 입력')

temperature = st.sidebar.slider(
    '온도 (°C)',
    min_value=70, max_value=100, value=85
)

humidity = st.sidebar.slider(
    '습도 (%)',
    min_value=30, max_value=70, value=50
)

speed = st.sidebar.slider(
    '속도 (RPM)',
    min_value=80, max_value=120, value=100
)
```

---

# 실습 3: 예측 로직

## 간단한 규칙 기반 예측

```python
# 예측 함수
def predict_quality(temp, humidity, speed):
    """간단한 규칙 기반 예측"""
    score = 0
    if temp > 90:
        score += 30
    if humidity > 60:
        score += 20
    if speed > 110:
        score += 15

    probability = min(score / 100, 1.0)
    return probability

# 예측 실행
prob = predict_quality(temperature, humidity, speed)
```

---

# 실습 4: 결과 표시

## metric과 progress

```python
# 메인 화면에 결과 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader('예측 결과')
    if prob > 0.3:
        st.error(f'⚠️ 불량 위험: {prob:.1%}')
    else:
        st.success(f'✅ 정상: 불량 확률 {prob:.1%}')

with col2:
    st.subheader('불량 확률')
    st.progress(prob)
    st.metric('확률', f'{prob:.1%}')
```

---

# 실습 5: 데이터 테이블

## 입력값 표시

```python
import pandas as pd

st.subheader('입력 데이터')

input_data = pd.DataFrame({
    '항목': ['온도', '습도', '속도'],
    '현재값': [f'{temperature}°C', f'{humidity}%', f'{speed}RPM'],
    '정상범위': ['70-90°C', '30-60%', '80-110RPM']
})

st.dataframe(input_data, use_container_width=True)
```

---

# 실습 6: 차트 추가

## 시각화

```python
import matplotlib.pyplot as plt

st.subheader('센서 현황')

# 막대 차트
fig, ax = plt.subplots()
values = [temperature, humidity, speed/1.2]
labels = ['온도', '습도', '속도']
colors = ['#ef4444' if v > 90 else '#22c55e' for v in values]

ax.bar(labels, values, color=colors)
ax.axhline(y=90, color='red', linestyle='--', label='경고선')
st.pyplot(fig)
```

---

# 실습 7: 파일 업로드

## CSV 데이터 분석

```python
st.subheader('데이터 업로드')

uploaded = st.file_uploader('CSV 파일', type='csv')

if uploaded:
    df = pd.read_csv(uploaded)
    st.write('데이터 미리보기:')
    st.dataframe(df.head())

    st.write('기초 통계:')
    st.write(df.describe())
```

---

# 실습 8: ML 모델 연동

## 학습된 모델 사용

```python
import joblib
import numpy as np

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    return joblib.load('quality_model.pkl')

# 예측 버튼
if st.button('AI 예측 실행'):
    model = load_model()
    features = np.array([[temperature, humidity, speed]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f'불량 예측! (확률: {proba:.1%})')
    else:
        st.success(f'정상 예측! (확률: {1-proba:.1%})')
```

---

# 실습 9: 세션 상태

## 값 유지하기

```python
# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state.history = []

# 예측 기록 저장
if st.button('예측 저장'):
    record = {
        'temp': temperature,
        'humidity': humidity,
        'prob': prob
    }
    st.session_state.history.append(record)

# 기록 표시
if st.session_state.history:
    st.write('예측 기록:')
    st.dataframe(pd.DataFrame(st.session_state.history))
```

---

# 실습 10: 캐싱

## 성능 최적화

```python
@st.cache_data  # 데이터 캐싱
def load_sensor_data():
    """대용량 데이터 로드"""
    return pd.read_csv('sensor_data.csv')

@st.cache_resource  # 리소스 캐싱
def load_ml_model():
    """ML 모델 로드"""
    return joblib.load('model.pkl')

# 처음만 로드, 이후 캐시 사용
df = load_sensor_data()
model = load_ml_model()
```

---

# 실습 정리

## 핵심 체크포인트

- [ ] streamlit run으로 앱 실행
- [ ] st.title, st.write로 텍스트 출력
- [ ] st.slider, st.selectbox로 입력 받기
- [ ] st.sidebar로 사이드바 레이아웃
- [ ] st.columns로 화면 분할
- [ ] st.pyplot, st.line_chart로 시각화
- [ ] @st.cache_data로 캐싱

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
matplotlib
```

1. GitHub에 코드 푸시
2. share.streamlit.io 접속
3. 저장소 연결
4. 배포 완료!

> 무료로 웹앱 호스팅 가능

---

# 다음 차시 예고

## 23차시: FastAPI로 예측 서비스 만들기

### 학습 내용
- FastAPI 기초
- REST API 만들기
- 모델 예측 API 구축

> Streamlit은 **UI**, FastAPI는 **백엔드 API**!

---

# 정리 및 Q&A

## 오늘의 핵심

1. **Streamlit**: Python만으로 웹앱 만들기
2. **입력 위젯**: slider, selectbox, button 등
3. **레이아웃**: sidebar, columns
4. **시각화**: pyplot, line_chart, bar_chart
5. **캐싱**: @st.cache_data, @st.cache_resource
6. **배포**: Streamlit Cloud

---

# 감사합니다

## 22차시: Streamlit으로 웹앱 만들기

**Python 코드 몇 줄로 품질 예측 웹앱을 만들었습니다!**
