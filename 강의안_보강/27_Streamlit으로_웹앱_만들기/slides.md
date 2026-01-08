---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# [27차시] Streamlit으로 웹앱 만들기

## ML 모델을 인터랙티브 웹으로

---

# 학습 목표

1. **Streamlit** 기본 위젯을 사용한다
2. **입력 위젯**으로 사용자 인터랙션을 구현한다
3. **품질 예측 웹앱**을 만들고 배포한다

---

# 지난 시간 복습

- **LLM API**: OpenAI / Claude 호출
- **프롬프트 엔지니어링**: 역할, 맥락, 형식
- **제조업 활용**: 불량 분석, 보고서 생성

**오늘**: 만든 모델을 웹에서 사용할 수 있게

---

# 수업 흐름

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | Streamlit 소개와 기본 위젯 |
| 대주제 2 | 10분 | 입력 위젯과 레이아웃 |
| 대주제 3 | 8분 | 품질 예측 앱 만들기 |
| 정리 | 2분 | 핵심 요약 |

---

<!-- _class: lead -->
# 대주제 1
## Streamlit 소개와 기본 위젯

---

# Streamlit이란?

**Python만으로 웹앱을 만드는 프레임워크**

- HTML, CSS, JavaScript 필요 없음
- Python 코드 = 웹 페이지
- 데이터 시각화, ML 앱에 최적화

```python
import streamlit as st

st.title("품질 예측 시스템")
st.write("센서 데이터를 입력하세요.")
```

---

# Streamlit 설치

```bash
pip install streamlit
```

**실행 방법**:
```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

---

# 첫 번째 앱

```python
# app.py
import streamlit as st

st.title("Hello, Streamlit!")
st.write("첫 번째 웹앱입니다.")
```

실행:
```bash
streamlit run app.py
```

**3줄로 웹앱 완성!**

---

# st.title, st.header, st.subheader

```python
import streamlit as st

st.title("제목 (가장 큼)")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
```

**텍스트 계층 구조**

---

# st.write - 만능 출력

```python
import streamlit as st
import pandas as pd

# 다양한 타입 출력
st.write("문자열")
st.write(123)
st.write([1, 2, 3])
st.write({"key": "value"})

# DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
st.write(df)
```

**타입 자동 감지하여 최적 렌더링**

---

# st.markdown

```python
import streamlit as st

st.markdown("""
# 마크다운 제목

**굵은 글씨**와 *기울임*

- 리스트 항목 1
- 리스트 항목 2

[링크](https://streamlit.io)
""")
```

마크다운 문법 그대로 사용

---

# st.dataframe - 인터랙티브 테이블

```python
import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'temperature': [200, 210, 220],
    'pressure': [50, 55, 60],
    'status': ['정상', '정상', '경고']
})

# 인터랙티브 (정렬, 필터)
st.dataframe(df)

# 정적 테이블
st.table(df)
```

---

# st.metric - KPI 표시

```python
import streamlit as st

col1, col2, col3 = st.columns(3)

col1.metric("불량률", "2.3%", "-0.2%")
col2.metric("가동률", "94.5%", "+1.2%")
col3.metric("생산량", "15,234", "+234")
```

**변화량(delta) 자동 색상 표시**

---

# st.pyplot - Matplotlib 차트

```python
import streamlit as st
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_xlabel("시간")
ax.set_ylabel("온도")

st.pyplot(fig)
```

---

# st.line_chart, st.bar_chart

```python
import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 15, 25]
})

# 라인 차트
st.line_chart(df)

# 바 차트
st.bar_chart(df)
```

**내장 차트 (Altair 기반)**

---

# 데이터 시각화 예제

```python
import streamlit as st
import pandas as pd
import numpy as np

st.title("생산 데이터 대시보드")

# 샘플 데이터
dates = pd.date_range('2026-01-01', periods=30)
df = pd.DataFrame({
    '생산량': np.random.randint(100, 150, 30),
    '불량수': np.random.randint(2, 10, 30)
}, index=dates)

st.line_chart(df)
```

---

<!-- _class: lead -->
# 대주제 2
## 입력 위젯과 레이아웃

---

# st.button - 버튼

```python
import streamlit as st

if st.button("분석 시작"):
    st.write("분석을 시작합니다...")
    # 분석 로직
    st.success("분석 완료!")
```

**버튼 클릭 시 True 반환**

---

# st.text_input - 텍스트 입력

```python
import streamlit as st

name = st.text_input("설비명", value="CNC 1호기")
st.write(f"입력된 설비: {name}")

# 숫자 입력
age = st.number_input("온도", min_value=0, max_value=500, value=200)
st.write(f"입력된 온도: {age}도")
```

---

# st.slider - 슬라이더

```python
import streamlit as st

# 단일 값
temperature = st.slider("온도", 100, 300, 200)
st.write(f"설정 온도: {temperature}도")

# 범위
temp_range = st.slider("온도 범위", 100, 300, (150, 250))
st.write(f"범위: {temp_range[0]} ~ {temp_range[1]}도")
```

---

# st.selectbox - 드롭다운

```python
import streamlit as st

# 단일 선택
line = st.selectbox(
    "생산 라인",
    ["라인 A", "라인 B", "라인 C"]
)
st.write(f"선택된 라인: {line}")

# 다중 선택
features = st.multiselect(
    "분석 항목",
    ["온도", "압력", "속도", "진동"]
)
st.write(f"선택된 항목: {features}")
```

---

# st.checkbox, st.radio

```python
import streamlit as st

# 체크박스
agree = st.checkbox("데이터 전처리 적용")
if agree:
    st.write("전처리가 적용됩니다.")

# 라디오 버튼
model = st.radio(
    "모델 선택",
    ["RandomForest", "XGBoost", "Neural Network"]
)
st.write(f"선택된 모델: {model}")
```

---

# st.file_uploader - 파일 업로드

```python
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터:")
    st.dataframe(df)
```

---

# st.columns - 열 레이아웃

```python
import streamlit as st

col1, col2, col3 = st.columns(3)

with col1:
    st.header("온도")
    temp = st.number_input("온도(도)", value=200)

with col2:
    st.header("압력")
    pres = st.number_input("압력(kPa)", value=50)

with col3:
    st.header("속도")
    speed = st.number_input("속도(rpm)", value=100)
```

---

# st.sidebar - 사이드바

```python
import streamlit as st

# 사이드바에 위젯 배치
st.sidebar.title("설정")
model = st.sidebar.selectbox("모델", ["RF", "XGB"])
threshold = st.sidebar.slider("임계값", 0.0, 1.0, 0.5)

# 메인 영역
st.title("품질 예측")
st.write(f"모델: {model}, 임계값: {threshold}")
```

---

# st.expander - 접기/펼치기

```python
import streamlit as st

with st.expander("상세 설정 보기"):
    st.write("고급 옵션")
    n_estimators = st.number_input("트리 수", value=100)
    max_depth = st.number_input("최대 깊이", value=10)
```

---

# st.tabs - 탭

```python
import streamlit as st

tab1, tab2, tab3 = st.tabs(["데이터", "분석", "결과"])

with tab1:
    st.header("데이터 입력")
    # 데이터 입력 위젯

with tab2:
    st.header("분석 설정")
    # 분석 옵션

with tab3:
    st.header("예측 결과")
    # 결과 표시
```

---

# st.form - 폼 (일괄 제출)

```python
import streamlit as st

with st.form("prediction_form"):
    st.write("센서 데이터 입력")

    temp = st.number_input("온도", value=200)
    pres = st.number_input("압력", value=50)
    speed = st.number_input("속도", value=100)

    submitted = st.form_submit_button("예측 실행")

    if submitted:
        st.write(f"온도: {temp}, 압력: {pres}, 속도: {speed}")
```

**폼 내 위젯은 제출 버튼 클릭 시만 반영**

---

# 상태 메시지

```python
import streamlit as st

st.success("성공 메시지")
st.info("정보 메시지")
st.warning("경고 메시지")
st.error("에러 메시지")
```

---

# st.progress, st.spinner

```python
import streamlit as st
import time

# 프로그레스 바
progress = st.progress(0)
for i in range(100):
    progress.progress(i + 1)
    time.sleep(0.01)

# 스피너
with st.spinner("분석 중..."):
    time.sleep(2)
st.success("완료!")
```

---

<!-- _class: lead -->
# 대주제 3
## 품질 예측 앱 만들기

---

# 앱 구조 설계

```
품질 예측 웹앱
├── 사이드바: 설정
│   ├── 모델 선택
│   └── 임계값 설정
├── 메인: 입력/결과
│   ├── 센서 데이터 입력
│   ├── 예측 버튼
│   └── 결과 표시
└── 탭: 추가 기능
    ├── 데이터 시각화
    └── 히스토리
```

---

# 기본 앱 템플릿

```python
import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="품질 예측 시스템",
    page_icon="🏭",
    layout="wide"
)

# 타이틀
st.title("🏭 제조 품질 예측 시스템")
st.markdown("센서 데이터를 입력하여 품질을 예측합니다.")
```

---

# 모델 로드

```python
import streamlit as st
import joblib

@st.cache_resource  # 모델 캐싱
def load_model():
    return joblib.load('quality_pipeline.pkl')

try:
    model = load_model()
    st.sidebar.success("모델 로드 완료")
except:
    st.sidebar.error("모델 로드 실패")
    model = None
```

**@st.cache_resource: 모델 한 번만 로드**

---

# 센서 입력 섹션

```python
st.header("📊 센서 데이터 입력")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("온도 (도)", 100, 300, 200)
    pressure = st.slider("압력 (kPa)", 20, 100, 50)
    speed = st.slider("속도 (rpm)", 50, 200, 100)

with col2:
    humidity = st.slider("습도 (%)", 20, 80, 50)
    vibration = st.slider("진동 (mm/s)", 0.0, 15.0, 5.0)
```

---

# 예측 실행

```python
if st.button("🔍 품질 예측", type="primary"):
    # 입력 데이터 구성
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'pressure': [pressure],
        'speed': [speed],
        'humidity': [humidity],
        'vibration': [vibration]
    })

    # 예측
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # 결과 표시
    if prediction[0] == 1:
        st.error(f"⚠️ 불량 예측 (확률: {probability[0][1]:.1%})")
    else:
        st.success(f"✅ 정상 예측 (확률: {probability[0][0]:.1%})")
```

---

# 결과 시각화

```python
import plotly.express as px

# 게이지 차트
fig = px.pie(
    values=[probability[0][0], probability[0][1]],
    names=['정상', '불량'],
    color_discrete_sequence=['#00CC96', '#EF553B']
)
st.plotly_chart(fig)

# 입력값 표시
st.subheader("입력 데이터")
st.dataframe(input_data.T.rename(columns={0: '값'}))
```

---

# 완성된 앱 코드 (요약)

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="품질 예측", layout="wide")
st.title("🏭 품질 예측 시스템")

# 모델 로드
model = joblib.load('quality_pipeline.pkl')

# 입력
col1, col2 = st.columns(2)
with col1:
    temp = st.slider("온도", 100, 300, 200)
    pres = st.slider("압력", 20, 100, 50)
with col2:
    speed = st.slider("속도", 50, 200, 100)
    vib = st.slider("진동", 0.0, 15.0, 5.0)

# 예측
if st.button("예측"):
    X = pd.DataFrame([[temp, pres, speed, 50, vib]],
                     columns=['temperature','pressure','speed','humidity','vibration'])
    pred = model.predict(X)[0]
    st.write("불량" if pred == 1 else "정상")
```

---

# Streamlit Cloud 배포

1. GitHub에 코드 업로드
2. https://share.streamlit.io 접속
3. GitHub 저장소 연결
4. "Deploy" 클릭

```
필요 파일:
├── app.py              # 메인 앱
├── requirements.txt    # 의존성
└── quality_pipeline.pkl # 모델 (또는 URL)
```

---

# requirements.txt

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.5.0
plotly>=5.15.0
```

---

# 배포 시 주의사항

| 항목 | 주의점 |
|-----|--------|
| **모델 크기** | 큰 모델은 로드 시간 증가 |
| **메모리** | 무료 티어 제한 있음 |
| **보안** | API 키는 Secrets 사용 |
| **캐싱** | @st.cache_resource 활용 |

---

# Secrets 관리

```python
# .streamlit/secrets.toml (로컬)
API_KEY = "your-api-key"

# 코드에서 접근
import streamlit as st
api_key = st.secrets["API_KEY"]
```

**Streamlit Cloud에서 설정 가능**

---

<!-- _class: lead -->
# 핵심 정리

---

# 오늘 배운 내용

1. **Streamlit 기본**
   - st.title, st.write, st.dataframe
   - st.pyplot, st.line_chart

2. **입력 위젯**
   - st.button, st.slider, st.selectbox
   - st.columns, st.sidebar, st.form

3. **앱 개발**
   - 모델 로드, 예측, 결과 표시
   - Streamlit Cloud 배포

---

# 핵심 코드

```python
import streamlit as st
import pandas as pd
import joblib

# 기본 출력
st.title("제목")
st.write("내용")

# 입력 위젯
value = st.slider("라벨", min, max, default)
if st.button("실행"):
    # 로직

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')
```

---

# 체크리스트

- [ ] Streamlit 설치 및 실행
- [ ] 기본 출력 위젯 사용
- [ ] 입력 위젯으로 인터랙션 구현
- [ ] 레이아웃 (columns, sidebar)
- [ ] 모델 로드 및 예측
- [ ] Streamlit Cloud 배포

---

# 다음 차시 예고

## [27차시] FastAPI로 예측 서비스 만들기

- FastAPI 기본 구조
- Pydantic 데이터 검증
- POST 엔드포인트 구현
- 예측 API 서버 만들기

---

<!-- _class: lead -->
# 수고하셨습니다!

## 실습: 품질 예측 웹앱 만들기

