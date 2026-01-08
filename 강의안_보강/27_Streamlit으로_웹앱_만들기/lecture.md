# 27차시: Streamlit으로 웹앱 만들기

## 학습 목표

1. **Streamlit** 기본 위젯을 사용함
2. **입력 위젯**으로 사용자 인터랙션을 구현함
3. **품질 예측 웹앱**을 만들고 배포함

---

## 강의 구성

| 구간 | 시간 | 내용 |
|:----:|:----:|------|
| 대주제 1 | 10분 | Streamlit 소개와 기본 위젯 |
| 대주제 2 | 10분 | 입력 위젯과 레이아웃 |
| 대주제 3 | 8분 | 품질 예측 앱 만들기 |
| 정리 | 2분 | 핵심 요약 |

---

## 대주제 1: Streamlit 소개와 기본 위젯

### 1.1 Streamlit이란?

**Python만으로 웹앱을 만드는 프레임워크**
- HTML, CSS, JavaScript 필요 없음
- Python 코드 = 웹 페이지
- 데이터 시각화, ML 앱에 최적화

```python
import streamlit as st

st.title("품질 예측 시스템")
st.write("센서 데이터를 입력하세요.")
```

### 1.2 Streamlit 설치 및 실행

```bash
# 설치
pip install streamlit

# 실행
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

### 1.3 첫 번째 앱

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

3줄로 웹앱 완성

### 1.4 텍스트 계층 구조

```python
import streamlit as st

st.title("제목 (가장 큼)")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
```

### 1.5 st.write - 만능 출력

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

타입 자동 감지하여 최적 렌더링

### 1.6 st.markdown

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

### 1.7 st.dataframe - 인터랙티브 테이블

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

### 1.8 st.metric - KPI 표시

```python
import streamlit as st

col1, col2, col3 = st.columns(3)

col1.metric("불량률", "2.3%", "-0.2%")
col2.metric("가동률", "94.5%", "+1.2%")
col3.metric("생산량", "15,234", "+234")
```

변화량(delta) 자동 색상 표시

### 1.9 st.pyplot - Matplotlib 차트

```python
import streamlit as st
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_xlabel("시간")
ax.set_ylabel("온도")

st.pyplot(fig)
```

### 1.10 st.line_chart, st.bar_chart

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

내장 차트 (Altair 기반)

### 1.11 데이터 시각화 예제

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

## 대주제 2: 입력 위젯과 레이아웃

### 2.1 st.button - 버튼

```python
import streamlit as st

if st.button("분석 시작"):
    st.write("분석을 시작합니다...")
    # 분석 로직
    st.success("분석 완료!")
```

버튼 클릭 시 True 반환

### 2.2 st.text_input - 텍스트 입력

```python
import streamlit as st

name = st.text_input("설비명", value="CNC 1호기")
st.write(f"입력된 설비: {name}")

# 숫자 입력
age = st.number_input("온도", min_value=0, max_value=500, value=200)
st.write(f"입력된 온도: {age}도")
```

### 2.3 st.slider - 슬라이더

```python
import streamlit as st

# 단일 값
temperature = st.slider("온도", 100, 300, 200)
st.write(f"설정 온도: {temperature}도")

# 범위
temp_range = st.slider("온도 범위", 100, 300, (150, 250))
st.write(f"범위: {temp_range[0]} ~ {temp_range[1]}도")
```

### 2.4 st.selectbox - 드롭다운

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

### 2.5 st.checkbox, st.radio

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

### 2.6 st.file_uploader - 파일 업로드

```python
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터:")
    st.dataframe(df)
```

### 2.7 st.columns - 열 레이아웃

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

### 2.8 st.sidebar - 사이드바

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

### 2.9 st.expander - 접기/펼치기

```python
import streamlit as st

with st.expander("상세 설정 보기"):
    st.write("고급 옵션")
    n_estimators = st.number_input("트리 수", value=100)
    max_depth = st.number_input("최대 깊이", value=10)
```

### 2.10 st.tabs - 탭

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

### 2.11 st.form - 폼 (일괄 제출)

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

폼 내 위젯은 제출 버튼 클릭 시만 반영

### 2.12 상태 메시지

```python
import streamlit as st

st.success("성공 메시지")
st.info("정보 메시지")
st.warning("경고 메시지")
st.error("에러 메시지")
```

### 2.13 st.progress, st.spinner

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

## 대주제 3: 품질 예측 앱 만들기

### 3.1 앱 구조 설계

```
품질 예측 웹앱
+-- 사이드바: 설정
|   +-- 모델 선택
|   +-- 임계값 설정
+-- 메인: 입력/결과
|   +-- 센서 데이터 입력
|   +-- 예측 버튼
|   +-- 결과 표시
+-- 탭: 추가 기능
    +-- 데이터 시각화
    +-- 히스토리
```

### 3.2 기본 앱 템플릿

```python
import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="품질 예측 시스템",
    page_icon="[공장]",
    layout="wide"
)

# 타이틀
st.title("[공장] 제조 품질 예측 시스템")
st.markdown("센서 데이터를 입력하여 품질을 예측합니다.")
```

### 3.3 모델 로드

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

`@st.cache_resource`: 모델 한 번만 로드

### 3.4 센서 입력 섹션

```python
st.header("[차트] 센서 데이터 입력")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("온도 (도)", 100, 300, 200)
    pressure = st.slider("압력 (kPa)", 20, 100, 50)
    speed = st.slider("속도 (rpm)", 50, 200, 100)

with col2:
    humidity = st.slider("습도 (%)", 20, 80, 50)
    vibration = st.slider("진동 (mm/s)", 0.0, 15.0, 5.0)
```

### 3.5 예측 실행

```python
if st.button("[돋보기] 품질 예측", type="primary"):
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
        st.error(f"[경고] 불량 예측 (확률: {probability[0][1]:.1%})")
    else:
        st.success(f"[체크] 정상 예측 (확률: {probability[0][0]:.1%})")
```

### 3.6 결과 시각화

```python
import plotly.express as px

# 파이 차트
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

### 3.7 완성된 앱 코드 (요약)

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="품질 예측", layout="wide")
st.title("[공장] 품질 예측 시스템")

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

### 3.8 Streamlit Cloud 배포

1. GitHub에 코드 업로드
2. https://share.streamlit.io 접속
3. GitHub 저장소 연결
4. "Deploy" 클릭

```
필요 파일:
+-- app.py              # 메인 앱
+-- requirements.txt    # 의존성
+-- quality_pipeline.pkl # 모델 (또는 URL)
```

### 3.9 requirements.txt

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.5.0
plotly>=5.15.0
```

### 3.10 배포 시 주의사항

| 항목 | 주의점 |
|-----|--------|
| **모델 크기** | 큰 모델은 로드 시간 증가 |
| **메모리** | 무료 티어 제한 있음 |
| **보안** | API 키는 Secrets 사용 |
| **캐싱** | @st.cache_resource 활용 |

### 3.11 Secrets 관리

```python
# .streamlit/secrets.toml (로컬)
API_KEY = "your-api-key"

# 코드에서 접근
import streamlit as st
api_key = st.secrets["API_KEY"]
```

Streamlit Cloud에서 설정 가능

---

## 실습: Iris 분류기 앱

### 데이터셋 및 모델 로드 (캐싱)

```python
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_iris_model():
    """Iris 데이터셋 로드 및 모델 학습 (캐싱)"""
    # 데이터 로드
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 스케일러
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 정확도 계산
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'accuracy': accuracy
    }

iris_data = load_iris_model()
```

### 입력 폼 구성

```python
st.header("Iris 품종 분류기")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**꽃받침 (Sepal)**")
        sepal_length = st.slider("길이 (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("너비 (cm)", 2.0, 4.5, 3.0, 0.1)

    with col2:
        st.markdown("**꽃잎 (Petal)**")
        petal_length = st.slider("길이 (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("너비 (cm)", 0.1, 2.5, 1.2, 0.1)

    submitted = st.form_submit_button("[돋보기] 품종 예측", type="primary")
```

### 예측 수행

```python
if submitted:
    # 입력 데이터 준비
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # 스케일링 및 예측
    input_scaled = iris_data['scaler'].transform(input_data)
    prediction = iris_data['model'].predict(input_scaled)[0]
    probabilities = iris_data['model'].predict_proba(input_scaled)[0]

    predicted_species = iris_data['target_names'][prediction]
    max_prob = probabilities[prediction]

    # 결과 표시
    st.subheader("예측 결과")
    st.success(f"예측 품종: **{predicted_species}**")
    st.metric("예측 신뢰도", f"{max_prob:.1%}")

    # 품종별 확률 표시
    st.markdown("**품종별 확률:**")
    for i, species in enumerate(iris_data['target_names']):
        prob = probabilities[i]
        st.write(f"- {species}: {prob:.1%}")
```

### 세션 상태로 히스토리 관리

```python
# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state.history = []

# 예측 결과 저장
if submitted:
    st.session_state.history.append({
        '시각': datetime.now().strftime('%H:%M:%S'),
        'Sepal L': sepal_length,
        'Sepal W': sepal_width,
        'Petal L': petal_length,
        'Petal W': petal_width,
        '예측': predicted_species,
        '신뢰도': f"{max_prob:.1%}"
    })

# 히스토리 표시
if st.session_state.history:
    st.subheader("예측 히스토리")
    history_df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(history_df, use_container_width=True)
```

---

## 핵심 정리

### 1. Streamlit 기본
- 설치: `pip install streamlit`
- 실행: `streamlit run app.py`
- 기본 출력: `st.title`, `st.write`, `st.markdown`

### 2. 출력 위젯
```python
st.title("제목")
st.write("내용")
st.dataframe(df)
st.metric("정확도", "95.3%", "+2.1%")
st.line_chart(df)
```

### 3. 입력 위젯
```python
if st.button("실행"):
    ...
value = st.slider("라벨", min, max, default)
option = st.selectbox("선택", ["A", "B", "C"])
text = st.text_input("입력")
```

### 4. 레이아웃
```python
col1, col2 = st.columns(2)
with col1:
    st.write("왼쪽")
with col2:
    st.write("오른쪽")

st.sidebar.title("사이드바")
```

### 5. 상태와 캐싱
```python
# 세션 상태
st.session_state.key = value

# 모델 캐싱 (리소스)
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# 데이터 캐싱
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

### 핵심 코드

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

## 체크리스트

- [ ] Streamlit 설치 및 실행
- [ ] 기본 출력 위젯 사용
- [ ] 입력 위젯으로 인터랙션 구현
- [ ] 레이아웃 (columns, sidebar)
- [ ] 모델 로드 및 예측
- [ ] Streamlit Cloud 배포
