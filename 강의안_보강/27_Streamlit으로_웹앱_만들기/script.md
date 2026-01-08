# [27차시] Streamlit으로 웹앱 만들기 - 강사 스크립트

## 수업 개요

| 항목 | 내용 |
|------|------|
| 차시 | 27차시 |
| 주제 | Streamlit으로 웹앱 만들기 |
| 시간 | 30분 (이론 15분 + 실습 13분 + 정리 2분) |
| 학습 목표 | Streamlit 기본 위젯, 입력 위젯, 품질 예측 웹앱 |

---

## 학습 목표

1. Streamlit 기본 위젯을 사용한다
2. 입력 위젯으로 사용자 인터랙션을 구현한다
3. 품질 예측 웹앱을 만들고 배포한다

---

## 시간 배분

| 구간 | 시간 | 내용 |
|------|------|------|
| 도입 | 2분 | 복습 및 학습목표 |
| 대주제 1 | 5분 | Streamlit 소개와 기본 위젯 |
| 대주제 2 | 5분 | 입력 위젯과 레이아웃 |
| 대주제 3 | 5분 | 품질 예측 앱 만들기 |
| 실습 | 11분 | 웹앱 개발 실습 |
| 정리 | 2분 | 요약 및 다음 차시 예고 |

---

## 상세 스크립트

### 도입부 (2분)

#### 슬라이드 1-3: 복습

> "지난 시간에 LLM API와 프롬프트 엔지니어링을 배웠습니다. 이제 이런 기능들을 사용자가 쉽게 쓸 수 있게 웹앱으로 만들어봅시다."

> "오늘 배울 Streamlit은 Python만으로 웹앱을 만들 수 있는 도구입니다. HTML이나 JavaScript를 몰라도 됩니다."

> "수업이 끝나면 품질 예측 모델을 웹에서 바로 사용할 수 있게 됩니다."

---

### 대주제 1: Streamlit 소개와 기본 위젯 (5분)

#### 슬라이드 4-6: Streamlit이란

> "Streamlit은 Python 코드만으로 웹앱을 만드는 프레임워크입니다."

```python
import streamlit as st

st.title("품질 예측 시스템")
st.write("센서 데이터를 입력하세요.")
```

> "이 3줄이면 웹 페이지가 만들어집니다. st.title로 제목, st.write로 내용을 표시해요."

> "설치는 pip install streamlit, 실행은 streamlit run app.py입니다."

---

#### 슬라이드 7-9: 기본 출력 위젯

> "텍스트 계층 구조가 있어요. st.title이 가장 크고, header, subheader 순입니다."

> "st.write는 만능입니다. 문자열, 숫자, 리스트, 딕셔너리, DataFrame까지 알아서 예쁘게 표시해줘요."

```python
st.write("문자열")
st.write({"key": "value"})
st.write(df)  # DataFrame도 자동 렌더링
```

---

#### 슬라이드 10-12: 데이터와 차트

> "st.dataframe은 인터랙티브 테이블입니다. 정렬, 필터가 가능해요."

> "st.metric은 KPI 표시에 좋습니다. 값과 변화량을 함께 보여주고, 증가/감소에 따라 색상이 바뀌어요."

```python
st.metric("불량률", "2.3%", "-0.2%")  # 녹색 화살표 (감소는 좋은 것)
```

> "차트도 간단해요. st.line_chart, st.bar_chart는 데이터만 주면 바로 그려줍니다."

---

### 대주제 2: 입력 위젯과 레이아웃 (5분)

#### 슬라이드 13-15: 입력 위젯

> "사용자로부터 입력을 받는 위젯들을 봅시다."

> "st.button은 버튼입니다. 클릭하면 True를 반환해서 if문으로 처리할 수 있어요."

```python
if st.button("분석 시작"):
    st.write("분석을 시작합니다...")
```

> "st.slider는 슬라이더, st.selectbox는 드롭다운입니다. 제조 데이터 입력에 딱이에요."

---

#### 슬라이드 16-18: 레이아웃

> "레이아웃으로 화면을 구성할 수 있습니다."

> "st.columns로 열을 나누고, st.sidebar로 사이드바를 만들어요."

```python
col1, col2 = st.columns(2)
with col1:
    temp = st.slider("온도", 100, 300, 200)
with col2:
    pres = st.slider("압력", 20, 100, 50)
```

> "st.form은 여러 입력을 모아서 한 번에 제출할 수 있게 해줍니다."

---

#### 슬라이드 19-21: 상태 메시지

> "결과 표시에는 상태 메시지를 사용합니다."

```python
st.success("성공!")    # 녹색
st.warning("주의!")    # 노란색
st.error("에러!")      # 빨간색
```

> "st.spinner로 로딩 표시, st.progress로 진행률을 보여줄 수 있어요."

---

### 대주제 3: 품질 예측 앱 만들기 (5분)

#### 슬라이드 22-24: 앱 구조

> "이제 품질 예측 앱을 만들어봅시다. 구조는 사이드바에 설정, 메인에 입력과 결과입니다."

> "첫 번째로 페이지 설정을 합니다."

```python
st.set_page_config(
    page_title="품질 예측 시스템",
    page_icon="🏭",
    layout="wide"
)
```

---

#### 슬라이드 25-27: 모델 로드

> "모델은 캐싱해서 로드합니다. @st.cache_resource를 쓰면 한 번만 로드해요."

```python
@st.cache_resource
def load_model():
    return joblib.load('quality_pipeline.pkl')

model = load_model()
```

> "모델이 클 때 특히 중요합니다. 매번 로드하면 느려지거든요."

---

#### 슬라이드 28-30: 예측과 결과

> "센서 입력을 받고 예측 버튼을 누르면 결과를 표시합니다."

```python
if st.button("예측"):
    input_df = pd.DataFrame([[temp, pres, speed, humid, vib]])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ 불량 예측")
    else:
        st.success("✅ 정상 예측")
```

---

### 실습편 (11분)

#### 슬라이드 31-33: 환경 설정

```python
# 필요 패키지 설치
pip install streamlit pandas numpy scikit-learn joblib matplotlib

# 앱 파일 생성
# app.py
```

---

#### 슬라이드 34-36: 기본 앱 작성

```python
import streamlit as st
import pandas as pd

st.title("🏭 품질 예측 시스템")
st.markdown("센서 데이터를 입력하여 품질을 예측합니다.")

# 센서 입력
st.header("📊 센서 데이터")
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("온도 (도)", 100, 300, 200)
    pressure = st.slider("압력 (kPa)", 20, 100, 50)

with col2:
    speed = st.slider("속도 (rpm)", 50, 200, 100)
    vibration = st.slider("진동 (mm/s)", 0.0, 15.0, 5.0)
```

---

#### 슬라이드 37-39: 예측 기능 추가

```python
# 예측 버튼
if st.button("🔍 품질 예측", type="primary"):
    # 입력 데이터 구성
    input_data = {
        'temperature': temperature,
        'pressure': pressure,
        'speed': speed,
        'vibration': vibration
    }

    # 결과 표시 (시뮬레이션)
    st.subheader("예측 결과")
    if temperature > 250 or vibration > 10:
        st.error("⚠️ 불량 가능성 높음")
    else:
        st.success("✅ 정상 예측")

    st.dataframe(pd.DataFrame([input_data]))
```

---

#### 슬라이드 40-42: 앱 실행

```bash
# 터미널에서 실행
streamlit run app.py

# 브라우저에서 확인
# http://localhost:8501
```

---

### 정리 (2분)

#### 슬라이드 43-44: 핵심 정리

> "오늘 배운 내용을 정리합니다."

> "**Streamlit**은 Python만으로 웹앱을 만드는 도구입니다. st.title, st.write, st.dataframe으로 출력하고, st.button, st.slider로 입력받습니다."

> "**레이아웃**은 st.columns, st.sidebar, st.form으로 구성합니다."

> "**모델 로드**는 @st.cache_resource로 캐싱하고, 예측 결과는 st.success, st.error로 표시합니다."

---

#### 슬라이드 45-46: 다음 차시 예고

> "다음 시간에는 FastAPI로 REST API 서버를 만듭니다. Streamlit이 사용자 UI라면, FastAPI는 다른 프로그램이 호출할 수 있는 API입니다."

> "오늘 수업 마무리합니다. 수고하셨습니다!"

---

## 예상 질문 및 답변

### Q1: Streamlit과 Flask/Django 차이가 뭔가요?

> "Streamlit은 데이터 앱에 특화되어 있고 훨씬 간단합니다. Flask/Django는 범용 웹 개발용이에요. ML 모델 데모나 대시보드에는 Streamlit이 빠릅니다."

### Q2: 무료로 배포할 수 있나요?

> "네, Streamlit Cloud에서 무료로 배포할 수 있어요. GitHub 저장소 연결하면 바로 배포됩니다."

### Q3: 여러 페이지를 만들 수 있나요?

> "네, pages 폴더에 파일을 추가하면 자동으로 다중 페이지가 됩니다. 사이드바에 네비게이션이 생겨요."

### Q4: 사용자별 세션 관리가 되나요?

> "네, st.session_state를 사용하면 사용자별 상태를 관리할 수 있습니다."

---

## 참고 자료

### 공식 문서
- [Streamlit 문서](https://docs.streamlit.io/)
- [Streamlit Cloud](https://share.streamlit.io/)

### 관련 차시
- 25차시: LLM API와 프롬프트 작성법
- 27차시: FastAPI로 예측 서비스 만들기

---

## 체크리스트

수업 전:
- [ ] Streamlit 설치 확인
- [ ] 예제 모델 파일 준비
- [ ] 브라우저 실행 환경 확인

수업 중:
- [ ] 기본 위젯 시연
- [ ] 입력 위젯 인터랙션 강조
- [ ] 실시간 앱 개발 시연

수업 후:
- [ ] 완성된 앱 코드 배포
- [ ] Streamlit Cloud 배포 가이드 공유

