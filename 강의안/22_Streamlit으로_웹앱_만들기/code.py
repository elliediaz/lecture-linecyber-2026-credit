"""
[22차시] Streamlit으로 웹앱 만들기 - 실습 코드
학습목표: Streamlit 기본 사용법, 대화형 위젯, ML 모델 예측 웹앱

이 파일은 Streamlit 앱 예제입니다.
실행: streamlit run code.py
"""

# Streamlit 설치 확인
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Streamlit이 설치되지 않았습니다.")
    print("   설치: pip install streamlit")
    print("   실행: streamlit run code.py")

import numpy as np
import pandas as pd

if STREAMLIT_AVAILABLE:
    # ============================================================
    # 페이지 설정
    # ============================================================
    st.set_page_config(
        page_title="품질 예측 시스템",
        page_icon="🏭",
        layout="wide"
    )

    # ============================================================
    # 제목
    # ============================================================
    st.title("🏭 제조 품질 예측 시스템")
    st.markdown("---")

    # ============================================================
    # 사이드바 - 입력 파라미터
    # ============================================================
    st.sidebar.header("📊 입력 파라미터")

    temperature = st.sidebar.slider(
        "온도 (°C)",
        min_value=70,
        max_value=100,
        value=85,
        step=1
    )

    humidity = st.sidebar.slider(
        "습도 (%)",
        min_value=30,
        max_value=70,
        value=50,
        step=1
    )

    speed = st.sidebar.slider(
        "속도 (rpm)",
        min_value=80,
        max_value=120,
        value=100,
        step=1
    )

    pressure = st.sidebar.slider(
        "압력 (bar)",
        min_value=0.8,
        max_value=1.2,
        value=1.0,
        step=0.05
    )

    # ============================================================
    # 메인 화면 - 결과 표시
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 입력값")
        input_df = pd.DataFrame({
            "파라미터": ["온도", "습도", "속도", "압력"],
            "값": [f"{temperature}°C", f"{humidity}%", f"{speed}rpm", f"{pressure}bar"]
        })
        st.dataframe(input_df, hide_index=True)

    with col2:
        st.subheader("📤 예측 결과")

        # 간단한 규칙 기반 예측 (실제로는 model.predict 사용)
        defect_prob = 0.1 + 0.02 * (temperature - 85) + 0.01 * (humidity - 50)
        defect_prob = max(0, min(1, defect_prob))

        if defect_prob > 0.3:
            prediction = "🔴 불량 위험"
            st.error(prediction)
        else:
            prediction = "🟢 정상"
            st.success(prediction)

        st.metric("불량 확률", f"{defect_prob:.1%}")
        st.progress(defect_prob)

    # ============================================================
    # 시각화
    # ============================================================
    st.markdown("---")
    st.subheader("📈 파라미터 분석")

    # 막대 차트
    chart_data = pd.DataFrame({
        "파라미터": ["온도", "습도", "속도", "압력"],
        "정규화 값": [
            (temperature - 70) / 30,
            (humidity - 30) / 40,
            (speed - 80) / 40,
            (pressure - 0.8) / 0.4
        ]
    })
    st.bar_chart(chart_data.set_index("파라미터"))

    # ============================================================
    # 배치 예측
    # ============================================================
    st.markdown("---")
    st.subheader("📁 배치 예측")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("업로드된 데이터:")
        st.dataframe(df.head())

        if st.button("배치 예측 실행"):
            st.write("예측 결과:")
            # 예측 로직 (실제로는 model.predict 사용)
            df["예측"] = np.random.choice(["정상", "불량"], len(df), p=[0.85, 0.15])
            st.dataframe(df)

    # ============================================================
    # 푸터
    # ============================================================
    st.markdown("---")
    st.markdown("*AI 기초체력훈련 | 22차시 Streamlit 실습*")

else:
    # Streamlit 없이 실행할 경우
    print("""
    ============================================================
    [22차시] Streamlit 웹앱 예제
    ============================================================

    이 파일은 Streamlit 앱입니다.

    실행 방법:
    1. Streamlit 설치: pip install streamlit
    2. 앱 실행: streamlit run code.py
    3. 브라우저에서 http://localhost:8501 접속

    주요 코드:

    import streamlit as st

    # 제목
    st.title('품질 예측 시스템')

    # 슬라이더 입력
    temp = st.slider('온도', 70, 100, 85)
    humidity = st.slider('습도', 30, 70, 50)

    # 예측 버튼
    if st.button('예측'):
        # 예측 수행
        result = model.predict([[temp, humidity]])[0]
        st.success(f'결과: {"불량" if result == 1 else "정상"}')

    # 사이드바
    st.sidebar.title('설정')
    option = st.sidebar.selectbox('옵션', ['A', 'B', 'C'])

    # 캐싱
    @st.cache_resource
    def load_model():
        return joblib.load('model.pkl')

    ============================================================
    """)
