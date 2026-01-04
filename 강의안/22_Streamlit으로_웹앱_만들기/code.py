# [22차시] Streamlit으로 웹앱 만들기 - 실습 코드
# 실행: streamlit run code.py

# Streamlit 설치 확인
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("=" * 60)
    print("Streamlit이 설치되지 않았습니다.")
    print("=" * 60)
    print()
    print("설치 방법:")
    print("  pip install streamlit")
    print()
    print("실행 방법:")
    print("  streamlit run code.py")
    print()

import numpy as np
import pandas as pd

if STREAMLIT_AVAILABLE:
    # ============================================================
    # 페이지 설정 (항상 맨 위에!)
    # ============================================================
    st.set_page_config(
        page_title="품질 예측 시스템",
        page_icon="🏭",
        layout="wide"
    )

    # ============================================================
    # 제목 및 소개
    # ============================================================
    st.title("🏭 제조 품질 예측 시스템")
    st.write("센서 데이터를 입력하면 품질을 예측합니다.")
    st.markdown("---")

    # ============================================================
    # 사이드바 - 입력 파라미터
    # ============================================================
    st.sidebar.header("📊 센서 입력")

    temperature = st.sidebar.slider(
        "온도 (°C)",
        min_value=70,
        max_value=100,
        value=85,
        step=1,
        help="정상 범위: 70-90°C"
    )

    humidity = st.sidebar.slider(
        "습도 (%)",
        min_value=30,
        max_value=70,
        value=50,
        step=1,
        help="정상 범위: 30-60%"
    )

    speed = st.sidebar.slider(
        "속도 (RPM)",
        min_value=80,
        max_value=120,
        value=100,
        step=1,
        help="정상 범위: 80-110 RPM"
    )

    # 라인 선택
    line = st.sidebar.selectbox(
        "생산 라인",
        ["A라인", "B라인", "C라인"]
    )

    # ============================================================
    # 예측 함수
    # ============================================================
    def predict_quality(temp, humidity, speed):
        """간단한 규칙 기반 예측 (실제로는 ML 모델 사용)"""
        score = 0

        # 온도 체크
        if temp > 90:
            score += 30
        elif temp > 85:
            score += 10

        # 습도 체크
        if humidity > 60:
            score += 20
        elif humidity > 55:
            score += 10

        # 속도 체크
        if speed > 110:
            score += 15
        elif speed > 105:
            score += 5

        probability = min(score / 100, 1.0)
        return probability

    # 예측 실행
    defect_prob = predict_quality(temperature, humidity, speed)

    # ============================================================
    # 메인 화면 - 결과 표시
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 입력 데이터")

        input_df = pd.DataFrame({
            "항목": ["온도", "습도", "속도", "라인"],
            "현재값": [f"{temperature}°C", f"{humidity}%", f"{speed}RPM", line],
            "정상범위": ["70-90°C", "30-60%", "80-110RPM", "-"]
        })

        st.dataframe(input_df, use_container_width=True, hide_index=True)

        # 각 항목 상태 표시
        st.write("**항목별 상태:**")
        if temperature > 90:
            st.warning(f"온도 {temperature}°C - 정상 범위 초과")
        else:
            st.success(f"온도 {temperature}°C - 정상")

        if humidity > 60:
            st.warning(f"습도 {humidity}% - 정상 범위 초과")
        else:
            st.success(f"습도 {humidity}% - 정상")

        if speed > 110:
            st.warning(f"속도 {speed}RPM - 정상 범위 초과")
        else:
            st.success(f"속도 {speed}RPM - 정상")

    with col2:
        st.subheader("📤 예측 결과")

        # 결과 메시지
        if defect_prob > 0.3:
            st.error(f"⚠️ 불량 위험! 확률: {defect_prob:.1%}")
        elif defect_prob > 0.1:
            st.warning(f"⚡ 주의 필요. 확률: {defect_prob:.1%}")
        else:
            st.success(f"✅ 정상 상태. 불량 확률: {defect_prob:.1%}")

        # 메트릭 표시
        st.metric(
            label="불량 확률",
            value=f"{defect_prob:.1%}",
            delta=f"{(defect_prob - 0.1):.1%}" if defect_prob > 0.1 else None,
            delta_color="inverse"
        )

        # 진행 바
        st.progress(defect_prob)

    # ============================================================
    # 시각화
    # ============================================================
    st.markdown("---")
    st.subheader("📈 센서 현황 시각화")

    col3, col4 = st.columns(2)

    with col3:
        # 막대 차트 (정규화된 값)
        chart_data = pd.DataFrame({
            "항목": ["온도", "습도", "속도"],
            "정규화값": [
                (temperature - 70) / 30 * 100,
                (humidity - 30) / 40 * 100,
                (speed - 80) / 40 * 100
            ]
        })
        st.bar_chart(chart_data.set_index("항목"))

    with col4:
        # 시계열 시뮬레이션
        np.random.seed(42)
        time_data = pd.DataFrame({
            "시간": pd.date_range("09:00", periods=10, freq="10min"),
            "온도": [temperature + np.random.randint(-3, 4) for _ in range(10)],
            "습도": [humidity + np.random.randint(-5, 6) for _ in range(10)]
        })
        st.line_chart(time_data.set_index("시간"))

    # ============================================================
    # 파일 업로드
    # ============================================================
    st.markdown("---")
    st.subheader("📁 배치 데이터 분석")

    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("**업로드된 데이터 (상위 5행):**")
        st.dataframe(df.head())

        st.write("**기초 통계:**")
        st.write(df.describe())

        if st.button("배치 예측 실행"):
            # 배치 예측 시뮬레이션
            with st.spinner("예측 중..."):
                import time
                time.sleep(1)

                # 결과 추가 (시뮬레이션)
                if 'temperature' in df.columns and 'humidity' in df.columns:
                    df['예측'] = df.apply(
                        lambda row: '불량 위험' if row.get('temperature', 85) > 90 else '정상',
                        axis=1
                    )
                else:
                    df['예측'] = np.random.choice(['정상', '불량 위험'], len(df), p=[0.85, 0.15])

                st.success("예측 완료!")
                st.dataframe(df)

                # 다운로드 버튼
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="결과 CSV 다운로드",
                    data=csv,
                    file_name="prediction_result.csv",
                    mime="text/csv"
                )

    # ============================================================
    # 세션 상태 - 예측 기록
    # ============================================================
    st.markdown("---")
    st.subheader("📋 예측 기록")

    # 세션 상태 초기화
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 기록 저장 버튼
    if st.button("현재 예측 저장"):
        record = {
            "시간": pd.Timestamp.now().strftime("%H:%M:%S"),
            "온도": temperature,
            "습도": humidity,
            "속도": speed,
            "라인": line,
            "불량확률": f"{defect_prob:.1%}"
        }
        st.session_state.history.append(record)
        st.success("예측이 저장되었습니다!")

    # 기록 표시
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # 기록 삭제 버튼
        if st.button("기록 삭제"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.info("저장된 예측 기록이 없습니다.")

    # ============================================================
    # 푸터
    # ============================================================
    st.markdown("---")
    st.markdown("*제조 AI 과정 | 22차시 Streamlit 실습*")

else:
    # Streamlit 없이 실행할 경우 예제 코드 출력
    print("""
============================================================
[22차시] Streamlit 웹앱 만들기 - 주요 코드 예시
============================================================

# 1. 기본 앱 구조
import streamlit as st

st.set_page_config(page_title='품질 예측', page_icon='🏭')
st.title('🏭 제조 품질 예측 시스템')

# 2. 입력 위젯
temperature = st.slider('온도 (°C)', 70, 100, 85)
humidity = st.slider('습도 (%)', 30, 70, 50)
line = st.selectbox('라인', ['A라인', 'B라인', 'C라인'])

# 3. 사이드바
st.sidebar.header('설정')
option = st.sidebar.selectbox('옵션', ['A', 'B', 'C'])

# 4. 컬럼 레이아웃
col1, col2 = st.columns(2)
with col1:
    st.write('왼쪽')
with col2:
    st.write('오른쪽')

# 5. 결과 표시
st.success('정상입니다!')
st.error('불량 위험!')
st.metric('확률', '15%')
st.progress(0.15)

# 6. 캐싱
@st.cache_resource
def load_model():
    import joblib
    return joblib.load('model.pkl')

# 7. 세션 상태
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button('클릭'):
    st.session_state.count += 1

st.write(f'클릭: {st.session_state.count}')

# 8. 파일 업로드
uploaded = st.file_uploader('CSV 업로드', type='csv')
if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    st.dataframe(df)

============================================================
실행 방법:
1. pip install streamlit
2. streamlit run code.py
3. 브라우저에서 http://localhost:8501 접속
============================================================
""")
