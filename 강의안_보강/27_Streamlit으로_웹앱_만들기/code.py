"""
[27차시] Streamlit으로 웹앱 만들기 - 실습 코드

학습 목표:
1. Streamlit 기본 위젯을 사용한다
2. 입력 위젯으로 사용자 인터랙션을 구현한다
3. 실제 ML 모델을 웹앱으로 배포한다

실습 환경: Python 3.8+, streamlit, sklearn

실행 방법:
    streamlit run code.py

또는 각 섹션별 파일로 저장하여 실행
"""

import pandas as pd
from datetime import datetime
import time

# Streamlit 설치 확인
try:
    import streamlit as st
    print(f"Streamlit 버전: {st.__version__}")
except ImportError:
    print("Streamlit 설치 필요: pip install streamlit")
    print("실행 방법: streamlit run code.py")
    exit(1)

# sklearn 설치 확인
try:
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    print("scikit-learn 로드 완료")
except ImportError:
    print("scikit-learn 설치 필요: pip install scikit-learn")
    exit(1)

# ============================================================
# 페이지 설정 (가장 먼저 호출해야 함)
# ============================================================
st.set_page_config(
    page_title="ML 예측 시스템",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 데이터셋 로드 및 모델 학습 (캐싱으로 성능 최적화)
# ============================================================
@st.cache_resource
def load_iris_model():
    """Iris 데이터셋 로드 및 모델 학습 (캐싱)"""
    try:
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
            'X': X,
            'y': y,
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

@st.cache_resource
def load_wine_model():
    """Wine 데이터셋 로드 및 모델 학습 (캐싱)"""
    try:
        # 데이터 로드
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = pd.Series(wine.target, name='target')

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
            'feature_names': wine.feature_names,
            'target_names': wine.target_names,
            'X': X,
            'y': y,
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

# 모델 로드
iris_data = load_iris_model()
wine_data = load_wine_model()

# ============================================================
# 1. 기본 출력 위젯
# ============================================================
st.title("🌸 [27차시] Streamlit으로 웹앱 만들기")
st.markdown("---")

# 탭으로 섹션 구분
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 기본 위젯",
    "🎛️ 입력 위젯",
    "📊 데이터/차트",
    "🌸 Iris 분류기",
    "📋 핵심 정리"
])

# ============================================================
# Tab 1: 기본 출력 위젯
# ============================================================
with tab1:
    st.header("1. 기본 출력 위젯")

    st.subheader("1.1 텍스트 계층 구조")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**코드:**")
        st.code("""
st.title("제목 (가장 큼)")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
st.caption("캡션 (작은 글씨)")
        """)

    with col2:
        st.markdown("**결과:**")
        st.markdown("### 제목 예시")
        st.markdown("#### 헤더 예시")
        st.markdown("##### 서브헤더 예시")
        st.text("일반 텍스트 예시")
        st.caption("캡션 예시")

    st.markdown("---")

    # st.write 만능 출력
    st.subheader("1.2 st.write - 만능 출력")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**문자열:**")
        st.write("Hello, Streamlit!")

        st.markdown("**숫자:**")
        st.write(42)

        st.markdown("**리스트:**")
        st.write([1, 2, 3, 4, 5])

    with col2:
        st.markdown("**딕셔너리:**")
        st.write({"species": "setosa", "petal_length": 1.4, "status": "정상"})

        st.markdown("**DataFrame (Iris 샘플):**")
        if iris_data:
            st.write(iris_data['X'].head(3))

    st.markdown("---")

    # 마크다운
    st.subheader("1.3 st.markdown")

    st.markdown("""
    마크다운 문법을 그대로 사용할 수 있습니다:

    - **굵은 글씨**
    - *기울임*
    - `코드`
    - [링크](https://streamlit.io)

    > 인용문도 가능합니다.
    """)

# ============================================================
# Tab 2: 입력 위젯
# ============================================================
with tab2:
    st.header("2. 입력 위젯")

    # 버튼
    st.subheader("2.1 st.button - 버튼")

    col1, col2 = st.columns(2)

    with col1:
        st.code("""
if st.button("분석 시작"):
    st.write("분석을 시작합니다...")
        """)

    with col2:
        if st.button("분석 시작 (예시)", key="btn1"):
            st.write("분석을 시작합니다...")
            st.success("완료!")

    st.markdown("---")

    # 슬라이더
    st.subheader("2.2 st.slider - 슬라이더")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider(
            "꽃받침 길이 (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.8,
            step=0.1,
            key="sepal_slider"
        )
        st.write(f"설정된 값: {sepal_length}cm")

    with col2:
        sepal_range = st.slider(
            "꽃받침 길이 범위",
            min_value=4.0,
            max_value=8.0,
            value=(4.5, 7.0),
            key="sepal_range"
        )
        st.write(f"범위: {sepal_range[0]} ~ {sepal_range[1]}cm")

    st.markdown("---")

    # 선택 위젯
    st.subheader("2.3 선택 위젯")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**selectbox - 드롭다운**")
        dataset = st.selectbox(
            "데이터셋",
            ["Iris", "Wine"],
            key="dataset_select"
        )
        st.write(f"선택: {dataset}")

    with col2:
        st.markdown("**multiselect - 다중 선택**")
        features = st.multiselect(
            "분석 특성",
            ["sepal length", "sepal width", "petal length", "petal width"],
            default=["sepal length"],
            key="feature_select"
        )
        st.write(f"선택: {features}")

    with col3:
        st.markdown("**radio - 라디오 버튼**")
        model = st.radio(
            "모델",
            ["RandomForest", "DecisionTree"],
            key="model_radio"
        )
        st.write(f"선택: {model}")

    st.markdown("---")

    # 텍스트/숫자 입력
    st.subheader("2.4 텍스트/숫자 입력")

    col1, col2 = st.columns(2)

    with col1:
        sample_name = st.text_input("샘플 이름", value="Sample_001", key="sample_name")
        st.write(f"입력: {sample_name}")

    with col2:
        threshold = st.number_input(
            "신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="threshold_input"
        )
        st.write(f"입력: {threshold}")

    st.markdown("---")

    # 체크박스
    st.subheader("2.5 체크박스")

    show_data = st.checkbox("Iris 데이터 표시", key="show_data_cb")
    if show_data and iris_data:
        st.dataframe(iris_data['X'].head(5))

# ============================================================
# Tab 3: 데이터와 차트
# ============================================================
with tab3:
    st.header("3. 데이터와 차트")

    if iris_data is None:
        st.error("데이터를 로드할 수 없습니다.")
    else:
        # DataFrame 표시
        st.subheader("3.1 DataFrame 표시 (Iris 데이터)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**st.dataframe (인터랙티브)**")
            st.dataframe(iris_data['X'].head(5), use_container_width=True)

        with col2:
            st.markdown("**st.table (정적)**")
            st.table(iris_data['X'].head(5))

        st.markdown("---")

        # Metric
        st.subheader("3.2 st.metric - KPI 표시")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("모델 정확도", f"{iris_data['accuracy']:.1%}", "+2.5%")
        col2.metric("총 샘플 수", len(iris_data['X']), "150")
        col3.metric("특성 수", len(iris_data['feature_names']), "4")
        col4.metric("클래스 수", len(iris_data['target_names']), "3")

        st.markdown("---")

        # 내장 차트
        st.subheader("3.3 내장 차트")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**st.line_chart - 특성 분포**")
            chart_data = iris_data['X'][['sepal length (cm)', 'petal length (cm)']].head(50)
            st.line_chart(chart_data)

        with col2:
            st.markdown("**st.bar_chart - 평균값**")
            mean_values = iris_data['X'].mean()
            st.bar_chart(mean_values)

        st.markdown("---")

        # Matplotlib
        st.subheader("3.4 Matplotlib 차트")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 산점도
        colors = ['red', 'green', 'blue']
        for i, species in enumerate(iris_data['target_names']):
            mask = iris_data['y'] == i
            axes[0].scatter(
                iris_data['X'].loc[mask, 'sepal length (cm)'],
                iris_data['X'].loc[mask, 'petal length (cm)'],
                c=colors[i],
                label=species,
                alpha=0.7
            )
        axes[0].set_xlabel('Sepal Length (cm)')
        axes[0].set_ylabel('Petal Length (cm)')
        axes[0].set_title('Iris: Sepal vs Petal Length')
        axes[0].legend()

        # 히스토그램
        axes[1].hist(iris_data['X']['sepal length (cm)'], bins=15, color='steelblue', edgecolor='white')
        axes[1].set_xlabel('Sepal Length (cm)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Sepal Length Distribution')

        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# Tab 4: Iris 분류기 앱
# ============================================================
with tab4:
    st.header("4. Iris 품종 분류기")

    if iris_data is None:
        st.error("모델을 로드할 수 없습니다.")
    else:
        st.markdown("""
        **Iris 데이터셋**을 사용한 실제 ML 분류 모델입니다.
        꽃받침(sepal)과 꽃잎(petal)의 크기를 입력하면 붓꽃 품종을 예측합니다.
        """)

        st.markdown("---")

        # 사이드바 설정
        st.sidebar.header("⚙️ 모델 설정")

        model_info = st.sidebar.selectbox(
            "모델 정보",
            ["Iris 분류기", "Wine 분류기"],
            key="model_info"
        )

        confidence_threshold = st.sidebar.slider(
            "신뢰도 임계값",
            0.0, 1.0, 0.5,
            key="conf_threshold"
        )

        st.sidebar.markdown("---")
        st.sidebar.info(f"모델 정확도: {iris_data['accuracy']:.1%}")

        # 메인 영역
        st.subheader("📊 꽃 특성 입력")

        # 폼 사용
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**꽃받침 (Sepal)**")
                input_sepal_length = st.slider(
                    "길이 (cm)",
                    min_value=4.0,
                    max_value=8.0,
                    value=5.8,
                    step=0.1,
                    key="input_sepal_length"
                )
                input_sepal_width = st.slider(
                    "너비 (cm)",
                    min_value=2.0,
                    max_value=4.5,
                    value=3.0,
                    step=0.1,
                    key="input_sepal_width"
                )

            with col2:
                st.markdown("**꽃잎 (Petal)**")
                input_petal_length = st.slider(
                    "길이 (cm)",
                    min_value=1.0,
                    max_value=7.0,
                    value=4.0,
                    step=0.1,
                    key="input_petal_length"
                )
                input_petal_width = st.slider(
                    "너비 (cm)",
                    min_value=0.1,
                    max_value=2.5,
                    value=1.2,
                    step=0.1,
                    key="input_petal_width"
                )

            submitted = st.form_submit_button("🔍 품종 예측", type="primary")

        if submitted:
            # 예측 수행
            with st.spinner("예측 중..."):
                time.sleep(0.3)  # UI 피드백용

                # 입력 데이터 준비
                input_data = [[
                    input_sepal_length,
                    input_sepal_width,
                    input_petal_length,
                    input_petal_width
                ]]

                # 스케일링 및 예측
                input_scaled = iris_data['scaler'].transform(input_data)
                prediction = iris_data['model'].predict(input_scaled)[0]
                probabilities = iris_data['model'].predict_proba(input_scaled)[0]

                predicted_species = iris_data['target_names'][prediction]
                max_prob = probabilities[prediction]

                # 결과 표시
                st.markdown("---")
                st.subheader("📋 예측 결과")

                col1, col2 = st.columns(2)

                with col1:
                    # 예측 결과
                    if max_prob >= confidence_threshold:
                        st.success(f"🌸 예측 품종: **{predicted_species}**")
                        st.metric("예측 신뢰도", f"{max_prob:.1%}")
                    else:
                        st.warning(f"⚠️ 신뢰도 낮음: {predicted_species} ({max_prob:.1%})")
                        st.caption(f"신뢰도가 임계값({confidence_threshold:.0%}) 미만입니다.")

                with col2:
                    # 각 품종별 확률
                    st.markdown("**품종별 확률:**")
                    for i, species in enumerate(iris_data['target_names']):
                        prob = probabilities[i]
                        bar_color = "🟢" if i == prediction else "⚪"
                        st.write(f"{bar_color} {species}: {prob:.1%}")

                # 입력 데이터 표시
                with st.expander("입력 데이터 상세"):
                    input_df = pd.DataFrame({
                        '특성': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        '입력값': [input_sepal_length, input_sepal_width, input_petal_length, input_petal_width],
                        '단위': ['cm', 'cm', 'cm', 'cm'],
                        '데이터 범위': ['4.3-7.9', '2.0-4.4', '1.0-6.9', '0.1-2.5']
                    })
                    st.dataframe(input_df, use_container_width=True)

        # 히스토리 섹션
        st.markdown("---")
        st.subheader("📜 예측 히스토리")

        # 세션 상태로 히스토리 관리
        if 'history' not in st.session_state:
            st.session_state.history = []

        if submitted:
            st.session_state.history.append({
                '시각': datetime.now().strftime('%H:%M:%S'),
                'Sepal L': input_sepal_length,
                'Sepal W': input_sepal_width,
                'Petal L': input_petal_length,
                'Petal W': input_petal_width,
                '예측': predicted_species,
                '신뢰도': f"{max_prob:.1%}"
            })

        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history[-10:])  # 최근 10개
            st.dataframe(history_df, use_container_width=True)

            if st.button("히스토리 초기화", key="clear_history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("예측 히스토리가 없습니다. 예측을 실행해보세요.")

        # 모델 성능 정보
        st.markdown("---")
        st.subheader("📈 모델 성능")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**데이터셋 정보:**")
            st.write(f"- 총 샘플 수: {len(iris_data['X'])}")
            st.write(f"- 특성 수: {len(iris_data['feature_names'])}")
            st.write(f"- 클래스: {', '.join(iris_data['target_names'])}")

        with col2:
            st.markdown("**모델 정보:**")
            st.write(f"- 알고리즘: Random Forest")
            st.write(f"- 테스트 정확도: {iris_data['accuracy']:.1%}")
            st.write(f"- 테스트 샘플 수: {len(iris_data['X_test'])}")

# ============================================================
# Tab 5: 핵심 정리
# ============================================================
with tab5:
    st.header("5. 핵심 정리")

    st.markdown("""
    ## [27차시 핵심 정리]

    ### 1. Streamlit 기본
    - **설치**: `pip install streamlit`
    - **실행**: `streamlit run app.py`
    - **기본 출력**: `st.title`, `st.write`, `st.markdown`

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

    ### 6. sklearn 데이터셋 활용
    ```python
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    target_names = iris.target_names
    ```

    ### 7. 배포
    1. GitHub에 코드 업로드
    2. share.streamlit.io 접속
    3. 저장소 연결 후 Deploy

    ### 필요 파일
    - `app.py` - 메인 앱
    - `requirements.txt` - 의존성
    - `model.pkl` - 모델 (선택)
    """)

    st.markdown("---")
    st.info("다음 차시 예고: FastAPI로 예측 서비스 만들기")

# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    [27차시] Streamlit으로 웹앱 만들기 | Iris 데이터셋 활용
</div>
""", unsafe_allow_html=True)
