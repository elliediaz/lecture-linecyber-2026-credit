"""
========================================
4차시: 공개 데이터셋 확보 및 데이터 생태계 이해
========================================
실습 코드

학습내용:
1. 공공데이터의 정의와 활용
2. 주요 데이터 포털의 특성
3. 데이터셋 구조와 용어
4. 실제 공개 데이터셋 로드 및 활용

사용 데이터셋:
- sklearn: Iris, Wine, Diabetes 등
- seaborn: tips, penguins, diamonds, titanic 등
- UCI ML Repository: Wine Quality 등
"""

# =====================================================
# 실습 환경 준비
# =====================================================

import pandas as pd
import numpy as np
import os
import json

print("=" * 60)
print("4차시: 공개 데이터셋 확보 및 데이터 생태계 이해")
print("=" * 60)


# =====================================================
# 실습 1: sklearn 내장 데이터셋 로드
# =====================================================
print("\n" + "=" * 60)
print("실습 1: sklearn 내장 데이터셋 로드")
print("=" * 60)

# 1-1. Iris 데이터셋 (붓꽃 분류)
print("\n[1-1. Iris 데이터셋 - 붓꽃 분류]")
try:
    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    print(f"데이터셋 설명: {iris.DESCR[:200]}...")
    print(f"\n데이터 크기: {iris_df.shape}")
    print(f"특성 이름: {iris.feature_names}")
    print(f"타겟 클래스: {list(iris.target_names)}")
    print(f"\n처음 5행:")
    print(iris_df.head())
    print(f"\n클래스별 개수:")
    print(iris_df['species'].value_counts())
except ImportError:
    print("sklearn이 설치되어 있지 않습니다. pip install scikit-learn")

# 1-2. Wine 데이터셋 (와인 품질 분류)
print("\n[1-2. Wine 데이터셋 - 와인 품질 분류]")
try:
    from sklearn.datasets import load_wine

    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target

    print(f"데이터 크기: {wine_df.shape}")
    print(f"특성 개수: {len(wine.feature_names)}")
    print(f"타겟 클래스: {list(wine.target_names)}")
    print(f"\n처음 5행:")
    print(wine_df.head())
except ImportError:
    print("sklearn이 설치되어 있지 않습니다.")

# 1-3. Diabetes 데이터셋 (당뇨병 진행 예측 - 회귀)
print("\n[1-3. Diabetes 데이터셋 - 당뇨병 진행 예측 (회귀)]")
try:
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target

    print(f"데이터 크기: {diabetes_df.shape}")
    print(f"특성 이름: {diabetes.feature_names}")
    print(f"타겟 변수: 1년 후 질병 진행도 (연속형)")
    print(f"\n처음 5행:")
    print(diabetes_df.head())
    print(f"\n타겟 변수 통계:")
    print(diabetes_df['target'].describe())
except ImportError:
    print("sklearn이 설치되어 있지 않습니다.")

# 1-4. Breast Cancer 데이터셋 (유방암 진단 - 이진 분류)
print("\n[1-4. Breast Cancer 데이터셋 - 유방암 진단 (이진 분류)]")
try:
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target

    print(f"데이터 크기: {cancer_df.shape}")
    print(f"특성 개수: {len(cancer.feature_names)}")
    print(f"타겟 클래스: {list(cancer.target_names)}")
    print(f"\n타겟 분포 (0=malignant, 1=benign):")
    print(cancer_df['target'].value_counts())
except ImportError:
    print("sklearn이 설치되어 있지 않습니다.")


# =====================================================
# 실습 2: seaborn 내장 데이터셋 로드
# =====================================================
print("\n" + "=" * 60)
print("실습 2: seaborn 내장 데이터셋 로드")
print("=" * 60)

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("seaborn이 설치되어 있지 않습니다. pip install seaborn")

if SEABORN_AVAILABLE:
    # 2-1. Tips 데이터셋 (레스토랑 팁)
    print("\n[2-1. Tips 데이터셋 - 레스토랑 팁 데이터]")
    tips = sns.load_dataset('tips')
    print(f"데이터 크기: {tips.shape}")
    print(f"변수: {list(tips.columns)}")
    print(f"\n처음 5행:")
    print(tips.head())

    # 2-2. Penguins 데이터셋 (펭귄 측정)
    print("\n[2-2. Penguins 데이터셋 - 펭귄 측정 데이터]")
    penguins = sns.load_dataset('penguins')
    print(f"데이터 크기: {penguins.shape}")
    print(f"변수: {list(penguins.columns)}")
    print(f"\n처음 5행:")
    print(penguins.head())
    print(f"\n종별 개수:")
    print(penguins['species'].value_counts())

    # 2-3. Diamonds 데이터셋 (다이아몬드 가격)
    print("\n[2-3. Diamonds 데이터셋 - 다이아몬드 가격]")
    diamonds = sns.load_dataset('diamonds')
    print(f"데이터 크기: {diamonds.shape}")
    print(f"변수: {list(diamonds.columns)}")
    print(f"\n처음 5행:")
    print(diamonds.head())
    print(f"\n가격 통계:")
    print(diamonds['price'].describe())

    # 2-4. Titanic 데이터셋 (타이타닉 생존자)
    print("\n[2-4. Titanic 데이터셋 - 타이타닉 생존자]")
    titanic = sns.load_dataset('titanic')
    print(f"데이터 크기: {titanic.shape}")
    print(f"변수: {list(titanic.columns)}")
    print(f"\n처음 5행:")
    print(titanic.head())
    print(f"\n생존 여부:")
    print(titanic['survived'].value_counts())

    # 2-5. 사용 가능한 모든 데이터셋 목록
    print("\n[2-5. seaborn에서 사용 가능한 모든 데이터셋]")
    print("sns.get_dataset_names():")
    print(sns.get_dataset_names())


# =====================================================
# 실습 3: UCI ML Repository 데이터 로드 (URL 직접 접근)
# =====================================================
print("\n" + "=" * 60)
print("실습 3: UCI ML Repository 데이터 로드")
print("=" * 60)

# 3-1. Wine Quality 데이터셋 (레드 와인)
print("\n[3-1. Wine Quality 데이터셋 - UCI Repository]")
print("URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/")

# UCI Wine Quality 데이터셋 직접 로드
WINE_QUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    wine_quality = pd.read_csv(WINE_QUALITY_URL, sep=';')
    print(f"\n레드 와인 품질 데이터 로드 성공!")
    print(f"데이터 크기: {wine_quality.shape}")
    print(f"변수: {list(wine_quality.columns)}")
    print(f"\n처음 5행:")
    print(wine_quality.head())
    print(f"\n품질 등급 분포:")
    print(wine_quality['quality'].value_counts().sort_index())
    print(f"\n기술통계:")
    print(wine_quality.describe().round(2))
except Exception as e:
    print(f"데이터 로드 실패 (인터넷 연결 확인): {e}")
    # 오프라인용 샘플 데이터
    wine_quality = pd.DataFrame({
        'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4],
        'volatile acidity': [0.70, 0.88, 0.76, 0.28, 0.70],
        'citric acid': [0.00, 0.00, 0.04, 0.56, 0.00],
        'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9],
        'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076],
        'free sulfur dioxide': [11, 25, 15, 17, 11],
        'total sulfur dioxide': [34, 67, 54, 60, 34],
        'density': [0.9978, 0.9968, 0.9970, 0.9980, 0.9978],
        'pH': [3.51, 3.20, 3.26, 3.16, 3.51],
        'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56],
        'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4],
        'quality': [5, 5, 5, 6, 5]
    })
    print("\n[오프라인 모드] 샘플 데이터 사용")
    print(wine_quality)

# 3-2. Iris 데이터셋 (UCI 원본)
print("\n[3-2. Iris 데이터셋 - UCI Repository 원본]")
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

try:
    iris_uci = pd.read_csv(
        IRIS_URL,
        header=None,
        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    )
    print(f"UCI Iris 데이터 로드 성공!")
    print(f"데이터 크기: {iris_uci.shape}")
    print(f"\n처음 5행:")
    print(iris_uci.head())
except Exception as e:
    print(f"데이터 로드 실패: {e}")


# =====================================================
# 실습 4: 다양한 포맷의 데이터 로드
# =====================================================
print("\n" + "=" * 60)
print("실습 4: 다양한 포맷의 데이터 로드 방법")
print("=" * 60)

# 4-1. CSV 로드 방법들
csv_examples = """
# 4-1. CSV 파일 로드 다양한 방법

# 기본 로드
df = pd.read_csv('data.csv')

# 인코딩 지정 (한글 파일)
df = pd.read_csv('data.csv', encoding='cp949')  # 윈도우 한글
df = pd.read_csv('data.csv', encoding='utf-8')  # UTF-8

# 구분자 지정
df = pd.read_csv('data.csv', sep=';')           # 세미콜론 구분
df = pd.read_csv('data.tsv', sep='\\t')         # 탭 구분

# 특정 열만 로드
df = pd.read_csv('data.csv', usecols=['col1', 'col2'])

# 데이터 타입 지정
df = pd.read_csv('data.csv', dtype={'col1': str, 'col2': float})

# 일부 행만 로드 (대용량 데이터)
df = pd.read_csv('data.csv', nrows=1000)

# 헤더가 없는 경우
df = pd.read_csv('data.csv', header=None, names=['A', 'B', 'C'])

# URL에서 직접 로드
df = pd.read_csv('https://example.com/data.csv')
"""
print(csv_examples)

# 4-2. Excel 로드
print("\n[4-2. Excel 파일 로드]")
excel_examples = """
# Excel 파일 로드
df = pd.read_excel('data.xlsx')

# 특정 시트 로드
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 여러 시트 로드
sheets = pd.read_excel('data.xlsx', sheet_name=None)  # 딕셔너리 반환
"""
print(excel_examples)

# 4-3. JSON 로드
print("\n[4-3. JSON 파일 로드]")
json_examples = """
# JSON 파일 로드
df = pd.read_json('data.json')

# 중첩된 JSON 평탄화
import json
with open('nested.json', 'r') as f:
    data = json.load(f)
df = pd.json_normalize(data)
"""
print(json_examples)


# =====================================================
# 실습 5: sklearn 데이터셋 상세 분석
# =====================================================
print("\n" + "=" * 60)
print("실습 5: sklearn 데이터셋 상세 분석")
print("=" * 60)

if 'iris_df' in dir():
    print("\n[Iris 데이터셋 상세 분석]")

    # 5-1. 데이터 구조
    print("\n5-1. 데이터 구조")
    print(f"전체 샘플 수: {len(iris_df)}")
    print(f"특성 수: {len(iris.feature_names)}")
    print(f"클래스 수: {len(iris.target_names)}")

    # 5-2. 특성별 통계
    print("\n5-2. 특성별 기술통계")
    print(iris_df.describe().round(2))

    # 5-3. 클래스별 평균
    print("\n5-3. 클래스별 평균")
    print(iris_df.groupby('species')[iris.feature_names].mean().round(2))

    # 5-4. 상관관계
    print("\n5-4. 특성 간 상관관계")
    print(iris_df[iris.feature_names].corr().round(2))


# =====================================================
# 실습 6: 데이터 구조 확인 함수
# =====================================================
print("\n" + "=" * 60)
print("실습 6: 데이터 구조 확인 함수")
print("=" * 60)

def check_dataset(df, name="Dataset"):
    """
    데이터셋 구조를 종합적으로 확인하는 함수

    Parameters:
    -----------
    df : pandas.DataFrame
        확인할 데이터프레임
    name : str
        데이터셋 이름
    """
    print("\n" + "=" * 60)
    print(f"[{name}] 구조 분석")
    print("=" * 60)

    # 1. 기본 정보
    print(f"\n1. 기본 정보")
    print(f"   - 행 수: {df.shape[0]:,}")
    print(f"   - 열 수: {df.shape[1]}")
    print(f"   - 메모리: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # 2. 데이터 타입
    print(f"\n2. 데이터 타입")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   - {dtype}: {count}개 열")

    # 3. 결측치
    print(f"\n3. 결측치")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"   - 총 결측치: {total_missing:,}개")
        print(f"   - 결측 열:")
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            print(f"     * {col}: {missing[col]:,}개 ({pct:.1f}%)")
    else:
        print("   - 결측치 없음")

    # 4. 컬럼 목록
    print(f"\n4. 컬럼 목록")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nunique = df[col].nunique()
        print(f"   {i:2d}. {col:25s} ({dtype}, {nunique} unique)")

    # 5. 수치형 변수 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n5. 수치형 변수 통계 (처음 5개)")
        print(df[numeric_cols[:5]].describe().round(2).to_string())

    # 6. 범주형 변수
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"\n6. 범주형 변수 (처음 3개)")
        for col in cat_cols[:3]:
            print(f"\n   [{col}]")
            print(f"   {df[col].value_counts().head(5).to_string()}")

    print("\n" + "=" * 60)

# 함수 테스트
if SEABORN_AVAILABLE:
    check_dataset(tips, "Tips 데이터셋")
    check_dataset(penguins, "Penguins 데이터셋")


# =====================================================
# 실습 7: 데이터셋 용어 실습
# =====================================================
print("\n" + "=" * 60)
print("실습 7: 데이터셋 용어 실습")
print("=" * 60)

if 'iris_df' in dir():
    # 7-1. 행(Row)과 열(Column)
    print("\n[1. 행(Row)과 열(Column)]")
    sample = iris_df.head(5)

    print(f"행(Row/Sample/Observation) 수: {len(sample)}")
    print(f"열(Column/Feature/Variable) 수: {len(sample.columns)}")
    print(f"\n첫 번째 행 (하나의 샘플):")
    print(sample.iloc[0])

    # 7-2. 독립변수(X)와 종속변수(y)
    print("\n[2. 독립변수(X)와 종속변수(y)]")
    feature_cols = iris.feature_names
    target_col = 'target'

    X = iris_df[feature_cols]
    y = iris_df[target_col]

    print(f"독립변수(X/Features) 크기: {X.shape}")
    print(f"종속변수(y/Target) 크기: {y.shape}")
    print(f"\nX (Features/Input/설명변수):")
    print(X.head(3))
    print(f"\ny (Target/Label/반응변수):")
    print(y.head(3).to_list())

    # 7-3. 수치형 vs 범주형
    print("\n[3. 수치형 vs 범주형]")
    numeric_cols = iris_df.select_dtypes(include=[np.number]).columns
    categorical_cols = iris_df.select_dtypes(include=['object', 'category']).columns

    print(f"수치형(Numerical) 변수: {list(numeric_cols)}")
    print(f"범주형(Categorical) 변수: {list(categorical_cols)}")

    # 7-4. 데이터 분할
    print("\n[4. 데이터 분할]")
    from sklearn.model_selection import train_test_split

    # 70/15/15 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    print(f"훈련 세트(Training): {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"검증 세트(Validation): {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"테스트 세트(Test): {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")


# =====================================================
# 실습 8: 여러 데이터셋 비교
# =====================================================
print("\n" + "=" * 60)
print("실습 8: 여러 데이터셋 비교")
print("=" * 60)

# 비교할 데이터셋 준비
datasets = {}

if 'iris_df' in dir():
    datasets["sklearn Iris"] = iris_df
if 'wine_df' in dir():
    datasets["sklearn Wine"] = wine_df
if SEABORN_AVAILABLE:
    datasets["seaborn Tips"] = tips
    datasets["seaborn Penguins"] = penguins
    datasets["seaborn Diamonds"] = diamonds
if 'wine_quality' in dir():
    datasets["UCI Wine Quality"] = wine_quality

# 비교표 생성
comparison = []
for name, df in datasets.items():
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
    missing_pct = df.isnull().mean().mean() * 100

    comparison.append({
        "데이터셋": name,
        "행 수": len(df),
        "열 수": len(df.columns),
        "수치형": numeric_count,
        "범주형": categorical_count,
        "결측률(%)": round(missing_pct, 1),
        "메모리(KB)": round(df.memory_usage(deep=True).sum() / 1024, 1)
    })

comparison_df = pd.DataFrame(comparison)
print("\n[데이터셋 비교표]")
print(comparison_df.to_string(index=False))


# =====================================================
# 실습 9: 데이터 로드 패턴 정리
# =====================================================
print("\n" + "=" * 60)
print("실습 9: 데이터 로드 패턴 정리")
print("=" * 60)

load_patterns = """
=== 데이터셋 로드 패턴 모음 ===

1. sklearn 데이터셋
   from sklearn.datasets import load_iris, load_wine, load_diabetes
   from sklearn.datasets import load_breast_cancer, load_digits

   data = load_iris()
   X, y = data.data, data.target
   df = pd.DataFrame(X, columns=data.feature_names)

2. seaborn 데이터셋
   import seaborn as sns

   tips = sns.load_dataset('tips')
   penguins = sns.load_dataset('penguins')
   diamonds = sns.load_dataset('diamonds')
   titanic = sns.load_dataset('titanic')

   # 사용 가능한 목록 확인
   print(sns.get_dataset_names())

3. UCI ML Repository (URL 직접)
   url = "https://archive.ics.uci.edu/ml/machine-learning-databases/..."
   df = pd.read_csv(url, sep=';')  # 또는 다른 구분자

4. Kaggle (kaggle API)
   # 터미널: pip install kaggle
   # kaggle datasets download -d username/dataset-name
   df = pd.read_csv('downloaded_file.csv')

5. 공공데이터포털 (data.go.kr)
   # 파일 다운로드 후
   df = pd.read_csv('공공데이터.csv', encoding='cp949')
   # 또는 API 호출
   import requests
   response = requests.get(api_url, params=params)

6. OpenML
   from sklearn.datasets import fetch_openml
   data = fetch_openml(name='wine-quality-red', version=1)
"""
print(load_patterns)


# =====================================================
# 실습 10: 결측치 확인 및 처리
# =====================================================
print("\n" + "=" * 60)
print("실습 10: 결측치 확인 (Penguins 데이터셋)")
print("=" * 60)

if SEABORN_AVAILABLE:
    penguins = sns.load_dataset('penguins')

    print("\n[결측치 확인 방법]")

    # 방법 1: isnull().sum()
    print("\n1. isnull().sum() - 열별 결측치 개수")
    print(penguins.isnull().sum())

    # 방법 2: info()
    print("\n2. info() - Non-Null 개수 확인")
    penguins.info()

    # 방법 3: 결측 비율
    print("\n3. 결측 비율")
    missing_ratio = penguins.isnull().mean() * 100
    print(missing_ratio[missing_ratio > 0].round(2))

    # 결측치 처리 예시
    print("\n[결측치 처리 예시]")
    print(f"원본 크기: {len(penguins)}")

    # dropna
    penguins_clean = penguins.dropna()
    print(f"dropna() 후 크기: {len(penguins_clean)}")

    # fillna
    penguins_filled = penguins.copy()
    for col in penguins.select_dtypes(include=[np.number]).columns:
        penguins_filled[col].fillna(penguins_filled[col].median(), inplace=True)
    print(f"수치형 열 중앙값 대체 후 결측치: {penguins_filled.isnull().sum().sum()}")


# =====================================================
# 실습 11: 메타데이터 작성
# =====================================================
print("\n" + "=" * 60)
print("실습 11: 메타데이터 작성")
print("=" * 60)

def create_metadata(df, name, source, description):
    """데이터셋 메타데이터 생성"""
    metadata = {
        "name": name,
        "version": "1.0",
        "created": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "source": source,
        "description": description,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "schema": []
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "unique_count": int(df[col].nunique())
        }

        if df[col].dtype in ['int64', 'float64']:
            col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            col_info["mean"] = round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else None
        else:
            col_info["sample_values"] = df[col].dropna().unique()[:5].tolist()

        metadata["schema"].append(col_info)

    return metadata

# 메타데이터 생성
if 'iris_df' in dir():
    metadata = create_metadata(
        df=iris_df,
        name="Iris_Dataset",
        source="sklearn / UCI ML Repository",
        description="붓꽃 품종 분류를 위한 4개 특성과 3개 클래스"
    )

    print("\n[생성된 메타데이터]")
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str)[:2000])


# =====================================================
# 실습 12: 데이터 저장 및 로드
# =====================================================
print("\n" + "=" * 60)
print("실습 12: 데이터 저장 및 로드")
print("=" * 60)

save_load_examples = """
[CSV 저장/로드]
# 저장
df.to_csv('data.csv', index=False, encoding='utf-8')

# 로드
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', encoding='cp949')  # 한글 인코딩

[Excel 저장/로드]
# 저장
df.to_excel('data.xlsx', index=False, sheet_name='Sheet1')

# 로드
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

[JSON 저장/로드]
# 저장
df.to_json('data.json', orient='records', force_ascii=False, indent=2)

# 로드
df = pd.read_json('data.json')

[Pickle 저장/로드 (Python 객체)]
# 저장
df.to_pickle('data.pkl')

# 로드
df = pd.read_pickle('data.pkl')
"""
print(save_load_examples)


# =====================================================
# 마무리
# =====================================================
print("\n" + "=" * 60)
print("4차시 실습 완료!")
print("=" * 60)

print("""
오늘 배운 핵심 내용:

1. sklearn 내장 데이터셋
   from sklearn.datasets import load_iris, load_wine, load_diabetes
   data = load_iris()
   X, y = data.data, data.target

2. seaborn 내장 데이터셋
   import seaborn as sns
   tips = sns.load_dataset('tips')
   penguins = sns.load_dataset('penguins')

3. UCI ML Repository (URL 직접 접근)
   url = "https://archive.ics.uci.edu/ml/..."
   df = pd.read_csv(url, sep=';')

4. 데이터 확인 필수 체크리스트
   df.shape      - 크기
   df.columns    - 컬럼 목록
   df.dtypes     - 데이터 타입
   df.isnull()   - 결측치
   df.describe() - 기술통계

5. 주요 공개 데이터셋
   - sklearn: iris, wine, diabetes, breast_cancer
   - seaborn: tips, penguins, diamonds, titanic
   - UCI: wine-quality, iris, adult 등

다음 시간: 5차시 - 기초 기술통계량과 탐색적 시각화
""")
