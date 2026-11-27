
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import re

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="산업용 로봇과 산업재해 분석",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 산업용 로봇 도입은 산업재해를 줄일 수 있을까?")

st.markdown("""
2004년~2025년의 공공데이터를 활용하여
**일반기계산업 내수 규모와 산업재해자 수의 관계**를 분석합니다.
""")

st.divider()


# ==========================================
# 2. 엑셀 파일 경로 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ROBOT_FILE = BASE_DIR / "1_로봇_연도별_내수_2004_2025.xlsx"

ACCIDENT_FILE = BASE_DIR / "2_산업재해_연도별_총합_2004_2025.xlsx"


# ==========================================
# 3. 데이터 전처리 함수
# ==========================================

def find_year_columns(df):
    """2004년, 2005년 등의 연도 열을 찾는 함수"""

    result = {}

    for col in df.columns:
        match = re.search(r"(20\d{2})", str(col))

        if match:
            year = int(match.group(1))

            if 2004 <= year <= 2025:
                result[year] = col

    return result


def find_year_col(df):
    """연도 또는 년도 열 찾기"""

    for col in df.columns:
        name = str(col).strip().lower()

        if name in ["연도", "년도", "year"]:
            return col

    return None


def find_value_col(df, keyword, preferred=None):
    """내수 또는 재해자 수 열 찾기"""

    if preferred and preferred in df.columns:
        return preferred

    for col in df.columns:
        if keyword in str(col):
            return col

    raise ValueError(
        f"엑셀 파일에서 '{keyword}' 관련 열을 찾을 수 없습니다. "
        f"현재 열 이름: {list(df.columns)}"
    )


def make_year_data(df, keyword, preferred=None):
    """연도별 데이터로 변환"""

    year_col = find_year_col(df)
    year_columns = find_year_columns(df)

    # 가로형 데이터: 2004년, 2005년 등이 열에 있는 경우
    if year_columns:
        result = []

        for year, col in year_columns.items():
            values = pd.to_numeric(df[col], errors="coerce")
            result.append({
                "연도": year,
                "값": values.sum(min_count=1)
            })

        return pd.DataFrame(result)

    # 세로형 데이터: 연도와 값이 각각 열에 있는 경우
    if year_col is not None:
        value_col = find_value_col(df, keyword, preferred)

        temp = pd.DataFrame({
            "연도": pd.to_numeric(df[year_col], errors="coerce"),
            "값": pd.to_numeric(df[value_col], errors="coerce")
        })

        temp = temp.dropna()

        temp["연도"] = temp["연도"].astype(int)

        return temp.groupby("연도", as_index=False)["값"].sum()

    # 연도 열이 없고 값만 있는 경우
    value_col = find_value_col(df, keyword, preferred)

    temp = pd.to_numeric(df[value_col], errors="coerce").dropna()

    years = list(range(2004, 2004 + len(temp)))

    if len(temp) > 22:
        raise ValueError("연도 열이 없는 데이터의 행 개수가 22개를 초과합니다.")

    return pd.DataFrame({
        "연도": years,
        "값": temp.values
    })


# ==========================================
# 4. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    if not ROBOT_FILE.exists():
        raise FileNotFoundError(
            f"로봇 엑셀 파일을 찾을 수 없습니다: {ROBOT_FILE}"
        )

    if not ACCIDENT_FILE.exists():
        raise FileNotFoundError(
            f"산업재해 엑셀 파일을 찾을 수 없습니다: {ACCIDENT_FILE}"
        )

    robot_raw = pd.read_excel(ROBOT_FILE)
    accident_raw = pd.read_excel(ACCIDENT_FILE)

    # 로봇 내수 데이터
    robot = make_year_data(
        robot_raw,
        keyword="내수",
        preferred="내수(십억원)"
    )

    robot = robot.rename(
        columns={"값": "로봇_내수"}
    )

    # 산업재해자 수 데이터
    accident = make_year_data(
        accident_raw,
        keyword="재해자",
        preferred="재해자수"
    )

    accident = accident.rename(
        columns={"값": "산업재해자수"}
    )

    # 연도를 기준으로 두 데이터 결합
    data = pd.merge(
        robot,
        accident,
        on="연도",
        how="inner"
    )

    data = data.sort_values("연도")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=["로봇_내수", "산업재해자수"]
    )

    data = data[
        (data["연도"] >= 2004) &
        (data["연도"] <= 2025)
    ].copy()

    return data


# ==========================================
# 5. 데이터 실행 및 오류 확인
# ==========================================

try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 문제가 발생했습니다.")
    st.exception(e)

    st.info("""
    확인할 사항

    1. 두 엑셀 파일이 모두 code 폴더 안에 있는지 확인하세요.
    2. 파일 이름이 코드와 정확히 같은지 확인하세요.
    3. 엑셀의 열 이름이 실제 데이터와 일치하는지 확인하세요.
    """)

    st.stop()


if len(df) < 3:
    st.error("회귀분석을 진행하기 위한 데이터가 부족합니다.")
    st.stop()


# ==========================================
# 6. 분석 기간 선택
# ==========================================

st.sidebar.header("⚙️ 분석 설정")

min_year = int(df["연도"].min())
max_year = int(df["연도"].max())

if min_year == max_year:
    st.error("분석 가능한 연도가 2개 이상 필요합니다.")
    st.stop()

year_range = st.sidebar.slider(
    "분석 연도 선택",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

df_filtered = df[
    (df["연도"] >= year_range[0]) &
    (df["연도"] <= year_range[1])
].copy()

if len(df_filtered) < 3:
    st.warning("회귀분석을 위해 최소 3개 연도를 선택해 주세요.")
    st.stop()


# ==========================================
# 7. 데이터 요약
# ==========================================

st.subheader("📋 분석 데이터 요약")

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 기간",
    f"{year_range[0]}~{year_range[1]}년"
)

col2.metric(
    "분석 연도 수",
    f"{len(df_filtered)}년"
)

col3.metric(
    "데이터 기준",
    "공공데이터"
)

with st.expander("통합 데이터 표 확인하기"):
    st.dataframe(
        df_filtered,
        use_container_width=True
    )


# ==========================================
# 8. 그래프 1 - 연도별 산업재해자 수
# ==========================================

st.divider()

st.header("1️⃣ 연도별 산업재해자 수")

st.markdown("""
연도에 따라 산업재해자 수가 어떻게 변화했는지 확인합니다.
""")

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered["산업재해자수"],
        mode="lines+markers",
        name="산업재해자 수",
        line=dict(
            color="#E67E22",
            width=3
        ),
        marker=dict(size=9),
        hovertemplate=
        "%{x}년<br>산업재해자 수: %{y:,.0f}명<extra></extra>"
    )
)

fig1.update_layout(
    title="연도별 산업재해자 수 변화",
    xaxis_title="연도",
    yaxis_title="산업재해자 수 (명)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ==========================================
# 9. 그래프 2 - 연도별 로봇 내수
# ==========================================

st.divider()

st.header("2️⃣ 연도별 로봇 내수")

st.markdown("""
일반기계산업의 내수 규모가 연도에 따라 어떻게 변화했는지 확인합니다.
""")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered["로봇_내수"],
        mode="lines+markers",
        name="일반기계 내수",
        line=dict(
            color="#2874A6",
            width=3
        ),
        marker=dict(size=9),
        hovertemplate=
        "%{x}년<br>내수: %{y:,.2f}십억원<extra></extra>"
    )
)

fig2.update_layout(
    title="연도별 일반기계산업 내수 변화",
    xaxis_title="연도",
    yaxis_title="내수 (십억원)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ==========================================
# 10. 그래프 3 - 회귀분석
# ==========================================


# ==========================================
# 10. 기준선 및 회귀분석 계산
# ==========================================

st.divider()

st.header("3️⃣ 로봇 내수에 따른 산업재해자 수")

st.markdown("""
로봇 내수를 독립변수(X), 산업재해자 수를 종속변수(Y)로 설정합니다.
기준선 모델과 선형 회귀모델을 비교합니다.
""")

x = df_filtered["로봇_내수"].values
y = df_filtered["산업재해자수"].values

# ------------------------------------------
# 기준선 모델
# ------------------------------------------

# 모든 연도의 산업재해자 수 평균 계산
baseline_value = np.mean(y)

# 모든 연도를 평균값으로 예측
baseline_pred = np.full(len(y), baseline_value)

# ------------------------------------------
# 선형 회귀모델
# ------------------------------------------

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r_squared = r_value ** 2

y_pred = slope * x + intercept

# 회귀직선 생성
x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept

# ------------------------------------------
# 모델 평가 지표 계산
# ------------------------------------------

# 평균절대오차(MAE)
baseline_mae = np.mean(np.abs(y - baseline_pred))
regression_mae = np.mean(np.abs(y - y_pred))

# 평균제곱근오차(RMSE)
baseline_rmse = np.sqrt(np.mean((y - baseline_pred) ** 2))
regression_rmse = np.sqrt(np.mean((y - y_pred) ** 2))


# ==========================================
# 11. 회귀분석 그래프
# ==========================================

fig3 = go.Figure()

# 실제 데이터
fig3.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=df_filtered["연도"].astype(str) + "년",
        textposition="top center",
        name="실제 데이터",
        marker=dict(size=11, color="#2874A6")
    )
)

# 기준선: 산업재해자 수 평균
fig3.add_trace(
    go.Scatter(
        x=x_line,
        y=np.full(len(x_line), baseline_value),
        mode="lines",
        name="기준선 모델 (평균 예측)",
        line=dict(color="#2ECC71", width=3, dash="dash")
    )
)

# 선형 회귀직선
fig3.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="선형 회귀모델",
        line=dict(color="#E74C3C", width=3)
    )
)

fig3.update_layout(
    title="로봇 내수와 산업재해자 수의 관계",
    xaxis_title="일반기계산업 내수 (십억원)",
    yaxis_title="산업재해자 수 (명)",
    template="plotly_white",
    hovermode="closest"
)

st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 12. 기준선과 회귀모델 비교
# ==========================================

st.subheader("📊 기준선 모델과 회귀모델 비교")

st.markdown("""
기준선 모델은 산업재해자 수의 평균을 예측하고,
회귀모델은 로봇 내수를 이용해 산업재해자 수를 예측합니다.
""")

comparison = pd.DataFrame({
    "모델": ["기준선 모델", "선형 회귀모델"],
    "MAE": [baseline_mae, regression_mae],
    "RMSE": [baseline_rmse, regression_rmse]
})

st.dataframe(
    comparison.style.format({
        "MAE": "{:,.2f}",
        "RMSE": "{:,.2f}"
    }),
    use_container_width=True
)

col1, col2 = st.columns(2)

col1.metric(
    "기준선 MAE",
    f"{baseline_mae:,.2f}명"
)

col2.metric(
    "회귀모델 MAE",
    f"{regression_mae:,.2f}명"
)

if regression_mae < baseline_mae:
    st.success(
        "선형 회귀모델이 기준선 모델보다 평균적으로 오차가 작습니다. "
        "로봇 내수를 활용한 예측이 단순 평균 예측보다 나은 결과를 보였습니다."
    )
elif regression_mae > baseline_mae:
    st.warning(
        "선형 회귀모델의 오차가 기준선 모델보다 큽니다. "
        "이번 데이터에서는 로봇 내수를 활용한 예측이 단순 평균 예측보다 "
        "나은 결과를 보이지 못했습니다."
    )
else:
    st.info("두 모델의 평균 예측 오차가 같습니다.")


# ==========================================
# 13. 회귀분석 통계 결과
# ==========================================

st.subheader("📈 회귀분석 통계 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("상관계수 (R)", f"{r_value:.4f}")
col2.metric("결정계수 (R²)", f"{r_squared:.4f}")
col3.metric("회귀계수", f"{slope:.4f}")
col4.metric("유의확률 (p-value)", f"{p_value:.4f}")

st.markdown(
    f"**회귀식:** 산업재해자 수 = "
    f"{slope:.4f} × 로봇 내수 + ({intercept:.2f})"
)

if slope < 0:
    st.info("로봇 내수가 증가할수록 산업재해자 수가 감소하는 방향의 관계입니다.")
elif slope > 0:
    st.info("로봇 내수가 증가할수록 산업재해자 수도 증가하는 방향의 관계입니다.")

if p_value < 0.05:
    st.success("통계적으로 유의미한 선형 관계가 관찰되었습니다.")
else:
    st.warning("통계적으로 유의미한 선형 관계라고 판단하기 어렵습니다.")



# ==========================================
# 11. 회귀분석 결과
# ==========================================

st.subheader("📈 회귀분석 통계 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "상관계수 (R)",
    f"{r_value:.4f}"
)

col2.metric(
    "결정계수 (R²)",
    f"{r_squared:.4f}"
)

col3.metric(
    "회귀계수",
    f"{slope:.4f}"
)

col4.metric(
    "유의확률 (p-value)",
    f"{p_value:.4f}"
)

st.markdown(
    f"**회귀식:** 산업재해자 수 = "
    f"{slope:.4f} × 로봇 내수 + ({intercept:.2f})"
)

if slope < 0:
    st.info(
        "음(-)의 관계가 나타났습니다. "
        "로봇 내수가 증가할수록 산업재해자 수가 감소하는 "
        "방향의 선형 관계입니다."
    )

elif slope > 0:
    st.info(
        "양(+)의 관계가 나타났습니다. "
        "로봇 내수가 증가할수록 산업재해자 수도 증가하는 "
        "방향의 선형 관계입니다."
    )

else:
    st.info("선형 회귀계수가 0에 가깝습니다.")

if p_value < 0.05:
    st.success(
        "p-value가 0.05보다 작아 통계적으로 유의미한 "
        "선형 관계가 관찰되었습니다."
    )
else:
    st.warning(
        "p-value가 0.05 이상이므로 통계적으로 유의미한 "
        "선형 관계라고 판단하기 어렵습니다."
    )


# ==========================================
# 12. 연구의 한계
# ==========================================

st.divider()

st.header("⚠️ 연구의 한계점")

st.markdown("""
- 로봇 내수 데이터는 일반기계산업 전체의 금액 기준 통계이므로 산업용 로봇만의 도입 규모를 정확하게 나타내지는 않습니다.
- 산업재해자 수는 근로자 수, 산업 규모, 안전관리 수준 등 다양한 요인의 영향을 받습니다.
- 두 변수 사이에 상관관계가 나타나더라도 로봇 도입이 산업재해를 직접 감소시켰다고 단정할 수 없습니다.
- 로봇 도입의 효과를 정확히 확인하려면 산업용 로봇 보급 대수, 산업별 근로자 수 등 추가적인 데이터가 필요합니다.
""")

st.caption(
    "본 앱은 공공데이터를 이용한 교육용 통계 분석 프로그램입니다."
)
