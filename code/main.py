
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from pathlib import Path

# 페이지 기본 설정
st.set_page_config(
    page_title="산업용 로봇 도입과 산업재해 회귀분석",
    layout="wide"
)

st.title("🤖 산업용 로봇 도입이 산업재해를 줄일 수 있을까?")
st.markdown(
    "2004년~2025년 데이터를 바탕으로 "
    "**일반기계산업 내수 규모(X)**와 **산업재해자 수(Y)**의 관계를 분석합니다."
)

# --------------------------------------------------
# 1. 엑셀 파일 경로 자동 찾기
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

def find_file(filename):
    candidates = [
        BASE_DIR / filename,
        PROJECT_DIR / filename,
        BASE_DIR / "data" / filename,
        PROJECT_DIR / "data" / filename
    ]

    for path in candidates:
        if path.exists():
            return path

    # 프로젝트 폴더 내부에서 파일명 검색
    matches = list(PROJECT_DIR.rglob(filename))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"{filename} 파일을 찾을 수 없습니다. "
        "GitHub에 엑셀 파일이 업로드되어 있는지 확인하세요."
    )


# --------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# --------------------------------------------------

@st.cache_data
def load_data():

    robot_path = find_file(
        "1_로봇_연도별_내수_2004_2025.xlsx"
    )

    accident_path = find_file(
        "2_산업중분류별_재해자수_전처리_2004_2025.xlsx"
    )

    df_robot = pd.read_excel(robot_path)
    df_acc = pd.read_excel(accident_path)

    # 산업재해 합계 행 제외
    df_acc = df_acc[
        df_acc["대업종"] != "합계"
    ].copy()

    years = list(range(2004, 2026))

    # 로봇 데이터 확인
    if "내수(십억원)" not in df_robot.columns:
        raise ValueError("로봇 엑셀에 '내수(십억원)' 열이 없습니다.")

    if len(df_robot) != len(years):
        raise ValueError(
            f"로봇 데이터 행 개수가 맞지 않습니다. "
            f"예상: {len(years)}개, 실제: {len(df_robot)}개"
        )

    # 연도별 산업재해자 수 계산
    total_acc = []
    mfg_acc = []
    mach_acc = []

    for year in years:

        col = f"{year}년 재해자수"

        if col not in df_acc.columns:
            raise ValueError(
                f"산업재해 엑셀에 '{col}' 열이 없습니다."
            )

        total_acc.append(
            pd.to_numeric(df_acc[col], errors="coerce").sum()
        )

        mfg_acc.append(
            pd.to_numeric(
                df_acc.loc[df_acc["대업종"] == "제조업", col],
                errors="coerce"
            ).sum()
        )

        mach_acc.append(
            pd.to_numeric(
                df_acc.loc[
                    df_acc["구분"] ==
                    "기계기구·금속·비금속광물제품제조업",
                    col
                ],
                errors="coerce"
            ).sum()
        )

    # 통합 데이터 생성
    merged = pd.DataFrame({
        "연도": years,
        "일반기계_내수(십억원)": pd.to_numeric(
            df_robot["내수(십억원)"],
            errors="coerce"
        ).values,
        "전체_산업_재해자수(명)": total_acc,
        "제조업_전체_재해자수(명)": mfg_acc,
        "기계·금속제조업_재해자수(명)": mach_acc
    })

    # 결측값 제거
    merged_clean = merged.dropna().copy()

    return merged, merged_clean


# 데이터 로딩 오류 처리
try:
    df_all, df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 3. 사이드바 분석 옵션
# --------------------------------------------------

st.sidebar.header("⚙️ 분석 옵션 설정")

target_col = st.sidebar.selectbox(
    "비교할 산업재해자 수 선택:",
    [
        "기계·금속제조업_재해자수(명)",
        "제조업_전체_재해자수(명)",
        "전체_산업_재해자수(명)"
    ]
)

year_range = st.sidebar.slider(
    "분석 연도 범위 선택:",
    2004,
    2024,
    (2004, 2024)
)

df_filtered = df[
    (df["연도"] >= year_range[0]) &
    (df["연도"] <= year_range[1])
].copy()


# --------------------------------------------------
# 4. 회귀분석 계산
# --------------------------------------------------

st.subheader("📊 분석 데이터")

st.write(
    f"분석 기간: {year_range[0]}년 ~ {year_range[1]}년"
)

st.write(f"분석 데이터 개수: {len(df_filtered)}개 연도")

st.dataframe(
    df_filtered,
    use_container_width=True
)

if len(df_filtered) < 3:
    st.warning("회귀분석을 하려면 최소 3개 연도의 데이터가 필요합니다.")
    st.stop()

x = df_filtered["일반기계_내수(십억원)"].values
y = df_filtered[target_col].values

if len(np.unique(x)) < 2:
    st.error("투자금 데이터의 값이 모두 같아 회귀분석을 할 수 없습니다.")
    st.stop()

slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

r_squared = r_val ** 2
y_pred = slope * x + intercept


# --------------------------------------------------
# 5. 핵심 통계 지표
# --------------------------------------------------

st.subheader("📌 회귀분석 핵심 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "회귀 기울기",
    f"{slope:.4f}"
)

col2.metric(
    "상관계수 (R)",
    f"{r_val:.4f}"
)

col3.metric(
    "결정계수 (R²)",
    f"{r_squared:.4f}"
)

col4.metric(
    "유의확률 (p-value)",
    f"{p_val:.4f}"
)


# --------------------------------------------------
# 6. 회귀분석 산점도
# --------------------------------------------------

st.divider()

st.subheader("📈 로봇 내수와 산업재해자 수의 회귀분석")

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=df_filtered["연도"].astype(str) + "년",
        textposition="top center",
        name="실제 데이터",
        marker=dict(
            size=11,
            color="#1F77B4"
        )
    )
)

sort_idx = np.argsort(x)

fig1.add_trace(
    go.Scatter(
        x=x[sort_idx],
        y=y_pred[sort_idx],
        mode="lines",
        name="회귀 직선",
        line=dict(
            color="#E74C3C",
            width=3
        )
    )
)

fig1.update_layout(
    title=f"일반기계 내수와 {target_col}의 관계",
    xaxis_title="일반기계산업 내수 (십억원)",
    yaxis_title="산업재해자 수 (명)",
    template="plotly_white",
    hovermode="closest"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# --------------------------------------------------
# 7. 연도별 추이 비교
# --------------------------------------------------

st.subheader("📊 연도별 내수 및 산업재해 변화")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered["일반기계_내수(십억원)"],
        name="일반기계 내수",
        mode="lines+markers",
        line=dict(color="#1F77B4", width=3)
    )
)

fig2.add_trace(
    go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered[target_col],
        name=target_col,
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#E67E22", width=3)
    )
)

fig2.update_layout(
    title="연도별 일반기계 내수와 산업재해자 수 변화",
    xaxis_title="연도",
    yaxis=dict(
        title="일반기계 내수 (십억원)"
    ),
    yaxis2=dict(
        title="산업재해자 수 (명)",
        overlaying="y",
        side="right"
    ),
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# --------------------------------------------------
# 8. 분석 결과 해석
# --------------------------------------------------

st.divider()

st.subheader("📝 데이터 분석 및 해석")

if slope < 0:

    st.info(
        f"음(-)의 관계가 나타났습니다. "
        f"회귀분석 결과 상관계수는 {r_val:.3f}입니다. "
        "일반기계 내수가 증가할수록 산업재해자 수가 감소하는 "
        "경향이 나타났습니다."
    )

elif slope > 0:

    st.warning(
        f"양(+)의 관계가 나타났습니다. "
        f"회귀분석 결과 상관계수는 {r_val:.3f}입니다. "
        "일반기계 내수가 증가할수록 산업재해자 수도 증가하는 "
        "경향이 나타났습니다."
    )

else:

    st.info("두 변수 사이에 선형적인 관계가 거의 나타나지 않습니다.")


if p_val < 0.05:
    st.success(
        "p-value가 0.05보다 작아 통계적으로 유의미한 "
        "선형 관계가 관찰되었습니다."
    )
else:
    st.warning(
        "p-value가 0.05 이상으로 통계적으로 유의미한 "
        "선형 관계가 확인되지 않았습니다."
    )


# --------------------------------------------------
# 9. 연구의 한계점
# --------------------------------------------------

st.divider()

st.subheader("⚠️ 연구의 한계점")

st.write("""
1. 본 분석에 사용된 내수 데이터는 일반기계산업 전체의 금액 기준 통계이므로
   산업용 로봇만의 투자 규모를 직접 나타내지는 않습니다.

2. 산업재해자 수는 산업 규모와 근로자 수의 변화에도 영향을 받습니다.

3. 회귀분석에서 상관관계가 발견되더라도 로봇 도입이
   산업재해를 직접 감소시키거나 증가시켰다고 단정할 수 없습니다.

4. 실제 로봇 도입 효과를 더 정확하게 알아보려면
   산업용 로봇 보급 대수와 산업별 근로자 수 등의 자료도 필요합니다.
""")


# --------------------------------------------------
# 10. 통합 데이터 확인
# --------------------------------------------------

with st.expander("📋 전체 통합 데이터 확인"):

    st.dataframe(
        df_all,
        use_container_width=True
    )
