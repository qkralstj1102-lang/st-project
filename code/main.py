import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(page_title="산업용 로봇 도입과 산업재해 회귀분석", layout="wide")

st.title("🤖 산업용 로봇(일반기계 내수) 도입이 산업재해를 줄일 수 있을까?")
st.markdown("2004년~2024년 데이터를 바탕으로 **일반기계산업 내수 규모(X)**와 **산업재해자 수(Y)** 간의 선형 회귀분석을 수행합니다.")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    df_robot = pd.read_excel("1_로봇_연도별_내수_2004_2025.xlsx")
    df_acc = pd.read_excel("2_산업중분류별_재해자수_전처리_2004_2025.xlsx")
    df_acc = df_acc[df_acc["대업종"] != "합계"].copy()

    years = list(range(2004, 2026))
    total_acc = [df_acc[f"{y}년 재해자수"].sum() for y in years]
    mfg_acc = [df_acc[df_acc["대업종"] == "제조업"][f"{y}년 재해자수"].sum() for y in years]
    mach_acc = [
        df_acc[df_acc["구분"] == "기계기구·금속·비금속광물제품제조업"][f"{y}년 재해자수"].values[0]
        for y in years
    ]

    merged = pd.DataFrame({
        "연도": years,
        "일반기계_내수(십억원)": df_robot["내수(십억원)"].values,
        "전체_산업_재해자수(명)": total_acc,
        "제조업_전체_재해자수(명)": mfg_acc,
        "기계·금속제조업_재해자수(명)": mach_acc,
    })
    # 2025년 내수 결측치(NaN) 제거 -> 2004~2024년 분석
    merged_clean = merged.dropna().copy()
    return merged, merged_clean

df_all, df = load_data()

# 2. 사이드바 옵션 선택
st.sidebar.header("⚙️ 분석 옵션 설정")
target_col = st.sidebar.selectbox(
    "비교할 재해자 수(Y축) 선택:",
    [
        "기계·금속제조업_재해자수(명)",
        "제조업_전체_재해자수(명)",
        "전체_산업_재해자수(명)",
    ],
    index=0
)

year_range = st.sidebar.slider("분석 연도 범위 선택:", 2004, 2024, (2004, 2024))
df_filtered = df[(df["연도"] >= year_range[0]) & (df["연도"] <= year_range[1])]

# 3. 선형 회귀분석 계산
x = df_filtered["일반기계_내수(십억원)"].values
y = df_filtered[target_col].values

slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
r_squared = r_val ** 2
y_pred = slope * x + intercept

# 4. 핵심 통계 지표 출력
col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀 기울기 (Slope)", f"{slope:.4f}")
col2.metric("상관계수 (R)", f"{r_val:.4f}")
col3.metric("결정계수 (R²)", f"{r_squared:.4f}")
col4.metric("유의확률 (p-value)", f"{p_val:.4f}")

# 5. 회귀분석 그래프 및 시계열 그래프 그리기
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"📈 선형 회귀분석: 내수 vs {target_col}")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(x, y, color="#1F4E78", s=60, alpha=0.85, label="실제 데이터 (연도별)")
    
    # 회귀선 정렬 후 그리기
    sort_idx = np.argsort(x)
    ax1.plot(x[sort_idx], y_pred[sort_idx], color="#D9534F", linewidth=2.2,
             label=f"회귀선: y = {slope:.4f}x + {intercept:.1f}")

    # 시작연도와 끝연도 텍스트 표시
    for idx in [0, len(df_filtered) - 1]:
        ax1.annotate(
            f"{int(df_filtered.iloc[idx]['연도'])}년",
            (x[idx], y[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax1.set_xlabel("일반기계산업 내수 (십억원)")
    ax1.set_ylabel(target_col)
    ax1.set_title(f"R = {r_val:.3f}, R² = {r_squared:.3f}, p = {p_val:.4f}")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    st.pyplot(fig1)

with right_col:
    st.subheader("📊 연도별 추이 비교 (이중축 그래프)")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2_twin = ax2.twinx()

    p1 = ax2.plot(df_filtered["연도"], df_filtered["일반기계_내수(십억원)"],
                  color="#1F4E78", marker="o", linewidth=2, label="일반기계 내수(십억원)")
    p2 = ax2_twin.plot(df_filtered["연도"], df_filtered[target_col],
                       color="#E67E22", marker="s", linewidth=2, linestyle="--", label=target_col)

    ax2.set_xlabel("연도")
    ax2.set_ylabel("일반기계 내수 (십억원)", color="#1F4E78")
    ax2_twin.set_ylabel(target_col, color="#E67E22")
    ax2.set_xticks(df_filtered["연도"][::2])
    ax2.grid(True, linestyle="--", alpha=0.4)

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")
    st.pyplot(fig2)

# 6. 데이터 해석 및 한계점 안내
st.divider()
st.subheader("📝 데이터 해석 및 연구 한계점")
if slope < 0 and p_val < 0.05:
    st.success(
        f"**통계적으로 유의미한 음(-)의 상관관계가 확인되었습니다 (R = {r_val:.3f}, p = {p_val:.4f}).** "
        "일반기계산업 내수가 증가할수록 해당 제조업 분야의 산업재해자 수는 감소하는 경향을 보입니다."
    )
elif slope > 0:
    st.warning(
        f"**양(+)의 상관관계가 나타납니다 (R = {r_val:.3f}, p = {p_val:.4f}).** "
        "전체 산업에는 로봇이 도입되지 않는 건설·물류·서비스업의 산재 증가분이 포함되어 있기 때문입니다. "
        "왼쪽 사이드바에서 **'기계·금속제조업_재해자수(명)'**로 변경해 비교해 보세요."
    )

st.info(
    "**데이터 한계점:** 본 분석에 사용된 내수 데이터는 '일반기계산업 전체'의 금액 기준 통계이므로, "
    "산업용 로봇만의 단독 보급 대수를 직접 분리하지 못했다는 한계가 있습니다."
)

with st.expander("📋 전처리된 통합 데이터 표 보기"):
    st.dataframe(df_all, use_container_width=True)
