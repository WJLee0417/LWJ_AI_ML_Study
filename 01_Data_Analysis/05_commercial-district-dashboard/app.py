"""Decision-support Streamlit dashboard for Seoul commercial districts."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "commercial_district_quarterly.csv"
RAW_PATH = ROOT / "data" / "raw"
# 원본 파일에는 신뢰할 수 있는 수정 시간이 없어, 정제 실행 시 이 값을 갱신한다.
DATA_REFRESHED_AT = "2026-09-01"
TIME_BANDS = {
    "오전 (06~11시)": "sales_06_11",
    "점심 (11~14시)": "sales_11_14",
    "저녁 (17~21시)": "sales_17_21",
}

st.set_page_config(page_title="서울 상권 분석", page_icon="☕", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def quarter_label(period: int) -> str:
    value = str(period)
    return f"{value[:4]} Q{value[-1]}"


def previous_period(periods: list[int], period: int) -> int | None:
    prior = [value for value in periods if value < period]
    return max(prior) if prior else None


st.title("서울 상권 분석 대시보드")
st.caption("상권·업종·분기별 매출, 점포 수, 유동인구를 함께 비교해 추가 조사 후보를 좁히는 도구입니다.")

if not DATA_PATH.exists():
    st.warning("정제 데이터가 없습니다. README의 원본 데이터 수집·정제 절차를 먼저 실행하세요.")
    st.stop()

data = load_data(str(DATA_PATH))
data["period"] = pd.to_numeric(data["period"], errors="coerce").astype("Int64")
data_updated = quarter_label(int(data["period"].max()))

st.info(
    f"데이터 기준: **{data_updated}** · 정제 데이터 갱신일: **{DATA_REFRESHED_AT}** · "
    "원본 수집일: **파일 메타데이터 미제공(수동 다운로드)** · "
    "유동인구 원본은 2025년 분기만 제공되어 유동인구 기반 지표는 2025년으로 제한됩니다."
)

districts = sorted(data["district_name"].dropna().unique())
industries = sorted(data["industry_name"].dropna().unique())
district = st.sidebar.selectbox("자치구", districts, index=districts.index("강남구") if "강남구" in districts else 0)
industry = st.sidebar.selectbox("업종", industries, index=industries.index("커피-음료") if "커피-음료" in industries else 0)
all_subset = data.query("district_name == @district and industry_name == @industry").copy()
periods = sorted(all_subset["period"].dropna().astype(int).unique())
selected_periods = st.sidebar.multiselect("분기", periods, default=periods, format_func=quarter_label)
if not selected_periods:
    st.info("하나 이상의 분기를 선택하세요.")
    st.stop()

subset = all_subset.loc[all_subset["period"].isin(selected_periods)].copy()
latest_period = max(selected_periods)
latest = subset.loc[subset["period"].eq(latest_period)].copy()
latest_label = quarter_label(latest_period)
population_available = latest["floating_population"].notna().any()

cols = st.columns(4)
cols[0].metric(f"{latest_label} 추정매출", f"{latest['estimated_sales'].sum():,.0f}원")
cols[1].metric("점포당 추정매출", f"{latest['sales_per_store'].mean():,.0f}원")
cols[2].metric("평균 점포 수", f"{latest['store_count'].mean():,.1f}")
cols[3].metric("평균 경쟁도", f"{latest['competition_percentile'].mean():.1f}")

trend = subset.groupby("period", as_index=False)["estimated_sales"].sum()
trend["period_label"] = trend["period"].map(quarter_label)
st.plotly_chart(
    px.line(trend, x="period_label", y="estimated_sales", markers=True, title="분기 추정매출 추이",
            labels={"period_label": "분기", "estimated_sales": "추정매출(원)"}),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    time_sales = pd.DataFrame(
        {"시간대": list(TIME_BANDS), "매출": [latest[column].sum() for column in TIME_BANDS.values()]}
    )
    time_sales["매출 비중(%)"] = time_sales["매출"] / latest["estimated_sales"].sum() * 100
    st.plotly_chart(
        px.bar(time_sales, x="시간대", y="매출 비중(%)", text=time_sales["매출 비중(%)"].round(1),
               title=f"{latest_label} 전체 매출 대비 시간대별 매출 비중"),
        use_container_width=True,
    )
with right:
    comparison = latest.sort_values("sales_per_store", ascending=False)
    st.plotly_chart(
        px.bar(comparison, x="district_label", y="sales_per_store", color="competition_percentile",
               title=f"{latest_label} 상권별 점포당 추정매출과 경쟁도",
               labels={"district_label": "상권", "sales_per_store": "점포당 추정매출(원)"}),
        use_container_width=True,
    )

st.subheader("상권 위치·성과 비교")
if latest[["x_coord", "y_coord"]].notna().all(axis=None):
    coordinate_plot = px.scatter(
        latest, x="x_coord", y="y_coord", size="estimated_sales", color="sales_per_store",
        hover_name="district_label",
        hover_data={"store_count": ":.0f", "competition_percentile": ":.1f", "x_coord": False, "y_coord": False},
        title=f"{latest_label} 좌표 기반 상권 산점도 (원 크기: 추정매출, 색: 점포당 추정매출)",
        labels={"x_coord": "X 좌표", "y_coord": "Y 좌표", "sales_per_store": "점포당 추정매출(원)"},
    )
    coordinate_plot.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(coordinate_plot, use_container_width=True)
else:
    st.warning("좌표 데이터가 없어 상권 위치 산점도를 표시할 수 없습니다.")

prior_period = previous_period(periods, latest_period)
st.subheader("점포 수 증감 × 매출 증감 사분면")
if prior_period is None:
    st.info("사분면을 만들려면 이전 분기 데이터가 필요합니다.")
else:
    current = all_subset.loc[all_subset["period"].eq(latest_period)]
    prior = all_subset.loc[all_subset["period"].eq(prior_period)]
    quadrant = current.merge(prior, on="district_code", suffixes=("", "_prior"))
    quadrant["sales_growth_pct"] = (quadrant["estimated_sales"] / quadrant["estimated_sales_prior"] - 1) * 100
    quadrant["store_growth_pct"] = (quadrant["store_count"] / quadrant["store_count_prior"] - 1) * 100
    quadrant = quadrant.dropna(subset=["sales_growth_pct", "store_growth_pct"])
    chart = px.scatter(
        quadrant, x="store_growth_pct", y="sales_growth_pct", hover_name="district_label",
        size="estimated_sales", color="sales_per_store",
        title=f"{quarter_label(prior_period)} → {latest_label}: 점포 수 증감과 매출 증감",
        labels={"store_growth_pct": "점포 수 증감률(%)", "sales_growth_pct": "매출 증감률(%)"},
    )
    chart.add_hline(y=0, line_dash="dash", line_color="gray")
    chart.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(chart, use_container_width=True)
    risk_count = ((quadrant["store_growth_pct"] > 0) & (quadrant["sales_growth_pct"] <= 0)).sum()
    st.caption(f"오른쪽 아래 사분면(점포 수 증가·매출 비증가)에는 {risk_count}개 상권이 있습니다.")

st.subheader("유동인구는 높지만 점포당 추정매출이 낮은 후보")
if not population_available:
    st.warning("선택한 분기는 유동인구 원본이 없습니다. 유동인구 기반 후보 비교는 2025년 분기만 가능합니다.")
else:
    high_traffic_low_sales = latest.loc[
        (latest["floating_population"] >= latest["floating_population"].median())
        & (latest["sales_per_store"] < latest["sales_per_store"].median())
    ].sort_values("sales_per_store")
    candidate_columns = [
        "district_label", "estimated_sales", "store_count", "floating_population",
        "sales_per_store", "competition_percentile",
    ]
    st.write(
        f"{district} {industry}의 {latest_label} 기준 후보는 **{len(high_traffic_low_sales)}곳**입니다. "
        "창업 추천 목록이 아니라 임대료·경쟁 점포·체류 수요를 추가 확인할 우선 조사 목록입니다."
    )
    st.dataframe(high_traffic_low_sales[candidate_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "후보 상권 CSV 다운로드",
        data=high_traffic_low_sales[candidate_columns].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=f"{district}_{industry}_{latest_period}_후보상권.csv",
        mime="text/csv",
    )

with st.expander("전체 상권 비교 테이블"):
    st.dataframe(
        latest.sort_values("sales_per_store", ascending=False)[
            ["district_label", "estimated_sales", "store_count", "store_growth_rate",
             "sales_per_store", "sales_per_population", "competition_percentile"]
        ],
        use_container_width=True,
        hide_index=True,
    )
