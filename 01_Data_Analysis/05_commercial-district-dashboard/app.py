"""Streamlit dashboard for the normalized Seoul commercial-district dataset."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "processed" / "commercial_district_quarterly.csv"
st.set_page_config(page_title="상권 분석", layout="wide")
st.title("서울 상권 분석 대시보드")
st.caption("2024년 이후 분기 자료만 비교합니다. 추정매출은 실제 손익이 아닙니다.")

if not DATA_PATH.exists():
    st.warning("정제 데이터가 없습니다. README의 원본 데이터 수집·정제 절차를 먼저 실행하세요.")
    st.stop()

data = pd.read_csv(DATA_PATH)
district = st.sidebar.selectbox("자치구", sorted(data["district_name"].dropna().unique()), index=0)
industry = st.sidebar.selectbox("업종", sorted(data["industry_name"].dropna().unique()))
subset = data.query("district_name == @district and industry_name == @industry").copy()
periods = sorted(subset["period"].unique())
selected_periods = st.sidebar.multiselect("분기", periods, default=periods)
subset = subset.loc[subset["period"].isin(selected_periods)]

latest = subset.loc[subset["period"].eq(max(selected_periods))]
cols = st.columns(4)
cols[0].metric("분기 추정매출", f"{latest['estimated_sales'].sum():,.0f}원")
cols[1].metric("점포당 추정매출", f"{latest['sales_per_store'].mean():,.0f}원")
cols[2].metric("평균 점포 수", f"{latest['store_count'].mean():,.1f}")
cols[3].metric("평균 경쟁도", f"{latest['competition_percentile'].mean():.1f}")

st.plotly_chart(
    px.line(subset.groupby("period", as_index=False)["estimated_sales"].sum(), x="period", y="estimated_sales",
            markers=True, title="분기 추정매출 추이"),
    use_container_width=True,
)
comparison = latest.sort_values("sales_per_store", ascending=False)
st.plotly_chart(
    px.bar(comparison, x="district_label", y="sales_per_store", color="competition_percentile",
           title="상권별 점포당 추정매출과 경쟁도"),
    use_container_width=True,
)
st.dataframe(
    comparison[["district_label", "estimated_sales", "store_count", "store_growth_rate",
                "sales_per_store", "sales_per_population", "competition_percentile"]],
    use_container_width=True,
)

high_traffic_low_sales = latest.loc[
    (latest["floating_population"] >= latest["floating_population"].median())
    & (latest["sales_per_store"] < latest["sales_per_store"].median())
]
st.subheader("분석 요약")
st.write(
    f"{district} {industry}에서 유동인구는 높지만 점포당 추정매출이 중앙값보다 낮은 상권은 "
    f"{len(high_traffic_low_sales)}곳입니다. 후보 상권은 임대료·경쟁 점포·시간대별 매출을 추가 확인해야 합니다."
)
