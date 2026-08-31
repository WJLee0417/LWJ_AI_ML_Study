"""Generate focused EDA figures and an evidence-based report for Seoul bike demand."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "SeoulBikeData.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "seoul_bike_cleaned.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_PATH = PROJECT_ROOT / "results" / "generated" / "analysis-report.md"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = list(range(1, 13))
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
RAIN_BAND_ORDER = ["0 mm", "0-1 mm", "1-5 mm", "5+ mm"]


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"원본 파일이 없습니다: {RAW_PATH}\n먼저 `python src/download_data.py`를 실행하세요."
        )

    data = pd.read_csv(RAW_PATH, encoding="unicode_escape")
    data["datetime"] = pd.to_datetime(data["Date"], format="%d/%m/%Y") + pd.to_timedelta(
        data["Hour"], unit="h"
    )
    data["month"] = data["datetime"].dt.month
    data["weekday"] = pd.Categorical(
        data["datetime"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True
    )
    data["is_weekend"] = data["datetime"].dt.dayofweek.ge(5).map(
        {True: "Weekend", False: "Weekday"}
    )
    data["rainfall_band"] = pd.cut(
        data["Rainfall(mm)"],
        bins=[-0.01, 0, 1, 5, float("inf")],
        labels=RAIN_BAND_ORDER,
    )
    return data.loc[data["Functioning Day"].eq("Yes")].copy()


def save_plot(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def create_figures(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    target = "Rented Bike Count"

    hourly = data.groupby("Hour", observed=True)[target].mean()
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o")
    plt.title("Average hourly bike demand")
    plt.xlabel("Hour")
    plt.ylabel("Average rented bikes")
    plt.xticks(range(0, 24, 2))
    save_plot("hourly-demand.png")

    weekday = data.groupby("weekday", observed=True)[target].mean().reindex(WEEKDAY_ORDER)
    plt.figure(figsize=(10, 4.5))
    sns.barplot(x=weekday.index, y=weekday.values, hue=weekday.index, legend=False, palette="Blues_d")
    plt.title("Average bike demand by weekday")
    plt.xlabel("")
    plt.ylabel("Average rented bikes")
    plt.xticks(rotation=25, ha="right")
    save_plot("weekday-demand.png")

    heatmap = data.pivot_table(
        index="weekday", columns="Hour", values=target, aggfunc="mean", observed=True
    ).reindex(WEEKDAY_ORDER)
    plt.figure(figsize=(13, 4.5))
    sns.heatmap(heatmap, cmap="YlGnBu", linewidths=0.2, cbar_kws={"label": "Average rented bikes"})
    plt.title("Average demand by weekday and hour")
    plt.xlabel("Hour")
    plt.ylabel("")
    save_plot("hour-weekday-heatmap.png")

    pattern = data.groupby(["Hour", "is_weekend"], observed=True)[target].mean().reset_index()
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(data=pattern, x="Hour", y=target, hue="is_weekend", marker="o")
    plt.title("Weekday vs weekend hourly demand")
    plt.xlabel("Hour")
    plt.ylabel("Average rented bikes")
    plt.xticks(range(0, 24, 2))
    save_plot("weekday-weekend-pattern.png")

    monthly = data.groupby("month", observed=True)[target].mean().reindex(MONTH_ORDER)
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", color="#2a7f62")
    plt.title("Average bike demand by month")
    plt.xlabel("Month")
    plt.ylabel("Average rented bikes")
    plt.xticks(MONTH_ORDER)
    save_plot("monthly-demand.png")

    seasonal = data.groupby("Seasons", observed=True)[target].mean().reindex(SEASON_ORDER)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=seasonal.index, y=seasonal.values, hue=seasonal.index, legend=False, palette="viridis")
    plt.title("Average bike demand by season")
    plt.xlabel("")
    plt.ylabel("Average rented bikes")
    save_plot("seasonal-demand.png")

    rainfall = data.groupby("rainfall_band", observed=True)[target].mean().reindex(RAIN_BAND_ORDER)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=rainfall.index, y=rainfall.values, hue=rainfall.index, legend=False, palette="mako")
    plt.title("Average bike demand by rainfall band")
    plt.xlabel("Rainfall")
    plt.ylabel("Average rented bikes")
    save_plot("rainfall-demand.png")


def write_report(data: pd.DataFrame) -> None:
    target = "Rented Bike Count"
    hourly = data.groupby("Hour", observed=True)[target].mean()
    weekday = data.groupby("weekday", observed=True)[target].mean().reindex(WEEKDAY_ORDER)
    pattern = data.groupby(["Hour", "is_weekend"], observed=True)[target].mean().unstack()
    seasonal = data.groupby("Seasons", observed=True)[target].mean().reindex(SEASON_ORDER)
    rainfall = data.groupby("rainfall_band", observed=True)[target].mean().reindex(RAIN_BAND_ORDER)
    peak_hour = int(hourly.idxmax())
    peak_weekday = str(weekday.idxmax())
    weekday_peak_hour = int(pattern["Weekday"].idxmax())
    weekend_peak_hour = int(pattern["Weekend"].idxmax())
    best_season = str(seasonal.idxmax())
    wet_drop = (1 - rainfall["5+ mm"] / rainfall["0 mm"]) * 100

    report = f"""# 서울 공공자전거 수요 EDA 보고서

## 분석 범위

- 운영일(`Functioning Day = Yes`) 시간 단위 관측치: `{len(data):,}`건
- 관측 기간: `{data['datetime'].min():%Y-%m-%d}` ~ `{data['datetime'].max():%Y-%m-%d}`
- 분석 대상: 시간당 평균 대여 자전거 수

## 질문 1. 수요가 가장 높은 시간대와 요일은 언제인가?

- 전체 평균 수요가 가장 높은 시간은 **{peak_hour:02d}시**다.
- 평균 수요가 가장 높은 요일은 **{peak_weekday}**다.
- 운영 제안: 전체 수요가 집중되는 시간보다 앞선 시간에 업무·상업 지역의 자전거와 거치 공간을 점검·재배치해야 한다.

## 질문 2. 평일 출퇴근과 주말 여가 수요는 어떻게 다른가?

- 평일 수요 최고점은 **{weekday_peak_hour:02d}시**, 주말 수요 최고점은 **{weekend_peak_hour:02d}시**다.
- 운영 제안: 평일에는 출퇴근 피크를 우선 대응하고, 주말에는 최고 수요 시간과 여가 수요가 이어지는 시간대를 중심으로 점검 인력을 배치해야 한다.

## 질문 3. 계절과 강수량은 수요에 어떤 영향을 미치는가?

- 평균 수요가 가장 높은 계절은 **{best_season}**이다.
- 강수량 0mm 대비 5mm 이상일 때의 평균 수요는 약 **{wet_drop:.1f}% 낮다**.
- 운영 제안: 강수 예보가 있는 날에는 평시 수요를 그대로 기준으로 재배치하지 말고, 강수 강도별 수요 감소를 반영한 운영 기준을 별도로 둬야 한다.

## 시각화

`results/figures/`에 시간·요일·시간대×요일·평일/주말·월·계절·강수량 그래프 7개를 생성했다.

## 한계

- 이 데이터는 시간 단위 집계라 개별 대여소의 수요 불균형을 설명하지 못한다.
- 날씨·계절·휴일은 함께 변할 수 있으므로, 그래프만으로 날씨가 수요를 직접 변화시켰다고 결론 낼 수 없다.
- 단일 연도 데이터이므로, 운영 정책 적용 전에는 다른 기간 데이터로 재검증이 필요하다.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    data = load_data()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    create_figures(data)
    write_report(data)
    print(f"Created processed data: {PROCESSED_PATH}")
    print(f"Created figures: {FIGURES_DIR}")
    print(f"Created report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
