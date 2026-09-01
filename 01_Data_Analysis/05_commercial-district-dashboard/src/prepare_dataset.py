"""Normalize Seoul commercial-district CSV files into one dashboard dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "commercial_district_quarterly.csv"

ALIASES = {
    "period": ["stdr_yyqu_cd", "기준_년분기_코드"],
    "district_type": ["trdar_se_cd", "상권_구분_코드"],
    "district_type_name": ["trdar_se_cd_nm", "상권_구분_코드_명"],
    "district_code": ["trdar_cd", "상권_코드"],
    "district_label": ["trdar_cd_nm", "상권_코드_명"],
    "industry_code": ["svc_induty_cd", "서비스_업종_코드"],
    "industry_name": ["svc_induty_cd_nm", "서비스_업종_코드_명"],
    "estimated_sales": ["thsmon_selng_amt", "당월_매출_금액", "분기당_매출_금액"],
    "store_count": ["stor_co", "점포_수"],
    "opening_stores": ["opbiz_stor_co", "개업_점포_수"],
    "closing_stores": ["clsbiz_stor_co", "폐업_점포_수"],
    "floating_population": ["tot_flpop_co", "총_유동인구_수"],
    "district_name": ["signgu_cd_nm", "자치구_코드_명"],
}


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("cp949", "utf-8-sig", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode: {path}")


def normalize(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    renamed = {}
    for target, candidates in ALIASES.items():
        source = next((column for column in candidates if column in frame.columns), None)
        if source:
            renamed[source] = target
    normalized = frame.rename(columns=renamed)
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; available columns: {list(frame.columns)}")
    return normalized


def read_many(prefix: str, required: list[str]) -> pd.DataFrame:
    paths = sorted(RAW.glob(f"{prefix}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No files matching {prefix}_*.csv in {RAW}")
    return pd.concat([normalize(read_csv(path), required) for path in paths], ignore_index=True)


def main() -> None:
    sales = normalize(
        read_many("sales", ["period", "district_code", "industry_code", "industry_name", "estimated_sales"]),
        ["period", "district_code", "industry_code", "industry_name", "estimated_sales"],
    )
    stores = normalize(
        read_many("stores", ["period", "district_code", "industry_code", "store_count"]),
        ["period", "district_code", "industry_code", "store_count"],
    )
    population = normalize(read_csv(RAW / "population.csv"), ["period", "district_code", "floating_population"])
    areas = normalize(read_csv(RAW / "areas.csv"), ["district_code", "district_label", "district_name"])

    key = ["period", "district_code", "industry_code"]
    dataset = sales[key + ["industry_name", "estimated_sales"]].merge(
        stores[key + ["store_count"]], on=key, how="inner", validate="one_to_one"
    ).merge(
        population[["period", "district_code", "floating_population"]],
        on=["period", "district_code"], how="left", validate="many_to_one"
    ).merge(
        areas[["district_code", "district_label", "district_name"]].drop_duplicates("district_code"),
        on="district_code", how="left", validate="many_to_one"
    )
    numeric = ["estimated_sales", "store_count", "floating_population"]
    dataset[numeric] = dataset[numeric].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.loc[dataset["period"].astype(str).str.startswith(("2024", "2025"))].copy()
    dataset["sales_per_store"] = dataset["estimated_sales"] / dataset["store_count"].replace(0, pd.NA)
    dataset["sales_per_population"] = dataset["estimated_sales"] / dataset["floating_population"].replace(0, pd.NA)
    dataset = dataset.sort_values(["district_code", "industry_code", "period"])
    dataset["store_growth_rate"] = dataset.groupby(["district_code", "industry_code"])["store_count"].pct_change() * 100
    dataset["competition_percentile"] = dataset.groupby(["period", "industry_code"])["store_count"].rank(pct=True) * 100
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Created {OUTPUT}: {len(dataset):,} rows")


if __name__ == "__main__":
    main()
