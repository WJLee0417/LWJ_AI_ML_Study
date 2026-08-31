"""Create customer-level features from the normalized Olist transaction tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PROJECT_DATA = PROJECT_ROOT.parent / "03_sql-analysis" / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"


def main() -> None:
    required = ["customers.csv", "orders.csv", "order_items.csv", "products.csv"]
    missing = [name for name in required if not (SQL_PROJECT_DATA / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Required SQL project outputs are missing: " + ", ".join(missing)
            + ". Run 03_sql-analysis/src/transform_data.py first."
        )

    customers = pd.read_csv(SQL_PROJECT_DATA / "customers.csv")
    orders = pd.read_csv(SQL_PROJECT_DATA / "orders.csv", parse_dates=["purchase_at"])
    items = pd.read_csv(SQL_PROJECT_DATA / "order_items.csv")
    products = pd.read_csv(SQL_PROJECT_DATA / "products.csv")

    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    delivered_items = delivered.merge(items, on="order_id").merge(
        products[["product_id", "product_category"]], on="product_id", how="left"
    )
    delivered_items["line_revenue"] = delivered_items["price"] + delivered_items["freight_value"]
    customer_orders = delivered.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id"
    )
    reference_date = customer_orders["purchase_at"].max()

    frequency = customer_orders.groupby("customer_unique_id").agg(
        purchase_frequency=("order_id", "nunique"),
        last_purchase_at=("purchase_at", "max"),
    )
    revenue = delivered_items.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id"
    ).groupby("customer_unique_id").agg(
        total_revenue=("line_revenue", "sum"),
        avg_order_value=("line_revenue", "sum"),
    )
    revenue["avg_order_value"] = revenue["avg_order_value"] / frequency["purchase_frequency"]
    category_revenue = delivered_items.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id"
    ).groupby(["customer_unique_id", "product_category"], as_index=False)["line_revenue"].sum()
    preferred_category = (
        category_revenue.sort_values(
            ["customer_unique_id", "line_revenue", "product_category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_unique_id")
        .set_index("customer_unique_id")["product_category"]
        .rename("preferred_category")
    )

    features = frequency.join(revenue).join(preferred_category).reset_index()
    features["recency_days"] = (reference_date - features["last_purchase_at"]).dt.days
    features["reference_date"] = reference_date.date().isoformat()
    features = features[
        [
            "customer_unique_id",
            "purchase_frequency",
            "total_revenue",
            "recency_days",
            "avg_order_value",
            "preferred_category",
            "reference_date",
        ]
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Created {OUTPUT_PATH}: {len(features):,} customer rows")


if __name__ == "__main__":
    main()
