"""Extract the Olist archive and create CSV files matching the MySQL schema."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "olist-brazilian-ecommerce.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def read_csv_from_zip(archive: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    member = next(name for name in archive.namelist() if name.endswith(filename))
    with archive.open(member) as source:
        return pd.read_csv(source)


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError("Run src/download_data.py before transforming data.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        customers = read_csv_from_zip(archive, "olist_customers_dataset.csv")
        orders = read_csv_from_zip(archive, "olist_orders_dataset.csv")
        items = read_csv_from_zip(archive, "olist_order_items_dataset.csv")
        products = read_csv_from_zip(archive, "olist_products_dataset.csv")
        translation = read_csv_from_zip(archive, "product_category_name_translation.csv")

    customers = customers[
        ["customer_id", "customer_unique_id", "customer_city", "customer_state"]
    ]
    products = products.merge(translation, on="product_category_name", how="left")
    products = products.assign(
        product_category=products["product_category_name_english"].fillna("unknown")
    )[
        [
            "product_id",
            "product_category",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ]
    orders = orders.rename(
        columns={
            "order_purchase_timestamp": "purchase_at",
            "order_approved_at": "approved_at",
            "order_delivered_carrier_date": "delivered_carrier_at",
            "order_delivered_customer_date": "delivered_at",
            "order_estimated_delivery_date": "estimated_delivery_at",
        }
    )[
        [
            "order_id",
            "customer_id",
            "order_status",
            "purchase_at",
            "approved_at",
            "delivered_carrier_at",
            "delivered_at",
            "estimated_delivery_at",
        ]
    ]
    items = items.rename(columns={"shipping_limit_date": "shipping_limit_at"})[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_at",
            "price",
            "freight_value",
        ]
    ]

    for name, frame in {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": items,
    }.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8")
        print(f"Created {name}.csv: {len(frame):,} rows")


if __name__ == "__main__":
    main()
