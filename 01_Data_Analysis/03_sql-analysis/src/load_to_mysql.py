"""Load transformed Olist CSV files into MySQL using environment-based credentials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pymysql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES = {
    "customers": ["customer_id", "customer_unique_id", "customer_city", "customer_state"],
    "products": [
        "product_id", "product_category", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ],
    "orders": [
        "order_id", "customer_id", "order_status", "purchase_at", "approved_at",
        "delivered_carrier_at", "delivered_at", "estimated_delivery_at",
    ],
    "order_items": [
        "order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_at",
        "price", "freight_value",
    ],
}


def connection() -> pymysql.Connection:
    required = ["MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
    )


def rows_for(frame: pd.DataFrame, columns: list[str]) -> list[tuple[object, ...]]:
    nullable = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    return list(nullable.itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete existing rows before loading.")
    args = parser.parse_args()
    if not all((PROCESSED_DIR / f"{table}.csv").exists() for table in TABLES):
        raise FileNotFoundError("Run src/transform_data.py before loading MySQL.")

    with connection() as conn, conn.cursor() as cursor:
        if args.reset:
            for table in ["order_items", "orders", "products", "customers"]:
                cursor.execute(f"DELETE FROM {table}")

        for table, columns in TABLES.items():
            frame = pd.read_csv(PROCESSED_DIR / f"{table}.csv")
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            cursor.executemany(
                f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})",
                rows_for(frame, columns),
            )
            print(f"Loaded {cursor.rowcount:,} rows into {table}")
        conn.commit()


if __name__ == "__main__":
    main()
