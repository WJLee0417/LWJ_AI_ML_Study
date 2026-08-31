"""Compare monthly delivered-order revenue calculated by Pandas and MySQL."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pymysql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_PATH = PROJECT_ROOT / "results" / "generated" / "monthly-revenue-verification.md"
MONTHLY_REVENUE_SQL = """
SELECT DATE_FORMAT(o.purchase_at, '%Y-%m') AS revenue_month,
       ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY revenue_month ORDER BY revenue_month
"""


def connection() -> pymysql.Connection:
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
    )


def main() -> None:
    orders = pd.read_csv(PROCESSED_DIR / "orders.csv", parse_dates=["purchase_at"])
    items = pd.read_csv(PROCESSED_DIR / "order_items.csv")
    delivered = orders.loc[orders["order_status"].eq("delivered"), ["order_id", "purchase_at"]]
    pandas_result = (
        delivered.merge(items, on="order_id")
        .assign(revenue=lambda frame: frame["price"] + frame["freight_value"],
                revenue_month=lambda frame: frame["purchase_at"].dt.strftime("%Y-%m"))
        .groupby("revenue_month", as_index=False)["revenue"].sum()
        .round({"revenue": 2})
    )

    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(MONTHLY_REVENUE_SQL)
        mysql_result = pd.DataFrame(cursor.fetchall(), columns=["revenue_month", "mysql_revenue"])

    compared = pandas_result.merge(mysql_result, on="revenue_month", how="outer")
    compared["difference"] = compared["revenue"] - compared["mysql_revenue"]
    passed = compared["difference"].abs().le(0.01).all()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        "# 월별 매출 검증\n\n"
        f"- 결과: {'PASS' if passed else 'FAIL'}\n"
        "- 기준: Pandas와 MySQL의 월별 매출 차이가 0.01 이하\n\n"
        + compared.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Verification {'passed' if passed else 'failed'}: {RESULT_PATH}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
