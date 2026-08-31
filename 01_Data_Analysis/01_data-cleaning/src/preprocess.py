"""Create reproducible cleaned tables and a data-quality report for Online Retail."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "results" / "generated" / "data-quality-report.md"
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def percent(part: int, whole: int) -> str:
    return "0.00%" if whole == 0 else f"{part / whole:.2%}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def read_source() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"원본 파일이 없습니다: {RAW_PATH}\n"
            "먼저 `python src/download_data.py`를 실행하세요."
        )

    orders = pd.read_excel(RAW_PATH)
    missing_columns = set(EXPECTED_COLUMNS) - set(orders.columns)
    if missing_columns:
        raise ValueError(f"예상한 컬럼이 없습니다: {sorted(missing_columns)}")
    return orders[EXPECTED_COLUMNS].copy()


def clean_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    metrics: dict[str, int] = {"raw_rows": len(orders)}
    metrics["duplicate_rows"] = int(orders.duplicated().sum())
    cleaned = orders.drop_duplicates().copy()
    metrics["after_deduplication"] = len(cleaned)

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype("string").str.strip()
    cleaned["StockCode"] = cleaned["StockCode"].astype("string").str.strip()
    cleaned["Description"] = cleaned["Description"].astype("string").str.strip()
    cleaned["Country"] = cleaned["Country"].astype("string").str.strip()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce").astype("Int64")

    cleaned["is_cancellation"] = (
        cleaned["InvoiceNo"].str.startswith("C", na=False) | (cleaned["Quantity"] < 0)
    )
    cleaned["is_valid_transaction"] = (
        ~cleaned["is_cancellation"]
        & (cleaned["Quantity"] > 0)
        & (cleaned["UnitPrice"] > 0)
        & cleaned["InvoiceDate"].notna()
    )
    cleaned["line_revenue"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned["order_year"] = cleaned["InvoiceDate"].dt.year.astype("Int64")
    cleaned["order_month"] = cleaned["InvoiceDate"].dt.to_period("M").astype("string")
    cleaned["order_day_of_week"] = cleaned["InvoiceDate"].dt.day_name()
    cleaned["order_hour"] = cleaned["InvoiceDate"].dt.hour.astype("Int64")

    metrics["cancellations"] = int(cleaned["is_cancellation"].sum())
    metrics["non_positive_quantity"] = int((cleaned["Quantity"] <= 0).sum())
    metrics["non_positive_unit_price"] = int((cleaned["UnitPrice"] <= 0).sum())
    metrics["missing_customer_id"] = int(cleaned["CustomerID"].isna().sum())
    metrics["invalid_invoice_date"] = int(cleaned["InvoiceDate"].isna().sum())
    metrics["valid_transactions"] = int(cleaned["is_valid_transaction"].sum())
    return cleaned, metrics


def write_report(raw: pd.DataFrame, cleaned: pd.DataFrame, metrics: dict[str, int]) -> None:
    raw_rows = metrics["raw_rows"]
    missing_rows = [
        [column, int(raw[column].isna().sum()), percent(int(raw[column].isna().sum()), raw_rows)]
        for column in raw.columns
    ]
    category_rows = [
        [column, int(raw[column].nunique(dropna=True))]
        for column in ["StockCode", "Description", "Country"]
    ]
    summary_rows = [
        ["원본 행", metrics["raw_rows"], "-"],
        ["완전 중복 행", metrics["duplicate_rows"], percent(metrics["duplicate_rows"], raw_rows)],
        ["중복 제거 후 행", metrics["after_deduplication"], percent(metrics["after_deduplication"], raw_rows)],
        ["취소 주문 행", metrics["cancellations"], percent(metrics["cancellations"], raw_rows)],
        ["유효 일반 주문 행", metrics["valid_transactions"], percent(metrics["valid_transactions"], raw_rows)],
    ]
    valid_orders = cleaned.loc[cleaned["is_valid_transaction"]]
    customer_orders = valid_orders.loc[valid_orders["CustomerID"].notna()]

    report = f"""# 데이터 품질 보고서

## 실행 요약

{markdown_table(["단계", "행 수", "원본 대비"], summary_rows)}

## 원본 프로파일

- 행·열 수: `{raw.shape[0]:,} × {raw.shape[1]}`
- 컬럼 타입: `{', '.join(f'{column}: {dtype}' for column, dtype in raw.dtypes.items())}`
- 완전 중복 행: `{metrics['duplicate_rows']:,}`

### 결측치

{markdown_table(["컬럼", "결측 수", "결측 비율"], missing_rows)}

### 범주형 값 수

{markdown_table(["컬럼", "고유값 수"], category_rows)}

## 품질 규칙 적용 결과

| 항목 | 행 수 | 설명 |
| --- | ---: | --- |
| 취소 주문 | {metrics['cancellations']:,} | `InvoiceNo`가 C로 시작하거나 수량이 음수 |
| 0 이하 수량 | {metrics['non_positive_quantity']:,} | 일반 판매로는 유효하지 않을 수 있음 |
| 0 이하 단가 | {metrics['non_positive_unit_price']:,} | 일반 판매로는 유효하지 않을 수 있음 |
| 고객 ID 결측 | {metrics['missing_customer_id']:,} | 고객 단위 분석에서 제외 |
| 날짜 변환 실패 | {metrics['invalid_invoice_date']:,} | 거래 분석에서 제외 |

## 분석용 데이터셋

- 거래 분석: 유효 일반 주문 `{len(valid_orders):,}`행
- 고객 분석: 유효 일반 주문 중 고객 ID 보유 `{len(customer_orders):,}`행
- 유효 일반 주문 매출 합계: `{valid_orders['line_revenue'].sum():,.2f}`

## 해석 시 유의점

- 취소 주문은 삭제하지 않고 상태로 보존했다. 매출 분석에서는 유효 일반 주문만 사용한다.
- 고객 ID 결측 주문은 거래 분석에는 남지만 고객 단위 분석에서는 제외된다.
- 극단값은 자동 삭제하지 않았다. 이상치 처리는 이후 분석 목적과 도메인 기준을 정한 뒤 별도 수행한다.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    raw = read_source()
    cleaned, metrics = clean_orders(raw)
    valid_orders = cleaned.loc[cleaned["is_valid_transaction"]].copy()
    customer_orders = valid_orders.loc[valid_orders["CustomerID"].notna()].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DIR / "orders_cleaned.csv", index=False, encoding="utf-8-sig")
    valid_orders.to_csv(
        PROCESSED_DIR / "orders_for_transaction_analysis.csv", index=False, encoding="utf-8-sig"
    )
    customer_orders.to_csv(
        PROCESSED_DIR / "orders_for_customer_analysis.csv", index=False, encoding="utf-8-sig"
    )
    write_report(raw, cleaned, metrics)
    print(f"Created cleaned tables in: {PROCESSED_DIR}")
    print(f"Created quality report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
