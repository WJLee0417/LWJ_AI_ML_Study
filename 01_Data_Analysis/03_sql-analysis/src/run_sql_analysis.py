"""Execute the ten portfolio SQL queries and record reproducible result samples."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pymysql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = PROJECT_ROOT / "sql" / "analysis_queries.sql"
RESULT_PATH = PROJECT_ROOT / "results" / "query-results.md"


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


def markdown_table(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    statements = [item.strip() for item in QUERY_PATH.read_text(encoding="utf-8").split(";") if item.strip()]
    sections = [
        "# SQL 분석 실행 결과",
        "",
        "MySQL 8.4 컨테이너에 Olist 변환 데이터를 적재한 뒤 실행한 결과다.",
        "각 표는 재현성을 위한 상위 5행 미리보기이며, `row_count`는 전체 결과 행 수다.",
        "",
    ]

    with connection() as conn, conn.cursor() as cursor:
        for statement in statements:
            if statement.upper().startswith("USE "):
                cursor.execute(statement)
                continue

            match = re.search(r"--\s*(\d{2}\..+)", statement)
            if not match:
                raise ValueError(f"Cannot identify query title: {statement[:80]}")
            title = match.group(1)
            cursor.execute(statement)
            columns = [column[0] for column in cursor.description]
            rows = list(cursor.fetchall())
            preview = rows[:5]

            sections.extend([
                f"## {title}",
                "",
                f"- 전체 결과 행 수: {len(rows):,}",
                "- 미리보기: 상위 5행",
                "",
                markdown_table(columns, preview),
                "",
            ])

    RESULT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote executed query results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
