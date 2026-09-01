from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_excel(ROOT / "data/raw/default of credit card clients.xls", header=1)
data = data.rename(columns={"default payment next month": "default_next_month"}).drop(columns="ID")
data.to_csv(ROOT / "data/processed/loan_default_cleaned.csv", index=False)
(ROOT / "data/data-dictionary.md").write_text(
    "# Data dictionary\n\nID is excluded. default_next_month is the target.",
    encoding="utf-8",
)
