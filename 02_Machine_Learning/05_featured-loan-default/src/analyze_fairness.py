"""Audit final-test prediction gaps by sex code and age band.

This script is an observational monitoring check. It does not establish causal
fairness and must not be used as an automated lending decision rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "loan_default_cleaned.csv"
MODEL_PATH = ROOT / "models" / "loan_default_model.joblib"
FINAL_TEST_PATH = ROOT / "results" / "final-test.json"
OUTPUT_PATH = ROOT / "results" / "fairness-by-group.csv"
REPORT_PATH = ROOT / "results" / "fairness-report.md"


def get_final_test_split() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH)
    target = data.pop("default_next_month")

    _, temporary_x, _, temporary_y = train_test_split(
        data, target, test_size=0.4, stratify=target, random_state=42
    )
    _, test_x, _, test_y = train_test_split(
        temporary_x, temporary_y, test_size=0.5, stratify=temporary_y, random_state=42
    )
    return test_x, test_y


def calculate_group_metrics(
    values: pd.Series, target: pd.Series, prediction: pd.Series, group_type: str
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for group in values.dropna().unique():
        mask = values == group
        group_target = target[mask]
        group_prediction = prediction[mask]
        positives = int(group_target.sum())
        negatives = int((group_target == 0).sum())
        true_positive = int(((group_target == 1) & (group_prediction == 1)).sum())
        false_positive = int(((group_target == 0) & (group_prediction == 1)).sum())

        rows.append(
            {
                "group_type": group_type,
                "group": str(group),
                "customers": int(mask.sum()),
                "observed_default_rate": group_target.mean(),
                "predicted_positive_rate": group_prediction.mean(),
                "recall": true_positive / positives if positives else None,
                "false_positive_rate": false_positive / negatives if negatives else None,
            }
        )
    return rows


def write_report(result: pd.DataFrame, threshold: float) -> None:
    display = result.copy()
    for column in [
        "observed_default_rate",
        "predicted_positive_rate",
        "recall",
        "false_positive_rate",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "-")

    columns = list(display.columns)
    markdown_table = "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| " + " | ".join(str(row[column]) for column in columns) + " |"
                for _, row in display.iterrows()
            ],
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# 공정성 점검 보고서",
                "",
                f"- 평가 데이터: 최종 Test 세트 (정책 임계값 {threshold:.2f})",
                "- 지표: 실제 연체율, 예측 양성률, Recall, False Positive Rate(FPR)",
                "- 목적: 성별 코드와 연령대별 결과 차이를 관찰하는 배포 전 점검입니다.",
                "",
                markdown_table,
                "",
                "## 해석 주의사항",
                "",
                "- 집단 간 차이는 데이터 분포와 표본 수의 영향을 받으며, 이 표만으로 차별 여부나 인과관계를 판단할 수 없습니다.",
                "- 원본은 특정 국가·시기·신용카드 고객 표본이므로 다른 대출 상품과 모집단에 일반화할 수 없습니다.",
                "- 성별과 연령은 공정성 점검용으로만 집계하며, 자동 승인·거절의 근거로 사용하면 안 됩니다.",
                "- 차이가 큰 경우 임계값 정책, 변수 사용 여부, 데이터 수집 편향을 함께 재검토해야 합니다.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    final_test = json.loads(FINAL_TEST_PATH.read_text(encoding="utf-8"))
    threshold = float(final_test["threshold"])
    test_x, test_y = get_final_test_split()
    probability = joblib.load(MODEL_PATH).predict_proba(test_x)[:, 1]
    prediction = pd.Series((probability >= threshold).astype(int), index=test_x.index)

    sex_group = test_x["SEX"].map({1: "남성(코드 1)", 2: "여성(코드 2)"})
    age_group = pd.cut(
        test_x["AGE"],
        bins=[17, 29, 39, 49, 59, 100],
        labels=["18~29세", "30~39세", "40~49세", "50~59세", "60세 이상"],
    )

    rows = calculate_group_metrics(sex_group, test_y, prediction, "성별")
    rows.extend(calculate_group_metrics(age_group, test_y, prediction, "연령대"))
    result = pd.DataFrame(rows).sort_values(["group_type", "group"]).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    write_report(result, threshold)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
