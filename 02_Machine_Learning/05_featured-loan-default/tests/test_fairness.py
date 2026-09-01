import pandas as pd

from src.analyze_fairness import calculate_group_metrics


groups = pd.Series(["A", "A", "B", "B"])
target = pd.Series([1, 0, 1, 0])
prediction = pd.Series([1, 1, 0, 0])
result = {row["group"]: row for row in calculate_group_metrics(groups, target, prediction, "테스트")}

assert result["A"]["customers"] == 2
assert result["A"]["recall"] == 1.0
assert result["A"]["false_positive_rate"] == 1.0
assert result["B"]["recall"] == 0.0
assert result["B"]["false_positive_rate"] == 0.0
